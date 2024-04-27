from jaxrl_m.typing import *
from jaxrl_m.networks import MLP, get_latent, default_init, ensemblize, CrossDomainAlign

import flax.linen as nn
import jax
import jax.numpy as jnp

from flax.struct import PyTreeNode
from jaxtyping import PRNGKeyArray
from jaxrl_m.common import TrainState
from jaxtyping import ArrayLike
import optax
from networks.common import FourierFeatures

class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x

class ICVFWithEncoder(nn.Module):
    encoder: nn.Module
    vf: nn.Module

    def get_encoder_latent(self, observations: jnp.ndarray) -> jnp.ndarray:     
        return get_latent(self.encoder, observations)
    
    def get_phi(self, observations: jnp.ndarray) -> jnp.ndarray:
        latent = get_latent(self.encoder, observations)
        return self.vf.get_phi(latent)

    def __call__(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf(latent_s, latent_g, latent_z)
    
    def get_info(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf.get_info(latent_s, latent_g, latent_z)

class SimpleVF(nn.Module):
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, train=None) -> jnp.ndarray:
        V_net = LayerNormMLP((*self.hidden_dims, 1), activate_final=False)
        v = V_net(observations)
        return jnp.squeeze(v, -1)

def create_icvf(icvf_cls_or_name, encoder=None, ensemble=True, **kwargs):    
    if isinstance(icvf_cls_or_name, str):
        icvf_cls = icvfs[icvf_cls_or_name]
    else:
        icvf_cls = icvf_cls_or_name

    if ensemble:
        vf = ensemblize(icvf_cls, 2, methods=['__call__', 'get_info', 'get_phi'])(**kwargs)
    else:
        vf = icvf_cls(**kwargs)
    
    if encoder is None:
        return vf

    return ICVFWithEncoder(encoder, vf)

##
#
# Actual ICVF definitions below
##

class ICVFTemplate(nn.Module):

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Returns useful metrics
        raise NotImplementedError
    
    def get_phi(self, observations):
        # Returns phi(s) for downstream use
        raise NotImplementedError
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        # Returns V(s, g, z)
        raise NotImplementedError

class MonolithicVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.net = network_cls((*self.hidden_dims, 1), activate_final=False)

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        x = jnp.concatenate([observations, outcomes, z], axis=-1)
        v = self.net(x)
        return {
            'v': jnp.squeeze(v, -1),
            'psi': outcomes,
            'z': z,
            'phi': observations,
        }
    
    def get_phi(self, observations):
        print('Warning: StandardVF does not define a state representation phi(s). Returning phi(s) = s')
        return observations
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, outcomes, z], axis=-1)
        v = self.net(x)
        return jnp.squeeze(v, -1)

class MultilinearVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.phi_net = network_cls(self.hidden_dims, activate_final=True, name='phi')
        self.psi_net = network_cls(self.hidden_dims, activate_final=True, name='psi')

        self.T_net =  network_cls(self.hidden_dims, activate_final=True, name='T')

        self.matrix_a = nn.Dense(self.hidden_dims[-1], name='matrix_a')
        self.matrix_b = nn.Dense(self.hidden_dims[-1], name='matrix_b')
        
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> jnp.ndarray:
        return self.get_info(observations, outcomes, intents)['v']
        
    def get_psi(self, observations):
        return self.psi_nety(observations)
    
    def get_phi(self, observations):
        return self.phi_net(observations)

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)

        # T(z) should be a dxd matrix, but having a network output d^2 parameters is inefficient
        # So we'll make a low-rank approximation to T(z) = (diag(Tz) * A * B * diag(Tz))
        # where A and B are (fixed) dxd matrices and Tz is a d-dimensional parameter dependent on z

        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)

        return {
            'v': v,
            'phi': phi,
            'psi': psi,
            'Tz': Tz,
            'z': z,
            'phi_z': phi_z,
            'psi_z': psi_z,
        }
        
class FourierMLP(nn.Module):
    fourier_net: nn.Module
    mlp: nn.Module
    
    @nn.compact
    def __call__(self, x):
        fourier_feat = self.fourier_net(x)
        return self.mlp(fourier_feat)
    
def apply_layernorm(x):
    net_def = nn.LayerNorm(use_bias=False, use_scale=False)
    return net_def.apply({'params': {}}, x)

class EncoderVF(PyTreeNode):
    rng: PRNGKeyArray
    net: TrainState
    target_net: TrainState
    
    @classmethod
    def create(
        cls,
        seed:int,
        observation_sample: ArrayLike,
        latent_dim: int = 4,
        hidden_dims: Sequence[int] = (8, 8, 8) # (32, 32, 32) - works good
    ):
        rng = jax.random.PRNGKey(seed)
        rng, key1, key2 = jax.random.split(rng, 3)
        
        # encoder_source = FourierMLP(
        #     fourier_net=FourierFeatures(output_size=32, learnable=True),
        #     mlp=MLP(hidden_dims=hidden_dims + (latent_dim, ), activate_final=True, activations=jax.nn.gelu
        # ))
        encoder_source = MLP(hidden_dims=hidden_dims + (latent_dim, ), activate_final=True,
                             activations=jax.nn.gelu)
        net_def = CrossDomainAlign(
            encoder=encoder_source
        )
        params = net_def.init(key1, observation_sample)['params']
        net = TrainState.create(
            model_def=net_def,
            params=params,
            tx=optax.adam(learning_rate=3e-4)
        )
        target_net = TrainState.create(
            model_def=net_def,
            params=params
        )
        return cls(
            rng=rng,
            net=net,
            target_net=target_net
        )
    
    @jax.jit
    def update(self, batch):
        def loss_fn(params):
            def get_v(params, obs, goal):
                encoded_s = apply_layernorm(self.net(obs, params=params))
                encoded_snext = apply_layernorm(self.net(goal, params=params))
                dist = optax.safe_norm(encoded_s - encoded_snext, 1e-3, axis=-1)
                #dist = jax.vmap(jnp.dot)(encoded_s, encoded_snext) # dot cost
                return -1 * dist
            
            V = get_v(params, batch.observations, batch.goals) # d(s, s+)
            nV = get_v(params, batch.next_observations, batch.goals) #d(s', s+)
            target_V = batch.rewards + 0.99 * batch.masks * nV

            def expectile_fn(diff, expectile:float=0.9):
                weight = jnp.where(diff >= 0, expectile, 1-expectile)
                return weight * diff ** 2
            
            diff = (V - target_V)
            loss = expectile_fn(diff, 0.9).mean()
            return loss, {'icvf_loss': loss}
            
        net, info = self.net.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        target_params = optax.incremental_update(self.net.params, self.target_net.params, 0.005)
        target_net = self.target_net.replace(params=target_params)
        return self.replace(net=net, target_net=target_net), info
    
icvfs = {
    'encodervf': EncoderVF,
    'multilinear': MultilinearVF,
    'monolithic': MonolithicVF,
}