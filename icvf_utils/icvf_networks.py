import flax.jax_utils
import flax.struct
import flax.struct
from jaxrl_m.typing import *
from jaxrl_m.networks import MLP, get_latent, ensemblize, CrossDomainAlign, RelativeRepresentation

import flax.linen as nn
import jax
import jax.numpy as jnp

from flax.struct import PyTreeNode
from jaxtyping import PRNGKeyArray
from jaxrl_m.common import TrainState
from jaxtyping import ArrayLike
import optax
from networks.common import FourierFeatures
import copy
import functools

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

def compute_source_encoder_loss(net, params, batch):
    def get_v(params, obs, goal):
        encoded_s = apply_layernorm(net(obs, params=params, method='encode_source'))
        encoded_snext = apply_layernorm(net(goal, params=params, method='encode_source'))
        dist = jax.vmap(jnp.dot)(encoded_s, encoded_snext) # dot cost
        return -1 * dist
    
    def get_v_ema(obs, goal):
        encoded_s = apply_layernorm(net(obs, method='encode_source_ema'))
        encoded_snext = apply_layernorm(net(goal, method='encode_source_ema'))
        dist = jax.vmap(jnp.dot)(encoded_s, encoded_snext) # dot cost
        return -1 * dist
    
    V = get_v(params, batch.observations, batch.goals) # d(s, s+)
    nV_1 = get_v_ema(batch.next_observations, batch.goals) # d(s', s+)
    nV_2 = get_v_ema(batch.next_goals, batch.observations) # d(s, s+')
    nV = jnp.maximum(nV_1, nV_2)
    target_V = batch.rewards + 0.99 * batch.masks * nV

    def expectile_fn(diff, expectile:float=0.9):
        weight = jnp.where(diff >= 0, expectile, 1-expectile)
        return weight * diff ** 2
    
    diff = (target_V - V)
    loss = expectile_fn(diff, 0.9).mean()
    return loss, {'source_encoder_loss': loss}

def compute_target_encoder_loss(net, params, batch):
    def get_v(params, obs, goal):
        encoded_s = apply_layernorm(net(obs, params=params, method='encode_target'))
        encoded_snext = apply_layernorm(net(goal, params=params, method='encode_target'))
        dist = jax.vmap(jnp.dot)(encoded_s, encoded_snext) # dot cost
        return -1 * dist
    
    def get_v_ema(obs, goal):
        encoded_s = apply_layernorm(net(obs, method='encode_target_ema'))
        encoded_snext = apply_layernorm(net(goal, method='encode_target_ema'))
        dist = jax.vmap(jnp.dot)(encoded_s, encoded_snext) # dot cost
        return -1 * dist
    
    V = get_v(params, batch.observations, batch.goals) # d(s, s+)
    nV_1 = get_v_ema(batch.next_observations, batch.goals) # d(s', s+)
    nV_2 = get_v_ema(batch.next_goals, batch.observations) # d(s, s+')
    nV = jnp.maximum(nV_1, nV_2)
    target_V = batch.rewards + 0.99 * batch.masks * nV

    def expectile_fn(diff, expectile:float=0.9):
        weight = jnp.where(diff >= 0, expectile, 1-expectile)
        return weight * diff ** 2
    
    diff = (target_V - V)
    loss = expectile_fn(diff, 0.9).mean()
    return loss, {'target_encoder_loss': loss}

def compute_not_distance(network, potentials, params, source_batch, target_batch): 
    encoded_source = network(source_batch.observations, params=params, method='encode_source')   
    encoded_target = network(target_batch.observations, params=params, method='encode_target')
    loss = -(jax.vmap(potentials.f)(encoded_source) + jax.vmap(potentials.g)(encoded_target)).mean()
    return loss

class JointNOTAgent(PyTreeNode):
    rng: PRNGKeyArray
    net: TrainState
    dual_potentials: Any
    neural_dual_agent: Any = flax.struct.field(pytree_node=False)
    
    @classmethod
    def create(
        cls,
        seed: int,
        source_obs: jnp.ndarray,
        target_obs: jnp.ndarray,
        latent_dim: int = 16,
        not_agent: Any = None,
        hidden_dims_source: Sequence[int] = (16, 16, 16, 16),
        hidden_dims_target: Sequence[int] = (16, 16, 16, 16),
        dual_potentials=None
    ):
        rng = jax.random.PRNGKey(seed)
        rng, key1 = jax.random.split(rng, 2)
        
        # encoder_source = FourierMLP(
        #     fourier_net=FourierFeatures(output_size=32, learnable=True),
        #     mlp=MLP(hidden_dims=hidden_dims + (latent_dim, ), activate_final=True, activations=jax.nn.gelu
        # ))
        #encoder_source = MLP(hidden_dims=hidden_dims, activate_final=True, activations=jax.nn.gelu)
        
        encoder_source = RelativeRepresentation(layer_norm=False, hidden_dims=hidden_dims_source + (latent_dim, ), bottleneck=False)
        encoder_target = RelativeRepresentation(layer_norm=False, hidden_dims=hidden_dims_target + (latent_dim, ), bottleneck=False)
        net_def = CrossDomainAlign(
            source_encoder=encoder_source,
            target_encoder=encoder_target,
            ema_encoder_source=copy.deepcopy(encoder_source),
            ema_encoder_target=copy.deepcopy(encoder_target),
            #not_estimator=not_agent
        )
        params = net_def.init(key1, source_obs, target_obs)['params']
        net = TrainState.create(
            model_def=net_def,
            params=params,
            tx=optax.multi_transform({'source_encoder': optax.chain(optax.zero_nans(), optax.adam(learning_rate=3e-4)),
                                      'target_encoder': optax.chain(optax.zero_nans(), optax.adam(learning_rate=3e-4)),
                                      "zero": optax.set_to_zero()},
                                      param_labels={'source_encoder': "source_encoder", 'target_encoder': 'target_encoder',
                                                    'ema_encoder_source': 'zero', 'ema_encoder_target': 'zero'}
        ))
        params['ema_encoder_source'] = net.params['source_encoder']
        params['ema_encoder_target'] = net.params['target_encoder']
        net = net.replace(params=params)
        return cls(rng=rng, net=net, dual_potentials=dual_potentials, neural_dual_agent=not_agent)
    
    def update_not(self, batch_source, batch_target):
        encoded_source = self.net(batch_source.observations, method='encode_source')
        encoded_target = self.net(batch_target.observations, method='encode_target')
        new_not_agent, loss, w_dist = self.neural_dual_agent.update(encoded_source, encoded_target)
        potentials = new_not_agent.to_dual_potentials(finetune_g=True)
        return self.replace(dual_potentials=potentials, neural_dual_agent=new_not_agent), encoded_source, encoded_target, {"loss": loss, "w_dist": w_dist}
    
    @functools.partial(jax.jit, static_argnames=('update_not'))
    def update(self, source_batch, target_batch, update_not: bool):
        def loss_fn(params):
            info = {}
            
            source_enc_loss, source_enc_info = compute_source_encoder_loss(self.net, params, source_batch)
            for k, v in source_enc_info.items():
                info[f'source_enc/{k}'] = v
            
            target_enc_loss, target_enc_info = compute_target_encoder_loss(self.net, params, target_batch)
            for k, v in target_enc_info.items():
                info[f'target_enc/{k}'] = v
                
            not_loss = jax.lax.cond(update_not, compute_not_distance, lambda *args: 0., self.net, self.dual_potentials, params, source_batch, target_batch)
            loss = source_enc_loss + target_enc_loss + 1.5 * not_loss
            return loss, info
        
        new_ema_params_source = optax.incremental_update(self.net.params['source_encoder'], self.net.params['ema_encoder_source'], 0.005)
        new_ema_params_target = optax.incremental_update(self.net.params['target_encoder'], self.net.params['ema_encoder_target'], 0.005)
        net, info = self.net.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        
        params = net.params
        params['ema_encoder_source'] = new_ema_params_source
        params['ema_encoder_target'] = new_ema_params_target
        new_net = net.replace(params=params)
        
        return self.replace(net=new_net), info
    
icvfs = {
    'encodervf': JointNOTAgent,
    'multilinear': MultilinearVF,
    'monolithic': MonolithicVF,
}