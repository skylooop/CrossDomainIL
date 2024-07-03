import flax.jax_utils
import flax.struct
import flax.struct
from jaxrl_m.typing import *
from jaxrl_m.networks import MLP, get_latent, ensemblize, CrossDomainNetwork, RelativeRepresentation, PhiValueDomain

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

def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def compute_value_loss_source(agent, params, batch):
    batch = batch._replace(rewards=(batch.rewards - 1.0) * 0.1)

    (next_o1, next_o2) = agent.net(batch.next_observations, batch.goals, method='ema_value_source_domain')
    (next_g1, next_g2) = agent.net(batch.observations, batch.next_goals, method='ema_value_source_domain')
    next_v1, next_v2 = jnp.maximum(next_o1, next_g1), jnp.maximum(next_o2, next_g2) 
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch.rewards + 0.99 * batch.masks * next_v
    
    (v1_t, v2_t) = agent.net(batch.observations, batch.goals, method='ema_value_source_domain')
    v_t = (v1_t + v2_t) / 2.
    adv = q - v_t
    
    q1 = batch.rewards + 0.99 * batch.masks * next_v1
    q2 = batch.rewards + 0.99 * batch.masks * next_v2
    (v1, v2) = agent.net(batch.observations, batch.goals, method='value_source_domain', params=params)
    v = (v1 + v2) / 2.
    value_loss1 = expectile_loss(adv, q1 - v1, 0.95).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, 0.95).mean()
    value_loss = value_loss1 + value_loss2
    return value_loss, {'value_source_loss': value_loss,
                        'adv_source_mean': adv.mean()}

def compute_value_loss_target(agent, params, batch):
    batch = batch._replace(rewards=(batch.rewards - 1.0) * 0.1) 

    (next_o1, next_o2) = agent.net(batch.next_observations, batch.goals, method='ema_value_target_domain')
    (next_g1, next_g2) = agent.net(batch.observations, batch.next_goals, method='ema_value_target_domain')
    next_v1, next_v2 = jnp.maximum(next_o1, next_g1), jnp.maximum(next_o2, next_g2) 
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch.rewards + 0.99 * batch.masks * next_v
    
    (v1_t, v2_t) = agent.net(batch.observations, batch.goals, method='ema_value_target_domain')
    v_t = (v1_t + v2_t) / 2.
    adv = q - v_t
    
    q1 = batch.rewards + 0.99 * batch.masks * next_v1
    q2 = batch.rewards + 0.99 * batch.masks * next_v2
    (v1, v2) = agent.net(batch.observations, batch.goals, method='value_target_domain', params=params)
    v = (v1 + v2) / 2.
    value_loss1 = expectile_loss(adv, q1 - v1, 0.95).mean()
    value_loss2 =  expectile_loss(adv, q2 - v2, 0.95).mean()
    value_loss = value_loss1 + value_loss2
    return value_loss, {'value_target_loss': value_loss,
                        'adv_target_mean': adv.mean()}
    

def compute_not_distance(network, potential_elems, potential_pairs, params, source_batch, target_batch): 
    encoded_source = network(source_batch.observations, params=params, method='phi_source_domain')
    encoded_target = network(target_batch.observations, params=params, method='phi_target_domain')
    ema_encoded_source = network(source_batch.observations, method='ema_phi_source_domain')
    ema_encoded_target = network(target_batch.observations, method='ema_phi_target_domain')
    
    T_src = jax.lax.stop_gradient(potential_elems.transport(ema_encoded_source, forward=True))
    T_tgt = jax.lax.stop_gradient(potential_elems.transport(ema_encoded_target, forward=False))
    
    squared_dist_target = ((T_tgt - encoded_target) ** 2).sum(axis=-1)
    v_target = jnp.maximum(squared_dist_target, 1e-6)

    squared_dist_src = ((T_src - encoded_source) ** 2).sum(axis=-1)
    v_src = jnp.maximum(squared_dist_src, 1e-6)
       
    loss = v_target.mean() + 0.2 * v_src.mean()
    return loss

class JointNOTAgent(PyTreeNode):
    rng: PRNGKeyArray
    net: TrainState

    @classmethod
    def create(
        cls,
        seed: int,
        source_obs: jnp.ndarray,
        target_obs: jnp.ndarray,
        latent_dim: int = 32,
        hidden_dims_source: Sequence[int] = (128, 128, 128), #128, 128, 128
        hidden_dims_target: Sequence[int] = (128, 128, 128),
    ):
        rng = jax.random.PRNGKey(seed)
        rng, key1 = jax.random.split(rng, 2)
        
        # encoder_source = FourierMLP(
        #     fourier_net=FourierFeatures(output_size=32, learnable=True),
        #     mlp=MLP(hidden_dims=hidden_dims + (latent_dim, ), activate_final=True, activations=jax.nn.gelu
        # ))
        #encoder_source = MLP(hidden_dims=hidden_dims, activate_final=True, activations=jax.nn.gelu)
        
        # encoder_source = RelativeRepresentation(layer_norm=False, ensemble=True, hidden_dims=hidden_dims_source + (latent_dim, ), bottleneck=False)
        # encoder_target = RelativeRepresentation(layer_norm=False, ensemble=True, hidden_dims=hidden_dims_target + (latent_dim, ), bottleneck=False)
        # Phi acts as an encoder for state-based envs
        value_def_source = PhiValueDomain(encoder=None, hidden_dims=hidden_dims_source, embedding_size=latent_dim, ensemble=True)
        value_def_target = PhiValueDomain(encoder=None, hidden_dims=hidden_dims_target, embedding_size=latent_dim, ensemble=True)
        
        value_def = CrossDomainNetwork(
            networks={
                'value_source_domain': value_def_source,
                'value_target_domain': value_def_target,
                'ema_value_source_domain': copy.deepcopy(value_def_source),
                'ema_value_target_domain': copy.deepcopy(value_def_target)}
        )
        params = value_def.init(key1, source_obs, source_obs, target_obs, target_obs)['params']
        net = TrainState.create(
            model_def=value_def,
            params=params,
            #tx=optax.adam(learning_rate=3e-4)
            tx=optax.multi_transform({'networks_value_source_domain': optax.chain(optax.zero_nans(), optax.adamw(learning_rate=1e-4, weight_decay=0.01)),
                                      'networks_value_target_domain': optax.chain(optax.zero_nans(), optax.adamw(learning_rate=1e-4, weight_decay=0.01)),
                                      "zero": optax.set_to_zero()},
                                      param_labels={'networks_value_source_domain': "networks_value_source_domain",
                                                    'networks_value_target_domain': 'networks_value_target_domain',
                                                    'networks_ema_value_source_domain': 'zero', 'networks_ema_value_target_domain': 'zero'}
        ))
        params = net.params
        params['networks_ema_value_source_domain'] = params['networks_value_source_domain']
        params['networks_ema_value_target_domain'] = params['networks_value_target_domain']
        net = net.replace(params=params)
        return cls(rng=rng, net=net)
    
    @functools.partial(jax.jit, static_argnames=('update_not'))
    def update(self, source_batch, target_batch, potential_elems, potential_pairs, update_not: bool):
        def loss_fn(params):
            info = {}
            
            value_loss_source, source_value_info = compute_value_loss_source(self, params, source_batch)
            for k, v in source_value_info.items():
                info[f'source_enc/{k}'] = v
        
            value_loss_target, target_value_info = compute_value_loss_target(self, params, target_batch)
            for k, v in target_value_info.items():
                info[f'target_enc/{k}'] = v
            
            not_loss = jax.lax.cond(update_not, compute_not_distance, lambda *args: 0., self.net, potential_elems, potential_pairs, params, source_batch, target_batch)
            loss = (value_loss_source + value_loss_target) + 0.01 * not_loss
            return loss, info
        
        new_ema_params_source = optax.incremental_update(self.net.params['networks_value_source_domain'], self.net.params['networks_ema_value_source_domain'], 0.005)
        new_ema_params_target = optax.incremental_update(self.net.params['networks_value_target_domain'], self.net.params['networks_ema_value_target_domain'], 0.005)
        net, info = self.net.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        params = net.params
        params['networks_ema_value_source_domain'] = new_ema_params_source
        params['networks_ema_value_target_domain'] = new_ema_params_target
        new_net = net.replace(params=params)
        
        return self.replace(net=new_net), info
    
    @jax.jit
    def ema_get_phi_source(self, obs):
        phi = self.net(obs, method='ema_phi_source_domain')
        return phi
    
    @jax.jit
    def ema_get_phi_target(self, obs):
        phi = self.net(obs, method='ema_phi_target_domain')
        return phi
    
icvfs = {
    'encodervf': JointNOTAgent,
    'multilinear': MultilinearVF,
    'monolithic': MonolithicVF,
}