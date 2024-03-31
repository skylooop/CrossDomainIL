import jax
import jax.numpy as jnp

from datasets.dataset import Batch
from networks.common import Model
from networks.critic_net import DoubleCritic
from networks.policies import _sample_actions

import optax
import flax.linen as nn

from jaxtyping import Array, Key
from typing import Tuple, Dict
import numpy as np

import functools

InfoDict = Dict[str, float]

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

@jax.jit
def update_temperature(temp: Model, entropy: float,
           target_entropy: float) -> Tuple[Model, InfoDict]:
    
    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info

@functools.partial(jax.jit,
                   static_argnames=('backup_entropy', 'update_target'))
def _update_jit(
    rng: Key, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, update_target: bool
) -> Tuple[Key, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


def update_critic(key: Key, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float,
           backup_entropy: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy: # if using entropy
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

def update_actor(key: Key, actor: Model, critic: Model, temp: Model,
           batch: Batch) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, batch.observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

class SAC:
    def __init__(self, observations, actions, **kwargs):
        self.name = kwargs.pop('name')
        self.discount = kwargs.pop('discount')
        self.tau = kwargs.pop('tau')
        self.target_entropy = kwargs.pop('target_entropy')
        self.target_update_period = kwargs.pop('target_update_period')
        self.backup_entropy = kwargs.pop('backup_entropy')
        
        self.observations = observations
        self.actions = actions

        if self.target_entropy is None:
            self.target_entropy = -actions.shape[-1] / 2
        else:
            self.target_entropy = kwargs.pop('target_entropy')
            
        rng = jax.random.PRNGKey(seed=kwargs.pop('seed'))
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        
        actor_def = kwargs.pop('actor_def')(action_dim=actions.shape[-1], init_mean=kwargs.pop('init_mean'),
                                            final_fc_init_scale=kwargs.pop('policy_final_fc_init_scale'))
        critic_def = kwargs.pop('critic_def')
        
        actor = Model.create(model_def=actor_def, inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=kwargs.pop('actor_lr')))
        critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adam(learning_rate=kwargs.pop('critic_lr')))
        target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
        
        # For entropy param optimization
        temp = Model.create(Temperature(initial_temperature=kwargs.pop('init_temperature')),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=kwargs.pop('temp_lr')))
        
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        self.step = 1
        
    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> Array:
        rng, actions = _sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations, temperature)
        self.rng = rng
        
        return jnp.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
        
        
        
        
        
        
        
        
        