import jax
import jax.numpy as jnp


class SAC:
    def __init__(self, observations, actions, **kwargs):
        self.name = kwargs.pop('name')
        self.seed = kwargs.pop('seed')
        
        self.actor_lr = kwargs.pop('actor_lr')
        self.critic_lr = kwargs.pop('critic_lr')
        self.temp_lr = kwargs.pop('temp_lr')
        self.hidden_dims = kwargs.pop('hidden_dims')
        self.discount = kwargs.pop('discount')
        self.tau = kwargs.pop('tau')
        self.target_update_period = kwargs.pop('target_update_period')
        self.target_entropy = kwargs.pop('target_entropy')
        self.backup_entropy = kwargs.pop('backup_entropy')
        self.init_temperature = kwargs.pop('init_temperature')
        self.init_mean = kwargs.pop('init_mean')
        self.policy_final_fc_init_scale = kwargs.pop('policy_final_fc_init_scale')
        
        self.observations = observations
        self.actions = actions
        