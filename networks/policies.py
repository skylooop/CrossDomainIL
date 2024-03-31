import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

import optax

import jaxtyping
from typing import Sequence, Union, Any, Optional

import distrax
from distrax import Distribution

from networks.common import MLP, default_init
import functools

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None
    
    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = distrax.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return distrax.Transformed(distribution=base_dist,
                                               bijector=distrax.Block(distrax.Tanh(), 1))
        else:
            return base_dist
    
@functools.partial(jax.jit, static_argnames=('actor_apply_fn'))
def _sample_actions(rng, actor_apply_fn, actor_params, observations, temperature):
    dist = actor_apply_fn({"params": actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key) 