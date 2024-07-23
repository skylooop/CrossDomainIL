from functools import partial
from typing import Dict, Tuple, Any, Callable
import jax
import jax.numpy as jnp
from flax import linen as flax_nn
import optax
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
import numpy as np
from flax.struct import PyTreeNode
import flax.linen as nn
from jaxrl_m.common import TrainState
from jaxrl_m.networks import MLP
import jax.nn as F
import functools

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred) 
    return loss


class Discriminator(PyTreeNode):
    state: TrainState
    key: jax.Array
    penalty_coef: float

    @classmethod
    def create(
        cls,
        obs: jnp.ndarray,
        learning_rate: float,
        discr_updates: int,
        transition_steps_decay: int,
        discr_final_lr: float,
        hidden_dims: list,
        l2_loss: float = 0.0,
        schedule_type: str = "linear",
        penalty_coef: float = 10.0):

        def scheduler(step_number):
            lr = jax.lax.select(
                step_number == 0,
                learning_rate,
                learning_rate / jnp.maximum((step_number // discr_updates), 1),
            )
            return lr

        rng = jax.random.PRNGKey(1)
        model = MLP(hidden_dims, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        params = model.init(rng, obs)['params']
        
        if schedule_type == "linear":
            schedule = optax.linear_schedule(
                init_value=learning_rate,
                end_value=discr_final_lr,
                transition_steps=transition_steps_decay * discr_updates,
            )
        elif schedule_type == "constant":
            schedule = learning_rate
        elif schedule_type == "harmonic":
            schedule = scheduler
        else:
            raise ValueError(f"Schedule type {schedule_type} not recognized")

        net = TrainState.create(
            model_def=model,
            params=params,
            tx=optax.adamw(learning_rate=schedule, weight_decay=l2_loss, eps=1e-5),
        )

        return cls(state=net, key=rng, penalty_coef=penalty_coef)
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames='penalty_coef')
    def batch_loss(
        state,
        params: FrozenDict,
        expert_batch: jnp.ndarray,
        imitation_batch: jnp.ndarray,
        key: Any,
        penalty_coef: float
    ) -> jnp.ndarray:
        def apply_disc(transition: jnp.ndarray) -> jnp.ndarray:
            return state(transition, params=params)

        def apply_scalar(params: FrozenDict, x: jnp.ndarray):
            return state(x, params=params)[0]

        def interpolate(alpha: float, expert_batch: jnp.ndarray, imitation_batch: jnp.ndarray):
            return alpha * expert_batch + (1 - alpha) * imitation_batch

        d_loss = d_logistic_loss(
            jax.vmap(apply_disc)(expert_batch), 
            jax.vmap(apply_disc)(imitation_batch)
        )
        alpha = jax.random.uniform(key, (imitation_batch.shape[0], 1))

        interpolated = jax.vmap(interpolate)(alpha, expert_batch, imitation_batch)
        gradients = jax.vmap(jax.grad(fun=apply_scalar, argnums=1), (None, 0))(
            params, interpolated
        )
        gradients = gradients.reshape((expert_batch.shape[0], -1))
        gradients_norm = jnp.sqrt(jnp.sum(gradients**2, axis=1) + 1e-12)
        grad_penalty = ((gradients_norm) ** 2).mean()

        # here we use 10 as a fixed parameter as a cost of the penalty.
        loss = d_loss + penalty_coef * grad_penalty

        return loss

    @jax.jit
    def update_step(self, expert_batch, imitation_batch):

        def loss_fn(params):
            info = {}
            loss = self.batch_loss(self.state, params, expert_batch, imitation_batch, self.key, self.penalty_coef)
            return loss, info

        new_state, info = self.state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        new_key, _ = jax.random.split(self.key)

        return self.replace(state=new_state, key=new_key), info
    
    def generator_losses(self, x) -> jnp.ndarray:
        d = self.state(x, params=self.state.params)
        loss = g_nonsaturating_loss(d).reshape(-1)
        return loss  