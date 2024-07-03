from functools import partial
from typing import Tuple, Any, Callable
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


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred) 
    return loss


class DiscModel(MLP):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = super().__call__(x)
        # x = nn.sigmoid(x)
        return x


class Discriminator(PyTreeNode):
    state: TrainState

    @classmethod
    def create(
        cls,
        obs: jnp.ndarray,
        learning_rate: float,
        discr_updates: int,
        transition_steps_decay: int,
        discr_final_lr: float,
        l2_loss: float = 0.0,
        schedule_type: str = "linear"):

        def scheduler(step_number):
            lr = jax.lax.select(
                step_number == 0,
                learning_rate,
                learning_rate / jnp.maximum((step_number // discr_updates), 1),
            )
            return lr

        rng = jax.random.PRNGKey(1)
        model = DiscModel([256, 256, 256, 1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
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

        return cls(state=net), rng

    @staticmethod
    def batch_loss(
        state,
        params: FrozenDict,
        expert_batch: jnp.ndarray,
        imitation_batch: jnp.ndarray,
        key: Any,
        discr_loss: str,
    ) -> jnp.ndarray:
        def loss_exp_bce(
            expert_transition: jnp.ndarray,
        ) -> jnp.ndarray:
            exp_d = state(expert_transition, params=params)
            # exp_loss = optax.sigmoid_binary_cross_entropy(exp_d, jnp.ones(expert_transition.shape[0]) * 1.0)
            return exp_d

        def loss_imit_bce(imitation_transition: jnp.ndarray) -> jnp.ndarray:
            imit_d = state(imitation_transition, params=params)
            # imit_loss = optax.sigmoid_binary_cross_entropy(imit_d, jnp.zeros(imitation_transition.shape[0]) * 1.0)
            return imit_d

        def loss_exp_mse(
            expert_transition: jnp.ndarray,
        ) -> jnp.ndarray:
            exp_d =  state(expert_transition, params=params)
            target = jnp.tile(jnp.array([1.0]), (exp_d.shape[0],))
            return optax.l2_loss(exp_d, target)

        def loss_imit_mse(imitation_transition: jnp.ndarray) -> jnp.ndarray:
            imit_d = state(imitation_transition, params=params)
            target = jnp.tile(jnp.array([0.0]), (imit_d.shape[0],))
            return optax.l2_loss(imit_d, target)

        def apply_scalar(
            params: FrozenDict,
            input: jnp.ndarray,
        ):
            return state(input,params=params )[0]

        def interpolate(
            alpha: float,
            expert_batch: jnp.ndarray,
            imitation_batch: jnp.ndarray,
        ):
            return alpha * expert_batch + (1 - alpha) * imitation_batch

        if discr_loss == "mse":
            loss_expert = loss_exp_mse
            loss_imitation = loss_imit_mse
        elif discr_loss == "bce":
            loss_expert = loss_exp_bce
            loss_imitation = loss_imit_bce

        # exp_loss = jnp.mean(jax.vmap(loss_expert)(expert_batch))
        # imit_loss = jnp.mean(jax.vmap(loss_imitation)(imitation_batch))
        d_loss = d_logistic_loss(
            jax.vmap(loss_expert)(expert_batch), 
            jax.vmap(loss_imitation)(imitation_batch)
        )
        alpha = jax.random.uniform(key, (imitation_batch.shape[0],))

        interpolated = jax.vmap(interpolate)(alpha, expert_batch, imitation_batch)
        gradients = jax.vmap(jax.grad(fun=apply_scalar, argnums=1), (None, 0))(
            params, interpolated
        )
        gradients = gradients.reshape((expert_batch.shape[0], -1))
        gradients_norm = jnp.sqrt(jnp.sum(gradients**2, axis=1) + 1e-12)
        grad_penalty = ((gradients_norm) ** 2).mean()

        # here we use 10 as a fixed parameter as a cost of the penalty.
        loss = d_loss + 10 * grad_penalty

        return loss

    @jax.jit
    def update_step(self, expert_batch, imitation_batch, norm_mean, norm_var, key):
        
        norm_expert_batch = (expert_batch - norm_mean) / jnp.sqrt(
            norm_var + 1e-8
        )

        def loss_fn(params):
            info = {}
            loss = self.batch_loss(self.state, params, norm_expert_batch, imitation_batch, key, "bce")
            return loss, info

        new_state, info = self.state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        new_key, key = jax.random.split(key)

        return self.replace(state=new_state), new_key, info

    def predict_reward(
        self,
        input,
    ) -> jnp.ndarray:
        d = self.state(input, params=self.state.params)
        return - g_nonsaturating_loss(d).reshape(-1)
    

    