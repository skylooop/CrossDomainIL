import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array
import functools

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

ModuleMethod = Union[
    str, Callable, None
]  # A method to be passed into TrainState.__call__
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x

class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x

class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class TrainState(flax.struct.PyTreeNode):
    """
    Core abstraction of a model in this repository.

    Creation:
    ```
        model_def = nn.Dense(12) # or any other flax.linen Module
        params = model_def.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))['params']
        model = TrainState.create(model_def, params, tx=None) # Optionally, pass in an optax optimizer
    ```

    Usage:
    ```
        y = model(jnp.ones((1, 4))) # By default, uses the `__call__` method of the model_def and params stored in TrainState
        y = model(jnp.ones((1, 4)), params=params) # You can pass in params (useful for gradient computation)
        y = model(jnp.ones((1, 4)), method=method) # You can apply a different method as well
    ```

    More complete example:
    ```
        def loss(params):
            y_pred = model(x, params=params)
            return jnp.mean((y - y_pred) ** 2)

        grads = jax.grad(loss)(model.params)
        new_model = model.apply_gradients(grads=grads) # Alternatively, new_model = model.apply_loss_fn(loss_fn=loss)
    ```
    """

    step: int
    apply_fn: Callable[..., Any] = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Params
    tx: Optional[optax.GradientTransformation] = nonpytree_field()
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Params,
        tx: Optional[optax.GradientTransformation] = None,
        **kwargs,
    ) -> "TrainState":
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        params=None,
        extra_variables: dict = None,
        method: ModuleMethod = None,
        **kwargs,
    ):
        """
        Internally calls model_def.apply_fn with the following logic:

        Arguments:
            params: If not None, use these params instead of the ones stored in the model.
            extra_variables: Additional variables to pass into apply_fn
            method: If None, use the `__call__` method of the model_def. If a string, uses
                the method of the model_def with that name (e.g. 'encode' -> model_def.encode).
                If a function, uses that function.

        """
        if params is None:
            params = self.params

        variables = {"params": params}

        if extra_variables is not None:
            variables = {**variables, **extra_variables}

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `opt_state` updated by applying `grads`, and additional attributes
            replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        """
        Takes a gradient step towards minimizing `loss_fn`. Internally, this calls
        `jax.grad` followed by `TrainState.apply_gradients`. If pmap_axis is provided,
        additionally it averages gradients (and info) across devices before performing update.
        """
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)

            return self.apply_gradients(grads=grads), info

        else:
            grads = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)

# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state

@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
