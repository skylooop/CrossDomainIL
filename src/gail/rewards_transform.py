from abc import abstractmethod
from flax.struct import PyTreeNode
from jax import numpy as jnp
import jax


class RewardsTransform(PyTreeNode):

    @abstractmethod
    def transform(self, r: jnp.ndarray) -> jnp.ndarray: pass
    @abstractmethod
    def update(self, r: jnp.ndarray): pass 


class RewardsStandartisation(RewardsTransform):
    mean: jnp.ndarray = 0.0
    var: jnp.ndarray = 1.0

    def update(self, r: jnp.ndarray):
        new_mean = self.mean * 0.99 + r.mean() * 0.01
        new_var = self.var * 0.99 + ((r - new_mean) ** 2).mean() * 0.01

        return self.replace(mean=new_mean, var=new_var)
    
    def transform(self, r: jnp.ndarray) -> jnp.ndarray:
        return (r - self.mean) / jnp.sqrt(self.var + 1e-10)
    

class NegativeShift(RewardsTransform):
    shift: jnp.ndarray = 1.0

    def update(self, r: jnp.ndarray):
        q = jnp.quantile(r, 0.9)
        return self.replace(shift = jnp.maximum(q, self.shift))
    
    def transform(self, r: jnp.ndarray) -> jnp.ndarray:
        return r - self.shift


