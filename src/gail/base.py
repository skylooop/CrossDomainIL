import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode
from agents.disc import Discriminator
from typing import List
from src.gail.rewards_transform import RewardsTransform


class GAIL(PyTreeNode):

    disc: Discriminator
    rewards_transform: List[RewardsTransform]

    @classmethod
    def create(cls, disc, rewards_transform: List[RewardsTransform]):
        return cls(disc=disc, rewards_transform=rewards_transform)

    @jax.jit
    def update(self, expert_batch, imitation_batch):
        rewards = -self.disc.generator_losses(imitation_batch)
        new_transform = []

        for t in self.rewards_transform:
            new_transform.append(t.update(rewards))
            rewards = t.transform(rewards)

        new_disc, info = self.disc.update_step(expert_batch, imitation_batch) 
        return self.replace(disc=new_disc, rewards_transform=new_transform), info

    def predict_reward(self, x) -> jnp.ndarray:
        rewards = -self.disc.generator_losses(x)
        for t in self.rewards_transform:
            rewards = t.transform(rewards)

        return rewards
    

    