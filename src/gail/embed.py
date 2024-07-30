from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode
from typing import List
from gail.disc import Discriminator
from agents.notdual import ENOTCustom, ENOTPotentialsCustom
from src.gail.rewards_transform import RewardsTransform
from src.gail.base import GAIL
from flax import linen as nn
import flax


class EncodersPair(PyTreeNode, ABC):

    @abstractmethod
    def agent_embed(self, x): pass

    @abstractmethod
    def expert_embed(self, x): pass


class EmbedGAIL(GAIL):

    encoders: EncodersPair
    potentials: ENOTPotentialsCustom
    enot: ENOTCustom = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, 
               disc: Discriminator, 
               rewards_transform: List[RewardsTransform], 
               encoders: EncodersPair,
               enot: ENOTCustom):
        return cls(disc=disc, rewards_transform=rewards_transform, 
                   encoders=encoders, potentials=enot.to_dual_potentials(), enot=enot)

    @jax.jit
    def update(self, y, next_y, x, next_x):

        # y = self.encoders.expert_embed(expert_obs)
        # next_y = self.encoders.expert_embed(next_expert_obs)
        # x = self.encoders.agent_embed(imitation_obs)
        # next_x = self.encoders.agent_embed(next_imitation_obs)

        y_pair = self.potentials.transport(jnp.concatenate([y, next_y], -1), forward=True)
        x_pair = jnp.concatenate([x, next_x], -1)

        new_obj, info = super().update(y_pair, x_pair)

        return new_obj, info
    
    @jax.jit
    def encode_pair(self, y, next_y, x, next_x):
        # encoded_source = self.encoders.expert_embed(expert_obs)
        # next_encoded_source = self.encoders.expert_embed(next_expert_obs)
        # encoded_target = self.encoders.agent_embed(imitation_obs)
        # next_encoded_target = self.encoders.agent_embed(next_imitation_obs)
        
        encoded_source_pair = jnp.concatenate([y, next_y], -1)
        encoded_target_pair = jnp.concatenate([x, next_x], -1)

        return encoded_source_pair, encoded_target_pair

    
    def update_ot(self, expert_embed, next_expert_embed, imitation_embed, next_imitation_embed):
        encoded_source_pair, encoded_target_pair = self.encode_pair(expert_embed, next_expert_embed, imitation_embed, next_imitation_embed)
        
        new_enot, loss_elems, w_dist_elems = self.enot.update(encoded_source_pair, encoded_target_pair)
        potentials = new_enot.to_dual_potentials(finetune_g=True)
        
        return self.replace(potentials=potentials, enot=new_enot)
        

    @jax.jit
    def predict_reward(self, imitation_obs, next_imitation_obs) -> jnp.ndarray:
        x = self.encoders.agent_embed(imitation_obs)
        next_x = self.encoders.agent_embed(next_imitation_obs)

        x_pair = jnp.concatenate([x, next_x], -1)

        return super().predict_reward(x_pair)