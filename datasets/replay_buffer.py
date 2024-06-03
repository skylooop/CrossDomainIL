from typing import Optional, Union

import gymnasium as gym
import numpy as np

from datasets.dataset import Dataset
import jax
import jax.numpy as jnp
import functools

class ReplayBuffer(Dataset):

    def __init__(self, observation_space: gym.spaces.Box,
                 action_space: Union[gym.spaces.Discrete,
                                     gym.spaces.Box], capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, *action_space.shape),
                           dtype=action_space.dtype)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0,
                         init_terminals=False)
        
        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int] = None):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples
        self.terminal_locs,  = np.nonzero(self.dones_float > 0)

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
#    @functools.partial(jax.jit, static_argnames=('batch_size'))
    def update_not_rewards(self, joint_ot_agent, potential_pairs, indx:int):
        observations = self.observations[indx]
        nobservations = self.next_observations[indx]
        encoded_target = joint_ot_agent.ema_get_phi_target(observations)
        encoded_target_next = joint_ot_agent.ema_get_phi_target(nobservations)
        # encoded_expert = jnp.concatenate(joint_ot_agent(expert_data.observations, method='ema_get_phi_source'), -1)
        # encoded_expert_next = jnp.concatenate(joint_ot_agent(expert_next_obs, method='ema_get_phi_source'), -1)
        f, g = potential_pairs.get_fg()
        reward = -jax.vmap(g)(jnp.concatenate([encoded_target, encoded_target_next], axis=-1)).mean()
        return reward
        
        