from typing import Optional, Union

import gymnasium as gym
import numpy as np

from gc_datasets.dataset import Dataset, Batch

def apply_noise_to_expert(expert_states):
    sizes = expert_states.shape[0] // 4
    # Apply additive noise (Gaussian noise)
    gaussian_noisy = expert_states[:sizes] + np.random.randn(*expert_states[:sizes].shape) * 0.1
    # Apply multiplicative noise (b = 0)
    multiplicative_noise = np.random.randn(expert_states[sizes:2*sizes].shape[0], expert_states[sizes:2*sizes].shape[0]) @ expert_states[sizes:2*sizes]
    # Doubly stochastic
    A = np.random.randn(expert_states[2*sizes:3*sizes].shape[0], expert_states[2*sizes:3*sizes].shape[0])
    rsum = None
    csum = None
    it = 0
    while ((np.any(rsum != 1)) | (np.any(csum != 1))) and it < 50:
        A /= A.sum(0)
        A = A / A.sum(1)[:, np.newaxis]
        rsum = A.sum(1)
        csum = A.sum(0)
        it += 1
    doubly_stochastic_noise = A @ expert_states[2*sizes:3*sizes]
    # Shuffle
    def shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)
    shuffled_noised = shuffle_along_axis(expert_states[3*sizes:], -1)
    expert_data = np.concatenate([gaussian_noisy, multiplicative_noise, doubly_stochastic_noise, shuffled_noised], axis=0)
    return expert_data #gaussian_noisy, multiplicative_noise, doubly_stochastic_noise, shuffled_noised

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

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
        self.capacity = capacity
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

    def apply_noise(self):
        self.observations = apply_noise_to_expert(self.observations[:self.size])
        self.next_observations = np.roll(self.observations, 1, 0)
    
    def random_shuffle(self):
        np.random.shuffle(self.observations[:self.size])#np.concatenate([np.random.shuffle(self.observations[:self.size]), np.empty((self.capacity - self.size, self.observations.shape[-1]))])
        self.next_observations = np.roll(self.observations, 1, 0)

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
        self.terminal_locs,  = np.nonzero(self.dones_float > 0)
