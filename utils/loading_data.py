import numpy as np
from datasets.replay_buffer import ReplayBuffer, Dataset
import gymnasium as gym

def prepare_buffers_for_il(cfg):
    expert_source = np.load(cfg.imitation_env.path_to_expert, allow_pickle=True).item()
    expert_random = np.load(cfg.imitation_env.path_to_random_expert, allow_pickle=True).item()
    source_random = np.load(cfg.imitation_env.path_to_random_target, allow_pickle=True).item()
    
    source_dataset = Dataset(observations=expert_source['observations'],
                           actions=expert_source['actions'],
                           rewards=expert_source['rewards'],
                           dones_float=expert_source['dones'],
                           masks=1.0 - expert_source['dones'],
                           next_observations=expert_source['next_observations'],
                           size=expert_source['observations'].shape[0])
    
    non_expert_source_dataset = Dataset(observations=expert_random['observations'],
                           actions=expert_random['actions'],
                           rewards=expert_random['rewards'],
                           dones_float=expert_random['dones'],
                           masks=1.0 - expert_random['dones'],
                           next_observations=expert_random['next_observations'],
                           size=expert_random['observations'].shape[0])
    
    target_dataset_random = Dataset(observations=source_random['observations'],
                           actions=source_random['actions'],
                           rewards=source_random['rewards'],
                           dones_float=source_random['dones'],
                           masks=1.0 - source_random['dones'],
                           next_observations=source_random['next_observations'],
                           size=source_random['observations'].shape[0])
    source_expert_buffer = ReplayBuffer(observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(source_dataset.observations.shape[-1], )), action_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(source_dataset.actions.shape[-1], )), capacity=cfg.algo.buffer_size)
    source_expert_buffer.initialize_with_dataset(source_dataset)
    source_random_buffer = ReplayBuffer(observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(source_dataset.observations.shape[-1], )), action_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(source_dataset.actions.shape[-1], )), capacity=cfg.algo.buffer_size)
    source_random_buffer.initialize_with_dataset(non_expert_source_dataset)
    target_random_buffer = ReplayBuffer(observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_dataset_random.observations.shape[-1], )), action_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_dataset_random.actions.shape[-1], )), capacity=cfg.algo.buffer_size)
    target_random_buffer.initialize_with_dataset(target_dataset_random)
    
    return source_expert_buffer, source_random_buffer, target_random_buffer
