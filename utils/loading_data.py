import numpy as np
from datasets.replay_buffer import ReplayBuffer, Dataset

def prepare_buffers_for_il(cfg, custom_npy: bool=True, target_antmaze: bool=True):
    
    if custom_npy:
        expert_source = np.load(cfg.imitation_env.path_to_expert, allow_pickle=True).item()
        expert_random = np.load(cfg.imitation_env.path_to_random_expert, allow_pickle=True).item()
        target_random = np.load(cfg.imitation_env.path_to_random_target, allow_pickle=True).item()

        
        expert_source['dones'][-1] = 1
        expert_random['dones'][-1] = 1
        target_random['dones'][-1] = 1
        
        target_random['observations'] = target_random['observations'].astype(np.float32)
        target_random['next_observations'] = target_random['next_observations'].astype(np.float32)
        
        expert_source['observations'] = expert_source['observations'].astype(np.float32)
        expert_source['next_observations'] = expert_source['next_observations'].astype(np.float32)
        
        expert_random['observations'] = expert_random['observations'].astype(np.float32)
        expert_random['next_observations'] = expert_random['next_observations'].astype(np.float32)
        
        expert_source_ds = Dataset(observations=expert_source['observations'],
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
        
        expert_source_ds2 = Dataset(observations=expert_source['observations'],
                            actions=expert_source['actions'],
                            rewards=expert_source['rewards'],
                            dones_float=expert_source['dones'],
                            masks=1.0 - expert_source['dones'],
                            next_observations=expert_source['next_observations'],
                            size=expert_source['observations'].shape[0])
        
        combined_source_ds = expert_source_ds2.add_data(observations=expert_random['observations'],
                            actions=expert_random['actions'],
                            rewards=expert_random['rewards'],
                            dones_float=expert_random['dones'],
                            masks=1.0 - expert_random['dones'],
                            next_observations=expert_random['next_observations'])
        
        target_dataset_random = Dataset(observations=target_random['observations'],
                            actions=target_random['actions'],
                            rewards=target_random['rewards'],
                            dones_float=target_random['dones'],
                            masks=1.0 - target_random['dones'],
                            next_observations=target_random['next_observations'],
                            size=target_random['observations'].shape[0])
        
        return expert_source_ds, non_expert_source_dataset, combined_source_ds, target_dataset_random
