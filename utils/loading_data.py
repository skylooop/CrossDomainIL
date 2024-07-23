import numpy as np
from gc_datasets.replay_buffer import Dataset

def prepare_buffers_for_il(cfg, custom_npy: bool=True,
                           clip_to_eps:bool = True, eps:float=1e-5):
    
    if custom_npy:
        expert_source = np.load(cfg.imitation_env.path_to_expert, allow_pickle=True).item()
    
        expert_source['dones'][-1] = 1
        
        lim = 1 - eps
        expert_source['actions'] = np.clip(expert_source['actions'], -lim, lim)
        expert_source['observations'] = expert_source['observations'].astype(np.float32)
        expert_source['next_observations'] = expert_source['next_observations'].astype(np.float32)
        
        expert_source_ds = Dataset(observations=expert_source['observations'],
                            actions=expert_source['actions'],
                            rewards=expert_source['rewards'],
                            dones_float=expert_source['dones'],
                            masks=1.0 - expert_source['dones'],
                            next_observations=expert_source['next_observations'],
                            size=expert_source['observations'].shape[0])
        if cfg.imitation_env.name != "Hopper":
            expert_random = np.load(cfg.imitation_env.path_to_random_expert, allow_pickle=True).item()
            target_random = np.load(cfg.imitation_env.path_to_random_target, allow_pickle=True).item()
            
            expert_random['dones'][-1] = 1
            target_random['dones'][-1] = 1

            expert_random['observations'] = expert_random['observations'].astype(np.float32)
            expert_random['next_observations'] = expert_random['next_observations'].astype(np.float32)

            target_random['observations'] = target_random['observations'].astype(np.float32)
            target_random['next_observations'] = target_random['next_observations'].astype(np.float32)

            expert_random['actions'] = np.clip(expert_random['actions'], -lim, lim)
            target_random['actions'] = np.clip(target_random['actions'], -lim, lim)

            non_expert_source_dataset = Dataset(observations=expert_random['observations'],
                                actions=expert_random['actions'],
                                rewards=expert_random['rewards'],
                                dones_float=expert_random['dones'],
                                masks=1.0 - expert_random['dones'],
                                next_observations=expert_random['next_observations'],
                                size=expert_random['observations'].shape[0])
            
            expert_source_ds_2 = Dataset(observations=expert_source['observations'],
                                actions=expert_source['actions'],
                                rewards=expert_source['rewards'],
                                dones_float=expert_source['dones'],
                                masks=1.0 - expert_source['dones'],
                                next_observations=expert_source['next_observations'],
                                size=expert_source['observations'].shape[0])
            
            combined_source_ds = expert_source_ds_2.add_data(observations=expert_random['observations'],
                                actions=expert_random['actions'],
                                rewards=expert_random['rewards'],
                                dones_float=expert_random['dones'],
                                masks=1.0 - expert_random['dones'],
                                next_observations=expert_random['next_observations'])
            return expert_source_ds, non_expert_source_dataset, combined_source_ds
        
        return expert_source_ds