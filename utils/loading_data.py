import numpy as np
from gc_datasets.replay_buffer import ReplayBuffer, Dataset

def prepare_buffers_for_il(cfg, target_obs_space, target_act_space, custom_npy: bool=True,
                           clip_to_eps:bool = True, eps:float=1e-5, target_antmaze:bool=True):
    #if target env antmaze -> set to true
    
    if custom_npy:
        expert_source = np.load(cfg.imitation_env.path_to_expert, allow_pickle=True).item()
        expert_random = np.load(cfg.imitation_env.path_to_random_expert, allow_pickle=True).item()
        target_random = np.load(cfg.imitation_env.path_to_random_target, allow_pickle=True).item()

        expert_source['dones'][-1] = 1
        expert_random['dones'][-1] = 1
        target_random['dones'][-1] = 1
        
        if clip_to_eps:
            lim = 1 - eps
            expert_source['actions'] = np.clip(expert_source['actions'], -lim, lim)
            expert_random['actions'] = np.clip(expert_random['actions'], -lim, lim)
            target_random['actions'] = np.clip(target_random['actions'], -lim, lim)
        
        if target_antmaze:
            dones_float = np.zeros_like(target_random['rewards'])
            traj_ends = np.zeros_like(target_random['rewards'])

            for i in range(len(dones_float) - 1):
                traj_end = (np.linalg.norm(target_random['observations'][i + 1] - target_random['next_observations'][i]) > 1e-6)
                traj_ends[i] = traj_end
                dones_float[i] = int(traj_end or target_random['dones'][i] == 1.0)
            dones_float[-1] = 1
            traj_ends[-1] = 1
        else:
            dones_float = target_random['dones'].copy()
            traj_ends = target_random['dones'].copy()
        
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
        
        # target_dataset_random = Dataset(observations=target_random['observations'],
        #                     actions=target_random['actions'],
        #                     rewards=target_random['rewards'],
        #                     dones_float=dones_float,
        #                     masks=1.0 - target_random['dones'],
        #                     next_observations=target_random['next_observations'],
        #                     size=target_random['observations'].shape[0])
        
        # target_random_buffer = ReplayBuffer(observation_space=target_obs_space, action_space=target_act_space, capacity=target_dataset_random.size)
        # target_random_buffer.initialize_with_dataset(target_dataset_random)
        return expert_source_ds, non_expert_source_dataset, combined_source_ds
