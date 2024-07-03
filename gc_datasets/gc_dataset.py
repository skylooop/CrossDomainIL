from datasets.dataset import Dataset, Batch, ICVF_output
import dataclasses
import numpy as np
import jax
import ml_collections
import collections

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float = 0.99
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    curr_goal_shift: int = 0
    
    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset.dones_float > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size-self.curr_goal_shift, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
            
        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        rewards = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            masks = (1.0 - success.astype(float))
        else:
            masks = np.ones(batch_size)
        goals = jax.tree_map(lambda arr: arr[goal_indx+self.curr_goal_shift], self.dataset.observations)

        return ICVF_output(observations=batch.observations, next_observations=batch.next_observations,
                           actions=batch.actions, rewards=rewards, masks=masks, goals=goals)

@dataclasses.dataclass
class GCSDataset(GCDataset):
    way_steps: int = 1
    intent_sametraj: bool = False
    high_p_randomgoal: float = 0
    
    @staticmethod
    def get_default_config():
        # FROM HIQL PAPER
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 1,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
        })
    
    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, icvf=False)
        goal_indx = self.sample_goals(indx)
        success = (indx == goal_indx)

        rewards = success.astype(float) * self.reward_scale + self.reward_shift
        
        if self.terminal:
            masks = (1.0 - success.astype(float))
        else:
            masks = np.ones(batch_size)
            
        goals = jax.tree_map(lambda arr: arr[goal_indx], self.dataset.observations)
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        way_indx = np.minimum(indx + self.way_steps, final_state_indx) # on trajectory
        low_goals = jax.tree_map(lambda arr: arr[way_indx], self.dataset.observations) # s_{t+k}

        distance = np.random.rand(batch_size)

        high_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        high_traj_target_indx = np.minimum(indx + self.way_steps, high_traj_goal_indx)

        high_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        high_random_target_indx = np.minimum(indx + self.way_steps, final_state_indx)

        pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
        high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
        high_target_idx = np.where(pick_random, high_random_target_indx, high_traj_target_indx)

        high_goals = jax.tree_map(lambda arr: arr[high_goal_idx], self.dataset.observations)
        high_targets = jax.tree_map(lambda arr: arr[high_target_idx], self.dataset.observations)

        return ICVF_output(observations=batch.observations, 
                           next_observations=batch.next_observations,
                           goals=goals,
                           rewards=rewards,
                           masks=masks,
                           actions=batch.actions)
                        #    low_goals=low_goals,
                        #    high_goals=high_goals,
                        #    high_targets=high_targets)

