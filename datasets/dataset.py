import collections
from typing import Tuple, Union

import numpy as np
from tqdm.auto import tqdm
import jax

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

ICVF_output = collections.namedtuple(
    'ICVF_output',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'goals', 'next_goals'])#, 'low_goals', 'high_goals', 'high_targets'])

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)

def _sample(
    dataset_dict, indx: np.ndarray
):
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch
    
class Dataset:
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int, init_terminals: bool = True):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        
        if init_terminals:
            self.terminal_locs,  = np.nonzero(dones_float > 0)
        
        self.p_trajgoal = 0.7
        self.p_currgoal = 0.5
        
    def sample_goals(self, indx):
        # batch_size = len(indx)
        # # Random goals
        # goal_indx = np.random.randint(self.size, size=batch_size)
        # goal_indx = np.where(np.random.rand(batch_size) < 0.8, indx, goal_indx)
        # return goal_indx
        
        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.size, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
            
        distance = np.random.rand(batch_size)
        middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < self.p_trajgoal, middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < self.p_currgoal, indx, goal_indx)
        return goal_indx
        
    def add_data(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray):
        self.size += len(observations)
        self.observations = np.concatenate([self.observations, observations], axis=0)
        self.actions = np.concatenate([self.actions, actions], axis=0)
        self.rewards = np.concatenate([self.rewards, rewards], axis=0)
        self.masks = np.concatenate([self.masks, masks], axis=0)
        self.dones_float = np.concatenate([self.dones_float, dones_float], axis=0)
        self.next_observations = np.concatenate([self.next_observations, next_observations], axis=0)
        self.terminal_locs, = np.nonzero(self.dones_float > 0)
        return self
    
    def sample(self, batch_size: int, icvf: bool = False) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        if icvf:
            goal_indx = self.sample_goals(indx)            
            success = (indx == goal_indx)
            rewards = success.astype(float)*0.1 - 0.1 # - 1
            return ICVF_output(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=rewards,
                     masks=1.0 - success.astype(float),
                     goals=jax.tree_util.tree_map(lambda arr: arr[goal_indx], self.observations),
                     next_observations=self.next_observations[indx],
                     next_goals=jax.tree_util.tree_map(lambda arr: arr[goal_indx], self.next_observations))
            
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

    def get_iter(self, batch_size):
        #for i in range(self.size // batch_size):
        while True:
            yield self.sample(batch_size).observations[:, :2] ####
            
    def get_initial_states(
        self,
        and_action: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        states = []
        if and_action:
            actions = []
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        for traj in trajs:
            states.append(traj[0][0])
            if and_action:
                actions.append(traj[0][1])

        states = np.stack(states, 0)
        if and_action:
            actions = np.stack(actions, 0)
            return states, actions
        else:
            return states

    def get_monte_carlo_returns(self, discount) -> np.ndarray:
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        mc_returns = []
        for traj in trajs:
            mc_return = 0.0
            for i, (_, _, reward, _, _, _) in enumerate(traj):
                mc_return += reward * (discount**i)
            mc_returns.append(mc_return)

        return np.asarray(mc_returns)

    def take_top(self, percentile: float = 100.0):
        assert percentile > 0.0 and percentile <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        N = int(len(trajs) * percentile / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def take_random(self, percentage: float = 100.0):
        assert percentage > 0.0 and percentage <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        np.random.shuffle(trajs)

        N = int(len(trajs) * percentage / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def train_validation_split(self,
                               train_fraction: float = 0.8
                               ) -> Tuple['Dataset', 'Dataset']:
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        train_size = int(train_fraction * len(trajs))

        np.random.shuffle(trajs)

        (train_observations, train_actions, train_rewards, train_masks,
         train_dones_float,
         train_next_observations) = merge_trajectories(trajs[:train_size])

        (valid_observations, valid_actions, valid_rewards, valid_masks,
         valid_dones_float,
         valid_next_observations) = merge_trajectories(trajs[train_size:])

        train_dataset = Dataset(train_observations,
                                train_actions,
                                train_rewards,
                                train_masks,
                                train_dones_float,
                                train_next_observations,
                                size=len(train_observations))
        valid_dataset = Dataset(valid_observations,
                                valid_actions,
                                valid_rewards,
                                valid_masks,
                                valid_dones_float,
                                valid_next_observations,
                                size=len(valid_observations))

        return train_dataset, valid_dataset