import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac import MlpPolicy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import os
os.environ['MUJOCO_GL']='egl'
import rootutils
ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')

SEED = 42
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import datasets
from imitation.data import huggingface_utils

# Download some expert trajectories from the HuggingFace Datasets Hub.
# dataset = datasets.load_dataset("HumanCompatibleAI/ppo-Pendulum-v1")
# env = gym.make("Pendulum-v1", render_mode="rgb_array")

from envs.maze_envs import CustomPointUMazeSize3Env


def load_expert(path, clip_to_eps: bool = True, eps: float=1e-5):
    expert_source = np.load(path, allow_pickle=True).item()
    expert_source['dones'][-1] = 1
    
    if clip_to_eps:
        lim = 1 - eps
        expert_source['actions'] = np.clip(expert_source['actions'], -lim, lim)
    
    
    expert_source['observations'] = expert_source['observations'].astype(np.float32)
    expert_source['next_observations'] = expert_source['next_observations'].astype(np.float32)

    return expert_source
        

gym.register(
    id="custom/ustomPointUMazeSize3Env",
    entry_point=CustomPointUMazeSize3Env,  # This can also be the path to the class, e.g. `observation_matching:ObservationMatchingEnv`
    max_episode_steps=1000,
)

env = gym.make("custom/ustomPointUMazeSize3Env", render_mode="rgb_array", disable_env_checker=True)
s0, _ = env.reset(seed=SEED, options={})

expert_data = load_expert("/home/m_bobrin/CrossDomainIL/prep_data/pointumaze/expert_source/trained_expert.npy")

T, D = expert_data['observations'].shape

split_expert_obs =np.stack(
    [traj for traj in np.split(expert_data['observations'], np.nonzero(expert_data['dones'] == 1)[0]) if len(traj) == 18]
)


d2 = datasets.Dataset.from_dict({
    "obs": split_expert_obs, 
    "acts": np.zeros((split_expert_obs.shape[0], split_expert_obs.shape[1] - 1, env.action_space.shape[-1])), 
    "infos": [["{}"] * (len(traj) - 1) for traj in split_expert_obs],
    "terminal": np.ones((len(split_expert_obs),)),
})

# Convert the dataset to a format usable by the imitation library.
expert_trajectories = huggingface_utils.TrajectoryDatasetSequence(d2)


def _make_env():
    _env = gym.make("custom/ustomPointUMazeSize3Env")
    _env = RolloutInfoWrapper(_env)
    return _env

venv = DummyVecEnv([_make_env for _ in range(4)])

from stable_baselines3 import sac


learner = sac.SAC(
    env=venv,
    policy="MlpPolicy",
    batch_size=128,
    gamma=0.99,
    learning_rate=2e-4,
    seed=SEED,
    device="cuda",
    verbose=0
)
reward_net = BasicRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
    use_action=False,
    use_next_state=True,
    hid_sizes = (128, 128)
)

gail_trainer = GAIL(
    demonstrations=expert_trajectories,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512 * 5,
    n_disc_updates_per_round=1,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True
)


learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 20, return_episode_rewards=True,
)

# train the learner and evaluate again
gail_trainer.train(50_000)  # Train for 800_000 steps to match expert.
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 20, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))

import gymnasium as gym
from gymnasium.utils.save_video import save_video

step_starting_index = 0
episode_index = 0
frames = []
st = s0
states = None
episode_starts = np.ones((1,), dtype=bool)

obs = s0

for i in range(1000):
   action, _state = learner.predict(obs, deterministic=True)
   obs, _, _, _, _ = env.step(action)
   frames.append(env.render())


save_video(
    frames=frames,
    video_folder="videos",
    fps=20,
    step_starting_index=step_starting_index,
    episode_index=episode_index
)

