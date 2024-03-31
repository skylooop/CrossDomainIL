import os
import warnings
warnings.filterwarnings('ignore')
os.environ['MUJOCO_GL']='egl'
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

from tqdm.auto import tqdm
import numpy as np
import random

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')

from utils.const import SUPPORTED_ENVS
from datasets.replay_buffer import ReplayBuffer

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo

@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="expert")
def collect_expert(cfg: DictConfig) -> None:
    #################
    # python src/run_expert.py hydra.job.chdir=True
    #################
    
    assert cfg.env.name in SUPPORTED_ENVS
    
    print(f"Collecting Expert data using {cfg.algo.name} on {cfg.env.name} env")
    print(OmegaConf.to_yaml(cfg))
    
    os.makedirs("saved_expert", exist_ok=True)
    print(f"Saving expert weights into {os.getcwd()}")
    
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    if cfg.env.name == "PointUMaze":
        from envs.maze_envs import CustomPointUMazeSize3Env
        episode_limit = 1000
        env = CustomPointUMazeSize3Env(render_mode='rgb_array')
        env = TimeLimit(env, max_episode_steps=episode_limit)
        env = RecordEpisodeStatistics(env)
        env = RecordVideo(env, video_folder='agent_video', step_trigger=lambda _: _ % cfg.video_log_interval == 0 and _ > cfg.video_log_interval)
        
    expert_agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, cfg.algo.buffer_size)
    
    eval_returns = []
    (observation, info), done = env.reset(), False
    
    for i in tqdm(range(1, cfg.max_steps + 1), smoothing=0.1):
        if i < cfg.algo.start_training:
            action = env.action_space.sample()
        else:
            action = expert_agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if not done:
            mask = 1.0
        else:
            mask = 0.0
        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation
        
        if done:
            for k,v in info['episode'].items():
                print(f"{k}/{v}")
            (observation, info), done = env.reset(), False
            
        if i >= cfg.algo.start_training:
            for _ in range(cfg.algo.updates_per_step):
                batch = replay_buffer.sample(cfg.algo.batch_size)
                update_info = expert_agent.update(batch)

            if i % cfg.log_interval == 0:
                for k, v in update_info.items():
                    print(f'training/{k}', v, i)
        
if __name__ == "__main__":
    collect_expert()