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

import wandb

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
    
    print(f"Saving expert weights into {os.getcwd()}")
    wandb.init(
        mode='offline',
        project=cfg.logger.project,
        config=dict(cfg),
        group="expert_" + f"{cfg.env.name}_{cfg.algo.name}",
    )
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    if cfg.env.name == "PointUMaze":
        from envs.maze_envs import CustomPointUMazeSize3Env
        env = CustomPointUMazeSize3Env()
        eval_env = CustomPointUMazeSize3Env()
        episode_limit = 1000
        
    elif cfg.env.name == "PointAntUMaze":
        from envs.maze_envs import CustomAntUMazeSize3Env
        episode_limit = 1000
        env = CustomAntUMazeSize3Env()
        eval_env = CustomAntUMazeSize3Env()
        
    elif cfg.env.name == 'InvertedPendulum-v2':
        from envs.envs import ExpertInvertedPendulumEnv
        env = ExpertInvertedPendulumEnv()
        eval_env = ExpertInvertedPendulumEnv()
        episode_limit = 1000
        
    elif cfg.env.name == 'InvertedDoublePendulum-v2':
        from envs.envs import ExpertInvertedDoublePendulumEnv
        env = ExpertInvertedDoublePendulumEnv()
        eval_env = ExpertInvertedDoublePendulumEnv()
        episode_limit = 1000
        
    elif cfg.env.name == 'Reacher2-v2':
        from envs.more_envs import CustomReacher2Env
        env = CustomReacher2Env(l2_penalty=True)
        eval_env = CustomReacher2Env(l2_penalty=True)
        episode_limit = 50
        
    elif cfg.env.name == 'Reacher3-v2':
        from envs.more_envs import CustomReacher3Env
        env = CustomReacher3Env(l2_penalty=True)
        eval_env = CustomReacher3Env(l2_penalty=True)
        episode_limit = 50
        
    elif cfg.env.name == 'HalfCheetah-v2':
        from envs.envs import ExpertHalfCheetahNCEnv
        env = ExpertHalfCheetahNCEnv()
        eval_env = ExpertHalfCheetahNCEnv()
        episode_limit = 1000
        
    env = TimeLimit(env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: (x + 1) % cfg.save_video_each_ep == 0)
    eval_env = RecordEpisodeStatistics(eval_env, deque_size=cfg.save_expert_episodes)
    
    expert_agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, cfg.algo.buffer_size)
    
    (observation, info), done = env.reset(seed=cfg.seed), False
    if cfg.train_expert:
        for i in tqdm(range(1, cfg.max_steps + 1)):
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
                wandb.log({f"Training/rewards": info['episode']['r'],
                        f"Training/episode length": info['episode']['l']}, step=i)
                (observation, info), done = env.reset(), False
                
            if i >= cfg.algo.start_training:
                for _ in range(cfg.algo.updates_per_step):
                    batch = replay_buffer.sample(cfg.algo.batch_size)
                    update_info = expert_agent.update(batch)

                if i % cfg.log_interval == 0:
                    wandb.log({f"Training/Critic loss": update_info['critic_loss'],
                            f"Training/Actor loss": update_info['actor_loss'],
                            f"Training/Entropy": update_info['entropy'],
                            f"Training/Temperature": update_info['temperature'],
                            f"Training/Temp loss": update_info['temp_loss']}, step=i)
                    
            if i % cfg.eval_interval == 0:
                eval_stats = evaluate(expert_agent, eval_env, cfg.eval_episodes)
                print(eval_stats)
                wandb.log({"Evaluation/rewards": eval_stats['r'],
                        "Evaluation/length": eval_stats['l']})
                
        save_expert(expert_agent, eval_env, cfg.save_expert_episodes, visual=False)
        
    print(f"Saving random policy")
    save_random_policy(eval_env, 10, visual=False)
    
def save_random_policy(env, n_episodes, visual):
    obs, nobs, rews, acts, dones, viz_obs = [], [], [], [], [], []
    
    for _ in tqdm(range(n_episodes), desc="Running random policy..."):
        (observation, info), done = env.reset(), False
        while not done:
            action = env.action_space.sample()
            obs.append(observation)
            acts.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            nobs.append(observation)
            rews.append(reward)
            dones.append(done)
            
            if visual:
                viz_obs.append(env.get_ims())
                
    env.close()       
    obs = np.stack(obs)
    nobs = np.stack(nobs)
    rews = np.array(rews)
    dones = np.array(dones)
    acts = np.stack(acts)
    
    os.makedirs(name='saved_prior')
    np.save("saved_prior/random_policy.npy", arr={
        'observations': obs,
        'next_observations': nobs,
        'rewards': rews,
        'dones': dones,
        'actions': acts,
        'images': viz_obs if visual else None
    })
    
    
def save_expert(agent, env, num_episodes: int, visual: bool = False):
    stats = {'r': [], 'l': [], 't': []}
    successes = None
    obs, nobs, rews, acts, dones, viz_obs = [], [], [], [], [], []
    
    for _ in tqdm(range(num_episodes), desc="Running trained expert..."):
        (observation, info), done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            obs.append(observation)
            acts.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            nobs.append(observation)
            rews.append(reward)
            dones.append(done)
            
            if visual:
                viz_obs.append(env.get_ims())
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']
    env.close()       
    obs = np.stack(obs)
    nobs = np.stack(nobs)
    rews = np.array(rews)
    dones = np.array(dones)
    acts = np.stack(acts)
    
    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    os.makedirs(name='saved_expert')
    np.save("saved_expert/trained_expert.npy", arr={
        'observations': obs,
        'next_observations': nobs,
        'rewards': rews,
        'dones': dones,
        'actions': acts,
        'images': viz_obs if visual else None
    })

def evaluate(agent, env, num_episodes: int):
    stats = {'r': [], 'l': [], 't': []}
    successes = None
    
    for _ in tqdm(range(num_episodes), desc="Evaluating agent.."):
        (observation, info), done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']
    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats

if __name__ == "__main__":
    try:
        collect_expert()
    except KeyboardInterrupt:
        wandb.finish()
        exit()