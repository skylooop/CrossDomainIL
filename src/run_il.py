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
import jax
import random
import optax
from ott.neural import models

import wandb

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')

from utils.const import SUPPORTED_ENVS
from utils.loading_data import prepare_buffers_for_il
from neuralot.neuraldual import W2NeuralDualCustom, NotNetwork, JointAgent
from networks.common import LayerNormMLP, TrainState
from datasets.replay_buffer import ReplayBuffer, Dataset

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo


@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="imitation")
def collect_expert(cfg: DictConfig) -> None:
    #################
    # python src/run_il.py hydra.job.chdir=True
    #################
    
    assert cfg.env.name in SUPPORTED_ENVS
    
    print(f"Collecting Expert data using {cfg.algo.name} on {cfg.env.name} env")
    print(OmegaConf.to_yaml(cfg))
    
    print(f"Saving expert weights into {os.getcwd()}")
    wandb.init(
        project=cfg.logger.project,
        config=dict(cfg),
        group="expert_" + f"{cfg.env.name}_{cfg.algo.name}",
    )
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    if cfg.imitation_env.name == "PointUMaze":
        from envs.maze_envs import CustomPointUMazeSize3Env
        env = CustomPointUMazeSize3Env()
        eval_env = CustomPointUMazeSize3Env()
        episode_limit = 1000
        
    elif cfg.imitation_env.name == "PointAntUMaze":
        from envs.maze_envs import CustomAntUMazeSize3Env
        episode_limit = 1000
        env = CustomAntUMazeSize3Env()
        eval_env = CustomAntUMazeSize3Env()
        
    elif cfg.imitation_env.name == 'InvertedPendulum-v2':
        from envs.envs import ExpertInvertedPendulumEnv
        env = ExpertInvertedPendulumEnv()
        eval_env = ExpertInvertedPendulumEnv()
        episode_limit = 1000
        
    elif cfg.imitation_env.name == 'InvertedDoublePendulum-v2':
        from envs.envs import ExpertInvertedDoublePendulumEnv
        env = ExpertInvertedDoublePendulumEnv()
        eval_env = ExpertInvertedDoublePendulumEnv()
        episode_limit = 1000
        
    elif cfg.imitation_env.name == 'Reacher2-v2':
        from envs.more_envs import CustomReacher2Env
        env = CustomReacher2Env(l2_penalty=True)
        eval_env = CustomReacher2Env(l2_penalty=True)
        episode_limit = 50
        
    elif cfg.imitation_env.name == 'Reacher3-v2':
        from envs.more_envs import CustomReacher3Env
        env = CustomReacher3Env(l2_penalty=True)
        eval_env = CustomReacher3Env(l2_penalty=True)
        episode_limit = 50
        
    elif cfg.imitation_env.name == 'HalfCheetah-v2':
        from envs.envs import ExpertHalfCheetahNCEnv
        env = ExpertHalfCheetahNCEnv()
        eval_env = ExpertHalfCheetahNCEnv()
        episode_limit = 1000
    
    source_expert_buffer, source_random_buffer, target_random_buffer = prepare_buffers_for_il(cfg=cfg, observation_space=env.observation_space, action_space=env.action_space)
    
    env = TimeLimit(env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: (x + 1) % cfg.save_video_each_ep == 0)
    eval_env = RecordEpisodeStatistics(eval_env, deque_size=cfg.save_expert_episodes)
    
    agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
    
    rng = jax.random.PRNGKey(cfg.seed)
    hidden_dims = [32, 16, 8, 2]
    neural_f = models.MLP(dim_hidden=[32, 32, 32, 32],is_potential=True)
    neural_g = models.MLP(dim_hidden=[32, 32, 32, 32],is_potential=True)
    optimizer_f = optax.adam(learning_rate=1e-4, b1=0.9, b2=0.99)
    optimizer_g = optimizer_f

    neural_dual = W2NeuralDualCustom(
        dim_data=hidden_dims[-1] * 2, 
        neural_f=neural_f,
        optimizer_f=optimizer_f,
        optimizer_g=optimizer_g,
        neural_g=neural_g,
        num_train_iters=2_000
    )
    encoder_expert = LayerNormMLP(hidden_dims=hidden_dims)
    encoder_agent = LayerNormMLP(hidden_dims=hidden_dims)
    net_def = NotNetwork(encoders={
        'expert_encoder': encoder_expert,
        'agent_encoder': encoder_agent}, expert_loss_coef=1.0)

    network = TrainState.create(
        model_def=net_def,
        params=net_def.init(rng, source_expert_buffer.observations[0], target_random_buffer.observations[0])['params'],
        tx=optax.adam(learning_rate=1e-4)
    )
    neural_ot_agent = JointAgent(network=network)
    
    (observation, info), done = env.reset(seed=cfg.seed), False
    
    for i in tqdm(range(1, cfg.max_steps + 1)):
        if i < cfg.algo.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if not done:
            mask = 1.0
        else:
            mask = 0.0
        # add reward from not after some training?
        target_random_buffer.insert(observation, action, reward, mask, float(done), next_observation)
        observation = next_observation
        
        if done:
            wandb.log({f"Training/rewards": info['episode']['r'],
                    f"Training/episode length": info['episode']['l']}, step=i)
            (observation, info), done = env.reset(), False
            
        if i >= cfg.algo.start_training:
            for _ in range(cfg.algo.updates_per_step):
                expert_data = source_expert_buffer.sample(cfg.algo.batch_size)
                agent_data = target_random_buffer.sample(cfg.algo.batch_size)
                if i % 10 == 0:
                    neural_ot_agent, update_info_encoders = neural_ot_agent.optimize_encoders(neural_dual, agent_data, expert_data)
                neural_dual, update_info_not = neural_ot_agent.optimize_not(neural_dual, agent_data, expert_data)
                actor_update_info = agent.update(agent_data)

            if i % cfg.log_interval == 0:
                wandb.log({f"Training/Critic loss": actor_update_info['critic_loss'],
                        f"Training/Actor loss": actor_update_info['actor_loss'],
                        f"Training/Entropy": actor_update_info['entropy'],
                        f"Training/Temperature": actor_update_info['temperature'],
                        f"Training/Temp loss": actor_update_info['temp_loss']}, step=i)
                
        if i % cfg.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, cfg.eval_episodes)
            wandb.log({"Evaluation/rewards": eval_stats['r'],
                    "Evaluation/length": eval_stats['l']})
    
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