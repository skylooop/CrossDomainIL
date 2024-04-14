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
import jax.numpy as jnp
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.geometry import pointcloud
import random
import optax
from ott.neural import models

import wandb

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')

import functools
from utils.const import SUPPORTED_ENVS
from utils.loading_data import prepare_buffers_for_il
from agents.notdual import JointAgent
from networks.common import LayerNormMLP

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo


@functools.partial(jax.jit, static_argnums=2)
def sinkhorn_loss(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: float = 0.001
) -> float:
    """Computes transport between (x, a) and (y, b) via Sinkhorn algorithm."""
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    sdiv = sinkhorn_divergence(
        pointcloud.PointCloud, x, y, epsilon=epsilon, a=a, b=b
    )
    return sdiv.divergence

@functools.partial(jax.jit, static_argnums=0)
def compute_reward_from_not(not_agent, expert_obs, expert_nobs, agent_observations, agent_next_observations):
    se = not_agent.encoders_state(expert_obs, method='encode_expert')
    se_next = not_agent.encoders_state(expert_nobs, method='encode_expert')
    expert_pairs = jnp.concatenate([se, se_next], axis=-1)
    
    sa = not_agent.encoders_state(agent_observations, method='encode_agent')
    sa_next = not_agent.encoders_state(agent_next_observations, method='encode_agent')
    agent_pairs = jnp.concatenate([sa, sa_next], axis=-1)
    
    #reward = potentials.distance(agent_pairs, expert_pairs)
    reward = jax.vmap(jax.vmap(not_agent.neural_dual_pairs.to_dual_potentials().distance, in_axes=(0, None)), in_axes=(None, 0))(agent_pairs, expert_pairs)
    return reward

@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="imitation")
def collect_expert(cfg: DictConfig) -> None:
    #################
    # python src/run_il.py hydra.job.chdir=True
    #################
    
    assert cfg.imitation_env.name in SUPPORTED_ENVS
    
    print(f"Collecting Expert data using {cfg.algo.name} on {cfg.imitation_env.name} env")
    print(OmegaConf.to_yaml(cfg))
    
    print(f"Saving expert weights into {os.getcwd()}")
    wandb.init(
        #mode="offline",
        project=cfg.logger.project,
        config=dict(cfg),
        group="expert_" + f"{cfg.imitation_env.name}_{cfg.algo.name}",
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
    
    source_expert_ds, source_random_ds, target_random_buffer = prepare_buffers_for_il(cfg=cfg, target_obs_space=eval_env.observation_space,
                                                                                          target_act_space=eval_env.action_space)
    
    env = TimeLimit(env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: (x + 1) % cfg.save_video_each_ep == 0)
    eval_env = RecordEpisodeStatistics(eval_env, deque_size=cfg.save_expert_episodes)
    
    agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
    
    hidden_dims = [32, 32, 16, 8, 2]
    encoder_expert = LayerNormMLP(hidden_dims=hidden_dims, activations=jax.nn.leaky_relu)
    encoder_agent = LayerNormMLP(hidden_dims=hidden_dims, activations=jax.nn.leaky_relu)

    neural_f = models.MLP(
        dim_hidden=[64, 64, 64, 64],
        is_potential=True,
    )
    neural_g = models.MLP(
        dim_hidden=[64, 64, 64, 64],
        is_potential=True,
    )
    
    optimizer_f = optax.adam(learning_rate=1e-4, b1=0.9, b2=0.99)
    optimizer_g = optimizer_f

    not_agent = JointAgent(
        encoder_agent, 
        encoder_expert, 
        agent_dim=eval_env.observation_space.shape[0],
        expert_dim=source_expert_ds.observations.shape[-1],
        embed_dim=hidden_dims[-1],
        neural_f=neural_f,
        neural_g=neural_g,
        optimizer_f=optimizer_f,
        optimizer_g=optimizer_g,
        expert_loss_coef=1.0)
    
    (observation, info), done = env.reset(seed=cfg.seed), False
    
    # PRETRAIN NOT
    pbar = tqdm(range(1500), leave=True) #1500
    for i in pbar:
        agent_data = target_random_buffer.sample(512)
        expert_data = source_expert_ds.sample(512)
        random_data = source_random_ds.sample(512)
        loss_elem, loss_pairs, w_dist_elem, w_dist_pairs = not_agent.optimize_not(agent_data, expert_data, random_data)

        if i % 100 == 0:
            se = not_agent.encoders_state(expert_data.observations, method='encode_expert')
            sa = not_agent.encoders_state(agent_data.observations, method='encode_agent')
            sink = sinkhorn_loss(sa, se)
            info = {"sink_dist": sink,
                          "w_dist_elem": w_dist_elem}
            pbar.set_postfix(info)
            wandb.log(info, step=i)
            #print(loss_elem, loss_pairs, w_dist_elem, w_dist_pairs, sink)
    
    pbar = tqdm(range(1, cfg.max_steps + 1), leave=True) #cfg.max_steps
    for i in pbar:
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

        target_random_buffer.insert(observation, action, reward, mask, float(done), next_observation)
        observation = next_observation
        
        if done:
            wandb.log({f"Training/rewards": info['episode']['r'],
                    f"Training/episode length": info['episode']['l']}, step=i)
            (observation, info), done = env.reset(), False
            
        if i >= cfg.algo.start_training:
            for _ in range(cfg.algo.updates_per_step):
                agent_data = target_random_buffer.sample(512)
                expert_data = source_expert_ds.sample(512)
                random_data = source_random_ds.sample(512)
                if i % 500 == 0:
                    loss_elem, loss_pairs, w_dist_elem, w_dist_pairs = not_agent.optimize_not(agent_data, expert_data, random_data)
                if i % 1000 == 0:
                    info = not_agent.optimize_encoders(agent_data, expert_data, random_data)
                if i % 2000 == 0:
                    pbar.set_postfix(info)
                    se = not_agent.encoders_state(expert_data.observations, method='encode_expert')
                    sa = not_agent.encoders_state(agent_data.observations, method='encode_agent')
                    sink = sinkhorn_loss(sa, se)
                    print({"sink_dist": sink.item(),
                            "w_dist_elem": w_dist_elem.item()})
                actor_update_info = agent.update(agent_data)
        ## Make rewarder
        agent_last_obs = target_random_buffer.observations[:512]
        agent_last_nobs = target_random_buffer.next_observations[:512]
        expert_data = source_expert_ds.sample(512)
        new_rewards = compute_reward_from_not(not_agent, expert_data.observations, expert_data.next_observations, agent_last_obs, agent_last_nobs)
        target_random_buffer.rewards[:512] = new_rewards.argmax(-1)
        
        
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