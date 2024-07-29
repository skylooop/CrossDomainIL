# SYSTEM
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['MUJOCO_GL']='egl'
os.environ['HYDRA_FULL_ERROR'] = '1'

# PATHS & CONFIGS
import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

from tqdm.auto import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer
import flax.linen as nn
import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from typing import *

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')
import gymnasium
from gc_datasets.replay_buffer import ReplayBuffer
from utils.loading_data import prepare_buffers_for_il
from main_agents.disc import Discriminator
from main_agents.dida_agent import DIDA
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, RecordVideo
from gc_datasets.dataset import Batch
from jaxrl_m.networks import RelativeRepresentation

class Identity(nn.Module):

    @nn.compact
    def __call__(self, x):
        return x

@jax.jit
def compute_reward_from_disc(agent, obs, next_obs):
    encoded_obs = agent.encoder(obs, method='encode_source')
    encoded_nobs = agent.encoder(next_obs, method='encode_source')
    reward = agent.policy_disc.state(
        jnp.concatenate([encoded_obs, encoded_nobs], axis=-1)
    ) 

    return reward.reshape(-1)

class NegativeMLP(nn.Module):
  """A simple MLP model of a potential used in default initialization."""

  dim_hidden: Sequence[int]
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply MLP transform."""
    for feat in self.dim_hidden[:-1]:
      x = self.act_fn(nn.Dense(feat)(x))
    return -jnp.abs(nn.Dense(self.dim_hidden[-1])(x))
  
def get_env(env_name: str, num_last_eps_info: int, eval_video_interval: int):
    if env_name == "PointUMaze":
        from envs.maze_envs import CustomPointUMazeSize3Env
        env = CustomPointUMazeSize3Env()
        eval_env = CustomPointUMazeSize3Env()
        episode_limit = 1000
        
    elif env_name == "PointAntUMaze":
        from envs.maze_envs import CustomAntUMazeSize3Env
        episode_limit = 1000
        env = CustomAntUMazeSize3Env()
        eval_env = CustomAntUMazeSize3Env()
        
    elif env_name == 'InvertedPendulum-v2':
        from envs.envs import ExpertInvertedPendulumEnv
        env = ExpertInvertedPendulumEnv()
        eval_env = ExpertInvertedPendulumEnv()
        episode_limit = 1000
        
    elif env_name == 'InvertedDoublePendulum-v2':
        from envs.envs import ExpertInvertedDoublePendulumEnv
        env = ExpertInvertedDoublePendulumEnv()
        eval_env = ExpertInvertedDoublePendulumEnv()
        episode_limit = 1000
        
    elif env_name == 'Reacher2-v2':
        from envs.more_envs import CustomReacher2Env
        env = CustomReacher2Env(l2_penalty=True)
        eval_env = CustomReacher2Env(l2_penalty=True)
        episode_limit = 50
        
    elif env_name == 'Reacher3-v2':
        from envs.more_envs import CustomReacher3Env
        env = CustomReacher3Env(l2_penalty=True)
        eval_env = CustomReacher3Env(l2_penalty=True)
        episode_limit = 50
        
    elif env_name == 'HalfCheetah-v2':
        from envs.envs import ExpertHalfCheetahNCEnv
        env = ExpertHalfCheetahNCEnv()
        eval_env = ExpertHalfCheetahNCEnv()
        episode_limit = 1000
    
    elif env_name == 'Hopper':
        env = gymnasium.make("Hopper-v4", render_mode='rgb_array')
        eval_env = gymnasium.make("Hopper-v4", render_mode='rgb_array')
        episode_limit = 1000

    env = TimeLimit(env, max_episode_steps=episode_limit)
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: x % eval_video_interval == 0)
    eval_env = RecordEpisodeStatistics(eval_env, buffer_length=num_last_eps_info)
    return env, eval_env

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def compute_sar(dida_agent, imitator_observations, noisy_expert_obs, anchor_obs, threshold_p: float = 2/3):
    observations = jnp.concatenate([imitator_observations, noisy_expert_obs, anchor_obs])
    bools = jnp.concatenate([
        jnp.zeros(imitator_observations.shape[0], dtype=jnp.int32), 
        jnp.ones(noisy_expert_obs.shape[0], dtype=jnp.int32), 
        jnp.ones(anchor_obs.shape[0], dtype=jnp.int32)
    ])
    p_acc_array = jnp.astype((dida_agent.noisy_disc.state(dida_agent.encoder(observations, method='encode_source')) > 0), jnp.int32)
    p_acc = (p_acc_array == bools).mean()
    
    alpha_1 = p_acc / threshold_p
    alpha_2 = (1 - p_acc) / (1 - threshold_p)
    mask = jnp.astype(threshold_p >= p_acc, jnp.int32)
    alpha = mask * alpha_1 + (1-mask) * alpha_2

    return alpha, p_acc

def mix_anchor_imitator(imitator_batch, anchor_batch):
    mix_batch = Batch(
        observations=jnp.concatenate([anchor_batch.observations, imitator_batch.observations.squeeze()], axis=0),
        next_observations=jnp.concatenate([anchor_batch.next_observations, imitator_batch.next_observations.squeeze()], axis=0),
        rewards=jnp.concatenate([anchor_batch.rewards, imitator_batch.rewards.squeeze()], axis=0),
        masks=jnp.concatenate([anchor_batch.masks, imitator_batch.masks.squeeze()], axis=0),
        actions=jnp.concatenate([anchor_batch.actions, imitator_batch.actions.squeeze()], axis=0)
    )
    return mix_batch

@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="imitation")
def dida_imitation(cfg: DictConfig) -> None:
    #################
    # python src/run_dida.py hydra.job.chdir=True
    #################
    print("CFG: \n")
    print(OmegaConf.to_yaml(cfg))
    print("\n")

    wandb.init(
        mode="offline",
        config=dict(cfg),
        group="DIDA",
    )
    env, eval_env = get_env(cfg.imitation_env.name, eval_video_interval=cfg.save_video_each_ep, num_last_eps_info=cfg.save_expert_episodes)
    if cfg.imitation_env.name != "Hopper":
        source_expert_ds, source_random_ds, combined_source_ds = prepare_buffers_for_il(cfg=cfg)
    else:
        source_expert_ds = prepare_buffers_for_il(cfg=cfg)

    imitator_buffer = ReplayBuffer(observation_space=eval_env.observation_space,
                                       action_space=eval_env.action_space, capacity=cfg.algo.buffer_size)
    noisy_expert_buffer = ReplayBuffer(observation_space=eval_env.observation_space,
                                       action_space=eval_env.action_space, capacity=cfg.algo.buffer_size)
    noisy_expert_buffer.initialize_with_dataset(source_expert_ds)
    noisy_expert_buffer.apply_noise()
    
    anchor_buffer = ReplayBuffer(observation_space=eval_env.observation_space,
                                       action_space=eval_env.action_space, capacity=cfg.algo.buffer_size)
    anchor_buffer.initialize_with_dataset(noisy_expert_buffer)
    anchor_buffer.random_shuffle()
    
    # agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
    #                                                  actions=env.action_space.sample()[None])
    
    rep_dim = 10
    D_noisy = Discriminator.create(jnp.ones((rep_dim, )), 2e-5, 5, 10000, 1e-5, hidden_dims=[128, 128, 1])
    D_policy = Discriminator.create(jnp.ones((2 * rep_dim, )), 2e-5, 1, 10000, 1e-5, hidden_dims=[128, 128, 1])

    dida_agent = DIDA.create(noisy_discr=D_noisy, policy_discr=D_policy, encoders={'source': RelativeRepresentation(hidden_dims=(128, 128, rep_dim), ensemble=False),
                                                                    'target': Identity()}, observation_dim=env.observation_space.sample().shape[0])
    agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
    (observation, info), done = env.reset(seed=cfg.seed), False
    os.makedirs("viz_plots", exist_ok=True)
    
    domain_weight = 0.2
    batch_size = 2048
    pbar = tqdm(range(1, cfg.max_steps + 1), leave=True)
    key = jax.random.PRNGKey(42)

    for i in pbar:
        if i < 5_000:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
            
        next_observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        mask = 1.0 if not done else 0.0

        ri = 0.0 if i < 1_000 else compute_reward_from_disc(dida_agent, observation[None,], next_observation[None,])[0] 
        imitator_buffer.insert(observation, action, float(ri), mask, float(done), next_observation)

        if done:
            wandb.log({f"Training/rewards": info['episode']['r'],
                    f"Training/episode length": info['episode']['l']})
            (observation, info), done = env.reset(), False

        # UPDATE DISCRIMINATOR
        if i >= 2050:
            key, rnd_key = jax.random.split(key, 2)
            noisy_expert_batch, _ = noisy_expert_buffer.sample(batch_size)
            anchor_batch, _ = anchor_buffer.sample(batch_size)
            imitator_batch, imitator_indx = imitator_buffer.sample(batch_size)
            
            if i % 5 == 0:
                dida_agent, grads_encoder_n = dida_agent.update_noise_discr(imitator_batch, noisy_expert_batch, anchor_batch, domain_weight)
            alpha, p_acc = compute_sar(dida_agent, imitator_batch.observations, noisy_expert_batch.observations, anchor_batch.observations)
            domain_weight = domain_weight * (2 / (1 + np.exp(-10 * (i - 1000) / cfg.max_steps)) - 1)
            das_probs = jax.nn.sigmoid(dida_agent.noisy_disc.state(dida_agent.encoder(imitator_batch.observations, method='encode_source'))).squeeze()
            p_das = jnp.clip(das_probs / (das_probs.sum(0)), min=0.1, max=0.9)
            # print(p_das.shape)
            # print(das_probs.shape)
            # print(imitator_batch.observations.shape[0])
            p_das_indxs = jax.random.choice(key=rnd_key, a=jnp.arange(imitator_batch.observations.shape[0]), replace=False, p=p_das, shape=(int(alpha * batch_size), ))
            das_chosen_imitator, _ = imitator_buffer.sample(indx=p_das_indxs)
            mix_data_batch = mix_anchor_imitator(das_chosen_imitator, anchor_batch)
            dida_agent, grads_encoder_p = dida_agent.update_policy_discr(noisy_expert_batch, mix_data_batch)
            dida_agent = dida_agent.update_encoder(grads_encoder_p, grads_encoder_n, domain_weight)

        if i >= 2050:
            actor_update_info = agent.update(imitator_batch)

        if i % cfg.log_interval == 0 and i >= 2050:
            tsne = TSNE()
            scaler = MinMaxScaler()
            noisy_expert_batch, _ = noisy_expert_buffer.sample(500)
            imitator_batch, _ = imitator_buffer.sample(500)
            viz_batch = jnp.concatenate([noisy_expert_batch.observations, imitator_batch.observations])
            proj_tsne = scaler.fit_transform(tsne.fit_transform(viz_batch))

            fig, ax = plt.subplots()
            ax.scatter(proj_tsne[:, 0], proj_tsne[:, 1], label="tsne", c=['purple'] * 500 + ['red'] * 500)
            fig.savefig(f"viz_plots/dida_{i}_tsne.png")
            # wandb.log({f"Training/Critic loss": actor_update_info['critic_loss'],
            #         f"Training/Actor loss": actor_update_info['actor_loss'],
            #         f"Training/Entropy": actor_update_info['entropy'],
            #         f"Training/Temperature": actor_update_info['temperature'],
            #         f"Training/Temp loss": actor_update_info['temp_loss']})
            
                
        if i % cfg.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, cfg.eval_episodes)
            print(eval_stats)
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
        dida_imitation()
    except KeyboardInterrupt:
        wandb.finish()
        exit()