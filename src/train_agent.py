import os
import warnings

import gym
import gymnasium
from matplotlib import axis


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
from orbax.checkpoint import PyTreeCheckpointer
from flax.training import orbax_utils
import optax

import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')

from gc_datasets.replay_buffer import ReplayBuffer
from gail.embed import EmbedGAIL, EncodersPair
from gail.rewards_transform import NegativeShift, RewardsStandartisation
import functools
from utils.const import SUPPORTED_ENVS
from utils.loading_data import prepare_buffers_for_il
from agents.notdual import ENOTCustom, W2NeuralDualCustom
from icvf_utils.icvf_networks import JointNOTAgent
from agents.disc import Discriminator
from run_il import get_dataset
from src.gail.base import GAIL
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, RecordVideo
from gc_datasets.dataset import Batch, Dataset
from ott.geometry import costs

def load_expert(path_to_expert):

    expert_source = np.load(path_to_expert, allow_pickle=True).item()


    expert_source['dones'][-1] = 1

    lim = 1 - 1e-5
    expert_source['actions'] = np.clip(expert_source['actions'], -lim, lim)
        
    
    expert_source['observations'] = expert_source['observations'].astype(np.float32)
    expert_source['next_observations'] = expert_source['next_observations'].astype(np.float32)
    
    
    return Dataset(observations=expert_source['observations'],
                        actions=expert_source['actions'],
                        rewards=expert_source['rewards'],
                        dones_float=expert_source['dones'],
                        masks=1.0 - expert_source['dones'],
                        next_observations=expert_source['next_observations'],
                        size=expert_source['observations'].shape[0])
    

class CoordEncoders(EncodersPair):

    def agent_embed(self, x): 
        return x

    def expert_embed(self, x): 
        return x


@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="imitation")
def collect_expert(cfg: DictConfig) -> None:
    #################
    # python src/train_agent.py hydra.job.chdir=True
    #################
    
    assert cfg.imitation_env.name in SUPPORTED_ENVS
    
    print(f"Collecting Expert data using {cfg.algo.name} on {cfg.imitation_env.name} env")
    print(OmegaConf.to_yaml(cfg))
    
    print(f"Saving expert weights into {os.getcwd()}")
    wandb.init(
        mode="online",
        config=dict(cfg),
        group="expert_" + f"{cfg.imitation_env.name}_{cfg.algo.name}",
    )
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    
    # from envs.maze_envs import CustomPointUMazeSize3Env
    # env = CustomPointUMazeSize3Env()
    # eval_env = CustomPointUMazeSize3Env()
    episode_limit = 1000
        
    # from envs.maze_envs import CustomAntUMazeSize3Env
    # episode_limit = 1000
    # env = CustomAntUMazeSize3Env()
    # eval_env = CustomAntUMazeSize3Env(),

    env = gymnasium.make("Hopper-v4", max_episode_steps=1000)
    eval_env = gymnasium.make("Hopper-v4", render_mode="rgb_array", max_episode_steps=1000)
    
    source_expert_ds = load_expert("saved_expert/trained_expert.npy")
    
    target_buffer_agent = ReplayBuffer(observation_space=eval_env.observation_space,
                                       action_space=eval_env.action_space, capacity=cfg.algo.buffer_size)
    
    buffer_disc = ReplayBuffer(observation_space=eval_env.observation_space,
                               action_space=eval_env.action_space, capacity=5000)
    
    # source_expert_ds = get_dataset(gym.make('hopper-expert-v2'), expert=True, num_episodes=10)
    
    env = TimeLimit(env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: (x + 1) % cfg.save_video_each_ep == 0)
   
    eval_env = RecordEpisodeStatistics(eval_env, deque_size=cfg.save_expert_episodes)
    
    agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
        

    from ott.neural.methods.expectile_neural_dual import MLP as ExpectileMLP, NegativeMLP
        
    
    latent_dim = 11

    neural_f = ExpectileMLP(dim_hidden=[128, 128, 128, latent_dim*2], act_fn=jax.nn.elu)
    neural_g = NegativeMLP(dim_hidden=[128, 128, 128, 1], act_fn=jax.nn.elu)
    optimizer_f = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)
    optimizer_g = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)


    not_proj = ENOTCustom(
            dim_data=latent_dim * 2, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.SqEuclidean(),
            expectile = 0.98,
            expectile_loss_coef = 1.0, # 0.4
            use_dot_product=False,
            is_bidirectional=False,
            target_weight=5.0
    )
    
    gail = EmbedGAIL.create(Discriminator.create(jnp.ones((latent_dim * 2,)), 5e-5, 10, 10000, 1e-5), 
                            [RewardsStandartisation()], 
                            CoordEncoders(), 
                            not_proj)
     
    
    (observation, info), done = env.reset(seed=cfg.seed), False
    

    os.makedirs("viz_plots", exist_ok=True)
    
    pbar = tqdm(range(1, cfg.max_steps + 1), leave=True)
    
    time_step = 0

    for i in pbar:
        if i < 10000:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
            
        next_observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        time_step += 1
        
        if not done:
            mask = 1.0
        else:
            mask = 0.
            time_step = 0

        if i < 3000:
            ri = 0.0
        else:
            ri = gail.predict_reward(observation[np.newaxis,], next_observation[np.newaxis,])[0] 
        
        target_buffer_agent.insert(observation, action, float(ri), mask, float(done), next_observation)
        buffer_disc.insert(observation, action, float(ri), mask, float(done), next_observation)
        
        observation = next_observation

        # if (observation[0] < 2 and observation[1] > 4) or terminated:
        #     print(f"YES: {terminated}")
            
        if done:
            wandb.log({f"Training/rewards": info['episode']['r'],
                    f"Training/episode length": info['episode']['l']})
            (observation, info), done = env.reset(), False
            
    
        if i == 3000:

            for _ in range(10000):
                target_data = target_buffer_agent.sample(1024)
                expert_data = source_expert_ds.sample(1024)
                gail = gail.update_ot(expert_data.observations, expert_data.next_observations, 
                                        target_data.observations, target_data.next_observations)


        if i >= 3000 and i % 1 == 0:

            for _ in range(1):
                target_data = buffer_disc.sample(1024)
                expert_data = source_expert_ds.sample(1024)
            
                gail = gail.update_ot(expert_data.observations, expert_data.next_observations, 
                                        target_data.observations, target_data.next_observations)
                gail, info = gail.update(expert_data.observations, expert_data.next_observations, 
                                        target_data.observations, target_data.next_observations)
                encoded_source, encoded_target = gail.encoders.expert_embed(expert_data.observations), gail.encoders.agent_embed(target_data.observations)
                

        if i >= 10000:

            for _ in range(1):
                target_data = target_buffer_agent.sample(512)
                actor_update_info = agent.update(target_data)


        if i % 5000 == 0 and  i > 0:
            fig, ax = plt.subplots()
            both = jnp.concatenate([encoded_target, encoded_source], axis=0)
            ax.scatter(both[:, 0], both[:, 1], c=['orange']*encoded_target.shape[0] + ['blue']*encoded_source.shape[0])
            fig.savefig(f"viz_plots/both_{i}.png")
            

        if i % cfg.log_interval == 0:
            print("rewards", target_data.rewards[:10])
            wandb.log({f"Training/Critic loss": actor_update_info['critic_loss'],
                    f"Training/Actor loss": actor_update_info['actor_loss'],
                    f"Training/Entropy": actor_update_info['entropy'],
                    f"Training/Temperature": actor_update_info['temperature'],
                    f"Training/Temp loss": actor_update_info['temp_loss']})
            
                
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
        collect_expert()
    except KeyboardInterrupt:
        wandb.finish()
        exit()