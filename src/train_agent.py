import os
from typing import Tuple
import warnings

import gymnasium
from matplotlib import axis


warnings.filterwarnings('ignore')

os.environ['MUJOCO_GL']='egl'
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf
from flax import linen as nn
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

from jaxrl_m.common import TrainState
from jaxrl_m.networks import RelativeRepresentation
from gc_datasets.replay_buffer import ReplayBuffer
from gail.embed import EmbedGAIL, EncodersPair
from gail.rewards_transform import NegativeShift, RewardsStandartisation
import functools
from utils.const import SUPPORTED_ENVS
from utils.loading_data import prepare_buffers_for_il
from agents.notdual import ENOTCustom, W2NeuralDualCustom
from icvf_utils.icvf_networks import JointNOTAgent
from gail.disc import Discriminator
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
    
    
    return Dataset(
        observations=expert_source['observations'],
        actions=expert_source['actions'],
        rewards=expert_source['rewards'],
        dones_float=expert_source['dones'],
        masks=1.0 - expert_source['dones'],
        next_observations=expert_source['next_observations'],
        size=expert_source['observations'].shape[0])
    

class DidaEncoders(EncodersPair):

    state: TrainState
    state_disc: Discriminator
    pair_disc: Discriminator
    weight: float

    @classmethod
    def create(cls, obs: jnp.ndarray, dims: Tuple[int], weight: float, num_train_iters: int, learning_rate: float):
        model = RelativeRepresentation(hidden_dims=dims, ensemble=False)
        rng = jax.random.PRNGKey(1)
        params = model.init(rng, obs)['params']

        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=num_train_iters, alpha=5e-2
        )
        
        net = TrainState.create(
            model_def=model,
            params=params,
            tx=optax.adamw(learning_rate=schedule),
            eps=1e-6
        )
 
        final_dim = dims[-1]
        state_disc = Discriminator.create(jnp.ones((final_dim,)), learning_rate=learning_rate, num_train_iters=num_train_iters)
        pair_disc = Discriminator.create(jnp.ones((final_dim * 2,)), learning_rate=learning_rate, num_train_iters=num_train_iters)

        return cls(state=net, weight=weight, state_disc=state_disc, pair_disc=pair_disc)

    def agent_embed(self, x): 
        return self.state(x, params=self.state.params)

    def expert_embed(self, x): 
        return self.state(x, params=self.state.params)
    
    @jax.jit
    def update(self, s_a, next_s_a, s_e, next_s_e):

        def loss_fn(params):
            embed = self.state(s_a, params=params)
            next_embed = self.state(next_s_a, params=params)
            pair = jnp.concatenate([embed, next_embed], -1)

            embed_e = self.state(s_e, params=params)
            next_embed_e = self.state(next_s_e, params=params)
            pair_e = jnp.concatenate([embed_e, next_embed_e], -1)

            state_loss = self.state_disc.generator_losses(embed).mean() + self.state_disc.real_generator_losses(embed_e).mean()
            pair_loss = self.pair_disc.disc_losses(pair).mean() + self.pair_disc.real_disc_losses(pair_e).mean()

            return pair_loss + state_loss * self.weight, (
                jax.lax.stop_gradient(embed), 
                jax.lax.stop_gradient(next_embed), 
                jax.lax.stop_gradient(embed_e), 
                jax.lax.stop_gradient(next_embed_e)
            )

        new_state, (embed, next_embed, embed_e, next_embed_e) = self.state.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        pair = jnp.concatenate([embed, next_embed], -1)
        pair_e = jnp.concatenate([embed_e, next_embed_e], -1)
  
        new_state_disc, _ = self.state_disc.update_step(embed_e, embed)
        new_pair_disc, _ = self.pair_disc.update_step(pair_e, pair)
        
        return self.replace(state=new_state, state_disc=new_state_disc, pair_disc=new_pair_disc), (
            embed, next_embed, embed_e, next_embed_e
        )
    
    @jax.jit
    def get_embeds(self, x, next_x, y, next_y):
        embed = self.agent_embed(x)
        next_embed = self.agent_embed(next_x)

        embed_y = self.expert_embed(y)
        next_embed_y = self.expert_embed(next_y)

        return embed, next_embed, embed_y, next_embed_y

    #     pair = jnp.concatenate([embed, next_embed], -1)
    #     pair_y = jnp.concatenate([embed_y, next_embed_y], -1)
  
    #     new_state_disc = self.state_disc.update_step(embed_y, embed)
    #     new_pair_disc = self.pair_disc.update_step(pair_y, pair)

    #     return self.replace(state_disc=new_state_disc, pair_disc=new_pair_disc)



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
                               action_space=eval_env.action_space, capacity=10000)
    
    # source_expert_ds = get_dataset(gym.make('hopper-expert-v2'), expert=True, num_episodes=10)
    
    env = TimeLimit(env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: (x + 1) % cfg.save_video_each_ep == 0)
   
    eval_env = RecordEpisodeStatistics(eval_env, deque_size=cfg.save_expert_episodes)
    
    agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
        

    from ott.neural.methods.expectile_neural_dual import MLP as ExpectileMLP, NegativeMLP
        
    
    latent_dim = 16

    neural_f = ExpectileMLP(dim_hidden=[128, 128, 128, latent_dim*2], act_fn=jax.nn.elu)
    neural_g = NegativeMLP(dim_hidden=[128, 128, 128, 1], act_fn=jax.nn.elu)
    schedule = optax.cosine_decay_schedule(
            init_value=3e-4, decay_steps=300_000, alpha=1e-2
    )
    optimizer_f = optax.adam(learning_rate=schedule, b1=0.9, b2=0.99)
    optimizer_g = optax.adam(learning_rate=schedule, b1=0.9, b2=0.99)


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

    gail = EmbedGAIL.create(Discriminator.create(jnp.ones((latent_dim * 2,)), 5e-5, 300_000), 
                            [RewardsStandartisation()], 
                            DidaEncoders.create(jnp.ones((11,)), latent_dim, weight=0.3, num_train_iters=300_000, learning_rate=1e-4), 
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
                target_data = buffer_disc.sample(2024)
                expert_data = source_expert_ds.sample(2024)

                embed_a, next_embed_a, embed_e, next_embed_e = gail.encoders.get_embeds(
                    target_data.observations, target_data.next_observations, 
                    expert_data.observations, expert_data.next_observations
                )
                gail = gail.update_ot(embed_e, next_embed_e, embed_a, next_embed_a)


        if i >= 3000 and i % 1 == 0:

            for _ in range(1):
                target_data = buffer_disc.sample(2024)
                expert_data = source_expert_ds.sample(2024)
                
                encoders, (embed_a, next_embed_a, embed_e, next_embed_e) = gail.encoders.update(
                    target_data.observations, target_data.next_observations, 
                    expert_data.observations, expert_data.next_observations
                )

                gail = gail.replace(encoders=encoders)

                gail = gail.update_ot(embed_e, next_embed_e, embed_a, next_embed_a)
                gail, info = gail.update(embed_e, next_embed_e, embed_a, next_embed_a)
                
                # encoded_source, encoded_target = gail.encoders.expert_embed(expert_data.observations), gail.encoders.agent_embed(target_data.observations)
                

        if i >= 10000:

            for _ in range(1):
                target_data = target_buffer_agent.sample(2012)
                actor_update_info = agent.update(target_data)


        if i % 5000 == 0 and  i > 0:
            fig, ax = plt.subplots()
            both = jnp.concatenate([embed_a, embed_a], axis=0)
            ax.scatter(both[:, 0], both[:, 1], c=['orange']*embed_a.shape[0] + ['blue']*embed_e.shape[0])
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