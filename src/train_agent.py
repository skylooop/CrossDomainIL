import os
import warnings

import gym
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
import functools
from utils.const import SUPPORTED_ENVS
from utils.loading_data import prepare_buffers_for_il
from agents.notdual import ENOTCustom, W2NeuralDualCustom
from icvf_utils.icvf_networks import JointNOTAgent
from agents.disc import Discriminator
from run_il import get_dataset

from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, RecordVideo

from gc_datasets.dataset import Batch

from ott.geometry import costs

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

def update_not(joint_ot_agent, not_agent_pairs, potential_proj, batch_source, batch_target):
    encoded_source = potential_proj.transport(
        joint_ot_agent.ema_get_phi_source(batch_source.observations),
        forward=True
    )
    encoded_target = joint_ot_agent.ema_get_phi_target(batch_target.observations)
    encoded_source_next = potential_proj.transport(
        joint_ot_agent.ema_get_phi_source(batch_source.next_observations),
         forward=True
    )
    encoded_target_next = joint_ot_agent.ema_get_phi_target(batch_target.next_observations)

    new_not_agent_pairs, loss_pairs, w_dist_pairs = not_agent_pairs.update(batch_source=jnp.concatenate((encoded_source, encoded_source_next), axis=-1),
                                                            batch_target=jnp.concatenate((encoded_target, encoded_target_next), axis=-1))

    # potentials_pairs = not_agent_pairs.to_dual_potentials(finetune_g=True)
    return new_not_agent_pairs, encoded_source, encoded_target


def update_disc(joint_ot_agent, D, potential_proj, batch_source, batch_target, d_key):
    encoded_source = potential_proj.transport(
        joint_ot_agent.ema_get_phi_source(batch_source.observations),
        forward=True
    )
    encoded_target = joint_ot_agent.ema_get_phi_target(batch_target.observations)
    encoded_source_next = potential_proj.transport(
        joint_ot_agent.ema_get_phi_source(batch_source.next_observations),
         forward=True
    )
    encoded_target_next = joint_ot_agent.ema_get_phi_target(batch_target.next_observations)

    new_D, d_key, _ = D.update_step(jnp.concatenate((encoded_source, encoded_source_next), axis=-1),
                                    jnp.concatenate((encoded_target, encoded_target_next), axis=-1), 1, 1, d_key)

    return new_D, d_key, encoded_source, encoded_target



def update_not_elem(joint_ot_agent, not_proj, expert_data, target_data):
    encoded_source = joint_ot_agent.ema_get_phi_source(expert_data.observations)
    encoded_target = joint_ot_agent.ema_get_phi_target(target_data.observations)
    # encoded_source_next = joint_ot_agent.get_phi_source(batch_source.next_observations)
    # encoded_target_next = joint_ot_agent.get_phi_target(batch_target.next_observations)
    
    new_not_agent_elems, loss_elems, w_dist_elems = not_proj.update(encoded_source, encoded_target)
    potentials = new_not_agent_elems.to_dual_potentials(finetune_g=True)
    
    return not_proj, potentials


def plot_embeddings(sn, se, tn) -> None:
    fig, ax = plt.subplots()
    
    ax.scatter(sn[:, 0], sn[:, 1], c='red', label='sn')
    ax.scatter(tn[:, 0], tn[:, 1], c='green', label='tn')
    ax.scatter(se[:, 0], se[:, 1], c='orange', label='se')
    plt.legend()
    plt.savefig('plots/embedding.png')
    wandb.log({'Embeddings': wandb.Image(fig)})
    plt.close()


@jax.jit
def compute_reward_from_not(joint_ot_agent, potential_pairs, obs, next_obs):
   
    # encoded_source = joint_ot_agent.ema_get_phi_source(expert_obs)
    encoded_target = joint_ot_agent.ema_get_phi_target(obs)
    # encoded_source_next = joint_ot_agent.ema_get_phi_source(expert_next_obs)
    encoded_target_next = joint_ot_agent.ema_get_phi_target(next_obs)

    f, g = potential_pairs.get_fg()
    f, g = jax.vmap(f), jax.vmap(g)
    reward = -g(
        jnp.concatenate([encoded_target, encoded_target_next], axis=-1)
    ) 
    # - f(
    #     jnp.concatenate([encoded_source, encoded_source_next], axis=-1)
    # ).mean()

    # tgt = jnp.concatenate([encoded_target, encoded_target_next], axis=-1)
    # T_tgt = potential_pairs.transport(tgt, forward=False)
    # reward = - ((T_tgt - tgt) ** 2).sum(axis=-1)

    return reward.reshape(-1)


@jax.jit
def compute_reward_from_disc(joint_ot_agent, D, obs, next_obs):
   
    # encoded_source = joint_ot_agent.ema_get_phi_source(expert_obs)
    encoded_target = joint_ot_agent.ema_get_phi_target(obs)
    # encoded_source_next = joint_ot_agent.ema_get_phi_source(expert_next_obs)
    encoded_target_next = joint_ot_agent.ema_get_phi_target(next_obs)

    reward = D.predict_reward(
        jnp.concatenate([encoded_target, encoded_target_next], axis=-1)
    ) 

    return reward.reshape(-1)


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
    
    # source_expert_ds, source_random_ds, combined_source_ds, target_ds = prepare_buffers_for_il(cfg=cfg)

    source_expert_ds, source_random_ds, combined_source_ds = prepare_buffers_for_il(cfg=cfg,
                                                                                                          target_obs_space=eval_env.observation_space,
                                                                                                          target_act_space=eval_env.action_space)
    
    target_buffer_agent = ReplayBuffer(observation_space=eval_env.observation_space,
                                       action_space=eval_env.action_space, capacity=cfg.algo.buffer_size)
    
    buffer_disc = ReplayBuffer(observation_space=eval_env.observation_space,
                               action_space=eval_env.action_space, capacity=5000)
    # source_expert_ds = get_dataset(gym.make("antmaze-umaze-v2"), expert=True, num_episodes=10)
    
    env = TimeLimit(env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: (x + 1) % cfg.save_video_each_ep == 0)
   
    eval_env = RecordEpisodeStatistics(eval_env, deque_size=cfg.save_expert_episodes)
    
    agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
        

    from ott.neural.methods.expectile_neural_dual import MLP as ExpectileMLP, NegativeMLP
        
    
    neural_f = ExpectileMLP(dim_hidden=[128, 128, 128, 2], act_fn=jax.nn.elu)
    neural_f_2 = ExpectileMLP(dim_hidden=[128, 128, 128, 1], act_fn=jax.nn.elu)
    neural_g = NegativeMLP(dim_hidden=[128, 128, 128, 1], act_fn=jax.nn.elu)
    neural_g_2 = ExpectileMLP(dim_hidden=[128, 128, 128, 1], act_fn=jax.nn.elu)
    optimizer_f = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)
    optimizer_g = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)
    latent_dim = 2

    not_pairs = ENOTCustom(
            dim_data=latent_dim * 2, 
            neural_f=neural_f_2,
            neural_g=neural_g_2,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.SqEuclidean(),
            expectile = 0.99,
            expectile_loss_coef = 0.5, # 0.4
            use_dot_product=False,
            is_bidirectional=True,
            target_weight=1
    )

    not_proj = ENOTCustom(
            dim_data=latent_dim, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.SqEuclidean(),
            expectile = 0.98,
            expectile_loss_coef = 1.0, # 0.4
            use_dot_product=False,
            is_bidirectional=False,
            target_weight=10.0
    )
    
    D, d_key = Discriminator.create(jnp.ones((4,)), 2e-5, 10, 10000, 1e-5)
    
    joint_ot_agent = JointNOTAgent.create(
        cfg.seed,
        latent_dim=latent_dim,
        target_obs=target_buffer_agent.observations[0],
        source_obs=source_expert_ds.observations[0],
    )
    
    checkpointer_net = PyTreeCheckpointer()

    restored_ckpt_target = checkpointer_net.restore("/home/nazar/projects/CDIL/saved_encoding_agent")
    net = joint_ot_agent.net.replace(params=restored_ckpt_target['net']['params'])
    joint_ot_agent = joint_ot_agent.replace(net=net)

    # potential_pairs_state_f_ckpt = checkpointer_potentials_pairs_state_f.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_pairs/state_f")
    # potential_pairs_state_g_ckpt = checkpointer_potentials_pairs_state_g.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_pairs/state_g")

    # potential_elems_state_f_ckpt = checkpointer_potentials_elems_state_f.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_elems/state_f")
    # potential_elems_state_g_ckpt = checkpointer_potentials_elems_state_g.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_elems/state_g")

    # state_f_elems = not_agent_elems.state_f.replace(params=potential_elems_state_f_ckpt['params'])
    # state_g_elems = not_agent_elems.state_g.replace(params=potential_elems_state_g_ckpt['params'])

    # state_f_pairs = not_agent_pairs.state_f.replace(params=potential_pairs_state_f_ckpt['params'])
    # state_g_pairs = not_agent_pairs.state_g.replace(params=potential_pairs_state_g_ckpt['params'])

    # not_agent_pairs.state_f = state_f_pairs
    # not_agent_pairs.state_g = state_g_pairs

    # not_agent_elems.state_f = state_f_elems
    # not_agent_elems.state_g = state_g_elems
    
    (observation, info), done = env.reset(seed=cfg.seed), False
    
    # potential_pairs = not_agent_pairs.to_dual_potentials()
    os.makedirs("viz_plots", exist_ok=True)
    
    pbar = tqdm(range(1, cfg.max_steps + 1), leave=True)
    for i in pbar:
        if i < 1000:
            action = env.action_space.sample()
        else:
            # indx = np.random.randint(target_buffer_agent.size - 1, size=256)
            # target_buffer_agent.update_not_rewards(joint_ot_agent, potential_pairs, indx=indx)
            action = agent.sample_actions(observation)
            
        next_observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if not done:
            mask = 1.0
        else:
            mask = 0.

        ri = compute_reward_from_disc(joint_ot_agent, D, observation[np.newaxis,], next_observation[np.newaxis,])[0] 
        
        target_buffer_agent.insert(observation, action, float(ri), mask, float(done), next_observation)
        buffer_disc.insert(observation, action, float(ri), mask, float(done), next_observation)
        
        observation = next_observation

        if (observation[0] < 2 and observation[1] > 4) or terminated:
            print(f"YES: {terminated}")
            
        if done:
            wandb.log({f"Training/rewards": info['episode']['r'],
                    f"Training/episode length": info['episode']['l']})
            (observation, info), done = env.reset(), False
            
        # if i % 1_000 == 0 or i == 0:
        #     os.makedirs(name='rewards_potential', exist_ok=True)
        #     np.save("rewards_potential/rewards.npy", arr={
        #         'rewards': target_buffer_agent.rewards,
        #     })

            # ckptr_agent = PyTreeCheckpointer()
            # ckptr_agent.save(
            #     os.getcwd() + "/saved_finetuned_g",
            #     potential_pairs.state_g,
            #     force=True,
            #     save_args=orbax_utils.save_args_from_target(potential_pairs.state_g),
            # )

        if i == 1000:

            for _ in range(10000):
                target_data = target_buffer_agent.sample(1024)
                expert_data = source_expert_ds.sample(1024)

                not_proj, potential_proj = update_not_elem(joint_ot_agent, not_proj, expert_data, target_data)
                # not_pairs, encoded_source, encoded_target = update_not(joint_ot_agent, not_pairs, potential_proj, expert_data, target_data)


        if i >= 1000 and i % 1 == 0:

            for _ in range(2):
                target_data = buffer_disc.sample(1024)
                expert_data = source_expert_ds.sample(1024)
                # target_data_2 = target_buffer_agent.sample(1024)

                not_proj, potential_proj = update_not_elem(joint_ot_agent, not_proj, expert_data, target_data)
                D, d_key, encoded_source, encoded_target = update_disc(joint_ot_agent, D, potential_proj, expert_data, target_data, d_key)
                # not_pairs, encoded_source, encoded_target = update_not(joint_ot_agent, not_pairs, potential_proj, expert_data, target_data)
            
        if i >= 5000:

            target_data = target_buffer_agent.sample(1024, goal_conditioned=True)
            expert_data = source_expert_ds.sample(1024, goal_conditioned=True)

            # if i % 20 == 0 and i > 20000:
            #     joint_ot_agent, info = joint_ot_agent.update(expert_data, target_data, potential_elems, potential_pairs, update_not=True)
           
            
            for _ in range(1):
                target_data = target_buffer_agent.sample(512)
                expert_data = source_expert_ds.sample(512)
                # not_agent_pairs, potential_pairs, encoded_source, encoded_target, not_info = update_not(joint_ot_agent, not_agent_pairs, expert_data, target_data)
                # rewards = compute_reward_from_disc(joint_ot_agent, D, target_data.observations, target_data.next_observations)
                # rewards = compute_reward_from_not(joint_ot_agent, not_pairs.to_dual_potentials(), target_data.observations, target_data.next_observations)
                rewards = target_data.rewards
                rewards = (rewards - jnp.mean(rewards)) / jnp.std(rewards)
                assert len(rewards) == 512

                target_data = Batch(observations=target_data.observations,
                     actions=target_data.actions,
                     rewards=np.asarray(rewards) - 1,
                     masks=target_data.masks,
                     next_observations=target_data.next_observations)

                actor_update_info = agent.update(target_data)

        if i % 5000 == 0 and  i > 10000:
            fig, ax = plt.subplots()
            both = jnp.concatenate([encoded_target, encoded_source], axis=0)
            ax.scatter(both[:, 0], both[:, 1], c=['orange']*encoded_target.shape[0] + ['blue']*encoded_source.shape[0])
            fig.savefig(f"viz_plots/both_{i}.png")
            

        if i % cfg.log_interval == 0:
            print("rewards", rewards[:10] - 1)
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