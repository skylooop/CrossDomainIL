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
from orbax.checkpoint import PyTreeCheckpointer
from flax.training import orbax_utils
import optax

import matplotlib.pyplot as plt
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')

import functools
from utils.const import SUPPORTED_ENVS
from utils.loading_data import prepare_buffers_for_il
from agents.notdual import ENOTCustom, W2NeuralDualCustom
from icvf_utils.icvf_networks import JointNOTAgent

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo

import ott
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

def plot_embeddings(sn, se, tn) -> None:
    fig, ax = plt.subplots()
    
    ax.scatter(sn[:, 0], sn[:, 1], c='red', label='sn')
    ax.scatter(tn[:, 0], tn[:, 1], c='green', label='tn')
    ax.scatter(se[:, 0], se[:, 1], c='orange', label='se')
    plt.legend()
    plt.savefig('plots/embedding.png')
    wandb.log({'Embeddings': wandb.Image(fig)})
    plt.close()

def update_not(joint_ot_agent, not_agent_elems, not_agent_pairs, batch_source, batch_target):
    encoded_source = joint_ot_agent.ema_get_phi_source(batch_source.observations)
    encoded_target = joint_ot_agent.ema_get_phi_target(batch_target.observations)
    # encoded_source_next = joint_ot_agent.get_phi_source(batch_source.next_observations)
    # encoded_target_next = joint_ot_agent.get_phi_target(batch_target.next_observations)
    
    new_not_agent_elems, loss_elems, w_dist_elems = not_agent_elems.update(encoded_source, encoded_target)
    potentials_elems = new_not_agent_elems.to_dual_potentials(finetune_g=True)
    # new_not_agent_pairs, loss_pairs, w_dist_pairs = not_agent_pairs.update(batch_source=jnp.concatenate((encoded_source, encoded_source_next), axis=-1),
    #                                                         batch_target=jnp.concatenate((encoded_target, encoded_target_next), axis=-1))

    potentials_pairs = not_agent_pairs.to_dual_potentials(finetune_g=True)
    return not_agent_pairs, new_not_agent_elems, potentials_elems, potentials_pairs, encoded_source, encoded_target, {"loss_elems": loss_elems, "w_dist_elems": w_dist_elems}

@jax.jit
def compute_reward_from_not(not_agent, potential_pairs, obs, next_obs, expert_obs, expert_next_obs):
    encoded_target = jnp.concatenate(not_agent(obs, method='encode_target'), -1)
    encoded_target_next = jnp.concatenate(not_agent(next_obs, method='encode_target'), -1)
    encoded_expert = jnp.concatenate(not_agent(expert_obs, method='encode_source'), -1)
    encoded_expert_next = jnp.concatenate(not_agent(expert_next_obs, method='encode_source'), -1)
    f, g = potential_pairs.get_fg()
    reward = -g(jnp.concatenate([encoded_target, encoded_target_next], axis=-1)) - jax.vmap(f)(jnp.concatenate([encoded_expert, encoded_expert_next], axis=-1)).mean()
    return reward

@jax.jit
def compute_reward_from_not_elem(not_agent, potential_elems, observation):
    encoded_target = jnp.concatenate(not_agent(observation, method='encode_target'), -1)
    f, g = potential_elems.get_fg()
    reward = g(encoded_target)
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
        mode="offline",
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
    
    source_expert_ds, source_random_ds, combined_source_ds, target_random_buffer = prepare_buffers_for_il(cfg=cfg,
                                                                                                          target_obs_space=eval_env.observation_space,
                                                                                                          target_act_space=eval_env.action_space)
    env = TimeLimit(env, max_episode_steps=episode_limit)
    env = RecordEpisodeStatistics(env)
    
    eval_env = TimeLimit(eval_env, max_episode_steps=episode_limit)
    eval_env = RecordVideo(eval_env, video_folder='agent_video', episode_trigger=lambda x: (x + 1) % cfg.save_video_each_ep == 0)
    eval_env = RecordEpisodeStatistics(eval_env, deque_size=cfg.save_expert_episodes)
    
    agent = hydra.utils.instantiate(cfg.algo)(observations=env.observation_space.sample()[None],
                                                     actions=env.action_space.sample()[None])
    if cfg.optimal_transport:
        from datasets.gc_dataset import GCSDataset
        from ott.neural.methods.expectile_neural_dual import MLP as ExpectileMLP
        from ott.neural.methods import neuraldual
        
        # neural_f = ott.neural.networks.potentials.PotentialMLP(
        #     dim_hidden=[512, 512, 512],
        #     is_potential=True,
        #     act_fn=jax.nn.gelu
        # )
        # neural_g = ott.neural.networks.potentials.PotentialMLP(
        #     dim_hidden=[512, 512, 512],
        #     is_potential=True,
        #     act_fn=jax.nn.gelu
        # )
        
        neural_f = ExpectileMLP(dim_hidden=[512, 512, 512, 1], act_fn=jax.nn.elu)
        neural_g = ExpectileMLP(dim_hidden=[512, 512, 512, 1], act_fn=jax.nn.elu)
        optimizer_f = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)
        optimizer_g = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)
        latent_dim = 32
        
        not_agent_elems = ENOTCustom(
            dim_data=latent_dim, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.SqEuclidean(),
            expectile = 0.99,
            expectile_loss_coef = 0.5, # 0.4
            use_dot_product=False,
            is_bidirectional=True
        )
        not_agent_pairs = ENOTCustom(
            dim_data=latent_dim * 2, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.SqEuclidean(),
            expectile = 0.99,
            expectile_loss_coef = 0.5, # 0.4
            use_dot_product=False,
            is_bidirectional=True
        )
        
        joint_ot_agent = JointNOTAgent.create(
            cfg.seed,
            latent_dim=latent_dim,
            target_obs=target_random_buffer.observations[0],
            source_obs=combined_source_ds.observations[0],
        )
        
        #gc_icvf_dataset_target = GCSDataset(dataset=target_random_buffer, **GCSDataset.get_default_config())
        
        #General Pretraining
        for i in tqdm(range(10_000), desc="Pretraining NOT", position=1, leave=False):
            target_data = target_random_buffer.sample(1024, goal_conditioned=True)
            source_data = combined_source_ds.sample(1024, goal_conditioned=True)
            not_agent_pairs, not_agent_elems, potential_elems, potential_pairs, encoded_source, encoded_target, not_info = update_not(joint_ot_agent, not_agent_elems, not_agent_pairs,
                                                                                                    source_data, target_data)
        os.makedirs("viz_plots", exist_ok=True)
        for i in tqdm(range(300_005), leave=True):
            target_data = target_random_buffer.sample(1024, goal_conditioned=True)
            source_data = combined_source_ds.sample(1024, goal_conditioned=True)
            if i % 5 == 0:
                joint_ot_agent, info = joint_ot_agent.update(source_data, target_data, potential_elems, potential_pairs, update_not=True)
            else:
                joint_ot_agent, info = joint_ot_agent.update(source_data, target_data, potential_elems, potential_pairs, update_not=False)
            
            if i % 10 == 0:
                for _ in range(30):
                    target_data = target_random_buffer.sample(1024, goal_conditioned=True)
                    source_data = combined_source_ds.sample(1024, goal_conditioned=True)
                    #source_data = source_expert_ds.sample(1024, icvf=True)
                    not_agent_pairs, not_agent_elems, potential_elems, potential_pairs, encoded_source, encoded_target, not_info = update_not(joint_ot_agent, not_agent_elems, not_agent_pairs, 
                                                                                                            source_data, target_data)
            if i % 5_000 == 1: # 20_000
                ckptr_agent = PyTreeCheckpointer()
                ckptr_agent.save(
                    os.getcwd() + "/saved_encoding_agent",
                    joint_ot_agent,
                    force=True,
                    save_args=orbax_utils.save_args_from_target(joint_ot_agent),
                )
                ckptr_pairs_potentials = PyTreeCheckpointer()
                ckptr_pairs_potentials.save(
                    os.getcwd() + "/saved_potentials_pairs/state_f",
                    not_agent_pairs.state_f,
                    force=True,
                    save_args=orbax_utils.save_args_from_target(not_agent_pairs.state_f),
                )
                ckptr_pairs_potentials.save(
                    os.getcwd() + "/saved_potentials_pairs/state_g",
                    not_agent_pairs.state_g,
                    force=True,
                    save_args=orbax_utils.save_args_from_target(not_agent_pairs.state_g),
                )
                ckptr_elems_potentials = PyTreeCheckpointer()
                ckptr_elems_potentials.save(
                    os.getcwd() + "/saved_potentials_elems/state_f",
                    not_agent_elems.state_f,
                    force=True,
                    save_args=orbax_utils.save_args_from_target(not_agent_elems.state_f),
                )
                ckptr_elems_potentials = PyTreeCheckpointer()
                ckptr_elems_potentials.save(
                    os.getcwd() + "/saved_potentials_elems/state_g",
                    not_agent_elems.state_g,
                    force=True,
                    save_args=orbax_utils.save_args_from_target(not_agent_elems.state_g),
                )
            if i % 5_010 == 0:
                # Target domain
                #pca = PCA(n_components=2)
                tsne = TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=cfg.seed)
                
                encoded_target = joint_ot_agent.ema_get_phi_target(target_data.observations)
                # encoded_target_next = joint_ot_agent.ema_get_phi_target(target_data.next_observations)
                # encoded_target = jnp.concatenate(encoded_target, -1)
                # encoded_target_next = jnp.concatenate(encoded_target_next, -1)
                
                #fitted_pca = pca.fit_transform(encoded_target)
                fitted_tsne = tsne.fit_transform(encoded_target)
                # fig, ax = plt.subplots()
                # ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
                # fig.savefig(f"viz_plots/target_{i}_pca.png")
                fig, ax = plt.subplots()
                ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
                fig.savefig(f"viz_plots/target_{i}_tsne.png")
                
                #####################################################################################
                # Source domain
                #pca = PCA(n_components=2)
                tsne = TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=cfg.seed)
                encoded_source = joint_ot_agent.ema_get_phi_source(source_data.observations)
                # encoded_source_next = joint_ot_agent.get_phi_source(source_data.next_observations)
                # encoded_source = jnp.concatenate(encoded_source, -1)
                # encoded_source_next = jnp.concatenate(encoded_source_next, -1)

                #fitted_pca = pca.fit_transform(encoded_source)
                fitted_tsne = tsne.fit_transform(encoded_source)

                # fig, ax = plt.subplots()
                # ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
                # fig.savefig(f"viz_plots/source_{i}_pca.png")

                fig, ax = plt.subplots()
                ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
                fig.savefig(f"viz_plots/source_{i}_tsne.png")
                
                ############################
                # BOTH
                tsne = TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=cfg.seed)
                both_domains = np.concatenate([encoded_target, encoded_source], axis=0)
                tsne_both = tsne.fit_transform(both_domains)
                
                fig, ax = plt.subplots()
                ax.scatter(tsne_both[:, 0], tsne_both[:, 1], c=['orange']*encoded_target.shape[0] + ['blue']*encoded_source.shape[0])
                fig.savefig(f"viz_plots/both_{i}_tsne.png")
                
                neural_dual_dist_elems = potential_elems.distance(encoded_source, encoded_target)
                # neural_dual_dist_pairs = potential_pairs.distance(jnp.concatenate((encoded_source, encoded_source_next), axis=-1),
                #                                                 jnp.concatenate((encoded_target, encoded_target_next), axis=-1))
                sinkhorn_dist_elems = sinkhorn_loss(encoded_source, encoded_target)
                # sinkhorn_dist_pairs = sinkhorn_loss(jnp.concatenate((encoded_source, encoded_source_next), axis=-1),
                #                                     jnp.concatenate((encoded_target, encoded_target_next), axis=-1))
                
                print(f"\nNeural dual distance between elements in source and target data: {neural_dual_dist_elems:.5f}")
                # print(f"Neural dual distance between pairs in source and target data: {neural_dual_dist_pairs:.5f}")
                print(f"Sinkhorn distance between elements in source and target data: {sinkhorn_dist_elems:.5f}")
                # print(f"Sinkhorn distance between pairs in source and target data: {sinkhorn_dist_pairs:.5f}")
        
        ########################## SAVING ####################################      
        ckptr_agent = PyTreeCheckpointer()
        ckptr_agent.save(
            os.getcwd() + "/saved_encoding_agent",
            joint_ot_agent,
            force=True,
            save_args=orbax_utils.save_args_from_target(joint_ot_agent),
        )
        ckptr_pairs_potentials = PyTreeCheckpointer()
        ckptr_pairs_potentials.save(
            os.getcwd() + "/saved_potentials_pairs/state_f",
            not_agent_pairs.state_f,
            force=True,
            save_args=orbax_utils.save_args_from_target(not_agent_pairs.state_f),
        )
        ckptr_pairs_potentials.save(
            os.getcwd() + "/saved_potentials_pairs/state_g",
            not_agent_pairs.state_g,
            force=True,
            save_args=orbax_utils.save_args_from_target(not_agent_pairs.state_g),
        )
        ckptr_elems_potentials = PyTreeCheckpointer()
        ckptr_elems_potentials.save(
            os.getcwd() + "/saved_potentials_elems/state_f",
            not_agent_elems.state_f,
            force=True,
            save_args=orbax_utils.save_args_from_target(not_agent_elems.state_f),
        )
        ckptr_elems_potentials = PyTreeCheckpointer()
        ckptr_elems_potentials.save(
            os.getcwd() + "/saved_potentials_elems/state_g",
            not_agent_elems.state_g,
            force=True,
            save_args=orbax_utils.save_args_from_target(not_agent_elems.state_g),
        )
    else:
        neural_f = ott.neural.networks.potentials.PotentialMLP(
            dim_hidden=[64, 64, 64, 64],
            is_potential=True,
            act_fn=jax.nn.elu
        )
        neural_g = ott.neural.networks.potentials.PotentialMLP(
            dim_hidden=[64, 64, 64, 64],
            is_potential=True,
            act_fn=jax.nn.elu
        )
        
        optimizer_f = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)
        optimizer_g = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.99)
        latent_dim = 16
        
        not_agent_elems = ENOTCustom(
            dim_data=latent_dim, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.SqEuclidean(),
            expectile = 0.99,
            expectile_loss_coef = 0.5, # 0.4
            num_train_iters=10_000
        )
        not_agent_pairs = ENOTCustom(
            dim_data=latent_dim * 2, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            cost_fn=costs.SqEuclidean(),
            expectile = 0.99,
            expectile_loss_coef = 0.5, # 0.4
            num_train_iters=10_000
        )
        joint_ot_agent = JointNOTAgent.create(
            cfg.seed,
            latent_dim=latent_dim,
            target_obs=target_random_buffer.observations[0],
            source_obs=combined_source_ds.observations[0],
        )
        
        checkpointer_net = PyTreeCheckpointer()
        checkpointer_potentials_pairs_state_f = PyTreeCheckpointer()
        checkpointer_potentials_elems_state_f = PyTreeCheckpointer()
        checkpointer_potentials_pairs_state_g = PyTreeCheckpointer()
        checkpointer_potentials_elems_state_g = PyTreeCheckpointer()

        restored_ckpt_target = checkpointer_net.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_encoding_agent")
        net = joint_ot_agent.net.replace(params=restored_ckpt_target['net']['params'])
        joint_ot_agent = joint_ot_agent.replace(net=net)

        potential_pairs_state_f_ckpt = checkpointer_potentials_pairs_state_f.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_pairs/state_f")
        potential_pairs_state_g_ckpt = checkpointer_potentials_pairs_state_g.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_pairs/state_g")

        potential_elems_state_f_ckpt = checkpointer_potentials_elems_state_f.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_elems/state_f")
        potential_elems_state_g_ckpt = checkpointer_potentials_elems_state_g.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-05-09/00-17-21/saved_potentials_elems/state_g")

        state_f_elems = not_agent_elems.state_f.replace(params=potential_elems_state_f_ckpt['params'])
        state_g_elems = not_agent_elems.state_g.replace(params=potential_elems_state_g_ckpt['params'])

        state_f_pairs = not_agent_pairs.state_f.replace(params=potential_pairs_state_f_ckpt['params'])
        state_g_pairs = not_agent_pairs.state_g.replace(params=potential_pairs_state_g_ckpt['params'])

        not_agent_pairs.state_f = state_f_pairs
        not_agent_pairs.state_g = state_g_pairs

        not_agent_elems.state_f = state_f_elems
        not_agent_elems.state_g = state_g_elems

        potential_pairs = not_agent_pairs.to_dual_potentials()
        potential_elems = not_agent_elems.to_dual_potentials()
    
    (observation, info), done = env.reset(seed=cfg.seed), False
    
    potential_pairs = not_agent_pairs.to_dual_potentials()
    potential_elems = not_agent_elems.to_dual_potentials()
    os.makedirs("viz_plots", exist_ok=True)
    
    pbar = tqdm(range(1, cfg.max_steps + 1), leave=True)
    expert_data = source_expert_ds.sample(256)
    for i in pbar:
        if i < cfg.algo.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, _, terminated, truncated, info = env.step(action) # reward
        done = terminated or truncated
        
        if not done:
            mask = 1.0
        else:
            mask = 0.
        reward = compute_reward_from_not(joint_ot_agent.net, potential_pairs, observation, next_observation,
                                         expert_data.observations, expert_data.next_observations)
        #reward = compute_reward_from_not_elem(joint_ot_agent.net, potential_elems, observation)
        
        target_random_buffer.insert(observation, action, reward, mask, float(done), next_observation)
        observation = next_observation
        if (observation[0] < 2 and observation[1] > 4) or terminated:
            print(f"YES: {terminated}")
            
        if done:
            wandb.log({f"Training/rewards": info['episode']['r'],
                    f"Training/episode length": info['episode']['l']})
            (observation, info), done = env.reset(), False
            
        if i % 1_000 == 0 or i == 0:
            os.makedirs(name='rewards_potential', exist_ok=True)
            np.save("rewards_potential/rewards.npy", arr={
                'rewards': target_random_buffer.rewards,
            })

            ckptr_agent = PyTreeCheckpointer()
            ckptr_agent.save(
                os.getcwd() + "/saved_finetuned_g",
                potential_pairs.state_g,
                force=True,
                save_args=orbax_utils.save_args_from_target(potential_pairs.state_g),
            )
            
        if i >= cfg.algo.start_training:
            for _ in range(cfg.algo.updates_per_step):
                target_data = target_random_buffer.sample(256)
                #source_data = combined_source_ds.sample(256)
                expert_data = source_expert_ds.sample(256)
                not_agent_pairs, not_agent_elems, potential_elems, potential_pairs, encoded_source, encoded_target, not_info = update_not(joint_ot_agent, not_agent_elems, not_agent_pairs,
                                                                                                        expert_data, target_data)
                actor_update_info = agent.update(target_data)

        if i % cfg.log_interval == 0:
            wandb.log({f"Training/Critic loss": actor_update_info['critic_loss'],
                    f"Training/Actor loss": actor_update_info['actor_loss'],
                    f"Training/Entropy": actor_update_info['entropy'],
                    f"Training/Temperature": actor_update_info['temperature'],
                    f"Training/Temp loss": actor_update_info['temp_loss']})
            #pca = PCA(n_components=2)
            tsne = TSNE(n_components=2, perplexity=40, n_iter=2000)
                
            encoded_target = joint_ot_agent.net(target_data.observations, method='encode_target')
            encoded_target_next = joint_ot_agent.net(target_data.next_observations, method='encode_target')
            encoded_target = jnp.concatenate(encoded_target, -1)
            encoded_target_next = jnp.concatenate(encoded_target_next, -1)
            
            #fitted_pca = pca.fit_transform(encoded_target)
            fitted_tsne = tsne.fit_transform(encoded_target)
            fig, ax = plt.subplots()
            # ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
            # fig.savefig(f"viz_plots/target_{i}_pca.png")
            fig, ax = plt.subplots()
            ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
            fig.savefig(f"viz_plots/target_{i}_tsne.png")
            
            #####################################################################################
            # Source domain
            tsne = TSNE(n_components=2, perplexity=40, n_iter=2000)
            encoded_source = joint_ot_agent.net(expert_data.observations, method='encode_source')
            encoded_source_next = joint_ot_agent.net(expert_data.next_observations, method='encode_source')
            encoded_source = jnp.concatenate(encoded_source, -1)
            encoded_source_next = jnp.concatenate(encoded_source_next, -1)

            fitted_tsne = tsne.fit_transform(encoded_source)

            # fig, ax = plt.subplots()
            # ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
            # fig.savefig(f"viz_plots/source_{i}_pca.png")

            fig, ax = plt.subplots()
            ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
            fig.savefig(f"viz_plots/source_{i}_tsne.png")
            
            ############################
            # BOTH
            tsne = TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=cfg.seed)
            both_domains = np.concatenate([encoded_target, encoded_source], axis=0)
            tsne_both = tsne.fit_transform(both_domains)
            
            fig, ax = plt.subplots()
            ax.scatter(tsne_both[:, 0], tsne_both[:, 1], c=['orange']*encoded_target.shape[0] + ['blue']*encoded_source.shape[0])
            fig.savefig(f"viz_plots/both_{i}_tsne.png")
            
            neural_dual_dist_elems = potential_elems.distance(encoded_source, encoded_target)
            # neural_dual_dist_pairs = potential_pairs.distance(jnp.concatenate((encoded_source, encoded_source_next), axis=-1),
            #                                                 jnp.concatenate((encoded_target, encoded_target_next), axis=-1))
            sinkhorn_dist_elems = sinkhorn_loss(encoded_source, encoded_target)
            # sinkhorn_dist_pairs = sinkhorn_loss(jnp.concatenate((encoded_source, encoded_source_next), axis=-1),
            #                                     jnp.concatenate((encoded_target, encoded_target_next), axis=-1))
            
            print(f"\nNeural dual distance between elements in source and target data: {neural_dual_dist_elems:.5f}")
            #print(f"Neural dual distance between pairs in source and target data: {neural_dual_dist_pairs:.5f}")
            print(f"Sinkhorn distance between elements in source and target data: {sinkhorn_dist_elems:.5f}")
            #print(f"Sinkhorn distance between pairs in source and target data: {sinkhorn_dist_pairs:.5f}")
                
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