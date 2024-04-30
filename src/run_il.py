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
from agents.notdual import NotAgent
from icvf_utils.icvf_networks import JointNOTAgent

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo

import ott

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

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped

@functools.partial(jax.jit, static_argnums=(1))
def compute_reward_from_not(encoders, f_potential, f_params, agent_observations, agent_next_observations):
    # sa = agent_observations[:2]
    # sa_next = agent_next_observations[:2]
    # sa = encoders(agent_observations, method='encode_agent')
    # sa_next = encoders(agent_next_observations, method='encode_agent')
    agent_pairs = jnp.concatenate([sa, sa_next], axis=-1)
    reward = f_potential(f_params)(agent_pairs) # *2 - jnp.linalg.norm(agent_pairs)# ** 2
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
    
    # if cfg.use_pretrained_icvf:
    #     print("Loading ICVF")
    #     from orbax.checkpoint import PyTreeCheckpointer
        
    #     checkpointer = PyTreeCheckpointer()
    #     restored_ckpt = checkpointer.restore("/home/m_bobrin/CrossDomainIL/outputs/2024-04-21/15-07-38/saved_model")
    #     value_def = create_icvf("multilinear", hidden_dims=[512, 512, 512], ensemble=True)
    #     icvf_value = TrainState.create(value_def, params=restored_ckpt['value']['params'])

    if cfg.optimal_transport:
        from datasets.gc_dataset import GCSDataset
        
        neural_f = ott.neural.networks.potentials.PotentialMLP(
            dim_hidden=[128, 128, 128, 128],
            is_potential=True,
            act_fn=jax.nn.leaky_relu
        )
        neural_g = ott.neural.networks.potentials.PotentialMLP(
            dim_hidden=[128, 128, 128, 128],
            is_potential=True,
            act_fn=jax.nn.leaky_relu
        )
        
        optimizer_f = optax.adam(learning_rate=1e-4, b1=0.9, b2=0.99)
        optimizer_g = optax.adam(learning_rate=1e-4, b1=0.9, b2=0.99)

        not_agent = NotAgent(
            embed_dim=8,
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            num_train_iters=10_000,
            expert_loss_coef=1.)
        
        joint_ot_agent = JointNOTAgent.create(
            cfg.seed,
            latent_dim=8,
            not_agent=not_agent,
            target_obs=target_random_buffer.observations[0],
            source_obs=combined_source_ds.observations[0],
        )
        #gc_icvf_dataset_target = GCSDataset(dataset=target_random_buffer, **GCSDataset.get_default_config())
    
    (observation, info), done = env.reset(seed=cfg.seed), False
    
    os.makedirs("viz_plots", exist_ok=True)
    for i in tqdm(range(300_000), leave=True):
        target_data = target_random_buffer.sample(512, icvf=True)
        source_data = combined_source_ds.sample(512, icvf=True)
        if i % 10_000 == 0:
            joint_ot_agent, info = joint_ot_agent.update(source_data, target_data, update_not=True)
        else:
            joint_ot_agent, info = joint_ot_agent.update(source_data, target_data, update_not=False)
        
        if i % 10_000 == 1:
            # Target domain
            pca = PCA(n_components=2)
            tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
            encoded_target = joint_ot_agent.net(target_data.observations, method='encode_target')
            fitted_pca = pca.fit_transform(encoded_target)
            fitted_tsne = tsne.fit_transform(encoded_target)
            fig, ax = plt.subplots()
            ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
            fig.savefig(f"viz_plots/target_{i}_pca.png")
            fig, ax = plt.subplots()
            ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
            fig.savefig(f"viz_plots/target_{i}_tsne.png")
            
            #####################################################################################
            # Source domain
            pca = PCA(n_components=2)
            tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
            encoded_source = joint_ot_agent.net(source_data.observations, method='encode_source')
            fitted_pca = pca.fit_transform(encoded_source)
            fitted_tsne = tsne.fit_transform(encoded_source)

            fig, ax = plt.subplots()
            ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
            fig.savefig(f"viz_plots/source_{i}_pca.png")

            fig, ax = plt.subplots()
            ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
            fig.savefig(f"viz_plots/source_{i}_tsne.png")
            
    ckptr = PyTreeCheckpointer()
    ckptr.save(
        os.getcwd() + "/saved_encoding_agent",
        joint_ot_agent,
        force=True,
        save_args=orbax_utils.save_args_from_target(joint_ot_agent),
    )
    
    # Stage 1. Pretrain Encoders
    ###
    # os.makedirs("viz_plots", exist_ok=True)
    # for i in tqdm(range(250_000), leave=True):
    #     agent_data = target_random_buffer.sample(256, icvf=True)
    #     icvf_enc_target, info = icvf_enc_target.update(agent_data)
        
    #     if i % 50_000 == 1:
            
    #         pca = PCA(n_components=2)
    #         tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
            
    #         fitted_pca = pca.fit_transform(icvf_enc_target.net(agent_data.observations[:]))
    #         fitted_tsne = tsne.fit_transform(icvf_enc_target.net(agent_data.observations[:]))

    #         fig, ax = plt.subplots()
    #         ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
    #         fig.savefig(f"viz_plots/target_{i}_pca.png")

    #         fig, ax = plt.subplots()
    #         ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
    #         fig.savefig(f"viz_plots/target_{i}_tsne.png")
        
    # ckptr = PyTreeCheckpointer()
    # ckptr.save(
    #     os.getcwd() + "/saved_encoder_target",
    #     icvf_enc_target,
    #     force=True,
    #     save_args=orbax_utils.save_args_from_target(icvf_enc_target),
    # )
    # print("Finished Training Target Domain Encoder")
    
    # for i in tqdm(range(250_000), leave=True):
    #     agent_data = combined_source_ds.sample(256, icvf=True)
    #     icvf_enc_source, info = icvf_enc_source.update(agent_data)
        
    #     if i % 50_000 == 1:
    #         pca = PCA(n_components=2)
    #         tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
            
    #         fitted_pca = pca.fit_transform(icvf_enc_source.net(agent_data.observations[:]))
    #         fitted_tsne = tsne.fit_transform(icvf_enc_source.net(agent_data.observations[:]))

    #         fig, ax = plt.subplots()
    #         ax.scatter(fitted_pca[:, 0], fitted_pca[:, 1], label="pca")
    #         fig.savefig(f"viz_plots/source_{i}_pca.png")

    #         fig, ax = plt.subplots()
    #         ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne")
    #         fig.savefig(f"viz_plots/source_{i}_tsne.png")
    # ckptr = PyTreeCheckpointer()
    # ckptr.save(
    #     os.getcwd() + "/saved_encoder_source",
    #     icvf_enc_source,
    #     force=True,
    #     save_args=orbax_utils.save_args_from_target(icvf_enc_source),
    # )
    # exit()
    exit()
    # Stage 2. PRETRAIN Neural OT
    pbar = tqdm(range(5_000), leave=True)
    for i in pbar:
        agent_data = target_random_buffer.sample(1024)
        expert_data = source_expert_ds.sample(1024)
        random_data = source_random_ds.sample(1024)
        loss_elem, loss_pairs, w_dist_elem, w_dist_pairs = not_agent.optimize_not(agent_data, expert_data, random_data)

        if i % 100 == 0:
            #se = not_agent.encoders_state(expert_data.observations, method='encode_expert')
            #sa = not_agent.encoders_state(agent_data.observations, method='encode_agent')
            sink = sinkhorn_loss(sa, se)
            info = {"sink_dist": sink,
                          "w_dist_elem": w_dist_elem,
                          "w_dist_pairs": w_dist_pairs}
            pbar.set_postfix(info)
            wandb.log(info)
    
    pbar = tqdm(range(1, cfg.max_steps + 1), leave=True)
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
            mask = 0.
        reward = compute_reward_from_not(not_agent.encoders_state, not_agent.neural_dual_pairs.state_f.potential_value_fn,
                                         not_agent.neural_dual_pairs.state_f.params, observation, next_observation)
        target_random_buffer.insert(observation, action, reward, mask, float(done), next_observation)
        observation = next_observation
        
        if done:
            wandb.log({f"Training/rewards": info['episode']['r'],
                    f"Training/episode length": info['episode']['l']})
            (observation, info), done = env.reset(), False

        if i % 10_000 == 0:
            os.makedirs(name='rewards_potential', exist_ok=True)
            np.save("rewards_potential/rewards.npy", arr={
                'rewards': target_random_buffer.rewards,
            })
        
        if i >= cfg.algo.start_training:
            for _ in range(cfg.algo.updates_per_step):
                agent_data = target_random_buffer.sample(256) # Maybe add prioritization?
                expert_data = source_expert_ds.sample(1024)
                random_data = source_random_ds.sample(256)
                loss_elem, loss_pairs, w_dist_elem, w_dist_pairs = not_agent.optimize_not(agent_data, expert_data, random_data)
                #if i % 500 == 0:
                    # if i % 10_000 == 0:
                    #     info = not_agent.optimize_encoders(agent_data, expert_data, random_data)
                if i % 10_000 == 0:
                    pbar.set_postfix(info)
                    # se = not_agent.encoders_state(expert_data.observations, method='encode_expert')
                    # sn = not_agent.encoders_state(random_data.observations, method='encode_expert')
                    #sa = not_agent.encoders_state(agent_data.observations, method='encode_agent')
                    sa = agent_data.observations[:, :2]
                    se = expert_data.observations[:, :2]
                    sn = random_data.observations[:, :2]
                    
                    sink = sinkhorn_loss(sa, se)
                    print({"sink_dist": sink.item(),
                            "w_dist_elem": w_dist_elem.item()})
                    os.makedirs("plots", exist_ok=True)
                    plot_embeddings(sn, se, sa)
                actor_update_info = agent.update(agent_data)

        if i % cfg.log_interval == 0:
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