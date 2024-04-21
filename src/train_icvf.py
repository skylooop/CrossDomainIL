import numpy as np
import jax
import os
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import wandb
import rootutils
import random
from orbax.checkpoint import PyTreeCheckpointer
from flax.training import orbax_utils

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')

from icvf_utils.gcdataset import GCSDataset
from jaxrl_m.dataset import Dataset
from icvf_utils.icvf_learner import create_learner
from icvf_utils.icvf_networks import create_icvf

@hydra.main(version_base='1.4', config_path=str(ROOT/"configs"), config_name="icvf")
def main(cfg: DictConfig) -> None:
    #################
    # python src/train_icvf.py hydra.job.chdir=True
    #################
    wandb.init(
        mode="offline",
        config=dict(cfg),
    )
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    target_random = np.load(hydra.utils.get_original_cwd() + "/prep_data/pointumaze/rand_target/random_policy.npy", allow_pickle=True).item()
    expert_source = np.load(hydra.utils.get_original_cwd() +"/prep_data/pointumaze/expert_source/trained_expert.npy", allow_pickle=True).item()
    
    target_random_ds = Dataset.create(observations=target_random['observations'],
                           actions=target_random['actions'],
                           rewards=target_random['rewards'],
                           dones_float=target_random['dones'],
                           masks=1.0 - target_random['dones'],
                           next_observations=target_random['next_observations'])
    expert_source_ds = Dataset.create(observations=expert_source['observations'],
                           actions=expert_source['actions'],
                           rewards=expert_source['rewards'],
                           dones_float=expert_source['dones'],
                           masks=1.0 - expert_source['dones'],
                           next_observations=expert_source['next_observations'])
    
    # Learn ICVF on agent data
    gc_dataset = GCSDataset(target_random_ds, **GCSDataset.get_default_config().to_dict())
    example_batch = gc_dataset.sample(1)

    hidden_dims = tuple([int(h) for h in cfg.algo.hidden_dims])
    value_def = create_icvf("multilinear", hidden_dims=hidden_dims)

    agent = create_learner(cfg.seed,
                    example_batch['observations'],
                    value_def,
                    **dict(cfg.algo))
    target_random_sample = target_random_ds.sample(20)
    source_expert_sample = expert_source_ds.sample(20)
    
    visualizer = DebugPlotGenerator(target_random_sample, gc_dataset=gc_dataset)

    for i in tqdm(range(1, cfg.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        batch = gc_dataset.sample(cfg.algo.batch_size)
        agent, update_info = agent.update(batch)

        if i % cfg.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            wandb.log(train_metrics, step=i)

        if i % cfg.eval_interval == 0:
            visualizations = visualizer.generate_debug_plots(agent)
            eval_metrics = {f'visualizations/{k}': v for k, v in visualizations.items()}
            wandb.log(eval_metrics, step=i)

        if i % cfg.save_interval == 0:
            ckptr = PyTreeCheckpointer()
            ckptr.save(
                os.getcwd() + "/saved_model",
                agent,
                force=True,
                save_args=orbax_utils.save_args_from_target(agent),
            )

###################################################################################################
#
# Creates wandb plots
#
###################################################################################################
class DebugPlotGenerator:
    def __init__(self, target_agent_samples, gc_dataset):
        init_state = np.copy(gc_dataset.dataset['observations'][0])
        init_state[:2] = (0, 0)
        self.example_trajectory = jnp.sort(target_agent_samples['observations'], axis=0)
        
        fig, ax = plt.subplots()
        ax.scatter(target_agent_samples['observations'][:, 0], target_agent_samples['observations'][:, 1])
        plt.tight_layout()
        plt.savefig("example_trajectory.png")
        plt.close()
        
    def generate_debug_plots(self, agent):
        example_trajectory = self.example_trajectory
        traj_metrics = get_traj_v(agent, example_trajectory)
        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(traj_metrics['dist_to_beginning'])
        ax[0].title.set_text('dist_to_beginning')
        
        ax[1].plot(traj_metrics['dist_to_middle'])
        ax[1].title.set_text('dist_to_middle')
        
        ax[2].plot(traj_metrics['dist_to_end'])
        ax[2].title.set_text('dist_to_end')
        plt.tight_layout()
        plt.savefig("icvf_eval.png")
        plt.close()
        return traj_metrics

###################################################################################################
#
# Helper functions for visualization
#
###################################################################################################

@jax.jit
def get_values(agent, observations, intent):
    def get_v(observations, intent):
        intent = intent.reshape(1, -1)
        intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
        v1, v2 = agent.value(observations, intent_tiled, intent_tiled)
        return (v1 + v2) / 2    
    return get_v(observations, intent)

@jax.jit
def get_policy(agent, observations, intent):
    def v(observations):
        def get_v(observations, intent):
            intent = intent.reshape(1, -1)
            intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
            v1, v2 = agent.value(observations, intent_tiled, intent_tiled)
            return (v1 + v2) / 2    
            
        return get_v(observations, intent).mean()

    grads = jax.grad(v)(observations)
    policy = grads[:, :2]
    return policy / jnp.linalg.norm(policy, axis=-1, keepdims=True)

@jax.jit
def get_debug_statistics(agent, batch):
    def get_info(s, g, z):
        if agent.config['no_intent']:
            return agent.value(s, g, jnp.ones_like(z), method='get_info')
        else:
            return agent.value(s, g, z, method='get_info')

    s = batch['observations']
    g = batch['goals']
    z = batch['desired_goals'] # = intent

    info_ssz = get_info(s, s, z)
    info_szz = get_info(s, z, z)
    info_sgz = get_info(s, g, z)
    info_sgg = get_info(s, g, g)
    info_szg = get_info(s, z, g)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': jnp.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(info_sgz['psi'], axis=-1).mean(),
        }
    else:
        stats = {}

    stats.update({
        'v_ssz': info_ssz['v'].mean(),
        'v_szz': info_szz['v'].mean(),
        'v_sgz': info_sgz['v'].mean(),
        'v_sgg': info_sgg['v'].mean(),
        'v_szg': info_szg['v'].mean(),
        'diff_szz_szg': (info_szz['v'] - info_szg['v']).mean(),
        'diff_sgg_sgz': (info_sgg['v'] - info_sgz['v']).mean(),
    })
    return stats

@jax.jit
def get_gcvalue(agent, s, g, z):
    v_sgz_1, v_sgz_2 = agent.value(s, g, z)
    return (v_sgz_1 + v_sgz_2) / 2

def get_v_zz(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal, goal)

def get_v_gz(agent, initial_state, target_goal, observations):
    initial_state = jnp.tile(initial_state, (observations.shape[0], 1))
    target_goal = jnp.tile(target_goal, (observations.shape[0], 1))
    return get_gcvalue(agent, initial_state, observations, target_goal)

@jax.jit
def get_traj_v(agent, observations):
    def get_v(s, g):
        return agent.value(s[None], g[None], g[None]).mean()
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }

####################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        wandb.finish()
        exit()