import os
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from icvf_utils.gcdataset import GCSDataset
from jaxrl_m.dataset import Dataset
from icvf_utils.icvf_learner import create_learner
from icvf_utils.icvf_networks import create_icvf

import wandb
import rootutils
import random

ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='requirements.txt')
import pickle

@hydra.main(version_base='1.4', config_path=str(ROOT/"configs"), config_name="icvf")
def main(cfg: DictConfig) -> None:
    #################
    # python src/train_icvf.py hydra.job.chdir=True
    #################
    wandb.init(
        mode="offline",
        project=cfg.logger.project,
        config=dict(cfg),
        group="expert_" + f"{cfg.imitation_env.name}_{cfg.algo.name}",
    )
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    pre_collected_data = np.load("../outputs/2024-04-17/13-56-58/saved_prior/random_policy.npy", allow_pickle=True).item()
    pre_collected_ds = Dataset.create(observations=pre_collected_data['observations'],
                           actions=pre_collected_data['actions'],
                           rewards=pre_collected_data['rewards'],
                           dones_float=pre_collected_data['dones'],
                           masks=1.0 - pre_collected_data['dones'],
                           next_observations=pre_collected_data['next_observations'])
    
    gc_dataset = GCSDataset(pre_collected_ds, **GCSDataset.get_default_config().to_dict())
    example_batch = gc_dataset.sample(1)

    hidden_dims = tuple([int(h) for h in cfg.algo.hidden_dims])
    value_def = create_icvf(hidden_dims=hidden_dims)

    agent = create_learner(cfg.seed,
                    example_batch['observations'],
                    value_def,
                    **dict(cfg.algo))

    visualizer = DebugPlotGenerator(gc_dataset)

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
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=FLAGS.config.to_dict()
            )

            fname = os.path.join(FLAGS.save_dir, 'step_{}_params.pkl'.format(i))
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)

###################################################################################################
#
# Creates wandb plots
#
###################################################################################################
class DebugPlotGenerator:
    def __init__(self, env_name, gc_dataset):
        if 'pointumaze' in env_name:
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (0, 0)
        # elif 'maze' in env_name:
        #     viz_env, viz_dataset = d4rl_pm.get_gcenv_and_dataset(env_name)
        #     init_state = np.copy(viz_dataset['observations'][0])
        #     init_state[:2] = (3, 4)
        #     viz_library = d4rl_pm
        #     self.viz_things = (viz_env, viz_dataset, viz_library, init_state)
        else:
            raise NotImplementedError('Visualization not implemented for this environment')

        intent_set_indx = np.array([184588, 62200, 162996, 110214, 4086, 191369, 92549, 12946, 192021])
        self.intent_set_batch = gc_dataset.sample(9, indx=intent_set_indx)
        self.example_trajectory = gc_dataset.sample(50, indx=np.arange(1000, 1050))

    def generate_debug_plots(self, agent):
        example_trajectory = self.example_trajectory
        intents = self.intent_set_batch['observations']
        (viz_env, viz_dataset, viz_library, init_state) = self.viz_things

        visualizations = {}
        traj_metrics = get_traj_v(agent, example_trajectory)
        value_viz = viz_utils.make_visual_no_image(traj_metrics, 
            [
            partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()
                ]
        )
        visualizations['value_traj_viz'] = wandb.Image(value_viz)

        if 'maze' in FLAGS.env_name:
            print('Visualizing intent policies and values')
            # Policy visualization
            methods = [
                partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_policies'] = wandb.Image(image)

            # Value visualization
            methods = [
                partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_values'] = wandb.Image(image)

            for idx in range(3):
                methods = [
                    partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx])),
                    partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                ]
                image = viz_library.make_visual(viz_env, viz_dataset, methods)
                visualizations[f'intent{idx}'] = wandb.Image(image)

            image_zz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_zz, agent),
            )
            image_gz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_gz, agent, init_state),
            )
            visualizations['v_zz'] = wandb.Image(image_zz)
            visualizations['v_gz'] = wandb.Image(image_gz)
        return visualizations

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
    z = batch['desired_goals']

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
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        return agent.value(s[None], g[None], g[None]).mean()
    observations = trajectory['observations']
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