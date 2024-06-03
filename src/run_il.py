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

from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, RecordVideo
import ott
from ott.geometry import costs

def SetColor(x, y):
    if(x < 6 and y < 2):
        return "green"
    elif(x >= 6 or (y >=1.5 and y < 4.5)):
        return "red"
    elif ( x<6.5 and y>4):
        return "purple"


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
        project="CrossDomain_IL",
        config=dict(cfg),
        group="expert_" + f"{cfg.imitation_env.name}_{cfg.algo.name}",
    )

    source_expert_ds, source_random_ds, combined_source_ds, target_random_ds = prepare_buffers_for_il(cfg=cfg)
    
    if cfg.optimal_transport:
        from ott.neural.methods.expectile_neural_dual import MLP as ExpectileMLP
        
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
            expectile_loss_coef = 0.5,
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
            expectile_loss_coef = 0.5,
            use_dot_product=False,
            is_bidirectional=True
        )
        joint_ot_agent = JointNOTAgent.create(
            cfg.seed,
            latent_dim=latent_dim,
            target_obs=target_random_ds.observations[0],
            source_obs=combined_source_ds.observations[0],
        )
        
        #General Pretraining
        for i in tqdm(range(2_000), desc="Pretraining NOT", position=0, leave=False):
            target_data = target_random_ds.sample(1024, goal_conditioned=True)
            source_data = combined_source_ds.sample(1024, goal_conditioned=True)
            not_agent_pairs, not_agent_elems, potential_elems, potential_pairs, encoded_source, encoded_target, not_info = update_not(joint_ot_agent, not_agent_elems, not_agent_pairs,
                                                                                                    source_data, target_data)
        max_steps = 300_005
        os.makedirs("viz_plots", exist_ok=True)
        
        for i in tqdm(range(max_steps), leave=True):
            target_data = target_random_ds.sample(1024, goal_conditioned=True)
            source_data = combined_source_ds.sample(1024, goal_conditioned=True)
            if i % 10 == 0:
                joint_ot_agent, info = joint_ot_agent.update(source_data, target_data, potential_elems, potential_pairs, update_not=True)
            else:
                joint_ot_agent, info = joint_ot_agent.update(source_data, target_data, potential_elems, potential_pairs, update_not=False)
            
            if i % 10 == 0:
                for _ in range(30):
                    target_data = target_random_ds.sample(1024, goal_conditioned=True)
                    source_data = combined_source_ds.sample(1024, goal_conditioned=True)
                    not_agent_pairs, not_agent_elems, potential_elems, potential_pairs, encoded_source, encoded_target, not_info = update_not(joint_ot_agent, not_agent_elems, not_agent_pairs, 
                                                                                                            source_data, target_data)
            
            if i % 5_000 == 1 or i == (max_steps - 1):
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
            if i % 5_010 == 0 or i == (max_steps - 1):
                colormap_target = list(map(SetColor, target_data.observations[:, 0], target_data.observations[:, 1]))
                colormap_source = list(map(SetColor, source_data.observations[:, 0], source_data.observations[:, 1]))
                #####################################################################################
                # Target domain
                #####################################################################################
                tsne = TSNE(n_components=2)
                
                encoded_target = joint_ot_agent.ema_get_phi_target(target_data.observations)
                fitted_tsne = tsne.fit_transform(encoded_target)
                fig, ax = plt.subplots()
                ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne", c=colormap_target)
                fig.savefig(f"viz_plots/target_{i}_tsne.png")
                
                #####################################################################################
                # Source domain
                #####################################################################################
                tsne = TSNE(n_components=2)
                encoded_source = joint_ot_agent.ema_get_phi_source(source_data.observations)
                fitted_tsne = tsne.fit_transform(encoded_source)

                fig, ax = plt.subplots()
                ax.scatter(fitted_tsne[:, 0], fitted_tsne[:, 1], label="tsne", c=colormap_source)
                fig.savefig(f"viz_plots/source_{i}_tsne.png")
                
                ############################
                # BOTH
                tsne = TSNE(n_components=2)
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
    
if __name__ == "__main__":
    try:
        collect_expert()
    except KeyboardInterrupt:
        wandb.finish()
        exit()