from ott.neural.solvers.neuraldual import W2NeuralDual
import jax
import jax.numpy as jnp
import functools
import flax.linen as nn
import flax

from typing import Dict, Sequence
from networks.common import TrainState

class W2NeuralDualCustom(W2NeuralDual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.back_and_forth = True
        self.step = 0
    
    def update(self, batch_agent, batch_expert):

        update_forward = not self.back_and_forth or self.step % 2 == 0
        train_batch = {}

        if update_forward:
            train_batch["source"] = batch_agent
            train_batch["target"] = batch_expert
            (self.state_f, self.state_g, loss, loss_f, loss_g, w_dist) = self.train_step_parallel(
                self.state_f,
                self.state_g,
                train_batch,
            )
        else:
            train_batch["target"] = batch_agent
            train_batch["source"] = batch_expert
            (self.state_g, self.state_f, loss, loss_f, loss_g, w_dist) = self.train_step_parallel(
                self.state_g,
                self.state_f,
                train_batch,
            )

        self.step += 1
       
        return self, loss, w_dist
    
    
@functools.partial(jax.jit, static_argnames=('not_solver'))
def _update_agent(not_solver, network, batch_agent, batch_expert):
    def loss_fn(params):
        loss, not_loss, expert_enc_loss = network(not_solver, batch_agent, batch_expert, params=params, method='loss_fn') # First loss
        return loss, {'loss': loss,
                      'not_distance': not_loss,
                      'expert_encoder_loss': expert_enc_loss}
    
    new_agent, info = network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
    return new_agent, info

class NotNetwork(nn.Module):
    encoders: Dict[str, nn.Module]
    expert_loss_coef: float = 0.5

    def compute_ot_distance(self, learned_potentials, encoded_s_agent, encoded_s_expert):
        
        agent_s = jnp.concatenate([encoded_s_agent, encoded_s_agent], axis=-1)
        expert_s = jnp.concatenate([encoded_s_expert, encoded_s_expert], axis=-1)

        return learned_potentials.distance(agent_s, expert_s)
    
    def compute_expert_encoder_loss(self, learned_potentials, encoded_s_expert, encoded_s_expert_next): # maximize target potential g w.r.t encoder
        s_pair = jnp.concatenate([encoded_s_expert, encoded_s_expert_next], axis=-1)
        g_value = jax.vmap(learned_potentials.g)(s_pair)
        return -2 * jnp.mean(g_value) + jnp.mean(jnp.sum(s_pair ** 2, axis=-1))
        
    def loss_fn(self, not_solver, batch_agent, batch_expert):
        learned_potentials = not_solver.to_dual_potentials()

        encoded_s_agent = self.encode_agent(batch_agent.observations)
        encoded_s_expert = self.encode_expert(batch_expert.observations)
        encoded_s_expert_next = self.encode_expert(batch_expert.next_observations)

        not_loss = self.compute_ot_distance(learned_potentials, encoded_s_agent, encoded_s_expert)
        expert_enc_loss = self.compute_expert_encoder_loss(learned_potentials, encoded_s_expert, encoded_s_expert_next)

        loss = not_loss - expert_enc_loss * self.expert_loss_coef

        return loss, not_loss, expert_enc_loss
        
    def encode_expert(self, expert_states):
        return self.encoders['expert_encoder'](expert_states)
    
    def encode_agent(self, agent_states):
        return self.encoders['agent_encoder'](agent_states)
    
    def __call__(self, expert_obs, agent_obs):
        rets = {
            "encoded_expert": self.encode_expert(expert_obs),
            "encoded_agent": self.encode_agent(agent_obs)
        }
        return rets

class JointAgent(flax.struct.PyTreeNode):
    network: TrainState
    
    def optimize_not(self, not_solver, batch_agent, batch_expert):
        SE = self.network(batch_expert.observations, method='encode_expert')
        SEnext = self.network(batch_expert.next_observations, method='encode_expert')
        SA = self.network(batch_agent.observations, method='encode_agent')
        SAnext = self.network(batch_agent.next_observations, method='encode_agent')
        
        # exp_reps = jnp.concatenate([SE, SEnext], axis=-1)
        exp_reps = jnp.concatenate([SE, SE], axis=-1)
        # ag_reps = jnp.concatenate([SA, SAnext], axis=-1)
        ag_reps = jnp.concatenate([SA, SA], axis=-1)

        # exp_reps = jnp.concatenate([exp_reps, exp_reps_2])
        # ag_reps = jnp.concatenate([ag_reps, ag_reps_2])

        updated_not_solver, loss, w_dist = not_solver.update(ag_reps, exp_reps)
        return updated_not_solver, loss, w_dist
    
    def optimize_encoders(self, not_solver, batch_agent, batch_expert):
        new_agent, info = _update_agent(
            not_solver, self.network, batch_agent, batch_expert
        )
        
        return self.replace(network=new_agent), info