from ott.neural.methods.neuraldual import W2NeuralDual
from ott.neural.methods.expectile_neural_dual import ExpectileNeuralDual
import jax.numpy as jnp
import jax
import numpy as np
import flax.linen as nn
from typing import Dict
from networks.common import TrainState
import jax.tree_util as jtu
from ott.problems.linear import potentials as dual_potentials
from ott.neural.networks.layers.conjugate import FenchelConjugateLBFGS


DEFAULT_CONJUGATE_SOLVER = FenchelConjugateLBFGS(
    gtol=1e-5,
    max_iter=40,
    max_linesearch_iter=20,
    linesearch_type="backtracking",
)

class Encoders(nn.Module):
    encoders: Dict[str, nn.Module]
    
    def encode_expert(self, expert_states):
        return self.encoders['expert_encoder'](expert_states)
    
    def encode_agent(self, agent_states):
        return self.encoders['agent_encoder'](agent_states)
    
    def __call__(self, agent_obs, expert_obs):
        rets = {
            "encoded_agent": self.encode_agent(agent_obs),
            "encoded_expert": self.encode_expert(expert_obs)
        }
        return rets
    
@jtu.register_pytree_node_class
class PotentialsCustom:

    def __init__(
      self, state_f, state_g, conjugate_solver
    ):
        
        self.state_f = state_f
        self.state_g = state_g
        self.conjugate_solver = conjugate_solver

    def tree_flatten(self):
        return [self.state_f, self.state_g], {
            "conjugate_solver": self.conjugate_solver
        }

    @classmethod
    def tree_unflatten(  # noqa: D102
        cls, aux_data, children
    ):
        return cls(*children, **aux_data)

    def get_fg(self):
    
        f_value = self.state_f.potential_value_fn(self.state_f.params)
        g_value_prediction = self.state_g.potential_value_fn(
            self.state_g.params, f_value
        )

        def g_value_finetuned(y: jnp.ndarray) -> jnp.ndarray:
            x_hat = jax.grad(g_value_prediction)(y)
            grad_g_y = jax.lax.stop_gradient(
                self.conjugate_solver.solve(f_value, y, x_init=x_hat).grad
            )
            return -f_value(grad_g_y) + jnp.dot(grad_g_y, y)
        
        return f_value, g_value_finetuned
    
    def distance(self, x, y):
        f, g = self.get_fg()
        C = jnp.mean(jnp.sum(x ** 2, axis=-1)) + \
            jnp.mean(jnp.sum(y ** 2, axis=-1))
        return C - 2 * (jax.vmap(f)(x) + jax.vmap(g)(y)).mean()
    


@jtu.register_pytree_node_class
class ENOTPotentialsCustom:

    def __init__(
      self, state_f, state_g, cost_fn, is_bidirectional 
    ):
        
        self.state_f = state_f
        self.state_g = state_g
        self.cost_fn = cost_fn
        self.is_bidirectional = is_bidirectional

    def tree_flatten(self):
        return [self.state_f, self.state_g], {
            "cost_fn": self.cost_fn,
            "is_bidirectional": self.is_bidirectional
        }

    @classmethod
    def tree_unflatten(  # noqa: D102
        cls, aux_data, children
    ):
        return cls(*children, **aux_data)

    def get_fg(self):
    
        grad_f = self.state_f.potential_gradient_fn(self.state_f.params)
        g_value = self.state_g.potential_value_fn(self.state_g.params, None)
        
        if self.is_bidirectional:
            transport = lambda x: self.cost_fn.twist_operator(x, grad_f(x), False)
        else:
            transport = lambda x: grad_f(x)

        def g_cost_conjugate(x: jnp.ndarray) -> jnp.ndarray:
            y_hat = jax.lax.stop_gradient(transport(x))
            return -g_value(y_hat) + self.cost_fn(x, y_hat)
            
        return g_cost_conjugate, g_value
    
    def distance(self, x, y):
        f, g = self.get_fg()
        return (jax.vmap(f)(x) + jax.vmap(g)(y)).mean()
    
    def transport(self, vec, forward: bool):
        twist_op = jax.vmap(self.cost_fn.twist_operator, in_axes=[0, 0, None])
        grad_f = jax.vmap(self.state_f.potential_gradient_fn(self.state_f.params))
        grad_g = jax.vmap(self.state_g.potential_gradient_fn(self.state_g.params))
        if forward:
            return twist_op(vec, grad_f(vec), False)
        return twist_op(vec, grad_g(vec), True)



class W2NeuralDualCustom(W2NeuralDual):
    def __init__(self, *args, **kwargs):
        kwargs["conjugate_solver"] = DEFAULT_CONJUGATE_SOLVER
        super().__init__(*args, **kwargs)
        self.back_and_forth = True
        self.step = 0
    
    def update(self, batch_source, batch_target):

        update_forward = not self.back_and_forth or self.step % 2 == 0
        train_batch = {}

        if update_forward:
            train_batch["source"] = batch_source
            train_batch["target"] = batch_target
            (self.state_f, self.state_g, loss, loss_f, loss_g, w_dist) = self.train_step_parallel(
                self.state_f,
                self.state_g,
                train_batch,
            )
        else:
            train_batch["target"] = batch_source
            train_batch["source"] = batch_target
            (self.state_g, self.state_f, loss, loss_f, loss_g, w_dist) = self.train_step_parallel(
                self.state_g,
                self.state_f,
                train_batch,
            )

        self.step += 1
       
        return self, loss, w_dist
    
    def to_dual_potentials(
      self, finetune_g: bool = True
    ) -> PotentialsCustom:
        
       return PotentialsCustom(self.state_f, self.state_g, self.conjugate_solver)
    

class ENOTCustom(ExpectileNeuralDual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
    
    def update(self, batch_source, batch_target):

        update_forward = self.step % 2 == 0
        train_batch = {}

        if update_forward:
            train_batch["source"] = batch_source
            train_batch["target"] = batch_target
            (self.state_f, self.state_g, loss, loss_f, loss_g, w_dist) = self.train_step(
                self.state_f,
                self.state_g,
                train_batch,
            )
        else:
            train_batch["target"] = batch_source
            train_batch["source"] = batch_target
            (self.state_g, self.state_f, loss, loss_f, loss_g, w_dist) = self.train_step(
                self.state_g,
                self.state_f,
                train_batch,
            )

        self.step += 1
       
        return self, loss, w_dist
    
    def to_dual_potentials(
      self, finetune_g: bool = True
    ) -> ENOTPotentialsCustom:
        
       return ENOTPotentialsCustom(self.state_f, self.state_g, self.cost_fn, self.is_bidirectional)
    
class NotAgent:
    def __init__(
        self,
        embed_dim: int,
        neural_f: nn.Module,
        neural_g: nn.Module,
        optimizer_f,
        optimizer_g,
        expert_loss_coef: float = 1.0,
        num_train_iters: int = 10_000):

        self.expert_loss_coef = expert_loss_coef
        self.neural_dual_elements = W2NeuralDualCustom(
            dim_data=embed_dim, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            num_train_iters=num_train_iters # 20_000
        )

        self.neural_dual_pairs = W2NeuralDualCustom(
            dim_data=embed_dim * 2, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            num_train_iters=num_train_iters # 20_000
        )

    @staticmethod
    def ot_distance_elements(learned_potentials, x, y):
        return learned_potentials.distance(x, y)
    
    @staticmethod
    def ot_distance_pairs(learned_potentials, x, y, x_next, y_next, sa_next, sa):
        x_pair = jnp.concatenate([x, x_next], axis=-1)
        y_pair = jnp.concatenate([y, y_next], axis=-1)
        sa_next = jnp.concatenate([sa, sa_next], axis=-1)
        
        return learned_potentials.distance(jnp.concatenate([sa_next, x_pair]), jnp.concatenate([y_pair, y_pair]))
      
    #@staticmethod
    # def encoders_loss(potentials_elem, potentials_pairs, sa, se, sn, sa_next, se_next, sn_next, expert_loss_coef):
        
    #     loss_elem = JointAgent.ot_distance_elements(
    #         potentials_elem, 
    #         jnp.concatenate([sa, sn], axis=0), # sa
    #         jnp.concatenate([se, se], axis=0),
    #     )
    #     #[sa, sn], [se, se], [sa_next, sn_next], [se_next, se_next]
    #     loss_pairs = JointAgent.ot_distance_pairs(potentials_pairs, sn, se, sn_next, se_next, sa_next, sa)
    #     # expert_enc_loss = JointAgent.compute_expert_encoder_loss(potentials_pairs, y, y_next)
    
    #     loss = loss_elem - loss_pairs * expert_loss_coef

    #     return loss, loss_elem, loss_pairs

    # PRETRAIN STAGE
    # def optimize_not(self, batch_agent, batch_expert, random_data):
    #     #sa, se, sn, sa_pairs, se_pairs, sn_pairs = compute_embeds(self.encoders_state, batch_agent, batch_expert, random_data)
    #     _, loss_elem, w_dist_elem = self.neural_dual_elements.update(np.concatenate([sa, sn]), np.concatenate([se, se]))
    #     _, loss_pairs, w_dist_pairs = self.neural_dual_pairs.update(np.concatenate([sa_pairs, sn_pairs]), np.concatenate([se_pairs, se_pairs]))
        
    #     return loss_elem, loss_pairs, w_dist_elem, w_dist_pairs
    
    def optimize_encoders(self, batch_agent, batch_expert, random_data):
        @jax.jit
        def update_step(potentials_elem, potentials_pairs, encoders, batch_agent, batch_expert, random_data):
            def loss_fn(params):
                se = batch_expert.observations[:, :2]
                se_next = batch_expert.next_observations[:, :2]
                sn = random_data.observations[:, :2]
                sn_next = random_data.next_observations[:, :2]
                sa = batch_agent.observations[:, :2]
                sa_next = batch_agent.next_observations[:, :2]
                
                # se = encoders(batch_expert.observations, params=params, method='encode_expert')
                # se_next = encoders(batch_expert.next_observations, params=params, method='encode_expert')
                # sa = encoders(batch_agent.observations, params=params, method='encode_agent')
                # sa_next = encoders(batch_agent.next_observations, params=params, method='encode_agent')
                # sn = encoders(random_data.observations, method='encode_expert')
                # sn_next = encoders(random_data.next_observations, method='encode_expert')

                loss, not_loss, expert_enc_loss = NotAgent.encoders_loss(potentials_elem, potentials_pairs, sa, se, sn, sa_next, se_next, sn_next, self.expert_loss_coef)

                return loss, {'loss': loss,
                            'not_distance': not_loss,
                            'expert_encoder_loss': expert_enc_loss}
            
            new_encoders, info = encoders.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

            return new_encoders, info
        
        potentials_elem = self.neural_dual_elements.to_dual_potentials()
        potentials_pairs = self.neural_dual_pairs.to_dual_potentials()

        self.encoders_state, info = update_step(
            potentials_elem, potentials_pairs, self.encoders_state, batch_agent, batch_expert, random_data
        )
        
        return info
    