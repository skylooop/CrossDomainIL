from ott.neural.solvers.neuraldual import W2NeuralDual
import jax.numpy as jnp
import jax
import numpy as np
import flax.linen as nn
from typing import Dict
import optax
from networks.common import TrainState

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
    
class JointAgent:

    def __init__(
        self,
        encoder_agent: nn.Module,
        encoder_expert: nn.Module,
        agent_dim: int,
        expert_dim: int,
        embed_dim: int,
        neural_f: nn.Module,
        neural_g: nn.Module,
        optimizer_f,
        optimizer_g,
        expert_loss_coef: float = 0.5,
        learning_rate: float = 1e-4,
        rng = jax.random.PRNGKey(42)) -> None:

        self.expert_loss_coef = expert_loss_coef

        encoders = Encoders({
            'agent_encoder': encoder_agent,
            'expert_encoder': encoder_expert
        })

        self.encoders_state = TrainState.create(
            model_def=encoders,
            params=encoders.init(rng, jnp.ones(agent_dim), jnp.ones(expert_dim))['params'],
            tx=optax.adam(learning_rate=learning_rate)
        )

        self.neural_dual_elements = W2NeuralDualCustom(
            dim_data=embed_dim, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            num_train_iters=5_000 # 20_000
        )

        self.neural_dual_pairs = W2NeuralDualCustom(
            dim_data=embed_dim * 2, 
            neural_f=neural_f,
            neural_g=neural_g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            num_train_iters=5_000 # 20_000
        )

    @staticmethod
    def ot_distance_elements(learned_potentials, x, y):
        return learned_potentials.distance(x, y)
    
    @staticmethod
    def ot_distance_pairs(learned_potentials, x, y, x_next, y_next):
        x_pair = jnp.concatenate([x, x_next], axis=-1)
        y_pair = jnp.concatenate([y, y_next], axis=-1)

        return learned_potentials.distance(x_pair, y_pair)
    
    @staticmethod
    def compute_expert_encoder_loss(learned_potentials, y, y_next): # maximize target potential g w.r.t encoder
        y_pair = jnp.concatenate([y, y_next], axis=-1)
        g_value = jax.vmap(learned_potentials.g)(y_pair)
        return jnp.mean((y_pair ** 2).sum(-1)) - 2 * jnp.mean(g_value)
    
    @staticmethod
    def encoders_loss(potentials_elem, potentials_pairs, sa, se, sn, sa_next, se_next, sn_next, expert_loss_coef):
        
        loss_elem = JointAgent.ot_distance_elements(
            potentials_elem, 
            jnp.concatenate([sa, sa], axis=0), 
            jnp.concatenate([se, sn], axis=0), 
        )
        loss_pairs = JointAgent.ot_distance_pairs(potentials_pairs, sn, se, sn_next, se_next)
        # expert_enc_loss = JointAgent.compute_expert_encoder_loss(potentials_pairs, y, y_next)
        
        loss = loss_elem - loss_pairs * expert_loss_coef

        return loss, loss_elem, loss_pairs
        
    def optimize_not(self, batch_agent, batch_expert, random_data):
        
        @jax.jit
        def compute_embeds(encoders, batch_agent, batch_expert, random_data):
            se = encoders(batch_expert.observations, method='encode_expert')
            se_next = encoders(batch_expert.next_observations, method='encode_expert')
            sn = encoders(random_data.observations, method='encode_expert')
            sn_next = encoders(random_data.next_observations, method='encode_expert')
            sa = encoders(batch_agent.observations, method='encode_agent')
            sa_next = encoders(batch_agent.next_observations, method='encode_agent')

            sa_pairs = jnp.concatenate([sa, sa_next], axis=-1)
            se_pairs = jnp.concatenate([se, se_next], axis=-1)
            sn_pairs = jnp.concatenate([sn, sn_next], axis=-1)

            return  sa, se, sn, sa_pairs, se_pairs, sn_pairs
        
        sa, se, sn, sa_pairs, se_pairs, sn_pairs = compute_embeds(self.encoders_state, batch_agent, batch_expert, random_data)

        _, loss_elem, w_dist_elem = self.neural_dual_elements.update(np.concatenate([sa, sa]), np.concatenate([se, sn]))
        _, loss_pairs, w_dist_pairs = self.neural_dual_pairs.update(np.concatenate([sa_pairs, sn_pairs]), np.concatenate([se_pairs, se_pairs]))
        
        return loss_elem, loss_pairs, w_dist_elem, w_dist_pairs
    
    def optimize_encoders(self, batch_agent, batch_expert, random_data):

        @jax.jit
        def update_step(potentials_elem, potentials_pairs, encoders, batch_agent, batch_expert, random_data):

            def loss_fn(params):

                se = encoders(batch_expert.observations, params=params, method='encode_expert')
                se_next = encoders(batch_expert.next_observations, params=params, method='encode_expert')
                sa = encoders(batch_agent.observations, params=params, method='encode_agent')
                sa_next = encoders(batch_agent.next_observations, params=params, method='encode_agent')
                sn = encoders(random_data.observations, method='encode_expert')
                sn_next = encoders(random_data.next_observations, method='encode_expert')

                loss, not_loss, expert_enc_loss = JointAgent.encoders_loss(potentials_elem, potentials_pairs, sa, se, sn, sa_next, se_next, sn_next, self.expert_loss_coef)

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
    