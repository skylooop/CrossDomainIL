import flax.linen as nn

import jax
import jax.numpy as jnp

from flax.struct import PyTreeNode
from main_agents.disc import Discriminator
from jaxrl_m.common import TrainState
import optax


class ReverseLayer(nn.Module):
    def __call__(self, x):
        return x

class EncoderModel(nn.Module):
    encoders: dict[str, nn.Module]
    reverse_layer: nn.Module

    def encode_source(self, x):
        return self.encoders['source'](x)
    
    def encode_target(self, x):
        return self.encoders['target'](x)

    def __call__(self, x):
        return {'source_enc': self.reverse_layer(self.encode_source(x)),
                'target_enc': self.reverse_layer(self.encode_target(x))}


class DIDA(PyTreeNode):
    noisy_disc: Discriminator
    policy_disc: Discriminator
    encoder: TrainState
    
    @classmethod
    def create(
        cls,
        observation_dim: int,
        noisy_discr: Discriminator,
        policy_discr: Discriminator,
        encoders: dict[str, nn.Module]
    ):
        rng = jax.random.PRNGKey(42)
        encoder_def = EncoderModel(encoders=encoders, reverse_layer=ReverseLayer())
        params = encoder_def.init(rng, jnp.ones_like(observation_dim, ))['params']
        encoder_model = TrainState.create(
            encoder_def, params=params, tx=optax.adam(3e-4)
        )
        return cls(noisy_disc=noisy_discr, policy_disc=policy_discr, encoder=encoder_model)
    
    @jax.jit
    def update_policy_discr(self, noisy_expert_batch, mix_data_batch):
        def loss_fn_policy_discr(params):
            encoded_mix = self.encoder(mix_data_batch.observations, method='encode_source')
            encoded_noisy_expert = self.encoder(noisy_expert_batch.observations, method='encode_source')
            
            discr_mix = self.policy_disc.state(encoded_mix, params=params)
            discr_noisy_expert = self.policy_disc.state(encoded_noisy_expert, params=params)
            noisy_expert_loss = optax.losses.sigmoid_binary_cross_entropy(discr_noisy_expert, jnp.ones_like(discr_noisy_expert))
            mix_loss = optax.losses.sigmoid_binary_cross_entropy(discr_mix, jnp.zeros_like(discr_mix))
            return mix_loss.mean() + noisy_expert_loss.mean()
        
        new_policy_disc = self.policy_disc.replace(state=self.policy_disc.state.apply_loss_fn(loss_fn=loss_fn_policy_discr, has_aux=False))
        return self.replace(policy_disc=new_policy_disc)

    @jax.jit
    def update_noise_discr(self, imitator_batch, noisy_expert_batch, anchor_batch):
        # sigma_imitator = jnp.concatenate([imitator_batch.observations, imitator_batch.next_observations], axis=-1)
        # sigma_anchor = jnp.concatenate([anchor_batch.observations, anchor_batch.next_observations], axis=-1)
        # sigma_noisy_expert = jnp.concatenate([noisy_expert_batch.observations, noisy_expert_batch.next_observations], axis=-1)
        
        def loss_fn_noise_discr(params):
            # encoded_imitator = self.encoder(sigma_imitator, method='encode_source')
            # encoded_noisy_expert = self.encoder(sigma_noisy_expert, method='encode_source')
            # encoded_anchor = self.encoder(sigma_anchor, method='encode_source')
            
            encoded_imitator = self.encoder(imitator_batch.observations, method='encode_source')
            encoded_noisy_expert = self.encoder(noisy_expert_batch.observations, method='encode_source')
            encoded_anchor = self.encoder(anchor_batch.observations, method='encode_source')
            
            discr_imitator = self.noisy_disc.state(encoded_imitator, params=params)
            discr_noisy_expert = self.noisy_disc.state(encoded_noisy_expert, params=params)
            discr_noisy_anchor = self.noisy_disc.state(encoded_anchor, params=params)
            imitator_loss = optax.losses.sigmoid_binary_cross_entropy(discr_imitator, jnp.zeros_like(discr_imitator))
            noisy_expert_loss = optax.losses.sigmoid_binary_cross_entropy(discr_noisy_expert, jnp.ones_like(discr_noisy_expert))
            anchor_loss = optax.losses.sigmoid_binary_cross_entropy(discr_noisy_anchor, jnp.ones_like(discr_noisy_anchor))

            return imitator_loss.mean() + noisy_expert_loss.mean() + anchor_loss.mean() #* lambda
            
        new_noise_discr = self.noisy_disc.replace(state=self.noisy_disc.state.apply_loss_fn(loss_fn=loss_fn_noise_discr, has_aux=False))
        return self.replace(noisy_disc=new_noise_discr)
    