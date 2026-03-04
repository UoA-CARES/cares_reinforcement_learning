"""
SAC-AE (Soft Actor-Critic with AutoEncoder)
--------------------------------------------

Original Paper: https://arxiv.org/abs/1910.01741

Original Code: https://github.com/denisyarats/pytorch_sac_ae/tree/master

SAC-AE extends Soft Actor-Critic (SAC) to learn directly
from high-dimensional image observations by jointly
learning a latent state representation.

Core Problem:
- Standard SAC assumes low-dimensional state inputs.
- Learning directly from raw pixels is unstable and
  sample-inefficient.
- Representation learning must be integrated with
  value learning.

Core Idea:
- Add a convolutional encoder to map images → latent vector z.
- Train a decoder to reconstruct the input (autoencoder loss).
- Use the shared encoder for actor and critic updates.

Architecture:
    Image → Encoder → latent z
    z → Actor π(a|z)
    z → Twin Critics Q1(z,a), Q2(z,a)
    z → Decoder → reconstructed image

Training Components:

1) RL Loss (standard SAC):
    - Critic TD loss
    - Actor entropy-regularized objective
    - Temperature tuning

2) Reconstruction Loss:
    L_rec = || x - x̂ ||²

Encoder Update:
- Critic gradients update encoder.
- Reconstruction loss also updates encoder.
- Actor does NOT update encoder (to stabilize learning).

Key Behaviour:
- Encoder learns compact state representation.
- Reconstruction stabilizes latent features.
- Critic drives representation toward task-relevant signals.

Advantages:
- Enables SAC to operate from raw pixels.
- Improves sample efficiency vs naïve pixel SAC.
- Simple integration without model-based rollouts.

SAC-AE = SAC + shared convolutional encoder +
         auxiliary reconstruction objective.
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.encoders.losses import AELoss
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.SACAE import Actor, Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    SARLObservation,
    SARLObservationTensors,
)
from cares_reinforcement_learning.algorithm.configurations import SACAEConfig


class SACAE(SARLAlgorithm[np.ndarray]):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        decoder_network: Decoder,
        config: SACAEConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.target_critic_net.eval()  # never in training mode - helps with batch/drop out layers

        # tie the encoder weights
        self.actor_net.encoder.copy_conv_weights_from(self.critic_net.encoder)

        self.encoder_tau = config.encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_update_freq = config.decoder_update_freq
        self.decoder_latent_lambda = config.autoencoder_config.latent_lambda

        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_weight_normalisation = config.per_weight_normalisation
        self.per_alpha = config.per_alpha
        self.min_priority = config.min_priority

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        self.target_entropy = -self.actor_net.num_actions

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

        self.ae_loss_function = AELoss(
            latent_lambda=config.autoencoder_config.latent_lambda
        )

        self.encoder_net_optimiser = torch.optim.Adam(
            self.critic_net.encoder.parameters(),
            **config.autoencoder_config.encoder_optim_kwargs,
        )
        self.decoder_net_optimiser = torch.optim.Adam(
            self.decoder_net.parameters(),
            **config.autoencoder_config.decoder_optim_kwargs,
        )

        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr, **config.alpha_lr_params
        )

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()

        with torch.no_grad():
            observation_tensors = memory_sampler.observation_to_tensors(
                [observation], self.device
            )

            if evaluation:
                _, _, action = self.actor_net(observation_tensors)
            else:
                action, _, _ = self.actor_net(observation_tensors)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()

        return ActionSample(action=action, source="policy")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def _update_critic(
        self,
        states: SARLObservationTensors,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: SARLObservationTensors,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            with fnc.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = (
                rewards * self.reward_scale + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = (q_values_one.detach() - q_target).abs()
        td_error_two = (q_values_two.detach() - q_target).abs()

        critic_loss_one = F.mse_loss(q_values_one, q_target, reduction="none")
        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = F.mse_loss(q_values_two, q_target, reduction="none")
        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        with torch.no_grad():
            # --- Twin critic disagreement (stability/uncertainty) ---
            # If this grows over training, critics are diverging / becoming inconsistent.
            info["q1_mean"] = q_values_one.mean().item()
            info["q2_mean"] = q_values_two.mean().item()
            info["q_twin_gap_abs_mean"] = (
                (q_values_one - q_values_two).abs().mean().item()
            )

            # --- Target critics disagreement (target stability) ---
            # Large/unstable gap here often means target critics are drifting or policy is visiting OOD actions.
            info["target_q1_mean"] = target_q_values_one.mean().item()
            info["target_q2_mean"] = target_q_values_two.mean().item()
            info["target_q_twin_gap_abs_mean"] = (
                (target_q_values_one - target_q_values_two).abs().mean().item()
            )

            # --- Soft target decomposition (SAC-specific) ---
            # min_target_q_mean: the conservative bootstrap value from twin critics (pre-entropy)
            # entropy_term_mean: magnitude of entropy regularization in the target (alpha * log_pi is usually negative)
            # soft_target_value_mean: the exact term used inside the Bellman target before reward/discount
            min_target_q = torch.minimum(target_q_values_one, target_q_values_two)

            # alpha_log_pi is typically negative; entropy_bonus is typically positive
            alpha_log_pi = self.alpha * next_log_pi
            # this is what gets ADDED to minQ in the target
            entropy_bonus = -self.alpha * next_log_pi

            soft_target_value = min_target_q + entropy_bonus  # == minQ - alpha*log_pi

            info["target_min_q_mean"] = min_target_q.mean().item()
            info["alpha_log_pi_mean"] = alpha_log_pi.mean().item()
            info["entropy_bonus_mean"] = entropy_bonus.mean().item()
            info["soft_target_value_mean"] = soft_target_value.mean().item()

            # --- Bellman target scale (reward scaling / discount sanity) ---
            # If q_target drifts upward without reward improvement, suspect reward_scale, gamma, or instability.
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std().item()

            # --- TD error diagnostics (Bellman fit quality) ---
            # td_abs_mean down over time is healthy; persistent growth/spikes often indicate critic instability.
            td1 = q_values_one - q_target  # signed
            td2 = q_values_two - q_target  # signed

            info["td1_mean"] = td1.mean().item()
            info["td1_std"] = td1.std().item()
            info["td1_abs_mean"] = td1.abs().mean().item()

            info["td2_mean"] = td2.mean().item()
            info["td2_std"] = td2.std().item()
            info["td2_abs_mean"] = td2.abs().mean().item()

            # --- Losses (optimization progress; less diagnostic than TD/twin gaps) ---
            info["critic_loss_one"] = critic_loss_one.item()
            info["critic_loss_two"] = critic_loss_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info, priorities

    def _update_actor_alpha(self, states: SARLObservationTensors) -> dict[str, Any]:
        info: dict[str, Any] = {}

        pi, log_pi, _ = self.actor_net(states, detach_encoder=True)

        with fnc.evaluating(self.critic_net):
            qf_pi_one, qf_pi_two = self.critic_net(states, pi, detach_encoder=True)

        min_qf_pi = torch.minimum(qf_pi_one, qf_pi_two)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # ---------------------------------------------------------
        # Stochastic Policy Gradient Strength (∇a [α log π(a|s) − Q(s,a)])
        # ---------------------------------------------------------
        # Measures how steep the entropy-regularized critic objective is
        # w.r.t. the sampled policy actions.
        #
        # ~0 early  -> critic surface and entropy term nearly flat;
        #              actor receives weak learning signal.
        #
        # Very large -> critic or entropy term is very sharp around policy
        #               actions; can lead to unstable or overly aggressive
        #               actor updates.
        dq_da = torch.autograd.grad(
            outputs=actor_loss,
            inputs=pi,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        with torch.no_grad():
            info["dq_da_abs_mean"] = dq_da.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da.norm(dim=1).quantile(0.95).item()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # Update the temperature (alpha)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        with torch.no_grad():
            # --- Policy entropy diagnostics (exploration health) ---
            # log_pi more negative -> higher entropy (more stochastic). Less negative -> lower entropy (more deterministic).
            info["log_pi_mean"] = log_pi.mean().item()
            info["log_pi_std"] = log_pi.std().item()

            # --- Action magnitude/saturation (tanh policies) ---
            # High saturation fraction can indicate the policy is slamming bounds; may reduce effective gradients.
            info["pi_action_abs_mean"] = pi.abs().mean().item()
            info["pi_action_std"] = pi.std().item()
            info["pi_action_saturation_frac"] = (pi.abs() > 0.95).float().mean().item()

            # --- On-policy critic signal ---
            # min_qf_pi_mean should generally increase as the policy improves (higher value actions under the policy).
            info["min_qf_pi_mean"] = min_qf_pi.mean().item()

            # --- Twin critics disagreement at policy actions (more relevant than replay actions) ---
            # Large gap here means critics disagree on what the current policy is doing (can destabilize actor updates).
            info["qf_pi_gap_abs_mean"] = (qf_pi_one - qf_pi_two).abs().mean().item()

            # --- Entropy gap (alpha tuning health) ---
            # entropy_gap ~ 0 means entropy matches target.
            # > 0: entropy too low -> alpha should increase; < 0: entropy too high -> alpha should decrease.
            entropy_gap = -(log_pi + self.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            # --- Losses and temperature ---
            info["actor_loss"] = actor_loss.item()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()
            info["log_alpha"] = self.log_alpha.item()

        return info

    def _update_autoencoder(self, states: torch.Tensor) -> dict[str, Any]:
        latent_samples = self.critic_net.encoder(states)
        reconstructed_data = self.decoder_net(latent_samples)

        ae_loss = self.ae_loss_function.calculate_loss(
            data=states,
            reconstructed_data=reconstructed_data,
            latent_sample=latent_samples,
        )

        self.encoder_net_optimiser.zero_grad()
        self.decoder_net_optimiser.zero_grad()
        ae_loss.backward()
        self.encoder_net_optimiser.step()
        self.decoder_net_optimiser.step()

        info = {
            "ae_loss": ae_loss.item(),
        }
        return info

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        self.learn_counter += 1

        # Sample and convert to tensors using multimodal sampling
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            _,  # extras ignored
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        assert observation_tensor.image_state_tensor is not None

        info: dict[str, Any] = {}

        # Update the Critic
        critic_info, priorities = self._update_critic(
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
        )
        info |= critic_info

        # Update the Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_info = self._update_actor_alpha(observation_tensor)
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            # Update the target networks - Soft Update
            self.soft_update_params(
                self.critic_net.critic.Q1, self.target_critic_net.critic.Q1, self.tau  # type: ignore
            )
            self.soft_update_params(
                self.critic_net.critic.Q2, self.target_critic_net.critic.Q2, self.tau  # type: ignore
            )
            self.soft_update_params(
                self.critic_net.encoder,
                self.target_critic_net.encoder,
                self.encoder_tau,
            )

        if self.learn_counter % self.decoder_update_freq == 0:
            ae_info = self._update_autoencoder(observation_tensor.image_state_tensor)
            info |= ae_info

        # Update the Priorities
        if self.use_per_buffer:
            memory_buffer.update_priorities(indices, priorities)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "decoder": self.decoder_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "encoder_optimizer": self.encoder_net_optimiser.state_dict(),
            "decoder_optimizer": self.decoder_net_optimiser.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().item(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "learn_counter": self.learn_counter,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.target_critic_net.load_state_dict(checkpoint["target_critic"])

        self.decoder_net.load_state_dict(checkpoint["decoder"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        self.encoder_net_optimiser.load_state_dict(checkpoint["encoder_optimizer"])
        self.decoder_net_optimiser.load_state_dict(checkpoint["decoder_optimizer"])

        self.log_alpha.data = torch.tensor(checkpoint["log_alpha"]).to(self.device)
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

        self.learn_counter = checkpoint.get("learn_counter", 0)
        logging.info("models, optimisers, and training state have been loaded...")
