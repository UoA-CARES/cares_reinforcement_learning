"""
MR.Q (Model-based Representations for Q-learning)
----------------------------------------------------

Original Paper: https://arxiv.org/abs/2501.16142

MR.Q combines model-based representation learning with an off-policy
actor-critic algorithm.

Core Components:
- State and state-action representations learned through a latent dynamics model.
- Multi-step latent dynamics, reward, and termination prediction.
- Decoupled representation learning and actor-critic optimisation.
- Twin critics with target policy smoothing.
- Multi-step value targets.
- Loss-Adjusted Prioritized (LAP) replay.
- Adaptive reward scaling.

The encoder is trained separately from the actor and critic using uniformly
sampled sequences. Actor-critic updates use prioritized sequences.

The target networks are periodically updated using hard parameter copies,
followed by a block of encoder training updates.
"""

import copy
import logging
import os
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import MRQConfig
from cares_reinforcement_learning.algorithm.schedulers import ExponentialScheduler
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.memory.memory_sampler import SARLSequenceTensorSample
from cares_reinforcement_learning.networks.MRQ import Actor, Critic, Encoder
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation


class TwoHot:
    """Two-hot encoding over symexp-spaced categorical reward bins."""

    def __init__(self, device: torch.device, lower: float, upper: float, count: int):
        if count < 2:
            raise ValueError("reward_bins must be at least 2")

        linear_bins = torch.linspace(lower, upper, count, device=device)

        self.bins = linear_bins.sign() * torch.expm1(linear_bins.abs())
        self.count = count

    def encode(self, rewards: torch.Tensor) -> torch.Tensor:
        flat_rewards = rewards.reshape(-1)

        upper_indices = torch.searchsorted(
            self.bins.contiguous(), flat_rewards.contiguous()
        ).clamp(1, self.count - 1)

        lower_indices = upper_indices - 1

        lower_bins = self.bins[lower_indices]
        upper_bins = self.bins[upper_indices]

        upper_weights = ((flat_rewards - lower_bins) / (upper_bins - lower_bins)).clamp(
            0.0, 1.0
        )

        targets = torch.zeros(
            flat_rewards.numel(), self.count, dtype=rewards.dtype, device=rewards.device
        )

        targets.scatter_(
            1, lower_indices.unsqueeze(-1), (1.0 - upper_weights).unsqueeze(-1)
        )

        targets.scatter_add_(
            1, upper_indices.unsqueeze(-1), upper_weights.unsqueeze(-1)
        )

        return targets.reshape(*rewards.shape[:-1], self.count)

    def cross_entropy(
        self, logits: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        targets = self.encode(rewards)

        return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)


class MRQ(SARLAlgorithm[np.ndarray]):
    """Continuous-control MR.Q with decoupled representation learning."""

    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        encoder_network: Encoder,
        config: MRQConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)
        self.encoder_net = encoder_network.to(self.device)

        # Target networks
        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_encoder_net = copy.deepcopy(self.encoder_net).to(self.device)

        self.target_actor_net.eval()
        self.target_critic_net.eval()
        self.target_encoder_net.eval()

        # General parameters
        self.action_num = self.actor_net.num_actions
        self.gamma = config.gamma

        self.learn_counter = 0

        # Training schedule
        self.target_update_freq = config.target_update_freq
        self.policy_update_freq = config.policy_update_freq

        self.encoder_horizon = config.encoder_horizon
        self.q_horizon = config.q_horizon

        # PER / LAP
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_alpha = config.per_alpha
        self.min_priority = config.min_priority

        # Exploration noise
        self.action_noise_scheduler = ExponentialScheduler(
            start_value=config.action_noise_start,
            end_value=config.action_noise_end,
            decay_steps=config.action_noise_decay,
        )

        self.action_noise = self.action_noise_scheduler.get_value(0)

        # Target policy smoothing
        self.policy_noise_scheduler = ExponentialScheduler(
            start_value=config.policy_noise_start,
            end_value=config.policy_noise_end,
            decay_steps=config.policy_noise_decay,
        )

        self.policy_noise = self.policy_noise_scheduler.get_value(0)
        self.policy_noise_clip = config.policy_noise_clip

        # Encoder loss
        self.dynamics_weight = config.dynamics_weight
        self.reward_weight = config.reward_weight
        self.terminal_weight = config.terminal_weight

        # Actor / critic optimisation
        self.pre_activation_weight = config.pre_activation_weight
        self.critic_grad_clip = config.critic_grad_clip

        # Reward scaling
        self.reward_scale = 1.0
        self.target_reward_scale = 0.0

        # Termination prediction is enabled after observing a terminal transition.
        self.seen_terminal = False

        self.two_hot = TwoHot(
            device=self.device,
            lower=config.reward_lower,
            upper=config.reward_upper,
            count=config.reward_bins,
        )

        # Optimisers
        self.actor_net_optimiser = torch.optim.AdamW(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )

        self.critic_net_optimiser = torch.optim.AdamW(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

        self.encoder_net_optimiser = torch.optim.AdamW(
            self.encoder_net.parameters(),
            lr=config.encoder_lr,
            **config.encoder_lr_params,
        )

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        state_tensor = torch.as_tensor(
            observation.vector_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            state_embedding = self.encoder_net.zs(state_tensor)

            action = self.actor_net.act(state_embedding)
            action = action.squeeze(0).cpu().numpy()

        if not evaluation:
            noise = np.random.normal(0.0, self.action_noise, size=action.shape)
            action = action + noise

        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        return ActionSample(action=action, source="policy")

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:
        state_tensor = torch.as_tensor(
            state.vector_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        action_tensor = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            state_embedding = self.encoder_net.zs(state_tensor)

            state_action_embedding = self.encoder_net.zsa(
                state_embedding, action_tensor
            )

            q_values_one, q_values_two = self.critic_net(state_action_embedding)

            q_value = torch.minimum(q_values_one, q_values_two)

        return q_value.item() * self.reward_scale

    @staticmethod
    def _masked_mean(
        loss: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Mean over the batch, masking invalid transitions."""

        return (loss * valid).mean()

    def _update_encoder(
        self, sample_tensor: SARLSequenceTensorSample
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        observations_tensor = sample_tensor.observations.vector_state
        actions_tensor = sample_tensor.actions
        rewards_tensor = sample_tensor.rewards
        terminated_tensor = sample_tensor.terminated
        valid_tensor = sample_tensor.valid

        horizon = self.encoder_horizon
        batch_size = observations_tensor.shape[0]

        # Target embeddings for the next H observations.
        with torch.no_grad():
            next_states = observations_tensor[:, 1 : horizon + 1]

            next_states = next_states.reshape(-1, observations_tensor.shape[-1])

            target_embeddings = self.target_encoder_net.zs(next_states)

            target_embeddings = target_embeddings.reshape(batch_size, horizon, -1)

        predicted_embedding = self.encoder_net.zs(observations_tensor[:, 0])

        encoder_loss_total = torch.zeros((), device=self.device)

        dynamics_loss_total = 0.0
        reward_loss_total = 0.0
        terminal_loss_total = 0.0

        for step in range(horizon):
            predicted_done, predicted_embedding, predicted_reward = (
                self.encoder_net.model_all(predicted_embedding, actions_tensor[:, step])
            )

            valid = valid_tensor[:, step]

            # Latent dynamics prediction
            dynamics_loss = self._masked_mean(
                (predicted_embedding - target_embeddings[:, step]).square(), valid
            )

            # Reward prediction
            reward_loss = self._masked_mean(
                self.two_hot.cross_entropy(predicted_reward, rewards_tensor[:, step]),
                valid,
            )

            # Termination prediction
            if self.seen_terminal:
                terminal_loss = self._masked_mean(
                    (predicted_done - terminated_tensor[:, step]).square(), valid
                )
            else:
                terminal_loss = torch.zeros_like(dynamics_loss)

            encoder_loss_total = encoder_loss_total + (
                self.dynamics_weight * dynamics_loss
                + self.reward_weight * reward_loss
                + self.terminal_weight * terminal_loss
            )

            dynamics_loss_total += dynamics_loss.detach().item()
            reward_loss_total += reward_loss.detach().item()
            terminal_loss_total += terminal_loss.detach().item()

        # Update encoder
        self.encoder_net_optimiser.zero_grad()

        encoder_loss_total.backward()

        self.encoder_net_optimiser.step()

        info["encoder_loss"] = encoder_loss_total.detach().item()
        info["dynamics_loss"] = dynamics_loss_total / horizon
        info["reward_prediction_loss"] = reward_loss_total / horizon
        info["terminal_prediction_loss"] = terminal_loss_total / horizon

        return info

    def _update_critic(
        self,
        observations_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        terminated_tensor: torch.Tensor,
        valid_tensor: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            # Encoder representations remain fixed during RL updates.
            state_embedding = self.encoder_net.zs(observations_tensor[:, 0])

            state_action_embedding = self.encoder_net.zsa(
                state_embedding, actions_tensor[:, 0]
            )

            # Find the final real next state for each sampled sequence.
            # Shorter sequences may occur at episode boundaries.
            sequence_lengths = valid_tensor.squeeze(-1).sum(dim=1).long()

            batch_indices = torch.arange(
                observations_tensor.shape[0], device=self.device
            )

            next_states = observations_tensor[batch_indices, sequence_lengths]

            next_embedding = self.target_encoder_net.zs(next_states)

            # Target policy smoothing
            next_actions = self.target_actor_net.act(next_embedding)

            target_noise = self.policy_noise * torch.randn_like(next_actions)

            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )

            next_actions = torch.clamp(next_actions + target_noise, -1.0, 1.0)

            next_state_action_embedding = self.target_encoder_net.zsa(
                next_embedding, next_actions
            )

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_state_action_embedding
            )

            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            target_q_values = target_q_values * self.target_reward_scale

            # Multi-step discounted return.
            cumulative_rewards = torch.zeros_like(target_q_values)
            discount = torch.ones_like(target_q_values)

            for step in range(self.q_horizon):
                valid = valid_tensor[:, step]

                cumulative_rewards = cumulative_rewards + (
                    discount * rewards_tensor[:, step] * valid
                )

                # Invalid (padded) transitions do not consume a timestep.
                discount = discount * torch.where(
                    valid.bool(),
                    self.gamma * (1.0 - terminated_tensor[:, step]),
                    torch.ones_like(valid),
                )

            # Normalise targets by the current reward scale.
            q_target = (cumulative_rewards + discount * target_q_values) / max(
                self.reward_scale, 1e-8
            )

        q_values_one, q_values_two = self.critic_net(state_action_embedding)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        critic_loss_one = F.smooth_l1_loss(q_values_one, q_target)

        critic_loss_two = F.smooth_l1_loss(q_values_two, q_target)

        critic_loss_total = 0.5 * (critic_loss_one + critic_loss_two)

        # Update critic
        self.critic_net_optimiser.zero_grad()

        critic_loss_total.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic_net.parameters(), self.critic_grad_clip
        )

        self.critic_net_optimiser.step()

        # LAP priorities
        priorities = (
            torch.maximum(td_error_one, td_error_two)
            .clamp_min(self.min_priority)
            .pow(self.per_alpha)
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

        # Diagnostics
        info["critic_loss_one"] = critic_loss_one.detach().item()
        info["critic_loss_two"] = critic_loss_two.detach().item()
        info["critic_loss_total"] = critic_loss_total.detach().item()

        info["q1_mean"] = q_values_one.detach().mean().item()
        info["q2_mean"] = q_values_two.detach().mean().item()

        info["target_q_mean"] = q_target.mean().item()

        info["td_abs_mean"] = torch.maximum(td_error_one, td_error_two).mean().item()

        info["reward_scale"] = float(self.reward_scale)

        return info, priorities

    def _update_actor(self, states: torch.Tensor) -> dict[str, Any]:
        info: dict[str, Any] = {}

        # State representations are fixed for the actor update.
        with torch.no_grad():
            state_embedding = self.encoder_net.zs(states)

        # The actor must retain gradients through:
        # action -> state-action encoder -> critic -> actor.
        #
        # Freeze the encoder and critic parameters without disabling
        # autograd through their computations.
        frozen_parameters = list(self.encoder_net.parameters()) + list(
            self.critic_net.parameters()
        )

        original_requires_grad = [
            parameter.requires_grad for parameter in frozen_parameters
        ]

        try:
            for parameter in frozen_parameters:
                parameter.requires_grad_(False)

            actions, pre_activation = self.actor_net(state_embedding)

            state_action_embedding = self.encoder_net.zsa(state_embedding, actions)

            q_values_one, q_values_two = self.critic_net(state_action_embedding)

            actor_q_values = 0.5 * (q_values_one + q_values_two)

            actor_loss = (
                -actor_q_values.mean()
                + self.pre_activation_weight * pre_activation.square().mean()
            )

            # Update actor
            self.actor_net_optimiser.zero_grad()

            actor_loss.backward()

            self.actor_net_optimiser.step()

        finally:
            for parameter, was_trainable in zip(
                frozen_parameters, original_requires_grad
            ):
                parameter.requires_grad_(was_trainable)

        info["actor_loss"] = actor_loss.detach().item()
        info["actor_q_mean"] = actor_q_values.detach().mean().item()

        return info

    def update_from_batch(
        self, episode_context: EpisodeContext, sample_tensor: SARLSequenceTensorSample
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}

        # Update noise schedules
        self.action_noise = self.action_noise_scheduler.get_value(
            episode_context.training_step
        )

        self.policy_noise = self.policy_noise_scheduler.get_value(
            episode_context.training_step
        )

        info["action_noise"] = float(self.action_noise)
        info["policy_noise"] = float(self.policy_noise)

        # Unpack sampled sequences
        observations_tensor = sample_tensor.observations.vector_state
        actions_tensor = sample_tensor.actions
        rewards_tensor = sample_tensor.rewards
        terminated_tensor = sample_tensor.terminated
        valid_tensor = sample_tensor.valid

        # Update critic
        critic_info, priorities = self._update_critic(
            observations_tensor,
            actions_tensor,
            rewards_tensor,
            terminated_tensor,
            valid_tensor,
        )

        info |= critic_info

        # Update actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_info = self._update_actor(observations_tensor[:, 0])

            info |= actor_info

        return info, priorities

    def update_target_networks(self) -> None:
        self.hard_update_params(self.actor_net, self.target_actor_net)
        self.hard_update_params(self.critic_net, self.target_critic_net)
        self.hard_update_params(self.encoder_net, self.target_encoder_net)

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        if len(memory_buffer) == 0:
            return info

        self.learn_counter += 1

        # MR.Q periodically synchronizes targets and trains the encoder.
        if (self.learn_counter - 1) % self.target_update_freq == 0:
            self.update_target_networks()

            self.target_reward_scale = self.reward_scale
            self.reward_scale = max(float(memory_buffer.reward_scale()), 1e-8)

            # Representation learning uses independently sampled uniform sequences.
            for _ in range(self.target_update_freq):
                encoder_sample_tensor, _ = memory_sampler.sample_sequences(
                    memory=memory_buffer,
                    batch_size=self.batch_size,
                    horizon=self.encoder_horizon,
                    device=self.device,
                    use_per_buffer=0,
                )

                if not self.seen_terminal:
                    self.seen_terminal = bool(
                        encoder_sample_tensor.terminated.any().item()
                    )

                encoder_info = self._update_encoder(encoder_sample_tensor)
                info |= encoder_info

        # Actor and critic share the same prioritized sequence batch.
        sample_tensor, indices = memory_sampler.sample_sequences(
            memory=memory_buffer,
            batch_size=self.batch_size,
            horizon=self.q_horizon,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
        )

        train_info, priorities = self.update_from_batch(episode_context, sample_tensor)
        info |= train_info

        # Update LAP replay priorities.
        if self.use_per_buffer:
            memory_buffer.update_priorities(indices, priorities)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        os.makedirs(filepath, exist_ok=True)

        checkpoint = {
            # Networks
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "encoder": self.encoder_net.state_dict(),
            # Target networks
            "target_actor": self.target_actor_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "target_encoder": self.target_encoder_net.state_dict(),
            # Optimisers
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "encoder_optimizer": self.encoder_net_optimiser.state_dict(),
            # Training state
            "learn_counter": self.learn_counter,
            "action_noise": self.action_noise,
            "policy_noise": self.policy_noise,
            "reward_scale": self.reward_scale,
            "target_reward_scale": self.target_reward_scale,
            "seen_terminal": self.seen_terminal,
        }

        torch.save(checkpoint, os.path.join(filepath, f"{filename}_checkpoint.pth"))

        logging.info("MRQ models, optimisers, and training state have been saved...")

    def load_models(
        self,
        filepath: str,
        filename: str,
        load_mode: Literal["resume", "transfer"] = "resume",
    ) -> None:
        if load_mode not in ("resume", "transfer"):
            raise ValueError(f"Unknown load mode: {load_mode}")

        checkpoint = torch.load(
            os.path.join(filepath, f"{filename}_checkpoint.pth"),
            map_location="cpu",
            weights_only=True,
        )

        # Load learned networks
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])
        self.encoder_net.load_state_dict(checkpoint["encoder"])

        if load_mode == "transfer":
            # Align targets with transferred networks.
            self.update_target_networks()

            # Reset training state.
            self.learn_counter = 0
            self.reward_scale = 1.0
            self.target_reward_scale = 0.0
            self.seen_terminal = False

            self.action_noise = self.action_noise_scheduler.get_value(0)
            self.policy_noise = self.policy_noise_scheduler.get_value(0)

            logging.info("MRQ model weights loaded for transfer...")
            return

        # Restore target networks
        self.target_actor_net.load_state_dict(checkpoint["target_actor"])

        self.target_critic_net.load_state_dict(checkpoint["target_critic"])

        self.target_encoder_net.load_state_dict(checkpoint["target_encoder"])

        # Restore optimisers
        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])

        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])

        self.encoder_net_optimiser.load_state_dict(checkpoint["encoder_optimizer"])

        # Restore training state
        self.learn_counter = checkpoint.get("learn_counter", 0)

        self.action_noise = checkpoint.get("action_noise", self.action_noise)

        self.policy_noise = checkpoint.get("policy_noise", self.policy_noise)

        self.reward_scale = checkpoint.get("reward_scale", 1.0)

        self.target_reward_scale = checkpoint.get("target_reward_scale", 0.0)

        self.seen_terminal = checkpoint.get("seen_terminal", False)

        logging.info("MRQ models, optimisers, and training state have been loaded...")
