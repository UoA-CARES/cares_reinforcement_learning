"""
Original Paper: https://arxiv.org/abs/2405.02576

Continues Distributed TD3
Each Critic outputs a normal distribution

Original Implementation: https://github.com/UoA-CARES/cares_reinforcement_learning/blob/1fce6fcde5183bafe4efce0aa30fc59f630a8429/cares_reinforcement_learning/algorithm/policy/CTD4.py
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.CTD4 import Actor, Critic
from cares_reinforcement_learning.util.configurations import CTD4Config


class CTD4(TD3):
    def __init__(
        self,
        actor_network: Actor,
        ensemble_critic: Critic,
        config: CTD4Config,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=ensemble_critic,
            config=config,
            device=device,
        )

        self.fusion_method = config.fusion_method

        self.policy_noise_decay = config.policy_noise_decay
        self.target_policy_noise_scale = config.target_policy_noise_scale
        self.min_policy_noise = config.min_policy_noise

        self.lr_ensemble_critic = config.critic_lr
        self.ensemble_critic_optimizers = [
            torch.optim.Adam(
                critic_net.parameters(),
                lr=self.lr_ensemble_critic,
                **config.critic_lr_params,
            )
            for critic_net in self.critic_net.critics
        ]

    def _fusion_kalman(
        self,
        std_1: torch.Tensor,
        mean_1: torch.Tensor,
        std_2: torch.Tensor,
        mean_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kalman_gain = (std_1**2) / (std_1**2 + std_2**2)
        fusion_mean = mean_1 + kalman_gain * (mean_2 - mean_1)
        fusion_variance = (
            (1 - kalman_gain) * std_1**2 + kalman_gain * std_2**2 + 1e-6
        )  # 1e-6 was included to avoid values equal to 0
        fusion_std = torch.sqrt(fusion_variance)
        return fusion_mean, fusion_std

    def _kalman(
        self, u_set: list[torch.Tensor], std_set: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Kalman fusion
        for i in range(len(u_set) - 1):
            if i == 0:
                x_1, std_1 = u_set[i], std_set[i]
                x_2, std_2 = u_set[i + 1], std_set[i + 1]
                fusion_u, fusion_std = self._fusion_kalman(std_1, x_1, std_2, x_2)
            else:
                x_2, std_2 = u_set[i + 1], std_set[i + 1]
                fusion_u, fusion_std = self._fusion_kalman(
                    fusion_std, fusion_u, std_2, x_2
                )
        return fusion_u, fusion_std

    def _average(
        self, u_set: list[torch.Tensor], std_set: list[torch.Tensor], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Average value among the critic predictions:
        fusion_u = (
            torch.mean(torch.concat(u_set, dim=1), dim=1)
            .unsqueeze(0)
            .reshape(batch_size, 1)
        )
        fusion_std = (
            torch.mean(torch.concat(std_set, dim=1), dim=1)
            .unsqueeze(0)
            .reshape(batch_size, 1)
        )
        return fusion_u, fusion_std

    def _minimum(
        self, u_set: list[torch.Tensor], std_set: list[torch.Tensor], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fusion_min = torch.min(torch.concat(u_set, dim=1), dim=1)
        fusion_u = fusion_min.values.unsqueeze(0).reshape(batch_size, 1)
        # # This corresponds to the std of the min U index. That is; the min cannot be got between the stds
        std_concat = torch.concat(std_set, dim=1)
        fusion_std = (
            torch.stack(
                [std_concat[i, fusion_min.indices[i]] for i in range(len(std_concat))]
            )
            .unsqueeze(0)
            .reshape(batch_size, 1)
        )
        return fusion_u, fusion_std

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> list[float]:
        batch_size = len(states)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)

            target_noise = self.target_policy_noise_scale * torch.randn_like(
                next_actions
            )
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            u_set = []
            std_set = []

            for target_critic_net in self.target_critic_net.critics:
                u, std = target_critic_net(next_states, next_actions)

                u_set.append(u)
                std_set.append(std)

            if self.fusion_method == "kalman":
                fusion_u, fusion_std = self._kalman(u_set, std_set)
            elif self.fusion_method == "average":
                fusion_u, fusion_std = self._average(u_set, std_set, batch_size)
            elif self.fusion_method == "minimum":
                fusion_u, fusion_std = self._minimum(u_set, std_set, batch_size)
            else:
                raise ValueError(
                    f"Invalid fusion method: {self.fusion_method}. Please choose between 'kalman', 'average', or 'minimum'."
                )

            # Create the target distribution = aX+b
            u_target = rewards + self.gamma * fusion_u * (1 - dones)
            std_target = self.gamma * fusion_std
            target_distribution = torch.distributions.normal.Normal(
                u_target, std_target
            )

        critic_loss_totals = []

        for critic_net, critic_net_optimiser in zip(
            self.critic_net.critics, self.ensemble_critic_optimizers
        ):
            u_current, std_current = critic_net(states, actions)
            current_distribution = torch.distributions.normal.Normal(
                u_current, std_current
            )

            # Compute each critic loss
            critic_individual_loss = torch.distributions.kl.kl_divergence(
                current_distribution, target_distribution
            ).mean()

            critic_net_optimiser.zero_grad()
            critic_individual_loss.backward()
            critic_net_optimiser.step()

            critic_loss_totals.append(critic_individual_loss.item())

        return critic_loss_totals

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_actor(self, states: torch.Tensor) -> float:  # type: ignore[override]
        batch_size = len(states)

        actor_q_u_set = []
        actor_q_std_set = []

        actions = self.actor_net(states)
        with hlp.evaluating(self.critic_net):
            for critic_net in self.critic_net.critics:
                actor_q_u, actor_q_std = critic_net(states, actions)

                actor_q_u_set.append(actor_q_u)
                actor_q_std_set.append(actor_q_std)

        if self.fusion_method == "kalman":
            # Kalman filter combination of all critics and then a single mean for the actor loss
            fusion_u_a, _ = self._kalman(actor_q_u_set, actor_q_std_set)

        elif self.fusion_method == "average":
            # Average combination of all critics and then a single mean for the actor loss
            fusion_u_a, _ = self._average(actor_q_u_set, actor_q_std_set, batch_size)

        elif self.fusion_method == "minimum":
            # Minimum all critics and then a single mean for the actor loss
            fusion_u_a, _ = self._minimum(actor_q_u_set, actor_q_std_set, batch_size)

        else:
            raise ValueError(
                f"Invalid fusion method: {self.fusion_method}. Please choose between 'kalman', 'average', or 'minimum'."
            )

        actor_loss = -fusion_u_a.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        return actor_loss.item()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        self.learn_counter += 1

        self.target_policy_noise_scale *= self.policy_noise_decay
        self.target_policy_noise_scale = max(
            self.min_policy_noise, self.target_policy_noise_scale
        )

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards_tensor = rewards_tensor.reshape(batch_size, 1)
        dones_tensor = dones_tensor.reshape(batch_size, 1)

        info: dict[str, Any] = {}

        # Update Critics
        critic_loss_totals = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        info["critic_loss_totals"] = critic_loss_totals

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_loss = self._update_actor(states_tensor)
            info["actor_loss"] = actor_loss

            # Update ensemble of target critics
            for critic_net, target_critic_net in zip(
                self.critic_net.critics, self.target_critic_net.critics
            ):
                hlp.soft_update_params(critic_net, target_critic_net, self.tau)

            # Update target actor
            hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

        return info
