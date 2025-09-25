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
from cares_reinforcement_learning.networks.CTD4 import Actor, Critic
from cares_reinforcement_learning.util.configurations import CTD4Config


class CTD4(TD3):
    critic_net: Critic
    target_critic_net: Critic

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

        self.lr_ensemble_critic = config.critic_lr
        self.ensemble_critic_optimizers = [
            torch.optim.Adam(
                critic_net.parameters(),
                lr=self.lr_ensemble_critic,
                **config.critic_lr_params,
            )
            for critic_net in self.critic_net.critics
        ]

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.unsqueeze(0)

        q_u_set = []
        q_std_set = []

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                for critic_net in self.critic_net.critics:
                    actor_q_u, actor_q_std = critic_net(state_tensor, action_tensor)

                    q_u_set.append(actor_q_u)
                    q_std_set.append(actor_q_std)

        fusion_u_a, _ = self._fuse_critic_outputs(1, q_u_set, q_std_set)

        return fusion_u_a.item()

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

    def _fuse_critic_outputs(
        self, batch_size: int, u_set: list[torch.Tensor], std_set: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        return fusion_u, fusion_std

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> tuple[dict[str, Any], np.ndarray]:
        batch_size = len(states)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)

            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )

            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            u_set = []
            std_set = []

            for target_critic_net in self.target_critic_net.critics:
                u, std = target_critic_net(next_states, next_actions)

                u_set.append(u)
                std_set.append(std)

            fusion_u, fusion_std = self._fuse_critic_outputs(batch_size, u_set, std_set)

            # Create the target distribution = aX+b
            u_target = rewards + self.gamma * fusion_u * (1 - dones)
            std_target = self.gamma * fusion_std

            target_distribution = torch.distributions.normal.Normal(
                u_target, std_target
            )

        critic_loss_totals = []
        critic_loss_elementwise = []

        for critic_net, critic_net_optimiser in zip(
            self.critic_net.critics, self.ensemble_critic_optimizers
        ):
            u_current, std_current = critic_net(states, actions)
            current_distribution = torch.distributions.normal.Normal(
                u_current, std_current
            )

            # Compute each critic los
            critic_elementwise_loss = torch.distributions.kl.kl_divergence(
                current_distribution, target_distribution
            )
            critic_loss_elementwise.append(critic_elementwise_loss)

            critic_loss = critic_elementwise_loss.mean()
            critic_loss_totals.append(critic_loss.item())

            critic_net_optimiser.zero_grad()
            critic_loss.backward()
            critic_net_optimiser.step()

        critic_losses = torch.stack(critic_loss_elementwise, dim=0)
        critic_losses = torch.max(critic_losses, dim=0).values

        # Update the Priorities - PER only
        priorities = (
            critic_losses.clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        critic_loss_total = np.mean(critic_loss_totals)

        info = {
            "critic_loss_total": critic_loss_total,
            "critic_loss_totals": critic_loss_totals,
        }

        return info, priorities

    def _update_actor(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        batch_size = len(states)

        actor_q_u_set = []
        actor_q_std_set = []

        actions = self.actor_net(states)
        with hlp.evaluating(self.critic_net):
            for critic_net in self.critic_net.critics:
                actor_q_u, actor_q_std = critic_net(states, actions)

                actor_q_u_set.append(actor_q_u)
                actor_q_std_set.append(actor_q_std)

        fusion_u_a, _ = self._fuse_critic_outputs(
            batch_size, actor_q_u_set, actor_q_std_set
        )

        actor_loss = -fusion_u_a.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        info = {
            "actor_loss": actor_loss.item(),
        }

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        super().save_models(filepath, filename)
        # Save each ensemble critic optimizer in a single file
        ensemble_optim_state = {
            f"optimizer_{idx}": opt.state_dict()
            for idx, opt in enumerate(self.ensemble_critic_optimizers)
        }
        torch.save(
            ensemble_optim_state,
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth",
        )

    def load_models(self, filepath: str, filename: str) -> None:
        super().load_models(filepath, filename)
        # Load each ensemble critic optimizer from the single file
        ensemble_optim_state = torch.load(
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth"
        )
        for idx, opt in enumerate(self.ensemble_critic_optimizers):
            opt.load_state_dict(ensemble_optim_state[f"optimizer_{idx}"])
