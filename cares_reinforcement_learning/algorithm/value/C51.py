"""
Original Paper:
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.C51 import Network
from cares_reinforcement_learning.util.configurations import C51Config


class C51(DQN):
    def __init__(
        self,
        network: Network,
        config: C51Config,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

    def _c51_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
        with torch.no_grad():
            # DQN
            next_action = self.target_network(next_states_tensor).argmax(1)
            next_dist = self.target_network.dist(next_states_tensor)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = (
                rewards_tensor.unsqueeze(1)
                + (1 - dones_tensor.unsqueeze(1)) * self.gamma * self.support
            )

            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.num_atoms)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.network.dist(states_tensor)
        log_p = torch.log(dist[range(batch_size), actions_tensor])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _double_c51_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.network(next_states_tensor).argmax(1)
            next_dist = self.target_network.dist(next_states_tensor)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = (
                rewards_tensor.unsqueeze(1)
                + (1 - dones_tensor.unsqueeze(1)) * self.gamma * self.support
            )
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.num_atoms)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.network.dist(states_tensor)
        log_p = torch.log(dist[range(batch_size), actions_tensor])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        compute_c51_loss_fn = (
            self._double_c51_loss if self.use_double_dqn else self._c51_loss
        )
        elementwise_loss = compute_c51_loss_fn(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            batch_size,
        )
        return elementwise_loss
