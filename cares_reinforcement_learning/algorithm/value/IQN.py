"""
Implicit Quantile Networks (IQN).

Paper: https://arxiv.org/abs/1806.06923

IQN extends DQN by learning a continuous quantile function
of the return distribution.

The network samples quantile fractions and predicts their
corresponding return values. Training uses pairwise quantile
Huber regression against distributional Bellman targets.
"""

from typing import Any

import torch

from cares_reinforcement_learning.algorithm.configurations import IQNConfig
from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.IQN import Network as IQNNetwork


class IQN(DQN):
    network: IQNNetwork
    target_network: IQNNetwork

    def __init__(
        self,
        network: IQNNetwork,
        config: IQNConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

        self.num_quantiles = config.num_quantiles
        self.num_target_quantiles = config.num_target_quantiles
        self.num_action_quantiles = config.num_action_quantiles
        self.kappa = config.kappa

        if self.kappa <= 0:
            raise ValueError("IQN kappa must be positive")

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:

        actions = actions_tensor.long().view(-1)

        # Predicted quantiles: [B, N, A]
        quantiles, taus = self.network.quantiles(
            states_tensor,
            num_quantiles=self.num_quantiles,
        )

        # Select the distribution for the sampled action.
        action_indices = actions.view(batch_size, 1, 1).expand(
            -1, self.num_quantiles, 1
        )
        pred_quantiles = quantiles.gather(dim=2, index=action_indices).squeeze(
            2
        )  # [B, N]

        with torch.no_grad():
            # Select next action using expected Q-values.
            if self.use_double_dqn:
                next_q = self.network(
                    next_states_tensor,
                    num_quantiles=self.num_action_quantiles,
                )
            else:
                next_q = self.target_network(
                    next_states_tensor,
                    num_quantiles=self.num_action_quantiles,
                )

            next_actions = next_q.argmax(dim=1)

            # Target quantiles: [B, N', A]
            next_quantiles, _ = self.target_network.quantiles(
                next_states_tensor,
                num_quantiles=self.num_target_quantiles,
            )

            next_indices = next_actions.view(batch_size, 1, 1).expand(
                -1, self.num_target_quantiles, 1
            )

            next_quantiles = next_quantiles.gather(dim=2, index=next_indices).squeeze(
                2
            )  # [B, N']

            target_quantiles = (
                rewards_tensor.view(-1, 1)
                + (1.0 - dones_tensor.view(-1, 1))
                * (self.gamma**self.n_step)
                * next_quantiles
            )

        # Pairwise TD errors: [B, N, N']
        td_errors = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)

        abs_errors = td_errors.abs()

        # Huber loss with threshold kappa.
        huber_loss = torch.where(
            abs_errors <= self.kappa,
            0.5 * td_errors.square(),
            self.kappa * (abs_errors - 0.5 * self.kappa),
        )

        # Quantile regression weights.
        quantile_weights = (taus.unsqueeze(2) - (td_errors.detach() < 0).float()).abs()

        # Average over target quantiles and sum over predicted quantiles.
        # Matches Equation 3 of the original IQN paper.
        elementwise_loss = (
            (quantile_weights * huber_loss / self.kappa).mean(dim=2).sum(dim=1)
        )

        info: dict[str, Any] = {}

        with torch.no_grad():
            # Expected action values
            q_pred = pred_quantiles.mean(dim=1)
            q_target = target_quantiles.mean(dim=1)

            # Scalar TD error
            td_error = q_target - q_pred

            # Standard DQN diagnostics
            info["q_value_mean"] = q_pred.mean().item()
            info["q_value_std"] = q_pred.std(unbiased=False).item()

            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            info["td_error_mean"] = td_error.mean().item()
            info["td_error_std"] = td_error.std(unbiased=False).item()
            info["td_error_abs_mean"] = td_error.abs().mean().item()

            # IQN-specific diagnostics
            info["quantile_td_error_abs_mean"] = abs_errors.mean().item()
            info["quantile_pred_std"] = (
                pred_quantiles.std(dim=1, unbiased=False).mean().item()
            )

            info["quantile_target_std"] = (
                target_quantiles.std(dim=1, unbiased=False).mean().item()
            )

        return elementwise_loss, info
