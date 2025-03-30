"""
Original Paper:
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.QRDQN import Network
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import QRDQNConfig


class QRDQN(DQN):
    def __init__(
        self,
        network: Network,
        config: QRDQNConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

        # QRDQN
        self.kappa = config.kappa

        self.quantiles = config.quantiles
        self.quantile_taus = torch.FloatTensor(
            [i / self.quantiles for i in range(1, self.quantiles + 1)]
        ).to(device)

    def calculate_quantile_huber_loss(self, td_errors, taus, kappa=1.0):
        # Calculate huber loss element-wisely.
        element_wise_huber_loss = hlp.calculate_huber_loss(td_errors, kappa, use_=False)
        print(f"1a element_wise_huber_loss: {element_wise_huber_loss.mean()}")
        print(f"1a element_wise_huber_loss: {element_wise_huber_loss.shape}")

        # Calculate quantile huber loss element-wisely.
        element_wise_quantile_huber_loss = (
            torch.abs(taus[..., None] - (td_errors.detach() < 0).float())
            * element_wise_huber_loss
            / kappa
        )

        print(
            f"1b element_wise_quantile_huber_loss: {element_wise_quantile_huber_loss.mean()}"
        )

        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(
            dim=1, keepdim=True
        )

        print(f"1c batch_quantile_huber_loss: {batch_quantile_huber_loss.mean()}")

        return batch_quantile_huber_loss

    def evaluate_quantile_at_action(self, s_quantiles, actions):
        batch_size = s_quantiles.shape[0]
        num_quantiles = s_quantiles.shape[1]

        # Reshape actions to (batch_size, 1, 1) so it can be broadcasted properly
        actions = actions.view(batch_size, 1, 1)

        # Expand actions to (batch_size, num_quantiles, 1) for indexing
        action_index = actions.expand(batch_size, num_quantiles, 1)

        # Calculate quantile values at specified actions.
        sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

        return sa_quantiles

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:

        # Predicted Q-value quantiles for current state
        current_quantile_values = self.network.calculate_quantiles(states_tensor)

        # Gather Q-value quantiles of actions actually taken
        current_action_q_values = self.evaluate_quantile_at_action(
            current_quantile_values, actions_tensor
        )

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.use_double_dqn:
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
                next_state_q_values = self.network.calculate_quantiles(
                    next_states_tensor
                ).mean(dim=1)
            else:
                next_state_q_values = self.target_network.calculate_quantiles(
                    next_states_tensor
                ).mean(dim=1)

            # Calculate greedy actions.
            next_state_best_actions = torch.argmax(
                next_state_q_values, dim=1, keepdim=True
            )

            # Calculate quantile values of next states and actions at tau_hats.
            next_quantile_values = self.target_network.calculate_quantiles(
                next_states_tensor
            )

            best_next_q_values = self.evaluate_quantile_at_action(
                next_quantile_values, next_state_best_actions
            )

            rewards_expanded = (
                rewards_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, self.quantiles, 1)
            )
            dones_expanded = (
                dones_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, self.quantiles, 1)
            )

            # Calculate target quantile values.
            target_q_values = (
                rewards_expanded
                + (1.0 - dones_expanded)
                * (self.gamma**self.n_step)
                * best_next_q_values
            )

        # Calculate TD errors.
        element_wise_loss = hlp.calculate_quantile_huber_loss(
            current_action_q_values.permute(0, 2, 1),
            target_q_values.squeeze(-1),
            self.quantile_taus,
            self.kappa,
            use_mean_reduction=False,
            use_pairwise_loss=False,
        )

        batch_quantile_huber_loss = element_wise_loss.sum(dim=1).mean(
            dim=1, keepdim=True
        )

        return batch_quantile_huber_loss
