"""
QR-DQN (Quantile Regression DQN)
----------------------------------

Original Paper: https://arxiv.org/pdf/1710.10044

QR-DQN extends DQN by learning a full return distribution
instead of a single expected Q-value.

Core Problem:
- Standard DQN estimates:
      Q(s,a) = E[Z(s,a)]
- The Bellman target contains uncertainty, but DQN collapses
  it to a scalar expectation.
- Modeling the distribution can improve stability and learning.

Core Idea:
- Represent the return distribution Z(s,a) using N quantiles.
- The network outputs:
      Zθ(s,a) = {θ₁, θ₂, ..., θ_N}
  corresponding to fixed quantile fractions τ_i.

Expected Q-value:
      Q(s,a) = mean_i θ_i

Target Construction:
- Sample next action via greedy selection on mean value.
- Target quantiles:
      y_i = r + γ (1 - done) θ'_j(s', a*)
- No projection step required (unlike C51).

Loss (Quantile Regression):
- Minimize quantile Huber loss between predicted
  and target quantiles:

      L = ρ_τ^κ ( y_j - θ_i )

where ρ is the quantile regression loss.

Key Behaviour:
- Captures uncertainty in return estimates.
- Reduces variance and improves learning stability.
- Often improves performance over scalar DQN.

QR-DQN = DQN + quantile-based distributional value learning.
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.QRDQN import Network
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import QRDQNConfig


class QRDQN(DQN):
    network: Network
    target_network: Network

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
        self.quantile_taus = torch.tensor(
            [i / (self.quantiles + 1) for i in range(1, self.quantiles + 1)],
            dtype=torch.float32,
            device=device,
        )

    def _evaluate_quantile_at_action(self, s_quantiles, actions):
        batch_size = s_quantiles.shape[0]

        # Reshape actions to (batch_size, 1, 1) so it can be broadcasted properly
        actions = actions.view(batch_size, 1, 1)

        # Expand actions to (batch_size, 1, num_quantiles) for indexing
        action_index = actions.expand(-1, -1, self.quantiles)

        # Calculate quantile values at specified actions.
        sa_quantiles = s_quantiles.gather(dim=1, index=action_index)

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
        current_action_q_values = self._evaluate_quantile_at_action(
            current_quantile_values, actions_tensor
        )

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.use_double_dqn:
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
                next_state_q_values = self.network.calculate_quantiles(
                    next_states_tensor
                ).mean(dim=-1)
            else:
                next_state_q_values = self.target_network.calculate_quantiles(
                    next_states_tensor
                ).mean(dim=-1)

            # Calculate greedy actions.
            next_state_best_actions = torch.argmax(
                next_state_q_values, dim=1, keepdim=True
            )

            # Calculate quantile values of next states and actions at tau_hats.
            next_quantile_values = self.target_network.calculate_quantiles(
                next_states_tensor
            )

            best_next_q_values = self._evaluate_quantile_at_action(
                next_quantile_values, next_state_best_actions
            )

            # Calculate target quantile values.
            target_q_values = (
                rewards_tensor[..., None, None]
                + (1.0 - dones_tensor[..., None, None])
                * (self.gamma**self.n_step)
                * best_next_q_values
            ).squeeze(1)

        # Calculate TD errors.
        element_wise_quantile_huber_loss = hlp.calculate_quantile_huber_loss(
            current_action_q_values,
            target_q_values,
            self.quantile_taus,
            self.kappa,
            use_mean_reduction=False,
            use_pairwise_loss=False,
        )

        element_wise_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(
            dim=1, keepdim=True
        )

        return element_wise_loss
