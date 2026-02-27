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

from typing import Any

import torch

import cares_reinforcement_learning.algorithm.lossess as loss
from cares_reinforcement_learning.algorithm.configurations import QRDQNConfig
from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.QRDQN import Network


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
    ) -> tuple[torch.Tensor, dict[str, Any]]:

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
        element_wise_quantile_huber_loss = loss.calculate_quantile_huber_loss(
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

        # -----------------------
        # Logging / diagnostics (QR-DQN)
        # -----------------------
        with torch.no_grad():
            info: dict[str, Any] = {}

            # Shapes (typical):
            # current_quantile_values: [B, A, N]
            # current_action_q_values: [B, 1, N]  (gathered at actions taken)
            # target_q_values:        [B, N]

            # Scalar Q-values via mean over quantiles (DQN-like view)
            q_mean = current_quantile_values.mean(dim=-1)  # [B, A]
            greedy_actions = q_mean.argmax(dim=1)  # [B]

            # Action histogram (batch-based, greedy under mean-Q)
            num_actions = self.network.num_actions
            counts = torch.bincount(greedy_actions, minlength=num_actions).float()
            probs = counts / counts.sum().clamp(min=1.0)
            entropy = -(probs * (probs + 1e-12).log()).sum()

            info["greedy_action_entropy"] = entropy.item()
            info["greedy_action_max_prob"] = probs.max().item()
            info["greedy_action_probs"] = probs.cpu().tolist()

            # Chosen-action distribution (predicted and target)
            pred_quantiles = current_action_q_values.squeeze(1)  # [B, N]
            targ_quantiles = target_q_values  # [B, N]

            pred_mean = pred_quantiles.mean(dim=1)  # [B]
            targ_mean = targ_quantiles.mean(dim=1)  # [B]

            # Scalar Q estimate from QR (mean over quantiles)
            info["pred_mean"] = pred_mean.mean().item()
            info["pred_mean_std"] = pred_mean.std().item()
            info["pred_mean_max"] = pred_mean.max().item()
            info["pred_mean_min"] = pred_mean.min().item()

            info["target_mean"] = targ_mean.mean().item()
            info["target_mean_std"] = targ_mean.std().item()
            info["target_mean_max"] = targ_mean.max().item()
            info["target_mean_min"] = targ_mean.min().item()

            # Scalar TD error (mean-return TD)
            td_mean = pred_mean - targ_mean  # [B]
            info["td_mean"] = td_mean.mean().item()
            info["td_std"] = td_mean.std().item()
            info["td_abs_mean"] = td_mean.abs().mean().item()

            # Distributional TD error (quantile-wise)
            td_q = pred_quantiles - targ_quantiles  # [B, N]
            info["td_q_abs_mean"] = td_q.abs().mean().item()
            info["td_q_abs_p95"] = td_q.abs().quantile(0.95).item()
            info["td_q_mean"] = td_q.mean().item()
            info["td_q_std"] = td_q.std().item()

            # Distribution spread / uncertainty (QR-specific)
            # Std over quantiles for the chosen action distribution
            pred_spread = pred_quantiles.std(dim=1)  # [B]
            targ_spread = targ_quantiles.std(dim=1)  # [B]
            info["pred_quantile_std_mean"] = pred_spread.mean().item()
            info["pred_quantile_std_p95"] = pred_spread.quantile(0.95).item()
            info["targ_quantile_std_mean"] = targ_spread.mean().item()
            info["targ_quantile_std_p95"] = targ_spread.quantile(0.95).item()

            # IQR (more robust than std)
            q25 = pred_quantiles.quantile(0.25, dim=1)
            q75 = pred_quantiles.quantile(0.75, dim=1)
            info["pred_quantile_iqr_mean"] = (q75 - q25).mean().item()

            # Tail means (risk-sensitive view; useful to see if distribution shifts correctly)
            # choose a small tail fraction
            tail_k = max(1, self.quantiles // 10)  # ~10% tail
            info["pred_cvar_low_mean"] = pred_quantiles[:, :tail_k].mean().item()
            info["pred_cvar_high_mean"] = pred_quantiles[:, -tail_k:].mean().item()
            info["targ_cvar_low_mean"] = targ_quantiles[:, :tail_k].mean().item()
            info["targ_cvar_high_mean"] = targ_quantiles[:, -tail_k:].mean().item()

            # Quantile huber loss stats (your actual objective)
            # element_wise_quantile_huber_loss is returned by your helper.
            # Log its mean/std/max for stability monitoring.
            info["quantile_huber_mean"] = element_wise_quantile_huber_loss.mean().item()
            info["quantile_huber_std"] = element_wise_quantile_huber_loss.std().item()
            info["quantile_huber_max"] = element_wise_quantile_huber_loss.max().item()

        return element_wise_loss, info
