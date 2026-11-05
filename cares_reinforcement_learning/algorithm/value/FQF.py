"""
Fully Parameterized Quantile Function (FQF).

Paper: https://arxiv.org/abs/1911.02140

FQF learns both the quantile values and quantile fractions
used to approximate the return distribution.

The quantile value network is trained using quantile Huber regression.
The fraction proposal network is trained using the analytical
1-Wasserstein fraction gradient.
"""

import logging
import math
from typing import Any, Literal

import torch

from cares_reinforcement_learning.algorithm.configurations import FQFConfig
from cares_reinforcement_learning.algorithm.value.DQN import DQN
from cares_reinforcement_learning.networks.FQF import Network as FQFNetwork
from cares_reinforcement_learning.networks.RainbowFQF import Network as RainbowNetwork


class FQF(DQN):
    network: FQFNetwork
    target_network: FQFNetwork

    def __init__(
        self,
        network: FQFNetwork | RainbowNetwork,
        config: FQFConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

        self.num_quantiles = config.num_quantiles
        self.kappa = config.kappa
        self.fraction_entropy_coeff = config.fraction_entropy_coeff

        if self.kappa <= 0:
            raise ValueError("FQF kappa must be positive")

        if self.num_quantiles < 2:
            raise ValueError("FQF requires at least two quantiles")

        # Fraction proposal network parameters.
        self.fraction_parameters = list(self.network.fraction_layer.parameters())

        self.fraction_optimiser = torch.optim.RMSprop(
            self.fraction_parameters, lr=config.fraction_lr
        )

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        actions = actions_tensor.long().view(-1)

        # State representation: [B, F]
        state_features = self.network.feature_layer(states_tensor)

        # Learned current-state quantile fractions.
        # Detach features so the fraction objective cannot update the encoder.
        fractions, midpoints, probabilities = self.network.propose_fractions(
            states_tensor,
            state_features=state_features.detach(),
        )

        # Predicted quantiles: [B, N, A]
        # Detach midpoints so the value loss cannot update the fraction network.
        quantiles = self.network.quantiles(
            states_tensor,
            taus=midpoints.detach(),
            state_features=state_features,
        )

        # Select the distribution for the sampled action.
        action_indices = actions.view(batch_size, 1, 1).expand(
            -1, self.num_quantiles, 1
        )

        # [B, N]
        pred_quantiles = quantiles.gather(
            dim=2,
            index=action_indices,
        ).squeeze(2)

        with torch.no_grad():
            # Select the greedy next action using expected Q-values.
            # Each network's forward pass proposes fractions for the next state.
            if self.use_double_dqn:
                next_q = self.network(next_states_tensor)
            else:
                next_q = self.target_network(next_states_tensor)

            next_actions = next_q.argmax(dim=1)

            # Target state representation.
            next_features = self.target_network.feature_layer(next_states_tensor)

            # Equation 6: the current prediction and Bellman target
            # share the same current-state quantile midpoints.
            next_quantiles = self.target_network.quantiles(
                next_states_tensor,
                taus=midpoints.detach(),
                state_features=next_features,
            )

            next_indices = next_actions.view(batch_size, 1, 1).expand(
                -1, self.num_quantiles, 1
            )

            # [B, N]
            next_quantiles = next_quantiles.gather(
                dim=2,
                index=next_indices,
            ).squeeze(2)

            # Distributional Bellman target.
            target_quantiles = (
                rewards_tensor.view(-1, 1)
                + (1.0 - dones_tensor.view(-1, 1))
                * (self.gamma**self.n_step)
                * next_quantiles
            )

        # Pairwise TD errors: [B, N, N]
        td_errors = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)

        abs_errors = td_errors.abs()

        # Quantile Huber loss.
        huber_loss = torch.where(
            abs_errors <= self.kappa,
            0.5 * td_errors.square(),
            self.kappa * (abs_errors - 0.5 * self.kappa),
        )

        # Quantile regression weights.
        quantile_weights = (
            midpoints.detach().unsqueeze(2) - (td_errors.detach() < 0).float()
        ).abs()

        # Equation 7 of the original FQF paper.
        # Average over target quantiles and sum over predictions.
        elementwise_loss = (
            (quantile_weights * huber_loss / self.kappa).mean(dim=2).sum(dim=1)
        )

        info: dict[str, Any] = {}

        with torch.no_grad():
            # Expected action values.
            q_pred = (pred_quantiles * probabilities).sum(dim=1)

            # Target quantiles were evaluated at current-state midpoints.
            q_target = (target_quantiles * probabilities).sum(dim=1)

            # Scalar TD error.
            td_error = q_target - q_pred

            # Standard DQN diagnostics.
            info["q_value_mean"] = q_pred.mean().item()
            info["q_value_std"] = q_pred.std(unbiased=False).item()

            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            info["td_error_mean"] = td_error.mean().item()
            info["td_error_std"] = td_error.std(unbiased=False).item()
            info["td_error_abs_mean"] = td_error.abs().mean().item()

            # Distributional diagnostics.
            info["quantile_td_error_abs_mean"] = abs_errors.mean().item()

            info["quantile_pred_std"] = (
                pred_quantiles.std(dim=1, unbiased=False).mean().item()
            )

            info["quantile_target_std"] = (
                target_quantiles.std(dim=1, unbiased=False).mean().item()
            )

            # -------------------------------------------------
            # Action-value separation and policy diagnostics
            # -------------------------------------------------

            # Expected Q-values across all actions: [B, A]
            # FQF uses learned fraction probabilities, not a uniform mean.
            q_values = (quantiles * probabilities.unsqueeze(2)).sum(dim=1)

            greedy_actions = q_values.argmax(dim=1)
            num_actions = self.network.num_actions

            counts = torch.bincount(greedy_actions, minlength=num_actions).float()

            probs = counts / counts.sum().clamp(min=1.0)

            entropy = -(probs * (probs + 1e-12).log()).sum()

            info["greedy_action_entropy"] = entropy.item()
            info["greedy_action_max_prob"] = probs.max().item()
            info["greedy_action_probs"] = probs.cpu().tolist()

            if num_actions > 1:
                info["greedy_action_entropy_normalised"] = (
                    entropy / math.log(num_actions)
                ).item()

                top_two = torch.topk(q_values, k=2, dim=1).values

                info["greedy_action_margin_mean"] = (
                    (top_two[:, 0] - top_two[:, 1]).mean().item()
                )
            else:
                info["greedy_action_entropy_normalised"] = 0.0
                info["greedy_action_margin_mean"] = 0.0

            info["q_action_std_mean"] = (
                q_values.std(dim=1, unbiased=False).mean().item()
            )

        # Compute the fraction objective using the existing forward-pass values.
        # Returns per-transition losses [B] for optional PER weighting.
        fraction_loss, fraction_info = self._compute_fraction_loss(
            states_tensor,
            actions_tensor,
            state_features.detach(),
            fractions,
            probabilities,
            pred_quantiles.detach(),
        )

        info |= fraction_info

        return elementwise_loss, {"fraction": fraction_loss}, info

    def _compute_fraction_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        state_features: torch.Tensor,
        fractions: torch.Tensor,
        probabilities: torch.Tensor,
        midpoint_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        actions = actions_tensor.long().view(-1)

        # Evaluate quantiles at internal fraction boundaries only.
        # Midpoint quantiles are reused from _compute_loss().
        with torch.no_grad():
            boundary_quantiles = self.network.quantiles(
                states_tensor,
                taus=fractions[:, 1:-1].detach(),
                state_features=state_features,
            )

            boundary_indices = actions.view(-1, 1, 1).expand(
                -1, self.num_quantiles - 1, 1
            )

            boundary_values = boundary_quantiles.gather(
                dim=2,
                index=boundary_indices,
            ).squeeze(2)

            # Analytical 1-Wasserstein gradient (Equation 5).
            fraction_gradients = (
                2.0 * boundary_values - midpoint_values[:, :-1] - midpoint_values[:, 1:]
            )

        # Per-transition surrogate objective: [B].
        fraction_loss = (fraction_gradients.detach() * fractions[:, 1:-1]).sum(dim=1)

        # Per-transition entropy regularisation: [B].
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)

        # Keep the complete objective per transition.
        # DQN.train() handles PER weighting and batch reduction.
        fraction_loss = fraction_loss - self.fraction_entropy_coeff * entropy

        info: dict[str, Any] = {
            "fraction_loss": fraction_loss.detach().mean().item(),
            "fraction_entropy": entropy.detach().mean().item(),
            "fraction_min_prob": probabilities.detach().min().item(),
            "fraction_max_prob": probabilities.detach().max().item(),
            "fraction_deviation_uniform": (
                probabilities.detach() - 1.0 / self.num_quantiles
            )
            .abs()
            .mean()
            .item(),
        }

        info["fraction_entropy_normalised"] = (
            entropy.detach().mean() / math.log(self.num_quantiles)
        ).item()

        return fraction_loss, info

    def _optimise(
        self,
        loss: torch.Tensor,
        auxiliary_losses: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Update the fraction proposal and quantile-value networks."""

        fraction_loss = auxiliary_losses.pop("fraction")

        # Update fraction proposal network.
        self.fraction_optimiser.zero_grad()
        fraction_loss.backward()

        info: dict[str, Any] = {}

        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.fraction_parameters,
                max_norm=self.max_grad_norm,
            )

            info["fraction_grad_norm"] = grad_norm.item()
            info["fraction_grad_clipped"] = grad_norm.item() > self.max_grad_norm

        self.fraction_optimiser.step()

        # DQN's Adam optimiser also contains the fraction parameters.
        # Clear their gradients so only RMSProp updates them.
        self.fraction_optimiser.zero_grad(set_to_none=True)

        # Update the quantile-value network using DQN's existing optimiser.
        info |= super()._optimise(loss, auxiliary_losses)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        # DQN saves the networks, main optimiser, and learning counter.
        super().save_models(filepath, filename)

        # Save the additional fraction optimiser.
        torch.save(
            self.fraction_optimiser.state_dict(),
            f"{filepath}/{filename}_fraction_optimizer.pth",
        )

        logging.info("FQF fraction optimiser has been saved...")

    def load_models(
        self,
        filepath: str,
        filename: str,
        load_mode: Literal["resume", "transfer"] = "resume",
    ) -> None:
        # DQN restores the network weights and handles resume/transfer.
        super().load_models(filepath, filename, load_mode)

        # Transfer preserves network weights but resets optimiser state.
        if load_mode == "transfer":
            return

        # Restore the additional fraction optimiser for training resume.
        fraction_state = torch.load(
            f"{filepath}/{filename}_fraction_optimizer.pth",
            map_location="cpu",
        )

        self.fraction_optimiser.load_state_dict(fraction_state)

        logging.info("FQF fraction optimiser has been loaded...")
