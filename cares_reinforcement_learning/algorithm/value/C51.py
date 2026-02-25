"""
C51 (Categorical Distributional DQN)
--------------------------------------

Original Paper: https://arxiv.org/pdf/1707.06887

C51 extends DQN by modeling the full return distribution
using a categorical representation over a fixed discrete support.

Core Problem:
- Standard DQN learns only the expected return:
      Q(s,a) = E[Z(s,a)]
- The Bellman target is a distribution, but DQN collapses
  it to a scalar, losing information.

Core Idea:
- Represent the return distribution Z(s,a) using
  a categorical distribution over N fixed atoms:

      z_i ∈ [V_min, V_max]

- The network outputs probabilities:
      p_i(s,a)

Expected Q-value:
      Q(s,a) = Σ_i z_i p_i(s,a)

Bellman Target (Distributional):
- Compute target distribution:
      Tz = r + γ (1 - done) z_i
- Because Tz may not align with fixed atoms,
  project the shifted distribution back onto
  the fixed support using a projection operator.

Loss:
- Cross-entropy between projected target
  distribution and predicted distribution.

Key Behaviour:
- Models full return distribution.
- More stable value learning.
- Improves performance over scalar DQN.

Limitations:
- Fixed support must be chosen carefully.
- Projection step adds implementation complexity.

C51 = DQN + categorical distributional value learning
      with fixed support and projection.
"""

from typing import Any

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.C51 import Network as C51Network
from cares_reinforcement_learning.networks.Rainbow import Network as RainbowNetwork
from cares_reinforcement_learning.util.configurations import C51Config, RainbowConfig


class C51(DQN):
    network: C51Network | RainbowNetwork
    target_network: C51Network | RainbowNetwork

    def __init__(
        self,
        network: C51Network | RainbowNetwork,
        config: C51Config | RainbowConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

        # C51
        self.num_atoms = config.num_atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(device)

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                next_action = self.network(next_states_tensor).argmax(1)
            else:
                # DQN
                next_action = self.target_network(next_states_tensor).argmax(1)

            next_dist = self.target_network.dist(next_states_tensor)
            next_dist = next_dist[range(batch_size), next_action]

            delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

            t_z = (
                rewards_tensor.unsqueeze(1)
                + (1 - dones_tensor.unsqueeze(1))
                * (self.gamma**self.n_step)
                * self.support
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

            proj_dist = torch.zeros_like(next_dist, device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.network.dist(states_tensor)
        log_p = torch.log(dist[range(batch_size), actions_tensor])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        info: dict[str, Any] = {}

        # -----------------------
        # Logging / diagnostics (C51)
        # -----------------------
        with torch.no_grad():
            # dist: [B, A, N]
            # proj_dist: [B, N]
            # chosen pred distribution: [B, N]
            pred_dist = dist[range(batch_size), actions_tensor]  # [B, N]

            # --- Expected values (scalar Q) ---
            # support: [N]
            q_pred = (pred_dist * self.support).sum(dim=1)  # [B]
            q_targ = (proj_dist * self.support).sum(dim=1)  # [B]

            td = q_pred - q_targ
            info["q_pred_mean"] = q_pred.mean().item()
            info["q_pred_std"] = q_pred.std().item()
            info["q_target_mean"] = q_targ.mean().item()
            info["q_target_std"] = q_targ.std().item()

            info["td_mean"] = td.mean().item()
            info["td_std"] = td.std().item()
            info["td_abs_mean"] = td.abs().mean().item()

            # --- Greedy action distribution under expected value ---
            q_all = (dist * self.support.view(1, 1, -1)).sum(dim=2)  # [B, A]
            greedy_actions = q_all.argmax(dim=1)  # [B]
            num_actions = self.network.num_actions
            counts = torch.bincount(greedy_actions, minlength=num_actions).float()
            probs = counts / counts.sum().clamp(min=1.0)
            entropy_actions = -(probs * (probs + 1e-12).log()).sum()
            info["greedy_action_entropy"] = entropy_actions.item()
            info["greedy_action_max_prob"] = probs.max().item()
            info["greedy_action_probs"] = probs.cpu().tolist()

            # --- Distribution entropy (collapse / over-uncertainty) ---
            pred_ent = -(pred_dist * (pred_dist + 1e-12).log()).sum(dim=1)  # [B]
            targ_ent = -(proj_dist * (proj_dist + 1e-12).log()).sum(dim=1)  # [B]
            info["pred_dist_entropy_mean"] = pred_ent.mean().item()
            info["pred_dist_entropy_std"] = pred_ent.std().item()
            info["target_dist_entropy_mean"] = targ_ent.mean().item()

            # --- Support saturation / projection health ---
            info["proj_mass_vmin"] = proj_dist[:, 0].mean().item()
            info["proj_mass_vmax"] = proj_dist[:, -1].mean().item()
            info["proj_mass_on_bounds"] = (
                (proj_dist[:, 0] + proj_dist[:, -1]).mean().item()
            )

        return elementwise_loss, info
