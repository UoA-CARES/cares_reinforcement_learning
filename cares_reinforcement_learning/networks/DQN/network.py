from typing import Any

import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.networks.noisy_linear import NoisyLinear
from cares_reinforcement_learning.algorithm.configurations import DQNConfig


class BaseNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "BaseDQN is an abstract class and cannot be instantiated directly."
        )

    @torch.no_grad()
    def _module_noise_stats(self, key: str, layer: nn.Module) -> dict[str, Any]:
        sigma_vals = []
        sigma_max_vals = []
        sigma_mu_ratios = []
        weight_noise_rms = []
        bias_noise_rms = []

        layer_stats: dict[str, Any] = {}
        layer_i = 0

        for m in layer.modules():
            if not isinstance(m, NoisyLinear):
                continue

            # sigma stats
            w_sigma = m.weight_sigma.detach().abs()
            b_sigma = m.bias_sigma.detach().abs()
            sigma_all = torch.cat([w_sigma.flatten(), b_sigma.flatten()])
            sigma_vals.append(sigma_all)

            sigma_max_vals.append(float(sigma_all.max().item()))

            # sigma / mu ratio (scale-invariant-ish)
            w_mu = m.weight_mu.detach().abs()
            b_mu = m.bias_mu.detach().abs()
            mu_all = torch.cat([w_mu.flatten(), b_mu.flatten()]).clamp(min=1e-12)
            sigma_mu_ratios.append((sigma_all.mean() / mu_all.mean()).item())

            # actual injected noise magnitude right now (depends on epsilon buffers)
            w_noise = (m.weight_sigma * m.weight_epsilon).detach()
            b_noise = (m.bias_sigma * m.bias_epsilon).detach()
            weight_noise_rms.append(w_noise.pow(2).mean().sqrt().item())
            bias_noise_rms.append(b_noise.pow(2).mean().sqrt().item())

            # optional per-layer (sometimes too noisy for logs)
            layer_stats[f"{key}_noisy_layer_{layer_i}_sigma_mean"] = float(
                sigma_all.mean().item()
            )
            layer_stats[f"{key}_noisy_layer_{layer_i}_sigma_mu_ratio"] = float(
                sigma_mu_ratios[-1]
            )
            layer_stats[f"{key}_noisy_layer_{layer_i}_weight_noise_rms"] = float(
                weight_noise_rms[-1]
            )
            layer_i += 1

        if not sigma_vals:
            return {}

        sigma_all = torch.cat([v.flatten() for v in sigma_vals])
        out: dict[str, Any] = {
            f"{key}_noisy_sigma_mean": float(sigma_all.mean().item()),
            f"{key}_noisy_sigma_std": float(sigma_all.std().item()),
            f"{key}_noisy_sigma_max": float(max(sigma_max_vals)),
            f"{key}_noisy_sigma_mu_ratio_mean": float(
                sum(sigma_mu_ratios) / len(sigma_mu_ratios)
            ),
            f"{key}_noisy_weight_noise_rms_mean": float(
                sum(weight_noise_rms) / len(weight_noise_rms)
            ),
            f"{key}_noisy_bias_noise_rms_mean": float(
                sum(bias_noise_rms) / len(bias_noise_rms)
            ),
        }

        # enable if you want layer-by-layer drilldown
        out.update(layer_stats)

        return out


class BaseDQN(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        network: nn.Module,
    ):
        super().__init__(observation_size=observation_size, num_actions=num_actions)

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.network(state)
        return output


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseDQN):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [64, 64]

        network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )


class Network(BaseDQN):
    def __init__(self, observation_size: int, num_actions: int, config: DQNConfig):

        network = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.network_config,
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )
