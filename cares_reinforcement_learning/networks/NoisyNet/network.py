from typing import Any

import torch
from torch import nn

from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.networks.noisy_linear import NoisyLinear
from cares_reinforcement_learning.util.configurations import NoisyNetConfig


class BaseNoisyNetwork(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int, network: nn.Module):
        super().__init__(observation_size=observation_size, num_actions=num_actions)
        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def reset_noise(self):
        for module in self.network.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()

    @torch.no_grad()
    def noise_stats(self) -> dict[str, Any]:
        sigma_vals = []
        sigma_max_vals = []
        sigma_mu_ratios = []
        weight_noise_rms = []
        bias_noise_rms = []

        layer_stats: dict[str, Any] = {}
        layer_i = 0

        for m in self.modules():
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
            layer_stats[f"noisy_layer_{layer_i}_sigma_mean"] = float(
                sigma_all.mean().item()
            )
            layer_stats[f"noisy_layer_{layer_i}_sigma_mu_ratio"] = float(
                sigma_mu_ratios[-1]
            )
            layer_stats[f"noisy_layer_{layer_i}_weight_noise_rms"] = float(
                weight_noise_rms[-1]
            )
            layer_i += 1

        if not sigma_vals:
            return {}

        sigma_all = torch.cat([v.flatten() for v in sigma_vals])
        out: dict[str, Any] = {
            "noisy_sigma_mean": float(sigma_all.mean().item()),
            "noisy_sigma_std": float(sigma_all.std().item()),
            "noisy_sigma_max": float(max(sigma_max_vals)),
            "noisy_sigma_mu_ratio_mean": float(
                sum(sigma_mu_ratios) / len(sigma_mu_ratios)
            ),
            "noisy_weight_noise_rms_mean": float(
                sum(weight_noise_rms) / len(weight_noise_rms)
            ),
            "noisy_bias_noise_rms_mean": float(
                sum(bias_noise_rms) / len(bias_noise_rms)
            ),
        }

        # enable if you want layer-by-layer drilldown
        out.update(layer_stats)

        return out


class DefaultNetwork(BaseNoisyNetwork):
    def __init__(self, observation_size: int, num_actions: int):
        network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            NoisyLinear(64, 64, sigma_init=1.0),
            nn.ReLU(),
            NoisyLinear(64, num_actions, sigma_init=0.5),
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )


class Network(BaseNoisyNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: NoisyNetConfig):

        network = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.network_config,
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )
