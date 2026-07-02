# cares_reinforcement_learning/networks/plasticity.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

import cares_reinforcement_learning.networks.activation_functions as caf
from cares_reinforcement_learning.algorithm.configurations import PlasticityConfig
from cares_reinforcement_learning.algorithm.plasticity_adam import PlasticityAdam

SUPPORTED_TRAINABLE_LAYERS = (nn.Linear,)


@dataclass
class FeatureSite:
    name: str
    producer_module: nn.Linear
    hook_module: nn.Module
    consumer_module: nn.Linear | None
    mode: str


class NetworkPlasticityManager:
    SUPPORTED_ACTIVATIONS = (
        nn.ReLU,
        nn.LeakyReLU,
        nn.ELU,
        nn.GELU,
        nn.SiLU,
        nn.Tanh,
        nn.Sigmoid,
        caf.GoLU,
    )

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: PlasticityConfig,
        name: str,
    ) -> None:
        if config.replacement_enabled and not isinstance(optimizer, PlasticityAdam):
            raise TypeError(
                "NetworkPlasticityManager requires PlasticityAdam when "
                "replacement_enabled=True so per-unit Adam state can be reset."
            )

        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.name = name

        self.enabled = config.enabled
        self.step_count = 0

        self.sites: list[FeatureSite] = []
        self.handles: list[Any] = []

        self.activity_ema: dict[str, torch.Tensor] = {}
        self.utility_ema: dict[str, torch.Tensor] = {}
        self.bias_corrected_utility: dict[str, torch.Tensor] = {}
        self.mean_feature_act: dict[str, torch.Tensor] = {}
        self.age: dict[str, torch.Tensor] = {}
        self.accumulated_replacements: dict[str, torch.Tensor] = {}

        self.last_activation: dict[str, torch.Tensor] = {}
        self.last_summary: dict[str, float] = {}

        if self.enabled:
            self._discover_sites()
            self._register_hooks()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _is_activation(self, module: nn.Module) -> bool:
        return isinstance(module, self.SUPPORTED_ACTIVATIONS)

    def _has_trainable_parameters(self, module: nn.Module) -> bool:
        return any(parameter.requires_grad for parameter in module.parameters())

    def _is_supported_trainable_layer(self, module: nn.Module) -> bool:
        return isinstance(module, SUPPORTED_TRAINABLE_LAYERS)

    def _discover_sites(self) -> None:
        for seq_name, seq in self.model.named_modules():
            if not isinstance(seq, nn.Sequential):
                continue

            children = list(seq.named_children())

            for idx, (child_name, child) in enumerate(children):
                if not self._is_supported_trainable_layer(child):
                    continue

                if idx + 1 >= len(children):
                    continue

                _, possible_activation = children[idx + 1]
                if not self._is_activation(possible_activation):
                    continue

                consumer_module: nn.Linear | None = None
                unsupported_between = False

                for _, later_module in children[idx + 2 :]:
                    if self._is_supported_trainable_layer(later_module):
                        consumer_module = later_module
                        break

                    if self._has_trainable_parameters(later_module):
                        unsupported_between = True
                        break

                if unsupported_between:
                    continue

                if consumer_module is None and not self.config.include_output_layer:
                    continue

                site_name = f"{seq_name}.{child_name}" if seq_name else child_name

                self.sites.append(
                    FeatureSite(
                        name=site_name,
                        producer_module=child,
                        hook_module=possible_activation,
                        consumer_module=consumer_module,
                        mode="post_activation",
                    )
                )

    def _register_hooks(self) -> None:
        for site in self.sites:
            handle = site.hook_module.register_forward_hook(self._make_hook(site))
            self.handles.append(handle)

    def _make_hook(self, site: FeatureSite):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not self.enabled:
                return

            if self.config.training_only and not self.model.training:
                return

            if not torch.is_tensor(output):
                return

            activation = output.detach()

            if activation.ndim != 2:
                return

            self._update_site(site, activation)

        return hook

    @torch.no_grad()
    def _update_site(self, site: FeatureSite, activation: torch.Tensor) -> None:
        num_units = activation.shape[1]
        device = activation.device
        decay = self.config.utility_decay

        if site.name not in self.activity_ema:
            self.activity_ema[site.name] = torch.zeros(num_units, device=device)
            self.utility_ema[site.name] = torch.zeros(num_units, device=device)
            self.bias_corrected_utility[site.name] = torch.zeros(
                num_units, device=device
            )
            self.mean_feature_act[site.name] = torch.zeros(num_units, device=device)
            self.age[site.name] = torch.zeros(num_units, device=device)
            self.accumulated_replacements[site.name] = torch.zeros(1, device=device)

        self.last_activation[site.name] = activation
        self.age[site.name].add_(1.0)

        bias_correction = 1.0 - torch.pow(
            torch.tensor(decay, device=device), self.age[site.name]
        )
        bias_correction = bias_correction.clamp_min(1e-12)

        batch_activity = (
            (activation.abs() > self.config.activity_threshold).float().mean(dim=0)
        )

        self.activity_ema[site.name].mul_(decay).add_((1.0 - decay) * batch_activity)

        self.mean_feature_act[site.name].mul_(decay).add_(
            (1.0 - decay) * activation.mean(dim=0)
        )

        if site.consumer_module is not None:
            output_weight_mag = site.consumer_module.weight.detach().abs().mean(dim=0)
            new_utility = output_weight_mag * activation.abs().mean(dim=0)
        else:
            new_utility = activation.abs().mean(dim=0)

        self.utility_ema[site.name].mul_(decay).add_((1.0 - decay) * new_utility)

        self.bias_corrected_utility[site.name] = (
            self.utility_ema[site.name] / bias_correction
        )

    @torch.no_grad()
    def summary(
        self, prefix: str | None = None, force: bool = False
    ) -> dict[str, float]:
        if not self.enabled:
            return {}

        self.step_count += 1

        should_log = force or self.step_count % self.config.log_interval == 0
        should_rank = self.config.compute_rank and (
            force or self.step_count % self.config.rank_interval == 0
        )

        if not should_log and not should_rank:
            return {}

        prefix = prefix or self.name
        info: dict[str, float] = {}

        dormant_fracs: list[float] = []
        utility_means: list[float] = []
        weight_abs_means: list[float] = []

        for site in self.sites:
            if site.name not in self.activity_ema:
                continue

            activity = self.activity_ema[site.name]
            utility = self.bias_corrected_utility[site.name]

            clean_name = site.name.replace(".", "_")
            dormant_frac = (
                (activity < self.config.dormant_threshold).float().mean().item()
            )

            info[f"{prefix}/{clean_name}/dormant_frac"] = dormant_frac
            info[f"{prefix}/{clean_name}/activity_mean"] = activity.mean().item()
            info[f"{prefix}/{clean_name}/activity_min"] = activity.min().item()
            info[f"{prefix}/{clean_name}/utility_mean"] = utility.mean().item()
            info[f"{prefix}/{clean_name}/utility_min"] = utility.min().item()
            info[f"{prefix}/{clean_name}/utility_p10"] = torch.quantile(
                utility, 0.10
            ).item()

            weight_abs_mean = site.producer_module.weight.detach().abs().mean().item()
            info[f"{prefix}/{clean_name}/weight_abs_mean"] = weight_abs_mean

            dormant_fracs.append(dormant_frac)
            utility_means.append(utility.mean().item())
            weight_abs_means.append(weight_abs_mean)

            if should_rank and site.name in self.last_activation:
                for key, value in self._rank_metrics(
                    self.last_activation[site.name]
                ).items():
                    info[f"{prefix}/{clean_name}/{key}"] = value

        if dormant_fracs:
            info[f"{prefix}/dormant_frac_mean"] = sum(dormant_fracs) / len(
                dormant_fracs
            )

        if utility_means:
            info[f"{prefix}/utility_mean"] = sum(utility_means) / len(utility_means)

        if weight_abs_means:
            info[f"{prefix}/weight_abs_mean"] = sum(weight_abs_means) / len(
                weight_abs_means
            )

        self.last_summary = info
        return info

    @torch.no_grad()
    def _rank_metrics(self, activation: torch.Tensor) -> dict[str, float]:
        if activation.shape[0] < 2:
            return {}

        x = activation.float()
        x = x - x.mean(dim=0, keepdim=True)

        try:
            singular_values = torch.linalg.svdvals(x)
        except RuntimeError:
            return {}

        if singular_values.numel() == 0:
            return {}

        fro_sq = torch.sum(singular_values**2)
        spectral_sq = singular_values[0] ** 2
        stable_rank = 0.0 if spectral_sq <= 0 else (fro_sq / spectral_sq).item()

        probs = singular_values / singular_values.sum().clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        effective_rank = torch.exp(entropy).item()

        return {
            "stable_rank": stable_rank,
            "effective_rank": effective_rank,
        }

    @torch.no_grad()
    def step_replacement(self) -> dict[str, float]:
        if not self.enabled or not self.config.replacement_enabled:
            return {}

        total_replaced = 0

        for site in self.sites:
            if site.consumer_module is None:
                continue

            if site.name not in self.bias_corrected_utility:
                continue

            age = self.age[site.name]
            utility = self.bias_corrected_utility[site.name]

            eligible_indices = torch.where(age > self.config.maturity_threshold)[0]

            if eligible_indices.numel() == 0:
                continue

            self.accumulated_replacements[site.name].add_(
                self.config.replacement_rate * eligible_indices.numel()
            )

            num_replace = int(self.accumulated_replacements[site.name].item())
            self.accumulated_replacements[site.name].sub_(float(num_replace))

            if num_replace <= 0:
                continue

            num_replace = min(num_replace, eligible_indices.numel())

            eligible_utility = utility[eligible_indices]
            relative_indices = torch.topk(-eligible_utility, k=num_replace).indices
            unit_indices = eligible_indices[relative_indices]

            for unit_idx_tensor in unit_indices:
                self._reset_unit(site, int(unit_idx_tensor.item()))
                total_replaced += 1

        return {f"{self.name}/units_replaced": float(total_replaced)}

    @torch.no_grad()
    def _reset_unit(self, site: FeatureSite, unit_idx: int) -> None:
        producer = site.producer_module
        consumer = site.consumer_module

        if consumer is None:
            return

        # Paper-faithful order:
        # 1. compensate consumer bias using current outgoing weights
        # 2. zero outgoing weights
        # 3. reinitialize producer weights / reset producer bias
        if consumer.bias is not None:
            decay = self.config.utility_decay
            age = self.age[site.name][unit_idx]
            bias_correction = 1.0 - decay ** age.item()
            bias_correction = max(bias_correction, 1e-12)

            bias_corrected_mean_feature = (
                self.mean_feature_act[site.name][unit_idx] / bias_correction
            )

            consumer.bias.data += (
                consumer.weight.data[:, unit_idx] * bias_corrected_mean_feature
            )

        consumer.weight.data[:, unit_idx] = 0.0

        self._reset_linear_output_unit(producer, unit_idx)

        self.utility_ema[site.name][unit_idx] = 0.0
        self.bias_corrected_utility[site.name][unit_idx] = 0.0
        self.mean_feature_act[site.name][unit_idx] = 0.0
        self.activity_ema[site.name][unit_idx] = 0.0
        self.age[site.name][unit_idx] = 0.0

        self._reset_optimizer_state_for_unit(site, unit_idx)

    @torch.no_grad()
    def _reset_linear_output_unit(self, layer: nn.Linear, unit_idx: int) -> None:
        bound = self._initialization_bound(layer)

        layer.weight.data[unit_idx, :] = 0.0
        layer.weight.data[unit_idx, :] += torch.empty(
            layer.in_features,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        ).uniform_(-bound, bound)

        if layer.bias is not None:
            layer.bias.data[unit_idx] = 0.0

    def _initialization_bound(self, layer: nn.Linear) -> float:
        init = self.config.init.lower()
        activation_name = self.config.activation_name.lower()

        if activation_name in ["swish", "silu", "elu", "golu"]:
            activation_name = "relu"

        if init == "default":
            return float((1.0 / layer.in_features) ** 0.5)

        if init == "xavier":
            gain = nn.init.calculate_gain(activation_name)
            return float(gain * (6.0 / (layer.in_features + layer.out_features)) ** 0.5)

        if init == "lecun":
            return float((3.0 / layer.in_features) ** 0.5)

        gain = nn.init.calculate_gain(activation_name)
        return float(gain * (3.0 / layer.in_features) ** 0.5)

    @torch.no_grad()
    def _reset_optimizer_state_for_unit(self, site: FeatureSite, unit_idx: int) -> None:
        producer = site.producer_module
        consumer = site.consumer_module

        self._zero_optimizer_state_slice(producer.weight, (unit_idx, slice(None)))

        if producer.bias is not None:
            self._zero_optimizer_state_slice(producer.bias, (unit_idx,))

        if consumer is not None:
            self._zero_optimizer_state_slice(consumer.weight, (slice(None), unit_idx))

    @torch.no_grad()
    def _zero_optimizer_state_slice(
        self,
        parameter: torch.nn.Parameter,
        index: tuple[Any, ...],
    ) -> None:
        state = self.optimizer.state.get(parameter)

        if not state:
            return

        for value in state.values():
            if not torch.is_tensor(value):
                continue

            if value.ndim == 0:
                continue

            try:
                value[index] = 0.0
            except (IndexError, RuntimeError):
                continue
