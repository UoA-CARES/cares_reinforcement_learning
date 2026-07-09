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
        self.total_units_replaced = 0

        self.replacement_strategy = config.replacement_strategy

        self.sites: list[FeatureSite] = []
        self.handles: list[Any] = []
        self.grad_handles: list[Any] = []

        # CBP / GnT utility tracking.
        self.utility_ema: dict[str, torch.Tensor] = {}
        self.bias_corrected_utility: dict[str, torch.Tensor] = {}
        self.mean_feature_act: dict[str, torch.Tensor] = {}
        self.age: dict[str, torch.Tensor] = {}
        self.accumulated_replacements: dict[str, torch.Tensor] = {}

        # Paper-style dead-unit tracking from the Nature CBP paper.
        # A unit is "dead" if it is active on less than 1% of samples in the
        # measurement window. For ReLU-style activations this follows their
        # `(features > 0).mean(dim=0)` logging.
        self.active_window_sum: dict[str, torch.Tensor] = {}
        self.active_window_count: dict[str, torch.Tensor] = {}
        self.active_lifetime_sum: dict[str, torch.Tensor] = {}
        self.active_lifetime_count: dict[str, torch.Tensor] = {}

        # Behaviour-window activation buffers for paper-aligned dead-unit and
        # representation-rank measurements. These buffers intentionally collect
        # activations when the network is in eval mode, which matches this PPO
        # implementation's rollout/action-collection path. They are not used by
        # CBP replacement itself.
        self.activation_window: dict[str, list[torch.Tensor]] = {}
        self.activation_window_rows: dict[str, int] = {}

        # ReDo-style dormancy tracking. This is a normalized activation-magnitude
        # metric and is intentionally named separately from paper-style dead units.
        self.redo_activation_abs_ema: dict[str, torch.Tensor] = {}
        self.redo_activity_frac_ema: dict[str, torch.Tensor] = {}

        # KNIFE-style RUA tracking.
        # Window values are reset on the configured summary/log interval.
        # Lifetime values are reset only when the unit itself is reset/replaced.
        self.update_activity_window_sum: dict[str, torch.Tensor] = {}
        self.update_activity_window_count: dict[str, torch.Tensor] = {}
        self.update_activity_lifetime_sum: dict[str, torch.Tensor] = {}
        self.update_activity_lifetime_count: dict[str, torch.Tensor] = {}

        self.last_activation: dict[str, torch.Tensor] = {}
        self.last_summary: dict[str, float] = {}

        if self.enabled:
            self._discover_sites()
            self._register_hooks()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

        for handle in self.grad_handles:
            handle.remove()
        self.grad_handles.clear()

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
            forward_handle = site.hook_module.register_forward_hook(
                self._make_forward_hook(site)
            )
            self.handles.append(forward_handle)

            grad_handle = site.producer_module.weight.register_hook(
                self._make_weight_grad_hook(site)
            )
            self.grad_handles.append(grad_handle)

    def _make_forward_hook(self, site: FeatureSite):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not self.enabled:
                return

            if not torch.is_tensor(output):
                return

            activation = output.detach()

            if activation.ndim != 2:
                return

            # Paper-aligned measurement window: collect behaviour activations
            # during rollout/action selection. In this PPO implementation the
            # networks are put in eval mode while collecting actions, so this
            # avoids mixing PPO minibatch-training activations into the rank and
            # dead-unit measurements.
            if not self.model.training:
                self._record_activation_window(site, activation)

            if self.config.training_only and not self.model.training:
                return

            self._update_activation_metrics(site, activation)

        return hook

    def _make_weight_grad_hook(self, site: FeatureSite):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self.enabled:
                return grad

            if self.config.training_only and not self.model.training:
                return grad

            if not torch.is_tensor(grad):
                return grad

            self._update_gradient_metrics(site, grad.detach())
            return grad

        return hook

    @torch.no_grad()
    def _ensure_site_state(
        self,
        site: FeatureSite,
        num_units: int,
        device: torch.device,
    ) -> None:
        if site.name in self.age:
            return

        self.utility_ema[site.name] = torch.zeros(num_units, device=device)
        self.bias_corrected_utility[site.name] = torch.zeros(num_units, device=device)
        self.mean_feature_act[site.name] = torch.zeros(num_units, device=device)
        self.age[site.name] = torch.zeros(num_units, device=device)
        self.accumulated_replacements[site.name] = torch.zeros(1, device=device)

        self.active_window_sum[site.name] = torch.zeros(num_units, device=device)
        self.active_window_count[site.name] = torch.zeros(num_units, device=device)
        self.active_lifetime_sum[site.name] = torch.zeros(num_units, device=device)
        self.active_lifetime_count[site.name] = torch.zeros(num_units, device=device)
        self.activation_window[site.name] = []
        self.activation_window_rows[site.name] = 0

        self.redo_activation_abs_ema[site.name] = torch.zeros(num_units, device=device)
        self.redo_activity_frac_ema[site.name] = torch.zeros(num_units, device=device)

        self.update_activity_window_sum[site.name] = torch.zeros(
            num_units, device=device
        )
        self.update_activity_window_count[site.name] = torch.zeros(
            num_units, device=device
        )
        self.update_activity_lifetime_sum[site.name] = torch.zeros(
            num_units, device=device
        )
        self.update_activity_lifetime_count[site.name] = torch.zeros(
            num_units, device=device
        )

    @torch.no_grad()
    def _record_activation_window(
        self,
        site: FeatureSite,
        activation: torch.Tensor,
    ) -> None:
        num_units = activation.shape[1]
        device = activation.device
        self._ensure_site_state(site, num_units, device)

        max_rows = int(self.config.activation_window_size)
        if max_rows <= 0:
            return

        chunk = activation.detach().float().cpu()
        self.activation_window[site.name].append(chunk)
        self.activation_window_rows[site.name] += int(chunk.shape[0])

        while (
            self.activation_window[site.name]
            and self.activation_window_rows[site.name] > max_rows
        ):
            removed = self.activation_window[site.name].pop(0)
            self.activation_window_rows[site.name] -= int(removed.shape[0])

    @torch.no_grad()
    def _activation_window_tensor(self, site_name: str) -> torch.Tensor | None:
        chunks = self.activation_window.get(site_name, [])
        if not chunks:
            return None

        window = torch.cat(chunks, dim=0)
        max_rows = int(self.config.activation_window_size)
        if max_rows > 0 and window.shape[0] > max_rows:
            window = window[-max_rows:]
        return window

    @torch.no_grad()
    def _update_activation_metrics(
        self,
        site: FeatureSite,
        activation: torch.Tensor,
    ) -> None:
        num_units = activation.shape[1]
        device = activation.device
        decay = self.config.utility_decay

        self._ensure_site_state(site, num_units, device)

        self.last_activation[site.name] = activation
        self.age[site.name].add_(1.0)

        bias_correction = 1.0 - torch.pow(
            torch.tensor(decay, device=device), self.age[site.name]
        )
        bias_correction = bias_correction.clamp_min(1e-12)

        activation_abs_mean = activation.abs().mean(dim=0)

        # Paper-style activity fraction: for ReLU networks, a unit is active when
        # its post-activation output is strictly positive. This is the metric used
        # in the original logging code: `(features > 0).float().mean(dim=0)`.
        batch_active_fraction = (activation > 0.0).float().mean(dim=0)
        self.active_window_sum[site.name].add_(batch_active_fraction)
        self.active_window_count[site.name].add_(1.0)
        self.active_lifetime_sum[site.name].add_(batch_active_fraction)
        self.active_lifetime_count[site.name].add_(1.0)

        # ReDo-style activity/dormancy diagnostics use activation magnitude, not
        # the paper-style firing fraction.
        self.redo_activation_abs_ema[site.name].mul_(decay).add_(
            (1.0 - decay) * activation_abs_mean
        )

        batch_activity = (
            (activation.abs() > self.config.activity_threshold).float().mean(dim=0)
        )
        self.redo_activity_frac_ema[site.name].mul_(decay).add_(
            (1.0 - decay) * batch_activity
        )

        self.mean_feature_act[site.name].mul_(decay).add_(
            (1.0 - decay) * activation.mean(dim=0)
        )

        if site.consumer_module is not None:
            output_weight_mag = site.consumer_module.weight.detach().abs().mean(dim=0)
            instantaneous_utility = output_weight_mag * activation_abs_mean
        else:
            instantaneous_utility = activation_abs_mean

        self.utility_ema[site.name].mul_(decay).add_(
            (1.0 - decay) * instantaneous_utility
        )

        self.bias_corrected_utility[site.name] = (
            self.utility_ema[site.name] / bias_correction
        )

    @torch.no_grad()
    def _update_gradient_metrics(
        self,
        site: FeatureSite,
        weight_grad: torch.Tensor,
    ) -> None:
        if weight_grad.ndim != 2:
            return

        num_units = weight_grad.shape[0]
        device = weight_grad.device

        self._ensure_site_state(site, num_units, device)

        weight = site.producer_module.weight.detach()

        grad_norm = weight_grad.norm(p=2, dim=1)
        weight_norm = weight.norm(p=2, dim=1)

        update_activity = grad_norm / (weight_norm + self.config.rua_eps)

        self.update_activity_window_sum[site.name].add_(update_activity)
        self.update_activity_window_count[site.name].add_(1.0)

        self.update_activity_lifetime_sum[site.name].add_(update_activity)
        self.update_activity_lifetime_count[site.name].add_(1.0)

    @torch.no_grad()
    def summary(
        self,
        prefix: str | None = None,
        force: bool = False,
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

        stagnant_threshold = self.config.stagnant_threshold
        volatile_threshold = self.config.volatile_threshold

        unit_count_total = 0.0
        dead_units_frac_weighted_sum = 0.0
        active_fraction_weighted_sum = 0.0
        dead_units_lifetime_frac_weighted_sum = 0.0
        active_lifetime_fraction_weighted_sum = 0.0

        redo_dormant_units_frac_weighted_sum = 0.0
        redo_activity_frac_weighted_sum = 0.0
        contribution_utility_mean_weighted_sum = 0.0

        update_activity_window_weighted_sum = 0.0
        stagnant_frac_window_weighted_sum = 0.0
        volatile_frac_window_weighted_sum = 0.0

        update_activity_lifetime_weighted_sum = 0.0
        stagnant_frac_lifetime_weighted_sum = 0.0
        volatile_frac_lifetime_weighted_sum = 0.0

        utility_p10s: list[float] = []
        rua_window_p10s: list[float] = []
        rua_lifetime_p10s: list[float] = []
        stable_ranks: list[float] = []
        effective_ranks: list[float] = []
        stable_rank_final_layer: float | None = None
        effective_rank_final_layer: float | None = None

        all_utility: list[torch.Tensor] = []
        all_rua_window: list[torch.Tensor] = []
        all_rua_lifetime: list[torch.Tensor] = []

        rua_sites_to_reset: list[str] = []

        for site in self.sites:
            if site.name not in self.age:
                continue

            clean_name = site.name.replace(".", "_")
            num_units = float(site.producer_module.out_features)

            activity_window = self._activation_window_tensor(site.name)
            if activity_window is not None and activity_window.shape[0] > 0:
                # Nature/CBP-paper metric: a ReLU unit is dead if it is active
                # on less than 1% of samples in the recent behaviour window.
                active_fraction = (activity_window > 0.0).float().mean(dim=0)
                dead_units_frac = (active_fraction < 0.01).float().mean().item()
                active_fraction_mean = active_fraction.mean().item()

                info[f"{prefix}/{clean_name}/dead_units_frac"] = dead_units_frac
                info[f"{prefix}/{clean_name}/active_fraction_mean"] = (
                    active_fraction_mean
                )
                info[f"{prefix}/{clean_name}/active_fraction_p10"] = torch.quantile(
                    active_fraction, 0.10
                ).item()
                info[f"{prefix}/{clean_name}/activity_window_size"] = float(
                    activity_window.shape[0]
                )

                dead_units_frac_weighted_sum += dead_units_frac * num_units
                active_fraction_weighted_sum += active_fraction_mean * num_units

                if should_rank:
                    rank_metrics = self._rank_metrics(activity_window)
                    for key, value in rank_metrics.items():
                        info[f"{prefix}/{clean_name}/{key}"] = value

                    if "stable_rank" in rank_metrics:
                        stable_ranks.append(rank_metrics["stable_rank"])
                        stable_rank_final_layer = rank_metrics["stable_rank"]

                    if "effective_rank" in rank_metrics:
                        effective_ranks.append(rank_metrics["effective_rank"])
                        effective_rank_final_layer = rank_metrics["effective_rank"]

            active_lifetime_count = self.active_lifetime_count[site.name].clamp_min(1.0)
            active_lifetime_fraction = (
                self.active_lifetime_sum[site.name] / active_lifetime_count
            )
            dead_units_lifetime_frac = (
                (active_lifetime_fraction < 0.01).float().mean().item()
            )
            active_lifetime_fraction_mean = active_lifetime_fraction.mean().item()

            info[f"{prefix}/{clean_name}/dead_units_lifetime_frac"] = (
                dead_units_lifetime_frac
            )
            info[f"{prefix}/{clean_name}/active_lifetime_fraction_mean"] = (
                active_lifetime_fraction_mean
            )

            dead_units_lifetime_frac_weighted_sum += (
                dead_units_lifetime_frac * num_units
            )
            active_lifetime_fraction_weighted_sum += (
                active_lifetime_fraction_mean * num_units
            )

            redo_activation_abs = self.redo_activation_abs_ema[site.name]
            redo_layer_mean_abs = redo_activation_abs.mean().clamp_min(1e-12)
            redo_dormancy_score = redo_activation_abs / redo_layer_mean_abs

            redo_dormant_units_frac = (
                (redo_dormancy_score <= self.config.dormant_threshold)
                .float()
                .mean()
                .item()
            )
            redo_activity_frac_mean = (
                self.redo_activity_frac_ema[site.name].mean().item()
            )

            info[f"{prefix}/{clean_name}/redo_dormant_units_frac"] = (
                redo_dormant_units_frac
            )
            info[f"{prefix}/{clean_name}/redo_dormancy_score_mean"] = (
                redo_dormancy_score.mean().item()
            )
            info[f"{prefix}/{clean_name}/redo_dormancy_score_p10"] = torch.quantile(
                redo_dormancy_score, 0.10
            ).item()
            info[f"{prefix}/{clean_name}/redo_activity_frac_mean"] = (
                redo_activity_frac_mean
            )

            redo_dormant_units_frac_weighted_sum += redo_dormant_units_frac * num_units
            redo_activity_frac_weighted_sum += redo_activity_frac_mean * num_units

            contribution_utility = self.bias_corrected_utility[site.name]

            contribution_utility_mean = contribution_utility.mean().item()
            contribution_utility_p10 = torch.quantile(contribution_utility, 0.10).item()

            info[f"{prefix}/{clean_name}/contribution_utility_mean"] = (
                contribution_utility_mean
            )
            info[f"{prefix}/{clean_name}/contribution_utility_p10"] = (
                contribution_utility_p10
            )
            info[f"{prefix}/{clean_name}/contribution_utility_min"] = (
                contribution_utility.min().item()
            )

            contribution_utility_mean_weighted_sum += (
                contribution_utility_mean * num_units
            )
            utility_p10s.append(contribution_utility_p10)
            all_utility.append(contribution_utility.detach().flatten())

            window_count = self.update_activity_window_count[site.name].clamp_min(1.0)
            update_activity_window = (
                self.update_activity_window_sum[site.name] / window_count
            )

            if torch.any(update_activity_window > 0):
                update_activity_window_mean = update_activity_window.mean().item()
                rua_window = (
                    update_activity_window
                    / update_activity_window.mean().clamp_min(1e-12)
                )

                rua_window_p10 = torch.quantile(rua_window, 0.10).item()
                stagnant_frac_window = (
                    (rua_window < stagnant_threshold).float().mean().item()
                )
                volatile_frac_window = (
                    (rua_window > volatile_threshold).float().mean().item()
                )

                info[f"{prefix}/{clean_name}/knife_update_activity_mean"] = (
                    update_activity_window_mean
                )
                info[f"{prefix}/{clean_name}/knife_rua_mean"] = rua_window.mean().item()
                info[f"{prefix}/{clean_name}/knife_rua_p10"] = rua_window_p10
                info[f"{prefix}/{clean_name}/knife_stagnant_units_frac"] = (
                    stagnant_frac_window
                )
                info[f"{prefix}/{clean_name}/knife_volatile_units_frac"] = (
                    volatile_frac_window
                )

                update_activity_window_weighted_sum += (
                    update_activity_window_mean * num_units
                )
                stagnant_frac_window_weighted_sum += stagnant_frac_window * num_units
                volatile_frac_window_weighted_sum += volatile_frac_window * num_units

                rua_window_p10s.append(rua_window_p10)
                all_rua_window.append(rua_window.detach().flatten())

            lifetime_count = self.update_activity_lifetime_count[site.name].clamp_min(
                1.0
            )
            update_activity_lifetime = (
                self.update_activity_lifetime_sum[site.name] / lifetime_count
            )

            if torch.any(update_activity_lifetime > 0):
                update_activity_lifetime_mean = update_activity_lifetime.mean().item()
                rua_lifetime = (
                    update_activity_lifetime
                    / update_activity_lifetime.mean().clamp_min(1e-12)
                )

                rua_lifetime_p10 = torch.quantile(rua_lifetime, 0.10).item()
                stagnant_frac_lifetime = (
                    (rua_lifetime < stagnant_threshold).float().mean().item()
                )
                volatile_frac_lifetime = (
                    (rua_lifetime > volatile_threshold).float().mean().item()
                )

                info[f"{prefix}/{clean_name}/knife_update_activity_lifetime_mean"] = (
                    update_activity_lifetime_mean
                )
                info[f"{prefix}/{clean_name}/knife_rua_lifetime_mean"] = (
                    rua_lifetime.mean().item()
                )
                info[f"{prefix}/{clean_name}/knife_rua_lifetime_p10"] = rua_lifetime_p10
                info[f"{prefix}/{clean_name}/knife_stagnant_units_lifetime_frac"] = (
                    stagnant_frac_lifetime
                )
                info[f"{prefix}/{clean_name}/knife_volatile_units_lifetime_frac"] = (
                    volatile_frac_lifetime
                )

                update_activity_lifetime_weighted_sum += (
                    update_activity_lifetime_mean * num_units
                )
                stagnant_frac_lifetime_weighted_sum += (
                    stagnant_frac_lifetime * num_units
                )
                volatile_frac_lifetime_weighted_sum += (
                    volatile_frac_lifetime * num_units
                )

                rua_lifetime_p10s.append(rua_lifetime_p10)
                all_rua_lifetime.append(rua_lifetime.detach().flatten())

            unit_count_total += num_units

            rua_sites_to_reset.append(site.name)

        if unit_count_total > 0:
            info[f"{prefix}/dead_units_frac"] = (
                dead_units_frac_weighted_sum / unit_count_total
            )
            info[f"{prefix}/active_fraction_mean"] = (
                active_fraction_weighted_sum / unit_count_total
            )
            info[f"{prefix}/dead_units_lifetime_frac"] = (
                dead_units_lifetime_frac_weighted_sum / unit_count_total
            )
            info[f"{prefix}/active_lifetime_fraction_mean"] = (
                active_lifetime_fraction_weighted_sum / unit_count_total
            )

            info[f"{prefix}/redo_dormant_units_frac"] = (
                redo_dormant_units_frac_weighted_sum / unit_count_total
            )
            info[f"{prefix}/redo_activity_frac_mean"] = (
                redo_activity_frac_weighted_sum / unit_count_total
            )
            info[f"{prefix}/contribution_utility_mean"] = (
                contribution_utility_mean_weighted_sum / unit_count_total
            )

            if update_activity_window_weighted_sum > 0:
                info[f"{prefix}/knife_update_activity_mean"] = (
                    update_activity_window_weighted_sum / unit_count_total
                )
                info[f"{prefix}/knife_stagnant_units_frac"] = (
                    stagnant_frac_window_weighted_sum / unit_count_total
                )
                info[f"{prefix}/knife_volatile_units_frac"] = (
                    volatile_frac_window_weighted_sum / unit_count_total
                )

            if update_activity_lifetime_weighted_sum > 0:
                info[f"{prefix}/knife_update_activity_lifetime_mean"] = (
                    update_activity_lifetime_weighted_sum / unit_count_total
                )
                info[f"{prefix}/knife_stagnant_units_lifetime_frac"] = (
                    stagnant_frac_lifetime_weighted_sum / unit_count_total
                )
                info[f"{prefix}/knife_volatile_units_lifetime_frac"] = (
                    volatile_frac_lifetime_weighted_sum / unit_count_total
                )

        self._add_weight_magnitude_metrics(info, prefix)

        if utility_p10s:
            info[f"{prefix}/contribution_utility_p10_mean"] = sum(utility_p10s) / len(
                utility_p10s
            )

        if all_utility:
            utility_global = torch.cat(all_utility)
            info[f"{prefix}/contribution_utility_p10_global"] = torch.quantile(
                utility_global, 0.10
            ).item()

        if rua_window_p10s:
            info[f"{prefix}/knife_rua_p10_mean"] = sum(rua_window_p10s) / len(
                rua_window_p10s
            )

        if all_rua_window:
            rua_window_global = torch.cat(all_rua_window)
            info[f"{prefix}/knife_rua_p10_global"] = torch.quantile(
                rua_window_global, 0.10
            ).item()

        if rua_lifetime_p10s:
            info[f"{prefix}/knife_rua_lifetime_p10_mean"] = sum(
                rua_lifetime_p10s
            ) / len(rua_lifetime_p10s)

        if all_rua_lifetime:
            rua_lifetime_global = torch.cat(all_rua_lifetime)
            info[f"{prefix}/knife_rua_lifetime_p10_global"] = torch.quantile(
                rua_lifetime_global, 0.10
            ).item()

        if stable_ranks:
            info[f"{prefix}/stable_rank_mean"] = sum(stable_ranks) / len(stable_ranks)

        if stable_rank_final_layer is not None:
            info[f"{prefix}/stable_rank_final_layer"] = stable_rank_final_layer

        if effective_ranks:
            info[f"{prefix}/effective_rank_mean"] = sum(effective_ranks) / len(
                effective_ranks
            )

        if effective_rank_final_layer is not None:
            info[f"{prefix}/effective_rank_final_layer"] = effective_rank_final_layer

        info[f"{prefix}/units_replaced_total"] = float(self.total_units_replaced)

        should_reset_window_metrics = self.model.training and should_log
        if should_reset_window_metrics:
            for site_name in rua_sites_to_reset:
                self.active_window_sum[site_name].zero_()
                self.active_window_count[site_name].zero_()
                self.update_activity_window_sum[site_name].zero_()
                self.update_activity_window_count[site_name].zero_()
                self.activation_window[site_name].clear()
                self.activation_window_rows[site_name] = 0

        self.last_summary = info
        return info

    @torch.no_grad()
    def _add_weight_magnitude_metrics(
        self,
        info: dict[str, float],
        prefix: str,
    ) -> None:
        total_abs_weight = 0.0
        total_weight_count = 0.0

        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            clean_name = module_name.replace(".", "_") if module_name else "linear"
            weight_abs_mean = module.weight.detach().abs().mean().item()
            weight_count = float(module.weight.numel())

            info[f"{prefix}/{clean_name}/average_weight_magnitude"] = weight_abs_mean

            total_abs_weight += weight_abs_mean * weight_count
            total_weight_count += weight_count

        if total_weight_count > 0:
            info[f"{prefix}/average_weight_magnitude"] = (
                total_abs_weight / total_weight_count
            )

    @torch.no_grad()
    def _rank_metrics(self, activation: torch.Tensor) -> dict[str, float]:
        if activation.shape[0] < 2:
            return {}

        x = activation.float()

        try:
            singular_values = torch.linalg.svdvals(x)
        except RuntimeError:
            return {}

        if singular_values.numel() == 0:
            return {}

        singular_sum = singular_values.sum()
        if singular_sum <= 0:
            return {
                "stable_rank": 0.0,
                "effective_rank": 0.0,
            }

        cumulative_ratio = torch.cumsum(singular_values, dim=0) / singular_sum
        stable_rank = float(torch.searchsorted(cumulative_ratio, 0.99).item() + 1)
        stable_rank = min(stable_rank, float(singular_values.numel()))

        probs = singular_values / singular_sum.clamp_min(1e-12)
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

        if self.replacement_strategy not in {"cbp", "gnt", "utility"}:
            raise NotImplementedError(
                f"replacement_strategy={self.replacement_strategy!r} is not "
                "implemented yet. Currently supported: 'cbp', 'gnt', 'utility'."
            )

        total_replaced = 0

        for site in self.sites:
            if site.consumer_module is None:
                continue

            unit_indices = self._select_units_for_replacement(site)

            for unit_idx_tensor in unit_indices:
                self._reset_unit(site, int(unit_idx_tensor.item()))
                total_replaced += 1

        self.total_units_replaced += total_replaced

        return {
            f"{self.name}/units_replaced": float(total_replaced),
            f"{self.name}/units_replaced_total": float(self.total_units_replaced),
        }

    @torch.no_grad()
    def _select_units_for_replacement(self, site: FeatureSite) -> torch.Tensor:
        if self.replacement_strategy in {"cbp", "gnt", "utility"}:
            return self._select_units_by_cbp_utility(site)

        raise NotImplementedError(
            f"replacement_strategy={self.replacement_strategy!r} is not implemented."
        )

    @torch.no_grad()
    def _select_units_by_cbp_utility(self, site: FeatureSite) -> torch.Tensor:
        if site.name not in self.bias_corrected_utility:
            return torch.empty(
                0, dtype=torch.long, device=site.producer_module.weight.device
            )

        age = self.age[site.name]
        utility = self.bias_corrected_utility[site.name]

        eligible_indices = torch.where(age > self.config.maturity_threshold)[0]

        if eligible_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=utility.device)

        expected_replacements = self.config.replacement_rate * eligible_indices.numel()

        if self.config.replacement_accumulate:
            # Algorithm-1 style deterministic accumulator: same long-run expected
            # replacement rate, but lower step-to-step variance and possible multi-unit
            # replacement when the accumulator exceeds 1.
            self.accumulated_replacements[site.name].add_(expected_replacements)
            num_replace = int(self.accumulated_replacements[site.name].item())
            self.accumulated_replacements[site.name].sub_(float(num_replace))
        else:
            # Reference-repo default behaviour: accumulate=False in GnT.
            # The integer part is always replaced, and the fractional part is
            # handled by one Bernoulli draw. For normal CBP settings where
            # replacement_rate * eligible_units < 1, this means replacing either
            # zero or one unit per site per update.
            expected_tensor = torch.tensor(
                expected_replacements,
                device=utility.device,
                dtype=torch.float32,
            )
            num_replace = int(torch.floor(expected_tensor).item())
            fractional_part = expected_tensor - float(num_replace)

            if torch.rand((), device=utility.device) < fractional_part:
                num_replace += 1

        if num_replace <= 0:
            return torch.empty(0, dtype=torch.long, device=utility.device)

        num_replace = min(num_replace, eligible_indices.numel())

        eligible_utility = utility[eligible_indices]
        relative_indices = torch.topk(-eligible_utility, k=num_replace).indices

        return eligible_indices[relative_indices]

    @torch.no_grad()
    def _reset_unit(self, site: FeatureSite, unit_idx: int) -> None:
        producer = site.producer_module
        consumer = site.consumer_module

        if consumer is None:
            return

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
        self.age[site.name][unit_idx] = 0.0

        self.active_window_sum[site.name][unit_idx] = 0.0
        self.active_window_count[site.name][unit_idx] = 0.0
        self.active_lifetime_sum[site.name][unit_idx] = 0.0
        self.active_lifetime_count[site.name][unit_idx] = 0.0

        self.redo_activation_abs_ema[site.name][unit_idx] = 0.0
        self.redo_activity_frac_ema[site.name][unit_idx] = 0.0

        self.update_activity_window_sum[site.name][unit_idx] = 0.0
        self.update_activity_window_count[site.name][unit_idx] = 0.0
        self.update_activity_lifetime_sum[site.name][unit_idx] = 0.0
        self.update_activity_lifetime_count[site.name][unit_idx] = 0.0

        self._reset_optimizer_state_for_unit(site, unit_idx)

    @torch.no_grad()
    def _reset_linear_output_unit(self, layer: nn.Linear, unit_idx: int) -> None:
        bound = self._initialization_bound(layer)

        layer.weight.data[unit_idx, :] = torch.empty(
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
