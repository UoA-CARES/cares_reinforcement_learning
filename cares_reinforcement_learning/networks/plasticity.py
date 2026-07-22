# cares_reinforcement_learning/networks/plasticity.py

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterator

import torch
from torch import nn
from torch.optim import Optimizer

import cares_reinforcement_learning.networks.activation_functions as caf
from cares_reinforcement_learning.algorithm.configurations import PlasticityConfig
from cares_reinforcement_learning.algorithm.plasticity_adam import PlasticityAdam

# =============================================================================
# References
#
# The diagnostics and replacement logic in this file reimplement metrics from
# three papers. Method-level comments below cite the relevant paper by short
# name; full references:
#
#   [Dohare2024]  Dohare, S., Hernandez-Garcia, J.F., Lan, Q., Rahman, P.,
#                 Mahmood, A.R., & Sutton, R.S. (2024). "Loss of Plasticity in
#                 Deep Continual Learning." Nature, 632, 768-774.
#                 -> dead units, stable rank / effective rank (Fig. 2d, 4b),
#                    CBP/GnT contribution-utility replacement.
#
#   [Sokar2023]   Sokar, G., Agarwal, R., Castro, P.S., & Evci, U. (2023).
#                 "The Dormant Neuron Phenomenon in Deep Reinforcement
#                 Learning." ICML 2023.
#                 -> ReDo dormancy score and dormant-unit fraction.
#
#   [Liu2026]     Liu, Z., Gao, Z., Qin, H., Hu, J., Wu, J., Zhu, M.,
#                 Zhang, H., Ma, C., Shen, S., & Wang, C. (2026). "Stagnant
#                 Neuron: Towards Understanding the Plasticity Loss in
#                 Multi-Agent Reinforcement Learning Value Factorization
#                 Methods." arXiv:2606.25335.
#                 -> KNIFE relative update activity (RUA), stagnant/volatile
#                    neuron fractions.
# =============================================================================

# =============================================================================
# Metrics tracked
#
# Basis for what's recorded per feature site (see summary() for the full
# call tree, and each _summarize_* method for the exact formula + citation):
#
#   dead_units_frac / active_fraction_*   Fraction of units that have gone
#     (recent + lifetime)                 permanently silent (or, for
#                                          active_fraction, how often the
#                                          rest of the layer still fires).
#                                          The most literal symptom of
#                                          plasticity loss: capacity that's
#                                          been switched off for good.
#
#   stable_rank / effective_rank /        How many independent directions
#   stable_rank_pct                       the layer's activations actually
#                                          span. Falls as units collapse into
#                                          redundant copies of each other --
#                                          an earlier warning than dead units,
#                                          since it catches units that are
#                                          still firing but no longer firing
#                                          *usefully*.
#
#   redo_dormant_units_frac /             Units whose activation magnitude
#   redo_activity_frac_mean               has shrunk to near-negligible
#                                          relative to their layer -- a
#                                          softer, earlier precursor to fully
#                                          dead units.
#
#   contribution_utility_*                How much a unit's output actually
#                                          moves the network's predictions.
#                                          Unlike the metrics above, this is
#                                          the mechanism (not just symptom):
#                                          low-utility units are reset by
#                                          step_replacement() (CBP/GnT).
#
#   knife_rua_* / knife_stagnant_*  /     Whether a unit's weights are still
#   knife_volatile_*                      meaningfully updating relative to
#                                          their own magnitude. Stagnant units
#                                          can look perfectly alive by every
#                                          activation-side metric above yet
#                                          have effectively stopped learning
#                                          -- the gradient-side failure mode
#                                          the others can miss. Volatile is
#                                          the opposite: unstable, oversized
#                                          updates.
#
# Each is reported per-site and aggregated network-wide (unit-count-weighted
# mean, plus p10/global-quantile tails where relevant) in
# _add_network_summary_metrics.
# =============================================================================

SUPPORTED_TRAINABLE_LAYERS = (nn.Linear,)


class CaptureMode(Enum):
    """Controls which plasticity state the hooks are allowed to update."""

    TRAINING = auto()
    METRICS = auto()


@dataclass
class FeatureSite:
    name: str
    producer_module: nn.Linear
    hook_module: nn.Module
    consumer_module: nn.Linear | None


@dataclass
class RunningAverageState:
    total: torch.Tensor
    count: torch.Tensor


@dataclass
class ActivityState:
    window: RunningAverageState
    lifetime: RunningAverageState
    behaviour_chunks: list[torch.Tensor]
    behaviour_rows: int = 0


@dataclass
class UtilityState:
    ema: torch.Tensor
    bias_corrected: torch.Tensor
    mean_feature_activation: torch.Tensor
    age: torch.Tensor
    replacement_accumulator: torch.Tensor


@dataclass
class RedoState:
    activation_abs_ema: torch.Tensor
    activity_fraction_ema: torch.Tensor


@dataclass
class KnifeState:
    window: RunningAverageState
    lifetime: RunningAverageState


@dataclass
class SiteState:
    utility: UtilityState
    activity: ActivityState
    redo: RedoState
    knife: KnifeState
    last_activation: torch.Tensor | None = None


@dataclass
class SiteSummary:
    num_units: float
    metrics: dict[str, float]
    distributions: dict[str, torch.Tensor]


class NetworkPlasticityManager:
    """Tracks and reports neuron-level plasticity diagnostics for a model.

    Two public entry points are called from the training loop:
      - summary(): periodically returns diagnostics (dead units, ReDo
        dormancy, KNIFE RUA/stagnant/volatile, CBP utility, rank).
      - step_replacement(): periodically resets low-utility units (CBP/GnT).

    The file below is organized into five sections, in this order:
      1. Construction & lifecycle           (__init__, close)
      2. Site discovery & hook registration (_discover_sites, _register_hooks, ...)
      3. Per-step metric recording           (called from the hooks above)
      4. summary() and its call tree
      5. step_replacement() and its call tree
    """

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

        # All per-layer metric state is grouped by feature site. This avoids
        # maintaining parallel dictionaries that must be initialized and reset
        # together.
        self.site_states: dict[str, SiteState] = {}

        self.last_summary: dict[str, float] = {}

        # Hooks are inactive by default. Callers explicitly opt in to the
        # exact forward/backward passes that should update plasticity state.
        # A stack safely restores the previous mode if capture contexts nest.
        # This avoids auxiliary forwards (evaluation, diagnostics, target
        # networks, bootstrapping, etc.) accidentally contributing to plasticity statistics.
        self._capture_modes: list[CaptureMode] = []

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

    @property
    def _active_capture_mode(self) -> CaptureMode | None:
        """Return the currently active capture mode, or None when inactive."""
        return self._capture_modes[-1] if self._capture_modes else None

    @contextmanager
    def _capture(self, mode: CaptureMode) -> Iterator[None]:
        """Temporarily enable one capture mode and restore the prior mode."""
        self._capture_modes.append(mode)
        try:
            yield
        finally:
            popped_mode = self._capture_modes.pop()
            if popped_mode is not mode:
                raise RuntimeError("Plasticity capture contexts exited out of order.")

    def capture_training(self):
        """Capture training activations and gradients inside this context."""
        return self._capture(CaptureMode.TRAINING)

    def capture_metrics(self):
        """Capture rollout activations for rank/activity diagnostics."""
        return self._capture(CaptureMode.METRICS)

    # =========================================================================
    # Site discovery & hook registration
    #
    # Walks the model once (at construction time) to find producer -> activation
    # -> consumer triples ("feature sites"), then attaches the forward/backward
    # hooks that feed the per-step recording methods in the next section.
    # =========================================================================

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
            if not self.enabled or self._active_capture_mode is None:
                return

            if not torch.is_tensor(output):
                return

            activation = output.detach()

            if activation.ndim != 2:
                return

            if self._active_capture_mode is CaptureMode.METRICS:
                self._record_activation_window(site, activation)
                return

            if self._active_capture_mode is CaptureMode.TRAINING:
                if self.config.training_only and not self.model.training:
                    return
                self._update_activation_metrics(site, activation)

        return hook

    def _make_weight_grad_hook(self, site: FeatureSite):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if (
                not self.enabled
                or self._active_capture_mode is not CaptureMode.TRAINING
            ):
                return grad

            if self.config.training_only and not self.model.training:
                return grad

            if not torch.is_tensor(grad):
                return grad

            self._update_gradient_metrics(site, grad.detach())
            return grad

        return hook

    # =========================================================================
    # Per-step metric recording (called from the hooks registered above)
    #
    # These run on every forward/backward pass and only accumulate state --
    # no metric computation, aggregation, or logging happens here. That's
    # all deferred to summary(), in the next section.
    # =========================================================================

    @torch.no_grad()
    def _ensure_site_state(
        self,
        site: FeatureSite,
        num_units: int,
        device: torch.device,
    ) -> SiteState:
        existing = self.site_states.get(site.name)
        if existing is not None:
            return existing

        def zeros() -> torch.Tensor:
            return torch.zeros(num_units, device=device)

        state = SiteState(
            utility=UtilityState(
                ema=zeros(),
                bias_corrected=zeros(),
                mean_feature_activation=zeros(),
                age=zeros(),
                replacement_accumulator=torch.zeros(1, device=device),
            ),
            activity=ActivityState(
                window=RunningAverageState(total=zeros(), count=zeros()),
                lifetime=RunningAverageState(total=zeros(), count=zeros()),
                behaviour_chunks=[],
            ),
            redo=RedoState(
                activation_abs_ema=zeros(),
                activity_fraction_ema=zeros(),
            ),
            knife=KnifeState(
                window=RunningAverageState(total=zeros(), count=zeros()),
                lifetime=RunningAverageState(total=zeros(), count=zeros()),
            ),
        )
        self.site_states[site.name] = state
        return state

    @torch.no_grad()
    def _record_activation_window(
        self,
        site: FeatureSite,
        activation: torch.Tensor,
    ) -> None:
        state = self._ensure_site_state(site, activation.shape[1], activation.device)

        max_rows = int(self.config.activation_window_size)
        if max_rows <= 0:
            return

        chunk = activation.detach().float().cpu()
        state.activity.behaviour_chunks.append(chunk)
        state.activity.behaviour_rows += int(chunk.shape[0])

        while (
            state.activity.behaviour_chunks and state.activity.behaviour_rows > max_rows
        ):
            removed = state.activity.behaviour_chunks.pop(0)
            state.activity.behaviour_rows -= int(removed.shape[0])

    @torch.no_grad()
    def _activation_window_tensor(self, state: SiteState) -> torch.Tensor | None:
        chunks = state.activity.behaviour_chunks
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
        state = self._ensure_site_state(site, activation.shape[1], activation.device)
        decay = self.config.utility_decay

        state.last_activation = activation
        state.utility.age.add_(1.0)

        bias_correction = 1.0 - torch.pow(
            torch.tensor(decay, device=activation.device), state.utility.age
        )
        bias_correction.clamp_min_(1e-12)

        activation_abs_mean = activation.abs().mean(dim=0)

        # Paper-style firing statistics. A ReLU unit is active when its
        # post-activation value is strictly positive.
        batch_active_fraction = (activation > 0.0).float().mean(dim=0)
        state.activity.window.total.add_(batch_active_fraction)
        state.activity.window.count.add_(1.0)
        state.activity.lifetime.total.add_(batch_active_fraction)
        state.activity.lifetime.count.add_(1.0)

        # ReDo tracks normalized activation magnitude separately from the
        # paper-style positive-activation statistic above.
        state.redo.activation_abs_ema.mul_(decay).add_(
            (1.0 - decay) * activation_abs_mean
        )
        batch_activity = (
            (activation.abs() > self.config.activity_threshold).float().mean(dim=0)
        )
        state.redo.activity_fraction_ema.mul_(decay).add_(
            (1.0 - decay) * batch_activity
        )

        state.utility.mean_feature_activation.mul_(decay).add_(
            (1.0 - decay) * activation.mean(dim=0)
        )

        if site.consumer_module is not None:
            output_weight_magnitude = (
                site.consumer_module.weight.detach().abs().mean(dim=0)
            )
            instantaneous_utility = output_weight_magnitude * activation_abs_mean
        else:
            instantaneous_utility = activation_abs_mean

        state.utility.ema.mul_(decay).add_((1.0 - decay) * instantaneous_utility)
        state.utility.bias_corrected.copy_(state.utility.ema / bias_correction)

    @torch.no_grad()
    def _update_gradient_metrics(
        self,
        site: FeatureSite,
        weight_grad: torch.Tensor,
    ) -> None:
        if weight_grad.ndim != 2:
            return

        state = self._ensure_site_state(site, weight_grad.shape[0], weight_grad.device)
        weight = site.producer_module.weight.detach()

        grad_norm = weight_grad.norm(p=2, dim=1)
        weight_norm = weight.norm(p=2, dim=1)
        update_activity = grad_norm / (weight_norm + self.config.rua_eps)

        state.knife.window.total.add_(update_activity)
        state.knife.window.count.add_(1.0)
        state.knife.lifetime.total.add_(update_activity)
        state.knife.lifetime.count.add_(1.0)

    # =========================================================================
    # summary() -- public entry point #1
    #
    # Called periodically by the training loop. Computes and returns
    # diagnostics from the state accumulated in the section above, at the
    # cadence configured in PlasticityConfig (log_interval / rank_interval /
    # knife_interval).
    #
    # Call tree, in the order methods appear below:
    #   summary()
    #     _summary_schedule()             -- decide what fires this call
    #     _summarize_site()  (per site)
    #       _summarize_recent_activity()    (+ _rank_metrics() if should_rank)
    #       _summarize_lifetime_activity()
    #       _summarize_redo()
    #       _summarize_utility()
    #       _summarize_knife()             (only if should_knife)
    #         _summarize_knife_average()
    #     _add_network_summary_metrics()  -- cross-site aggregation
    #       _weighted_site_mean()
    #       _add_mean_and_global_p10()
    #       _add_rank_aggregates()
    #     _add_weight_magnitude_metrics()
    #     _reset_summary_windows()
    # =========================================================================

    @torch.no_grad()
    def summary(
        self,
        prefix: str | None = None,
        force: bool = False,
    ) -> dict[str, float]:
        """Return diagnostics at their configured reporting cadences."""
        if not self.enabled:
            return {}

        self.step_count += 1
        should_log, should_rank, should_knife = self._summary_schedule(force)
        if not should_log and not should_rank:
            return {}

        prefix = prefix or self.name
        info: dict[str, float] = {}
        site_summaries: list[SiteSummary] = []

        for site in self.sites:
            state = self.site_states.get(site.name)
            if state is None:
                continue

            summary = self._summarize_site(
                site=site,
                state=state,
                should_rank=should_rank,
                should_knife=should_knife,
            )
            site_summaries.append(summary)

            clean_name = site.name.replace(".", "_")
            for metric_name, value in summary.metrics.items():
                info[f"{prefix}/{clean_name}/{metric_name}"] = value

        self._add_network_summary_metrics(info, prefix, site_summaries)
        self._add_weight_magnitude_metrics(info, prefix)
        info[f"{prefix}/units_replaced_total"] = float(self.total_units_replaced)

        self._reset_summary_windows(should_log, should_knife)
        self.last_summary = info
        return info

    def _summary_schedule(self, force: bool) -> tuple[bool, bool, bool]:
        should_log = force or self.step_count % self.config.log_interval == 0
        should_rank = self.config.compute_rank and (
            force or self.step_count % self.config.rank_interval == 0
        )

        # KNIFE is deliberately gated as a multiple of log_interval, rather
        # than an independent step-count check like rank_interval, so its
        # accumulation window always aligns with a should_log reset (see
        # _reset_summary_windows). Otherwise the window could be logged
        # without being reset, silently reporting a stale multi-interval
        # average under the same key as a fresh one.
        log_count = self.step_count // self.config.log_interval
        should_knife = force or (
            should_log and log_count % self.config.knife_interval == 0
        )
        return should_log, should_rank, should_knife

    @torch.no_grad()
    def _summarize_site(
        self,
        site: FeatureSite,
        state: SiteState,
        should_rank: bool,
        should_knife: bool,
    ) -> SiteSummary:
        result = SiteSummary(
            num_units=float(site.producer_module.out_features),
            metrics={},
            distributions={},
        )

        self._summarize_recent_activity(result, state, should_rank)
        self._summarize_lifetime_activity(result, state)
        self._summarize_redo(result, state)
        self._summarize_utility(result, state)

        if should_knife:
            self._summarize_knife(result, state)

        return result

    @torch.no_grad()
    def _summarize_recent_activity(
        self,
        result: SiteSummary,
        state: SiteState,
        should_rank: bool,
    ) -> None:
        # dead_units_frac / rank metrics below reproduce the Fig. 2d/4b
        # diagnostics from [Dohare2024] -- see References at the top of
        # this file.
        #
        # What these mean: a unit is "dead" once it stops firing (outputs
        # <=0 on ~every input, so its ReLU-family activation gradient is
        # permanently zero) -- it can no longer learn, and neither can
        # anything downstream that depended on it. dead_units_frac rising
        # over training is the most direct, literal symptom of plasticity
        # loss: capacity in the layer is being permanently switched off.
        # active_fraction_mean/p10 are the complementary "how alive is the
        # rest of the layer" view -- p10 in particular flags a long tail of
        # barely-firing units before they've fully died.
        activity_window = self._activation_window_tensor(state)
        if activity_window is None or activity_window.shape[0] == 0:
            return

        active_fraction = (activity_window > 0.0).float().mean(dim=0)
        result.metrics.update(
            {
                "dead_units_frac": ((active_fraction < 0.01).float().mean().item()),
                "active_fraction_mean": active_fraction.mean().item(),
                "active_fraction_p10": torch.quantile(active_fraction, 0.10).item(),
                "activity_window_size": float(activity_window.shape[0]),
            }
        )

        if should_rank:
            result.metrics.update(
                self._rank_metrics(activity_window, num_units=result.num_units)
            )

    @torch.no_grad()
    def _rank_metrics(
        self, activation: torch.Tensor, num_units: float | None = None
    ) -> dict[str, float]:
        # Stable rank / effective rank, per [Dohare2024] Fig. 2d & 4b.
        #
        # What this means: a healthy layer's units respond somewhat
        # independently to different inputs, so its batch of activations
        # spans most of its available dimensions (high rank). As plasticity
        # is lost, units collapse toward a handful of shared response
        # patterns -- effectively becoming redundant copies of each other --
        # so the activation matrix's rank shrinks well below its unit count
        # even before those units go fully dead. A falling stable/effective
        # rank is an earlier, softer warning sign than dead_units_frac: it
        # catches representational collapse in units that are still firing,
        # just no longer firing usefully.
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
            metrics = {
                "stable_rank": 0.0,
                "effective_rank": 0.0,
            }
            if num_units:
                metrics["stable_rank_pct"] = 0.0
            return metrics

        cumulative_ratio = torch.cumsum(singular_values, dim=0) / singular_sum
        stable_rank = float(torch.searchsorted(cumulative_ratio, 0.99).item() + 1)
        stable_rank = min(stable_rank, float(singular_values.numel()))

        probs = singular_values / singular_sum.clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        effective_rank = torch.exp(entropy).item()

        metrics = {
            "stable_rank": stable_rank,
            "effective_rank": effective_rank,
        }

        # [Dohare2024] (Fig. 2d, Fig. 4b) plots stable rank scaled to the
        # layer's maximum possible rank (its unit count), i.e. 0-100. This
        # denominator is the layer width, not min(n_samples, n_units) from
        # the SVD itself -- it's "how much of this layer's capacity is
        # actually being used," not "how much of the sampled batch's rank."
        if num_units:
            metrics["stable_rank_pct"] = 100.0 * stable_rank / num_units

        return metrics

    @torch.no_grad()
    def _summarize_lifetime_activity(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        # Same dead-unit definition as _summarize_recent_activity [Dohare2024],
        # applied to the lifetime running average instead of the log window.
        active_fraction = (
            state.activity.lifetime.total / state.activity.lifetime.count.clamp_min(1.0)
        )
        result.metrics.update(
            {
                "dead_units_lifetime_frac": (
                    (active_fraction < 0.01).float().mean().item()
                ),
                "active_lifetime_fraction_mean": active_fraction.mean().item(),
            }
        )

    @torch.no_grad()
    def _summarize_redo(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        # ReDo dormancy score, per [Sokar2023]: a unit's normalized
        # activation magnitude relative to its layer's mean. A unit is
        # dormant when this score falls at or below dormant_threshold
        # (the paper's default is 0.1).
        #
        # What this means: this is a softer, magnitude-based precursor to
        # dead_units_frac above. dead_units_frac only catches units that
        # have gone fully silent; a unit can still fire above zero but at
        # near-negligible magnitude compared to its layer -- barely
        # contributing to anything downstream, and likely to go fully dead
        # soon. redo_dormant_units_frac catches those "going, going, gone"
        # units earlier. redo_activity_frac_mean is the fraction of time a
        # unit's activation exceeds a fixed threshold at all, independent of
        # relative magnitude -- a second, complementary view of the same
        # underlying decline.
        activation_abs = state.redo.activation_abs_ema
        dormancy_score = activation_abs / activation_abs.mean().clamp_min(1e-12)
        result.metrics.update(
            {
                "redo_dormant_units_frac": (
                    (dormancy_score <= self.config.dormant_threshold)
                    .float()
                    .mean()
                    .item()
                ),
                "redo_dormancy_score_mean": dormancy_score.mean().item(),
                "redo_dormancy_score_p10": torch.quantile(dormancy_score, 0.10).item(),
                "redo_activity_frac_mean": (
                    state.redo.activity_fraction_ema.mean().item()
                ),
            }
        )

    @torch.no_grad()
    def _summarize_utility(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        # Contribution utility, per [Dohare2024]'s CBP/GnT algorithm: a
        # bias-corrected EMA of |output weight| * |mean feature activation|,
        # used both for logging here and to rank units for replacement in
        # _select_units_by_cbp_utility below.
        #
        # What this means: unlike the metrics above (which describe symptoms
        # of plasticity loss), this one is the mechanism CBP uses to *treat*
        # it. A unit with low contribution utility is one whose output barely
        # moves the network's predictions -- either because it rarely fires,
        # or because the layer after it has learned to ignore it. Such units
        # are "dead weight": still consuming capacity but not contributing
        # useful representational diversity, and prime candidates for reset
        # (see step_replacement()). contribution_utility_p10/min falling
        # toward zero over training is a leading indicator that an
        # increasing share of the layer has stopped pulling its weight, even
        # before dead_units_frac or the rank metrics show a problem.
        utility = state.utility.bias_corrected
        result.metrics.update(
            {
                "contribution_utility_mean": utility.mean().item(),
                "contribution_utility_p10": torch.quantile(utility, 0.10).item(),
                "contribution_utility_min": utility.min().item(),
            }
        )
        result.distributions["contribution_utility"] = utility.detach().flatten()

    @torch.no_grad()
    def _summarize_knife(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        # KNIFE relative update activity (RUA), per [Liu2026]. See
        # _summarize_knife_average for the per-window computation.
        self._summarize_knife_average(state.knife.window, result, lifetime=False)
        self._summarize_knife_average(state.knife.lifetime, result, lifetime=True)

    @torch.no_grad()
    def _summarize_knife_average(
        self,
        average: RunningAverageState,
        result: SiteSummary,
        lifetime: bool,
    ) -> None:
        # [Liu2026]: update activity UA_i = ||grad_i|| / (||weight_i|| + eps),
        # time-averaged, then normalized by the layer's mean UA to get RUA_i.
        # A unit is stagnant when RUA_i falls below stagnant_threshold
        # (paper default 0.25) and volatile when it exceeds volatile_threshold.
        #
        # What this means: this is a gradient-side view of plasticity loss,
        # complementary to the activation-side metrics above (dead units,
        # ReDo, rank). A unit can still be firing normally yet be stagnant:
        # its weights have stopped meaningfully updating relative to their
        # own magnitude, so gradient descent is no longer changing what it
        # computes -- learning has effectively frozen for that unit even
        # though it hasn't gone dark. Volatile is the opposite failure mode:
        # a unit whose updates are disproportionately large relative to its
        # weight, indicating unstable, possibly non-converging learning
        # rather than a healthy steady state. Rising knife_stagnant_units_frac
        # over training says "an increasing share of this layer has stopped
        # learning, even though it still looks active by every other metric
        # here" -- this is the failure mode the other metrics can miss.
        update_activity = average.total / average.count.clamp_min(1.0)
        if not torch.any(update_activity > 0):
            return

        rua = update_activity / update_activity.mean().clamp_min(1e-12)
        suffix = "_lifetime" if lifetime else ""
        result.metrics.update(
            {
                f"knife_update_activity{suffix}_mean": (update_activity.mean().item()),
                f"knife_rua{suffix}_mean": rua.mean().item(),
                f"knife_rua{suffix}_p10": torch.quantile(rua, 0.10).item(),
                f"knife_stagnant_units{suffix}_frac": (
                    (rua < self.config.stagnant_threshold).float().mean().item()
                ),
                f"knife_volatile_units{suffix}_frac": (
                    (rua > self.config.volatile_threshold).float().mean().item()
                ),
            }
        )
        result.distributions[f"knife_rua{suffix}"] = rua.detach().flatten()

    @torch.no_grad()
    def _add_network_summary_metrics(
        self,
        info: dict[str, float],
        prefix: str,
        site_summaries: list[SiteSummary],
    ) -> None:
        if not site_summaries:
            return

        weighted_metrics = (
            "dead_units_frac",
            "active_fraction_mean",
            "dead_units_lifetime_frac",
            "active_lifetime_fraction_mean",
            "redo_dormant_units_frac",
            "redo_activity_frac_mean",
            "contribution_utility_mean",
            "knife_update_activity_mean",
            "knife_stagnant_units_frac",
            "knife_volatile_units_frac",
            "knife_update_activity_lifetime_mean",
            "knife_stagnant_units_lifetime_frac",
            "knife_volatile_units_lifetime_frac",
        )
        for metric_name in weighted_metrics:
            value = self._weighted_site_mean(site_summaries, metric_name)
            if value is not None:
                info[f"{prefix}/{metric_name}"] = value

        self._add_mean_and_global_p10(
            info,
            prefix,
            site_summaries,
            metric_name="contribution_utility_p10",
            distribution_name="contribution_utility",
        )
        self._add_mean_and_global_p10(
            info,
            prefix,
            site_summaries,
            metric_name="knife_rua_p10",
            distribution_name="knife_rua",
        )
        self._add_mean_and_global_p10(
            info,
            prefix,
            site_summaries,
            metric_name="knife_rua_lifetime_p10",
            distribution_name="knife_rua_lifetime",
        )
        self._add_rank_aggregates(info, prefix, site_summaries)

    @staticmethod
    def _weighted_site_mean(
        site_summaries: list[SiteSummary],
        metric_name: str,
    ) -> float | None:
        total_units = sum(summary.num_units for summary in site_summaries)
        if total_units <= 0:
            return None

        weighted_sum = 0.0
        found_value = False
        for summary in site_summaries:
            value = summary.metrics.get(metric_name)
            if value is None:
                continue
            weighted_sum += value * summary.num_units
            found_value = True

        return weighted_sum / total_units if found_value else None

    @staticmethod
    def _add_mean_and_global_p10(
        info: dict[str, float],
        prefix: str,
        site_summaries: list[SiteSummary],
        metric_name: str,
        distribution_name: str,
    ) -> None:
        metric_values = [
            summary.metrics[metric_name]
            for summary in site_summaries
            if metric_name in summary.metrics
        ]
        distributions = [
            summary.distributions[distribution_name]
            for summary in site_summaries
            if distribution_name in summary.distributions
        ]
        if not metric_values or not distributions:
            return

        info[f"{prefix}/{metric_name}_mean"] = sum(metric_values) / len(metric_values)
        info[f"{prefix}/{metric_name}_global"] = torch.quantile(
            torch.cat(distributions), 0.10
        ).item()

    @staticmethod
    def _add_rank_aggregates(
        info: dict[str, float],
        prefix: str,
        site_summaries: list[SiteSummary],
    ) -> None:
        for metric_name in (
            "stable_rank",
            "stable_rank_pct",
            "effective_rank",
        ):
            values = [
                summary.metrics[metric_name]
                for summary in site_summaries
                if metric_name in summary.metrics
            ]
            if values:
                info[f"{prefix}/{metric_name}_mean"] = sum(values) / len(values)
                info[f"{prefix}/{metric_name}_final_layer"] = values[-1]

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
    def _reset_summary_windows(
        self,
        should_log: bool,
        should_knife: bool,
    ) -> None:
        if not self.model.training:
            return

        for state in self.site_states.values():
            if should_log:
                state.activity.window.total.zero_()
                state.activity.window.count.zero_()
                state.activity.behaviour_chunks.clear()
                state.activity.behaviour_rows = 0

            if should_knife:
                state.knife.window.total.zero_()
                state.knife.window.count.zero_()

    # =========================================================================
    # step_replacement() -- public entry point #2
    #
    # Called periodically by the training loop to reset low-utility units
    # (CBP/GnT-style generate-and-test). Independent of summary() above --
    # it reads state.utility but doesn't depend on any summary() call having
    # happened first.
    #
    # Call tree:
    #   step_replacement()
    #     _select_units_for_replacement()
    #       _select_units_by_cbp_utility()
    #     _reset_unit()  (per selected unit)
    #       _reset_linear_output_unit()
    #         _initialization_bound()
    #       _reset_optimizer_state_for_unit()
    #         _zero_optimizer_state_slice()
    # =========================================================================

    @torch.no_grad()
    def step_replacement(self) -> dict[str, float]:
        if not self.enabled or not self.config.replacement_enabled:
            return {}

        if self.replacement_strategy != "cbp":
            raise NotImplementedError(
                f"replacement_strategy={self.replacement_strategy!r} is not "
                "implemented yet. Currently supported: 'cbp'"
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
        if self.replacement_strategy == "cbp":
            return self._select_units_by_cbp_utility(site)

        raise NotImplementedError(
            f"replacement_strategy={self.replacement_strategy!r} is not implemented."
        )

    @torch.no_grad()
    def _select_units_by_cbp_utility(self, site: FeatureSite) -> torch.Tensor:
        # Continual Backprop / Generate-and-Test unit selection, per
        # [Dohare2024]: replace the lowest-contribution-utility units among
        # those past maturity_threshold, at replacement_rate per step.
        state = self.site_states.get(site.name)
        if state is None:
            return torch.empty(
                0, dtype=torch.long, device=site.producer_module.weight.device
            )

        age = state.utility.age
        utility = state.utility.bias_corrected

        eligible_indices = torch.where(age > self.config.maturity_threshold)[0]

        if eligible_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=utility.device)

        expected_replacements = self.config.replacement_rate * eligible_indices.numel()

        if self.config.replacement_accumulate:
            # Algorithm-1 style deterministic accumulator: same long-run expected
            # replacement rate, but lower step-to-step variance and possible multi-unit
            # replacement when the accumulator exceeds 1.
            accumulator = state.utility.replacement_accumulator
            accumulator.add_(expected_replacements)
            num_replace = int(accumulator.item())
            accumulator.sub_(float(num_replace))
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

        state = self.site_states[site.name]

        if consumer.bias is not None:
            decay = self.config.utility_decay
            age = state.utility.age[unit_idx]
            bias_correction = 1.0 - decay ** age.item()
            bias_correction = max(bias_correction, 1e-12)

            bias_corrected_mean_feature = (
                state.utility.mean_feature_activation[unit_idx] / bias_correction
            )

            consumer.bias.data += (
                consumer.weight.data[:, unit_idx] * bias_corrected_mean_feature
            )

        consumer.weight.data[:, unit_idx] = 0.0

        self._reset_linear_output_unit(producer, unit_idx)

        state.utility.ema[unit_idx] = 0.0
        state.utility.bias_corrected[unit_idx] = 0.0
        state.utility.mean_feature_activation[unit_idx] = 0.0
        state.utility.age[unit_idx] = 0.0

        state.activity.window.total[unit_idx] = 0.0
        state.activity.window.count[unit_idx] = 0.0
        state.activity.lifetime.total[unit_idx] = 0.0
        state.activity.lifetime.count[unit_idx] = 0.0

        state.redo.activation_abs_ema[unit_idx] = 0.0
        state.redo.activity_fraction_ema[unit_idx] = 0.0

        state.knife.window.total[unit_idx] = 0.0
        state.knife.window.count[unit_idx] = 0.0
        state.knife.lifetime.total[unit_idx] = 0.0
        state.knife.lifetime.count[unit_idx] = 0.0

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
