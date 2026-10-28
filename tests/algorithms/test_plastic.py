import math

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import PlasticityConfig
from cares_reinforcement_learning.algorithm.plasticity_adam import PlasticityAdam
from cares_reinforcement_learning.networks.plasticity import NetworkPlasticityManager


def close(actual, expected, description, atol=1e-5):
    assert math.isclose(
        float(actual), float(expected), rel_tol=1e-5, abs_tol=atol
    ), f"{description}: expected {expected}, got {actual}"
    print(f"PASS {description}: {float(actual):.6g}")


def make_model():
    model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.fill_(1.0)
        model[2].weight.copy_(torch.tensor([[0.0, 1.0, 2.0, 9.0]]))
        model[2].bias.zero_()
    return model


def make_config(**overrides):
    config = PlasticityConfig(
        enabled=True,
        replacement_enabled=False,
        rank_enabled=True,
        activity_window_env_steps=128,
    )
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AssertionError(f"PlasticityConfig has no field {key!r}")
        setattr(config, key, value)
    return config


def test_knife_and_capture():
    print("\n=== KNIFE gradients and capture separation ===")
    model = make_model()
    config = make_config(knife_stagnant_threshold=0.25, knife_volatile_threshold=2.0)
    manager = NetworkPlasticityManager(
        model, torch.optim.SGD(model.parameters(), lr=0.0), config, name="knife"
    )
    try:
        site = manager.sites[0]
        # With x=0, each producer weight row has norm 1, but its gradient would
        # be zero. Instead use x=1: each weight gradient is [0,1,2,9].
        x = torch.ones(1, 1)
        model.train()
        with manager.capture_training():
            model(x).sum().backward()
        state = manager.site_states[site.name]
        raw = torch.tensor([0.0, 1.0, 2.0, 9.0])
        assert torch.allclose(state.knife.window.total, raw)
        values = manager.summary(env_step=1, force=True)
        p = "knife/0"
        close(
            values[f"{p}/knife_update_activity_mean"], 3.0, "mean KNIFE update activity"
        )
        close(values[f"{p}/knife_rua_mean"], 1.0, "mean relative update activity")
        close(values[f"{p}/knife_stagnant_units_frac"], 0.25, "stagnant fraction")
        close(values[f"{p}/knife_volatile_units_frac"], 0.25, "volatile fraction")
        assert not hasattr(
            state.knife, "lifetime"
        ), "KNIFE lifetime state must be absent"
        assert not any(
            "lifetime" in key for key in values
        ), "KNIFE lifetime metrics should no longer be emitted"
        assert torch.equal(state.knife.window.total, torch.zeros(4))
        assert torch.equal(state.knife.window.count, torch.zeros(4))

        # Uncaptured backward must not count, even though hooks are installed.
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()
        assert torch.equal(state.knife.window.count, torch.zeros(4))
        print("PASS uncaptured backward does not affect KNIFE state")

        model.zero_grad(set_to_none=True)

        with manager.capture_metrics():
            model(x).sum().backward()

        assert state.activity.behaviour_rows == 1

        print("PASS capture_metrics does not change KNIFE statistics")

        # A second captured interval must use only new gradients, not the
        # previous interval's values (and must not report lifetime metrics).
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model[2].weight.copy_(torch.tensor([[9.0, 2.0, 1.0, 0.0]]))
        with manager.capture_training():
            model(x).sum().backward()
        assert torch.allclose(
            state.knife.window.total, torch.tensor([9.0, 2.0, 1.0, 0.0])
        )
        assert torch.equal(state.knife.window.count, torch.ones(4))
        second = manager.summary(env_step=2, force=True)
        close(second[f"{p}/knife_update_activity_mean"], 3.0, "second interval UA")
        close(second[f"{p}/knife_rua_mean"], 1.0, "second interval RUA")
        close(
            second[f"{p}/knife_stagnant_units_frac"], 0.25, "second interval stagnant"
        )
        close(
            second[f"{p}/knife_volatile_units_frac"], 0.25, "second interval volatile"
        )
        assert not any("lifetime" in key for key in second)
        assert torch.equal(state.knife.window.total, torch.zeros(4))
        assert torch.equal(state.knife.window.count, torch.zeros(4))
        print("PASS KNIFE summaries use independent intervals without lifetime state")
    finally:
        manager.close()


def test_cbp_replacement():
    print("\n=== CBP selection, replacement, optimizer-state reset ===")
    torch.manual_seed(42)
    model = make_model()
    optimizer = PlasticityAdam(model.parameters(), lr=0.01)
    # Populate Adam moments for every parameter. Afterwards restore known
    # weights so replacement selection is completely deterministic.
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.fill_(1.0)
        model[2].weight.copy_(torch.tensor([[1.0, 0.1, 2.0, 3.0]]))
        model[2].bias.zero_()

    config = make_config(
        replacement_enabled=True,
        replacement_strategy="cbp",
        maturity_threshold_updates=0,
        replacement_rate=0.25,
        replacement_accumulate=True,
    )
    manager = NetworkPlasticityManager(model, optimizer, config, name="cbp")
    try:
        model.train()
        with manager.capture_training():
            model(torch.zeros(2, 1))  # ReLU outputs exactly 1 for every unit.
        producer_before = model[0].weight.detach().clone()
        consumer_before = model[2].weight.detach().clone()
        result = manager.step_replacement()
        close(result["cbp/units_replaced"], 1, "number of replacements")
        close(result["cbp/units_replaced_total"], 1, "cumulative replacements")
        state = manager.site_states[manager.sites[0].name]
        # The second feature has unique lowest instantaneous utility (0.1).
        assert torch.equal(model[2].weight[0, 1].detach(), torch.tensor(0.0))
        assert torch.equal(
            model[2].weight[0, [0, 2, 3]].detach(), consumer_before[0, [0, 2, 3]]
        )
        assert torch.equal(
            model[0].weight[[0, 2, 3]].detach(), producer_before[[0, 2, 3]]
        )
        # The selected feature must receive newly initialised incoming weights.
        # The original row contains 1.0; replacement samples from a smaller range.
        assert not torch.equal(
            model[0].weight[1].detach(), producer_before[1]
        ), "Replaced neuron incoming weights were not reinitialised"
        assert model[0].bias[1].item() == 0.0
        close(model[2].bias[0].item(), 0.1, "outgoing bias compensation")
        assert torch.equal(state.utility.age, torch.tensor([1.0, 0.0, 1.0, 1.0]))
        assert state.utility.bias_corrected[1].item() == 0.0
        print("PASS only lowest-utility neuron selected; other neurons untouched")

        # Reset per-element Adam step counters and moments, leaving other units intact.
        checks = [
            (model[0].weight, (1, slice(None)), (0, slice(None))),
            (model[0].bias, (1,), (0,)),
            (model[2].weight, (slice(None), 1), (slice(None), 0)),
        ]
        for param, reset_slice, untouched_slice in checks:
            opt_state = optimizer.state[param]
            for name in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name == "max_exp_avg_sq" and name not in opt_state:
                    continue  # Only present if AMSGrad is enabled.
                assert name in opt_state, f"Missing {name} for optimizer state"
                tensor = opt_state[name]
                assert torch.all(tensor[reset_slice] == 0), (name, "not reset")
                assert torch.all(tensor[untouched_slice] != 0), (
                    name,
                    "other unit changed",
                )
        print("PASS Adam step and moment slices cleared only for replaced feature")

        # No stored activation should remain after utility was consumed.
        assert state.last_activation is None
        print("PASS captured activation consumed after utility update")
    finally:
        manager.close()


def test_metric_formulations():
    """Numerical regression checks for existing formulas and emitted metric keys."""
    print("\n=== Existing metric formulations ===")
    model = make_model()
    manager = NetworkPlasticityManager(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        make_config(redo_dormant_threshold=0.1),
        name="metrics",
    )
    try:
        site = manager.sites[0]
        state = manager._ensure_site_state(site, 4, torch.device("cpu"))
        # Rows are feature vectors; known singular values [4, 2, 1, 0].
        features = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.0]))
        with manager.capture_metrics():
            model(torch.ones(1, 1))
        state.activity.behaviour_chunks = [features]
        state.activity.behaviour_rows = 4
        state.redo.activation_abs_ema.copy_(torch.tensor([0.0, 1.0, 2.0, 9.0]))
        state.redo.activity_fraction_ema.copy_(torch.tensor([0.0, 0.5, 1.0, 1.0]))

        values = manager.summary(env_step=1, force=True)
        prefix = "metrics/0"
        close(values[f"{prefix}/dead_units_frac"], 0.25, "frequency-dead fraction")
        close(
            values[f"{prefix}/active_fraction_mean"],
            0.1875,
            "sample activation fraction",
        )
        close(
            values[f"{prefix}/redo_dormant_units_frac"], 0.25, "ReDo dormant fraction"
        )
        close(
            values[f"{prefix}/redo_dormancy_score_p10"], 0.1, "ReDo relative score p10"
        )
        close(
            values[f"{prefix}/redo_activity_frac_mean"],
            0.625,
            "ReDo threshold activity",
        )
        close(values[f"{prefix}/stable_rank"], 3.0, "99%-cumulative rank")
        close(values[f"{prefix}/stable_rank_pct"], 75.0, "rank normalized by width")
        expected_erank = torch.exp(
            -(
                torch.tensor([4 / 7, 2 / 7, 1 / 7])
                * torch.log(torch.tensor([4 / 7, 2 / 7, 1 / 7]))
            ).sum()
        ).item()
        close(
            values[f"{prefix}/effective_rank"], expected_erank, "entropy effective rank"
        )
        close(values[f"{prefix}/contribution_utility_mean"], 0.0, "initial CBP utility")
        close(
            values["metrics/average_weight_magnitude"],
            2.0,
            "parameter-weighted mean magnitude",
        )
    finally:
        manager.close()


def test_rank_and_activity_independent_cadence():
    """Rank calculations can run without the general statistics interval."""
    model = make_model()
    manager = NetworkPlasticityManager(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        make_config(
            statistics_interval_env_steps=100,
            rank_interval_env_steps=5,
            knife_interval_env_steps=100,
        ),
        name="cadence",
    )
    try:
        with manager.capture_metrics():
            model(torch.tensor([[0.0], [1.0], [2.0]]))
        values = manager.summary(env_step=5)
        assert "cadence/0/stable_rank" in values
        assert "cadence/0/effective_rank" in values
        assert "cadence/0/redo_dormant_units_frac" not in values
    finally:
        manager.close()


if __name__ == "__main__":
    test_knife_and_capture()
    test_cbp_replacement()
    test_metric_formulations()
    test_rank_and_activity_independent_cadence()
    print("\nAll extended plasticity tests passed.")
