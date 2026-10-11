"""Deterministic gradient, state-tracking, and CBP replacement tests.

Run from the CARES-RL environment:
    python test_plasticity_extended.py

Uses the actual NetworkPlasticityManager and PlasticityAdam implementations.
"""

import math

import torch
from cares_reinforcement_learning.algorithm.configurations import PlasticityConfig
from cares_reinforcement_learning.algorithm.plasticity_adam import PlasticityAdam
from cares_reinforcement_learning.networks.plasticity import NetworkPlasticityManager
from torch import nn


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
        assert torch.allclose(state.knife.lifetime.total, raw)
        values = manager.summary(env_step=1, force=True)
        p = "knife/0"
        close(
            values[f"{p}/knife_update_activity_mean"], 3.0, "mean KNIFE update activity"
        )
        close(values[f"{p}/knife_rua_mean"], 1.0, "mean relative update activity")
        close(values[f"{p}/knife_stagnant_units_frac"], 0.25, "stagnant fraction")
        close(values[f"{p}/knife_volatile_units_frac"], 0.25, "volatile fraction")
        close(
            values[f"{p}/knife_stagnant_units_lifetime_frac"],
            0.25,
            "lifetime stagnant fraction",
        )
        assert torch.equal(state.knife.window.count, torch.zeros(4))
        assert torch.equal(state.knife.lifetime.count, torch.ones(4))

        # Uncaptured backward must not count, even though hooks are installed.
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()
        assert torch.equal(state.knife.window.count, torch.zeros(4))
        assert torch.equal(state.knife.lifetime.count, torch.ones(4))
        print("PASS uncaptured backward does not affect KNIFE state")

        # METRICS capture must not affect training/lifetime/KNIFE statistics.
        before_lifetime = state.activity.lifetime.count.clone()
        with manager.capture_metrics():
            model(x)
        assert torch.equal(state.activity.lifetime.count, before_lifetime)
        assert torch.equal(state.knife.lifetime.count, torch.ones(4))
        print("PASS capture_metrics does not change training statistics")
    finally:
        manager.close()


def test_tracking():
    print("\n=== Repeated observation tracking ===")
    model = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0], [-1.0]]))
        model[0].bias.zero_()
    config = make_config()
    manager = NetworkPlasticityManager(
        model, torch.optim.SGD(model.parameters(), lr=0), config, name="tracking"
    )
    try:
        model.train()
        # First batch: feature 0 active, feature 1 inactive.
        with manager.capture_training():
            model(torch.ones(2, 1))
        # Second batch: the reverse, with different batch size. These are
        # averages per forward batch (not per individual sample).
        with manager.capture_training():
            model(-torch.ones(3, 1))
        state = manager.site_states[manager.sites[0].name]
        assert torch.equal(state.activity.lifetime.total, torch.ones(2))
        assert torch.equal(state.activity.lifetime.count, torch.full((2,), 2.0))
        assert torch.equal(state.activity.window.count, torch.full((2,), 2.0))
        print("PASS running totals and counts reflect two training forwards")
        values = manager.summary(env_step=1, force=True)
        close(
            values["tracking/0/active_lifetime_fraction_mean"],
            0.5,
            "lifetime activity average",
        )
        assert torch.equal(state.activity.window.count, torch.zeros(2))
        assert torch.equal(state.activity.lifetime.count, torch.full((2,), 2.0))
        print("PASS recent summary window reset, lifetime retained")

        # Third batch changes lifetime mean from 1/2 to (2/3, 1/3).
        with manager.capture_training():
            model(torch.ones(1, 1))
        values = manager.summary(env_step=2, force=True)
        close(
            values["tracking/0/active_lifetime_fraction_mean"],
            0.5,
            "lifetime mean across three batches",
        )
        assert torch.allclose(
            state.activity.lifetime.total / state.activity.lifetime.count,
            torch.tensor([2 / 3, 1 / 3]),
        )
        print("PASS per-neuron lifetime activity retained across summaries")
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
        assert model[0].bias[1].item() == 0.0
        close(model[2].bias[0].item(), 0.1, "outgoing bias compensation")
        assert torch.equal(state.utility.age, torch.tensor([1.0, 0.0, 1.0, 1.0]))
        assert state.utility.bias_corrected[1].item() == 0.0
        print("PASS only lowest-utility neuron selected; other neurons untouched")

        # Reset relevant Adam first/second moments, but not those of other units.
        checks = [
            (model[0].weight, (1, slice(None)), (0, slice(None))),
            (model[0].bias, (1,), (0,)),
            (model[2].weight, (slice(None), 1), (slice(None), 0)),
        ]
        for param, reset_slice, untouched_slice in checks:
            opt_state = optimizer.state[param]
            for name in ("exp_avg", "exp_avg_sq"):
                assert name in opt_state, f"Missing {name} for optimizer state"
                tensor = opt_state[name]
                assert torch.all(tensor[reset_slice] == 0), (name, "not reset")
                assert torch.all(tensor[untouched_slice] != 0), (
                    name,
                    "other unit changed",
                )
        print("PASS Adam moment slices cleared only for replaced feature")

        # No stored activation should remain after utility was consumed.
        assert state.last_activation is None
        print("PASS captured activation consumed after utility update")
    finally:
        manager.close()


if __name__ == "__main__":
    test_knife_and_capture()
    test_tracking()
    test_cbp_replacement()
    print("\nAll extended plasticity tests passed.")
