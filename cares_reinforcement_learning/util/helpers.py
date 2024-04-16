import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def plot_reward_curve(data_reward):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x="step", y="episode_reward", title="Reward Curve")
    plt.show()


def weight_init(m):
    """
    Initialize the world model with orthogonal initializer for diversity.
    It works better than Xavier, which is commented.

    Keyword arguments:
        m -- the layer to be initialized.
    """
    if isinstance(m, torch.nn.Linear):
        #     torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def normalize_observation(observation, statistics):
    """
    This normalization is applied to world models inputs.
    Normalize the states based on the statistics from experience replay.

    Keyword arguments:
        obs -- input states
        statistics -- statistics from experience replay (no default)
    """
    return (observation - statistics["observation_mean"]) / statistics[
        "observation_std"
    ]


def denormalize_observation_delta(normalized_delta, statistics):
    """
    This denormlizing is applied to world models predicitons to restore the range of state difference between next and
    current.

    Keyword arguments:
        obs -- input states
        statistics -- statistics from experience replay (no default)
    """
    return (normalized_delta * statistics["delta_std"]) + statistics["delta_mean"]


def normalize_observation_delta(delta, statistics):
    """
    This normalization is applied to world models' target lables. The world model is predicting the difference between
    current states and next states. This normalization is applied to the deltas.

    """
    return (delta - statistics["delta_mean"]) / statistics["delta_std"]


def denormalize(action, max_action_value, min_action_value):
    """
    return action in environment range [max_action_value, min_action_value]
    """
    max_range_value = max_action_value
    min_range_value = min_action_value
    max_value_in = 1
    min_value_in = -1
    action_denorm = (action - min_value_in) * (max_range_value - min_range_value) / (
        max_value_in - min_value_in
    ) + min_range_value
    return action_denorm


def normalize(action, max_action_value, min_action_value):
    """
    return action in algorithm range [-1, +1]
    """
    max_range_value = 1
    min_range_value = -1
    max_value_in = max_action_value
    min_value_in = min_action_value
    action_norm = (action - min_value_in) * (max_range_value - min_range_value) / (
        max_value_in - min_value_in
    ) + min_range_value
    return action_norm


def compare_models(model_1, model_2):
    """
    This function helps to compare two models
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismatch found at", key_item_1[0])
            else:
                raise ValueError(
                    f"Models are not equal. {key_item_1[0]} is not equal to {key_item_2[0]}"
                )
    return models_differ == 0


def prioritized_approximate_loss(x, min_priority, alpha):
    return torch.where(
        x.abs() < min_priority,
        (min_priority**alpha) * 0.5 * x.pow(2),
        min_priority * x.abs().pow(1.0 + alpha) / (1.0 + alpha),
    ).mean()


def huber(x, min_priority):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).mean()


def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = (
        samples[:, None, None, :] - quantiles[:, :, :, None]
    )  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (
        torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
    ).mean()
    return loss
