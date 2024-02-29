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
