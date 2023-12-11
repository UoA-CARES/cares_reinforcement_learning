import torch

# pylint: disable=unused-argument


def td_error(indices, info: dict, params: dict):
    return torch.abs(info["q_target"] - info["q_values_min"]).detach().cpu().numpy()


def std(indices, info: dict, params: dict):
    return [1.0] * len(indices)
