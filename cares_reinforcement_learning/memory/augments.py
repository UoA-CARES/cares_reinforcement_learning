import torch


def td_error(info: dict):
    return torch.abs(info['q_target'] - info['q_values_min']).detach().cpu().numpy()


def std(info: dict):
    return [1.0] * len(info['q_target'])
