import torch


# No priority memory buffer
def standard(info: dict, params: dict):
    return info["indices"], torch.tensor([1.0] * len(info["indices"]))


# Algorithm based priority calculation - e.g. RDTD3
def algorithm_priority(info: dict, params: dict):
    return info["indices"], info["priorities"]


# Known as PER the standard td_error based PER: https://arxiv.org/abs/1511.05952
def td_error(info: dict, params: dict):
    return (
        info["indices"],
        torch.abs(info["q_target"] - info["q_values_min"])
        .detach()
        .cpu()
        .numpy()
        .squeeze(),
    )
