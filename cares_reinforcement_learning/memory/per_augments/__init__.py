import torch


def td_error(info: dict):
    q_values = torch.minimum(info["q_values_one"], info["q_values_two"])
    return torch.abs(info['q_target'] - q_values).detach().cpu().numpy()
