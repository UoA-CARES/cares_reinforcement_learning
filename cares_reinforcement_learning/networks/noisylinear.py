import torch
from torch import nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
        bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        # pylint: disable-next=not-callable
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        std = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)

        self.weight_sigma.data.fill_(self.sigma_init * std)
        self.bias_sigma.data.fill_(self.sigma_init * std)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.data.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.data.copy_(epsilon_out)

        # Print the noise values
        # print(
        #     f"Weight epsilon mean: {self.weight_epsilon.mean().item():.6f}, "
        #     f"std: {self.weight_epsilon.std().item():.6f}"
        # )
        # print(
        #     f"Bias epsilon mean: {self.bias_epsilon.mean().item():.6f}, "
        #     f"std: {self.bias_epsilon.std().item():.6f}"
        # )

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        # print(f"Raw noise stats: mean {x.mean().item():.6f}, std {x.std().item():.6f}")
        return x.sign().mul(x.abs().sqrt())