import torch
import torch.nn as nn
# This (tries to) implement SimBa architecture, which is taken from https://arxiv.org/pdf/2410.09754

class RunningStatsNorm(nn.Module):
    """ This keeps a running average and variance for each input to normalize observations.

    Args:
        input_dim: how many values are in the input
        epsilon: a small constant
    """

    def __init__(self, input_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_count = x.shape[0]
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)

                old_count = self.count
                new_count = old_count + batch_count
                delta = batch_mean - self.running_mean

                new_mean = self.running_mean + delta * (batch_count / new_count)

                old_sum_of_squares = self.running_var * old_count
                batch_sum_of_squares = batch_var * batch_count
                mean_correction = delta**2 * ((old_count * batch_count) / new_count)
                new_var = (old_sum_of_squares + batch_sum_of_squares + mean_correction) / new_count

                self.running_mean.copy_(new_mean)
                self.running_var.copy_(new_var)
                self.count.copy_(new_count)

        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)


class SimBaBlock(nn.Module):
    """
        This is the residual processing block.

        Args:
            hidden_dim: internal network dimensions
        """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), # This uses the paper's recommended 4 times, it is not fixed
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class SimBaNetwork(nn.Module):
    """
    This standardizes the input, converts it to the network's size, puts it through a stack of
    processing blocks, restandardizes it and converts it to the final output.

    Args:
        input_dim: input dimensions
        output_dim: output dimensions
        hidden_dim: internal network dimensions
        num_blocks: the number of processing blocks
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
    ):
        super().__init__()

        self.rsnorm = RunningStatsNorm(input_dim)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [SimBaBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rsnorm(x)
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_norm(x)
        return self.output(x)