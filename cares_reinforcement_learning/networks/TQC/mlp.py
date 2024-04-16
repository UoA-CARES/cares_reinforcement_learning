from torch import nn
from torch.nn import functional as F


class Mlp(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            self.add_module(f"fc{i}", fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, state):
        h = state
        for fc in self.fcs:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output
