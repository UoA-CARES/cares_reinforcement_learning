from torch import nn


class DuelingNetwork(nn.Module):
    def __init__(self, observation_space_size, action_num):
        super().__init__()
        self.input_dim = observation_space_size
        self.output_dim = action_num

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, self.output_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals
