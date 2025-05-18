"""
Ensemble of Predictive Discrete Model (EPPM)
Predict outputs  a point estimate e.g. discrete value
"""
# ============= LSTM Version 4.0  ===========================
# - Replaced MLP-based EPDM with LSTM-based EPDM (LSTMEPDM)
# - Supports sequence inputs: (batch_size, seq_len, latent/action)
# - Introduced internal LSTMNet with:
#     • 2-layer LSTM
#     • LayerNorm for stability
#     • Linear projection for output
# - Added dropout (0.2) to LSTM layers for regularization
# - Only the last LSTM output is used for latent prediction
# - Updated EPDM alias to use LSTMEPDM by default
# ===============================================================

import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp

from cares_reinforcement_learning.util.configurations import NaSATD3Config


class BaseEPDM(nn.Module):
    def __init__(self, prediction_net: nn.Module):
        super().__init__()
        self.prediction_net = prediction_net
        self.apply(hlp.weight_init)

    def forward(self, latent_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """
        Expects:
            latent_seq: (batch_size, seq_len, latent_dim)
            action_seq: (batch_size, seq_len, action_dim)
        Returns:
            predicted next_latent: (batch_size, latent_dim)
        """
        x = torch.cat([latent_seq, action_seq], dim=-1)  # (batch, seq_len, latent + action)
        out = self.prediction_net(x)  # (batch_size, latent_dim)
        return out


class LSTMEPDM(BaseEPDM):
    def __init__(self, latent_dim: int, action_dim: int, config: NaSATD3Config):
        hidden_size = 128
        input_size = latent_dim + action_dim
        num_layers = 2
        dropout = 0.2

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    batch_first=True)
                self.layer_norm = nn.LayerNorm(hidden_size)
                self.linear = nn.Linear(hidden_size, latent_dim)

            # lstm_out: (batch_size, seq_len, hidden_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)  
                lstm_out = self.layer_norm(lstm_out)
                last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
                return self.linear(last_output)  # (batch_size, latent_dim)

        prediction_net = LSTMNet()
        super().__init__(prediction_net=prediction_net)


# Final model alias
EPDM = LSTMEPDM
