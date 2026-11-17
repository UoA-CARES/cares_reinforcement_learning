import math

import torch
import torch.nn.functional as F
from torch import nn


class SIMBABlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        return input_value + self.model(input_value)


def l2_normalise(
    input_value: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(input_value, ord=2, dim=dim, keepdim=True)

    return input_value / norm.clamp_min(eps)


class SIMBAV2Scaler(nn.Module):
    """
    Learnable element-wise scaler used by SimbaV2.

    The parameter itself is initialized to `scale`, while its effective
    initial value during the forward pass is `init`.
    """

    def __init__(
        self,
        size: int,
        init: float = 1.0,
        scale: float = 1.0,
    ):
        super().__init__()

        if scale == 0.0:
            raise ValueError("SIMBAV2Scaler scale must be non-zero.")

        self.scaler = nn.Parameter(
            torch.full(
                (size,),
                fill_value=scale,
                dtype=torch.float32,
            )
        )

        self.forward_scale = init / scale

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        return input_value * self.scaler * self.forward_scale


class SIMBAV2Linear(nn.Linear):
    """
    Bias-free linear layer with unit-normalized weight vectors.

    SimbaV2 weights are initialized orthogonally and projected onto
    the unit hypersphere. The algorithm must call project_weights()
    again after optimizer updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        nn.init.orthogonal_(self.weight)

        self.project_weights()

    @torch.no_grad()
    def project_weights(self) -> None:
        self.weight.copy_(
            l2_normalise(
                self.weight,
                dim=1,
            )
        )


class SIMBAV2InputEmbedding(nn.Module):
    """
    SimbaV2 hyperspherical input embedding.

    Input observations are expected to already have RSNorm applied.

    The embedding:
        input
        -> append constant coordinate
        -> L2 normalize
        -> unit-normalized linear layer
        -> learnable scaler
        -> L2 normalize
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        c_shift: float = 3.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = hidden_size

        self.c_shift = c_shift

        self.linear = SIMBAV2Linear(
            in_features=input_size + 1,
            out_features=hidden_size,
        )

        scaler_init = math.sqrt(2.0 / hidden_size)

        self.scaler = SIMBAV2Scaler(
            size=hidden_size,
            init=scaler_init,
            scale=scaler_init,
        )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        shift = torch.full(
            (*input_value.shape[:-1], 1),
            fill_value=self.c_shift,
            dtype=input_value.dtype,
            device=input_value.device,
        )

        output = torch.cat(
            (input_value, shift),
            dim=-1,
        )

        output = l2_normalise(output)

        output = self.linear(output)
        output = self.scaler(output)

        return l2_normalise(output)


class SIMBAV2Block(nn.Module):
    """
    SimbaV2 hyperspherical feature block.

    Structure:
        Linear(d -> 4d)
        -> Scaler
        -> ReLU
        -> Linear(4d -> d)
        -> L2 normalization
        -> learnable LERP with residual
        -> L2 normalization
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        expansion_ratio: int = 4,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        if expansion_ratio < 1:
            raise ValueError("expansion_ratio must be at least 1.")

        expanded_size = hidden_size * expansion_ratio

        self.linear1 = SIMBAV2Linear(
            in_features=hidden_size,
            out_features=expanded_size,
        )

        mlp_scaler_init = math.sqrt(2.0 / expanded_size)

        self.mlp_scaler = SIMBAV2Scaler(
            size=expanded_size,
            init=mlp_scaler_init,
            scale=mlp_scaler_init,
        )

        self.linear2 = SIMBAV2Linear(
            in_features=expanded_size,
            out_features=hidden_size,
        )

        self.alpha_scaler = SIMBAV2Scaler(
            size=hidden_size,
            init=1.0 / (num_blocks + 1),
            scale=1.0 / math.sqrt(hidden_size),
        )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        residual = input_value

        output = self.linear1(input_value)
        output = self.mlp_scaler(output)

        # Prevent an entirely zero vector before hyperspherical projection.
        output = F.relu(output) + 1e-8

        output = self.linear2(output)
        output = l2_normalise(output)

        output = residual + self.alpha_scaler(output - residual)

        return l2_normalise(output)


class SIMBAV2Encoder(nn.Module):
    """
    Reusable SimbaV2 feature encoder.

    RSNorm is deliberately handled outside of this module by the
    algorithm.

    Structure:
        SIMBAV2InputEmbedding
        -> SIMBAV2Block x num_blocks
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_blocks: int,
        c_shift: float = 3.0,
        expansion_ratio: int = 4,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        self.input_size = input_size
        self.output_size = hidden_size

        self.embedding = SIMBAV2InputEmbedding(
            input_size=input_size,
            hidden_size=hidden_size,
            c_shift=c_shift,
        )

        self.blocks = nn.Sequential(
            *[
                SIMBAV2Block(
                    hidden_size=hidden_size,
                    num_blocks=num_blocks,
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        output = self.embedding(input_value)
        return self.blocks(output)

    @torch.no_grad()
    def project_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, SIMBAV2Linear):
                module.project_weights()
