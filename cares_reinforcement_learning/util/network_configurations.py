from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainableLayer(BaseModel):
    layer_category: Literal["trainable"] = "trainable"  # Discriminator field
    layer_type: str
    in_features: int | None = None
    out_features: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class NormLayer(BaseModel):
    layer_category: Literal["norm"] = "norm"  # Discriminator field
    layer_type: str
    in_features: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class FunctionLayer(BaseModel):
    layer_category: Literal["function"] = "function"  # Discriminator field
    layer_type: str
    params: dict[str, Any] = Field(default_factory=dict)


class FiLMLayer(BaseModel):
    layer_category: Literal["film"] = "film"  # Discriminator field


class ResidualLayer(BaseModel):
    layer_category: Literal["residual"] = "residual"  # Discriminator field
    main_layers: list[TrainableLayer | NormLayer | FunctionLayer | ResidualLayer | FiLMLayer]
    shortcut_layer: TrainableLayer | None = None
    use_padding: bool = False

class MLPConfig(BaseModel):
    layers: list[TrainableLayer | NormLayer | FunctionLayer | ResidualLayer | FiLMLayer]

