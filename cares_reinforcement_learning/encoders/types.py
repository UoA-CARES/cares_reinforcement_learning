from typing import TypedDict, List
import torch


class AECompositeState(TypedDict):
    observations: List[torch.Tensor]
    info: torch.Tensor
