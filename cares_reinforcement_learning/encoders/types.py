from typing import TypedDict, List
import torch


class AECompositeState(TypedDict):
    """
    Contain 'image' and 'vector', both are torch.Tensor.

        'image' should be of size: (batch x channels x img_axis_1 x img_axis_2) | for 1d 'images': (batch x channels x length)

        'vector' should be of size: (batch x length)

    CHANNELS INCLUDE STACK SIZE FOR TEMPORAL INFO. e.g. a stack of 3 RGB images have 9 channels
    """

    image: torch.Tensor
    vector: torch.Tensor
