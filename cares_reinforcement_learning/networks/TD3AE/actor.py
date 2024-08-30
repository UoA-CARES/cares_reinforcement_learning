import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Encoder
from cares_reinforcement_learning.networks.TD3 import Actor as TD3Actor
from cares_reinforcement_learning.encoders.types import AECompositeState
from typing import Optional

class Actor(TD3Actor):
    def __init__(
        self,
        encoder: Encoder,
        num_actions: int,
        hidden_size: list[int] = None,
        info_vector_size:Optional[int] = 0
    ):
        if hidden_size is None:
            hidden_size = [1024, 1024]

        #                 fc input dim                         output dim     hidden size
        super().__init__(encoder.latent_dim + info_vector_size, num_actions, hidden_size)

        self.encoder = encoder
        self.info_vector_size = info_vector_size

        self.apply(hlp.weight_init)

    def forward(
        self, state:AECompositeState, detach_encoder: bool = False
    ) -> torch.Tensor:
        
        state_latent_list = []

        # Detach at the CNN layer to prevent backpropagation through the encoder
        # encoder should return Tensor of size (batch * latent_dim)
        state_latent_list.append(self.encoder(state['image'], detach_cnn=detach_encoder))
        
        # make sure info vector has content
        if self.info_vector_size != 0:
            # 'vector' should have size (batch * however many features)
            state_latent_list.append(state['vector'])
            # combine tensor along the "not batch" axis, i.e. result in shape (batch * whatever it is now)
            state_latent = torch.cat(state_latent_list, -1)
        
        # or just use image
        else:
            state_latent = state_latent_list[0]

        return super().forward(state_latent)
