from dataclasses import replace
import logging
import os
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F

from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import LSDConfig
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import memory_sampler
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.LSD.encoder import Encoder
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation


class LSD(SARLAlgorithm[np.ndarray]):
    def __init__(
        self,
        skills_agent: SAC,
        encoder: Encoder,
        config: LSDConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="usd", config=config, device=device)

        self.skills_agent = skills_agent
        self.encoder_net = encoder.to(device)

        self.skill_dim = config.skill_dim # num of discrete skills or dimensions for continuouse skill
        self.is_discrete = config.is_discrete
        self.updates_per_iteration = config.updates_per_iteration

        # set skill prior
        if self.is_discrete:
            self.p_z = np.full(
                self.skill_dim, 1.0 / self.skill_dim, dtype=np.float32
            )  # (K,)
        else:
            self.p_z = None # ignore as using standard normal built into np

        # sample first skill
        self._sample_new_z()
        
        # create state encoder optimiser
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder_net.parameters(),
            lr=config.encoder_lr
        )
    
    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:

        observation = replace(
            observation,
            vector_state=self._concat_state_latent(observation.vector_state),
        )

        action_sample = self.skills_agent.act(observation, evaluation)
        action_sample.extras["skill"] = self.z
        return action_sample
    
    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        # skip if there are not enough samples in mem buffer
        if len(memory_buffer) < self.batch_size:
            return {}
        
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            train_data,
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # DIAYN does not use PER
        )

        batch_size = len(observation_tensor.vector_state_tensor)

        # Extract skills from the sampled transitions into tensor
        skills = [extra["skill"] for extra in train_data]
        if self.is_discrete:
            z_code_tensor= torch.tensor(skills, dtype=torch.long, device=self.device) #(B,) each entry is the number for the skill
            # Concatenate zs (skills) as one-hot to states
            # pylint: disable-next=not-callable
            skill_tensor = F.one_hot(z_code_tensor, num_classes=self.skill_dim).to(
                observation_tensor.vector_state_tensor.dtype
            ) # (B,skill_dim), one hot for chosen skill
        else:
            skill_tensor = torch.tensor(skills) #(B,skill_dim)

        # Concate 
        states_z_tensor = torch.cat(
            [observation_tensor.vector_state_tensor, skill_tensor], dim=1
        )
        next_states_z_tensor = torch.cat(
            [next_observation_tensor.vector_state_tensor, skill_tensor], dim=1
        )
        
        
        for _ in range(self.updates_per_iteration):
            # Calc loss for both sac agent and encoder net
            representation_skill_prod = self.compute_representation_skill_prod(
                observation_tensor.vector_state_tensor,
                next_observation_tensor.vector_state_tensor,
                skill_tensor
            ) #(B,)

            # update sac agent
            observation_z_tensor = replace(
                observation_tensor,
                vector_state_tensor=states_z_tensor,
            )
            next_observation_z_tensor = replace(
                next_observation_tensor,
                vector_state_tensor=next_states_z_tensor,
            )

            
            agent_info, _ = self.skills_agent.update_from_batch(
                observation_tensor=observation_z_tensor,
                actions_tensor=actions_tensor,
                rewards_tensor=representation_skill_prod.detach().unsqueeze(-1), #(B,) -> (B,1), needed for some reason
                next_observation_tensor=next_observation_z_tensor,
                dones_tensor=dones_tensor,
                weights_tensor=weights_tensor
            )

            # print(observation_z_tensor.vector_state_tensor.shape)
            # print(representation_skill_prod.detach().unsqueeze(-1).shape)
            info |= agent_info

            # Update state encoder
            representation_skill_prod_mean = representation_skill_prod.mean() #()
            encoder_loss = -representation_skill_prod_mean

            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()
        
        return info

    
    def compute_representation_skill_prod(
            self,
            state_tensor: torch.Tensor, #(B, state size)
            next_state_tensor: torch.Tensor, #(B, state size)
            skill_tensor: torch.Tensor #(B, skill_dim)
        ):
        '''
        Calculate (phi(s')-phi(s))z
        '''
        current_z = self.encoder_net(state_tensor) #(B, skill_dim)
        next_z = self.encoder_net(next_state_tensor) #(B, skill_dim)
        delta_z:torch.Tensor = next_z - current_z #(B, skill_dim)

        if self.is_discrete:
            #(B,1,skill_dim)
            delta_z = delta_z.reshape(delta_z.size(0), 1, delta_z.size(1)) 
            #(B,skill_dim,skill_dim). Batches of one hot encoding for each skill:
            # e.g. [[1,0,0],[0,1,0],[0,0,1],...]
            eye_z = torch.eye(self.skill_dim, device=self.device).reshape(1, self.skill_dim, self.skill_dim).expand(delta_z.size(0), -1, -1)
            #(B,skill_dim), effectively: each option gets a score of how much latent change align with that dimension
            logits = (eye_z * delta_z).sum(dim=2)

            #(B,skill_dim), onehot for chosen skill -> centered around 0 and scaled
            masks = (skill_tensor - skill_tensor.mean(dim=1, keepdim=True)) * (self.skill_dim) / (self.skill_dim - 1 if self.skill_dim != 1 else 1)

            return (logits * masks).sum(dim=1) #(B,)

        else:
            return (delta_z * skill_tensor).sum(dim=1) #(B,)
            


    def episode_done(self):
        '''
        Sample new skill from p(z) when episode finishes.
        '''
        self._sample_new_z()
        return super().episode_done()
    

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state = replace(
            state, vector_state=self._concat_state_latent(state.vector_state)
        )
        return self.skills_agent._calculate_value(state, action)

    ###### LATENT RELATED UTILS ############################################################
    def _concat_state_latent(self, state: np.ndarray) -> np.ndarray:
        if self.is_discrete:
            z_one_hot = np.zeros(self.skill_dim)
            z_one_hot[self.z] = 1
            return np.concatenate([state, z_one_hot])
        else:
            return np.concatenate([state, self.z])
        
    def _sample_new_z(self):
        ''' Sample new skill z following p(z)'''
        # NOTE: different z in continuous and discrete case 
        #   - Discrete: z is the number for the skill (NOT one-hot)
        #   - Continuous: z is sampled (skill_dim) shaped tensor 
        if self.is_discrete:
            self.z = np.random.choice(self.skill_dim, p=self.p_z)
        else:
            self.z = np.random.randn(self.skill_dim)


    ###### SAVING UTIL ####################################################################

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Save skills agent
        self.skills_agent.save_models(filepath, f"{filename}_skill_agent")

        # Save LSD-specific state in a single checkpoint
        checkpoint = {
            "encoder_state_dict": self.encoder_net.state_dict(),
            "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
            "z": self.z,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_lsd.pth")
        logging.info("LSD models and state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.skills_agent.load_models(filepath, f"{filename}_skill_agent")

        checkpoint = torch.load(f"{filepath}/{filename}_diayn.pth")

        self.encoder_net.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer_state_dict"]
        )

        self.z = checkpoint.get("z", self.z)
        logging.info("LSD models and state have been loaded...")

    
