import logging
import math
import os
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.CIC import SkillEncoder, StateEncoder, TransitionEncoder
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.algorithm.configurations import CICConfig



class CIC(SARLAlgorithm[np.ndarray]):
    def __init__(
            self,
            skills_agent: SARLAlgorithm[np.ndarray],
            skill_encoder: SkillEncoder,
            state_encoder: StateEncoder,
            transition_encoder: TransitionEncoder,
            config: CICConfig, 
            device: torch.device):
        
        super().__init__(policy_type="usd", config=config, device=device)

        # get all networks from factory, set up optimizer
        self.skills_agent = skills_agent
        self.skill_encoder = skill_encoder.to(device)
        self.state_encoder = state_encoder.to(device)
        self.transition_encoder = transition_encoder.to(device)

        self.cic_optimizer = torch.optim.Adam(
            list(skill_encoder.parameters()) +
            list(state_encoder.parameters()) +
            list(transition_encoder.parameters()),
            lr=config.cic_lr
        )

        # CIC related stuff
        self.cpc_temp = config.cpc_temp

        # APT related stuff
        self.knn_k = config.knn_k
        self.is_using_knn_avg = config.is_using_knn_avg
        self.is_using_rms = config.is_using_rms
        self.knn_clip = config.knn_clip

        self.rms = RMS(device=device) # runing moving average to smooth out knn used for entropy estimation

        # skill related setups
        self.num_skills = config.num_skills # for evaluation
        self.z_dim = config.z_dim
        self.num_steps_per_resample_skill = config.num_steps_per_resample_skill
        self.z = np.random.randn(self.z_dim).astype(np.float32)  # z ~ N(0, I)

        # bank of z to evaluate with
        rng = np.random.default_rng(100)
        self.eval_z_radius = getattr(config, "eval_z_radius", 2.0)
        self.eval_z_bank = rng.standard_normal(
            size=(self.num_skills, self.z_dim)
        ).astype(np.float32)
        norms = np.linalg.norm(self.eval_z_bank, axis=1, keepdims=True) + 1e-8
        self.eval_z_bank = (self.eval_z_bank / norms) * self.eval_z_radius
    
    #### Reward & loss calculation #########################################################
    def compute_apt_reward(self, state:torch.Tensor, next_state:torch.Tensor):
        state_latent:torch.Tensor = self.state_encoder(state)
        next_state_latent:torch.Tensor = self.state_encoder(next_state)

        b1, b2 = state_latent.size(0), next_state_latent.size(0)
        sim_matrix = torch.norm(state_latent[:, None, :].view(b1, 1, -1) - next_state_latent[None, :, :].view(1, b2, -1), dim=-1, p=2)
        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.is_using_knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.is_using_rms:
                moving_mean, moving_std = self.rms(reward)
                reward = reward / moving_std
            reward = torch.max(reward - self.knn_clip, torch.zeros_like(reward).to(self.device))  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.is_using_rms:
                moving_mean, moving_std = self.rms(reward)
                reward = reward / moving_std
            reward = torch.max(reward - self.knn_clip, torch.zeros_like(reward).to(self.device))
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        
        reward = torch.log(reward + 1.0) #?
        
        return reward.unsqueeze(-1)
    

    def compute_cpc_loss(self, state:torch.Tensor, next_state:torch.Tensor, skill:torch.Tensor):
        # compute transition representation
        state_latent = self.state_encoder(state)
        next_state_latent = self.state_encoder(next_state)
        transition_latent = self.transition_encoder(
            torch.cat([state_latent,next_state_latent], dim=1)
        )
        transition_latent = F.normalize(transition_latent, dim=1)

        # compute skill representation
        skill_latent = self.skill_encoder(skill)
        skill_latent = F.normalize(skill_latent, dim=1)

        # similarity matrix
        cov = torch.mm(transition_latent, skill_latent.T) #(b,b)
        sim = torch.exp(cov / self.cpc_temp) #(b,b), exp

        # negatives
        eps = 1e-6 # for numerical stability
        negative = sim.sum(dim=-1) # (b,) full row sum
        row_sub = row_sub = torch.full(negative.shape, math.e ** (1 / self.cpc_temp), device=self.device)
          #torch.Tensor(negative.shape, device=self.device).fill_(math.e**(1 / self.cpc_temp))#.to(negative.device)
        negative = torch.clamp(negative - row_sub, min=eps)  # clamp for numerical stability

        # positives (extract diagonal)
        positive = torch.exp(
            torch.sum(transition_latent*skill_latent, dim=-1)
            /self.cpc_temp
        ) #(b,)

        # loss
        loss = -torch.log(positive / (negative + eps)) #(b,)
        return loss
    
    #### TRAINNNNN #################################################################
    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext
    ) -> dict[str, Any]:
        
        info: dict[str, Any] = {}

        if len(memory_buffer) < self.batch_size:
            return {}

        sample_tensor, _ = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # DIAYN does not use PER
        )

        train_data = sample_tensor.train_data
        observation_tensor = sample_tensor.observation
        next_observation_tensor = sample_tensor.next_observation
        actions_tensor = sample_tensor.action
        dones_tensor = sample_tensor.done
        weights_tensor = sample_tensor.weights

    
        z_list = [extra["z"] for extra in train_data]
        z_tensor = torch.tensor(
            np.asarray(z_list), dtype=torch.float32, device=self.device
        ) # (B, z_dim)

        states_z_tensor = torch.cat(
            [observation_tensor.vector_state, z_tensor], 
            dim=1,
        )
        next_states_z_tensor = torch.cat(
            [next_observation_tensor.vector_state, z_tensor], dim=1
        )
        observation_z_tensor = replace(
            observation_tensor,
            vector_state=states_z_tensor,
        )
        next_observation_z_tensor = replace(
            next_observation_tensor,
            vector_state=next_states_z_tensor,
        )

        #---- CIC update, constrastive
        cpc_loss = self.compute_cpc_loss(observation_tensor.vector_state, next_observation_tensor.vector_state, z_tensor)
        cpc_loss = cpc_loss.mean()
        info["contrastive_loss"] = cpc_loss.item()
        self.cic_optimizer.zero_grad()
        cpc_loss.backward()
        self.cic_optimizer.step()

        #---- skill agent update
        with torch.no_grad():
            rewards_tensor = self.compute_apt_reward(observation_tensor.vector_state, next_observation_tensor.vector_state)
        agent_info, _ = hlp.update_skill_agent_from_batch(
            self.skills_agent,
            episode_context=episode_context,
            observation_tensor=observation_z_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_z_tensor,
            dones_tensor=dones_tensor,
            weights_tensor=weights_tensor,
        )
        info |= agent_info

        if episode_context.training_step % self.num_steps_per_resample_skill == 0:
            self._sample_skill()         

        return info

    #### Skill related utils ###############################################################
    def set_skill(self, skill: int, evaluation: bool = False) -> None:
        if skill < 0 or skill >= self.num_skills:
            raise ValueError(f"Skill index {skill} is out of bounds.")
        self.z = self.eval_z_bank[skill]
    
    def _sample_skill(self):
        self.z = np.random.randn(self.z_dim).astype(np.float32)

    def _concat_state_latent(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate([state, self.z])
    
    #### algorithm stuff ######################################################
    def act(
            self, observation: SARLObservation, evaluation:bool = False
    ) -> ActionSample[np.ndarray]:
        
        observation = replace(
            observation,
            vector_state = self._concat_state_latent(observation.vector_state)
        )

        action_sample = self.skills_agent.act(observation, evaluation)
        action_sample.extras["z"] = self.z.copy()

        return action_sample
    
    def episode_done(self):
        # self._sample_random_skill()
        return super().episode_done()

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state = replace(
            state, vector_state=self._concat_state_latent(state.vector_state)
        )

        return self.skills_agent._calculate_value(state, action)
    
    #### Saving & Loading Utils #############################################################################
    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        # Save skills agent and discriminator
        self.skills_agent.save_models(filepath, f"{filename}_skill_agent")

        checkpoint = {
            "skill_encoder": self.skill_encoder.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "transition_encoder": self.transition_encoder.state_dict(),
            "cic_optimizer": self.cic_optimizer.state_dict()
        }

        torch.save(checkpoint, f"{filepath}/{filename}_cic.pth")
        logging.info("models, optimisers, and CIC state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.skills_agent.load_models(filepath, f"{filename}_skill_agent")

        checkpoint = torch.load(f"{filepath}/{filename}_cic.pth")

        self.skill_encoder.load_state_dict(checkpoint["skill_encoder"])
        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.transition_encoder.load_state_dict(checkpoint["transition_encoder"])
        self.cic_optimizer.load_state_dict(checkpoint["cic_optimizer"])

        logging.info("models, optimisers, and CIC state have been loaded...")

## Running moving average util
# TODO: see if this can be moved to actual util file
class RMS(object):
    def __init__(self, device:torch.device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
