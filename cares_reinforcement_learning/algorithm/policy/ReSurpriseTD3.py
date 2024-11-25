import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from cares_reinforcement_learning.memory import ManageBuffers

# TODO no sure how to import this, the ensemble will be the same? Can I pass this form outside?
from cares_reinforcement_learning.networks.NaSATD3.EPDM import EPDM


class ReSurpriseTD3:
    def __init__(
        self,
        actor_network: nn.Module,
        critic_network: nn.Module,
        gamma: float,
        tau: float,
        ensemble_size: int,
        action_num: int,
        latent_size: int,
        intrinsic_on: bool,
        actor_lr: float,
        critic_lr: float,
        epm_lr: float,
        device: str,
    ):
        self.type = "policy"
        self.device = device

        self.gamma = gamma
        self.tau = tau

        self.noise_clip = 0.5
        self.policy_noise = 0.2

        self.ensemble_size = ensemble_size
        self.latent_size = latent_size
        self.intrinsic_on = intrinsic_on

        self.learn_counter = 0
        self.policy_update_freq = 2

        self.action_num = action_num

        self.actor = actor_network.to(device)
        self.critic = critic_network.to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)


        self.ensemble_predictive_model = nn.ModuleList()
        networks = [
            EPDM(self.latent_size, self.action_num) for _ in range(self.ensemble_size)
        ]
        self.ensemble_predictive_model.extend(networks)
        self.ensemble_predictive_model.to(self.device)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )

        
        self.epm_lr = epm_lr
        self.epm_optimizers = [
            torch.optim.Adam(
                self.ensemble_predictive_model[i].parameters(),
                lr=self.epm_lr,
                weight_decay=1e-3,
            )
            for i in range(self.ensemble_size)
        ]

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0.1
    ) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                action += noise_scale * np.random.randn(
                    self.action_num
                )  # this is part the TD3 too, add noise to the action
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor.train()
        return action

    def train_policy(self, memory: ManageBuffers, batch_size: int) -> None:
        self.actor.train()
        self.critic.train()

        self.learn_counter += 1

        experiences = memory.short_term_memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)
        # goals       = torch.FloatTensor(np.asarray(goals)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.critic_target(
                next_states, next_actions
            )
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic(states, actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()
        


        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_q_one, actor_q_two = self.critic(
                states, self.actor(states))
            actor_q_values = torch.minimum(actor_q_one, actor_q_two)
            actor_loss = -actor_q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network params
            for target_param, param in zip(
                self.critic_target.Q1.parameters(), self.critic.Q1.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

            for target_param, param in zip(
                self.critic_target.Q2.parameters(), self.critic.Q2.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

            for target_param, param in zip(
                self.actor_target.act_net.parameters(), self.actor.act_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

            # Note: the encoders in target networks are the same of main networks, so I wont update them

        # TODO split the tuple non-tuple
        # Update intrinsic models
        if self.intrinsic_on:
            self.train_predictive_model(states, actions, next_states)

    def get_intrinsic_reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> float:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            next_state_tensor = next_state_tensor.unsqueeze(0)
            action_tensor = torch.FloatTensor(action).to(self.device)
            action_tensor = action_tensor.unsqueeze(0)

            surprise_rate = self.get_surprise_rate(
                state_tensor, action_tensor, next_state_tensor
            )

        a = 1.0
        reward_surprise = surprise_rate * a


        return reward_surprise 

    def get_surprise_rate(
        self,
        state_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        next_state_tensor: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            predict_vector_set = []
            for network in self.ensemble_predictive_model:
                network.eval()
                predicted_vector = network(state_tensor, action_tensor)
                predict_vector_set.append(predicted_vector.detach().cpu().numpy())
            ensemble_vector = np.concatenate(predict_vector_set, axis=0)
            z_next_latent_prediction = np.mean(ensemble_vector, axis=0)
            z_next_latent_true = next_state_tensor.detach().cpu().numpy()[0]
            mse = (np.square(z_next_latent_prediction - z_next_latent_true)).mean()
        return mse


    def train_predictive_model(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> None:

        for predictive_network, optimizer in zip(
            self.ensemble_predictive_model, self.epm_optimizers
        ):
            predictive_network.train()
            # Get the deterministic prediction of each model
            prediction_vector = predictive_network(states, actions)
            # Calculate Loss of each model
            loss = F.mse_loss(prediction_vector, next_states)
            # Update weights and bias
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic.state_dict(), f"{path}/{filename}_critic.pht")
        torch.save(
            self.ensemble_predictive_model.state_dict(),
            f"{path}/{filename}_ensemble.pht",
        )
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
