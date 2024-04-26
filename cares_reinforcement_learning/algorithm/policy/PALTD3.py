"""
Original Paper: https://arxiv.org/abs/2007.06049
"""

import copy
import logging
import os
import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as helpers
from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


class PALTD3:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        tau: float,
        per_alpha: float,
        min_priority: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)

        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        self.gamma = gamma
        self.tau = tau

        self.per_alpha = per_alpha
        self.min_priority = min_priority

        self.noise_clip = 0.5
        self.policy_noise = 0.2

        self.learn_counter = 0
        self.policy_update_freq = 2

        self.action_num = action_num

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0.1
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor_net.train()
        return action

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        pal_loss_one = helpers.prioritized_approximate_loss(
            td_error_one, self.min_priority, self.per_alpha
        )
        pal_loss_two = helpers.prioritized_approximate_loss(
            td_error_two, self.min_priority, self.per_alpha
        )
        critic_loss_total = pal_loss_one + pal_loss_two
        critic_loss_total /= (
            torch.max(td_error_one, td_error_two)
            .clamp(min=self.min_priority)
            .pow(self.per_alpha)
            .mean()
            .detach()
        )

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        if self.learn_counter % self.policy_update_freq == 0:
            # actor_q_one, actor_q_two = self.critic_net(states, self.actor_net(states))
            # actor_q_values = torch.minimum(actor_q_one, actor_q_two)

            # Update Actor
            actor_q_values, _ = self.critic_net(states, self.actor_net(states))
            actor_loss = -actor_q_values.mean()

            self.actor_net_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net_optimiser.step()

            # Update target network params
            for target_param, param in zip(
                self.target_critic_net.Q1.parameters(), self.critic_net.Q1.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

            for target_param, param in zip(
                self.target_critic_net.Q2.parameters(), self.critic_net.Q2.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

            for target_param, param in zip(
                self.target_actor_net.parameters(), self.actor_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
