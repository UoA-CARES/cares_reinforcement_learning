"""
Original Paper: https://arxiv.org/abs/2209.00532

https://github.com/h-yamani/RD-PER-baselines/blob/main/LA3P/LA3P/Code/SAC/LA3P_SAC.py
"""

import copy
import logging
import os

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as helpers
from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


class LA3PSAC:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        tau: float,
        reward_scale: float,
        per_alpha: float,
        min_priority: float,
        prioritized_fraction: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_critic_net = copy.deepcopy(self.critic_net)  # .to(device)

        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        self.per_alpha = per_alpha
        self.min_priority = min_priority
        self.prioritized_fraction = prioritized_fraction

        self.learn_counter = 0
        self.policy_update_freq = 1

        self.target_entropy = -action_num

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0.0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net(state_tensor)
            else:
                (_, _, action) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp()

    def _update_target_network(self) -> None:
        # Update target network params
        for target_param, param in zip(
            self.target_critic_net.parameters(), self.critic_net.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

    def _train_actor(self, states: np.ndarray) -> None:
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)

        # Update Actor
        next_actions, next_log_pi, _ = self.actor_net(states)
        target_q_values_one, target_q_values_two = self.target_critic_net(
            states, next_actions
        )
        min_q_values = torch.minimum(target_q_values_one, target_q_values_two)
        actor_loss = ((self.alpha * next_log_pi) - min_q_values).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(
            self.log_alpha * (next_log_pi + self.target_entropy).detach()
        ).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def _train_critic(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        uniform_sampling: bool,
    ) -> np.ndarray:
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(len(rewards), 1)
        dones = dones.unsqueeze(0).reshape(len(dones), 1)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = (
                rewards * self.reward_scale + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        if uniform_sampling:
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
        else:
            huber_lose_one = helpers.huber(td_error_one, self.min_priority)
            huber_lose_two = helpers.huber(td_error_two, self.min_priority)
            critic_loss_total = huber_lose_one + huber_lose_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        torch.mean(critic_loss_total).backward()
        self.critic_net_optimiser.step()

        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(min=self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        return priorities

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        uniform_batch_size = int(batch_size * (1 - self.prioritized_fraction))
        priority_batch_size = int(batch_size * self.prioritized_fraction)

        policy_update = self.learn_counter % self.policy_update_freq

        ######################### UNIFORM SAMPLING #########################
        experiences = memory.sample_uniform(uniform_batch_size)
        states, actions, rewards, next_states, dones, indices = experiences

        priorities = self._train_critic(
            states,
            actions,
            rewards,
            next_states,
            dones,
            uniform_sampling=True,
        )

        memory.update_priorities(indices, priorities)

        # Train Actor
        self._train_actor(states)

        if policy_update:
            self._update_target_network()

        ######################### CRITIC PRIORITIZED SAMPLING #########################
        experiences = memory.sample_priority(batch_size, sampling="simple")
        states, actions, rewards, next_states, dones, indices, _ = experiences

        priorities = self._train_critic(
            states,
            actions,
            rewards,
            next_states,
            dones,
            uniform_sampling=False,
        )

        memory.update_priorities(indices, priorities)
        if policy_update:
            self._update_target_network()

        ######################### ACTOR PRIORITIZED SAMPLING #########################
        experiences = memory.sample_inverse_priority(priority_batch_size)
        states, actions, rewards, next_states, dones, indices, _ = experiences

        self._train_actor(states)

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
