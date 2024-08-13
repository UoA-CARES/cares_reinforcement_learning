"""
Original Paper: https://arxiv.org/pdf/1910.07207
Code based on: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer


class SACD:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        tau: float,
        reward_scale: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        target_entropy_multiplier: float,
        device: torch.device,
    ):
        self.type = "discrete_policy"
        self.device = device

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        self.learn_counter = 0
        self.policy_update_freq = 1

        # For smaller action spaces, set the multiplier to lower values (probs should be a config option)
        self.target_entropy = -np.log(1.0 / action_num) * target_entropy_multiplier

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        # Temperature (alpha) for the entropy loss
        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.action_num = action_num

    # pylint: disable-next=unused-argument
    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation:
                (_, _, action) = self.actor_net(state_tensor)
                # action = np.argmax(action_probs)
            else:
                (action, _, _) = self.actor_net(state_tensor)
                # action = np.random.choice(a=self.action_num, p=action_probs)
        self.actor_net.train()
        return action

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            _, (action_probs, log_actions_probs), _ = self.actor_net(next_states)

            qf1_next_target, qf2_next_target = self.target_critic_net(next_states)

            min_qf_next_target = action_probs * (
                torch.minimum(qf1_next_target, qf2_next_target)
                - self.alpha * log_actions_probs
            )

            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            # TODO: Investigate
            next_q_value = (
                rewards * self.reward_scale
                + (1.0 - dones) * min_qf_next_target * self.gamma
            )

        q_values_one, q_values_two = self.critic_net(states)

        gathered_q_values_one = q_values_one.gather(1, actions.long().unsqueeze(-1))
        gathered_q_values_two = q_values_two.gather(1, actions.long().unsqueeze(-1))

        critic_loss_one = F.mse_loss(gathered_q_values_one, next_q_value)
        critic_loss_two = F.mse_loss(gathered_q_values_two, next_q_value)
        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

    def _update_actor_alpha(self, states: torch.Tensor) -> None:
        _, (action_probs, log_action_probs), _ = self.actor_net(states)

        qf1_pi, qf2_pi = self.critic_net(states)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        inside_term = self.alpha * log_action_probs - min_qf_pi
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()

        new_log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature (alpha)
        alpha_loss = -(
            self.log_alpha * (new_log_action_probs + self.target_entropy).detach()
        ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # Update the Critic
        self._update_critic(states, actions, rewards, next_states, dones)

        # Update the Actor and Alpha
        self._update_actor_alpha(states)

        if self.learn_counter % self.policy_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

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
