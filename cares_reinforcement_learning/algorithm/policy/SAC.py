"""
Original Paper: https://arxiv.org/abs/1812.05905
Code based on: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py.

This code runs automatic entropy tuning
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.SAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import SACConfig


class SAC:
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SACConfig,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.target_critic_net.eval()  # never in training mode - helps with batch/drop out layers

        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        self.target_entropy = -self.actor_net.num_actions

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        # Temperature (alpha) for the entropy loss
        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr
        )

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # pylint: disable-next=unused-argument

        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            if evaluation:
                (_, _, action) = self.actor_net(state_tensor)
            else:
                (action, _, _) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
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
    ) -> tuple[float, float, float]:
        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
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

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        return critic_loss_one.item(), critic_loss_two.item(), critic_loss_total.item()

    def _update_actor_alpha(self, states: torch.Tensor) -> tuple[float, float]:
        pi, log_pi, _ = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            qf1_pi, qf2_pi = self.critic_net(states, pi)

        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature (alpha)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards_tensor = rewards_tensor.unsqueeze(0).reshape(batch_size, 1)
        dones_tensor = dones_tensor.unsqueeze(0).reshape(batch_size, 1)

        info = {}

        # Update the Critic
        critic_loss_one, critic_loss_two, critic_loss_total = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["critic_loss"] = critic_loss_total

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_loss, alpha_loss = self._update_actor_alpha(states_tensor)
            info["actor_loss"] = actor_loss
            info["alpha_loss"] = alpha_loss
            info["alpha"] = self.alpha.item()

        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
