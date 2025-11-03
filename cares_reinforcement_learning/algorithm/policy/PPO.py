"""
Original Paper:
                https://arxiv.org/abs/1707.06347
Good Explanation:
                https://www.youtube.com/watch?v=5P7I-xPq8u8
Code based on:
                https://github.com/ericyangyu/PPO-for-Beginners
                https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.networks.PPO import Actor, Critic
from cares_reinforcement_learning.util.configurations import PPOConfig
from cares_reinforcement_learning.util.training_context import (
    TrainingContext,
    ActionContext,
)


class PPO(VectorAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PPOConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.gamma = config.gamma
        self.action_num = self.actor_net.num_actions
        self.device = device

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.updates_per_iteration = config.updates_per_iteration
        self.eps_clip = config.eps_clip

        self.cov_var = torch.full(size=(self.action_num,), fill_value=0.5).to(
            self.device
        )
        self.cov_mat = torch.diag(self.cov_var)

    def _calculate_log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        self.actor_net.eval()
        with torch.no_grad():
            mean = self.actor_net(state)

            dist = MultivariateNormal(mean, self.cov_mat)
            log_prob = dist.log_prob(action)

        self.actor_net.train()
        return log_prob

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        self.actor_net.eval()
        state = action_context.state

        assert isinstance(state, np.ndarray)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)

            mean = self.actor_net(state_tensor)
            dist = MultivariateNormal(mean, self.cov_mat)

            # Sample an action from the distribution and get its log prob
            sample = dist.sample()

            action = sample.cpu().data.numpy().flatten()

        self.actor_net.train()

        return action

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            value = self.critic_net(state_tensor)

        return value[0].item()

    def _evaluate_policy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.critic_net(state).squeeze()  # shape 5000
        mean = self.actor_net(state)  # shape, 5000, 1
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(action)  # shape, 5000
        return v, log_prob

    def _calculate_rewards_to_go(
        self, batch_rewards: torch.Tensor, batch_dones: torch.Tensor
    ) -> torch.Tensor:
        rtgs: list[float] = []
        discounted_reward = 0
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)):
            discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
            rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(
            rtgs, dtype=torch.float32, device=self.device
        )  # shape 5000
        return batch_rtgs

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        # pylint: disable-next=unused-argument

        memory = training_context.memory
        batch_size = training_context.batch_size

        experiences = memory.flush()
        states, actions, rewards, next_states, dones = experiences

        # Convert to tensors using helper method (no next_states needed for PPO, so pass dummy data)
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            _,  # next_states not used in PPO
            dones_tensor,
            _,  # weights not needed
        ) = tu.batch_to_tensors(
            np.asarray(states),
            np.asarray(actions),
            np.asarray(rewards),
            np.asarray(next_states),
            np.asarray(dones),
            self.device,
        )

        log_probs_tensor = self._calculate_log_prob(states_tensor, actions_tensor)

        # compute reward to go:
        rtgs = self._calculate_rewards_to_go(rewards_tensor, dones_tensor)
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-7)

        # calculate advantages
        v, _ = self._evaluate_policy(states_tensor, actions_tensor)

        advantages = rtgs.detach() - v.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        td_errors = torch.abs(advantages).data.cpu().numpy()

        for _ in range(self.updates_per_iteration):
            v, curr_log_probs = self._evaluate_policy(states_tensor, actions_tensor)

            # Calculate ratios
            ratios = torch.exp(curr_log_probs - log_probs_tensor.detach())

            # Finding Surrogate Loss
            surrogate_lose_one = ratios * advantages
            surrogate_lose_two = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            actor_loss = (-torch.minimum(surrogate_lose_one, surrogate_lose_two)).mean()
            critic_loss = F.mse_loss(v, rtgs)

            self.actor_net_optimiser.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_net_optimiser.step()

            self.critic_net_optimiser.zero_grad()
            critic_loss.backward()
            self.critic_net_optimiser.step()

        info: dict[str, Any] = {}
        info["td_errors"] = td_errors
        info["critic_loss"] = critic_loss.item()
        info["actor_loss"] = actor_loss.item()

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        logging.info("models and optimisers have been loaded...")
