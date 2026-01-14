"""
Original Paper:
                https://arxiv.org/abs/1707.06347
Good Explanation:
                https://www.youtube.com/watch?v=5P7I-xPq8u8
Code based on:
                https://github.com/ericyangyu/PPO-for-Beginners
                https://github.com/nikhilbarhate99/PPO-PyTorch
Update network based on:
                https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
                https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
#from torch.distributions import MultivariateNormal
from torch.distributions import Normal # add for Diagonal Gaussia

import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.networks.PPO2 import Actor, Critic
from cares_reinforcement_learning.util.configurations import PPO2Config
from cares_reinforcement_learning.util.training_context import (
    TrainingContext,
    ActionContext,
)


class PPO2(VectorAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PPO2Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.gamma = config.gamma
        self.lambda_gae = config.lambda_gae # GAE-lambda
        self.action_num = self.actor_net.num_actions
        self.device = device

        # Diagonal Gaussia, initial std ≈ 0.6
        self.log_std = torch.nn.Parameter(
            torch.full((self.action_num,), -0.5, device=self.device)
        )

        # self.actor_net_optimiser = torch.optim.Adam(
        #     self.actor_net.parameters(), lr=config.actor_lr
        # )

        # add to actor nn.parameter 
        self.actor_net_optimiser = torch.optim.Adam(
            [
                {'params': self.actor_net.parameters()},
                {'params': [self.log_std]}
            ], 
            lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.updates_per_iteration = config.updates_per_iteration
        self.eps_clip = config.eps_clip

        # remove MultivariateNormal
        # self.cov_var = torch.full(size=(self.action_num,), fill_value=0.5).to(
        #    self.device
        # )
        # self.cov_mat = torch.diag(self.cov_var)


        # for debug
        print("--- PPO2 Actor Architecture ---")
        print(self.actor_net)
        print("--- PPO2 Critic Architecture ---")
        print(self.critic_net)
        print("--- PPO2 initailed ---")

        print("--- Check log_std status ---")
        print(f"log_std resides in PPO2: {self.log_std.device}, Shape: {self.log_std.shape}")
        all_params = []
        for group in self.actor_net_optimiser.param_groups:
            all_params.extend(group['params'])
        print(f"Is log_std in Optimizer? {any(p is self.log_std for p in all_params)}")

    def _get_action_dist(self, mean: torch.Tensor):
        std = torch.exp(self.log_std)
        # using Diagonal Gaussia
        dist = torch.distributions.Normal(mean, std)
        return dist

    def _calculate_log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        self.actor_net.eval()
        with torch.no_grad():
            mean = self.actor_net(state)

            #dist = MultivariateNormal(mean, self.cov_mat)
            dist = self._get_action_dist(mean)
            #log_prob = dist.log_prob(action)
            log_prob = dist.log_prob(action).sum(axis=-1)

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
            #dist = MultivariateNormal(mean, self.cov_mat)
            dist = self._get_action_dist(mean)

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
        #dist = MultivariateNormal(mean, self.cov_mat)
        dist = self._get_action_dist(mean)
        #log_prob = dist.log_prob(action)  # shape, 5000
        log_prob = dist.log_prob(action).sum(axis=-1) # Log-prob of a multi-dimensional action is the sum of log-probs of each dimension

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

    def _calculate_rewards_to_go_episode_end(
        self,
        batch_rewards: torch.Tensor,
        batch_dones: torch.Tensor,
        batch_episode_end: torch.Tensor,
        batch_next_states: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_values = self.critic_net(batch_next_states).cpu().numpy()
        rewards = batch_rewards.cpu().numpy()
        dones = batch_dones.cpu().numpy()
        episode_ends = batch_episode_end.cpu().numpy()

        rtgs = []
        discounted_reward = 0

        for i in reversed(range(len(rewards))):
            if episode_ends[i]:
                if dones[i]:
                    discounted_reward = rewards[i]
                else:
                    # Target = r + gamma * V(s_next)
                    discounted_reward = rewards[i] + self.gamma * next_values[i]
            else:
                # Target = r + gamma * discounted_reward
                discounted_reward = rewards[i] + self.gamma * discounted_reward
            rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(rtgs, dtype=torch.float32, device=self.device).squeeze() # shape 5000
        return batch_rtgs

    def _discounted_cumulative_sum_gae(
        self, batch_deltas: torch.Tensor, batch_episode_end: torch.Tensor
    ) -> torch.Tensor:
        dis_sum: list[float] = []
        discounted_sum = 0

        for d, e in zip(reversed(batch_deltas), reversed(batch_episode_end)):
            discounted_sum = d.item() + self.gamma * self.lambda_gae * (1 - e.item()) * discounted_sum
            dis_sum.insert(0, discounted_sum)
        batch_dis_sum = torch.tensor(dis_sum, dtype=torch.float32, device=self.device)

        return batch_dis_sum

    def _GAE_calculator(
        self,
        batch_states: torch.Tensor,
        batch_reward: torch.Tensor,
        batch_next_states: torch.Tensor,
        batch_dones: torch.Tensor,
        batch_episode_end: torch.Tensor,
    ) -> torch.Tensor:
        # GAE (Generalized Advantage Estimation) calculation based on PPO paper:
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)                       --- Eq. (12)
        # A_hat_t = delta_t + (gamma * lambda) * A_hat_{t+1}                --- Eq. (11)
        #
        # Mathematically, Eq. (11) is equivalent to:
        # A_hat_t = sum_{k=0}^{T-t-1} (gamma * lambda)^k * delta_{t+k}
        advantages = 0
        with torch.no_grad():
            batch_values = self.critic_net(batch_states)
            batch_next_values = self.critic_net(batch_next_states)

        deltas = batch_reward + self.gamma * batch_next_values * (1 - batch_dones) - batch_values
        advantages = self._discounted_cumulative_sum_gae(deltas, batch_episode_end).squeeze() # shape 5000

        return advantages


    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        # pylint: disable-next=unused-argument

        memory = training_context.memory
        batch_size = training_context.batch_size

        experiences = memory.flush()
        #states, actions, rewards, next_states, dones = experiences
        states       = experiences[0]
        actions      = experiences[1]
        rewards      = experiences[2]
        next_states  = experiences[3]
        dones        = experiences[4]
        episode_end  = experiences[5] if len(experiences) > 5 else dones

        # Convert to tensors using helper method (no next_states needed for PPO, so pass dummy data)
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            #_,  # next_states not used in PPO
            next_states_tensor, # next_states for rtgs_episode_end estimated when truncated = 1
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
        # Convert episode_end to tensors
        episode_end_tensor = torch.tensor(np.asarray(episode_end), dtype=torch.long, device=self.device)
        episode_end_tensor = episode_end_tensor.reshape(len(episode_end_tensor), 1)

        log_probs_tensor = self._calculate_log_prob(states_tensor, actions_tensor)

        # compute reward to go:
        #rtgs = self._calculate_rewards_to_go(rewards_tensor, dones_tensor)
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-7)

        # new rtgs to covert dones and truncated and calculate in differnt methods
        # dones: task fail, reward = last_reward, 0 or -100, depends on tasks
        # truncated: steps reach the max_steps in tasks, reward = last_reward + gamma*critic.net(next_state)
        rtgs_episode_end = self._calculate_rewards_to_go_episode_end(rewards_tensor, dones_tensor, episode_end_tensor, next_states_tensor) # shape 5000

        # calculate advantages
        v, _ = self._evaluate_policy(states_tensor, actions_tensor)

        # for shape verfication
        # if rtgs_episode_end.shape != v.shape:
        #     logging.warning(f"check 1: rtgs_episode_end Shape Mismatch! rtgs: {rtgs_episode_end.shape}, v: {v.shape}")
        #     rtgs_episode_end = rtgs_episode_end.view_as(v)
        #advantages = rtgs.detach() - v.detach()
        #advantages = rtgs_episode_end.detach() - v.detach() # use rtgs_episode_end

        # use GAE as advantages
        advantages = self._GAE_calculator(states_tensor,
                                          rewards_tensor,
                                          next_states_tensor,
                                          dones_tensor,
                                          episode_end_tensor
                                          ) # shape 5000
        # for shape verfication
        if advantages.shape != v.shape:
            logging.warning(f"check 3: advantages Shape Mismatch! advantages: {advantages.shape}, v: {v.shape}")
            advantages = advantages.view_as(v)

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
            #critic_loss = F.mse_loss(v, rtgs)

            # for shape verfication
            if rtgs_episode_end.shape != v.shape:
                logging.warning(f"check 2: rtgs_episode_end Shape Mismatch! rtgs: {rtgs_episode_end.shape}, v: {v.shape}")
                rtgs_episode_end = rtgs_episode_end.view_as(v)

            critic_loss = F.mse_loss(v, rtgs_episode_end) # use rtgs_episode_end

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

        current_std = torch.exp(self.log_std).mean().item() # add std to log
        info["step_std"] = current_std

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "log_std": self.log_std, # add log_std to checkpoint
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])
        self.log_std.data = checkpoint["log_std"].data # load log_std on parameter

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        logging.info("models and optimisers have been loaded...")
