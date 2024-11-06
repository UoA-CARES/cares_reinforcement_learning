"""
Original Paper: https://arxiv.org/pdf/2101.05982.pdf
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
from cares_reinforcement_learning.util.configurations import REDQConfig


class REDQ:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        config: REDQConfig,
        device: torch.device,
    ):
        self.type = "policy"
        self.gamma = config.gamma
        self.tau = config.tau

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        self.device = device

        self.num_sample_critics = config.num_sample_critics

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )

        self.target_entropy = -self.actor_net.num_actions

        # ------------- Ensemble of critics ------------------#
        self.ensemble_size = config.ensemble_size
        self.ensemble_critics = torch.nn.ModuleList()

        critics = [critic_network for _ in range(self.ensemble_size)]
        self.ensemble_critics.extend(critics)
        self.ensemble_critics.to(device)

        # Ensemble of target critics
        self.target_ensemble_critics = copy.deepcopy(self.ensemble_critics).to(device)

        lr_ensemble_critic = config.critic_lr
        self.ensemble_critics_optimizers = [
            torch.optim.Adam(
                self.ensemble_critics[i].parameters(), lr=lr_ensemble_critic
            )
            for i in range(self.ensemble_size)
        ]
        # -----------------------------------------#

        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

    # pylint: disable-next=unused-argument
    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # pylint: disable-next=unused-argument

        # note that when evaluating this algorithm we need to select mu as action
        # so _, _, action = self.actor_net.sample(state_tensor)
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
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _update_critics(
        self,
        idx: list[int],
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> list[float]:
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one = self.target_ensemble_critics[idx[0]](
                next_states, next_actions
            )

            target_q_values_two = self.target_ensemble_critics[idx[1]](
                next_states, next_actions
            )

            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        critic_loss_totals = []

        for critic_net, critic_net_optimiser in zip(
            self.ensemble_critics, self.ensemble_critics_optimizers
        ):
            q_values = critic_net(states, actions)

            critic_loss_total = 0.5 * F.mse_loss(q_values, q_target)

            critic_net_optimiser.zero_grad()
            critic_loss_total.backward()
            critic_net_optimiser.step()

            critic_loss_totals.append(critic_loss_total.item())

        return critic_loss_totals

    def _update_actor_alpha(
        self, idx: list[int], states: torch.Tensor
    ) -> tuple[float, float]:
        pi, log_pi, _ = self.actor_net(states)

        qf1_pi = self.target_ensemble_critics[idx[0]](states, pi)
        qf2_pi = self.target_ensemble_critics[idx[1]](states, pi)

        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
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
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # replace=False so that not picking the same idx twice
        idx = np.random.choice(
            self.ensemble_size, self.num_sample_critics, replace=False
        )

        info = {}

        # Update the Critics
        critic_loss_totals = self._update_critics(
            idx, states, actions, rewards, next_states, dones
        )
        info["critic_loss_totals"] = critic_loss_totals

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor
            actor_loss, alpha_loss = self._update_actor_alpha(idx, states)
            info["actor_loss"] = actor_loss
            info["alpha_loss"] = alpha_loss
            info["alpha"] = self.alpha.item()

        if self.learn_counter % self.target_update_freq == 0:
            # Update ensemble of target critics
            for critic_net, target_critic_net in zip(
                self.ensemble_critics, self.target_ensemble_critics
            ):
                hlp.soft_update_params(critic_net, target_critic_net, self.tau)

        return info

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(
            self.ensemble_critics.state_dict(), f"{path}/{filename}_ensemble.pht"
        )
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        actor_path = f"{path}/{filename}_actor.pht"
        ensemble_path = f"{path}/{filename}_ensemble.pht"

        self.actor_net.load_state_dict(torch.load(actor_path))
        self.ensemble_critics.load_state_dict(torch.load(ensemble_path))
        logging.info("models has been loaded...")
