"""
Original Paper: https://arxiv.org/pdf/2101.05982.pdf
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.IDC import Actor, Critic
from cares_reinforcement_learning.util.configurations import IDCConfig

# Train all critics on the same batch and use the critic with the lowest overall td_error on the batch for the actor
# REDQ just randomly samples and uses the same critics for updating the critic and actor
# we would train all critics and use the one with the lowest td_error to update the actor.

# Track the average td_error for each critic and use the one with the lowest average td_error to update the actor
# Avergage td_error and standard deviation of td_error for each critic
# and use the one with the lowest average td_error and standard deviation of td_error to update the actor - or weighted average


class IDC(SAC):
    critic_net: Critic
    target_critic_net: Critic

    def __init__(
        self,
        actor_network: Actor,
        ensemble_critic: Critic,
        config: IDCConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=ensemble_critic,
            config=config,
            device=device,
        )

        self.ensemble_size = config.ensemble_size

        self.lr_ensemble_critic = config.critic_lr
        self.ensemble_critic_optimizers = [
            torch.optim.Adam(
                critic_net.parameters(),
                lr=self.lr_ensemble_critic,
                **config.critic_lr_params
            )
            for critic_net in self.critic_net.critics
        ]

        self.std_weight = config.std_weight

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                q_values = self.critic_net(state_tensor, action_tensor)
                q_value = q_values.mean()

        return q_value.item()

    def _choose_critic(self, scores: list[float], select_lowest: bool = False) -> int:
        if select_lowest:
            return scores.index(min(scores))

        # Bias toward lower scores
        inverted = [1 / (s + 1e-6) for s in scores]
        total = sum(inverted)
        weights = [w / total for w in inverted]

        return random.choices(range(len(scores)), weights=weights, k=1)[0]

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[list[float], list[float]]:
        scores = []
        critic_loss_totals = []

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

        for critic_id, (critic_net, target_critic, critic_net_optimiser) in enumerate(
            zip(
                self.critic_net.critics,
                self.target_critic_net.critics,
                self.ensemble_critic_optimizers,
            )
        ):
            with torch.no_grad():
                # shape (batch_size, num_critics, 1)
                target_q_values = target_critic(next_states, next_actions)

                target_q_values = target_q_values - self.alpha * next_log_pi

                q_target = rewards + self.gamma * (1 - dones) * target_q_values

            q_values = critic_net(states, actions)

            td_error = (q_values - q_target).abs()

            # Mean TD-error
            mean_td_error = td_error.mean()

            # Standard deviation of TD-error
            std_td_error = td_error.std(unbiased=False)

            score = mean_td_error + self.std_weight * std_td_error
            scores.append(score)

            critic_loss_total = 0.5 * F.mse_loss(q_values, q_target)

            critic_net_optimiser.zero_grad()
            critic_loss_total.backward()
            critic_net_optimiser.step()

            critic_loss_totals.append(critic_loss_total.item())

        return critic_loss_totals, scores

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_actor_alpha(  # type: ignore[override]
        self,
        states: torch.Tensor,
        best_critic_id: int,
    ) -> tuple[float, float]:
        pi, log_pi, _ = self.actor_net(states)

        qf_pi = self.target_critic_net.critics[best_critic_id](states, pi)

        actor_loss = ((self.alpha * log_pi) - qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item()

    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
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
        rewards_tensor = rewards_tensor.reshape(batch_size, 1)
        dones_tensor = dones_tensor.reshape(batch_size, 1)

        info: dict[str, Any] = {}

        # Update the Critics
        critic_loss_totals, scores = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        info["critic_loss_totals"] = critic_loss_totals
        info["scores"] = scores

        critic_id = self._choose_critic(scores, select_lowest=False)

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor
            actor_loss, alpha_loss = self._update_actor_alpha(states_tensor, critic_id)
            info["actor_loss"] = actor_loss
            info["alpha_loss"] = alpha_loss
            info["alpha"] = self.alpha.item()

        if self.learn_counter % self.target_update_freq == 0:
            # Update ensemble of target critics
            for critic_net, target_critic_net in zip(
                self.critic_net.critics, self.target_critic_net.critics
            ):
                hlp.soft_update_params(critic_net, target_critic_net, self.tau)

        return info
