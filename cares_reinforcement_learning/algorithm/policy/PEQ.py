"""
Original Paper: https://arxiv.org/pdf/2101.05982.pdf
"""

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.PEQ import Actor, Critic
from cares_reinforcement_learning.util.configurations import PEQConfig

# Train all critics on the same batch and use the critic with the lowest overall td_error on the batch for the actor
# REDQ just randomly samples and uses the same critics for updating the critic and actor
# we would train all critics and use the one with the lowest td_error to update the actor.

# Track the average td_error for each critic and use the one with the lowest average td_error to update the actor
# Avergage td_error and standard deviation of td_error for each critic
# and use the one with the lowest average td_error and standard deviation of td_error to update the actor - or weighted average

# Use TQC critics


class PEQ(SAC):
    critic_net: Critic
    target_critic_net: Critic

    def __init__(
        self,
        actor_network: Actor,
        ensemble_critic: Critic,
        config: PEQConfig,
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
                **config.critic_lr_params,
            )
            for critic_net in self.critic_net.critics
        ]

        self.critic_selection_strategy = config.critic_selection_strategy

        self.w_mean = config.w_mean
        self.w_std = config.w_std
        self.w_deviation = config.w_deviation

        self.ema_alpha = config.ema_alpha
        self.critic_td_avgs: dict[int, dict[str, float]] = {
            i: {"mean": 0.0, "mean_sq": 1e-4, "std": 1e-2, "deviation": 0.0}
            for i in range(self.ensemble_size)
        }

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

    def _get_normalized_critic_stats(self) -> dict[int, dict[str, float]]:
        # Helper normalization function for a list of floats
        def normalize(lst: list[float]) -> list[float]:
            min_val = min(lst)
            max_val = max(lst)
            denom = max_val - min_val + 1e-8  # prevent division by zero
            return [(x - min_val) / denom for x in lst]

        means = [self.critic_td_avgs[i]["mean"] for i in range(self.ensemble_size)]
        stds = [self.critic_td_avgs[i]["std"] for i in range(self.ensemble_size)]
        deviations = [
            self.critic_td_avgs[i]["deviation"] for i in range(self.ensemble_size)
        ]

        norm_means = normalize(means)
        norm_stds = normalize(stds)
        norm_devs = normalize(deviations)

        return {
            i: {
                "mean": norm_means[i],
                "std": norm_stds[i],
                "deviation": norm_devs[i],
            }
            for i in range(self.ensemble_size)
        }

    def _ema_calculation(self, critic_id: int, value: float, param: str):
        self.critic_td_avgs[critic_id][param] = (
            self.ema_alpha * self.critic_td_avgs[critic_id][param]
            + (1 - self.ema_alpha) * value
        )

    def _exponential_moving_average(
        self, critic_id: int, mean: float, mean_sq: float, deviation: float
    ) -> None:
        self._ema_calculation(critic_id, mean, "mean")
        self._ema_calculation(critic_id, mean_sq, "mean_sq")
        self._ema_calculation(critic_id, deviation, "deviation")

        mean = self.critic_td_avgs[critic_id]["mean"]
        mean_sq = self.critic_td_avgs[critic_id]["mean_sq"]
        self.critic_td_avgs[critic_id]["std"] = math.sqrt(mean_sq - mean**2)

    def _choose_critic(self) -> int:
        def score(stats: dict[str, float]) -> float:
            return (
                self.w_mean * stats["mean"]
                + self.w_std * stats["std"]
                + self.w_deviation * stats["deviation"]
            )

        normalized_stats = self._get_normalized_critic_stats()
        scores = [score(stats) for stats in normalized_stats.values()]

        if self.critic_selection_strategy == "random":
            return random.randint(0, len(scores) - 1)

        if self.critic_selection_strategy == "lowest":
            return scores.index(min(scores))

        if self.critic_selection_strategy == "highest":
            return scores.index(max(scores))

        if self.critic_selection_strategy == "weighted":
            # Weigth toward lower scores
            inverted = [1 / (s + 1e-6) for s in scores]
            total = sum(inverted)
            weights = [w / total for w in inverted]

            return random.choices(range(len(scores)), weights=weights, k=1)[0]

        raise ValueError(
            f"Unknown critic selection strategy: {self.critic_selection_strategy}"
        )

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[list[float], int]:

        critic_loss_totals = []

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

        q_targets = []
        all_q_values = []
        all_td_errors = []

        for critic_id, (critic_net, target_critic, critic_net_optimiser) in enumerate(
            zip(
                self.critic_net.critics,
                self.target_critic_net.critics,
                self.ensemble_critic_optimizers,
            )
        ):
            with torch.no_grad():
                target_q_values = target_critic(next_states, next_actions)
                target_q_values = target_q_values - self.alpha * next_log_pi
                q_target = rewards + self.gamma * (1 - dones) * target_q_values
                q_targets.append(q_target)

            q_values = critic_net(states, actions)
            all_q_values.append(q_values.squeeze(-1))  # (batch_size,)
            td_error = (q_values - q_target).abs()
            all_td_errors.append(td_error.squeeze(-1))  # match shape (batch_size,)

        # Stack across critics: shape (num_critics, batch_size)
        q_tensor = torch.stack(all_q_values, dim=0)
        td_tensor = torch.stack(all_td_errors, dim=0)

        # Ensemble mean Q-values: (batch_size,)
        ensemble_mean = q_tensor.mean(dim=0)

        # Compute deviation per critic from ensemble mean: (num_critics,)
        critic_deviations = torch.abs(q_tensor - ensemble_mean).mean(dim=1)

        # Compute mean and mean square TD-error per critic
        mean_td_errors = td_tensor.mean(dim=1)  # (num_critics,)
        mean_sq_td_errors = (td_tensor**2).mean(dim=1)  # (num_critics,)
        std_td_errors = td_tensor.std(dim=1, unbiased=False)  # (num_critics,)

        # Update EMA statistics outside the loop
        for critic_id in range(len(self.critic_net.critics)):
            self._exponential_moving_average(
                critic_id=critic_id,
                mean=mean_td_errors[critic_id].item(),
                mean_sq=mean_sq_td_errors[critic_id].item(),
                deviation=critic_deviations[critic_id].item(),
            )

        critic_id = self._choose_critic()
        q_target = q_targets[critic_id]

        for critic_net, critic_net_optimiser in zip(
            self.critic_net.critics, self.ensemble_critic_optimizers
        ):
            q_values = critic_net(states, actions)

            critic_loss_total = 0.5 * F.mse_loss(q_values, q_target)

            critic_net_optimiser.zero_grad()
            critic_loss_total.backward()
            critic_net_optimiser.step()

            critic_loss_totals.append(critic_loss_total.item())

        return critic_loss_totals, critic_id

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
        critic_loss_totals, critic_id = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        info["critic_loss_totals"] = critic_loss_totals
        info["critic_id"] = critic_id

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
