"""
Original Paper: https://arxiv.org/pdf/2101.05982.pdf
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.REDQ import Actor, Critic
from cares_reinforcement_learning.util.configurations import REDQConfig
from cares_reinforcement_learning.util.training_context import TrainingContext


class REDQ(SAC):
    critic_net: Critic
    target_critic_net: Critic

    def __init__(
        self,
        actor_network: Actor,
        ensemble_critic: Critic,
        config: REDQConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=ensemble_critic,
            config=config,
            device=device,
        )

        self.num_sample_critics = config.num_sample_critics
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

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, Any]:
        # replace=False so that not picking the same idx twice
        idx = np.random.choice(
            self.ensemble_size, self.num_sample_critics, replace=False
        )

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one = self.target_critic_net.critics[idx[0]](
                next_states, next_actions
            )

            target_q_values_two = self.target_critic_net.critics[idx[1]](
                next_states, next_actions
            )

            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        critic_loss_totals = []

        for critic_net, critic_net_optimiser in zip(
            self.critic_net.critics, self.ensemble_critic_optimizers
        ):
            q_values = critic_net(states, actions)

            critic_loss = 0.5 * F.mse_loss(q_values, q_target)

            critic_net_optimiser.zero_grad()
            critic_loss.backward()
            critic_net_optimiser.step()

            critic_loss_totals.append(critic_loss.item())

        critic_loss_total = np.mean(critic_loss_totals)
        info = {
            "idx": idx,
            "critic_loss_total": critic_loss_total,
            "critic_loss_totals": critic_loss_totals,
        }

        return info

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_actor_alpha(  # type: ignore[override]
        self,
        states: torch.Tensor,
    ) -> dict[str, Any]:
        pi, log_pi, _ = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            q_values = self.critic_net(states, pi)
            min_qf_pi = q_values.mean(dim=1)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
        }

        return info

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size

        # Use the helper to sample and prepare tensors in one step
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            _,
            _,
        ) = tu.sample_batch_to_tensors(
            memory=memory,
            batch_size=batch_size,
            device=self.device,
            use_per_buffer=0,  # REDQ uses uniform sampling
        )

        info: dict[str, Any] = {}

        # Update the Critics
        critic_info = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        info |= critic_info

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor
            actor_info = self._update_actor_alpha(states_tensor)
            info |= actor_info
            info["alpha"] = self.alpha.item()

        if self.learn_counter % self.target_update_freq == 0:
            # Update ensemble of target critics
            for critic_net, target_critic_net in zip(
                self.critic_net.critics, self.target_critic_net.critics
            ):
                hlp.soft_update_params(critic_net, target_critic_net, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        super().save_models(filepath, filename)
        # Save each ensemble critic optimizer in a single file
        ensemble_optim_state = {
            f"optimizer_{idx}": opt.state_dict()
            for idx, opt in enumerate(self.ensemble_critic_optimizers)
        }
        torch.save(
            ensemble_optim_state,
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth",
        )

    def load_models(self, filepath: str, filename: str) -> None:
        super().load_models(filepath, filename)
        # Load each ensemble critic optimizer from the single file
        ensemble_optim_state = torch.load(
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth"
        )
        for idx, opt in enumerate(self.ensemble_critic_optimizers):
            opt.load_state_dict(ensemble_optim_state[f"optimizer_{idx}"])
