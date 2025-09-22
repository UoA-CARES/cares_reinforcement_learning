"""
https://openreview.net/pdf?id=PDgZ3rvqHn
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.SDAR import Actor, Critic
from cares_reinforcement_learning.util.configurations import SDARConfig


class SDAR(VectorAlgorithm):
    actor_network: Actor
    critic_network: Critic

    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SDARConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        # SAC-style initialization
        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_weight_normalisation = config.per_weight_normalisation
        self.per_alpha = config.per_alpha
        self.min_priority = config.min_priority

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        # Networks
        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_critic_net.eval()

        self.target_entropy = -self.actor_net.num_actions

        # Optimizers
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

        # Alpha (entropy regularization)
        alpha_init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(alpha_init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr, **config.alpha_lr_params
        )

        # SDAR-specific initialization
        self.prev_action_tensor = torch.zeros(
            (1, self.actor_net.num_actions), device=self.device
        )

        self.force_act = True

        self.target_beta = -0.5 * self.actor_net.num_actions

        # Beta (action regularization specific to SDAR)
        beta_init_temperature = 1.0
        self.log_beta = torch.tensor(np.log(beta_init_temperature)).to(device)
        self.log_beta.requires_grad = True
        self.log_beta_optimizer = torch.optim.Adam(
            [self.log_beta], lr=config.beta_lr, **config.beta_lr_params
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def episode_done(self):
        # Reset the previous action to the dummy action
        self.prev_action_tensor = torch.zeros(
            (1, self.actor_net.num_actions), device=self.device
        )
        self.force_act = True

    def select_action_from_policy(
        self,
        state: np.ndarray,
        evaluation: bool = False,
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            if evaluation:
                (_, _, action, *_) = self.actor_net(
                    state_tensor, self.prev_action_tensor, force_act=self.force_act
                )
            else:
                (action, _, *_) = self.actor_net(
                    state_tensor, self.prev_action_tensor, force_act=self.force_act
                )

            self.prev_action_tensor = action
            self.force_act = False

            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, *_ = self.actor_net(
                    next_states, actions, force_act=False
                )

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

        critic_loss_one = F.mse_loss(q_values_one, q_target, reduction="none")
        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = F.mse_loss(q_values_two, q_target, reduction="none")
        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        info = {
            "critic_loss_one": critic_loss_one.item(),
            "critic_loss_two": critic_loss_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_actor_alpha(  # type: ignore[override]
        self,
        states: torch.Tensor,
        prev_actions: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        (
            sample_action,
            log_pi,
            _,
            act_probs,
            binary_mask,
            log_beta,
        ) = self.actor_net(states, prev_actions, force_act=False)

        with hlp.evaluating(self.critic_net):
            qf1_pi, qf2_pi = self.critic_net(states, sample_action)

        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) + (self.beta * log_beta) - min_qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature (alpha)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # === Update α_β (for β) ===
        beta_loss = -(self.log_beta * (log_beta + self.target_beta).detach()).mean()

        self.log_beta_optimizer.zero_grad()
        beta_loss.backward()
        self.log_beta_optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "beta_loss": beta_loss.item(),
            "log_pi": log_pi.mean().item(),
            "log_beta": log_beta.mean().item(),
            "act_prob_mean": act_probs.mean().item(),
            "b_mean": binary_mask.mean().item(),
        }

        return info

    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
        self.learn_counter += 1

        # state_i, action_i, reward_i, next_state_i, done_i, ..._i, state_i+1, action_i+1, reward_i+1, next_state_i+1, done_i+1, ..._+i
        experiences = memory.sample_consecutive(batch_size)

        (
            _,
            prev_actions,
            _,
            _,
            _,
            states,
            actions,
            rewards,
            next_states,
            dones,
            _,
        ) = experiences

        weights = [1.0] * batch_size

        batch_size = len(states)

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)
        weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)

        prev_actions = torch.FloatTensor(np.asarray(prev_actions)).to(self.device)

        # Reshape to batch_size x whatever
        rewards_tensor = rewards_tensor.reshape(batch_size, 1)
        dones_tensor = dones_tensor.reshape(batch_size, 1)
        weights_tensor = weights_tensor.reshape(batch_size, 1)

        info: dict[str, Any] = {}

        # Update the Critic
        critic_info, priorities = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
        )
        info |= critic_info

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(
                states_tensor, prev_actions, weights_tensor
            )
            info |= actor_info
            info["alpha"] = self.alpha.item()
            info["beta"] = self.beta.item()

        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "log_beta": float(self.log_beta.detach().cpu().item()),
            "log_beta_optimizer": self.log_beta_optimizer.state_dict(),
            "target_beta": self.target_beta,
            "learn_counter": int(self.learn_counter),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.target_critic_net.load_state_dict(checkpoint["target_critic"])
        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])

        self.log_alpha.data = torch.tensor(checkpoint["log_alpha"]).to(self.device)
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

        self.log_beta.data = torch.tensor(checkpoint["log_beta"]).to(self.device)
        self.log_beta_optimizer.load_state_dict(checkpoint["log_beta_optimizer"])
        self.target_beta = checkpoint.get("target_beta", self.target_beta)

        self.learn_counter = checkpoint.get("learn_counter", 0)

        logging.info("models, optimisers, and training state have been loaded...")
