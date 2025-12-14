"""
Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting."

Original Paper: https://dl.acm.org/doi/abs/10.1145/122344.122377

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
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.DynaSAC import Actor, Critic
from cares_reinforcement_learning.networks.world_models.ensemble_integrated import (
    EnsembleWorldReward,
)
from cares_reinforcement_learning.util.configurations import DynaSACConfig
from cares_reinforcement_learning.util.training_context import (
    TrainingContext,
    ActionContext,
)


class DynaSAC(VectorAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        world_network: EnsembleWorldReward,
        config: DynaSACConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="mbrl", config=config, device=device)

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(self.device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        self.gamma = config.gamma
        self.tau = config.tau

        self.num_samples = config.num_samples
        self.horizon = config.horizon
        self.action_num = self.actor_net.num_actions

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        self.target_entropy = -self.action_num

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        # Set to initial alpha to 1.0 according to other baselines.
        self.log_alpha = torch.tensor(np.log(1.0)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr
        )

        # World model
        self.world_model = world_network

    @property
    def _alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        # pylint: disable-next=unused-argument

        state = action_context.state
        evaluation = action_context.evaluation

        # note that when evaluating this algorithm we need to select mu as
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net(state_tensor)
            else:
                (_, _, action) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def _update_critic_actor(self, states, actions, rewards, next_states, dones):
        # Update Critic
        self._update_critic(states, actions, rewards, next_states, dones)

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            self._update_actor(states)

        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

    def _update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)
            target_q_one, target_q_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                torch.minimum(target_q_one, target_q_two) - self._alpha * next_log_pi
            )
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)
        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

    def _update_actor(self, states):
        pi, first_log_p, _ = self.actor_net(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = ((self._alpha * first_log_p) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # Update the temperature
        alpha_loss = -(
            self.log_alpha * (first_log_p + self.target_entropy).detach()
        ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def _dyna_generate_and_train(self, next_states: torch.Tensor) -> None:
        pred_states = []
        pred_actions = []
        pred_rs = []
        pred_n_states = []

        pred_state = next_states

        for _ in range(self.horizon):
            pred_state = torch.repeat_interleave(pred_state, self.num_samples, dim=0)
            # This part is controversial. But random actions is empirically better.
            rand_acts = np.random.uniform(-1, 1, (pred_state.shape[0], self.action_num))
            pred_acts = torch.FloatTensor(rand_acts).to(self.device)
            pred_next_state, _, _, _ = self.world_model.pred_next_states(
                pred_state, pred_acts
            )

            pred_reward, _ = self.world_model.pred_rewards(pred_state, pred_acts)
            pred_states.append(pred_state)
            pred_actions.append(pred_acts.detach())
            pred_rs.append(pred_reward.detach())
            pred_n_states.append(pred_next_state.detach())
            pred_state = pred_next_state.detach()

        pred_states = torch.vstack(pred_states)
        pred_actions = torch.vstack(pred_actions)
        pred_rs = torch.vstack(pred_rs)
        pred_n_states = torch.vstack(pred_n_states)

        # Pay attention to here! It is dones in the Cares RL Code!
        pred_dones = torch.FloatTensor(np.zeros(pred_rs.shape)).to(self.device)

        # states, actions, rewards, next_states, not_dones
        self._update_critic_actor(
            pred_states, pred_actions, pred_rs, pred_n_states, pred_dones
        )

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device).unsqueeze(1)

        # Step 1 train as usual
        self._update_critic_actor(states, actions, rewards, next_states, dones)

        # # # Step 2 Dyna add more data
        self._dyna_generate_and_train(next_states=next_states)

        return {}

    def train_world_model(self, memory: MemoryBuffer, batch_size: int) -> None:
        experiences = memory.sample_consecutive(batch_size)

        (
            states,
            actions,
            rewards,
            next_states,
            _,
            _,
            next_actions,
            next_rewards,
            _,
            _,
            _,
        ) = experiences

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        next_rewards = (
            torch.FloatTensor(np.asarray(next_rewards)).to(self.device).unsqueeze(1)
        )
        next_actions = torch.FloatTensor(np.asarray(next_actions)).to(self.device)

        # Step 1 train the world model.
        self.world_model.train_world(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_actions=next_actions,
            next_rewards=next_rewards,
        )

    def set_statistics(self, stats: dict) -> None:
        self.world_model.set_statistics(stats)

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            # Save log_alpha as a float, not a numpy array
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "learn_counter": self.learn_counter,
        }

        # add world model state if it supports it

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

        self.learn_counter = checkpoint.get("learn_counter", 0)

        # TODO add world model loading if needed

        logging.info("models, optimisers, and training state have been loaded...")
