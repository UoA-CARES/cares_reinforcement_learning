"""
Original Paper: https://arxiv.org/pdf/2101.05982.pdf
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


class REDQ:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        tau: float,
        ensemble_size: int,
        num_sample_critics: int,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):
        self.type = "policy"
        self.gamma = gamma
        self.tau = tau

        self.learn_counter = 0
        self.policy_update_freq = 1

        self.device = device

        self.target_entropy = -action_num

        self.num_sample_critics = num_sample_critics

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )

        # ------------- Ensemble of critics ------------------#
        self.ensemble_size = ensemble_size
        self.ensemble_critics = torch.nn.ModuleList()

        critics = [critic_network for _ in range(self.ensemble_size)]
        self.ensemble_critics.extend(critics)
        self.ensemble_critics.to(device)

        # Ensemble of target critics
        self.target_ensemble_critics = copy.deepcopy(self.ensemble_critics).to(device)

        lr_ensemble_critic = critic_lr
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
        # note that when evaluating this algorithm we need to select mu as action
        # so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (
                    action,
                    _,
                    _,
                ) = self.actor_net.sample(state_tensor)
            else:
                (
                    _,
                    _,
                    action,
                ) = self.actor_net.sample(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp()

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
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

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net.sample(next_states)

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

        for critic_net, critic_net_optimiser in zip(
            self.ensemble_critics, self.ensemble_critics_optimizers
        ):
            q_values = critic_net(states, actions)

            critic_loss_total = 0.5 * F.mse_loss(q_values, q_target)

            # Update the Critic
            critic_net_optimiser.zero_grad()
            critic_loss_total.backward()
            critic_net_optimiser.step()

        pi, log_pi, _ = self.actor_net.sample(states)

        qf1_pi = self.target_ensemble_critics[idx[0]](states, pi)
        qf2_pi = self.target_ensemble_critics[idx[1]](states, pi)

        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learn_counter % self.policy_update_freq == 0:
            # Update ensemble of target critics
            for critic_net, target_critic_net in zip(
                self.ensemble_critics, self.target_ensemble_critics
            ):
                for target_param, param in zip(
                    target_critic_net.parameters(), critic_net.parameters()
                ):
                    target_param.data.copy_(
                        param.data * self.tau + target_param.data * (1.0 - self.tau)
                    )

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

    def load_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        actor_path = f"{path}/{filename}_actor.pht"
        ensemble_path = f"{path}/{filename}_ensemble.pht"

        self.actor_net.load_state_dict(torch.load(actor_path))
        self.ensemble_critics.load_state_dict(torch.load(ensemble_path))
        logging.info("models has been loaded...")
