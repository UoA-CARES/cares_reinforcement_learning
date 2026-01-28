"""
Original Paper:
                https://proceedings.mlr.press/v80/oh18b/oh18b.pdf
Code based on:
                https://github.com/junhyukoh/self-imitation-learning/blob/master/baselines/common/self_imitation.py
                https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/sil.py

"""
# to do: clean up the import list for SIL
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal # add for Diagonal Gaussia

import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.util.training_context import (
    TrainingContext,
    ActionContext,
)

from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import SILConfig
from cares_reinforcement_learning.networks.common import SquashedNormal # for SAC dist


class SIL(VectorAlgorithm):
    def __init__(
        self,
        main_algorithm: Any, # Use Any or 'VectorAlgorithm' to avoid circular import
        config: SILConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.device = device
        # input main algorithm hyperparamteter from agent
        self.main_algo = main_algorithm
        self.config = config

        # SIL hyperparameter
        self.sil_update_interval = config.sil_update_interval
        self.sil_n_update = config.sil_n_update #update times after policy train
        self.sil_clip = config.sil_clip # sil clip value, using in advanagtes
        self.sil_batch_size = config.batch_size
        self.sil_weight = config.sil_weight # to do: how to set for different algos

        self.use_per_buffer = config.use_per_buffer
        self.sil_per_sampling_strategy  = config.sil_per_sampling_strategy
        self.sil_per_weight_normalisation  = config.sil_per_weight_normalisation
        self.sil_beta  = config.sil_beta
        self.sil_d_beta  = config.sil_d_beta
        self.sil_per_alpha = config.sil_per_alpha
        self.sil_min_priority  = config.sil_min_priority

        self.sil_policy_update_freq = config.sil_policy_update_freq

        self.sil_learn_counter = 0

        # get training hyperparameter from main algorithms
        self._extract_hyperparameters()
        # to do: gamma, ect.

        # connect to main algorithm's network
        self._connect_network()

        # initial SIL memory
        self.sil_memory = self._create_sil_memory()

        # SIL initial check
        self._sil_initial_check()


    def _extract_hyperparameters(self):
        """
        Priority: 1. Main Agent Attributes -> 2. SIL Config
        """
        params_to_sync = ["actor_lr", "critic_lr", "gamma"] # remove batch_size, SIL should have same or small batch_size for update
        
        for param in params_to_sync:
            # Look for attribute directly on the agent instance (e.g., self.main_algo.gamma)
            main_val = getattr(self.main_algo, param, None)
            sil_val = getattr(self.config, param, None)

            if main_val is not None:
                setattr(self, f"sil_{param}", main_val)
            elif sil_val is not None:
                setattr(self, f"sil_{param}", sil_val)
                logging.info(f"SIL: Parameter '{param}' inherited from SIL Config: {sil_val}")
            else:
                error_msg = f"Critical Parameter '{param}' missing in both Agent and SIL Config!"
                logging.error(error_msg)
                raise AttributeError(error_msg)


    def _connect_network(self):
        """connect SIL to agnet network, which created by main algorithm"""
        # to do: how about anthers' parameter in some main_algo: such as temperature alpha?
        try:
            self.actor_net = self.main_algo.actor_net
            self.critic_net = self.main_algo.critic_net
            self.actor_net_optimiser = self.main_algo.actor_net_optimiser
            self.critic_net_optimiser = self.main_algo.critic_net_optimiser
        except AttributeError as e:
            raise AttributeError(f"SIL connection failed: Main algorithm missing core components. {e}")


    def _create_sil_memory(self):
        """
        create a unique sil_memory
        parameter priority: main_algo > SIL > default
        Note: SIL paper high light there is no using IS(Important sampling) for per, beta=0.0, d_beta=0.0
        """
        # check the unique of sil_memory to avoid memory overwirte
        if hasattr(self, "sil_memory") and self.sil_memory is not None:
            error_msg = "Duplicate SIL Memory detected! SIL memory should only be initialized once."
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # buffer size
        capacity = getattr(self.main_algo, "buffer_size", None)
        if capacity is None:
            capacity = getattr(self.config, "buffer_size", 1000000)
            logging.info(f"SIL: Using buffer_size: {capacity}")
        # sil PER Alpha
        per_alpha = self.sil_per_alpha
        if not (0 <= per_alpha <= 1):
            raise ValueError(f"SIL Error: Invalid per_alpha ({per_alpha}). Must be between 0 and 1.")
        print(f"SIL: Creating unique PER Buffer [Capacity: {capacity}, Alpha: {per_alpha},  Beta: {self.sil_beta},  D_beta: {self.sil_d_beta}],")
        print("-" * 60)

        # create sil_memory
        return MemoryBuffer(
            max_capacity=capacity,
            beta = self.sil_beta,
            d_beta = self.sil_d_beta,
        )


    def _sil_initial_check(self):
        """
        Check and print SIL initialization status for debugging.
        Verifies:
        1. Connectivity to main algorithm's networks and optimizers.
        2. Live parameters from the instantiated memory buffer.
        3. Essential SIL hyperparameters.
        """
        main_name = self.main_algo.__class__.__name__

        # 1. Network & Optimizer Connection Check
        required_components = {
            "Actor Network": self.actor_net,
            "Critic Network": self.critic_net,
            "Actor Optimiser": self.actor_net_optimiser,
            "Critic Optimiser": self.critic_net_optimiser
        }

        print("\n" + "="*60)
        print(f" SIL MODULE INITIALIZED FOR [ {main_name} ] ")
        print("-" * 60)

        all_systems_go = True
        for name, comp in required_components.items():
            status = "OK" if comp is not None else "MISSING"
            if comp is None: all_systems_go = False
            print(f"{name:<25}: {status}")

        print("-" * 60)

        # 2. Live Memory Parameters (Extracted directly from the active instance)
        # Assuming sil_memory is already created via create_sil_memory()
        mem_capacity = getattr(self.sil_memory, "max_capacity", "N/A")
        mem_alpha    = self.sil_per_alpha
        mem_beta    = getattr(self.sil_memory, "beta", "N/A")
        mem_d_beta    = getattr(self.sil_memory, "d_beta", "N/A")

        memory_params = {
            "Memory Capacity": mem_capacity,
            "Memory PER Alpha": mem_alpha,
            "Memory PER beta": mem_beta,
            "Memory PER d_beta": mem_d_beta,
        }

        for name, val in memory_params.items():
            print(f"{name:<25}: {val}")
            if val == "N/A": all_systems_go = False

        print("-" * 60)

        # 3. Essential Training Hyperparameters (Inherited or Configured)
        status_table = {
            "SIL Batch Size": getattr(self, "sil_batch_size", None),
            "SIL Actor LR": getattr(self, "sil_actor_lr", None),
            "SIL Critic LR": getattr(self, "sil_critic_lr", None),
            "SIL Gamma": getattr(self, "sil_gamma", None),
            "SIL N-Updates": getattr(self, "sil_n_update", None),
            "SIL Advantage Clip": getattr(self, "sil_clip", None),
            "Device": self.device
        }

        for key, value in status_table.items():
            status_str = str(value) if value is not None else "MISSING"
            if value is None: all_systems_go = False
            print(f"{key:<25}: {status_str}")

        print("="*60 + "\n")

        # Final Verification
        if not all_systems_go:
            error_msg = f"SIL Initial Check FAILED: Missing components or parameters in {main_name}."
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        logging.info(f"SIL module successfully attached and verified for {main_name}")

    def MC_return_calculator(
            self, batch_rewards: torch.Tensor, batch_episode_ends: torch.Tensor
        ) -> torch.Tensor:
        batch_returns = torch.zeros_like(batch_rewards)
        running_return = 0
        for t in reversed(range(len(batch_rewards))):
            mask = 1.0 - batch_episode_ends[t]
            running_return = batch_rewards[t] + self.gamma * running_return * mask
            batch_returns[t] = running_return

        return batch_returns

    def step(
            self,
            batch_states: torch.Tensor,
            batch_actions: torch.Tensor,
            batch_rewards: torch.Tensor,
            batch_next_states: torch.Tensor,
            batch_dones: torch.Tensor,
            batch_episode_ends: torch.Tensor,
            )-> torch.Tensor:
        # the process workflow diffs to source code,
        # in source code: filter episode having any reward > 0 in first for MC return calculate
        # to do: compare the results and figure why

        # for debug
        num_step = batch_states.shape[0]
        num_step_episodes = int(batch_episode_ends.sum())
        num_step_dones = int(batch_dones.sum())
        # print("-" * 60)
        # print(f"Total input for SIL: ")
        # print(f"{num_step} steps, {num_step_episodes} episodes, {num_step_dones} dones   .")

        # calculate MC return for all episodes in memory
        batch_returns = self.MC_return_calculator(batch_rewards, batch_episode_ends)

        advantages = self.sil_advantages_calculator(batch_states, batch_actions, batch_returns).detach()
        masks = (advantages > 0).flatten()

        f_states = batch_states[masks]
        f_actions = batch_actions[masks]
        f_returns = batch_returns[masks]
        f_next_states = batch_next_states[masks]
        f_dones = batch_dones[masks]
        f_episode_ends = batch_episode_ends[masks]
        f_advantages = advantages[masks]

        # check the lens and shape before adding to sil_memory
        tensors = [f_states, f_actions, f_returns, f_next_states, f_dones, f_episode_ends]

        if not all(len(t) == len(tensors[0]) for t in tensors):
            error_msg = f"SIL Error: Batch length mismatch among inputs."
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # add experience to sil_memory
        for i in range(len(f_states)):
            self.sil_memory.add(
                f_states[i].cpu().numpy(),
                f_actions[i].cpu().numpy(),
                f_returns[i].cpu().numpy(),
                f_next_states[i].cpu().numpy(),
                f_dones[i].cpu().numpy(),
                f_episode_ends[i].cpu().numpy(),
            )

        # to do: update priority in sil_memory before sampling
        # priorities = ((f_advantages)
        #               .clamp(self.sil_min_priority)
        #               .pow(self.sil_per_alpha)
        #               .cpu()
        #               .data.numpy()
        #               .flatten())
        # self.sil_memory.update_priorities(indices, priorities) #indices not available before sampling

        # for debug
        filtered_num = f_states.shape[0]
        # print(f" SIL: filtered {filtered_num} s-a good experiences to SIL_memory")
        # print("-" * 60)

    def _get_nlog_p(self, states, actions ):
        # get -log_prob of action from actor, with grad

        algo_name = self.main_algo.__class__.__name__

        # PPO2
        if algo_name == "PPO2SIL":
            _ , log_p = self.main_algo._evaluate_policy(states, actions)
            nlog_p = - log_p

        # SAC
        if algo_name == "SACSIL":
            # to do: double check the log of log p
            '''
            _, log_p = self.actor_net(states)
            nlog_p = - log_p
            '''
            # need get log prob out of forward, follow code same as forward only replase pred action by sample action
            '''
            x = self.actor_net(states)
            '''
            x = self.actor_net.act_net(states)
            mu = self.actor_net.mean_linear(x)
            log_std = self.actor_net.log_std_linear(x)

            # Bound the action to finite interval.
            # Apply an invertible squashing function: tanh
            # employ the change of variables formula to compute the likelihoods of the bounded actions

            # constrain log_std inside [log_std_min, log_std_max]
            log_std = torch.tanh(log_std)

            log_std_min, log_std_max = self.actor_net.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

            std = log_std.exp()

            dist = SquashedNormal(mu, std)
            #sample = dist.rsample()
            log_pi = dist.log_prob(actions).sum(-1, keepdim=True) # using action from sil_memory
            nlog_p = -log_pi

        # TD3
        if algo_name == "TD3SIL":
            # to do : how to appoximate nlog_prob of TD3 actor action
            pred_actions = self.actor_net(states)
            pred_mes = torch.pow(pred_actions - actions, 2).sum(dim=-1, keepdim=True)
            nlog_p = pred_mes

        return nlog_p

    def _get_value(self, states, actions):
        # get v(value) from critic, with grad

        algo_name = self.main_algo.__class__.__name__

        # PPO2
        if algo_name == "PPO2SIL":
            v = self.critic_net(states) # shape: [batch_size, 1]

        # SAC
        # to do: which value sould be use: from critic_net, or target_critic_net ?
        # current use mini q value from critic_net
        if algo_name == "SACSIL":
            q1, q2 = self.critic_net(states, actions)
            v = torch.min(q1, q2)

        # TD3
        # to do : doble confirm the value
        if algo_name == "TD3SIL":
            q1, q2 = self.critic_net(states, actions)
            v = torch.min(q1, q2)

        return v

    def sil_advantages_calculator(self,states, actions, returns):
        # R - V, with grad
        advantages = returns - self._get_value(states, actions) # shape[]

        return advantages


    def sil_update_networks(
        self,
        sil_memory: MemoryBuffer,
        indices: np.ndarray,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        returns_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        episode_ends_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        advantages = self.sil_advantages_calculator(states_tensor, actions_tensor, returns_tensor)

        # SIL critic loss calculation:
        # sil_value_loss  = 1/2 * || (R - V_θ(s))+ ||^2
        clipped_advantages = torch.clamp(advantages, min=0.0)
        sil_critic_loss = ((0.5 * torch.pow(clipped_advantages, 2)).mean())*self.sil_weight

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        sil_critic_loss.backward()
        self.critic_net_optimiser.step()

        info |= {"sil_critic_loss": sil_critic_loss.item()}



        if self.sil_learn_counter % self.sil_policy_update_freq == 0:
            # Update the Actor
            # sil_policy_loss = -log π_θ(a|s) * (R - V_θ(s))+
            nlog_p = self._get_nlog_p(states_tensor, actions_tensor)
            # to do: should using latest advs in here?
            curr_advanatges = self.sil_advantages_calculator(states_tensor, actions_tensor, returns_tensor).detach()
            clipped_curr_advanatges = torch.clamp(curr_advanatges, min=0.0)
            sil_actor_loss = ((nlog_p * clipped_curr_advanatges).mean())*self.sil_weight

            self.actor_net_optimiser.zero_grad()
            sil_actor_loss.backward()
            self.actor_net_optimiser.step()

            info |= {"sil_actor_loss": sil_actor_loss.item()}

        # update priority base on advs
        priorities = ((curr_advanatges)
                      .clamp(self.sil_min_priority)
                      .pow(self.sil_per_alpha)
                      .cpu()
                      .data.numpy()
                      .flatten())
        sil_memory.update_priorities(indices, priorities) #indices not available before sampling

            # to do: how to update target network params
            # hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
            # hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

        return info

    def train(self,):

        # set min_data_size = sil batch_size
        # only start SIL sampling and train when good experiences garther than min_data_size
        info = {}
        self.sil_learn_counter = 0
        sil_min_batch_size = self.sil_batch_size
        curr_experiences_num = self.sil_memory.__len__()
        if (curr_experiences_num >= sil_min_batch_size):
            # start SIL train

            # update update n times after main algorithm update
            # print(f"SIL_train: update {self.sil_n_update} times after main algorithm update")
            for x in range(self.sil_n_update):
                (
                    states_tensor,
                    actions_tensor,
                    returns_tensor,
                    next_states_tensor,
                    dones_tensor,
                    episode_ends_tensor,
                    weights_tensor,
                    indices,
                ) = tu.sample_batch_to_tensors(
                    memory=self.sil_memory,
                    batch_size=sil_min_batch_size,
                    device=self.device,
                    use_per_buffer= self.use_per_buffer,
                    per_sampling_strategy = self.sil_per_sampling_strategy,
                    per_weight_normalisation = self.sil_per_weight_normalisation,
                )

                info = self.sil_update_networks(
                    self.sil_memory,
                    indices,
                    states_tensor,
                    actions_tensor,
                    returns_tensor,
                    next_states_tensor,
                    episode_ends_tensor,
                    weights_tensor,
                )
                self.sil_learn_counter += 1
                print(f"SIL trained {self.sil_learn_counter}")

                print("-" * 60)
            return info
        else:
            print(f"only {curr_experiences_num} data, less than SIL batch size {sil_min_batch_size}, skip SIL train")
            info = {"sil_status": "skipped"}

            return info


######to do must have below function ####
    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        return np.array([])

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        return {}

    def save_models(self, filepath: str, filename: str) -> None:
        pass

    def load_models(self, filepath: str, filename: str) -> None:
        pass