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

        # # SIL hyperparameter--> move to main_algos
        # self.sil_update_interval = config.sil_update_interval
        # self.sil_n_update = config.sil_n_update #update times after policy train
        # self.sil_clip = config.sil_clip # sil clip value, using in advanagtes
        # self.sil_scaler = config.sil_scaler # for fn_reward() temporary solution
        # self.sil_max_nlog = config.sil_max_nlog # to do: how to select
        # self.sil_max_grad_norm = config.sil_max_grad_norm # to do: how to select
        # self.sil_weight = config.sil_weight # to do: how to set for different algos

        self.sil_batch_size = config.sil_batch_size

        self.use_per_buffer = config.use_per_buffer
        self.sil_per_sampling_strategy  = config.sil_per_sampling_strategy
        self.sil_per_weight_normalisation  = config.sil_per_weight_normalisation
        self.sil_beta  = config.sil_beta
        self.sil_d_beta  = config.sil_d_beta
        self.sil_per_alpha = config.sil_per_alpha
        self.sil_min_priority  = config.sil_min_priority

        self.sil_policy_update_freq = config.sil_policy_update_freq

        self.sil_learn_counter = 0
        self._temp_buffer = [] # for observe and step in SAC/TD3, save the roll-out data until episode end

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
        # to do: set sil_lr to network
        params_to_sync = ["actor_lr", "critic_lr", "gamma"] # remove batch_size, SIL should have same or small batch_size for update
        sil_params_main_algos = ["sil_update_interval", "sil_n_update", "sil_batch_size", "sil_scaler", "sil_clip", "sil_max_nlog", "sil_max_grad_norm", "sil_weight", "sil_weight_v"]
        
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
        # get sil parameter from main algos run command
        for param in sil_params_main_algos:
            # Look for attribute directly on the agent instance (e.g., self.main_algo.gamma)
            main_val = getattr(self.main_algo, param, None)
            sil_val = getattr(self.config, param, None)

            if main_val is not None:
                setattr(self, f"{param}", main_val)
            elif sil_val is not None:
                setattr(self, f"{param}", sil_val)
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
        print(f" SIL V0.1: experiences filter: reward > 0")
        print(f" SIL V0.1: Only clamp on advantages: min=0; max=sil_clip")
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
            "sil_batch_size": getattr(self, "sil_batch_size", None),
            "sil_actor_lr": getattr(self, "sil_actor_lr", None),
            "sil_critic_lr": getattr(self, "sil_critic_lr", None),
            "sil_gamma": getattr(self, "sil_gamma", None),
            "sil_update_interval": getattr(self, "sil_update_interval", None),
            "sil_n_update": getattr(self, "sil_n_update", None),
            "sil_batch_size": getattr(self, "sil_batch_size", None),
            "sil_scaler": getattr(self, "sil_scaler", None), # for fn_reward() temporary solution
            "sil_clip": getattr(self, "sil_clip", None),
            "sil_weight": getattr(self, "sil_weight", None),
            "sil_weight_v": getattr(self, "sil_weight_v", None),
            "sil_max_nlog": getattr(self, "sil_max_nlog", None),
            "sil_max_grad_norm": getattr(self, "sil_max_grad_norm", None),
            "device": self.device
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

    def fn_reward(self, reward):
        # TODO what reward scaler should use for sil: Static Scaling, Sign-based, Return-based RMS
        # how to balance cross-domain generalization and data precision in SIL advantegs
        # Temporary solution here
        sil_scaler = self.sil_scaler
        fn_reward = reward*sil_scaler
        return fn_reward

    def MC_return_calculator(
            self,
            batch_rewards: torch.Tensor,
            batch_episode_ends: torch.Tensor
        ) -> torch.Tensor:
        batch_returns = torch.zeros_like(batch_rewards)
        running_return = 0
        for t in reversed(range(len(batch_rewards))):
            mask = 1.0 - batch_episode_ends[t]
            running_return = batch_rewards[t] + self.gamma * running_return * mask
            batch_returns[t] = running_return

        # print(f"---batch_returns.shape: {batch_returns.shape}")
        return batch_returns #[batch_size]

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

        # reward scalling
        scaled_rewards = self.fn_reward(batch_rewards)

        # calculate MC return for all episodes in memory
        batch_returns = self.MC_return_calculator(scaled_rewards, batch_episode_ends)

        # to do: how to filter good experiences in step stage for all tasks?
        # advantages = self.sil_advantages_calculator(batch_states, batch_actions, batch_returns).detach()
        # masks = (advantages > 0).flatten()   # advanatges is changing with critic

        # solution 1: same sa source code, using reward > 0, difference: add s-a or add whole episode
        rewards = batch_rewards.view(-1)
        masks = (rewards > 0).flatten()

        f_states = batch_states[masks]
        f_actions = batch_actions[masks]
        f_returns = batch_returns[masks]
        f_next_states = batch_next_states[masks]
        f_dones = batch_dones[masks]
        f_episode_ends = batch_episode_ends[masks]

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

        # for debug
        filtered_num = f_states.shape[0]
        print(f" SIL: filtered {filtered_num} s-a good experiences to SIL_memory")
        print("-" * 60)

    def observe_step(
            self,
            state,       # np.ndarray
            action,      # np.ndarray
            reward,      # float / np.ndarray
            next_state,  # np.ndarray
            done,        # bool
            episode_end, # bool
        ):
        """
        oberserve roll-out and step data when episode end
        design for SAC, TD3, which return data each step
        """
        if not hasattr(self, '_temp_buffer'):
            self._temp_buffer = []

        self._temp_buffer.append((state, action, reward, next_state, done, episode_end))

        if episode_end:
            if not self._temp_buffer:
                return
            states, actions, rewards, next_states, dones, episode_ends = zip(*self._temp_buffer)
            batch_states = torch.from_numpy(np.array(states)).float().to(self.device)
            batch_actions = torch.from_numpy(np.array(actions)).float().to(self.device)
            batch_rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
            batch_next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
            batch_dones = torch.from_numpy(np.array(dones)).float().to(self.device)
            batch_episode_ends = torch.from_numpy(np.array(episode_ends)).float().to(self.device)
            # for debug
            print(f"--- Episode End. Processing {len(self._temp_buffer)} steps for SIL ---")

            self.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_episode_ends)
            # Note: must be clear and read for next episode
            self._temp_buffer.clear()

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
            # new limit for SACSIL
            sil_log_std_min = -5.0
            sil_log_std_max = log_std_max

            #log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            log_std = sil_log_std_min + 0.5 * (sil_log_std_max - sil_log_std_min) * (log_std + 1)

            std = log_std.exp()

            dist = SquashedNormal(mu, std)
            #sample = dist.rsample()
            log_pi = dist.log_prob(actions).sum(-1, keepdim=True) # using action from sil_memory
            nlog_p = -log_pi
            nlog_p = nlog_p.view(-1)

        # TD3
        if algo_name == "TD3SIL":
            # to do : how to appoximate nlog_prob of TD3 actor action
            pred_actions = self.actor_net(states)
            pred_mes = torch.pow(pred_actions - actions, 2).sum(dim=-1, keepdim=True)
            nlog_p = pred_mes
            nlog_p = nlog_p.view(-1)

        # print(f"---_get_nlog_p, nlog_p.shape: {nlog_p.shape}")
        return nlog_p

    def _get_value(self, states, actions):
        # get v(value) from critic, with grad

        algo_name = self.main_algo.__class__.__name__

        # PPO2
        if algo_name == "PPO2SIL":
            v = self.critic_net(states).squeeze(-1) # shape:[batch_size]

        # SAC
        # to do: which value sould be use: from critic_net, or target_critic_net ?
        # current use mini q value from critic_net
        if algo_name == "SACSIL":
            q1, q2 = self.critic_net(states, actions)
            v = torch.min(q1, q2).squeeze(-1) # shape:[batch_size]

        # TD3
        # to do : doble confirm the value
        if algo_name == "TD3SIL":
            q1, q2 = self.critic_net(states, actions)
            v = torch.min(q1, q2).squeeze(-1) # shape:[batch_size]

        # print(f"---_get_value, v.shape: {v.shape}")
        return v

    def sil_advantages_calculator(self, states, actions, returns):
        # R - V, with grad

        # for debug: shape issue
        if returns.dim() > 1:
            returns = returns.squeeze(-1)
        v = self._get_value(states, actions)
        if v .dim() > 1:
            v = v.squeeze(-1)

        advantages = returns - self._get_value(states, actions) # shape[Batch_size]
        # print(f"---sil_advantages_calculator, advantages.shape: {advantages.shape}")

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
        clipped_advantages = torch.clamp(advantages, min=0.0, max = self.sil_clip).view(-1)

        # check shape
        # print('------SIL critic loss calculate------')
        # print(f"advantages.shape: {advantages.shape}, clipped_advantages.shape: {clipped_advantages.shape}")
        sil_critic_loss = ((0.5 * torch.pow(clipped_advantages, 2)).mean())*self.sil_weight_v*self.sil_weight

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        sil_critic_loss.backward()
        total_norm_critic = torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=self.sil_max_grad_norm)
        self.critic_net_optimiser.step()

        info |= {
            "sil/critic/adv_mean": advantages.mean().item(),
            "sil/critic/adv_max": advantages.max().item(),
            "sil/critic/clipped_adv_mean": clipped_advantages.mean().item(),
            "sil/critic_loss": sil_critic_loss.item(),
            "sil/total_norm_critic": total_norm_critic.item()
            }

        #info |= {"sil_critic_loss": sil_critic_loss.item()}


        if self.sil_learn_counter % self.sil_policy_update_freq == 0:
            # Update the Actor
            # sil_policy_loss = -log π_θ(a|s) * (R - V_θ(s))+
            nlog_p = self._get_nlog_p(states_tensor, actions_tensor)
            clipped_nlog_p = torch.clamp(nlog_p, max= self.sil_max_nlog)
            # to do: should using latest advs in here?
            curr_advanatges = self.sil_advantages_calculator(states_tensor, actions_tensor, returns_tensor).detach()
            clipped_curr_advanatges = torch.clamp(curr_advanatges, min=0.0, max = self.sil_clip)

            #check shape
            # print('------SIL actor loss calculate------')
            # print(f"nlog_p.shape: {nlog_p.shape}, clipped_curr_advanatges.shape: {clipped_curr_advanatges.shape}")
            sil_actor_loss = ((clipped_nlog_p * clipped_curr_advanatges).mean())*self.sil_weight

            self.actor_net_optimiser.zero_grad()
            sil_actor_loss.backward()
            total_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=self.sil_max_grad_norm)
            self.actor_net_optimiser.step()

            info |= {
                "sil/actor/nlog_p_mean": nlog_p.mean().item(),
                "sil/actor/nlog_p_max": nlog_p.max().item(),
                "sil/actor/clipped_nlog_p_mean": clipped_nlog_p.mean().item(),
                "sil/actor/curr_advs_mean": curr_advanatges.mean().item(),
                "sil/actor/curr_advs_max": curr_advanatges.max().item(),
                "sil/actor/clipped_curr_advs_mean": clipped_curr_advanatges.mean().item(),
                "sil/actor_loss": sil_actor_loss.item(),
                "sil/total_norm_actor": total_norm_actor.item()
                }

            #info |= {"sil_actor_loss": sil_actor_loss.item()}

        # update priority base on advs
        # TODO how to set up initial priority before sampling?
        # check teh shape
        # print('------SIL per priority------')
        # print(f"curr_advanatges.shape: {curr_advanatges.shape}")
        priorities = ((clipped_advantages)     # to do: here should be clipped_curr_advanatges ?
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
        sil_min_batch_size = self.sil_batch_size
        curr_experiences_num = self.sil_memory.__len__()
        if (curr_experiences_num >= sil_min_batch_size):
            # start SIL train

            # update update n times after main algorithm update
            # print(f"SIL_train: update {self.sil_n_update} times after main algorithm update")
            for x in range(self.sil_n_update):
                self.sil_learn_counter += 1
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
                # print(f"SIL trained {self.sil_learn_counter}")

                # print("-" * 60)
            return info
        else:
            # print(f"only {curr_experiences_num} data, less than SIL batch size {sil_min_batch_size}, skip SIL train")
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
