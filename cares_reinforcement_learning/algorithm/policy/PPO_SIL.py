"""
Original Paper:
                https://arxiv.org/abs/1707.06347
Good Explanation:
                https://www.youtube.com/watch?v=5P7I-xPq8u8
Code based on:
                https://github.com/ericyangyu/PPO-for-Beginners
                https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.PPO import Actor, Critic
from cares_reinforcement_learning.util.configurations import PPO_SILConfig ##update
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory #
#from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.memory.SIL import SelfImitation




class PPO_SIL(VectorAlgorithm): ##updated
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PPO_SILConfig,  ##
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.gamma = config.gamma
        self.action_num = self.actor_net.num_actions
        self.device = device

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.updates_per_iteration = config.updates_per_iteration
        self.eps_clip = config.eps_clip
        self.cov_var = torch.full(size=(self.action_num,), fill_value=0.5).to(
            self.device
        )
        self.cov_mat = torch.diag(self.cov_var)

        self.n_update = config.n_update
        self.sil_clip = config.sil_clip
        self.per_alpha = config.per_alpha
        self.per_beta = config.per_beta


        #self.alg_config = config  ## add for memory_SIL
        self.sil = SelfImitation(self.actor_net, 
                                 self.critic_net, 
                                 self.actor_net_optimiser, 
                                 self.critic_net_optimiser, 
                                 self.gamma, 
                                 self.device,
                                 self.sil_clip, 
                                 self.per_alpha, 
                                 self.per_beta, 
                                 self.n_update)
        # for debug
        # self.sil.printing_config()




    def _calculate_log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        self.actor_net.eval()
        with torch.no_grad():
            mean = self.actor_net(state)

            dist = MultivariateNormal(mean, self.cov_mat)
            log_prob = dist.log_prob(action)

        self.actor_net.train()
        return log_prob

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)

            mean = self.actor_net(state_tensor)
            dist = MultivariateNormal(mean, self.cov_mat)

            # Sample an action from the distribution and get its log prob
            sample = dist.sample()

            action = sample.cpu().data.numpy().flatten()

        self.actor_net.train()

        return action

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            value = self.critic_net(state_tensor)

        return value[0].item()

    def _evaluate_policy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.critic_net(state).squeeze()  # shape 5000 #squeeze()移除1的维度
        mean = self.actor_net(state)  # shape, 5000, 1
        dist = MultivariateNormal(mean, self.cov_mat)  #构建一个多元正态分布（高斯分布）,PPO 在连续动作空间中策略建模的常规方式
        log_prob = dist.log_prob(action)  # shape, 5000
        return v, log_prob

    def _calculate_rewards_to_go(
        self, batch_rewards: torch.Tensor, batch_dones: torch.Tensor
    ) -> torch.Tensor:
        rtgs: list[float] = []
        discounted_reward = 0
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)): #定义计算折扣累计奖励
            discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
            rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(rtgs, dtype=torch.float).to(self.device)  # shape 5000
        return batch_rtgs

    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
        # pylint: disable-next=unused-argument
 


        experiences = memory.flush()    ##copy and clearn the memory
        states, actions, rewards, _, dones, truncated = experiences   ##_ is next_status


        # for debug
        # print('===data from PPO memory===')
        # print("TYPES (new):", type(states), type(actions), type(rewards), type(_), type(dones))
        # print("LENGTHS (new):", len(states), len(actions), len(rewards), len(_), len(dones)) 

        ###filter r>0 trajectory and add to memory_SIL and calculate rtgs 
        self.sil.setp(states, actions, rewards, _, dones, truncated)  ##add experience to SIL memory

        #print('completed sil.step()')


        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        log_probs_tensor = self._calculate_log_prob(states_tensor, actions_tensor)

        # compute reward to go:
        rtgs = self._calculate_rewards_to_go(rewards_tensor, dones_tensor) #rtgs是实际计算得到的折扣累计奖励
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-7)
        # print('calcualted PPO rtgs')
        #print(rtgs)

        # calculate advantages
        v, _ = self._evaluate_policy(states_tensor, actions_tensor) #这是old advantages: v, log_prob

        advantages = rtgs.detach() - v.detach() #
        # print('calcualted old advantages')
        #print(advantages)


        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        td_errors = torch.abs(advantages).data.cpu().numpy()

        # for debug
        # print('===PPO input and output ===')
        # print("states_tensor:", type(states_tensor), states_tensor.shape)
        # print("actions_tensor:", type(actions_tensor), actions_tensor.shape)
        # print("rewards_tensor:", type(rewards_tensor), rewards_tensor.shape)
        # print("dones_tensor:", type(dones_tensor), dones_tensor.shape)

        # print("log_probs_tensor:", type(log_probs_tensor), log_probs_tensor.shape)
        # print("rtgs:", type(rtgs), rtgs.shape)
        # print("value v:", type(v), v.shape)
        # print("advantages:", type(advantages), advantages.shape)
        # print("normalized advantages:", type(advantages), advantages.shape)
        # print("td_errors:", type(td_errors), td_errors.shape)




        for _ in range(self.updates_per_iteration):   ## can set up as 4, and G set as 1.
            v, curr_log_probs = self._evaluate_policy(states_tensor, actions_tensor) #更新后的,用于比较

            # Calculate ratios
            ratios = torch.exp(curr_log_probs - log_probs_tensor.detach())

            # Finding Surrogate Loss
            surrogate_lose_one = ratios * advantages
            surrogate_lose_two = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            actor_loss = (-torch.minimum(surrogate_lose_one, surrogate_lose_two)).mean() #PPO 特有的剪切目标函数 #.mean()整个batch 的loss求平均
            critic_loss = F.mse_loss(v, rtgs) #rtgs折扣累计奖励 #F.mse_loss() 是 PyTorch 内置的 均方误差（Mean Squared Error） 损失

            self.actor_net_optimiser.zero_grad() #清除之前计算残留的梯度（每次优化前都要做）,Pytorch默认累加
            actor_loss.backward(retain_graph=True) #自动执行反向传播，计算 loss 对 actor 参数的梯度,保留计算图(因为还要给critic网络使用)
            self.actor_net_optimiser.step() #更新actor网络

            self.critic_net_optimiser.zero_grad()
            critic_loss.backward()       #计算loss对aritic的梯度
            self.critic_net_optimiser.step()#更新actor网络

            # print('updated PPO network')
        
        
        ####SIL update loop####
        #n_update = 4  ### wait add this para to sil configuration
        self.sil.train()


        info: dict[str, Any] = {}
        info["td_errors"] = td_errors
        info["critic_loss"] = critic_loss.item()
        info["actor_loss"] = actor_loss.item()

        ##here add SIL_train loop


        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
