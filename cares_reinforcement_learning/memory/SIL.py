# First version under development of self immitation

import numpy as np
import torch
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory 

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.configurations import AlgorithmConfig

from torch.distributions import MultivariateNormal  #for _evaluate_policy function
import os  ##for print data
import csv


class SelfImitation:
    def __init__(
            self,
            actor=None, 
            critic=None, 
            optimizer_actor=None, 
            optimizer_critic=None, 
            gamma=None,

            device=None, 
            sil_clip= 20 ,     ###sil advs clip: differ from PPO clip eps_clip, use sil_clip, try  20
            per_alpha = None,
            per_beta = None,
            n_update = None
            ):
        self.actor_net = actor   #change to meet functions
        self.critic_net = critic   #change to meet functions
        self.actor_net_optimiser = optimizer_actor
        self.critic_net_optimiser = optimizer_critic
        self.gamma = gamma
        self.device = device  # self.device = torch.device is worng, only get type.
        self.sil_clip = sil_clip
        self.sil_beta = per_beta # Beta parameter for prioritized sampling  ##this should be for memory_SIL
        self.alpha = per_alpha ## alpha for PRE

        self.memory_SIL = MemoryBuffer(beta = per_beta)


        self.batch_size = 512   ###this should be define by algorithm?  original = 256
        self.mini_batch_size = 64  #only ready and train when data greater than this 

        # how to use value and logp function from algorithm?
        self.action_num = self.actor_net.num_actions  ##checked, no issue
        self.cov_var = torch.full(size=(self.action_num,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var) #add this for functions
        self.n_update = n_update

        self.max_nlogp = 5 #same as sil source code

        #if using total loss
        self.w_value = 0.01 #same as sil source code, spreated update without this
        self.w_entropy = 0.01 #same as sil source code,  spreate update without this

        #for debug
        self.train_count = 0
        
        print('sil instantion done, created memory_sil')

    # for debug
    def printing_config(self):
        print("==== SelfImitation Config ====")
        attrs = [
            ("gamma", self.gamma),
            ("device", self.device),
            ("sil_clip", self.sil_clip),
            ("sil_beta", self.sil_beta),
            ("alpha", self.alpha),
            ("memory_SIL", self.memory_SIL),
            ("batch_size", self.batch_size),
            ("mini_batch_size", self.mini_batch_size),
            ("n_update", self.n_update),
            ("max_nlogp", self.max_nlogp),
            ("w_value", self.w_value),
            ("w_entropy", self.w_entropy),
            ("train_count", self.train_count),
        ]
        for name, value in attrs:
            print(f"{name} ({type(value)}): {value}")
        print("==============================")


    #copy from PPO.py, for train, input shape: batch_size, bos_dim . action_dim
    def _evaluate_policy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.critic_net(state).squeeze()  
        mean = self.actor_net(state)  
        dist = MultivariateNormal(mean, self.cov_mat)  
        log_prob = dist.log_prob(action)  
        return v, log_prob
    

    #copy from PPO.py
    def _calculate_rewards_to_go(
        self, batch_rewards: torch.Tensor, batch_dones: torch.Tensor
    ) -> torch.Tensor:
        rtgs: list[float] = []
        discounted_reward = 0
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)): #定义计算折扣累计奖励
            discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
            rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(rtgs, dtype=torch.float).to(self.device)  
        return batch_rtgs
    

    
    
    #copy from TD3_sil sil.py
    def add_episode(self, trajectory):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals= []

        for (state, action, reward, next_state, terminal) in trajectory:
            if state is not None:
                states.append(state)
            else:
                states.append(None)
            actions.append(action)
            #rewards.append(np.sign(reward))   ###no, this only get sign: -1, 0, 1
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(terminal)

        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        #print('memory_SIL: rewards  ')

        terminals_tensor = torch.LongTensor(np.asarray(terminals)).to(self.device)
        #print('memory_SIL: terminals  ')

        # for debug, check setp trajectory lens before calcualte returns
        # print('---SIL add_episode data ---')
        # names = ("states", "actions", "rewards", "next_states", "terminals")
        # lists = (states, actions, rewards, next_states, terminals)
        # lens = list(map(len, lists))
        # print("[SIL] lens:", " ".join(f"{n}={l}" for n, l in zip(names, lens)))
        # if len(set(lens)) != 1:
        #     raise ValueError(f"[SIL] length mismatch: {dict(zip(names, lens))}")


        rtgs = self._calculate_rewards_to_go(rewards_tensor, terminals_tensor)  #return Tensor, len: 1000
        #print('memory_SIL: rtgs ')
        # print("rtgs_tensor:", type(rtgs), rtgs.shape)

        rtgs = rtgs.tolist() # convert to python list, consistent with other
        # print("rtgs:", type(rtgs), len(rtgs))




        ## the add function with inital priority setting using max_priority
        for (state, action, rtg, next_state, terminal) in list(zip(states, actions, rtgs, next_states, terminals)):   ##input rewards and calculate retgs
            self.memory_SIL.add(state, action, rtg, next_state, terminal)     # memory will set the new sapmle as max_priority
        

        


    def setp(self, states, actions, rewards, next_states, dones, truncated):
        trajectory = []
        trajectory_num = 0
        terminals = []
        for i in range(len(dones)):
            terminal = bool(dones[i] or truncated[i])
            terminals.append(terminal)
            step = [
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                terminals[i]
            ]
            trajectory.append(step)
            #print('add a trajectory')

            if terminals[i]:
                has_prositive_reward = any(step[2] > 0 for step in trajectory)  #any rewards >0
                
                if has_prositive_reward:
                    #add trahectory to memory_SIL with rtgs
                    self.add_episode(trajectory)

                    trajectory_num +=1
            
                trajectory = []
        # print('Saved trajectories:', trajectory_num)


    def train(self):
        if self.n_update == 0:
            print('n_update=0, without SIL train ')
            return 0, 0
        
        self.train_count += 1 
        mean_adv, valid_samples_num = 0, 0  #train function return
        
        if self.n_update < 1:
            update_ratio = int(1/self.n_update)
            if self.train_count % update_ratio == 0:
                mean_adv, valid_samples_num = self._train()
                print('SIL trained')
        else:
            for n in range(self.n_update):
                mean_adv, valid_samples_num = self._train()
                print('SIL trained')
            
    def _train(self): #first follow sil version:
        print('start SIL train')
        states, actions, rtgs, next_states, terminals, idxes, weights = self.sample_batch(self.batch_size) #prioritied sample, beta
        mean_adv=None 
        valid_samples_num =None   #why return this(num_valid_samples) in TD3_SIL?


        # for debug

        # print("---[SIL] Sampled batch details---")
        # print("  states      :", type(states), " | len =", len(states), " | sample[0] shape =", np.array(states[0]).shape)
        # print("  actions  :", type(actions), " | len =", len(actions), " | sample[0] shape =", np.array(actions[0]).shape)
        # print("  rtgs     :", type(rtgs), " | len =", len(rtgs), " | sample[0] =", rtgs[0])
        # print("  next_states :", type(next_states), " | len =", len(next_states), " | sample[0] shape =", np.array(next_states[0]).shape)
        # print("  terminals    :", type(terminals), " | len =", len(terminals), " | sample[0] =", terminals[0], " | type =", type(terminals[0]))
        # print("  idxes    :", type(idxes), " | len =", len(idxes), " | sample[0] =", idxes[0])
        # print("  weights  :", type(weights), " | len =", len(weights), " | sample[0] =", weights[0])


        # for debug
        # if not hasattr(self, "export_count"):
        #     self.export_count = 0

        # if self.export_count < 5:
        #     save_dir = "buffer_exports_sil"
        #     os.makedirs(save_dir, exist_ok=True)
        #     file_path = os.path.join(save_dir, f"batch_{self.export_count + 1}.csv")
            
        #     with open(file_path, mode='w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(["states", "actions", "rtgs", "next_states", "terminals", "idxes", "weights"])
                
        #         for i in range(len(states)):
        #             writer.writerow([
        #                 states[i].tolist(), 
        #                 actions[i].tolist(), 
        #                 rtgs[i], 
        #                 next_states[i].tolist(), 
        #                 terminals[i], 
        #                 idxes[i], 
        #                 weights[i]
        #             ])
        #     print(f"[SIL] Exported buffer batch {self.export_count+1} to {file_path}")
        #     self.export_count += 1


        if states is not None:
           #convert to Tensor
           #should set up device in here?: torch.FloatTensor(np.asarray(next_states)).to(self.device)

            states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
            actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
            returns_tensor = torch.FloatTensor(np.asarray(rtgs)).to(self.device)
            weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device) 

            # calculate the value_critic before SIL update
            v, log_probs_tensor = self._evaluate_policy(states_tensor, actions_tensor)

            #calculate the advs: R- V before SIL update, here retuns = rtgs
            advantages = returns_tensor - v.detach()

            #avtanges = R-V, for SIL loss: (R-V)+, 
            #and clip by sil_clip(source code: 1, but should depend on task)
            clipped_advantages = torch.clamp(advantages, 0, self.sil_clip)  ##self.sil_clip = sil_clip = 20, source code setting is 1. wait to test


            #mask the advs>0
            masks = (advantages > 0).float()
            valid_samples_num = masks.sum().item()
            samples_num = max(valid_samples_num, self.mini_batch_size)  # to avoid 0, min_batch = 64(can be less?)

            #calculate mean_adv sa TD3_SIL version, but why return this?
            mean_adv = (clipped_advantages.sum()/samples_num).item()


            #log_probs, actor loss
            #clip by max_nlogp, the source code use 5, but why? Depends on task or algorithm?
            #log_probs_tensor = self._calculate_log_prob(states_tensor, actions_tensor)   ##
            nlogp = -log_probs_tensor

            # clamp ase SIL source code using Straight-Through Estimator, STE, default max_nlogp = 5
            clipped_nlogp = nlogp + (torch.clamp(nlogp, max= self.max_nlogp) - nlogp).detach()

            #pg_loss = sum(weights*advs*clipped_nlogp)/samples_num
            ## sil_code:self.pg_loss = tf.reduce_sum(self.W * self.adv * clipped_nlogp) / self.num_samples
            actor_loss_sil = (torch.sum(weights_tensor*clipped_advantages*clipped_nlogp))/samples_num


            # Entropy, optional part, to do


            #value loss, critic loss, 
            A_plus = torch.clamp((returns_tensor-v), 0, self.sil_clip)
            critic_loss_sil = torch.sum(weights_tensor*0.5*(A_plus.pow(2)))/samples_num


            # for debug

            # print("states_tensor:", type(states_tensor), states_tensor.shape)
            # print("actions_tensor:", type(actions_tensor), actions_tensor.shape)
            # print("returns_tensor:", type(returns_tensor), returns_tensor.shape)
            # print("weights_tensor:", type(weights_tensor), weights_tensor.shape)
            # print("value v:", type(v), v.shape)
            # print("log_probs_tensor:", type(log_probs_tensor), log_probs_tensor.shape)
            # print("advantages:", type(advantages), advantages.shape)
            # print("masks:", type(masks), masks.shape, "valid_samples_num:", masks.sum().item())
            # print("clipped_nlogp:", type(clipped_nlogp), clipped_nlogp.shape)

            # print("actor_loss_sil:", float(actor_loss_sil))
            # print("critic_loss_sil:", float(critic_loss_sil))


            #update networks
            self.actor_net_optimiser.zero_grad()
            actor_loss_sil.backward(retain_graph=True)
            self.actor_net_optimiser.step()

            self.critic_net_optimiser.zero_grad()
            critic_loss_sil.backward()
            self.critic_net_optimiser.step() 


            #update memory_SIL sample priority            
            ###need to update priority base on TD error and parameter pre_alpha)
            PER_advantages = (clipped_advantages + self.memory_SIL.min_priority) ** self.alpha ##to adviod the 0 in priority
            # print("PER_advantages:", type(PER_advantages), PER_advantages.shape)           

            self.memory_SIL.update_priorities(idxes, PER_advantages.detach().cpu().numpy().flatten())  ##need to double check the function usage
            # print('sil train done')


        else:
            print('states is None, skiped SIL update')

        return mean_adv, valid_samples_num
    

    def sample_batch(self, batch_size):

        if len(self.memory_SIL) > self.mini_batch_size:
            batch_size = min(batch_size, len(self.memory_SIL))
            states, actions, rtgs, next_states, dones, idxes, weights = self.memory_SIL.sample_priority(batch_size)   #beta
            return states, actions, rtgs, next_states, dones, idxes, weights
            #return self.memory_SIL.sample_priority(batch_size)
        else:
            return None, None, None, None, None, None,None





        





    

        

    

        
