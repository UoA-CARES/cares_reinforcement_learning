#D4PG

"""
Original Paper:
                https://arxiv.org/abs/1804.08617

Code based on:
                https://github.com/schatty/d4pg-pytorch/blob/master/models/d4pg/d4pg.py
                https://github.com/bmaxdk/DeepRL-ND-Continuous-Control/blob/main/D4PG/D4PG_agent.py
"""
import logging
import os
import numpy as np
import torch
#import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim 
#import queue 


class D4PG: 
   def __init__(self,
                config,
                network,
                device, 
                v_Max, 
                v_Min, 
                gamma, 
                tau, 
                actor_lr, 
                critic_lr,
                trajectory_length,
                n_atoms,
                batch_size,
                replay_size,
                exploration_size, 
                action_size
                exploration_constant,
                ):
      
      self.type = "policy"
      self.actor_net = network.to(device)
      self.critic_net = network.to(device)
      self.device = device 
      self.gamma = gamma
      self.v_max = v_Max
      self.v_min = v_Min
      self.atoms = n_atoms 
      self.batch_size = batch_size 
      self.actor_lr = actor_lr
      self.critic_lr = critic_lr
      self.tau = tau
      self.delta_z = (self.v_max - self.v_min)/(self.atoms - 1)
      self.eps_decay = exploration_constant
      self.replay_size = replay_size 
      self.action_size = action_size 
      self.state_size = exploration_size 
      self.eps = 0.3 
      self.seed = random.seed(seed)

for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data)

for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data)   

    self.actor_net_optimiser  = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
    self.critic_net_optimiser = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
    #self.value_criterion = nn.BCELoss(reduction='none')

def select_action_from_policy(self, state) 
 self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
#def reset(self): 
   #    self.noise.reset()




#Update networks:  theta = theta + alpha, w = w + βt* δw

#Update steps 
def _update_policy(self, experiences):
        states, actions, rewards, weights, next_states, dones = experiences
                #batch_size = len(states)
        info = {}
        
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)
        weights = torch.LongTensor(np.asarray(weights)).to(self.device)


     #Predicting next action with target policy network
        next_action = self.target_actor_net(next_states)
        target_value = self.target_critic_net.get_probs(next_states, next_action.detach())
    #--Update critic-- 
        #rewards = torch.unsqueeze(rewards,0).reshape(self.batch_size, 1)
       # dones = torch.unsqueeze(dones,0).reshape(self.batch.size, 1)
       
        # Get projected distribution
        target_z_projected = _l2_project(next_distr_v=target_value,
                                         rewards_v=reward,
                                         dones_mask_t=done,
                                         gamma=self.gamma ** self.n_step_return,
                                         n_atoms=self.num_atoms,
                                         v_min=self.v_min,
                                         v_max=self.v_max,
                                         delta_z=self.delta_z)
        target_z_projected = torch.from_numpy(target_z_projected).float().to(self.device)

        critic_value = self.critic_net.get_probs(state, action)
        critic_value = critic_value.to(self.device)

        critic_loss = self.critic_criterion(critic_value, target_z_projected)
        critic_loss = critic_loss.mean(axis=1)

#-- Update step -----------------------# 
        critic_loss = -critic_loss.mean()
        self.critic_net.optimiser.zero_grad() 
        critic_loss.backward() 
        self.critic_net.optimiser.step() 

   
   

   

    #Update Actor Network
        actor_loss = self.critic_net.get_probs(states, self.actor_net(states))
        actor_loss = -actor_loss.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_net.optimiser.step()
   
        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        if update_step.value % 100 == 0: 
            try: 
                params = [p.data.cpu().detach().numpy() for p in self.policy_net.parameters()]
                self.learner_w_queue.put_nowait(params)
            except:
                pass


def run (self, training, batch_q, replay_q, update_policy): 
     torch.set_num_threads(4) 
     while update_policy.value < self.num_train_steps: 
        try: 
             experiences = batch_q.get_no_wait()
        except queue.Empty: 
             continue 
        
        self._update_policy(experiences, replay_q, update_policy)
        update_policy.value += 1
        
        if update_policy.value % 1000 == 0:
                print("Training step ", update_policy.value)

        training_on.value = 0

        logging.info['actor_loss'] = actor_loss
        logging.info['critic_loss'] = critic_loss
        logging.info['q_values_min'] = q_values
        logging.info['q_values'] = q_values
         
        return logging.info
     
#Save and load models 

def save_models(self, filename, filepath='models'):
        path = f"{filepath}/models" if filepath != 'models' else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f'{path}/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'{path}/{filename}_critic.pht')
        logging.info("models has been saved...")

def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != 'models' else filepath

        self.actor_net.load_state_dict(torch.load(f'{path}/{filename}_actor.pht'))
        self.critic_net.load_state_dict(torch.load(f'{path}/{filename}_critic.pht'))
        logging.info("models has been loaded...")
