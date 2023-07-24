"""
references: https://github.com/Kchu/DeepRL_PyTorch/blob/master/Distributional_RL/1_C51.py 
"""

import os
import torch
import logging
import numpy as np
import torch.nn.functional as F


class C51:

    def __init__(self,
                 network,
                 gamma,
                 device,
                 vMin = -10,
                 vMax = 10,
                 n = 50,
                 num_atoms = 51):
        self.type = "value"
        self.network = network.to(device)
        self.device = device
        self.gamma = gamma
        
        # evenly spaced discrete value distribution range
        self.value_range = torch.linspace(vMin,vMax, num_atoms)
        self.vMin = vMin
        self.vMax = vMax
        self.n = n 
        self.num_atoms = num_atoms
        

    def select_action_from_policy(self, state):
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values_dist = self.network(state_tensor)
            # sum of distribution across n-atoms, omit n-atoms (3rd dim)
            q_values = torch.sum(q_values_dist*self.value_range.view(1,1,-1), dim = 2)
            action = torch.argmax(q_values).item()
        self.actor_net.train()
        return action

    def train_policy(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        info = {}

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # TESTING
        print(states.shape) # (BATZ, SIZE STATE)
        print(actions.shape) # (BZ, ACTION SIZE)
        print(rewards.shape) # (BZ, 1)
        print(dones.shape) # (BZ, 1)

        # Generate Q Values given state at time t and t + 1
        q_values_dist = self.network(states)
        
        next_q_values_dist = self.network(next_states)
        next_q_values = torch.sum(next_q_values_dist*self.value_range.view(1,1,-1), dim = 2)
        best_next_q_values = torch.max(next_q_values, dim=1).values

        # Reduce both distributions to (m,n_atoms)
        # Get an array of the observation set and its selected action distribution
        m_size = q_values_dist.size(0)
        m_action = [torch.index_select(q_values_dist[i],0,actions[i]) for i in range(m_size)]
        next_m_action = [torch.index_select(next_q_values_dist[i],0,best_next_q_values[i]) for i in range(m_size)]
        # Remove the 2nd dim
        q_values_dist = torch.stack(m_action).squeeze(1)
        next_q_values_dist = torch.stack(next_m_action).squeeze(1).data.cpu().numpy() # convert to numpy

        # Projection step 
        next_v_range = np.expand_dims(rewards,1) + self.gamma*np.expand_dims((1. - dones),1)*np.expand_dims(self.value_range.data.cpu().numpy(),0)
        next_v_pos = np.zeros_like(next_v_range)
        next_v_range = np.clip(next_v_range, self.vMin, self.vMax)
        
        delta_z = (self.vMax - self.vMin)/self.n
        next_v_pos = (next_v_range - self.vMin)/delta_z

        # Distribution step for each (m,num_atoms) pair
        q_target = np.zeros((m_size,self.num_atoms))
        u = np.ceil(next_v_pos).astype(int)
        l = np.floor(next_v_pos).astype(int)

        for i in range(m_size):
            for j in range(self.num_atoms): 
                q_target[i,u[i,j]] += (next_q_values_dist*(next_v_pos-l))[i,j]
                q_target[i,l[i,j]] += (next_q_values_dist*(u-next_v_pos))[i,j]

        q_target = torch.FloatTensor(q_target)

        # Calculate loss and weights
        # TESTING: test cross-entropy and KL divergence
        loss = F.kl_div(F.log_softmax(q_values_dist, dim = 1), q_target, reduction="batchmean")
        loss = torch.mean(loss)

        # Update the Network
        self.network.optimiser.zero_grad()
        loss.backward()
        self.network.optimiser.step()

        info['q_target'] = q_target
        info['q_values_min'] = q_values_dist
        info['network_loss'] = loss
        
        return info

    def save_models(self, filename, filepath='models'):
        path = f"{filepath}/models" if filepath != 'models' else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.network.state_dict(), f'{path}/{filename}_network.pht')
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != 'models' else filepath

        self.network.load_state_dict(torch.load(f'{path}/{filename}_network.pht'))
        logging.info("models has been loaded...")
