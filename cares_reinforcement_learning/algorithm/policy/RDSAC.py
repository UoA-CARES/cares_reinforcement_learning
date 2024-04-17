import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F

class RDSAC:
    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 action_num,
                 state_dim,
                 device):
      
        self.type = "policy"
        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau

        self.learn_counter = 0 
        self.policy_update_freq = 2  

        self.action_num = action_num
        self.state_dim = state_dim
        self.device = device

        self.target_entropy = -action_num
       

        init_temperature = 0.01
        
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

        # new ones
        self.alpha = 0.7 # 0.4 0.6
        self.min_priority = 1
        self.noise_clip = 0.5  
        self.policy_noise = 0.2  

        # RD-PER parameters
        self.scale_r = 1.0
        self.scale_s = 1.0
        self.update_step = 0
    
    def div(self,target):
        return target[:, 0], target[:, 1], target[:, 2:]

    def select_action_from_policy(self, state, evaluation=False, noise_scale=0):
       
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
    def alpha_entropy(self):
        return self.log_alpha.exp()
    
    def train_policy(self, experience):

        self.learn_counter += 1
        info = {}

        # Sample replay buffer
        states, actions, rewards, next_states, dones, indices, weights = experience

        dones = dones.reshape(-1, 1)

        # Get current Q estimates way2 (2)
        q_values_one, q_values_two = self.critic_net(states.detach(), actions.detach())
        values1, rew1, next_states1 = self.div(q_values_one)
        values2, rew2, next_states2 = self.div(q_values_two)

        diff_rew1 = 0.5 * torch.pow(rew1.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
        diff_rew2 = 0.5 * torch.pow(rew2.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
        diff_next_states1 = 0.5 * torch.mean(torch.pow(next_states1.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim), 2.0), -1).reshape(-1, 1)
        diff_next_states2 = 0.5 * torch.mean(torch.pow(next_states2.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim), 2.0), -1).reshape(-1, 1)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net.sample(next_states)
            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            next_values1, _, _ = self.div(target_q_values_one)
            next_values2, _, _ = self.div(target_q_values_two)
            min_next_target = torch.minimum(next_values1, next_values2).reshape(-1, 1)  # torch.min
            target_q_values = min_next_target - self.alpha_entropy * next_log_pi
            q_target = rewards + self.gamma * (1 - dones) * target_q_values
            
        #############################################
        diff_td1 = F.mse_loss(values1.reshape(-1, 1), q_target, reduction='none')
        diff_td2 = F.mse_loss(values2.reshape(-1, 1), q_target, reduction='none')
        critic_total_loss1 = (diff_td1 + self.scale_r * diff_rew1 + self.scale_s * diff_next_states1)
        critic_total_loss2 = (diff_td2 + self.scale_r * diff_rew2 + self.scale_s * diff_next_states2)

        critic1_loss = (diff_rew1.reshape(-1, 1))
        critic2_loss = (diff_rew2.reshape(-1, 1))

        critic_loss_total = (critic_one_loss * weights).mean() + ( critic_two_loss * weights
        ).mean()

        # train critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        
        
        pi, log_pi, _ = self.actor_net.sample(states)
        qf1__pi, qf2__pi = self.critic_net(states, pi)
        qf1_pi, _, _ = self.div(qf1__pi)
        qf2_pi, _, _ = self.div(qf2__pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = torch.mean(((self.alpha_entropy * log_pi) - min_qf_pi) * weights)

        # Update the Actor
        self.actor_net.optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net.optimiser.step()

        # policy loss

        numpy_td = torch.cat([diff_td1, diff_td2], -1)
        numpy_td = torch.mean(numpy_td, 1)
        numpy_td = numpy_td.view(-1, 1)
        numpy_td = numpy_td[:, 0].detach().data.cpu().numpy()

        numpy_r = torch.cat([diff_rew1, diff_rew2], -1)
        numpy_r = torch.mean(numpy_r, 1)
        numpy_r = numpy_r.view(-1, 1)
        numpy_r = numpy_r[:, 0].detach().data.cpu().numpy()

        numpy_s = torch.cat([diff_next_states1, diff_next_states2], -1)
        numpy_s = torch.mean(numpy_s, 1)
        numpy_s = numpy_s.view(-1, 1)
        numpy_s = numpy_s[:, 0].detach().data.cpu().numpy()
        numpy_next_q_value = q_target[:, 0].detach().data.cpu().numpy()

        ############################
        # calculate priority

        priorities = torch.max(critic1_loss, critic2_loss).clamp(min=self.min_priority).pow(self.alpha).cpu().data.numpy().flatten()
        
        
        # update the temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        

        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))



            # Update target network params
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            info['actor_loss'] = actor_loss

        ################################################
        # Update Scales
        if self.update_step == 0:
            self.scale_r = np.mean(numpy_td) / (np.mean(numpy_r))
            self.scale_s = np.mean(numpy_td) / (np.mean(numpy_s))
        self.update_step += 1

        info["q_target"] = q_target
        info["q_values_one"] = q_values_one
        info["q_values_two"] = q_values_two
        info["q_values_min"] = torch.minimum(q_values_one, q_values_two)
        info["critic_loss_total"] = critic_loss_total
        info["critic_loss_one"] = critic1_loss
        info["critic_loss_two"] = critic2_loss
        info["priorities"] = priorities
        info["indices"] = indices

        return info

    def save_models(self, filename, filepath="models"):
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
