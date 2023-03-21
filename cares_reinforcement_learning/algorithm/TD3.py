import os
import numpy as np
import torch
import torch.nn.functional as F




def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


class TD3:
    def __init__(self,
                 actor_network,
                 critic_network,
                 max_actions,
                 min_actions,
                 gamma,
                 tau,
                 device):

        self.actor_net  = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net  = actor_network.to(device)
        self.target_critic_net = critic_network.to(device)

        # ----------------- copy weights and bias from main to target networks ----------#
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

        self.gamma = gamma
        self.tau   = tau

        self.max_actions = max_actions
        self.min_actions = min_actions

        self.learn_counter      = 0
        self.policy_update_freq = 2

        self.device = device

    def select_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device) # this line should be here. all the env will need it
            action       = self.actor_net(state_tensor)
            action       = action.cpu().data.numpy().flatten() # this line should be here. all the env will need it
        return action

    def train_policy(self, experiences):
        self.learn_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=self.min_actions, max=self.max_actions)

            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        critic_loss_1 = F.mse_loss(q_values_one, q_target)
        critic_loss_2 = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        # Update the Critic
        self.critic_net.optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net.optimiser.step()

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_q_one, actor_q_two = self.critic_net(states, self.actor_net(states))

            actor_q_values = torch.minimum(actor_q_one, actor_q_two)
            actor_loss     = -actor_q_values.mean()

            self.actor_net.optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net.optimiser.step()

            # Update target network params
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save_models(self, filename):
        dir_exists = os.path.exists("models")

        if not dir_exists:
            os.makedirs("models")
        torch.save(self.actor_net.state_dict(),  f'models/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'models/{filename}_critic.pht')
        print("models has been loaded...")  # TODO change the print for logging.info


    def load_models(self, filename):
        self.actor_net.load_state_dict(torch.load(f'models/{filename}_actor.pht'))
        self.critic_net.load_state_dict(torch.load(f'models/{filename}_critic.pht'))
        print("models has been loaded...")  # TODO change the print for logging.info