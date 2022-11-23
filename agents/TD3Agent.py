import torch


class TD3Agent:

    def __int__(self, actor_net, target_actor_net, critic_one, target_critic_one, critic_two, target_critic_two, memory,
                gamma, tau):
        self.actor = actor_net
        self.targ_actor = target_actor_net

        self.critic_two = critic_one
        self.targ_critic_one = target_critic_one

        self.critic_two = critic_two
        self.targ_critic_two = target_critic_two

        self.memory = memory

        self.gamma = gamma
        self.tau = tau

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy()

        return action[0]

    def learn(self, batch_size):

        # Only begin learning when there's enough experience in buffer to sample from
        if len(self.memory.buffer) < batch_size:
            return

