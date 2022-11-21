import torch
from ..util import MemoryBuffer

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")


class DDPGAgent:
    """
    Reinforcement Learning Agent using DDPG algorithm to learn
    """

    def __init__(self, actor_net: torch.nn.Module, target_actor_net: torch.nn.Module, critic_net: torch.nn.Module,
                 target_critic_net: torch.nn.Module, memory: MemoryBuffer, gamma: float,
                 tau: float) -> None:
        """
        Constructor used to create DDPGAgent

        Input:
            `actor_net`: Neural Network used to approximate the policy \n
            `target_actor_net`: Lagging Neural Network used to control over estimation \n
            `critic_net`: Neural Network approximating the Q function, used to critique the policy \n
            `target_critic_net`: Lagging Neural Network used to control over estimation \n
            `memory`: buffer used to store experience/transitions \n
            `gamma`: discount rate \n
            `tau`: polyak averaging constant, lagging constant \n
        """
        self.actor = actor_net.to(DEVICE)
        self.target_actor = target_actor_net.to(DEVICE)
        self.critic = critic_net.to(DEVICE)
        self.target_critic = target_critic_net.to(DEVICE)

        self.memory = memory

        self.gamma = gamma
        self.tau = tau

    def choose_action(self, state):
        """
        Given an observation, produce an action to take.

        Input:
            `state`: the observation used to produce the action

        Returns:
            `action`: an action to take

        We use the Actor Network to produce an action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(DEVICE)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy()

        return action[0]

    def learn(self, batch_size):
        """
        Initiate Memory Replay
        """

        # Only begin memory replay when the experience buffer has enough experience to sample the desired sample size
        if len(self.memory.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1).to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.unsqueeze(0).reshape(batch_size, 1).to(DEVICE)

        # print("States", states.shape)
        # print("Actions", actions.shape)
        # print("Rewards", rewards.shape)
        # print("Next_States", next_states.shape)
        # print("Dones", dones.shape)

        # We do not want the gradients calculated for any of the target networks, we manually update the parameters
        with torch.no_grad():

            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)

            q_target = rewards + self.gamma * ~dones * next_q_values

        q_values = self.critic(states, actions)

        # Update the Critic Network
        critic_loss = self.critic.loss(q_values, q_target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update the Actor Network
        actor_q = self.critic(states, self.actor(states))
        actor_loss = -actor_q.mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks' params
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
