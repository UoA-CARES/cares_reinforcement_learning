import torch
from gym import Env
import numpy as np

from ..util import MemoryBuffer, OUNoise
from ..networks import Actor, Critic

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

    def __init__(self,
                 env: Env,
                 memory: MemoryBuffer = MemoryBuffer(10_000),
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 learning_rate: float = 0.001,
                 actor_net: torch.nn.Module = None,
                 target_actor_net: torch.nn.Module = None,
                 critic_net: torch.nn.Module = None,
                 target_critic_net: torch.nn.Module = None
                 ) -> None:
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
        self.env = env

        obs_size = env.observation_space.shape[0]
        act_size = env.action_space.shape[0]

        self.actor = actor_net or Actor(env.observation_space, env.action_space, learning_rate)
        self.target_actor = target_actor_net or Actor(env.observation_space, env.action_space, learning_rate)
        self.critic = critic_net or Critic(obs_size, act_size, learning_rate)
        self.target_critic = target_critic_net or Critic(obs_size, act_size, learning_rate)

        self.memory = memory

        self.noise = OUNoise(env.action_space)

        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate

        self.step_counter = 0

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
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy()

        # action = self.noise.get_action(action[0], self.step_counter)

        gau_noise = np.random.normal(0, scale=0.3 * self.env.action_space.high.max(),
                                 size=self.env.action_space.shape[0])
        gau_action = np.clip(action[0] + gau_noise, self.env.action_space.low, self.env.action_space.high)
        # print(f"OUNoise {noise}")
        # print(f"Gaussian Noise {gau_noise}")

        self.step_counter += 1

        return gau_action

    def learn(self, batch_size):
        """
        Initiate Memory Replay
        """

        # Only begin memory replay when the experience buffer has enough experience to sample the desired sample size
        if len(self.memory.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.FloatTensor(np.asarray(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.LongTensor(dones)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # We do not want the gradients calculated for any of the target networks, we manually update the parameters
        with torch.no_grad():

            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)

            q_target = rewards + self.gamma * (1 - dones) * next_q_values

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

    def fill_buffer(self):

        while len(self.memory.buffer) != self.memory.buffer.maxlen:
            state, _ = self.env.reset()

            while True:
                action = self.env.action_space.sample()
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                self.memory.add(state, action, reward, new_state, terminated)

                state = new_state

                if terminated or truncated:
                    break
