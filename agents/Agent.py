class Agent(object):
    """
    A superclass for all reinforcement learning agents. Holds methods all agents use.
    """

    def __init__(self, env, memory):
        self.env = env
        self.memory = memory

    def train(self, episode_num, batch_size):
        # Track the reward over EPISODE_NUM episodes
        historical_reward = []

        for episode in range(0, episode_num):

            # Initial State
            state, _ = self.env.reset()

            episode_reward = 0

            while True:

                action = self.choose_action(state)

                # Take the next action and observe the effect
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Add the experience to the memory buffer
                self.memory.add(state, action, reward, next_state, terminated)

                for _ in range(1, 10):
                    self.learn(batch_size)

                state = next_state
                episode_reward += reward

                if terminated or truncated:
                    break

            historical_reward.append(episode_reward)

            print(f"Episode #{episode} Reward {episode_reward}")

        # Data collected during run, for plotting
        reward_data = historical_reward

        return reward_data

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

    def choose_action(self, state):
        raise NotImplementedError("Child class must implement this function")

    def learn(self, batch_size):
        raise NotImplementedError("Child class must implement this function")
