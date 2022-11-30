from torch.utils.tensorboard import SummaryWriter


def train(agent, episode_num, batch_size, env):
    # Track the reward over EPISODE_NUM episodes
    historical_reward = []
    writer = SummaryWriter()

    for episode in range(0, episode_num):

        # Initial State
        state, _ = env.reset()

        episode_reward = 0

        while True:

            action = agent.choose_action(state)

            # Take the next action and observe the effect
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Add the experience to the memory buffer
            agent.memory.add(state, action, reward, next_state, terminated)

            # Train the neural network when the size of the memory buffer is greater than or equal to the batch size
            for _ in range(1, 10):
                agent.learn(batch_size)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        writer.add_scalar("Reward/train", episode_reward, episode)
        historical_reward.append(episode_reward)

        print(f"Episode #{episode} Reward {episode_reward}")

    writer.flush()
    writer.close()
    # Data collected during run, for plotting
    reward_data = historical_reward

    return reward_data


def fill_buffer(agent, env):
    print("Filling Buffer Started")

    memory = agent.memory

    while len(memory.buffer) != memory.buffer.maxlen:
        state, _ = env.reset()

        while True:
            action = env.action_space.sample()
            new_state, reward, terminated, truncated, _ = env.step(action)

            memory.add(state, action, reward, new_state, terminated)

            state = new_state

            if terminated or truncated:
                break

    print("Buffer Filling Complete!")


def save_models(self):
    torch.save(self.actor.state_dict(), f'trained_models/Normal-TD3_actor_{self.env_name}.pht')
    print("models have been saved...")
