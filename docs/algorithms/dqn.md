# Deep Q-Network (DQN)

## Overview

Deep Q-Network (DQN) is a value-based reinforcement learning algorithm that uses a deep neural network to approximate the Q-value function. It was the first algorithm to demonstrate successful learning of control policies directly from high-dimensional sensory input, achieving human-level performance on Atari 2600 games.

**Paper**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., Nature 2015)

**Action Space**: Discrete

**Policy Type**: Off-policy, value-based

## How It Works

DQN learns an action-value function $Q(s, a)$ that estimates the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter.

### Key Components

1. **Q-Network**: A neural network that takes state $s$ as input and outputs Q-values for all possible actions.

2. **Target Network**: A copy of the Q-network used to compute target Q-values, stabilizing training. Updated either by a hard update every `target_update_freq` steps or via soft (Polyak) averaging controlled by `tau`.

3. **Experience Replay Buffer**: Stores transitions $(s, a, r, s')$ and samples random mini-batches for training, breaking temporal correlations. The library supports both uniform sampling and Prioritized Experience Replay (PER) via `use_per_buffer`.

### Update Rule

The Q-network is updated by minimizing the temporal difference (TD) loss:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
$$

Where:
- $\theta$: Q-network parameters
- $\theta^-$: Target network parameters
- $\gamma$: Discount factor
- $D$: Experience replay buffer

When `use_double_dqn` is enabled, the online network selects the best next action and the target network evaluates it, reducing Q-value overestimation.

### Exploration (ε-greedy)

DQN explores with an ε-greedy schedule that decays linearly from `start_epsilon` to `end_epsilon` over `decay_steps`, managed by an internal linear scheduler. When `n_step > 1`, bootstrapping uses the discounted return over $n$ steps.

## Configuration

The DQN configuration is provided by the `DQNConfig` class in [`cares_reinforcement_learning/algorithm/configurations.py`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/configurations.py).

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 1e-3 | Learning rate for the Q-network optimizer |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `tau` | float | 1.0 | Soft-update coefficient (1.0 = hard update) |
| `batch_size` | int | 32 | Mini-batch size sampled from the replay buffer |
| `target_update_freq` | int | 1000 | Steps between target network updates |
| `max_grad_norm` | float \| None | None | Optional gradient norm clipping |
| `use_double_dqn` | int | 0 | Enable Double DQN action selection |
| `n_step` | int | 1 | Number of steps for n-step bootstrapping |

### Exploration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_epsilon` | float | 1.0 | Initial exploration rate |
| `end_epsilon` | float | 1e-3 | Final exploration rate |
| `decay_steps` | int | 100000 | Steps over which epsilon decays linearly |

### Prioritized Experience Replay (PER)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_per_buffer` | int | 0 | Enable prioritized experience replay |
| `per_alpha` | float | 0.6 | PER prioritization exponent |
| `min_priority` | float | 1e-6 | Minimum priority value |

### Network Configuration

The Q-network architecture is configured through `network_config` (an `MLPConfig`). See [MLP Configuration](../user_guide/mlp_configuration.md) for how to specify layer types, activation functions, and residual connections.

## Running DQN

### With the Command-Line Interface (recommended)

The library is configuration-driven. The quickest way to train a DQN agent is through the `cares-rl` CLI:

```bash
# Train DQN on CartPole-v1 with default hyperparameters
cares-rl train cli --gym openai --task CartPole-v1 DQN

# Override hyperparameters directly from the command line
cares-rl train cli --gym openai --task CartPole-v1 DQN --lr 0.001 --gamma 0.99 --batch_size 32

# Train with full reproducibility via configuration files
cares-rl train config --data_path ~/my_experiment/
```

For more details on the `cares-rl` CLI and configuration files, see the [Experiments guide](../user_guide/experiment.md).

### Programmatic Usage

Algorithms are created through the [`AlgorithmFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/algorithm_factory.py) and memories through the [`MemoryFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/memory/memory_factory.py). The factory builds the correct network and algorithm from the configuration:

```python
import numpy as np

from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
from cares_reinforcement_learning.algorithm.configurations import DQNConfig
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.experience import SingleAgentExperience
from cares_reinforcement_learning.types.observation import SARLObservation

# 1. Configure the algorithm
config = DQNConfig(lr=1e-3, gamma=0.99, batch_size=32)

# 2. Build the agent and replay buffer from the config
factory = AlgorithmFactory()
agent = factory.create_network(
    observation_size={"image": None, "vector": observation_size},  # int obs dim
    action_num=action_num,  # discrete action count
    config=config,
)

memory_buffer = MemoryFactory().create_memory(config)

# 3. Training loop
observation = SARLObservation(vector_state=env.reset())
episode = 0
episode_step = 0
episode_return = 0.0

for step in range(total_steps):
    action_sample = agent.act(observation)
    action = action_sample.action

    next_obs, reward, done, truncated, _ = env.step(action)
    next_observation = SARLObservation(vector_state=next_obs)

    experience = SingleAgentExperience(
        observation=observation,
        next_observation=next_observation,
        action=action,
        reward=float(reward),
        done=bool(done),
        truncated=bool(truncated),
        train_data={},
        info={},
    )
    memory_buffer.add(experience)

    episode_return += reward
    episode_step += 1
    observation = next_observation

    if done or truncated:
        episode += 1
        observation = SARLObservation(vector_state=env.reset())
        episode_return = 0.0
        episode_step = 0

    # Train once enough transitions are available
    if len(memory_buffer) >= config.batch_size:
        episode_context = EpisodeContext(
            training_step=step,
            episode=episode,
            episode_steps=episode_step,
            episode_reward=episode_return,
            episode_done=bool(done),
        )
        metrics = agent.train(memory_buffer, episode_context)
```

Note: `agent.act()` returns an [`ActionSample`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/types/action.py) — access the selected action through `action_sample.action`. Observations are wrapped in `SARLObservation` with the vector state under `vector_state`.

## Stability Metrics

`agent.train()` returns a dictionary of metrics. Monitor the following to assess DQN training stability:

### Loss Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `loss` | Gradually decreases then stabilizes | Sudden spikes, continuous growth, NaN |
| `q_value_mean` | Increases during early learning, then plateaus | Monotonic decrease, divergence |
| `q_value_max` | Should remain bounded | Exponential growth (overestimation) |

### TD-Error Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `td_error_mean` | Tends toward 0 as learning progresses | Persistent large magnitude |
| `td_error_abs_mean` | Decreases and stabilizes | Large and growing (unstable target) |
| `overestimation_gap` | Small when using Double DQN | Large when overestimation is present |

### Exploration Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `epsilon` | Linearly decays from `start_epsilon` to `end_epsilon` | Not decaying (stuck at 1.0) |

### Performance Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `episode_return` | Improves over time, with noise | No improvement after exploration decay |
| `evaluation_return` | Smoother version of episode return | Consistently below random baseline |

## Common Issues and Solutions

### 1. Q-value Divergence

**Symptom**: `q_value_max` grows exponentially, loss becomes NaN.

**Causes**:
- Learning rate too high
- Target network updating too frequently
- Missing gradient clipping

**Solutions**:
- Reduce `lr` to 1e-4 or lower
- Increase `target_update_freq` to 2000+
- Enable gradient clipping with `max_grad_norm=10.0`
- Enable `use_double_dqn` to reduce overestimation

### 2. No Learning Progress

**Symptom**: Episode return stays at baseline after exploration decay.

**Causes**:
- `decay_steps` too small (exploration ends too early)
- `batch_size` too small for the task
- Network architecture too simple for the task

**Solutions**:
- Increase `decay_steps` to 200k–500k
- Increase `batch_size` to 64 or 128
- Enlarge the `network_config` MLP (e.g. 256 hidden units)

### 3. Training Instability

**Symptom**: Performance oscillates wildly, frequent catastrophic forgetting.

**Causes**:
- High variance in reward signal
- Unstable bootstrapping targets

**Solutions**:
- Apply reward scaling or normalization
- Enable `use_double_dqn` for more stable targets
- Try `DuelingDQN` or `Rainbow` variants for improved stability

## Variants

For improved performance, consider these DQN variants available in the library:

- **DoubleDQN**: Reduces Q-value overestimation by decoupling action selection from evaluation
- **DuelingDQN**: Separates state-value and advantage estimation for better value learning
- **Rainbow**: Combines Double, Dueling, PER, NoisyNet, Distributional (C51) and Multi-step
- **PERDQN**: Prioritized experience replay for more efficient learning
- **NoisyNet**: Parameter-space exploration via noisy linear layers (ε-greedy disabled)
- **C51 / QRDQN**: Distributional RL — learn a distribution over returns instead of the expectation
- **NoisyNet and Rainbow** use noisy linear layers and typically run with `start_epsilon=end_epsilon=0`

See the [Algorithms overview](index.md) for the full list of available algorithms.

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
2. van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
3. Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML*.
4. Schaul, T., et al. (2016). Prioritized Experience Replay. *ICLR*.
5. Bellemare, M., et al. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*.
