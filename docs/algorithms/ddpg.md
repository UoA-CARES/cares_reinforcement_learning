# Deep Deterministic Policy Gradient (DDPG)

## Overview

Deep Deterministic Policy Gradient (DDPG) is an off-policy actor-critic algorithm for continuous action spaces. It combines a deterministic policy with deep function approximation and target networks, learning a deterministic policy $\pi(s) \to a$ alongside a Q-function $Q(s, a)$.

**Paper**: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2016)

**Action Space**: Continuous (bounded to $[-1, 1]$)

**Policy Type**: Off-policy, deterministic actor-critic

!!! note "Implementation detail"
    In this library, DDPG is implemented for **bounded continuous control**. The actor network ends with a $\tanh$ output layer, so actions lie in $[-1, 1]$. During training, exploration is handled by adding Gaussian noise to the action (see the exploration parameters below); when using `evaluation=True` in `act()`, no noise is added. When using a Gymnasium-style environment, wrap actions so that the environment preserves this $[-1, 1]$ convention.

## How It Works

DDPG learns a deterministic policy $\mu_\theta(s)$ and a Q-function $Q_\phi(s, a)$. Stability comes from replay buffers and slowly-updated target networks.

### Key Components

1. **Actor Network**: A deterministic policy $\mu_\theta(s)$ that maps states directly to actions (with a $\tanh$ output layer for the $[-1, 1]$ bound).

2. **Critic Network**: Estimates the action-value function $Q_\phi(s, a)$.

3. **Target Networks**: Copies of the actor and critic, softly updated to stabilize bootstrapping.

4. **Replay Buffer**: Stores off-policy experience; transitions are sampled uniformly for each update.

### Critic Update

The critic minimizes the mean squared Bellman error:

$$
L(\phi) = \mathbb{E}\left[ \left( Q_\phi(s, a) - y \right)^2 \right], \qquad y = r + \gamma (1 - d) Q_{\phi'}(s', \mu_{\theta'}(s'))
$$

Where:
- $y$: Bellman target computed with the **target** actor and critic
- $d$: Done flag (`done`), which zeroes the bootstrap term at terminal states

### Actor Update

The actor is updated by the deterministic policy gradient, implemented as maximizing the critic's estimate of the current state:

$$
\nabla_\theta J \approx \mathbb{E}\left[ \nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s) \right]
$$

In practice this is implemented by minimizing $-Q_\phi(s, \mu_\theta(s))$ over the batch.

### Exploration

- Exploration noise is added **externally** to the actor's action during training.
- The noise scale follows an exponential scheduler (`action_noise_start` → `action_noise_end` over `action_noise_decay` training steps).
- The noisy action is clipped to $[-1, 1]$.

### Target Network Update

Target networks are softly updated after every training step:

$$
\theta_{\text{target}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{target}}
$$

## Configuration

The DDPG configuration is provided by the `DDPGConfig` class in [`cares_reinforcement_learning/algorithm/configurations.py`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/configurations.py).

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_lr` | float | 1e-4 | Learning rate for the actor optimizer |
| `critic_lr` | float | 1e-3 | Learning rate for the critic optimizer |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `tau` | float | 0.005 | Soft target update coefficient |

### Exploration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_noise_start` | float | 0.2 | Initial Gaussian noise scale (std) added to actions |
| `action_noise_end` | float | 0.05 | Final noise scale after decay |
| `action_noise_decay` | int | 1000000 | Training steps over which the noise decays exponentially |

### Shared Parameters (from `AlgorithmConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 256 | Batch size sampled from the replay buffer per update |
| `buffer_size` | int | 1000000 | Replay buffer capacity |
| `G` | int | 1 | Updates per training step (UTD ratio) |
| `number_steps_per_train_policy` | int | 1 | Steps collected per policy update call |
| `max_steps_exploration` | int | 1000 | Maximum steps of pure exploration before training starts |

### Network Configuration

The actor and critic architectures are configured through `actor_config` and `critic_config` (both `MLPConfig`). The default actor is a 1024→1024 MLP with a $\tanh$ output; the default critic is a 1024→1024 MLP with a single scalar output. See [MLP Configuration](../user_guide/mlp_configuration.md) for how to specify layer types and activation functions.

## Running DDPG

### With the Command-Line Interface (recommended)

The library is configuration-driven. The quickest way to train a DDPG agent is through the `cares-rl` CLI:

```bash
# Train DDPG on Pendulum-v1 (continuous control) with default hyperparameters
cares-rl train cli --gym openai --task Pendulum-v1 DDPG

# Override hyperparameters directly from the command line
cares-rl train cli --gym openai --task Pendulum-v1 DDPG --actor_lr 1e-4 --critic_lr 1e-3

# Train with full reproducibility via configuration files
cares-rl train config --data_path ~/my_experiment/
```

For more details on the `cares-rl` CLI and configuration files, see the [Experiments guide](../user_guide/experiment.md).

### Programmatic Usage

Algorithms are created through the [`AlgorithmFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/algorithm_factory.py) and memories through the [`MemoryFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/memory/memory_factory.py). The factory builds the correct actor/critic networks and algorithm from the configuration. Because DDPG is off-policy, experiences are stored in the replay buffer and the agent can be trained at any point:

```python
import numpy as np

from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
from cares_reinforcement_learning.algorithm.configurations import DDPGConfig
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.experience import SingleAgentExperience
from cares_reinforcement_learning.types.observation import SARLObservation

# 1. Configure the algorithm
config = DDPGConfig(actor_lr=1e-4, critic_lr=1e-3)

# 2. Build the agent and replay buffer from the config
factory = AlgorithmFactory()
agent = factory.create_network(
    observation_size={"image": None, "vector": observation_size},  # int obs dim
    action_num=action_num,  # continuous action dimension
    config=config,
)

memory_buffer = MemoryFactory().create_memory(config)

# 3. Off-policy training loop
observation = SARLObservation(vector_state=env.reset())
training_step = 0

for step in range(total_steps):
    # Act (adds exploration noise internally unless evaluation=True)
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
    observation = next_observation

    # Train whenever the buffer has enough data (off-policy)
    if len(memory_buffer) >= config.batch_size:
        episode_context = EpisodeContext(
            training_step=training_step,  # drives the noise scheduler decay
            episode=step,
            episode_steps=1,
            episode_reward=0.0,
            episode_done=False,
        )
        metrics = agent.train(memory_buffer, episode_context)
        training_step += 1
```

Note: `agent.act()` returns an [`ActionSample`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/types/action.py); the action is available at `action_sample.action`. Unlike on-policy algorithms such as PPO, no `log_prob`/`value` extras are required — pass an empty `train_data={}`. The `episode_context.training_step` is used by the exploration noise scheduler, so increment it with each `train()` call.

## Stability Metrics

`agent.train()` returns a dictionary of metrics. Monitor the following to assess DDPG training stability:

### Critic Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `critic_loss` | Decreases then stabilizes | Continuous growth, NaN |
| `q_mean` / `q_std` | Q-values grow toward the true return scale | Exploding or oscillating Q-values |
| `q_target_mean` / `q_target_std` | Stable Bellman target scale | Drift upward without reward improvement (check `gamma`, reward scale) |
| `td_mean` / `td_std` / `td_abs_mean` | `td_abs_mean` decreases over time | Persistent growth or spikes (critic instability) |

### Actor Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `actor_loss` | Negative, magnitude shrinks as Q improves | Becomes large positive (unstable actor update) |
| `actor_q_mean` | Increases over training | Flat or decreasing (weak learning signal) |
| `dq_da_abs_mean` / `dq_da_norm_mean` / `dq_da_norm_p95` | Small positive gradient magnitude | ~0 early (critic flat — no signal) or very large (critic too sharp) |
| `pi_action_saturation_frac` | Low fraction of actions at ±1 | Consistently > 0.8 (actor slamming bounds, weak gradient through tanh) |
| `pi_action_mean` / `pi_action_std` / `pi_action_abs_mean` | Actions stay within $[-1, 1]$ | Actions pinned at bounds for long periods |

### Performance Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `episode_return` | Improves over time | No improvement after many updates |
| `evaluation_return` | Smoother improvement | Consistently below baseline |

## Common Issues and Solutions

### 1. No Learning Progress / Flat Critic

**Symptom**: `actor_q_mean` stays flat, `dq_da_abs_mean` near 0.

**Causes**:
- `actor_lr` too low
- Critic not yet accurate enough to provide signal

**Solutions**:
- Increase `actor_lr` to 3e-4 or 1e-3
- Ensure enough exploration steps before training begins
- Check that the reward signal is properly scaled

### 2. Action Saturation at Bounds

**Symptom**: `pi_action_saturation_frac` consistently > 0.8, learning stalls.

**Causes**:
- Actor output saturating the $\tanh$ non-linearity
- Exploration noise too large

**Solutions**:
- Reduce `action_noise_start`
- Reduce `actor_lr`
- Check reward scaling (large rewards push actions to bounds)

### 3. Diverging Q-Values

**Symptom**: `q_mean` / `q_target_mean` explode, `td_abs_mean` grows.

**Causes**:
- Critic learning rate too high
- Reward scale too large
- `gamma` too close to 1 for the task horizon

**Solutions**:
- Reduce `critic_lr` to 3e-4 or 1e-4
- Scale rewards to a reasonable range
- Consider switching to TD3 (twin critics) if overestimation persists

### 4. Overestimation Bias

**Symptom**: Q-values are much higher than observed returns; policy behaves erratically.

**Causes**: Deterministic actor-critic overestimation (a known weakness of DDPG).

**Solutions**:
- Switch to **TD3** or **SAC**, which are designed to mitigate this
- Reduce `tau` (e.g. 0.001) for slower target updates

## Comparison with Other Algorithms

| Aspect | DDPG | TD3 | SAC | PPO |
|--------|------|-----|-----|-----|
| Policy Type | Off-policy | Off-policy | Off-policy | On-policy |
| Policy | Deterministic | Deterministic | Stochastic (max-entropy) | Stochastic |
| Critic Count | 1 | 2 (min) | 2 (min) | 1 (value) |
| Action Space | Continuous | Continuous | Continuous | Continuous |
| Sample Efficiency | Medium | Medium-High | High | Low |
| Stability | Low-Medium | High | High | High |
| Implementation Complexity | Low | Medium | High | Medium |

**When to choose DDPG**:
- Simple continuous-control baselines where implementation simplicity matters
- As a starting point before moving to TD3/SAC if instability appears
- When deterministic policies with external noise are sufficient

## References

1. Lillicrap, T. P., et al. (2016). Continuous Control with Deep Reinforcement Learning. *ICLR* / *arXiv preprint arXiv:1509.02971*.
2. Silver, D., et al. (2014). Deterministic Policy Gradient Algorithms. *ICML*.
3. Fujimoto, S., et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods (TD3). *ICML* / *arXiv preprint arXiv:1802.09477*.
