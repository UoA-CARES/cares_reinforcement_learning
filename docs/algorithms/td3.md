# Twin Delayed Deep Deterministic Policy Gradient (TD3)

## Overview

Twin Delayed Deep Deterministic Policy Gradient (TD3) is an off-policy actor-critic algorithm for continuous control that improves DDPG-style learning stability by addressing value overestimation and brittle policy updates. It uses two critics (taking the minimum), target policy smoothing, and delayed actor updates.

**Paper**: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018)

**Action Space**: Continuous (bounded to $[-1, 1]$)

**Policy Type**: Off-policy, deterministic actor-critic

!!! note "Implementation detail"
    In this library, TD3 is implemented for **bounded continuous control**. The actor network ends with a $\tanh$ output layer, so actions lie in $[-1, 1]$. During training, exploration is handled by adding Gaussian noise to the action (see `action_noise_*` parameters below); when using `evaluation=True` in `act()`, no noise is added. Target policy smoothing uses a separate noise scheduler (`policy_noise_*`) clipped by `policy_noise_clip`. When using a Gymnasium-style environment, wrap actions so that the environment preserves this $[-1, 1]$ convention.

## How It Works

TD3 is DDPG + twin critics + target action smoothing + delayed policy updates.

### Key Components

1. **Actor Network**: A deterministic policy $\mu_\theta(s)$ mapping states to actions (with a $\tanh$ output layer for the $[-1, 1]$ bound).

2. **Twin Critics**: Two Q-networks $Q_{\phi_1}(s,a)$, $Q_{\phi_2}(s,a)$; the minimum is used in targets to reduce overestimation bias.

3. **Target Networks**: Copies of the actor and both critics, softly updated.

4. **Replay Buffer**: Stores off-policy experience; transitions are sampled (uniformly or with PER) for each update.

### Clipped Double Q-Learning (Twin Critics)

The critic target uses the conservative estimate:

$$
y = r + \gamma (1-d) \min\left( Q_{\phi_1'}(s', a'), Q_{\phi_2'}(s', a') \right)
$$

Both critics are trained with the same target, which reduces the overestimation bias typical of single-critic deterministic methods.

### Target Policy Smoothing

To avoid exploiting sharp Q-function peaks, the target action is perturbed with clipped noise:

$$
a' = \text{clip}\left( \mu_{\theta'}(s') + \text{clip}(\varepsilon, -c, c), -1, 1 \right), \qquad \varepsilon \sim \mathcal{N}(0, \sigma_{\text{policy}})
$$

Where:
- $\sigma_{\text{policy}}$ is controlled by the `policy_noise_*` scheduler
- $c$ is `policy_noise_clip`

### Delayed Policy Updates

- The critics are updated every training step.
- The actor (and target networks) are updated only every `policy_update_freq` training steps (default 2), so the critic has time to become more accurate first.

### Exploration

- Exploration noise is added **externally** to the actor's action during training.
- The noise scale follows an exponential scheduler (`action_noise_start` → `action_noise_end` over `action_noise_decay` training steps).
- The noisy action is clipped to $[-1, 1]$.

### Target Network Update

Target networks are softly updated whenever the actor is updated:

$$
\theta_{\text{target}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{target}}
$$

## Configuration

The TD3 configuration is provided by the `TD3Config` class in [`cares_reinforcement_learning/algorithm/configurations.py`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/configurations.py).

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_lr` | float | 3e-4 | Learning rate for the actor optimizer |
| `critic_lr` | float | 3e-4 | Learning rate for the critic optimizer |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `tau` | float | 0.005 | Soft target update coefficient |
| `policy_update_freq` | int | 2 | Training steps between actor/target updates |

### Exploration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_noise_start` | float | 0.1 | Initial Gaussian noise scale (std) added to actions |
| `action_noise_end` | float | 0.1 | Final exploration noise scale after decay |
| `action_noise_decay` | int | 1 | Training steps over which the exploration noise decays |

### Target Policy Smoothing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_noise_start` | float | 0.2 | Initial target smoothing noise scale (std) |
| `policy_noise_end` | float | 0.2 | Final smoothing noise scale after decay |
| `policy_noise_decay` | int | 1 | Training steps over which the smoothing noise decays |
| `policy_noise_clip` | float | 0.5 | Maximum magnitude of the clipped smoothing noise |

### Prioritized Experience Replay (PER)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_per_buffer` | int | 0 | Enable prioritized experience replay |
| `per_sampling_strategy` | str | "stratified" | PER sampling strategy |
| `per_weight_normalisation` | str | "batch" | Importance-weight normalization |
| `beta` | float | 0.4 | PER importance-sampling exponent |
| `per_alpha` | float | 0.6 | PER prioritization exponent |
| `min_priority` | float | 1e-6 | Minimum priority value |

### Shared Parameters (from `AlgorithmConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 256 | Batch size sampled from the replay buffer per update |
| `buffer_size` | int | 1000000 | Replay buffer capacity |
| `G` | int | 1 | Updates per training step (UTD ratio) |
| `number_steps_per_train_policy` | int | 1 | Steps collected per policy update call |
| `max_steps_exploration` | int | 1000 | Maximum steps of pure exploration before training starts |

### Network Configuration

The actor and critic architectures are configured through `actor_config` and `critic_config` (both `MLPConfig`, default 256→256 for the actor with a $\tanh$ output, 256→256→1 for the critics). See [MLP Configuration](../user_guide/mlp_configuration.md) for how to specify layer types and activation functions.

## Running TD3

### With the Command-Line Interface (recommended)

The library is configuration-driven. The quickest way to train a TD3 agent is through the `cares-rl` CLI:

```bash
# Train TD3 on Pendulum-v1 (continuous control) with default hyperparameters
cares-rl train cli --gym openai --task Pendulum-v1 TD3

# Override hyperparameters directly from the command line
cares-rl train cli --gym openai --task Pendulum-v1 TD3 --actor_lr 3e-4 --critic_lr 3e-4 --policy_update_freq 2

# Train with full reproducibility via configuration files
cares-rl train config --data_path ~/my_experiment/
```

For more details on the `cares-rl` CLI and configuration files, see the [Experiments guide](../user_guide/experiment.md).

### Programmatic Usage

Algorithms are created through the [`AlgorithmFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/algorithm_factory.py) and memories through the [`MemoryFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/memory/memory_factory.py). The factory builds the deterministic actor and twin critics from the configuration. Because TD3 is off-policy, experiences are stored in the replay buffer and the agent can be trained at any point:

```python
import numpy as np

from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
from cares_reinforcement_learning.algorithm.configurations import TD3Config
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.experience import SingleAgentExperience
from cares_reinforcement_learning.types.observation import SARLObservation

# 1. Configure the algorithm
config = TD3Config(actor_lr=3e-4, critic_lr=3e-4)

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

Note: `agent.act()` returns an [`ActionSample`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/types/action.py); the action is available at `action_sample.action`. Unlike on-policy algorithms such as PPO, no `log_prob`/`value` extras are required — pass an empty `train_data={}`. The `episode_context.training_step` is used by both the exploration and smoothing noise schedulers, so increment it with each `train()` call.

## Stability Metrics

`agent.train()` returns a dictionary of metrics. Monitor the following to assess TD3 training stability:

### Noise Diagnostics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `action_noise` | Decays per scheduler | N/A (report only) |
| `policy_noise` | Decays per scheduler | N/A (report only) |
| `target_noise_abs_mean` | Small smoothing magnitude | N/A (report only) |
| `target_noise_clip_frac` | Low fraction of clipped noise | Consistently high early (clip too small or noise too large) |

### Critic Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `critic_loss_one` / `critic_loss_two` / `critic_loss_total` | Decrease then stabilize | Continuous growth, NaN |
| `q1_mean` / `q2_mean` | Grow toward the true return scale | Exploding or oscillating Q-values |
| `q_twin_gap_abs_mean` | Small and stable | Growing gap (critics diverging / inconsistent) |
| `target_q1_mean` / `target_q2_mean` / `target_q_twin_gap_abs_mean` | Stable targets | Large unstable gap (target drift / OOD actions) |
| `td1_abs_mean` / `td2_abs_mean` | Decrease over time | Persistent growth or spikes (critic instability) |
| `q_target_mean` / `q_target_std` | Stable Bellman target scale | Drift upward without reward improvement |

### Actor Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `actor_loss` | Negative, magnitude shrinks as Q improves | Becomes large positive (unstable actor update) |
| `actor_q_mean` | Increases over training | Flat or decreasing (weak learning signal) |
| `dq_da_abs_mean` / `dq_da_norm_mean` / `dq_da_norm_p95` | Small positive gradient magnitude | ~0 early (no signal) or very large (unstable) |
| `pi_action_saturation_frac` | Low fraction of actions at ±1 | Consistently > 0.8 (actor slamming bounds) |
| `pi_action_mean` / `pi_action_std` / `pi_action_abs_mean` | Actions stay within $[-1, 1]$ | Actions pinned at bounds for long periods |

### Performance Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `episode_return` | Improves over time | No improvement after many updates |
| `evaluation_return` | Smoother improvement | Consistently below baseline |

## Common Issues and Solutions

### 1. Actor Updates Too Frequently / Unstable

**Symptom**: `actor_loss` oscillates, `dq_da_*` metrics spike.

**Causes**:
- Critic not accurate enough before actor updates
- `policy_update_freq` too low

**Solutions**:
- Increase `policy_update_freq` (e.g. 3–4)
- Reduce `actor_lr`
- Reduce `tau` for slower target updates

### 2. Overestimation Still Present

**Symptom**: Q-values consistently higher than observed returns; policy behaves erratically.

**Causes**:
- Smoothing noise too small (`policy_noise_start` too low)
- Twin critics not used effectively

**Solutions**:
- Increase `policy_noise_start` / `policy_noise_end`
- Verify `policy_noise_clip` is not too tight
- Reduce `critic_lr`

### 3. Target Noise Clipped Too Often

**Symptom**: `target_noise_clip_frac` consistently high early in training.

**Causes**:
- `policy_noise_clip` too small relative to `policy_noise`

**Solutions**:
- Increase `policy_noise_clip` (e.g. 0.5–0.7)
- Reduce `policy_noise_start`

### 4. No Learning Progress

**Symptom**: `actor_q_mean` flat, episode return stays at baseline.

**Causes**:
- `actor_lr` too low
- Exploration noise too small
- Network architecture too small for the task

**Solutions**:
- Increase `actor_lr` to 1e-3
- Increase `action_noise_start`
- Enlarge `actor_config` / `critic_config` MLPs

## Comparison with Other Algorithms

| Aspect | TD3 | DDPG | SAC | PPO |
|--------|-----|------|-----|-----|
| Policy Type | Off-policy | Off-policy | Off-policy | On-policy |
| Policy | Deterministic | Deterministic | Stochastic (max-entropy) | Stochastic |
| Critic Count | 2 (min) | 1 | 2 (min) | 1 (value) |
| Target Smoothing | Yes | No | No | N/A |
| Delayed Updates | Yes | No | No | N/A |
| Action Space | Continuous | Continuous | Continuous | Continuous |
| Sample Efficiency | Medium-High | Medium | High | Low |
| Stability | High | Low-Medium | High | High |
| Implementation Complexity | Medium | Low | High | Medium |

**When to choose TD3**:
- Continuous control where DDPG is unstable due to overestimation
- Deterministic policies with conservative updates are preferred
- A simpler, more stable alternative to SAC when tuning budget is limited

## References

1. Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML* / *arXiv preprint arXiv:1802.09477*.
2. Lillicrap, T. P., et al. (2016). Continuous Control with Deep Reinforcement Learning. *ICLR* / *arXiv preprint arXiv:1509.02971*.
3. Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *ICML* / *arXiv preprint arXiv:1801.01290*.
