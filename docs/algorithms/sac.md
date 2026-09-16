# Soft Actor-Critic (SAC)

## Overview

Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm for continuous control that augments the reward objective with an entropy term to encourage exploration. It learns a stochastic Gaussian policy and uses twin Q-critics with automatic temperature tuning, making it one of the most sample-efficient and stable off-policy algorithms in continuous control.

**Paper**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018)

**Action Space**: Continuous (bounded to $[-1, 1]$)

**Policy Type**: Off-policy, stochastic (maximum entropy) actor-critic

!!! note "Implementation detail"
    In this library, SAC is implemented for **bounded continuous control**. The actor is a Gaussian policy in pre-squash space passed through a $\tanh$ squashing function, so actions lie in $[-1, 1]$; log-probabilities include the correct change-of-variables correction. The entropy temperature $\alpha$ is **learned automatically** (initialized at 1.0, target entropy $= -\text{action\_num}$). When calling `act(evaluation=True)`, the **mean** of the policy is used as the action; otherwise a stochastic sample is returned. When using a Gymnasium-style environment, wrap actions so that the environment preserves this $[-1, 1]$ convention.

## How It Works

SAC maximizes the expected return *plus* the expected policy entropy:

$$
J(\pi) = \mathbb{E}\left[ \sum_t \left( r_t + \alpha H(\pi(\cdot|s_t)) \right) \right]
$$

where $\alpha$ is the entropy temperature. This encourages broader exploration while remaining off-policy via the replay buffer.

### Key Components

1. **Actor Network**: A stochastic policy $\pi(a|s)$ — a Gaussian policy with tanh squashing (`TanhGaussianPolicy`). The per-action log standard deviation is bounded by `log_std_bounds`.

2. **Twin Critics**: Two Q-networks $Q_{\phi_1}(s,a)$, $Q_{\phi_2}(s,a)$; the minimum of the two is used in targets to reduce overestimation bias.

3. **Target Critics**: Softly-updated copies of the critics used for stable bootstrapping.

4. **Automatic Temperature Tuning**: The temperature $\alpha$ is adjusted to match a target entropy ($-\text{action\_num}$) via gradient descent on `log_alpha`.

5. **Replay Buffer**: Stores off-policy experience; transitions are sampled (uniformly or with PER) for each update.

### Critic Update

The critics minimize the mean squared Bellman error against a soft target:

$$
y = r \cdot \text{reward\_scale} + \gamma (1-d) \left( \min(Q_{\phi_1'}, Q_{\phi_2'})(s', a') - \alpha \log \pi(a'|s') \right)
$$

$$
L(\phi_i) = \mathbb{E}\left[ \left( Q_{\phi_i}(s, a) - y \right)^2 \right], \quad i \in \{1, 2\}
$$

Where:
- $a' \sim \pi(\cdot|s')$: next action sampled from the current policy
- $\alpha \log \pi(a'|s')$: entropy regularization inside the target (typically negative)
- `reward_scale`: optional reward scaling applied before discounting

### Actor Update

The actor minimizes:

$$
J_\pi = \mathbb{E}\left[ \alpha \log \pi(a|s) - \min(Q_{\phi_1}, Q_{\phi_2})(s, a) \right]
$$

### Temperature Update

The temperature is updated to match the target entropy:

$$
J_\alpha = \mathbb{E}\left[ -\alpha \left( \log \pi(a|s) + H_{\text{target}} \right) \right], \qquad H_{\text{target}} = -\text{action\_num}
$$

### Update Frequencies

- The actor and temperature are updated every `policy_update_freq` training steps.
- Target critics are softly updated every `target_update_freq` training steps:
  $\phi' \leftarrow \tau \phi + (1-\tau) \phi'$

## Configuration

The SAC configuration is provided by the `SACConfig` class in [`cares_reinforcement_learning/algorithm/configurations.py`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/configurations.py).

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_lr` | float | 3e-4 | Learning rate for the actor optimizer |
| `critic_lr` | float | 3e-4 | Learning rate for the critic optimizer |
| `alpha_lr` | float | 3e-4 | Learning rate for the temperature $\alpha$ |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `tau` | float | 0.005 | Soft target update coefficient |
| `reward_scale` | float | 1.0 | Reward scaling applied in the Bellman target |
| `log_std_bounds` | list[float] | [-20.0, 2.0] | Bounds on the learnable log standard deviation |
| `policy_update_freq` | int | 1 | Training steps between actor/temperature updates |
| `target_update_freq` | int | 1 | Training steps between target critic updates |

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

The actor and critic architectures are configured through `actor_config` and `critic_config` (both `MLPConfig`, default 256→256). See [MLP Configuration](../user_guide/mlp_configuration.md) for how to specify layer types and activation functions.

## Running SAC

### With the Command-Line Interface (recommended)

The library is configuration-driven. The quickest way to train a SAC agent is through the `cares-rl` CLI:

```bash
# Train SAC on Pendulum-v1 (continuous control) with default hyperparameters
cares-rl train cli --gym openai --task Pendulum-v1 SAC

# Override hyperparameters directly from the command line
cares-rl train cli --gym openai --task Pendulum-v1 SAC --actor_lr 3e-4 --critic_lr 3e-4

# Train with full reproducibility via configuration files
cares-rl train config --data_path ~/my_experiment/
```

For more details on the `cares-rl` CLI and configuration files, see the [Experiments guide](../user_guide/experiment.md).

### Programmatic Usage

Algorithms are created through the [`AlgorithmFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/algorithm_factory.py) and memories through the [`MemoryFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/memory/memory_factory.py). The factory builds the stochastic actor and twin critics from the configuration. Because SAC is off-policy, experiences are stored in the replay buffer and the agent can be trained at any point:

```python
import numpy as np

from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
from cares_reinforcement_learning.algorithm.configurations import SACConfig
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.experience import SingleAgentExperience
from cares_reinforcement_learning.types.observation import SARLObservation

# 1. Configure the algorithm
config = SACConfig(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4)

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

for step in range(total_steps):
    # Act: stochastic sample during training
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
            training_step=step,
            episode=step,
            episode_steps=1,
            episode_reward=0.0,
            episode_done=False,
        )
        metrics = agent.train(memory_buffer, episode_context)
```

Note: `agent.act()` returns an [`ActionSample`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/types/action.py); the action is available at `action_sample.action`. During training, `act()` returns a stochastic sample; call `agent.act(observation, evaluation=True)` to use the policy **mean** for deterministic evaluation. No `log_prob`/`value` extras are required — pass an empty `train_data={}`.

## Stability Metrics

`agent.train()` returns a dictionary of metrics. Monitor the following to assess SAC training stability:

### Critic Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `critic_loss_one` / `critic_loss_two` / `critic_loss_total` | Decrease then stabilize | Continuous growth, NaN |
| `q1_mean` / `q2_mean` | Grow toward the true return scale | Exploding or oscillating Q-values |
| `q_twin_gap_abs_mean` | Small and stable | Growing gap (critics diverging / inconsistent) |
| `target_q1_mean` / `target_q2_mean` / `target_q_twin_gap_abs_mean` | Stable targets | Large unstable gap (target drift / OOD actions) |
| `soft_target_value_mean` | Stable target value | Drift upward without reward improvement (check `reward_scale`, `gamma`) |
| `td1_abs_mean` / `td2_abs_mean` | Decrease over time | Persistent growth or spikes (critic instability) |
| `q_target_mean` / `q_target_std` | Stable Bellman target scale | Drift upward without reward improvement |

### Actor & Temperature Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `actor_loss` | Negative, magnitude shrinks as Q improves | Becomes large positive (unstable actor update) |
| `min_qf_pi_mean` | Increases over training | Flat or decreasing (weak learning signal) |
| `log_pi_mean` | Moderate (more negative = more entropy) | Collapses toward 0 (policy too deterministic) |
| `entropy_gap_mean` | Near 0 (entropy matches target) | Consistently > 0 (entropy too low — $\alpha$ should rise) or < 0 |
| `alpha` / `log_alpha` | Converges to a stable value | Oscillating wildly or hitting extreme values |
| `alpha_loss` | Small and stable | Persistent large values |
| `dq_da_abs_mean` / `dq_da_norm_mean` / `dq_da_norm_p95` | Small positive gradient magnitude | ~0 early (no signal) or very large (unstable) |
| `pi_action_saturation_frac` | Low fraction of actions at ±1 | Consistently > 0.8 (policy slamming bounds) |
| `qf_pi_gap_abs_mean` | Small | Large gap (critics disagree on current policy actions) |

### Performance Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `episode_return` | Improves over time | No improvement after many updates |
| `evaluation_return` | Smoother improvement | Consistently below baseline |

## Common Issues and Solutions

### 1. Entropy Collapse (Policy Becomes Deterministic Too Early)

**Symptom**: `log_pi_mean` approaches 0, `entropy_gap_mean` consistently > 0.

**Causes**:
- `log_std_bounds` too tight
- Reward scale too large (overwhelms entropy bonus)

**Solutions**:
- Widen `log_std_bounds` (e.g. [-20, 2])
- Reduce `reward_scale`
- Lower `alpha_lr` for slower temperature adaptation

### 2. Unstable Twin Critics

**Symptom**: `q_twin_gap_abs_mean` grows over training.

**Causes**:
- Critic learning rate too high
- Reward scale too large

**Solutions**:
- Reduce `critic_lr` to 1e-4
- Scale rewards (e.g. `reward_scale` 0.1–1.0 for the task)
- Increase `target_update_freq` / reduce `tau` for slower targets

### 3. Q-Value Explosion

**Symptom**: `q1_mean`/`q2_mean` grow without bound; `td1_abs_mean`/`td2_abs_mean` spike.

**Causes**:
- `gamma` too close to 1 for the task horizon
- Reward scale too large
- Critic too large/too fast

**Solutions**:
- Lower `gamma` to match the task horizon
- Scale rewards
- Reduce `critic_lr`

### 4. No Learning Progress

**Symptom**: `min_qf_pi_mean` flat, episode return stays at baseline.

**Causes**:
- `actor_lr` too low
- Network architecture too small for the task
- Exploration steps insufficient

**Solutions**:
- Increase `actor_lr` to 1e-3
- Enlarge `actor_config` / `critic_config` MLPs
- Increase `max_steps_exploration`

## Comparison with Other Algorithms

| Aspect | SAC | DDPG | TD3 | PPO |
|--------|-----|------|-----|-----|
| Policy Type | Off-policy | Off-policy | Off-policy | On-policy |
| Policy | Stochastic (max-entropy) | Deterministic | Deterministic | Stochastic |
| Critic Count | 2 (min) | 1 | 2 (min) | 1 (value) |
| Action Space | Continuous | Continuous | Continuous | Continuous |
| Sample Efficiency | High | Medium | Medium-High | Low |
| Stability | High | Low-Medium | High | High |
| Implementation Complexity | High | Low | Medium | Medium |
| Hyperparameter Sensitivity | Medium | High | Medium | Low |

**When to choose SAC**:
- Sample-efficient off-policy learning for continuous control
- Tasks that benefit from stochastic exploration
- When automatic temperature tuning is preferred over hand-tuned exploration

## References

1. Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *ICML* / *arXiv preprint arXiv:1801.01290*.
2. Haarnoja, T., et al. (2018). Soft Actor-Critic Algorithms and Applications. *arXiv preprint arXiv:1812.05905*.
3. Fujimoto, S., et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods (TD3). *ICML* / *arXiv preprint arXiv:1802.09477*.
