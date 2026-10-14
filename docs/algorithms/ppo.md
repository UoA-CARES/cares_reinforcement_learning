# Proximal Policy Optimization (PPO)

## Overview

Proximal Policy Optimization (PPO) is an on-policy policy gradient algorithm that alternates between sampling data through interaction with the environment and optimizing a clipped surrogate objective. PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, making it one of the most widely used RL algorithms in practice.

**Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)

**Action Space**: Continuous (bounded to $[-1, 1]$)

**Policy Type**: On-policy, policy gradient

!!! note "Implementation detail"
    In this library, PPO is implemented for **bounded continuous control**. The actor is a Gaussian policy in pre-squash space that is passed through a $\tanh$ squashing function, so actions lie in $[-1, 1]$. Log-probabilities include the correct change-of-variables correction for the $\tanh$ transformation. When using a Gymnasium-style environment, wrap actions so that the environment preserves this $[-1, 1]$ convention.

## How It Works

PPO learns a stochastic policy $\pi_\theta(a|s)$ that directly maps states to action distributions. The key innovation is the clipped surrogate objective, which prevents excessively large policy updates that can destabilize training.

### Key Components

1. **Actor Network**: Outputs the mean of a Gaussian distribution; the per-action standard deviations are maintained by a learnable `log_std` parameter optimized jointly with the actor.

2. **Critic Network**: Estimates the state-value function $V(s)$ for advantage estimation.

3. **Generalized Advantage Estimation (GAE)**: Computes advantage estimates by balancing bias and variance through a parameter $\lambda$.

4. **Rollout Buffer**: Collects on-policy experience (squashed actions, corrected log-probabilities and critic values) and is flushed after each policy update.

### Clipped Surrogate Objective

PPO optimizes the following objective:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: Probability ratio
- $A_t$: Advantage estimate (GAE)
- $\epsilon$: Clip parameter (`eps_clip`, typically 0.2)

The clipping ensures the policy update does not deviate too far from the old policy, improving training stability.

### Update Procedure

- Rollout collection is strictly on-policy using the current stochastic policy.
- Advantages are computed with GAE; returns for the critic are `advantage + value`.
- Advantages are normalized across the batch for numerical stability.
- Updates are performed over multiple epochs of minibatch SGD (`updates_per_iteration`).
- Gradient norm clipping is applied to both actor and critic (`max_grad_norm`).
- An optional entropy bonus encourages exploration (`entropy_start` / `entropy_end`, decayed by a linear scheduler over `entropy_decay` steps).
- If `target_kl` is set, minibatch/epoch updates stop early once the approximated KL exceeds the threshold.

## Configuration

The PPO configuration is provided by the `PPOConfig` class in [`cares_reinforcement_learning/algorithm/configurations.py`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/configurations.py).

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_lr` | float | 3e-4 | Learning rate for the actor optimizer |
| `critic_lr` | float | 1e-3 | Learning rate for the critic optimizer |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `eps_clip` | float | 0.2 | PPO clipping parameter ($\epsilon$) |
| `gae_lambda` | float | 0.95 | GAE lambda for advantage estimation |
| `max_grad_norm` | float \| None | 0.5 | Maximum gradient norm for clipping |
| `updates_per_iteration` | int | 10 | Optimization epochs per rollout |
| `minibatch_size` | int | 1000 | Minibatch size per gradient step |
| `number_steps_per_train_policy` | int | 10000 | Rollout steps collected per policy update |
| `target_kl` | float \| None | None | Early-stopping KL threshold (disabled by default) |

### Exploration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_start` | float | 0.0 | Initial entropy bonus coefficient |
| `entropy_end` | float | 0.0 | Final entropy bonus coefficient |
| `entropy_decay` | int | 0 | Steps over which the entropy coefficient decays |
| `log_std_bounds` | list[float] | [-5.0, -0.5] | Bounds on the learnable log standard deviation |

### Value Normalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_value_normalisation` | int | 0 | Enable running value normalization for the critic |

### Network Configuration

The actor and critic architectures are configured through `actor_config` and `critic_config` (both `MLPConfig`). See [MLP Configuration](../user_guide/mlp_configuration.md) for how to specify layer types and activation functions.

## Running PPO

### With the Command-Line Interface (recommended)

The library is configuration-driven. The quickest way to train a PPO agent is through the `cares-rl` CLI:

```bash
# Train PPO on Pendulum-v1 (continuous control) with default hyperparameters
cares-rl train cli --gym openai --task Pendulum-v1 PPO

# Override hyperparameters directly from the command line
cares-rl train cli --gym openai --task Pendulum-v1 PPO --actor_lr 3e-4 --critic_lr 1e-3 --eps_clip 0.2

# Train with full reproducibility via configuration files
cares-rl train config --data_path ~/my_experiment/
```

For more details on the `cares-rl` CLI and configuration files, see the [Experiments guide](../user_guide/experiment.md).

### Programmatic Usage

Algorithms are created through the [`AlgorithmFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/algorithm_factory.py) and memories through the [`MemoryFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/memory/memory_factory.py). Because PPO is on-policy, the rollout buffer is flushed once per policy update:

```python
import numpy as np

from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
from cares_reinforcement_learning.algorithm.configurations import PPOConfig
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.experience import SingleAgentExperience
from cares_reinforcement_learning.types.observation import SARLObservation

# 1. Configure the algorithm
config = PPOConfig(actor_lr=3e-4, critic_lr=1e-3)

# 2. Build the agent and rollout buffer from the config
factory = AlgorithmFactory()
agent = factory.create_network(
    observation_size={"image": None, "vector": observation_size},  # int obs dim
    action_num=action_num,  # continuous action dimension
    config=config,
)

memory_buffer = MemoryFactory().create_memory(config)

# 3. Collect an on-policy rollout and update
observation = SARLObservation(vector_state=env.reset())

for update in range(num_updates):
    # Collect rollout
    for step in range(config.number_steps_per_train_policy):
        action_sample = agent.act(observation)
        action = action_sample.action
        log_prob = action_sample.extras["log_prob"]
        value = action_sample.extras["value"]

        next_obs, reward, done, truncated, _ = env.step(action)
        next_observation = SARLObservation(vector_state=next_obs)

        experience = SingleAgentExperience(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=float(reward),
            done=bool(done),
            truncated=bool(truncated),
            train_data={"log_prob": log_prob, "value": value},
            info={},
        )
        memory_buffer.add(experience)
        observation = next_observation

    # On-policy update: flush the rollout buffer
    episode_context = EpisodeContext(
        training_step=update,
        episode=update,
        episode_steps=config.number_steps_per_train_policy,
        episode_reward=0.0,
        episode_done=False,
    )
    metrics = agent.train(memory_buffer, episode_context)
```

Note: `agent.act()` returns an [`ActionSample`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/types/action.py). The action is available at `action_sample.action`, and the on-policy quantities needed for training (`log_prob`, `value`) are returned in `action_sample.extras`.

## Stability Metrics

`agent.train()` returns a dictionary of metrics. Monitor the following to assess PPO training stability:

### Policy Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `actor_loss` | Negative, gradually approaches 0 | Becomes positive (policy worse than old), large magnitude |
| `approx_kl` | Small and stable | Consistently exceeds ~0.05 (updates too large) |
| `clip_frac` | 0.05–0.3, stable | Near 0 (no clipping) or consistently > 0.5 (too aggressive) |
| `entropy` | Stable, does not collapse too early | Drops to near 0 too quickly (premature convergence) |
| `log_std_mean` | Converges within `log_std_bounds` | Hitting bounds (exploration collapse or runaway noise) |

### Value Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `critic_loss` | Decreases then stabilizes | Continuous growth, NaN |

### Performance Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `episode_return` | Improves over time | No improvement after many updates |
| `evaluation_return` | Smoother improvement | Consistently below baseline |

## Common Issues and Solutions

### 1. Policy Collapse / Premature Convergence

**Symptom**: `entropy` drops to near 0, policy becomes deterministic, performance plateaus at a suboptimal level.

**Causes**:
- `entropy_start` too low
- Learning rate too high
- Too many epochs per rollout

**Solutions**:
- Increase `entropy_start` (and set `entropy_decay` for a scheduled decay)
- Reduce `actor_lr` to 1e-4 or 5e-5
- Reduce `updates_per_iteration` to 4–8

### 2. Large KL Divergence

**Symptom**: `approx_kl` consistently exceeds ~0.05, training unstable.

**Causes**:
- Learning rate too high
- Too many epochs per rollout
- Rollout too short (high variance)

**Solutions**:
- Reduce learning rate by 50%
- Reduce `updates_per_iteration` to 5–8
- Increase `number_steps_per_train_policy`
- Set `target_kl` (e.g. 0.02) to enable early stopping

### 3. Value Function Divergence

**Symptom**: `critic_loss` grows, values diverge from actual returns.

**Causes**:
- Critic learning rate too high
- Reward scale too large

**Solutions**:
- Reduce `critic_lr`
- Normalize or scale rewards
- Enable `use_value_normalisation`

### 4. No Learning Progress

**Symptom**: Episode return stays at baseline after many updates.

**Causes**:
- Rollout too short for the task
- Learning rate too low
- Network architecture too simple

**Solutions**:
- Increase `number_steps_per_train_policy`
- Increase learning rate to 5e-4 or 1e-3
- Enlarge the `actor_config` / `critic_config` MLPs

## Comparison with Other Algorithms

| Aspect | PPO | DQN | SAC | TD3 |
|--------|-----|-----|-----|-----|
| Policy Type | On-policy | Off-policy | Off-policy | Off-policy |
| Action Space | Continuous | Discrete | Continuous | Continuous |
| Sample Efficiency | Low | High | High | High |
| Stability | High | Medium | High | High |
| Implementation Complexity | Medium | Low | High | Medium |
| Hyperparameter Sensitivity | Low | Medium | High | Medium |

**When to choose PPO**:
- Continuous action spaces with bounded (normalized) actions
- On-policy learning is acceptable
- Simplicity and robustness are prioritized over sample efficiency
- Reproducibility and ease of tuning are important

## References

1. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
2. Schulman, J., et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.
3. Schulman, J., et al. (2015). Trust Region Policy Optimization. *ICML*.
