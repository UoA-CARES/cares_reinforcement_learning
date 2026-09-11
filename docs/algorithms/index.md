# Algorithms

CARES Reinforcement Learning provides a modular set of reinforcement learning algorithms for single-agent (value-based, policy-based) and multi-agent settings, as well as unsupervised skill discovery (USD) methods. All algorithms share a consistent interface through the [`AlgorithmFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/algorithm/algorithm_factory.py) and [`MemoryFactory`](https://github.com/UoA-CARES/cares_reinforcement_learning/blob/main/cares_reinforcement_learning/memory/memory_factory.py).

## Value-Based Algorithms (Discrete Actions)

| Algorithm | Key Features | Reference |
|-----------|--------------|-----------|
| [DQN](dqn.md) | Experience replay, target network, ε-greedy | [Mnih et al., 2015](https://www.nature.com/articles/nature14236) |
| DoubleDQN | Decoupled action selection and evaluation | [van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461) |
| DuelingDQN | State-value and advantage decomposition | [Wang et al., 2016](https://arxiv.org/abs/1511.06581) |
| NoisyNet | Noisy linear layers for exploration | [Fortunato et al., 2018](https://arxiv.org/abs/1706.10295) |
| PERDQN | Prioritized experience replay + Double DQN | [Schaul et al., 2016](https://arxiv.org/abs/1511.05952) |
| C51 | Distributional RL (categorical) | [Bellemare et al., 2017](https://arxiv.org/abs/1707.06887) |
| QRDQN | Distributional RL (quantile regression) | [Dabney et al., 2018](https://arxiv.org/abs/1710.10044) |
| Rainbow | Double, Dueling, PER, NoisyNet, C51, multi-step | [Hessel et al., 2018](https://arxiv.org/abs/1710.02298) |

## Policy-Based Algorithms (Continuous Actions)

| Algorithm | Key Features | Reference |
|-----------|--------------|-----------|
| [PPO](ppo.md) | Clipped surrogate objective, GAE, on-policy | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) |
| DDPG | Deterministic policy gradient | [Lillicrap et al., 2016](https://arxiv.org/abs/1509.02971) |
| TD3 | Twin delayed DDPG, target policy smoothing | [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477) |
| SAC | Soft actor-critic, maximum entropy | [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) |
| SACD | SAC for discrete action spaces | [Christodoulou, 2019](https://arxiv.org/abs/1910.07207) |
| SACAE | SAC with an autoencoder for image observations | — |
| TD3AE | TD3 with an autoencoder for image observations | — |
| TD7 | TD3 with state disentanglement and normalized value targets | [Ota et al., 2023](https://openreview.net/forum?id=Hk2V4EqKEA) |
| REDQ | Randomized ensemble double Q-learning | [Chen et al., 2021](https://arxiv.org/abs/2101.05982) |
| DroQ | SAC variant with dropout-regularized critics | [Hiraoka et al., 2022](https://arxiv.org/abs/2110.02034) |
| CrossQ | SAC variant using batch normalization in critic | [Bhatt et al., 2023](https://arxiv.org/abs/2302.02259) |
| TQC | SAC variant with truncated quantile critics | [Kuznetsov et al., 2020](https://arxiv.org/abs/2005.04269) |
| CTD4 | Continuous tempered distributional derivative | [Cetin et al., 2024](https://arxiv.org/abs/2409.11548) |

## Policy-Based Variants (Prioritized / Research Extensions)

These algorithms extend the base algorithms (mostly TD3 and SAC) with prioritized experience replay, loss-adjusted prioritization, or other research modifications. They share the same usage interface as their base algorithm.

| Algorithm | Description |
|-----------|-------------|
| PERTD3 | TD3 with prioritized experience replay |
| PERSAC | SAC with prioritized experience replay |
| LAPTD3 | TD3 variant with loss-adjusted prioritization |
| LAPSAC | SAC variant with loss-adjusted prioritization |
| LA3PTD3 | TD3 variant with look-ahead loss-adjusted prioritization |
| LA3PSAC | SAC variant with look-ahead loss-adjusted prioritization |
| MAPERTD3 | TD3 variant with multi-step prioritized experience replay |
| MAPERSAC | SAC variant with multi-step prioritized experience replay |
| PALTD3 | TD3 variant with prioritized attention learning |
| RDSAC | SAC variant with prioritized experience replay and regularization |
| RDTD3 | TD3 variant with regularization |
| SDAR | SAC variant with a state-dependent action selector |
| NaSATD3 | TD3 variant for image observations with intrinsic rewards |

## Multi-Agent Algorithms

The library provides multi-agent reinforcement learning (MARL) algorithms for cooperative/competitive settings. See the [MARL Cross Play](../user_guide/marl_cross_play.md) guide for usage and configuration.

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| MADDPG | CTDE | Multi-agent DDPG |
| MATD3 | CTDE | Multi-agent TD3 |
| MASAC | CTDE | Multi-agent SAC |
| MAPPO | CTDE | Multi-agent PPO |
| M3DDPG | CTDE | Minimax multi-agent DDPG |
| QMIX | Value-based | Monotonic value function factorization |
| IMARL | — | Individual vs. shared reward MARL |
| CrossMARL | — | Cross-play MARL |

## Unsupervised Skill Discovery (USD)

| Algorithm | Key Features | Reference |
|-----------|--------------|-----------|
| DIAYN | Skill discovery via mutual information | [Eysenbach et al., 2019](https://arxiv.org/abs/1802.06070) |
| DADS | Dynamics-aware skill discovery | [Sharma et al., 2020](https://arxiv.org/abs/1907.01657) |

## Selecting an Algorithm

### For discrete action spaces
- **Start with DQN** as a baseline.
- Use **Rainbow** for the strongest combination of DQN improvements.
- Use **C51 / QRDQN** if you need distributional value estimates.
- Use **NoisyNet** for parameter-space exploration (no ε-greedy).

### For continuous action spaces
- **Start with TD3 or SAC** as stable baselines (SAC for sample efficiency, TD3 for stability).
- Use **PPO** for on-policy learning with simple, robust tuning.
- Use **REDQ / DroQ / CrossQ** for sample-efficient large-batch training.
- Use **TD3AE / SACAE / NaSATD3** for image-based observations.

### For multi-agent problems
- Start with **MADDPG / MASAC** for actor-critic style learning.
- Use **QMIX** for value-based cooperative settings.
- Use **MAPPO** for on-policy multi-agent learning.

### For skill discovery
- Use **DIAYN / DADS** to learn reusable skills without external rewards.

## Stability Metrics

All algorithms log a shared set of metrics for monitoring training stability:

- **Loss**: Policy/actor loss and value/critic loss
- **Returns**: Episode return (mean, max, min over evaluation episodes)
- **Q-values**: Estimated Q-values (mean, max, min) for value-based methods
- **Exploration**: Epsilon (for value-based) or entropy / log_std (for policy-based)
- **Learning rate**: Current learning rate

See the individual algorithm pages for algorithm-specific metrics and stability guidance.
