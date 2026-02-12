"""
Double DQN (Double Deep Q-Network)
------------------------------------

Original Paper: https://arxiv.org/abs/1509.06461

Code Based On: https://github.com/dxyang/DQN_pytorch

Double DQN modifies DQN to reduce overestimation bias
caused by the max operator in the Bellman target.

Core Problem:
- Standard DQN uses:
      y = r + γ max_a Q_target(s', a)
- The same network both selects and evaluates the max action.
- This introduces positive bias due to noise in Q estimates.

Core Idea:
- Decouple action selection from action evaluation.
- Use the online network to select the best action.
- Use the target network to evaluate that action.

Target Update:
    a* = argmax_a Q_online(s', a)
    y  = r + γ Q_target(s', a*)

This simple change reduces systematic overestimation.

Architecture:
- Same as DQN:
      Q_online(s,a)
      Q_target(s,a)
- Replay buffer and ε-greedy exploration unchanged.

Loss:
      L = (Q_online(s,a) - y)^2

Key Behaviour:
- Lower overestimation bias.
- More stable value learning.
- Often improves final performance over vanilla DQN.

Double DQN = DQN with decoupled action selection
             and action evaluation in the target.
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.DoubleDQN import Network
from cares_reinforcement_learning.util.configurations import DoubleDQNConfig


class DoubleDQN(DQN):
    def __init__(
        self,
        network: Network,
        config: DoubleDQNConfig,
        device: torch.device,
    ):
        super().__init__(network, config, device)
