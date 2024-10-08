"""
This module provides functions to create different types of reinforcement learning agents
with their corresponding network architectures.
"""

import copy
import inspect
import logging
import sys

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.util.configurations import AlgorithmConfig

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name


def create_DQN(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.value import DQN
    from cares_reinforcement_learning.networks.DQN import Network

    network = Network(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = DQN(network=network, config=config, device=device)
    return agent


def create_DuelingDQN(observation_size, action_num, config: AlgorithmConfig):
    """
    Original paper https://arxiv.org/abs/1511.06581
    """
    from cares_reinforcement_learning.algorithm.value import DQN
    from cares_reinforcement_learning.networks.DuelingDQN import DuelingNetwork

    network = DuelingNetwork(
        observation_size, action_num, hidden_size=config.hidden_size
    )

    device = hlp.get_device()
    agent = DQN(network=network, config=config, device=device)
    return agent


def create_DoubleDQN(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.value import DoubleDQN
    from cares_reinforcement_learning.networks.DoubleDQN import Network

    network = Network(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = DoubleDQN(
        network=network,
        config=config,
        device=device,
    )
    return agent


def create_PPO(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PPO
    from cares_reinforcement_learning.networks.PPO import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = PPO(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DynaSAC(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.
    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC
    from cares_reinforcement_learning.networks.DynaSAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldReward

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()

    world_model = EnsembleWorldReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        lr=config.world_model_lr,
        device=device,
    )

    agent = DynaSAC(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        config=config,
        device=device,
    )
    return agent


def create_SAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import SAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = SAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_SACAE(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import SACAE
    from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
    from cares_reinforcement_learning.networks.SACAE import Actor, Critic

    ae_factory = AEFactory()
    autoencoder = ae_factory.create_autoencoder(
        observation_size=observation_size["image"], config=config.autoencoder_config
    )

    actor_encoder = copy.deepcopy(autoencoder.encoder)
    critic_encoder = copy.deepcopy(autoencoder.encoder)

    vector_observation_size = (
        observation_size["vector"] if config.vector_observation else 0
    )

    actor = Actor(
        vector_observation_size,
        actor_encoder,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(
        vector_observation_size,
        critic_encoder,
        action_num,
        hidden_size=config.hidden_size,
    )

    device = hlp.get_device()
    agent = SACAE(
        actor_network=actor,
        critic_network=critic,
        decoder_network=autoencoder.decoder,
        config=config,
        device=device,
    )
    return agent


def create_SACD(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import SACD
    from cares_reinforcement_learning.networks.SACD import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = hlp.get_device()
    agent = SACD(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DDPG(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import DDPG
    from cares_reinforcement_learning.networks.DDPG import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = DDPG(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_TD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import TD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_TD3AE(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import TD3AE
    from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
    from cares_reinforcement_learning.networks.TD3AE import Actor, Critic

    ae_factory = AEFactory()
    autoencoder = ae_factory.create_autoencoder(
        observation_size=observation_size["image"], config=config.autoencoder_config
    )

    actor_encoder = copy.deepcopy(autoencoder.encoder)
    critic_encoder = copy.deepcopy(autoencoder.encoder)

    vector_observation_size = (
        observation_size["vector"] if config.vector_observation else 0
    )

    actor = Actor(
        vector_observation_size,
        actor_encoder,
        action_num,
        hidden_size=config.hidden_size,
    )
    critic = Critic(
        vector_observation_size,
        critic_encoder,
        action_num,
        hidden_size=config.hidden_size,
    )

    device = hlp.get_device()
    agent = TD3AE(
        actor_network=actor,
        critic_network=critic,
        decoder_network=autoencoder.decoder,
        config=config,
        device=device,
    )
    return agent


def create_NaSATD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import NaSATD3
    from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
    from cares_reinforcement_learning.networks.NaSATD3 import Actor, Critic

    ae_factory = AEFactory()
    autoencoder = ae_factory.create_autoencoder(
        observation_size=observation_size["image"], config=config.autoencoder_config
    )

    vector_observation_size = (
        observation_size["vector"] if config.vector_observation else 0
    )

    actor = Actor(
        vector_observation_size,
        action_num,
        autoencoder,
        hidden_size=config.hidden_size,
    )
    critic = Critic(
        vector_observation_size,
        action_num,
        autoencoder,
        hidden_size=config.hidden_size,
    )

    device = hlp.get_device()
    agent = NaSATD3(
        autoencoder=autoencoder,
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_CTD4(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import CTD4
    from cares_reinforcement_learning.networks.CTD4 import Actor, EnsembleCritic

    device = hlp.get_device()

    ensemble_critics = EnsembleCritic(
        config.ensemble_size,
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
    )

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)

    agent = CTD4(
        actor_network=actor,
        ensemble_critics=ensemble_critics,
        config=config,
        device=device,
    )

    return agent


def create_RDTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import RDTD3
    from cares_reinforcement_learning.networks.RDTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = RDTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_RDSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import RDSAC
    from cares_reinforcement_learning.networks.RDSAC import Actor, Critic

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = RDSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_PERTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PERTD3
    from cares_reinforcement_learning.networks.PERTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = PERTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_PERSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PERSAC
    from cares_reinforcement_learning.networks.PERSAC import Actor, Critic

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = PERSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LAPTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LAPTD3
    from cares_reinforcement_learning.networks.LAPTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = LAPTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LAPSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LAPSAC
    from cares_reinforcement_learning.networks.LAPSAC import Actor, Critic

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = LAPSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_PALTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PALTD3
    from cares_reinforcement_learning.networks.PALTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = PALTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_MAPERTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import MAPERTD3
    from cares_reinforcement_learning.networks.MAPERTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = MAPERTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_MAPERSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import MAPERSAC
    from cares_reinforcement_learning.networks.MAPERSAC import Actor, Critic

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = MAPERSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_REDQ(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import REDQ
    from cares_reinforcement_learning.networks.REDQ import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = REDQ(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_TQC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import TQC
    from cares_reinforcement_learning.networks.TQC import Actor, Critic

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(
        observation_size,
        action_num,
        config.num_quantiles,
        config.num_nets,
        hidden_size=config.hidden_size,
    )

    device = hlp.get_device()
    agent = TQC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LA3PTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LA3PTD3
    from cares_reinforcement_learning.networks.LA3PTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, hidden_size=config.hidden_size)
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = LA3PTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LA3PSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LA3PSAC
    from cares_reinforcement_learning.networks.LA3PSAC import Actor, Critic

    actor = Actor(
        observation_size,
        action_num,
        hidden_size=config.hidden_size,
        log_std_bounds=config.log_std_bounds,
    )
    critic = Critic(observation_size, action_num, hidden_size=config.hidden_size)

    device = hlp.get_device()
    agent = LA3PSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


class NetworkFactory:
    def create_network(
        self,
        observation_size,
        action_num,
        config: AlgorithmConfig,
    ):
        algorithm = config.algorithm

        agent = None
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isfunction(obj):
                if name == f"create_{algorithm}":
                    agent = obj(observation_size, action_num, config)

        if agent is None:
            logging.warning(f"Unkown {agent} algorithm.")

        return agent
