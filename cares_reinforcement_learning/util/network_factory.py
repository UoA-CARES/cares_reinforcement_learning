import inspect
import logging
import sys

import torch

from cares_reinforcement_learning.util.configurations import AlgorithmConfig


# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name


def create_DQN(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.value import DQN
    from cares_reinforcement_learning.networks.DQN import Network

    network = Network(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQN(
        network=network, gamma=config.gamma, network_lr=config.lr, device=device
    )
    return agent


def create_DuelingDQN(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.value import DQN
    from cares_reinforcement_learning.networks.DuelingDQN import DuelingNetwork

    network = DuelingNetwork(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQN(
        network=network, gamma=config.gamma, network_lr=config.lr, device=device
    )
    return agent


def create_DoubleDQN(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.value import DoubleDQN
    from cares_reinforcement_learning.networks.DoubleDQN import Network

    network = Network(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DoubleDQN(
        network=network,
        gamma=config.gamma,
        network_lr=config.lr,
        tau=config.tau,
        device=device,
    )
    return agent


def create_PPO(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PPO
    from cares_reinforcement_learning.networks.PPO import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPO(
        actor_network=actor,
        critic_network=critic,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        updates_per_iteration=config.updates_per_iteration,
        eps_clip=config.eps_clip,
        action_num=action_num,
        device=device,
    )
    return agent

def create_DynaSAC_MaxBatchReweight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_MaxBatchReweight
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_MaxBatchReweight(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        device=device,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        num_samples=config.num_samples,
    )
    return agent

def create_DynaSAC_ExaBatchReweight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_ExaBatchReweight
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_ExaBatchReweight(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        device=device,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        num_samples=config.num_samples,
    )
    return agent


def create_DynaSAT_BatchReweight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAT_BatchReweight
    from cares_reinforcement_learning.networks.SAC import Actor, TriCritic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneReward

    actor = Actor(observation_size, action_num)
    critic = TriCritic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAT_BatchReweight(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        device=device,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        num_samples=config.num_samples,
    )
    return agent


def create_DynaSAT(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAT
    from cares_reinforcement_learning.networks.SAC import Actor, TriCritic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneReward

    actor = Actor(observation_size, action_num)
    critic = TriCritic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        lr=config.world_model_lr,
        device=device,
    )

    agent = DynaSAT(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        num_samples=config.num_samples,
        device=device,
    )
    return agent


def create_DynaSAC_BatchReweight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_BatchReweight
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_BatchReweight(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        device=device,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        num_samples=config.num_samples,
    )
    return agent


def create_DynaSAC_Var(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_Var
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_Var(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        device=device,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        num_samples=config.num_samples,
    )
    return agent


def create_DynaSAC(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneReward(
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
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        num_samples=config.num_samples,
        device=device,
    )
    return agent



def create_SAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import SAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SAC(
        actor_network=actor,
        critic_network=critic,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        gamma=config.gamma,
        tau=config.tau,
        reward_scale=config.reward_scale,
        action_num=action_num,
        device=device,
    )
    return agent


def create_DDPG(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import DDPG
    from cares_reinforcement_learning.networks.DDPG import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPG(
        actor_network=actor,
        critic_network=critic,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        device=device,
    )
    return agent


def create_TD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import TD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        device=device,
    )
    return agent


def create_NaSATD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import NaSATD3
    from cares_reinforcement_learning.networks.NaSATD3 import (
        Actor,
        Critic,
        Decoder,
        Encoder,
    )

    encoder = Encoder(latent_dim=config.latent_size)
    decoder = Decoder(latent_dim=config.latent_size)

    actor = Actor(config.latent_size, action_num, encoder)
    critic = Critic(config.latent_size, action_num, encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = NaSATD3(
        encoder_network=encoder,
        decoder_network=decoder,
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        ensemble_size=config.ensemble_size,
        action_num=action_num,
        latent_size=config.latent_size,
        intrinsic_on=config.intrinsic_on,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        encoder_lr=config.encoder_lr,
        decoder_lr=config.decoder_lr,
        epm_lr=config.epm_lr,
        device=device,
    )
    return agent


def create_CTD4(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import CTD4
    from cares_reinforcement_learning.networks.CTD4 import (
        Actor,
        EnsembleCritic,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble_critics = EnsembleCritic(
        config.ensemble_size, observation_size, action_num
    )

    actor = Actor(observation_size, action_num)

    agent = CTD4(
        actor_network=actor,
        ensemble_critics=ensemble_critics,
        action_num=action_num,
        gamma=config.gamma,
        tau=config.tau,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        fusion_method=config.fusion_method,
        device=device,
    )

    return agent


def create_RDTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import RDTD3
    from cares_reinforcement_learning.networks.RDTD3 import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RDTD3(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_PERTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PERTD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PERTD3(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_LAPTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LAPTD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = LAPTD3(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_LAPSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LAPSAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = LAPSAC(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        reward_scale=config.reward_scale,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        device=device,
    )
    return agent


def create_PALTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PALTD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PALTD3(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_MAPERTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import MAPERTD3
    from cares_reinforcement_learning.networks.MAPERTD3 import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = MAPERTD3(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_REDQ(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import REDQ
    from cares_reinforcement_learning.networks.REDQ import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = REDQ(
        actor_network=actor,
        critic_network=critic,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        ensemble_size=config.ensemble_size,
        num_sample_critics=config.num_sample_critics,
        action_num=action_num,
        device=device,
    )
    return agent


def create_TQC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import TQC
    from cares_reinforcement_learning.networks.TQC import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num, config.num_quantiles, config.num_nets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TQC(
        actor_network=actor,
        critic_network=critic,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        gamma=config.gamma,
        tau=config.tau,
        top_quantiles_to_drop=config.top_quantiles_to_drop,
        action_num=action_num,
        device=device,
    )
    return agent


def create_PERSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import PERSAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PERSAC(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_RDSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import RDSAC
    from cares_reinforcement_learning.networks.RDSAC import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RDSAC(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_MAPERSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import MAPERSAC
    from cares_reinforcement_learning.networks.MAPERSAC import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = MAPERSAC(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        device=device,
    )
    return agent


def create_LA3PTD3(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LA3PTD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = LA3PTD3(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        prioritized_fraction=config.prioritized_fraction,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        device=device,
    )
    return agent


def create_LA3PSAC(observation_size, action_num, config: AlgorithmConfig):
    from cares_reinforcement_learning.algorithm.policy import LA3PSAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = LA3PSAC(
        actor_network=actor,
        critic_network=critic,
        gamma=config.gamma,
        tau=config.tau,
        reward_scale=config.reward_scale,
        per_alpha=config.per_alpha,
        min_priority=config.min_priority,
        prioritized_fraction=config.prioritized_fraction,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        device=device,
    )
    return agent


class NetworkFactory:
    def create_network(self, observation_size, action_num, config: AlgorithmConfig):
        algorithm = config.algorithm

        agent = None
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isfunction(obj):
                if name == f"create_{algorithm}":
                    agent = obj(observation_size, action_num, config)

        if agent is None:
            logging.warning(f"Unkown failed to return None: returned {agent}")

        return agent
