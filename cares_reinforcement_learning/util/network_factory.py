import inspect
import logging
import sys

import cares_reinforcement_learning.util.configurations as acf
import cares_reinforcement_learning.util.helpers as hlp

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name
###################################
#         DQN Algorithms          #
###################################


def create_STEVE_MEAN(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import STEVE_MEAN
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models.ensemble_world_ensemble_sas_reward import \
        EnsembleWorldEnsembleSASReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldEnsembleSASReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_world_models=config.num_world_models,
        num_reward_models=config.num_reward_models,
        lr=config.world_model_lr,
        device=device,
    )

    agent = STEVE_MEAN(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        gamma=config.gamma,
        tau=config.tau,
        action_num=action_num,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        horizon=config.horizon,
        L=config.num_critic_models,
        device=device,
    )
    return agent


def create_DynaSAC_SA(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_SA
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models.ensmeble_sa_world import EnsembleWorldAndOneSAReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneSAReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        lr=config.world_model_lr,
        device=device,
    )

    agent = DynaSAC_SA(
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




def create_DynaSAC_SAS(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_SAS
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneSASReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneSASReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        lr=config.world_model_lr,
        device=device,
    )

    agent = DynaSAC_SAS(
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


def create_DynaSAC_SAS_Immersive_Weight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_SAS_Immersive_Weight
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneSASReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneSASReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_SAS_Immersive_Weight(
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
        threshold_scale=config.threshold_scale,
        reweight_critic=config.reweight_critic,
        reweight_actor=config.reweight_actor,
        mode=config.mode,
        sample_times=config.sample_times,
    )
    return agent




def create_DynaSAC_BIVReweight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_BIVReweight
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneNSReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneNSReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_BIVReweight(
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
        threshold_scale=config.threshold_scale,
        reweight_critic=config.reweight_critic,
        reweight_actor=config.reweight_actor,
        mode=config.mode,
        sample_times=config.sample_times,
    )
    return agent


def create_DynaSAC_SUNRISEReweight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_SUNRISEReweight
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneNSReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneNSReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_SUNRISEReweight(
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
        threshold_scale=config.threshold_scale,
        reweight_critic=config.reweight_critic,
        reweight_actor=config.reweight_actor,
        mode=config.mode,
        sample_times=config.sample_times,
    )
    return agent


def create_DynaSAC_UWACReweight(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.

    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_UWACReweight
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneNSReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneNSReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        device=device,
        lr=config.world_model_lr,
    )

    agent = DynaSAC_UWACReweight(
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
        threshold_scale=config.threshold_scale,
        reweight_critic=config.reweight_critic,
        reweight_actor=config.reweight_actor,
        mode=config.mode,
        sample_times=config.sample_times,
    )
    return agent


def create_DynaSAC(observation_size, action_num, config: AlgorithmConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.
    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC_NS
    from cares_reinforcement_learning.networks.SAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldAndOneNSReward

    actor = Actor(observation_size, action_num)
    critic = Critic(observation_size, action_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = EnsembleWorldAndOneNSReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        lr=config.world_model_lr,
        device=device,
    )

    agent = DynaSAC_NS(
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


def create_SAC(observation_size, action_num, config: acf.SACConfig):
    from cares_reinforcement_learning.algorithm.policy import SAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = SAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DynaSAC(observation_size, action_num, config: acf.DynaSACConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.
    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC
    from cares_reinforcement_learning.networks.DynaSAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldReward

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)
    device = hlp.get_device()

    world_model = EnsembleWorldReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        lr=config.world_model_lr,
        device=device,
    )
    agent=DynaSAC(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        config=config,
        device=device,
    )
    return agent


class NetworkFactory:
    def create_network(
        self,
        observation_size,
        action_num: int,
        config: acf.AlgorithmConfig,
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
