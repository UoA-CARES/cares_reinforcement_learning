"""
This module provides functions to create different types of reinforcement learning agents
with their corresponding network architectures.
"""

import inspect
import logging
import sys

import numpy as np

import cares_reinforcement_learning.algorithm.configurations as acf
import cares_reinforcement_learning.util.helpers as hlp

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name


###################################
#         DQN Algorithms          #
###################################
def create_DQN(observation_size, action_num, config: acf.DQNConfig):
    from cares_reinforcement_learning.algorithm.value import DQN
    from cares_reinforcement_learning.networks.DQN import Network

    network = Network(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = DQN(network=network, config=config, device=device)
    return agent


def create_PERDQN(observation_size, action_num, config: acf.PERDQNConfig):
    from cares_reinforcement_learning.algorithm.value import PERDQN
    from cares_reinforcement_learning.networks.PERDQN import Network

    network = Network(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = PERDQN(network=network, config=config, device=device)
    return agent


def create_DuelingDQN(observation_size, action_num, config: acf.DuelingDQNConfig):
    from cares_reinforcement_learning.algorithm.value import DuelingDQN
    from cares_reinforcement_learning.networks.DuelingDQN import Network

    network = Network(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = DuelingDQN(network=network, config=config, device=device)
    return agent


def create_DoubleDQN(observation_size, action_num, config: acf.DoubleDQNConfig):
    from cares_reinforcement_learning.algorithm.value import DoubleDQN
    from cares_reinforcement_learning.networks.DoubleDQN import Network

    network = Network(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = DoubleDQN(
        network=network,
        config=config,
        device=device,
    )
    return agent


def create_NoisyNet(observation_size, action_num, config: acf.NoisyNetConfig):
    from cares_reinforcement_learning.algorithm.value import NoisyNet
    from cares_reinforcement_learning.networks.NoisyNet import Network

    network = Network(observation_size["vector"], action_num, config)

    device = hlp.get_device()
    agent = NoisyNet(network=network, config=config, device=device)
    return agent


def create_C51(observation_size, action_num, config: acf.C51Config):
    from cares_reinforcement_learning.algorithm.value import C51
    from cares_reinforcement_learning.networks.C51 import Network

    network = Network(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = C51(network=network, config=config, device=device)
    return agent


def create_QRDQN(observation_size, action_num, config: acf.QRDQNConfig):
    from cares_reinforcement_learning.algorithm.value import QRDQN
    from cares_reinforcement_learning.networks.QRDQN import Network

    network = Network(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = QRDQN(network=network, config=config, device=device)
    return agent


def create_Rainbow(observation_size, action_num, config: acf.RainbowConfig):
    from cares_reinforcement_learning.algorithm.value import Rainbow
    from cares_reinforcement_learning.networks.Rainbow import Network

    network = Network(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = Rainbow(network=network, config=config, device=device)
    return agent


###################################
#         PPO Algorithms          #
###################################


def create_PPO(observation_size, action_num, config: acf.PPOConfig):
    from cares_reinforcement_learning.algorithm.policy import PPO
    from cares_reinforcement_learning.networks.PPO import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], config=config)

    device = hlp.get_device()
    agent = PPO(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


###################################
#         SAC Algorithms          #
###################################


def create_SAC(observation_size, action_num, config: acf.SACConfig):
    from cares_reinforcement_learning.algorithm.policy import SAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = SAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_SACAE(observation_size, action_num, config: acf.SACAEConfig):
    from cares_reinforcement_learning.algorithm.policy import SACAE
    from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
    from cares_reinforcement_learning.networks.SACAE import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    ae_config = config.autoencoder_config
    decoder = Decoder(
        observation_size["image"],
        out_dim=actor.encoder.out_dim,
        latent_dim=ae_config.latent_dim,
        num_layers=ae_config.num_layers,
        num_filters=ae_config.num_filters,
        kernel_size=ae_config.kernel_size,
    )

    device = hlp.get_device()
    agent = SACAE(
        actor_network=actor,
        critic_network=critic,
        decoder_network=decoder,
        config=config,
        device=device,
    )
    return agent


def create_PERSAC(observation_size, action_num, config: acf.PERSACConfig):
    from cares_reinforcement_learning.algorithm.policy import PERSAC
    from cares_reinforcement_learning.networks.PERSAC import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = PERSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_REDQ(observation_size, action_num, config: acf.REDQConfig):
    from cares_reinforcement_learning.algorithm.policy import REDQ
    from cares_reinforcement_learning.networks.REDQ import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    ensemble_critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = REDQ(
        actor_network=actor,
        ensemble_critic=ensemble_critic,
        config=config,
        device=device,
    )
    return agent


def create_TQC(observation_size, action_num, config: acf.TQCConfig):
    from cares_reinforcement_learning.algorithm.policy import TQC
    from cares_reinforcement_learning.networks.TQC import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = TQC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LAPSAC(observation_size, action_num, config: acf.LAPSACConfig):
    from cares_reinforcement_learning.algorithm.policy import LAPSAC
    from cares_reinforcement_learning.networks.LAPSAC import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = LAPSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LA3PSAC(observation_size, action_num, config: acf.LA3PSACConfig):
    from cares_reinforcement_learning.algorithm.policy import LA3PSAC
    from cares_reinforcement_learning.networks.LA3PSAC import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = LA3PSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_MAPERSAC(observation_size, action_num, config: acf.MAPERSACConfig):
    from cares_reinforcement_learning.algorithm.policy import MAPERSAC
    from cares_reinforcement_learning.networks.MAPERSAC import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = MAPERSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_RDSAC(observation_size, action_num, config: acf.RDSACConfig):
    from cares_reinforcement_learning.algorithm.policy import RDSAC
    from cares_reinforcement_learning.networks.RDSAC import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = RDSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DroQ(observation_size, action_num, config: acf.DroQConfig):
    from cares_reinforcement_learning.algorithm.policy import DroQ
    from cares_reinforcement_learning.networks.DroQ import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = DroQ(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_CrossQ(observation_size, action_num, config: acf.CrossQConfig):
    from cares_reinforcement_learning.algorithm.policy import CrossQ
    from cares_reinforcement_learning.networks.CrossQ import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = CrossQ(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_SDAR(observation_size, action_num, config: acf.SDARConfig):
    from cares_reinforcement_learning.algorithm.policy import SDAR
    from cares_reinforcement_learning.networks.SDAR import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = SDAR(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_SACD(observation_size, action_num, config: acf.SACDConfig):
    from cares_reinforcement_learning.algorithm.policy import SACD
    from cares_reinforcement_learning.networks.SACD import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = SACD(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DIAYN(observation_size, action_num, config: acf.DIAYNConfig):
    from cares_reinforcement_learning.algorithm.usd import DIAYN
    from cares_reinforcement_learning.networks.DIAYN import Discriminator

    sac_observation_size = {
        "vector": observation_size["vector"] + config.num_skills,
    }

    agent = create_SAC(sac_observation_size, action_num, config=config)

    discriminator = Discriminator(observation_size["vector"], config=config)

    device = hlp.get_device()
    agent = DIAYN(
        skills_agent=agent,
        discriminator_network=discriminator,
        config=config,
        device=device,
    )
    return agent


def create_DADS(observation_size, action_num, config: acf.DADSConfig):
    from cares_reinforcement_learning.algorithm.usd import DADS
    from cares_reinforcement_learning.networks.DADS import SkillDynamicsModel

    sac_observation_size = {
        "vector": observation_size["vector"] + config.z_dim,
    }

    agent = create_SAC(sac_observation_size, action_num, config=config)

    discriminator = SkillDynamicsModel(
        observation_size=observation_size["vector"],
        config=config,
    )

    device = hlp.get_device()
    agent = DADS(
        skills_agent=agent,
        discriminator_network=discriminator,
        config=config,
        device=device,
    )
    return agent


###################################
#         TD3 Algorithms          #
###################################


def create_DDPG(observation_size, action_num, config: acf.DDPGConfig):
    from cares_reinforcement_learning.algorithm.policy import DDPG
    from cares_reinforcement_learning.networks.DDPG import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = DDPG(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_TD3(observation_size, action_num, config: acf.TD3Config):
    from cares_reinforcement_learning.algorithm.policy import TD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_TD3AE(observation_size, action_num, config: acf.TD3AEConfig):
    from cares_reinforcement_learning.algorithm.policy import TD3AE
    from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
    from cares_reinforcement_learning.networks.TD3AE import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    ae_config = config.autoencoder_config
    decoder = Decoder(
        observation_size["image"],
        out_dim=actor.encoder.out_dim,
        latent_dim=ae_config.latent_dim,
        num_layers=ae_config.num_layers,
        num_filters=ae_config.num_filters,
        kernel_size=ae_config.kernel_size,
    )

    device = hlp.get_device()
    agent = TD3AE(
        actor_network=actor,
        critic_network=critic,
        decoder_network=decoder,
        config=config,
        device=device,
    )
    return agent


def create_NaSATD3(observation_size, action_num, config: acf.NaSATD3Config):
    from cares_reinforcement_learning.algorithm.policy import NaSATD3
    from cares_reinforcement_learning.networks.NaSATD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = NaSATD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_PERTD3(observation_size, action_num, config: acf.PERTD3Config):
    from cares_reinforcement_learning.algorithm.policy import PERTD3
    from cares_reinforcement_learning.networks.PERTD3 import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = PERTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LAPTD3(observation_size, action_num, config: acf.LAPTD3Config):
    from cares_reinforcement_learning.algorithm.policy import LAPTD3
    from cares_reinforcement_learning.networks.LAPTD3 import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = LAPTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_PALTD3(observation_size, action_num, config: acf.PALTD3Config):
    from cares_reinforcement_learning.algorithm.policy import PALTD3
    from cares_reinforcement_learning.networks.PALTD3 import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = PALTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LA3PTD3(observation_size, action_num, config: acf.LA3PTD3Config):
    from cares_reinforcement_learning.algorithm.policy import LA3PTD3
    from cares_reinforcement_learning.networks.LA3PTD3 import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = LA3PTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_MAPERTD3(observation_size, action_num, config: acf.MAPERTD3Config):
    from cares_reinforcement_learning.algorithm.policy import MAPERTD3
    from cares_reinforcement_learning.networks.MAPERTD3 import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = MAPERTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_RDTD3(observation_size, action_num, config: acf.RDTD3Config):
    from cares_reinforcement_learning.algorithm.policy import RDTD3
    from cares_reinforcement_learning.networks.RDTD3 import Actor, Critic

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)

    device = hlp.get_device()
    agent = RDTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_CTD4(observation_size, action_num, config: acf.CTD4Config):
    from cares_reinforcement_learning.algorithm.policy import CTD4
    from cares_reinforcement_learning.networks.CTD4 import Actor, Critic

    device = hlp.get_device()

    actor = Actor(observation_size["vector"], action_num, config=config)
    ensemble_critic = Critic(observation_size["vector"], action_num, config=config)

    agent = CTD4(
        actor_network=actor,
        ensemble_critic=ensemble_critic,
        config=config,
        device=device,
    )

    return agent


def create_TD7(observation_size, action_num, config: acf.TD7Config):
    from cares_reinforcement_learning.algorithm.policy import TD7
    from cares_reinforcement_learning.networks.TD7 import Actor, Critic, Encoder

    device = hlp.get_device()

    actor = Actor(observation_size["vector"], action_num, config=config)
    critic = Critic(observation_size["vector"], action_num, config=config)
    encoder = Encoder(observation_size["vector"], action_num, config=config)

    agent = TD7(
        actor_network=actor,
        critic_network=critic,
        encoder_network=encoder,
        config=config,
        device=device,
    )

    return agent


###################################
#         MARL Algorithms         #
###################################


def _build_actor_critic_mappings(
    all_agent_ids: list[str],
    env_teams: dict[str, list[str]],
    parameter_sharing_scope: str,
    algo_name: str,
) -> tuple[
    dict[str, str],  # agent_id_to_actor_id
    dict[str, list[str]],  # actor_id_to_agent_ids
    dict[str, str],  # agent_id_to_critic_id
    dict[str, list[str]],  # critic_id_to_agent_ids
]:
    """
    Build actor/critic ownership mappings for MARL parameter sharing.

    The mappings define:
        - which learning unit provides the ACTOR for each env agent,
        - which learning unit provides the CRITIC for each env agent,
        - and which env agents are assigned to each learning unit.

    Example environment
    -------------------
    Env agents:
        adversary_0
        adversary_1
        agent_0

    Teams:
        {
            "adversary": ["adversary_0", "adversary_1"],
            "agent": ["agent_0"],
        }

    ------------------------------------------------------------------
    individual
    ------------------------------------------------------------------

    One actor + one critic per env agent.

    agent_id_to_actor_id:
        {
            "adversary_0": "adversary_0",
            "adversary_1": "adversary_1",
            "agent_0": "agent_0",
        }

    actor_id_to_agent_ids:
        {
            "adversary_0": ["adversary_0"],
            "adversary_1": ["adversary_1"],
            "agent_0": ["agent_0"],
        }

    agent_id_to_critic_id:
        {
            "adversary_0": "adversary_0",
            "adversary_1": "adversary_1",
            "agent_0": "agent_0",
        }

    critic_id_to_agent_ids:
        {
            "adversary_0": ["adversary_0"],
            "adversary_1": ["adversary_1"],
            "agent_0": ["agent_0"],
        }

    ------------------------------------------------------------------
    team_critic
    ------------------------------------------------------------------

    One actor per env agent, but one shared critic per team.

    agent_id_to_actor_id:
        {
            "adversary_0": "adversary_0",
            "adversary_1": "adversary_1",
            "agent_0": "agent_0",
        }

    actor_id_to_agent_ids:
        {
            "adversary_0": ["adversary_0"],
            "adversary_1": ["adversary_1"],
            "agent_0": ["agent_0"],
        }

    agent_id_to_critic_id:
        {
            "adversary_0": "adversary",
            "adversary_1": "adversary",
            "agent_0": "agent",
        }

    critic_id_to_agent_ids:
        {
            "adversary": ["adversary_0", "adversary_1"],
            "agent": ["agent_0"],
        }

    ------------------------------------------------------------------
    team_all
    ------------------------------------------------------------------

    One shared actor + one shared critic per team.

    agent_id_to_actor_id:
        {
            "adversary_0": "adversary",
            "adversary_1": "adversary",
            "agent_0": "agent",
        }

    actor_id_to_agent_ids:
        {
            "adversary": ["adversary_0", "adversary_1"],
            "agent": ["agent_0"],
        }

    agent_id_to_critic_id:
        {
            "adversary_0": "adversary",
            "adversary_1": "adversary",
            "agent_0": "agent",
        }

    critic_id_to_agent_ids:
        {
            "adversary": ["adversary_0", "adversary_1"],
            "agent": ["agent_0"],
        }
    """
    if parameter_sharing_scope == "individual":
        actor_id_to_agent_ids = {agent_id: [agent_id] for agent_id in all_agent_ids}
        agent_id_to_actor_id = {agent_id: agent_id for agent_id in all_agent_ids}

        critic_id_to_agent_ids = {agent_id: [agent_id] for agent_id in all_agent_ids}
        agent_id_to_critic_id = {agent_id: agent_id for agent_id in all_agent_ids}

    elif parameter_sharing_scope == "team_critic":
        actor_id_to_agent_ids = {agent_id: [agent_id] for agent_id in all_agent_ids}
        agent_id_to_actor_id = {agent_id: agent_id for agent_id in all_agent_ids}

        critic_id_to_agent_ids = env_teams
        agent_id_to_critic_id = {
            agent_id: team_id
            for team_id, team_agent_ids in env_teams.items()
            for agent_id in team_agent_ids
        }

    elif parameter_sharing_scope == "team_all":
        actor_id_to_agent_ids = env_teams
        agent_id_to_actor_id = {
            agent_id: team_id
            for team_id, team_agent_ids in env_teams.items()
            for agent_id in team_agent_ids
        }

        critic_id_to_agent_ids = env_teams
        agent_id_to_critic_id = {
            agent_id: team_id
            for team_id, team_agent_ids in env_teams.items()
            for agent_id in team_agent_ids
        }

    else:
        raise ValueError(f"Unknown {algo_name} {parameter_sharing_scope=}")

    return (
        agent_id_to_actor_id,
        actor_id_to_agent_ids,
        agent_id_to_critic_id,
        critic_id_to_agent_ids,
    )


def _build_learning_units(
    actor_id_to_agent_ids: dict[str, list[str]],
    critic_id_to_agent_ids: dict[str, list[str]],
    observation_size,
    action_num: int,
    config,
    device,
    Algorithm,
    Actor,
    Critic,
    critic_uses_action_num=True,
) -> dict:
    learning_units = {}

    # A learning unit may exist because it owns:
    #   - an actor,
    #   - a critic,
    #   - or both.
    #
    # individual:
    #   actor IDs = critic IDs = env agents
    #
    # team_critic:
    #   actor IDs  = env agents
    #   critic IDs = teams
    #
    # team_all:
    #   actor IDs  = teams
    #   critic IDs = teams
    #
    # Therefore we build the union of all actor/critic IDs that require a
    # physical learning-unit container.
    learning_unit_ids = list(
        dict.fromkeys(
            list(actor_id_to_agent_ids.keys()) + list(critic_id_to_agent_ids.keys())
        )
    )

    for learning_unit_id in learning_unit_ids:
        # Recover a representative environment agent for network construction.
        #
        # The actor network API expects an env-agent ID so it can determine:
        #   - observation dimensions
        #   - action dimensions
        #   - agent-specific observation structure
        #
        # A learning unit is guaranteed to appear in at least one of:
        #   actor_id_to_agent_ids
        #   critic_id_to_agent_ids
        #
        # individual:
        #   learning_unit_id = adversary_0
        #   representative_agent_id = adversary_0
        #
        # team_critic:
        #   actor learning units:
        #       adversary_0 -> adversary_0
        #
        #   critic learning units:
        #       adversary -> adversary_0
        #
        # team_all:
        #   adversary -> adversary_0

        controlled_agent_ids = actor_id_to_agent_ids.get(
            learning_unit_id
        ) or critic_id_to_agent_ids.get(learning_unit_id)

        if controlled_agent_ids is None:
            raise ValueError(
                f"No controlled agents found for learning unit '{learning_unit_id}'"
            )

        # Recover the environment agents associated with this learning unit.
        #
        # We first check whether the learning unit owns an actor.
        # If not, we fall back to the critic ownership mapping.
        #
        # This is needed because in some configurations a learning unit may exist
        # purely due to actor ownership or purely due to critic ownership.
        representative_agent_id = controlled_agent_ids[0]

        actor = Actor(
            observation_size=observation_size,
            num_actions=action_num,
            config=config,
            agent_id=representative_agent_id,
        )

        if critic_uses_action_num:
            critic = Critic(
                observation_size=observation_size,
                num_actions=action_num,
                config=config,
            )
        else:
            critic = Critic(
                observation_size=observation_size,
                config=config,
            )

        learning_units[learning_unit_id] = Algorithm(
            actor_network=actor,
            critic_network=critic,
            config=config,
            device=device,
        )

    return learning_units


def create_MADDPG(observation_size, action_num, config: acf.MADDPGConfig):
    from cares_reinforcement_learning.algorithm.marl import MADDPG
    from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
    from cares_reinforcement_learning.networks.MADDPG import Actor, Critic

    all_agent_ids = list(observation_size["obs"].keys())
    env_teams = observation_size["teams"]
    device = hlp.get_device()

    (
        agent_id_to_actor_id,
        actor_id_to_agent_ids,
        agent_id_to_critic_id,
        critic_id_to_agent_ids,
    ) = _build_actor_critic_mappings(
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        parameter_sharing_scope=config.parameter_sharing_scope,
        algo_name="MADDPG",
    )

    learning_units = _build_learning_units(
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        observation_size=observation_size,
        action_num=action_num,
        config=config,
        device=device,
        Algorithm=DDPG,
        Actor=Actor,
        Critic=Critic,
    )

    return MADDPG(
        learning_units=learning_units,
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        agent_id_to_actor_id=agent_id_to_actor_id,
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        agent_id_to_critic_id=agent_id_to_critic_id,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        config=config,
        device=device,
    )


def create_M3DDPG(observation_size, action_num, config: acf.M3DDPGConfig):
    from cares_reinforcement_learning.algorithm.marl import M3DDPG
    from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
    from cares_reinforcement_learning.networks.M3DDPG import Actor, Critic

    all_agent_ids = list(observation_size["obs"].keys())
    env_teams = observation_size["teams"]
    device = hlp.get_device()

    (
        agent_id_to_actor_id,
        actor_id_to_agent_ids,
        agent_id_to_critic_id,
        critic_id_to_agent_ids,
    ) = _build_actor_critic_mappings(
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        parameter_sharing_scope=config.parameter_sharing_scope,
        algo_name="M3DDPG",
    )

    learning_units = _build_learning_units(
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        observation_size=observation_size,
        action_num=action_num,
        config=config,
        device=device,
        Algorithm=DDPG,
        Actor=Actor,
        Critic=Critic,
    )

    return M3DDPG(
        learning_units=learning_units,
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        agent_id_to_actor_id=agent_id_to_actor_id,
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        agent_id_to_critic_id=agent_id_to_critic_id,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        config=config,
        device=device,
    )


def create_MATD3(observation_size, action_num, config: acf.MATD3Config):
    from cares_reinforcement_learning.algorithm.marl import MATD3
    from cares_reinforcement_learning.algorithm.policy.TD3 import TD3
    from cares_reinforcement_learning.networks.MATD3 import Actor, Critic

    device = hlp.get_device()
    all_agent_ids = list(observation_size["obs"].keys())
    env_teams = observation_size["teams"]

    (
        agent_id_to_actor_id,
        actor_id_to_agent_ids,
        agent_id_to_critic_id,
        critic_id_to_agent_ids,
    ) = _build_actor_critic_mappings(
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        parameter_sharing_scope=config.parameter_sharing_scope,
        algo_name="MATD3",
    )

    learning_units = _build_learning_units(
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        observation_size=observation_size,
        action_num=action_num,
        config=config,
        device=device,
        Algorithm=TD3,
        Actor=Actor,
        Critic=Critic,
    )

    return MATD3(
        learning_units=learning_units,
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        agent_id_to_actor_id=agent_id_to_actor_id,
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        agent_id_to_critic_id=agent_id_to_critic_id,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        config=config,
        device=device,
    )


def create_MASAC(observation_size, action_num, config: acf.MASACConfig):
    from cares_reinforcement_learning.algorithm.marl.MASAC import MASAC
    from cares_reinforcement_learning.algorithm.policy.SAC import SAC
    from cares_reinforcement_learning.networks.MASAC import Actor, Critic

    device = hlp.get_device()
    all_agent_ids = list(observation_size["obs"].keys())
    env_teams = observation_size["teams"]

    (
        agent_id_to_actor_id,
        actor_id_to_agent_ids,
        agent_id_to_critic_id,
        critic_id_to_agent_ids,
    ) = _build_actor_critic_mappings(
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        parameter_sharing_scope=config.parameter_sharing_scope,
        algo_name="MATD3",
    )

    learning_units = _build_learning_units(
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        observation_size=observation_size,
        action_num=action_num,
        config=config,
        device=device,
        Algorithm=SAC,
        Actor=Actor,
        Critic=Critic,
    )

    return MASAC(
        learning_units=learning_units,
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        agent_id_to_actor_id=agent_id_to_actor_id,
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        agent_id_to_critic_id=agent_id_to_critic_id,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        config=config,
        device=device,
    )


def create_MAPPO(observation_size, action_num, config: acf.MAPPOConfig):
    from cares_reinforcement_learning.algorithm.marl import MAPPO
    from cares_reinforcement_learning.algorithm.policy.PPO import PPO
    from cares_reinforcement_learning.networks.MAPPO import Actor, Critic

    device = hlp.get_device()
    all_agent_ids = list(observation_size["obs"].keys())
    env_teams = observation_size["teams"]

    (
        agent_id_to_actor_id,
        actor_id_to_agent_ids,
        agent_id_to_critic_id,
        critic_id_to_agent_ids,
    ) = _build_actor_critic_mappings(
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        parameter_sharing_scope=config.parameter_sharing_scope,
        algo_name="MAPPO",
    )

    learning_units = _build_learning_units(
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        observation_size=observation_size,
        action_num=action_num,
        config=config,
        device=device,
        Algorithm=PPO,
        Actor=Actor,
        Critic=Critic,
        critic_uses_action_num=False,
    )

    return MAPPO(
        learning_units=learning_units,
        all_agent_ids=all_agent_ids,
        env_teams=env_teams,
        agent_id_to_actor_id=agent_id_to_actor_id,
        actor_id_to_agent_ids=actor_id_to_agent_ids,
        agent_id_to_critic_id=agent_id_to_critic_id,
        critic_id_to_agent_ids=critic_id_to_agent_ids,
        config=config,
        device=device,
    )


def create_QMIX(observation_size, action_num, config: acf.QMIXConfig):
    from cares_reinforcement_learning.algorithm.marl import QMIX
    from cares_reinforcement_learning.networks.QMIX import (
        SharedMultiAgentNetwork,
        QMixer,
    )

    network = SharedMultiAgentNetwork(
        observation_size=observation_size,
        num_actions=action_num,
        config=config,
    )

    mixer = QMixer(observation_size=observation_size, config=config)

    device = hlp.get_device()
    agent = QMIX(network=network, mixer=mixer, config=config, device=device)
    return agent


def _create_independant_agents(
    observation_size, action_num, create_network, config: acf.AlgorithmConfig
):
    obs_shapes = observation_size["obs"]  # dict[str → obs_dim]

    agents = {}
    for agent_name in obs_shapes.keys():
        agent_obs = {}
        agent_obs["vector"] = obs_shapes[agent_name]
        network = create_network(
            observation_size=agent_obs,
            action_num=action_num,
            config=config,
        )
        agents[agent_name] = network

    return agents


def _create_imarl_learning_units(
    observation_size,
    action_num,
    create_network,
    config,
) -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, list[str]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, str],
]:
    obs_shapes = observation_size["obs"]
    agent_ids = sorted(obs_shapes.keys())
    env_teams = observation_size["teams"]

    team_ids = sorted(env_teams.keys())
    agent_id_to_team_id = {
        agent_id: team_id
        for team_id in team_ids
        for agent_id in sorted(env_teams[team_id])
    }

    agent_identity_vectors = {
        agent_id: np.eye(len(agent_ids), dtype=np.float32)[index]
        for index, agent_id in enumerate(agent_ids)
    }
    team_identity_vectors = {
        team_id: np.eye(len(team_ids), dtype=np.float32)[index]
        for index, team_id in enumerate(team_ids)
    }

    parameter_sharing_scope = config.parameter_sharing_scope
    if parameter_sharing_scope == "individual":
        learning_units = _create_independant_agents(
            observation_size,
            action_num,
            create_network,
            config,
        )
        agent_id_to_learning_unit_id = {
            agent_id: agent_id for agent_id in learning_units.keys()
        }
        learning_unit_id_to_agent_ids = {agent_id: [agent_id] for agent_id in agent_ids}
        return (
            learning_units,
            agent_id_to_learning_unit_id,
            learning_unit_id_to_agent_ids,
            agent_identity_vectors,
            team_identity_vectors,
            agent_id_to_team_id,
        )

    if parameter_sharing_scope != "shared":
        raise ValueError(
            f"Unknown IMARL parameter_sharing_scope={parameter_sharing_scope}"
        )

    if not agent_ids:
        raise ValueError("IMARL requires at least one agent observation shape.")

    shared_obs_size = obs_shapes[agent_ids[0]]
    for agent_id in agent_ids[1:]:
        if obs_shapes[agent_id] != shared_obs_size:
            raise ValueError(
                "Shared IMARL networks require identical per-agent observation sizes. "
                f"Got {agent_ids[0]}={shared_obs_size} and {agent_id}={obs_shapes[agent_id]}."
            )

    learning_unit_id_to_agent_ids = {"shared": agent_ids}

    extra_obs_dim = 0
    if config.use_team_id and len(agent_ids) > 1:
        extra_obs_dim += len(team_ids)
    if config.use_agent_id and len(agent_ids) > 1:
        extra_obs_dim += len(agent_ids)

    shared_observation_size = {"vector": shared_obs_size + extra_obs_dim}
    shared_learning_unit = create_network(
        observation_size=shared_observation_size,
        action_num=action_num,
        config=config,
    )

    learning_units = {"shared": shared_learning_unit}
    agent_id_to_learning_unit_id = {agent_id: "shared" for agent_id in agent_ids}
    return (
        learning_units,
        agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids,
        agent_identity_vectors,
        team_identity_vectors,
        agent_id_to_team_id,
    )


def create_IDDPG(observation_size, action_num, config: acf.IDDPGConfig):
    from cares_reinforcement_learning.algorithm.marl import IDDPG

    device = hlp.get_device()
    (
        learning_units,
        agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids,
        agent_identity_vectors,
        team_identity_vectors,
        agent_id_to_team_id,
    ) = _create_imarl_learning_units(observation_size, action_num, create_DDPG, config)

    iddpg_agent = IDDPG(
        learning_units=learning_units,
        agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
        agent_identity_vectors=agent_identity_vectors,
        team_identity_vectors=team_identity_vectors,
        agent_id_to_team_id=agent_id_to_team_id,
        config=config,
        device=device,
    )
    return iddpg_agent


def create_ITD3(observation_size, action_num, config: acf.ITD3Config):
    from cares_reinforcement_learning.algorithm.marl import ITD3

    device = hlp.get_device()
    (
        learning_units,
        agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids,
        agent_identity_vectors,
        team_identity_vectors,
        agent_id_to_team_id,
    ) = _create_imarl_learning_units(observation_size, action_num, create_TD3, config)

    itd3_agent = ITD3(
        learning_units=learning_units,
        agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
        agent_identity_vectors=agent_identity_vectors,
        team_identity_vectors=team_identity_vectors,
        agent_id_to_team_id=agent_id_to_team_id,
        config=config,
        device=device,
    )
    return itd3_agent


def create_ISAC(observation_size, action_num, config: acf.ISACConfig):
    from cares_reinforcement_learning.algorithm.marl import ISAC

    device = hlp.get_device()
    (
        learning_units,
        agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids,
        agent_identity_vectors,
        team_identity_vectors,
        agent_id_to_team_id,
    ) = _create_imarl_learning_units(observation_size, action_num, create_SAC, config)

    isac_agent = ISAC(
        learning_units=learning_units,
        agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
        agent_identity_vectors=agent_identity_vectors,
        team_identity_vectors=team_identity_vectors,
        agent_id_to_team_id=agent_id_to_team_id,
        config=config,
        device=device,
    )
    return isac_agent


def create_IPPO(observation_size, action_num, config: acf.IPPOConfig):
    from cares_reinforcement_learning.algorithm.marl import IPPO

    device = hlp.get_device()
    (
        learning_units,
        agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids,
        agent_identity_vectors,
        team_identity_vectors,
        agent_id_to_team_id,
    ) = _create_imarl_learning_units(observation_size, action_num, create_PPO, config)

    ippo_agent = IPPO(
        learning_units=learning_units,
        agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
        learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
        agent_identity_vectors=agent_identity_vectors,
        team_identity_vectors=team_identity_vectors,
        agent_id_to_team_id=agent_id_to_team_id,
        config=config,
        device=device,
    )
    return ippo_agent


def create_CrossMARL(observation_size, action_num, config: acf.CrossMARLConfig):
    from cares_reinforcement_learning.algorithm.marl import CrossMARL

    device = hlp.get_device()

    env_teams = observation_size["teams"]  # dict[str → list[str]]

    learning_team_name = config.learning_team_name
    learning_team = (
        env_teams[learning_team_name] if learning_team_name is not None else {}
    )

    learning_obs_shapes = {
        "obs": {
            agent_name: observation_size["obs"][agent_name]
            for agent_name in learning_team
        },
        "state": observation_size["state"],
        "num_agents": len(learning_team),
        "teams": env_teams,
    }

    algorithm_factory = AlgorithmFactory()

    agents = {}

    for team_name in env_teams.keys():
        agent_obs = observation_size
        if team_name == learning_team_name:
            agent_obs = learning_obs_shapes

        agent = algorithm_factory.create_network(
            observation_size=agent_obs,
            action_num=action_num,
            config=config.agents_config[team_name],
        )
        agents[team_name] = agent

    multimarl_agent = CrossMARL(
        agent_networks=agents, env_teams=env_teams, config=config, device=device
    )
    return multimarl_agent


####################################
#      Algorithm Factory Utils     #
####################################


def _compare_mlp_parts(obj1: acf.AlgorithmConfig, obj2: acf.AlgorithmConfig) -> bool:
    # Extract fields where the value is of type mlp_type
    def get_mlp_fields(obj):
        return {
            name: value.model_dump()
            for name, value in obj.__dict__.items()
            if isinstance(value, acf.MLPConfig)
        }

    mlp_fields1 = get_mlp_fields(obj1)
    mlp_fields2 = get_mlp_fields(obj2)

    return mlp_fields1 == mlp_fields2


class AlgorithmFactory:
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
            logging.warning(f"Unknown {algorithm} algorithm.")
        else:
            if config.model_path is not None:
                logging.info(f"Loading model weights from {config.model_path}")
                agent.load_models(filepath=config.model_path, filename=config.algorithm)

            if not _compare_mlp_parts(
                type(config)(algorithm=config.algorithm, gamma=config.gamma), config
            ):
                logging.warning(
                    "The network architecture has changed from the default configuration."
                )

        return agent
