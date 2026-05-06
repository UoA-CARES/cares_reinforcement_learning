import numpy as np

from cares_reinforcement_learning.types.experience import (
    MultiAgentExperience,
    SingleAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


def get_sarl_observation(state_size: int = 4) -> SARLObservation:
    """Create a dummy SARL observation for testing."""
    return SARLObservation(vector_state=np.array([1.0] * state_size), image_state=None)


def get_indexed_sarl_observation(index: int, state_size: int = 4) -> SARLObservation:
    """Create an indexed SARL observation for testing."""
    return SARLObservation(
        vector_state=np.array([float(index)] * state_size), image_state=None
    )


def get_marl_observation(state_size: int = 4, num_agents: int = 2) -> MARLObservation:
    """Create a dummy MARL observation for testing."""
    agents = [f"agent_{i}" for i in range(num_agents)]

    agent_states = {agent_id: np.array([1.0] * state_size) for agent_id in agents}

    available_actions = {
        agent_id: np.ones((state_size,), dtype=bool) for agent_id in agents
    }

    return MARLObservation(
        global_state=np.array([1.0] * state_size),
        agent_states=agent_states,
        available_actions=available_actions,
    )


def get_indexed_marl_observation(
    index: int, state_size: int = 4, num_agents: int = 2
) -> MARLObservation:
    """Create an indexed MARL observation for testing."""
    agents = [f"agent_{i}" for i in range(num_agents)]

    agent_states = {
        agent_id: np.array([float(index)] * state_size) for agent_id in agents
    }

    available_actions = {
        agent_id: np.ones((state_size,), dtype=bool) for agent_id in agents
    }

    return MARLObservation(
        global_state=np.array([float(index)] * state_size),
        agent_states=agent_states,
        available_actions=available_actions,
    )


def create_sarl_experience(
    observation: SARLObservation | None = None,
    next_observation: SARLObservation | None = None,
) -> SingleAgentExperience:
    """Create a SingleAgentExperience for testing."""
    if observation is None:
        observation = get_sarl_observation()
    if next_observation is None:
        next_observation = get_sarl_observation()

    return SingleAgentExperience(
        observation=observation,
        next_observation=next_observation,
        action=np.array([0.5]),
        reward=1.0,
        done=False,
        truncated=False,
        info={},
    )


def create_marl_experience(
    observation: MARLObservation | None = None,
    next_observation: MARLObservation | None = None,
) -> MultiAgentExperience:
    """Create a MultiAgentExperience for testing."""
    if observation is None:
        observation = get_marl_observation()
    if next_observation is None:
        next_observation = get_marl_observation()

    actions = {}
    reward = {}
    done = {}
    truncated = {}
    for agent in observation.agent_states:
        reward[agent] = 10.0
        done[agent] = False
        truncated[agent] = False
        actions[agent] = np.array(list(range(2)), dtype=np.float32)

    return MultiAgentExperience(
        observation=observation,
        next_observation=next_observation,
        action=actions,
        reward=reward,
        done=done,
        truncated=truncated,
        info={},
    )
