import inspect
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cares_reinforcement_learning.algorithm import configurations
from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
from cares_reinforcement_learning.algorithm.configurations import AlgorithmConfig
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.experience import (
    MultiAgentExperience,
    SingleAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)

TEST_SEED = 1234

CAPACITY = 5
ACTION_NUM = 2

OBSERVATION_SIZE_VECTOR = 5
OBSERVATION_SIZE_IMAGE = (9, 32, 32)

OBSERVATION_SIZE_MARL = {
    "obs": {
        "agent_0": 10,
        "agent_1": 10,
        "agent_2": 10,
    },
    "state": 30,
    "num_agents": 3,
    "teams": {
        "team_0": [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
    },
}


# CrossMARL is intentionally excluded from these generic tests.
#
# CrossMARL is a harness around other MARL algorithms, requires
# configured frozen model paths, and deliberately does not support
# generic transfer loading.
GENERIC_TEST_EXCLUSIONS = {
    "CrossMARL",
}


def reset_test_seed(seed: int = TEST_SEED) -> None:
    """Reset common random sources used by algorithms/tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_algorithm_cases():
    """Discover all algorithm configurations as individual pytest cases."""
    cases = []

    for name, cls in inspect.getmembers(
        configurations,
        inspect.isclass,
    ):
        if not issubclass(cls, AlgorithmConfig):
            continue

        if cls == AlgorithmConfig:
            continue

        algorithm = name.replace("Config", "")

        if algorithm in GENERIC_TEST_EXCLUSIONS:
            continue

        cases.append(
            pytest.param(
                algorithm,
                cls,
                id=algorithm,
            )
        )

    return cases


ALGORITHM_CASES = get_algorithm_cases()


def create_sarl_observation(
    observation_size: dict,
    image_state: bool = False,
) -> SARLObservation:
    """Create a SARL observation for testing."""
    vector_state = np.arange(
        observation_size["vector"],
        dtype=np.float32,
    )

    if image_state:
        image = np.random.randint(
            255,
            size=observation_size["image"],
            dtype=np.uint8,
        )
    else:
        image = None

    return SARLObservation(
        vector_state=vector_state,
        image_state=image,
    )


def create_marl_observation(
    observation_size: dict,
    action_num: int,
) -> MARLObservation:
    """Create a MARL observation for testing."""
    global_state = np.arange(
        observation_size["state"],
        dtype=np.float32,
    )

    agent_states = {
        agent_id: np.arange(
            obs_dim,
            dtype=np.float32,
        )
        for agent_id, obs_dim in observation_size["obs"].items()
    }

    available_actions = {
        agent_id: np.ones(
            action_num,
            dtype=np.float32,
        )
        for agent_id in observation_size["obs"].keys()
    }

    return MARLObservation(
        global_state=global_state,
        agent_states=agent_states,
        available_actions=available_actions,
    )


def populate_buffer_sarl(
    memory_buffer,
    capacity: int,
    observation_size: dict,
    agent,
    image_state: bool = False,
):
    """Populate a SARL buffer using the algorithm's real action interface."""
    for _ in range(capacity):
        observation = create_sarl_observation(
            observation_size,
            image_state,
        )

        next_observation = create_sarl_observation(
            observation_size,
            image_state,
        )

        action_sample = agent.act(
            observation,
            evaluation=False,
        )

        experience = SingleAgentExperience(
            observation=observation,
            next_observation=next_observation,
            action=action_sample.action,
            reward=10.0,
            done=False,
            truncated=False,
            train_data=dict(action_sample.extras),
            info={},
        )

        memory_buffer.add(experience)

    return memory_buffer


def populate_buffer_marl(
    memory_buffer,
    capacity: int,
    observation_size: dict,
    action_num: int,
    agent,
):
    """Populate a MARL buffer using the algorithm's real action interface."""
    for _ in range(capacity):
        observation = create_marl_observation(
            observation_size,
            action_num,
        )

        next_observation = create_marl_observation(
            observation_size,
            action_num,
        )

        action_sample = agent.act(
            observation,
            evaluation=False,
        )

        reward = {agent_id: 10.0 for agent_id in observation.agent_states}

        done = {agent_id: False for agent_id in observation.agent_states}

        truncated = {agent_id: False for agent_id in observation.agent_states}

        experience = MultiAgentExperience(
            observation=observation,
            next_observation=next_observation,
            action=action_sample.action,
            reward=reward,
            done=done,
            truncated=truncated,
            train_data=dict(action_sample.extras),
            info={},
        )

        memory_buffer.add(experience)

    return memory_buffer


def get_observation_size(
    alg_config: AlgorithmConfig,
) -> dict:
    """Return the appropriate test observation shape."""
    if alg_config.marl_observation:
        return OBSERVATION_SIZE_MARL

    return {
        "image": OBSERVATION_SIZE_IMAGE,
        "vector": OBSERVATION_SIZE_VECTOR,
    }


def create_agent(
    factory: AlgorithmFactory,
    config_cls,
):
    """Create a fresh config and algorithm instance."""
    alg_config = config_cls()

    observation_size = get_observation_size(alg_config)

    agent = factory.create_network(
        observation_size=observation_size,
        action_num=ACTION_NUM,
        config=alg_config,
    )

    return (
        alg_config,
        observation_size,
        agent,
    )


def create_populated_memory(
    memory_factory: MemoryFactory,
    alg_config: AlgorithmConfig,
    agent,
    observation_size: dict,
    capacity: int = CAPACITY,
):
    memory_buffer = memory_factory.create_memory(alg_config)

    if alg_config.marl_observation:
        return populate_buffer_marl(
            memory_buffer,
            capacity,
            observation_size,
            ACTION_NUM,
            agent,
        )

    return populate_buffer_sarl(
        memory_buffer,
        capacity,
        observation_size,
        agent,
        image_state=alg_config.image_observation,
    )


def create_training_context() -> EpisodeContext:
    """Create the common training context used by tests."""
    return EpisodeContext(
        training_step=1,
        episode=1,
        episode_steps=1,
        episode_reward=10.0,
        episode_done=True,
    )


def calculate_test_value(
    agent,
    observation,
    action,
) -> float:
    reset_test_seed(TEST_SEED + 1)

    if hasattr(agent, "set_skill"):
        agent.set_skill(
            0,
            evaluation=True,
        )

    reset_noise = getattr(
        agent,
        "_reset_noise",
        None,
    )

    if callable(reset_noise):
        reset_noise()

    return agent._calculate_value(
        observation,
        action,
    )


def assert_checkpoint_value_equal(
    actual: Any,
    expected: Any,
) -> None:
    """
    Recursively compare values loaded from two checkpoints.

    This lets the persistence test verify complete saved state
    without knowing whether an algorithm contains actors, critics,
    target networks, optimisers, normalisers, encoders, counters,
    entropy state, etc.
    """
    if isinstance(actual, torch.Tensor):
        assert isinstance(
            expected,
            torch.Tensor,
        )

        torch.testing.assert_close(
            actual.detach().cpu(),
            expected.detach().cpu(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        return

    if isinstance(actual, np.ndarray):
        assert isinstance(
            expected,
            np.ndarray,
        )

        np.testing.assert_array_equal(
            actual,
            expected,
        )
        return

    if isinstance(actual, dict):
        assert isinstance(
            expected,
            dict,
        )

        assert actual.keys() == expected.keys()

        for key in actual:
            assert_checkpoint_value_equal(
                actual[key],
                expected[key],
            )

        return

    if isinstance(actual, (list, tuple)):
        assert isinstance(
            expected,
            type(actual),
        )

        assert len(actual) == len(expected)

        for actual_item, expected_item in zip(
            actual,
            expected,
        ):
            assert_checkpoint_value_equal(
                actual_item,
                expected_item,
            )

        return

    if isinstance(actual, float):
        assert actual == pytest.approx(
            expected,
            rel=0,
            abs=0,
            nan_ok=True,
        )
        return

    assert actual == expected


def assert_checkpoint_directories_equal(
    source_dir: Path,
    roundtrip_dir: Path,
) -> None:
    """
    Compare the full checkpoint directory trees.

    Resume loading followed immediately by another save should
    reproduce the same persisted state.
    """
    source_files = sorted(
        path.relative_to(source_dir) for path in source_dir.rglob("*") if path.is_file()
    )

    roundtrip_files = sorted(
        path.relative_to(roundtrip_dir)
        for path in roundtrip_dir.rglob("*")
        if path.is_file()
    )

    assert source_files == roundtrip_files

    torch_checkpoint_suffixes = {
        ".pt",
        ".pth",
        ".pht",
    }

    for relative_path in source_files:
        source_path = source_dir / relative_path

        roundtrip_path = roundtrip_dir / relative_path

        if source_path.suffix in torch_checkpoint_suffixes:
            source_checkpoint = torch.load(
                source_path,
                map_location="cpu",
            )

            roundtrip_checkpoint = torch.load(
                roundtrip_path,
                map_location="cpu",
            )

            assert_checkpoint_value_equal(
                source_checkpoint,
                roundtrip_checkpoint,
            )

        else:
            assert source_path.read_bytes() == roundtrip_path.read_bytes()


@pytest.mark.parametrize(
    "algorithm, config_cls",
    ALGORITHM_CASES,
)
def test_algorithm_smoke(
    algorithm,
    config_cls,
):
    """
    Broad smoke test for every algorithm.

    Preserves the original checks:
    - algorithm can be created,
    - value calculation returns a float,
    - a training call returns a dictionary,
    - intrinsic reward calculation runs when enabled.
    """
    reset_test_seed()

    factory = AlgorithmFactory()
    memory_factory = MemoryFactory()

    (
        alg_config,
        observation_size,
        agent,
    ) = create_agent(
        factory,
        config_cls,
    )

    assert agent is not None, f"{algorithm} was not created successfully"

    if agent.policy_type == "mbrl":
        pytest.skip("MBRL algorithms are outside " "this generic RL smoke test")

    memory_buffer = create_populated_memory(
        memory_factory,
        alg_config,
        agent,
        observation_size,
    )

    sample = memory_buffer.sample_uniform(1)
    experience = sample.experiences[0]

    value = agent._calculate_value(
        experience.observation,
        experience.action,
    )

    assert isinstance(
        value,
        float,
    ), (
        f"{algorithm} did not return a float " "value for the calculated value"
    )

    info = agent.train(
        memory_buffer,
        create_training_context(),
    )

    assert isinstance(
        info,
        dict,
    ), (
        f"{algorithm} did not return a " "dictionary of training info"
    )

<<<<<<< HEAD
        memory_buffer = memory_factory.create_memory(alg_config)

        if alg_config.marl_observation:
            observation_size = observation_size_marl
        else:
            observation_size = {
                "image": observation_size_image,
                "vector": observation_size_vector,
            }

        agent = factory.create_network(
            observation_size=observation_size,
            action_num=action_num,
            config=alg_config,
            action_sampler=None,
=======
    intrinsic_on = (
        bool(alg_config.intrinsic_on)
        if hasattr(
            alg_config,
            "intrinsic_on",
>>>>>>> main
        )
        else False
    )

    if intrinsic_on:
        sample = memory_buffer.sample_uniform(1)
        experience = sample.experiences[0]

        # Existing smoke behaviour: simply ensure the intrinsic
        # reward path executes without error.
        agent.get_intrinsic_reward(
            experience.observation,
            experience.action,
            experience.next_observation,
        )


@pytest.mark.parametrize(
    "algorithm, config_cls",
    ALGORITHM_CASES,
)
def test_algorithm_persistence(
    tmp_path,
    algorithm,
    config_cls,
):
    """
    Validate resume and transfer persistence generically.

    Resume contract:
    - exercise a source agent,
    - save it,
    - load into a completely fresh instance,
    - immediately re-save it,
    - compare the complete checkpoint trees,
    - compare observable value behaviour.

    Transfer contract:
    - load the same saved model into another fresh instance,
    - confirm its model can be used,
    - confirm fresh training can begin successfully.

    The test intentionally avoids knowing the internal architecture
    of individual algorithms.
    """
    reset_test_seed()

    factory = AlgorithmFactory()
    memory_factory = MemoryFactory()

    # ==============================================================
    # Source agent
    # ==============================================================

    (
        source_config,
        observation_size,
        source_agent,
    ) = create_agent(
        factory,
        config_cls,
    )

    assert source_agent is not None, f"{algorithm} was not created successfully"

    if source_agent.policy_type == "mbrl":
        pytest.skip("MBRL algorithms are outside " "this generic persistence test")

    training_capacity = max(
        CAPACITY,
        int(
            getattr(
                source_agent,
                "batch_size",
                CAPACITY,
            )
        ),
    )

    source_memory = create_populated_memory(
        memory_factory,
        source_config,
        source_agent,
        observation_size,
        capacity=training_capacity,
    )

    # Capture a stable observation/action pair before training.
    #
    # Some on-policy algorithms may consume or clear their
    # buffer during train().
    sample = source_memory.sample_uniform(1)
    experience = sample.experiences[0]

    test_observation = experience.observation
    test_action = experience.action

    # Exercise training before saving so persistence is not
    # tested solely against a newly constructed object.
    source_info = source_agent.train(
        source_memory,
        create_training_context(),
    )

    assert isinstance(
        source_info,
        dict,
    )

    assert source_info, (
        f"{algorithm} did not perform a training " "update before persistence testing"
    )

    source_dir = tmp_path / "source"

    source_agent.save_models(
        source_dir,
        algorithm,
    )

    # ==============================================================
    # Resume
    # ==============================================================

    (
        _,
        _,
        resume_agent,
    ) = create_agent(
        factory,
        config_cls,
    )

    assert resume_agent is not None

    resume_agent.load_models(
        source_dir,
        algorithm,
        load_mode="resume",
    )

    # The strongest generic persistence check:
    #
    #     source save
    #          ↓
    #     fresh agent
    #          ↓
    #     resume load
    #          ↓
    #     immediate save
    #
    # should reproduce exactly the same persisted state.
    roundtrip_dir = tmp_path / "roundtrip"

    resume_agent.save_models(
        roundtrip_dir,
        algorithm,
    )

    assert_checkpoint_directories_equal(
        source_dir,
        roundtrip_dir,
    )

    # Also check observable behaviour.
    #
    # This catches state that affects the algorithm but was
    # accidentally omitted entirely from persistence.
    source_value = calculate_test_value(
        source_agent,
        test_observation,
        test_action,
    )

    resume_value = calculate_test_value(
        resume_agent,
        test_observation,
        test_action,
    )

    assert resume_value == pytest.approx(
        source_value,
        rel=1e-5,
        abs=1e-6,
    ), (
        f"{algorithm} resume load changed " "observable value behaviour"
    )

    # ==============================================================
    # Transfer
    # ==============================================================

    (
        transfer_config,
        _,
        transfer_agent,
    ) = create_agent(
        factory,
        config_cls,
    )

    assert transfer_agent is not None

    transfer_agent.load_models(
        source_dir,
        algorithm,
        load_mode="transfer",
    )

    # Transfer intentionally does NOT reproduce the entire
    # checkpoint:
    #
    # - optimisers are fresh,
    # - counters are fresh,
    # - normalisers may be fresh,
    # - exploration state is fresh,
    # - target networks may be rebuilt.
    #
    # Therefore the generic transfer test only validates that
    # the transferred learned model can be used successfully.
    transfer_value = calculate_test_value(
        transfer_agent,
        test_observation,
        test_action,
    )

    assert transfer_value == pytest.approx(
        source_value,
        rel=1e-5,
        abs=1e-6,
    ), (
        f"{algorithm} transfer load did not " "reproduce source model behaviour"
    )

    # Transfer represents a new run, so use a fresh memory
    # buffer rather than the source run's potentially mutated
    # replay/rollout buffer.
    transfer_memory = create_populated_memory(
        memory_factory,
        transfer_config,
        transfer_agent,
        observation_size,
    )

    transfer_info = transfer_agent.train(
        transfer_memory,
        create_training_context(),
    )

    assert isinstance(
        transfer_info,
        dict,
    ), (
        f"{algorithm} transfer-loaded " "agent could not train"
    )
