import cares_reinforcement_learning.envs.configurations as cfg
from cares_reinforcement_learning.envs.marl.marl_environment import (
    MARLEnvironment,
)
from cares_reinforcement_learning.envs.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import GymEnvironmentConfig

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel


class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(
        self,
        config: GymEnvironmentConfig,
        train_seed: int,
        eval_seed: int,
        image_observation: bool,
    ) -> tuple[SARLEnvironment | MARLEnvironment, SARLEnvironment | MARLEnvironment]:

        env: SARLEnvironment | MARLEnvironment
        eval_env: SARLEnvironment | MARLEnvironment
        match config:
            # ---------- SARL ----------
            case cfg.DMCSConfig():
                from cares_reinforcement_learning.envs.sarl.dmcs.dmcs_environment import (
                    DMCSEnvironment,
                )

                env = DMCSEnvironment(
                    config, train_seed, image_observation=image_observation
                )
                eval_env = DMCSEnvironment(
                    config, eval_seed, image_observation=image_observation
                )

            case cfg.OpenAIConfig():
                from cares_reinforcement_learning.envs.sarl.openai.openai_environment import (
                    OpenAIEnvironment,
                )

                env = OpenAIEnvironment(
                    config, train_seed, image_observation=image_observation
                )
                eval_env = OpenAIEnvironment(
                    config, eval_seed, image_observation=image_observation
                )

            case cfg.PyBoyConfig():
                from cares_reinforcement_learning.envs.sarl.pyboy.pyboy_environment import (
                    PyboyEnvironment,
                )

                env = PyboyEnvironment(
                    config, train_seed, image_observation=image_observation
                )
                eval_env = PyboyEnvironment(
                    config, eval_seed, image_observation=image_observation
                )

            case cfg.ShowdownConfig():
                from cares_reinforcement_learning.envs.sarl.showdown.showdown_environment import (
                    ShowdownEnvironment,
                )

                env = ShowdownEnvironment(
                    config, train_seed, image_observation=image_observation
                )
                eval_env = ShowdownEnvironment(
                    config,
                    eval_seed,
                    image_observation=image_observation,
                    evaluation=True,
                )

            case cfg.DroneConfig():
                from cares_reinforcement_learning.envs.sarl.drone.drone_environment import (
                    DroneEnvironment,
                )

                env = DroneEnvironment(
                    config, train_seed, image_observation=image_observation
                )
                # intentional: shared env
                eval_env = env

            case cfg.GripperConfig():
                from cares_reinforcement_learning.envs.sarl.gripper.gripper_environment import (
                    GripperEnvironment,
                )

                env = GripperEnvironment(
                    config, train_seed, image_observation=image_observation
                )
                # intentional: shared env
                eval_env = env

            case cfg.F1TenthConfig():
                from cares_reinforcement_learning.envs.sarl.f1tenth.f1tenth_environment import (
                    F1TenthEnvironment,
                )

                env = F1TenthEnvironment(config, train_seed)
                eval_env = env

            # ---------- MARL ----------
            case cfg.MPEConfig():
                from cares_reinforcement_learning.envs.marl.mpe.mpe import (
                    MPE2Environment,
                )

                env = MPE2Environment(config, train_seed)
                eval_env = MPE2Environment(config, eval_seed)

            case cfg.SMACConfig():
                from cares_reinforcement_learning.envs.marl.smac.smac import (
                    SMACEnvironment,
                )

                env = SMACEnvironment(config, train_seed)
                eval_env = SMACEnvironment(config, eval_seed)

            case cfg.SMAC2Config():
                from cares_reinforcement_learning.envs.marl.smac2.smac2 import (
                    SMAC2Environment,
                )

                env = SMAC2Environment(config, train_seed)
                eval_env = SMAC2Environment(config, eval_seed)

            case _:
                raise ValueError(f"Unknown environment: {type(config)}")

        return env, eval_env
