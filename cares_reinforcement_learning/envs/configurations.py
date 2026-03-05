"""
Configuration class for Gym Environments.
"""

from pathlib import Path
from typing import ClassVar

from cares_reinforcement_learning.algorithm.configurations import SubscriptableClass

file_path = Path(__file__).parent.resolve()


class GymEnvironmentConfig(SubscriptableClass):
    """
    Configuration class for Gym Environment.

    Attributes:
        frames_to_stack (int): Number of frames to stack for image observation (default: 3)
        frame_width (int): Width of the image frames (default: 84)
        frame_height (int): Height of the image frames (default: 84)
        grey_scale (bool): Whether to convert frames to grayscale (default: False)
        display (int): Display mode for the environment (default: 0)
        save_train_checkpoints (int): Whether to save training checkpoints (default: 0)
    """

    gym: ClassVar[str]
    domain: str = ""
    task: str

    display: int = 0
    save_train_checkpoints: int = 0

    # stochastic noise configuration
    state_std: float = 0.0
    action_std: float = 0.0

    # image observation configurations
    frames_to_stack: int = 3
    frame_width: int = 84
    frame_height: int = 84
    grey_scale: int = 0

    record_video_fps: int = 30

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["gym"] = self.__class__.gym
        return data


class OpenAIConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "openai"


class DMCSConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "dmcs"


class PyBoyConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "pyboy"

    rom_path: str = f"{Path.home()}/cares_rl_configs"
    act_freq: int = 24
    emulation_speed: int = 0
    headless: int = 1


class ShowdownConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "showdown"


class DroneConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "drone"
    task: str = "move_3d"
    use_simulator: int = 1  # 1 for simulator (DroneSim), 0 for real drone (Drone)


class GripperConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "gripper"

    gripper_id: int


class SMACConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "smac"

    task: str = "3m"

    record_video_fps: int = 5


class SMAC2Config(GymEnvironmentConfig):
    gym: ClassVar[str] = "smac2"

    task: str = "10gen_terran"

    n_units: int = 3
    n_enemies: int = 3

    record_video_fps: int = 5


class MPEConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "mpe"

    continuous_actions: int = 0

    record_video_fps: int = 5
