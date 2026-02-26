"""
Example usage of the TrainingRunner class.
This demonstrates how to replace the main() function logic with the new runner.
"""

import multiprocessing
import sys

import cares_reinforcement_learning.runners.execution_logger as logs
from cares_reinforcement_learning.runners.execution_coordinator import (
    ExecutionCoordinator,
)
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.rl_parser import RLParser


def main_with_runner():
    """
    Simplified main function using TrainingRunner.
    This replaces the complex main() function in run.py
    """
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Start method has already been set, likely by another part of the code or a library.
        pass

    # Set up logging first - choose your preferred preset
    logs.LoggingPresets.production()  # or .production(), .quiet(), .debug()

    # Get the main logger for this function
    logger = logs.get_main_logger()

    # Parse configurations (same as before)
    parser = RLParser()
    configurations = parser.parse_args()

    # Create the execution coordinator
    coordinator = ExecutionCoordinator(configurations)

    # Device validation
    device = hlp.get_device()
    logger.info(f"Device: {device}")

    # Interactive prompts
    run_name = input(
        "Double check your experiment configurations :) Press ENTER to continue. (Optional - Enter a name for this run)\n"
    )

    if device.type == "cpu":
        no_gpu_answer = input(
            "Device being set as CPU - No cuda or mps detected. Do you want to continue? Note: Training will be slower on cpu only. [y/n]"
        )
        if no_gpu_answer not in ["y", "Y"]:
            logger.info(
                "Terminating Experiment - check CUDA or mps is installed correctly."
            )
            sys.exit()

    # Checkpoint warnings
    if coordinator.env_config.save_train_checkpoints:
        logger.warning(
            "Training checkpoints will be saved - be aware this will increase disk usage (memory buffer)."
        )
        if coordinator.alg_config.image_observation:
            no_gpu_answer = input(
                "Image observations are being used with checkpoints - this will take up a lot of disk space: Do you want to disable this? [y/n]"
            )
            if no_gpu_answer in ["y", "Y"]:
                logger.info("Disabling training checkpoint saving.")
                coordinator.env_config.save_train_checkpoints = False

    # Setup directories and logging
    coordinator.setup_logging_and_directories(run_name)

    logger.info(f"Command: {coordinator.run_config.command}")
    logger.info(f"Data Path: {coordinator.run_config.data_path}")

    # Run the training process
    coordinator.run()


if __name__ == "__main__":
    main_with_runner()
