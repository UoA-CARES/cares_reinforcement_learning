"""
ExecutionCoordinator class for orchestrating reinforcement learning execution across multiple seeds.
This class handles the parallel execution, configuration management, and coordination
of training, evaluation, and testing runs for statistical validation.
"""

import concurrent.futures
import logging
import multiprocessing
import time
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any

import yaml
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

import cares_reinforcement_learning.runners.execution_logger as logs
from cares_reinforcement_learning.envs.configurations import GymEnvironmentConfig
from cares_reinforcement_learning.runners.evaluation_runner import EvaluationRunner
from cares_reinforcement_learning.runners.training_runner import TrainingRunner
from cares_reinforcement_learning.algorithm.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.util.record import Record
from cares_reinforcement_learning.util.rl_parser import RunConfig

# Module-level loggers - created once when module loads
logger = logs.get_main_logger()
parallel_logger = logs.get_parallel_logger()


class ExecutionCoordinator:
    """
    Orchestrates reinforcement learning execution across multiple seeds with support
    for parallel execution, configuration management, and different run modes.

    Supports:
    - Training: Full training runs with progress tracking
    - Evaluation: Testing all checkpoints from training runs
    - Testing: Testing final models only with specified episodes
    """

    def __init__(self, configurations: dict[str, Any]):
        """
        Initialize the ExecutionCoordinator with parsed configurations.

        Args:
            configurations: Dictionary containing all parsed configurations
        """
        self.configurations = configurations
        self.run_config: RunConfig = configurations["run_config"]
        self.env_config: GymEnvironmentConfig = configurations["env_config"]
        self.training_config: TrainingConfig = configurations["train_config"]
        self.alg_config: AlgorithmConfig = configurations["alg_config"]

        self.train_seeds: list[int] = self.training_config.seeds

        self.max_workers: int = self.training_config.max_workers
        self.max_workers = min(len(self.train_seeds), self.max_workers)
        self.max_workers = max(1, self.max_workers)

        self.base_log_dir: str | None = None

        # Log all configurations for debugging
        self._print_configurations()

    def _print_configurations(self) -> None:
        """Log all configurations for debugging and reproducibility."""
        logger.info(
            "\n---------------------------------------------------\n"
            "RUN CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(f"\n{yaml.dump(self.run_config.dict(), default_flow_style=False)}")

        logger.info(
            "\n---------------------------------------------------\n"
            "ENVIRONMENT CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(f"\n{yaml.dump(self.env_config.dict(), default_flow_style=False)}")

        logger.info(
            "\n---------------------------------------------------\n"
            "ALGORITHM CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(f"\n{yaml.dump(self.alg_config.dict(), default_flow_style=False)}")

        logger.info(
            "\n---------------------------------------------------\n"
            "TRAINING CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(
            f"\n{yaml.dump(self.training_config.dict(), default_flow_style=False)}"
        )

    def setup_logging_and_directories(self, run_name: str = "") -> None:
        """
        Set up logging directories and validate configurations.

        Args:
            run_name: Optional name for the training run

        Returns:
            Base log directory path
        """
        self.base_log_dir = Record.create_base_directory(
            domain=self.env_config.domain,
            task=self.env_config.task,
            gym=self.env_config.gym,
            algorithm=self.alg_config.algorithm,
            run_name=run_name,
        )

        logger.info(f"Base Log Directory: {self.base_log_dir}")

    def _listen_for_progress(self, queue: Queue, futures: list[Any]) -> None:
        progress = Progress(
            TextColumn("[bold blue]{task.fields[seed]}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[status]}"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )

        tasks: dict[int, int] = {}
        done_seeds: set[int] = set()

        with progress:
            while True:
                try:
                    msg = queue.get_nowait()
                except Empty:
                    msg = None

                if msg:
                    seed = msg["seed"]

                    if seed not in tasks:
                        total = msg.get("total", 1)
                        tasks[seed] = progress.add_task(
                            f"Seed {seed}",
                            total=total,
                            seed=f"Seed {seed}",
                            status=msg.get("status", ""),
                        )

                    progress.update(
                        tasks[seed],  # type: ignore[arg-type]
                        completed=msg.get("step", 0),
                        status=msg.get("status", ""),
                    )

                    if msg.get("status") == "done":
                        done_seeds.add(seed)
                        progress.console.log(f"[green]Seed {seed} completed!")

                if len(done_seeds) == len(futures):
                    break

    def _test_single_seed(
        self,
        seed: int,
        progress_queue: Queue | None = None,
        save_configurations: bool = False,
    ) -> None:
        """
        Execute testing for a single seed.
        This delegates all setup to EvaluationRunner but calls run_test().

        Args:
            seed: Random seed for this run
            progress_queue: Queue for progress updates (if any)
            save_configurations: Whether to save configurations to disk
        """
        if self.base_log_dir is None:
            raise ValueError("Base log directory must be set before running seeds")

        if not self.run_config.data_path:
            raise ValueError("Data path is required for test command")

        if not self.run_config.eval_seed:
            raise ValueError("Evaluation seed is required for test command")

        eval_seed = self.run_config.eval_seed
        eval_runner = EvaluationRunner(
            train_seed=seed,  # This finds the trained model based on original seed
            eval_seed=eval_seed,  # This sets the random seed for testing against
            configurations=self.configurations,
            base_log_dir=self.base_log_dir,
            former_base_path=self.run_config.data_path,
            num_eval_episodes=self.run_config.episodes,  # Use episodes from run_config
            save_configurations=save_configurations,
            progress_queue=progress_queue,
        )
        eval_runner.run_test()  # Call run_test instead of run_evaluation

    def _test_parallel_seeds(self) -> None:
        """
        Execute testing across multiple seeds in parallel.
        """
        logger.info(f"Running testing with {self.max_workers} parallel workers")

        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()

            # Use ProcessPoolExecutor with limited workers
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [
                    # Save configs only for first seed
                    executor.submit(
                        self._test_single_seed,
                        seed=seed,
                        progress_queue=progress_queue,  # type: ignore[arg-type]
                        save_configurations=(i == 0),
                    )
                    for i, seed in enumerate(self.train_seeds)
                ]

                self._listen_for_progress(progress_queue, futures)  # type: ignore[arg-type]

        logger.info(f"Completed testing for all {len(self.train_seeds)} seeds")

    def _test_sequential_seeds(self) -> None:
        """
        Execute testing across multiple seeds sequentially.
        Useful for debugging or when parallel execution is not desired.
        """
        logs.set_logger_level("parallel", logging.INFO)
        logs.set_logger_level("seed", logging.INFO)

        for iteration, seed in enumerate(self.train_seeds):
            logger.info(
                f"Running testing seed {iteration+1}/{len(self.train_seeds)} with Seed: {seed}"
            )
            # Save configs only for first seed
            self._test_single_seed(seed=seed, save_configurations=(iteration == 0))

        logger.info(
            f"Completed testing for all {len(self.train_seeds)} seeds sequentially"
        )

    def _test(self) -> None:
        """
        Execute testing across multiple seeds.
        Chooses between parallel and sequential execution based on configuration.
        """
        if not self.run_config.data_path:
            raise ValueError("Data path is required for test command")
        if not self.run_config.episodes:
            raise ValueError("Episodes count is required for test command")

        if self.base_log_dir is None:
            raise ValueError("Base log directory must be set before running seeds")

        start_time = time.time()
        if self.max_workers > 1 and len(self.train_seeds) > 1:
            self._test_parallel_seeds()
        else:
            self._test_sequential_seeds()

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Testing completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )

    def _evaluate_single_seed(
        self,
        seed: int,
        progress_queue: Queue | None = None,
        save_configurations: bool = False,
    ) -> None:
        """
        Execute evaluation for a single seed.

        Args:
            seed: Random seed for this run
            progress_queue: Queue for progress updates (if any)
            save_configurations: Whether to save configurations to disk
        """
        if self.base_log_dir is None:
            raise ValueError("Base log directory must be set before running seeds")

        if not self.run_config.data_path:
            raise ValueError("Data path is required for evaluate command")

        eval_runner = EvaluationRunner(
            train_seed=seed,
            eval_seed=seed,
            configurations=self.configurations,
            base_log_dir=self.base_log_dir,
            former_base_path=self.run_config.data_path,
            save_configurations=save_configurations,
            progress_queue=progress_queue,
        )
        eval_runner.run_evaluation()

    def _evaluate_parallel_seeds(self) -> None:
        """
        Execute evaluation across multiple seeds in parallel.
        """
        logger.info(f"Running evaluation with {self.max_workers} parallel workers")

        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()

            # Use ProcessPoolExecutor with limited workers
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [
                    # Save configs only for first seed
                    executor.submit(
                        self._evaluate_single_seed,
                        seed=seed,
                        progress_queue=progress_queue,  # type: ignore[arg-type]
                        save_configurations=(i == 0),
                    )
                    for i, seed in enumerate(self.train_seeds)
                ]

                self._listen_for_progress(progress_queue, futures)  # type: ignore[arg-type]

        logger.info(f"Completed evaluation for all {len(self.train_seeds)} seeds")

    def _evaluate_sequential_seeds(self) -> None:
        """
        Execute evaluation across multiple seeds sequentially.
        Useful for debugging or when parallel execution is not desired.
        """
        logs.set_logger_level("parallel", logging.INFO)
        logs.set_logger_level("seed", logging.INFO)

        for iteration, seed in enumerate(self.train_seeds):
            logger.info(
                f"Running evaluation seed {iteration+1}/{len(self.train_seeds)} with Seed: {seed}"
            )
            self._evaluate_single_seed(
                seed=seed,
                save_configurations=(
                    iteration == 0
                ),  # Save configs only for first seed
            )

        logger.info(
            f"Completed evaluation for all {len(self.train_seeds)} seeds sequentially"
        )

    def _evaluate(self) -> None:
        """
        Execute evaluation across multiple seeds.
        Chooses between parallel and sequential execution based on configuration.
        """
        if not self.run_config.data_path:
            raise ValueError("Data path is required for evaluate command")

        if self.base_log_dir is None:
            raise ValueError("Base log directory must be set before running seeds")

        start_time = time.time()
        if self.max_workers > 1 and len(self.train_seeds) > 1:
            self._evaluate_parallel_seeds()
        else:
            self._evaluate_sequential_seeds()

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Evaluation completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )

    def _train_single_seed(
        self,
        seed: int,
        progress_queue: Queue | None = None,
        save_configurations: bool = False,
    ) -> None:
        """
        Execute training for a single seed.

        Args:
            seed: Random seed for this run
            progress_queue: Queue for progress updates (if any)
            save_configurations: Whether to save configurations to disk
        """
        if self.base_log_dir is None:
            raise ValueError("Base log directory must be set before running seeds")

        resume_path = None
        if self.run_config.command == "resume":
            resume_path = self.run_config.data_path

        runner = TrainingRunner(
            train_seed=seed,
            configurations=self.configurations,
            base_log_dir=self.base_log_dir,
            progress_queue=progress_queue,
            resume_path=resume_path,
            save_configurations=save_configurations,
        )

        runner.run_training()

    def _train_parallel_seeds(self) -> None:
        """
        Execute training across multiple seeds in parallel.
        """
        logger.info(f"Running with {self.max_workers} parallel workers")

        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()

            # Use ProcessPoolExecutor with limited workers
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [
                    # Save configs only for first seed
                    executor.submit(
                        self._train_single_seed,
                        seed=seed,
                        progress_queue=progress_queue,  # type: ignore[arg-type]
                        save_configurations=(i == 0),
                    )
                    for i, seed in enumerate(self.train_seeds)
                ]

                self._listen_for_progress(progress_queue, futures)  # type: ignore[arg-type]

        logger.info(f"Completed all {len(self.train_seeds)} seeds")

    def _train_sequential_seeds(self) -> None:
        """
        Execute training across multiple seeds sequentially.
        Useful for debugging or when parallel execution is not desired.
        """
        logs.set_logger_level("parallel", logging.INFO)
        logs.set_logger_level("seed", logging.INFO)

        for iteration, seed in enumerate(self.train_seeds):
            logger.info(
                f"Running seed {iteration+1}/{len(self.train_seeds)} with Seed: {seed}"
            )
            self._train_single_seed(
                seed=seed,
                save_configurations=(
                    iteration == 0
                ),  # Save configs only for first seed
            )

        logger.info(f"Completed all {len(self.train_seeds)} seeds sequentially")

    def _train(self) -> None:
        """
        Execute training across multiple seeds.
        Chooses between parallel and sequential execution based on configuration.
        """
        start_time = time.time()
        if self.max_workers > 1 and len(self.train_seeds) > 1:
            self._train_parallel_seeds()
        else:
            self._train_sequential_seeds()

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Training completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )

    def run(self) -> None:
        """
        Main entry point to run the execution process.
        """
        if self.run_config.command in ["train", "resume"]:
            self._train()
        elif self.run_config.command == "evaluate":
            self._evaluate()
        elif self.run_config.command == "test":
            self._test()
        else:
            raise ValueError(f"Unknown command: {self.run_config.command}")
