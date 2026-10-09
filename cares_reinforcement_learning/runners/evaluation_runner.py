"""
EvaluationRunner class for sequential evaluation of multiple model checkpoints.

This module provides a clean interface for loading different model checkpoints
from a training run and evaluating each one, allowing for performance tracking
across training steps.
"""

import time
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from natsort import natsorted

from cares_reinforcement_learning.runners.base_runner import BaseRunner


class EvaluationRunner(BaseRunner):
    """
    Handles sequential evaluation of multiple model checkpoints.

    This class loads different model checkpoints from a training run and evaluates
    each one, providing performance tracking across different training steps.
    """

    def __init__(
        self,
        train_seed: int,
        eval_seed: int,
        configurations: dict[str, Any],
        base_log_dir: str,
        former_base_path: str,
        checkpoint: str | None = None,
        num_eval_episodes: int | None = None,
        save_configurations: bool = False,
        progress_queue: Queue | None = None,
    ):
        """
        Initialize EvaluationRunner for sequential checkpoint evaluation.

        Args:
            train_seed: Random seed for this training run
            eval_seed: Random seed for this evaluation run
            configurations: Dictionary containing all parsed configurations
            base_log_dir: Base directory for logging evaluation results
            former_base_path: Base path to the trained model directory (contains seed subdirs)
            checkpoint: Name of the specific checkpoint to evaluate (if None, uses the default evaluation strategy)
            num_eval_episodes: Number of episodes to run per checkpoint (if None, uses config default)
            save_configurations: Whether to save configurations to disk
            progress_queue: Queue for progress updates (if any)
        """
        # Initialize the base runner with evaluation-specific settings
        super().__init__(
            train_seed=train_seed,
            eval_seed=eval_seed,
            configurations=configurations,
            base_log_dir=base_log_dir,
            save_configurations=save_configurations,
            num_eval_episodes=num_eval_episodes,
        )

        # EvaluationRunner-specific attributes
        self.former_model_base_path = Path(former_base_path)
        self.former_model_seed_path = self.former_model_base_path / str(self.train_seed)
        self.progress_queue = progress_queue
        self.checkpoint = checkpoint

        # Update logging for evaluation context
        self.logger.info(
            f"[SEED {self.train_seed}] will run {self.number_eval_episodes} episodes per checkpoint on [SEED {self.eval_seed}]"
        )

    def _report_progress(
        self, checkpoint: int, total_checkpoints: int, status: str
    ) -> None:
        """Report progress to the main thread if a queue is provided."""
        if self.progress_queue is not None:
            self.progress_queue.put(
                {
                    "seed": self.train_seed,
                    "step": checkpoint,
                    "total": total_checkpoints,
                    "status": status,
                }
            )

    def _discover_checkpoints(self) -> list[dict[str, Any]]:
        """
        Discover all available model checkpoints for this seed.

        Returns:
            List of checkpoint info dictionaries with 'path' and 'step' keys
        """
        if not self.former_model_seed_path.exists():
            raise FileNotFoundError(
                f"No model directory found for seed {self.eval_seed} at {self.former_model_seed_path}"
            )

        models_path = self.former_model_seed_path / "models"
        if not models_path.exists():
            raise FileNotFoundError(f"No models directory found at {models_path}")

        checkpoints: list[dict[str, Any]] = []

        folders = list(models_path.glob("*"))

        # # Sort folders and remove the final and best model folders
        folders = natsorted(folders)[:-2]

        for folder in folders:
            step = int(folder.name)
            if step is not None:
                checkpoints.append(
                    {
                        "path": folder,
                        "step": step,
                    }
                )

        return checkpoints

    def _discover_test_checkpoint(self) -> dict[str, Any]:
        """
        Discover the model checkpoint to test.

        If a model is specified, select that directory within models/.
        Otherwise, prefer final, then best, then the latest numbered checkpoint.
        """
        models_path = self.former_model_seed_path / "models"

        if not models_path.is_dir():
            raise FileNotFoundError(f"No models directory found at {models_path}")

        # Explicit model selection
        if self.checkpoint is not None:
            # Only allow a directory name, not an arbitrary path
            if (
                not self.checkpoint
                or self.checkpoint in (".", "..")
                or Path(self.checkpoint).name != self.checkpoint
                or "\\" in self.checkpoint
            ):
                raise ValueError(f"Invalid model directory name: {self.checkpoint}")

            model_path = models_path / self.checkpoint

            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"Model '{self.checkpoint}' not found at {model_path}"
                )

            return {
                "path": model_path,
                "step": self.checkpoint,
            }

        # Default behaviour: final -> best -> latest numbered checkpoint
        for name in ("final", "best"):
            model_path = models_path / name

            if model_path.is_dir():
                return {
                    "path": model_path,
                    "step": name,
                }

        numbered_checkpoints = [
            (int(folder.name), folder)
            for folder in models_path.iterdir()
            if folder.is_dir() and folder.name.isdigit()
        ]

        if numbered_checkpoints:
            step, folder = max(numbered_checkpoints, key=lambda item: item[0])

            return {
                "path": folder,
                "step": step,
            }

        raise FileNotFoundError(f"No model checkpoints found at {models_path}")

    def _load_checkpoint(self, checkpoint_info: dict[str, Any]) -> bool:
        """
        Load a specific model checkpoint into the agent.

        Args:
            checkpoint_info: Checkpoint information dictionary

        Returns:
            True if loading succeeded, False otherwise
        """
        checkpoint_path = checkpoint_info["path"]
        step = checkpoint_info["step"]

        self.logger.info(f"[SEED {self.eval_seed}] (step {step})")

        try:
            self.agent.load_models(
                checkpoint_path,
                self.alg_config.algorithm,
                load_mode="resume",
            )
            self.logger.info(
                f"[SEED {self.eval_seed}] Successfully loaded checkpoint: {step}"
            )
            return True
        except (FileNotFoundError, OSError, RuntimeError) as e:
            self.logger.warning(
                f"[SEED {self.eval_seed}] Failed to load checkpoint {step}: {e}"
            )
            return False

    def _evaluate_checkpoint(self, checkpoint_info: dict[str, Any]) -> bool:
        """
        Evaluate a single checkpoint.

        Args:
            checkpoint_info: Checkpoint information dictionary

        Returns:
            True if evaluation succeeded, False otherwise
        """
        # Load the checkpoint
        if not self._load_checkpoint(checkpoint_info):
            self.logger.error(
                f"[SEED {self.eval_seed}] Failed to load checkpoint {checkpoint_info['step']}, skipping"
            )
            return False

        step = checkpoint_info["step"]

        self.logger.info(
            f"[SEED {self.eval_seed}] Evaluating checkpoint: (step {step})"
        )

        start_time = time.time()

        if self.agent.policy_type == "usd":
            results = self._evaluate_usd_skills(checkpoint_info["step"], f"{step}")
        else:
            results = self._evaluate_agent_episodes(checkpoint_info["step"], f"{step}")

        end_time = time.time()
        evaluation_time = end_time - start_time

        self.logger.info(
            f"[SEED {self.eval_seed}] Completed evaluation of {step}: "
            f"Avg reward: {results.get('avg_reward', 'N/A'):.2f}, "
            f"Time: {evaluation_time:.1f}s"
        )

        return True

    def run_evaluation(self) -> None:
        """
        Execute evaluation of all discovered checkpoints.
        """
        self.logger.info(f"[SEED {self.eval_seed}] Starting checkpoint evaluation")

        # Discover all checkpoints
        checkpoints = self._discover_checkpoints()

        if not checkpoints:
            self.logger.warning(
                f"[SEED {self.eval_seed}] No checkpoints found to evaluate"
            )
            self._report_progress(0, 0, "done")
            return

        self._report_progress(0, len(checkpoints), "evaluating")

        for i, checkpoint_info in enumerate(checkpoints):
            self.logger.info(
                f"[SEED {self.eval_seed}] Processing checkpoint {i + 1}/{len(checkpoints)}"
            )
            self._evaluate_checkpoint(checkpoint_info)
            # Report progress after completing each checkpoint
            self._report_progress(i + 1, len(checkpoints), "evaluating")

        # Save all results
        self.record.save()

        self._report_progress(len(checkpoints), len(checkpoints), "done")
        self.logger.info(f"[SEED {self.eval_seed}] evaluation completed.")

    def run_test(self) -> None:
        """
        Execute testing on a selected model checkpoint.

        Unlike evaluation which tests all checkpoints, testing evaluates
        only the selected model for the specified number of episodes.
        """
        # Discover the final checkpoint
        checkpoint = self._discover_test_checkpoint()

        self.logger.info(
            f"[SEED {self.train_seed}] Starting testing with "
            f"{self.number_eval_episodes} episodes on "
            f"model '{checkpoint['step']}' [SEED {self.eval_seed}]"
        )

        self._report_progress(0, 1, "starting")

        self.logger.info(
            f"[SEED {self.eval_seed}] Testing checkpoint: {checkpoint['step']}"
        )

        if not self._evaluate_checkpoint(checkpoint):
            raise RuntimeError(
                f"[SEED {self.eval_seed}] Testing failed for checkpoint {checkpoint['step']}"
            )

        # Save results
        self.record.save()

        self._report_progress(1, 1, "done")
        self.logger.info(f"[SEED {self.eval_seed}] testing completed.")
