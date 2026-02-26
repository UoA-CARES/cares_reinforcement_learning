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

    def _discover_final_checkpoint(self) -> dict[str, Any] | None:
        """
        Discover the final model checkpoint for this seed.

        Returns:
            Checkpoint info dictionary with 'path' and 'step' keys, or None if not found
        """
        if not self.former_model_seed_path.exists():
            raise FileNotFoundError(
                f"No model directory found for seed {self.eval_seed} at {self.former_model_seed_path}"
            )

        models_path = self.former_model_seed_path / "models"
        if not models_path.exists():
            raise FileNotFoundError(f"No models directory found at {models_path}")

        folders = list(models_path.glob("*"))
        folders = natsorted(folders)

        # Look for 'final' folder first, then 'best', then highest numbered folder
        for folder_name in ["final", "best"]:
            for folder in folders:
                if folder.name == folder_name:
                    return {
                        "path": folder,
                        "step": folder_name,
                    }

        # If no 'final' or 'best' folder, use the highest numbered checkpoint
        numbered_folders = []
        for folder in folders:
            try:
                step = int(folder.name)
                numbered_folders.append((step, folder))
            except ValueError:
                continue

        if numbered_folders:
            step, folder = max(numbered_folders)
            return {
                "path": folder,
                "step": step,
            }

        return None

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
            self.agent.load_models(checkpoint_path, self.alg_config.algorithm)
            self.logger.info(
                f"[SEED {self.eval_seed}] Successfully loaded checkpoint: {step}"
            )
            return True
        except (FileNotFoundError, OSError, RuntimeError) as e:
            self.logger.warning(
                f"[SEED {self.eval_seed}] Failed to load checkpoint {step}: {e}"
            )
            return False

    def _evaluate_checkpoint(self, checkpoint_info: dict[str, Any]) -> None:
        """
        Evaluate a single checkpoint.

        Args:
            checkpoint_info: Checkpoint information dictionary

        Returns:
            Dictionary with evaluation results
        """
        # Load the checkpoint
        if not self._load_checkpoint(checkpoint_info):
            self.logger.error(
                f"[SEED {self.eval_seed}] Failed to load checkpoint {checkpoint_info['step']}, skipping"
            )
            return

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
        Execute testing on the final model checkpoint only.

        Unlike evaluation which tests all checkpoints, testing only evaluates
        the final trained model for the specified number of episodes.
        """
        self.logger.info(
            f"[SEED {self.train_seed}] Starting testing with {self.number_eval_episodes} episodes on final model [SEED {self.eval_seed}]"
        )

        # Discover the final checkpoint
        final_checkpoint = self._discover_final_checkpoint()

        if not final_checkpoint:
            self.logger.warning(
                f"[SEED {self.eval_seed}] No final checkpoint found to test"
            )
            self._report_progress(0, 0, "done")
            return

        self._report_progress(0, 1, "starting")

        self.logger.info(
            f"[SEED {self.eval_seed}] Testing final checkpoint: {final_checkpoint['step']}"
        )

        self._evaluate_checkpoint(final_checkpoint)

        # Save results
        self.record.save()

        self._report_progress(1, 1, "done")
        self.logger.info(f"[SEED {self.eval_seed}] testing completed.")
