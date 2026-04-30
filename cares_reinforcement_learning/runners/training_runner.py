import time
from dataclasses import replace
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from cares_reinforcement_learning.memory.memory_buffer import (
    MARLMemoryBuffer,
    SARLMemoryBuffer,
)
from cares_reinforcement_learning.runners.base_runner import BaseRunner, EpisodeStats
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.util.repetition_manager import RepetitionManager


class TrainingRunner(BaseRunner):
    """
    Handles the training loop for a single seed with integrated logging.

    This class encapsulates all training logic for a single seed, providing
    clean separation between orchestration (TrainingCoordinator) and execution (TrainingRunner).
    """

    def __init__(
        self,
        train_seed: int,
        configurations: dict[str, Any],
        base_log_dir: str,
        progress_queue: Queue | None = None,
        resume_path: str | None = None,
        save_configurations: bool = False,
        eval_seed: int | None = None,
    ):
        """
        Initialize TrainingRunner with all component creation and setup.

        Args:
            train_seed: Random seed for this training run
            configurations: Dictionary containing all parsed configurations
            base_log_dir: Base directory for logging
            progress_queue: Queue for progress updates (if any)
            resume_path: Path to resume from (if None, start fresh training)
            save_configurations: Whether to save configurations to disk
            eval_seed: Separate evaluation seed (if None, uses train_seed)
        """
        # Resolve eval_seed before calling super()
        eval_seed = eval_seed if eval_seed is not None else train_seed

        # Initialize the base runner
        super().__init__(
            train_seed=train_seed,
            eval_seed=eval_seed,
            configurations=configurations,
            base_log_dir=base_log_dir,
            save_configurations=save_configurations,
        )

        # TrainingRunner-specific setup
        self.progress_queue = progress_queue
        self.display = bool(self.training_config.display)

        # Create memory (needed for training)
        self.memory_buffer = self.memory_factory.create_memory(self.alg_config)

        # Handle resume logic - this must modify our local variables
        self.start_training_step = 0
        if resume_path is not None:
            self.start_training_step, self.memory_buffer = self._handle_resume(
                resume_path,
                self.alg_config.algorithm,
            )

        # Set up memory in record
        self.record.set_memory_buffer(self.memory_buffer)

        # Algorithm Training parameters
        self.max_steps_training = self.alg_config.max_steps_training
        self.max_steps_exploration = self.alg_config.max_steps_exploration
        self.number_steps_per_train_policy = (
            self.alg_config.number_steps_per_train_policy
        )
        self.batch_size = self.alg_config.batch_size
        self.G = self.alg_config.G  # pylint: disable=invalid-name

        # Evaluation parameters (some inherited from BaseRunner)
        self.number_steps_per_evaluation = (
            self.training_config.number_steps_per_evaluation
        )

        # Episode repetition setup
        self.repetition_manager = RepetitionManager(
            max_repetitions=self.alg_config.repetition_num_episodes
        )

        self.logger.info(f"[SEED {self.train_seed}] training instance setup complete")

    def _handle_resume(
        self,
        data_path: str,
        algorithm: str,
    ) -> tuple[int, SARLMemoryBuffer | MARLMemoryBuffer]:
        """
        Handle all resume logic and return starting step and loaded memory.

        Args:
            data_path: Path to the checkpoint data
            algorithm: Algorithm name for loading models

        Returns:
            Tuple of (starting_training_step, loaded_memory)
        """
        restart_path = Path(data_path) / str(self.train_seed)

        # Check if seed directory exists
        if not restart_path.exists():
            self.logger.warning(
                f"[SEED {self.train_seed}] No checkpoint found at {restart_path}, starting fresh training"
            )
            return 0, self.memory_buffer

        self.logger.info(
            f"[SEED {self.train_seed}] Restarting from path: {restart_path}"
        )

        self.logger.info(
            f"[SEED {self.train_seed}] Loading training and evaluation data"
        )
        self.record.load(restart_path)

        self.logger.info(f"[SEED {self.train_seed}] Loading memory buffer")
        try:
            loaded_memory = self.memory_buffer.load(restart_path / "memory", "memory")  # type: ignore
        except FileNotFoundError:
            self.logger.warning(
                f"[SEED {self.train_seed}] No memory buffer found at {restart_path / 'memory'}, starting with empty memory"
            )
            loaded_memory = self.memory_buffer

        self.logger.info(f"[SEED {self.train_seed}] Loading agent models")
        try:
            self.agent.load_models(restart_path / "models" / "checkpoint", algorithm)  # type: ignore
        except FileNotFoundError:
            self.logger.warning(
                f"[SEED {self.train_seed}] No agent models found at {restart_path / 'models' / 'checkpoint'}, starting with fresh models"
            )

        start_training_step = self.record.get_last_logged_step()
        self.logger.info(
            f"[SEED {self.train_seed}] Resuming from step: {start_training_step}"
        )

        return start_training_step, loaded_memory

    def _report_progress(self, episode: int, step: int, status: str) -> None:
        """Report progress to the main thread if a queue is provided."""
        if self.progress_queue is not None:
            self.progress_queue.put(
                {
                    "seed": self.train_seed,
                    "episode": episode,
                    "step": step,
                    "total": self.max_steps_training,
                    "status": status,
                }
            )

    def _select_exploration_action(self, train_step_counter: int) -> ActionSample:
        """Handle exploration phase action selection."""
        self.logger.info(
            f"Running Exploration Steps {train_step_counter + 1}/{self.max_steps_exploration}"
        )

        return ActionSample(self.env.sample_action(), source="exploration")

    def _select_repetition_action(self, episode_timesteps: int) -> ActionSample:
        """Handle episode repetition action selection."""
        action = self.repetition_manager.get_repetition_action(episode_timesteps)

        return action

    def _select_policy_action(self, state) -> ActionSample:
        """Handle policy-based action selection."""
        action = self.agent.act(state, evaluation=False)

        return action

    def _select_action(
        self, train_step_counter: int, episode_step: int, state
    ) -> ActionSample:
        if train_step_counter < self.max_steps_exploration:
            action = self._select_exploration_action(train_step_counter)
        elif self.repetition_manager.should_repeat(episode_step):
            action = self._select_repetition_action(episode_step)
        else:
            action = self._select_policy_action(state)

        return action

    def _update_policy(
        self,
        train_step_counter: int,
        episode_num: int,
        episode_timesteps: int,
        episode_reward: float,
        episode_done: bool,
    ) -> dict:
        """Execute policy training step."""
        episode_context = EpisodeContext(
            training_step=train_step_counter,
            episode=episode_num + 1,
            episode_steps=episode_timesteps,
            episode_reward=episode_reward,
            episode_done=episode_done,
        )

        train_info = {}
        for _ in range(self.G):
            train_info = self.agent.train(self.memory_buffer, episode_context)

        return train_info

    def _finalise_episode(
        self,
        train_step_counter: int,
        episode_reward: float,
    ) -> None:
        """Handle episode completion and repetition logic."""
        in_training_phase = train_step_counter > self.max_steps_exploration
        self.repetition_manager.finish_episode(episode_reward, in_training_phase)

    def _run_evaluation(self, train_step_counter: int) -> None:
        """Execute evaluation phase."""
        self.logger.info("*************--Evaluation Loop--*************")

        if self.agent.policy_type == "usd":
            self._evaluate_usd_skills(
                train_step_counter + 1, f"{train_step_counter + 1}"
            )
        else:
            self._evaluate_agent_episodes(
                train_step_counter + 1, f"{train_step_counter + 1}"
            )

        self.logger.info("--------------------------------------------")

    def run_training(self) -> None:
        """
        Execute the main training loop with proper cleanup.

        This is the main entry point that orchestrates the entire training process
        for this seed, including exploration, training, and evaluation phases.
        """
        self.logger.info(
            f"Training {self.max_steps_training} Exploration {self.max_steps_exploration} "
            f"Evaluation {self.number_steps_per_evaluation}"
        )

        self._report_progress(0, 0, "starting")

        start_time = time.time()

        # Initialize training state
        episode_num = 0
        episode_stats = EpisodeStats(n_agents=self.env.num_agents)

        state = self.env.reset()
        episode_start = time.time()

        # Main training loop
        train_step_counter = self.start_training_step
        for train_step_counter in range(
            self.start_training_step, int(self.max_steps_training)
        ):
            info: dict = {}

            episode_stats.step()

            status = (
                "training"
                if train_step_counter >= self.max_steps_exploration
                else "exploration"
            )
            self._report_progress(episode_num + 1, train_step_counter + 1, status)

            # Determine action based on training phase
            action_sample = self._select_action(
                train_step_counter, episode_stats.steps, state
            )

            # Record action and execute step
            self.repetition_manager.record_action(action_sample)
            info |= self.repetition_manager.get_status_info()

            experience = self.env.step(action_sample.action)
            experience = replace(
                experience, train_data={**experience.train_data, **action_sample.extras}
            )

            state = experience.next_observation

            episode_end = experience.done_flag | experience.truncated_flag

            if self.display:
                self.env.render()

            # Calculate total reward (extrinsic + intrinsic)
            # TODO bring back for intrinsic rewards and modify for MARL
            # if train_step_counter > self.max_steps_exploration:
            #     intrinsic_reward = self.agent.get_intrinsic_reward(
            #         state, action, next_state
            #     )
            #     total_reward += intrinsic_reward
            #     info["intrinsic_reward"] = intrinsic_reward

            # Store experience in memory
            self.memory_buffer.add(experience)  # type: ignore

            episode_stats.update_reward(experience.reward)

            # Train policy if conditions are met
            if (
                train_step_counter >= self.max_steps_exploration
                and (train_step_counter + 1) % self.number_steps_per_train_policy == 0
            ):
                train_info = self._update_policy(
                    train_step_counter,
                    episode_num,
                    episode_stats.steps,
                    episode_stats.get_episode_reward(),
                    episode_end,
                )
                info |= train_info

            # Evaluate agent periodically
            if (train_step_counter + 1) % self.number_steps_per_evaluation == 0:
                self._report_progress(
                    episode_num + 1, train_step_counter + 1, "evaluation"
                )
                self._run_evaluation(train_step_counter)

            # Handle episode completion
            if episode_end:
                episode_time = time.time() - episode_start

                info.update(episode_stats.summary())

                # Log training data
                self.record.log_train(
                    total_steps=train_step_counter + 1,
                    episode=episode_num + 1,
                    episode_time=episode_time,
                    **experience.info,
                    **info,
                    display=True,
                )

                # Handle any logic at episode end
                self._finalise_episode(
                    train_step_counter, episode_stats.get_episode_reward()
                )

                # Reset for next episode
                state = self.env.reset()
                episode_stats.reset()

                episode_num += 1
                self.agent.episode_done()

                episode_start = time.time()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(
            f"Training completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )

        # Save record and report completion
        self.record.save()
        self._report_progress(episode_num + 1, train_step_counter + 1, "done")
