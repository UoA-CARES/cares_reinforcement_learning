import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import cares_reinforcement_learning.plotter as plt
import cares_reinforcement_learning.runners.execution_logger as logs
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.configurations import SubscriptableClass
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer


class Record:
    """
    Class for recording training and evaluation data, managing directories, saving configurations,
    handling video recording, and saving/loading agent models and memory buffers.
    """

    def __init__(
        self,
        base_directory: str,
        algorithm: str,
        task: str,
        agent: Algorithm | None = None,
        record_video: bool = True,
        record_plot: bool = True,
        plot_interval: int = 500,
        record_checkpoints: bool = False,
        checkpoint_interval: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initializes the Record instance with directories, logging settings, and recording options.
        Args:
            base_directory (str): The base directory for saving logs and models.
            algorithm (str): The name of the algorithm being used.
            task (str): The name of the task being performed.
            agent (Algorithm, optional): The reinforcement learning agent. Defaults to None.
            record_video (bool, optional): Whether to record videos during training/evaluation. Defaults to True.
            record_plot (bool, optional): Whether to enable plotting. Defaults to True.
            plot_interval (int, optional): Interval (in episodes) at which to update plots. Defaults to 1.
            record_checkpoints (bool, optional): Whether to save checkpoints of the agent and memory buffer. Defaults to False.
            checkpoint_interval (int, optional): Interval (in episodes) at which to save checkpoints. Defaults to 1.
        """

        self.best_reward = float("-inf")

        self.base_directory = f"{base_directory}"
        self.sub_directory = ""

        self.current_sub_directory = self.base_directory

        self.algorithm = algorithm
        self.task = task

        self.agent = agent

        self.train_rows: list[dict] = []
        self.train_columns: list[str] = []

        self.eval_rows: list[dict] = []
        self.eval_columns: list[str] = []

        self.record_video = record_video
        self.video: cv2.VideoWriter | None = None

        self.log_count = 0

        self.record_checkpoints = record_checkpoints
        self.checkpoint_interval = max(1, checkpoint_interval)
        self.memory_buffer: MemoryBuffer | None = None

        self.record_plot = record_plot
        self.train_plot_interval = max(1, plot_interval)

        self.logger = logger or logs.get_record_logger()

        self.__initialise_base_directory()

    def set_sub_directory(self, sub_directory: str) -> None:
        self.sub_directory = sub_directory
        self.current_sub_directory = f"{self.base_directory}/{sub_directory}"

        self.log_count = 0
        self.best_reward = float("-inf")

        self.train_rows = []
        self.train_columns = []

        self.eval_rows = []
        self.eval_columns = []

        self.__initialise_sub_directory()

    def set_agent(self, agent: Algorithm) -> None:
        self.agent = agent

    def set_memory_buffer(self, memory_buffer: MemoryBuffer) -> None:
        self.memory_buffer = memory_buffer

    def save_config(self, configuration: SubscriptableClass, file_name: str) -> None:
        # Ensure directory exists with race condition protection
        os.makedirs(self.base_directory, exist_ok=True)

        with open(
            f"{self.base_directory}/{file_name}.json", "w", encoding="utf-8"
        ) as outfile:
            json.dump(configuration.model_dump(exclude_none=False), outfile)

    def save_configurations(self, configurations: dict) -> None:
        for config_name, config in configurations.items():
            if config_name == "run_config":
                continue
            self.logger.info(f"Saving {config_name} configuration")
            self.save_config(config, config_name)

    def enable_record_video(self) -> None:
        self.record_video = True

    def disable_record_video(self) -> None:
        self.record_video = False

    def enable_record_memory(self) -> None:
        self.record_checkpoints = True

    def disable_record_memory(self) -> None:
        self.record_checkpoints = False

    def start_video(self, file_name: str, frame: np.ndarray, fps: int = 30) -> None:
        if not self.record_video:
            self.logger.warning("Video recording is disabled")
            return

        video_name = f"{self.current_sub_directory}/videos/{file_name}.mp4"
        height, width, _ = frame.shape
        self.video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        self.log_video(frame)

    def log_video(self, frame: np.ndarray) -> None:
        if not self.record_video:
            return
        if self.video is None:
            self.logger.warning(
                "Video recording is not started - use start_video method first - no video is being recorded"
            )
            return
        self.video.write(frame)

    def stop_video(self) -> None:
        if self.video is None:
            return
        self.video.release()

    def _save_checkpoint(self, total_steps: int) -> None:
        if not self.record_checkpoints:
            self.logger.warning("Checkpoint saving is disabled")
            return

        if self.memory_buffer is None:
            self.logger.warning(
                "Memory Buffer is not set - use set_memory_buffer method first"
            )
            return

        if self.agent is None:
            self.logger.warning("Agent is not set - use set_agent method first")
            return

        self.memory_buffer.save(
            filepath=f"{self.current_sub_directory}/memory", file_name="memory"
        )

        self.save_agent(f"{self.algorithm}", "checkpoint")

    def save_agent(self, file_name: str, folder_name: str) -> None:
        if self.agent is not None:
            self.agent.save_models(
                f"{self.current_sub_directory}/models/{folder_name}", f"{file_name}"
            )

    def _plot_train(self) -> None:
        if not self.train_rows:
            return

        train_dataframe = pd.DataFrame(
            self.train_rows,
            columns=self.train_columns,
        )

        plt.plot_train(
            train_dataframe,
            f"Training-{self.algorithm}-{self.task}",
            self.algorithm,
            self.current_sub_directory,
            "train",
            20,
        )

    def _append_or_rewrite_data(
        self,
        data: list[dict],
        logs: dict,
        filename: str,
        known_columns: list[str],
    ) -> None:
        path = Path(self.current_sub_directory) / "data" / filename

        new_columns = [column for column in logs if column not in known_columns]

        data.append(dict(logs))

        if new_columns:
            known_columns.extend(new_columns)

            # Rewrite only when the CSV schema changes.
            pd.DataFrame(
                data,
                columns=known_columns,
            ).to_csv(
                path,
                index=False,
            )
            return

        # Normal path: append one row.
        pd.DataFrame(
            [logs],
            columns=known_columns,
        ).to_csv(
            path,
            mode="a",
            header=not path.exists() or path.stat().st_size == 0,
            index=False,
        )

    def _print_log(self, **logs) -> None:
        string_values = []
        for key, val in logs.items():
            if isinstance(val, list):
                formatted_list = [f"{str(i)[0:10]:6s}" for i in val]
                string_values.append(f"{key}: {formatted_list}")
            else:
                string_values.append(f"{key}: {str(val)[0:10]:6s}")

        string_out = " | ".join(string_values)
        string_out = "| " + string_out + " |"

        self.logger.info(string_out)

    def log_train(
        self,
        display: bool = False,
        **logs,
    ) -> None:
        self.log_count += 1

        if display:
            self._print_log(**logs)

        self._append_or_rewrite_data(
            data=self.train_rows,
            logs=logs,
            filename="train.csv",
            known_columns=self.train_columns,
        )

        if self.record_checkpoints and self.log_count % self.checkpoint_interval == 0:
            self._save_checkpoint(logs["total_steps"])

        if self.record_plot and self.log_count % self.train_plot_interval == 0:
            self._plot_train()

        reward = logs["episode_reward"]

        if reward > self.best_reward:
            self.logger.info(
                f"New highest reward of {reward} during training! " "Saving model..."
            )
            self.best_reward = reward
            self.save_agent(
                self.algorithm,
                "highest_reward",
            )

    def get_last_logged_step(self) -> int:
        if not self.train_rows:
            return 0

        return int(self.train_rows[-1].get("total_steps", 0))

    def _plot_eval(self) -> None:
        if not self.eval_rows:
            return

        eval_dataframe = pd.DataFrame(
            self.eval_rows,
            columns=self.eval_columns,
        )

        plt.plot_eval(
            eval_dataframe,
            f"Evaluation-{self.algorithm}-{self.task}",
            self.algorithm,
            self.current_sub_directory,
            "eval",
        )

    def log_eval(
        self,
        display: bool = False,
        **logs,
    ) -> None:
        if display:
            self._print_log(**logs)

        self._append_or_rewrite_data(
            data=self.eval_rows,
            logs=logs,
            filename="eval.csv",
            known_columns=self.eval_columns,
        )

        if self.record_plot:
            self._plot_eval()

        self.save_agent(
            self.algorithm,
            f"{logs['total_steps']}",
        )

    def save(self) -> None:
        self.logger.info("Saving final outputs")

        if self.train_columns:
            pd.DataFrame(
                self.train_rows,
                columns=self.train_columns,
            ).to_csv(
                Path(self.current_sub_directory) / "data" / "train.csv",
                index=False,
            )

        if self.eval_columns:
            pd.DataFrame(
                self.eval_rows,
                columns=self.eval_columns,
            ).to_csv(
                Path(self.current_sub_directory) / "data" / "eval.csv",
                index=False,
            )

        self._plot_eval()
        self._plot_train()

        self.save_agent(
            self.algorithm,
            "final",
        )

        self.stop_video()

    def load(self, base_directory: Path) -> None:
        train_path = Path(base_directory) / "data" / "train.csv"
        eval_path = Path(base_directory) / "data" / "eval.csv"

        self.train_rows = []
        self.train_columns = []

        self.eval_rows = []
        self.eval_columns = []

        if train_path.exists() and train_path.stat().st_size > 0:
            train_dataframe = pd.read_csv(train_path)

            self.train_columns = list(train_dataframe.columns)
            self.train_rows = train_dataframe.to_dict(orient="records")
            self.log_count = len(self.train_rows)

            if "episode_reward" in train_dataframe.columns:
                reward_values = pd.to_numeric(
                    train_dataframe["episode_reward"],
                    errors="coerce",
                )

                if reward_values.notna().any():
                    self.best_reward = float(reward_values.max())

        if eval_path.exists() and eval_path.stat().st_size > 0:
            eval_dataframe = pd.read_csv(eval_path)

            self.eval_columns = list(eval_dataframe.columns)
            self.eval_rows = eval_dataframe.to_dict(orient="records")

    def __initialise_base_directory(self) -> None:
        # Use exist_ok=True to handle race conditions in parallel execution
        os.makedirs(self.base_directory, exist_ok=True)

    def __initialise_sub_directory(self) -> None:
        # Create subdirectories with race condition protection using exist_ok=True
        subdirs = ["data", "models", "figures", "videos", "memory"]

        for subdir in subdirs:
            dir_path = f"{self.current_sub_directory}/{subdir}"
            os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def create_base_directory(
        gym: str,
        domain: str,
        task: str,
        algorithm: str,
        run_name: str = "",
        base_dir: str | None = None,
        format_str: str | None = None,
        **names: dict[str, str],
    ) -> str:
        """
        Creates a base directory path for logging based on the provided parameters and environment variables.
        Args:
            gym (str): The name of the gym environment.
            domain (str): The domain of the task.
            task (str): The specific task within the domain.
            algorithm (str): The name of the algorithm being used.
            run_name (str): The name of the run.
            base_dir (str, optional): The base directory for logs overrides CARES_LOG_BASE_DIR variable.
            format_str (str, optional): Template for the log path overrides CARES_LOG_PATH_TEMPLATE variable.
            names (dict): Additional names to be included in the log path.
        Returns:
            str: The constructed base directory path for logging.
        Environment Variables:
            CARES_LOG_PATH_TEMPLATE (str): Template for the log path. Defaults to "{algorithm}/{algorithm}-{gym}-{domain_task}-{run_name}{date}".
            CARES_LOG_BASE_DIR (str): Base directory for logs. Defaults to "{Path.home()}/cares_rl_logs".
        """

        default_log_path = os.environ.get(
            "CARES_LOG_PATH_TEMPLATE",
            "{algorithm}/{algorithm}-{domain_task}-{run_name}{date}",
        )

        format_str = default_log_path if format_str is None else format_str

        default_base_dir = os.environ.get(
            "CARES_LOG_BASE_DIR", f"{Path.home()}/cares_rl_logs"
        )

        base_dir = default_base_dir if base_dir is None else base_dir

        date = datetime.now().strftime("%y_%m_%d_%H-%M-%S")

        domain_with_hyphen_or_empty = f"{domain}-" if domain != "" else ""
        domain_task = domain_with_hyphen_or_empty + task

        run_name_with_hyphen_or_empty = f"{run_name}-" if run_name != "" else ""

        log_dir = format_str.format(
            algorithm=algorithm,
            domain=domain,
            task=task,
            gym=gym,
            run_name=run_name_with_hyphen_or_empty,
            domain_task=domain_task,
            date=date,
            **names,
        )
        return f"{base_dir}/{log_dir}"
