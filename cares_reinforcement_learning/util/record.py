import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch import nn

import cares_reinforcement_learning.util.plotter as plt
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.configurations import SubscriptableClass


class Record:
    """
    A class that represents a record for logging and saving data during training and evaluation.

    Args:
        log_dir (str): The log directory.
        algorithm (str): The algorithm name.
        task (str): The task name.
        network (nn.Module, optional): The neural network model. Defaults to None.
    """

    def __init__(
        self,
        base_directory: str,
        algorithm: str,
        task: str,
        agent: nn.Module | None = None,
        record_video: bool = True,
        record_memory: bool = False,
    ) -> None:

        self.best_reward = float("-inf")

        self.base_directory = f"{base_directory}"
        self.sub_directory = ""

        self.current_sub_directory = self.base_directory

        self.algorithm = algorithm
        self.task = task

        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()

        self.agent = agent

        self.record_memory = record_memory
        self.memory_buffer: MemoryBuffer | None = None

        self.record_video = record_video
        self.video: cv2.VideoWriter | None = None

        self.log_count = 0

        self.__initialise_base_directory()

    def set_sub_directory(self, sub_directory: str) -> None:
        self.sub_directory = sub_directory
        self.current_sub_directory = f"{self.base_directory}/{sub_directory}"

        self.log_count = 0

        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()

        self.__initialise_sub_directory()

    def set_agent(self, agent: nn.Module) -> None:
        self.agent = agent

    def set_memory_buffer(self, memory_buffer: MemoryBuffer) -> None:
        self.memory_buffer = memory_buffer

    def save_config(self, configuration: SubscriptableClass, file_name: str) -> None:
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

        with open(
            f"{self.base_directory}/{file_name}.json", "w", encoding="utf-8"
        ) as outfile:
            json.dump(configuration.dict(exclude_none=True), outfile)

    def save_configurations(self, configurations: dict) -> None:
        for config_name, config in configurations.items():
            if config_name == "run_config":
                continue
            logging.info(f"Saving {config_name} configuration")
            self.save_config(config, config_name)

    def enable_record_video(self) -> None:
        self.record_video = True

    def disable_record_video(self) -> None:
        self.record_video = False

    def enable_record_memory(self) -> None:
        self.record_memory = True

    def disable_record_memory(self) -> None:
        self.record_memory = False

    def start_video(self, file_name: str, frame, fps=30):
        if not self.record_video:
            logging.warning("Video recording is disabled")
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
            logging.warning(
                "Video recording is not started - use start_video method first - no video is being recorded"
            )
            return
        self.video.write(frame)

    def stop_video(self) -> None:
        if self.video is None:
            return
        self.video.release()

    def save_memory(self):
        if not self.record_memory:
            return

        if self.memory_buffer is None:
            logging.warning(
                "Memory Buffer is not set - use set_memory_buffer method first"
            )
            return

        self.memory_buffer.save(
            filepath=f"{self.current_sub_directory}/memory", file_name="memory"
        )

    def save_agent(self, file_name: str, folder_name: str) -> None:
        if self.agent is not None:
            self.agent.save_models(
                f"{self.current_sub_directory}/models/{folder_name}", f"{file_name}"
            )

    def _save_data(
        self, data_frame: pd.DataFrame, filename: str, logs: dict, display: bool = True
    ) -> None:
        if data_frame.empty:
            logging.warning("Trying to save an Empty Dataframe")

        data_frame.to_csv(f"{self.current_sub_directory}/data/{filename}", index=False)

        string_values = []
        for key, val in logs.items():
            if isinstance(val, list):
                formatted_list = [f"{str(i)[0:10]:6s}" for i in val]
                string_values.append(f"{key}: {formatted_list}")
            else:
                string_values.append(f"{key}: {str(val)[0:10]:6s}")

        string_out = " | ".join(string_values)
        string_out = "| " + string_out + " |"

        if display:
            logging.info(string_out)

    def log_train(self, display: bool = False, **logs) -> None:
        self.log_count += 1

        self.train_data = pd.concat(
            [self.train_data, pd.DataFrame([logs])], ignore_index=True
        )
        self._save_data(self.train_data, "train.csv", logs, display=display)

        self.save_memory()

        plt.plot_train(
            self.train_data,
            f"Training-{self.algorithm}-{self.task}",
            f"{self.algorithm}",
            self.current_sub_directory,
            "train",
            20,
        )

        reward = logs["episode_reward"]

        if reward > self.best_reward:
            logging.info(
                f"New highest reward of {reward} during training! Saving model..."
            )
            self.best_reward = reward

            self.save_agent(f"{self.algorithm}", "highest_reward")

    def log_eval(self, display: bool = False, **logs) -> None:
        self.eval_data = pd.concat(
            [self.eval_data, pd.DataFrame([logs])], ignore_index=True
        )
        self._save_data(self.eval_data, "eval.csv", logs, display=display)

        plt.plot_eval(
            self.eval_data,
            f"Evaluation-{self.algorithm}-{self.task}",
            f"{self.algorithm}",
            self.current_sub_directory,
            "eval",
        )

        # self.save_agent(f"{self.algorithm}", f"{logs['total_steps']}")

    def save(self) -> None:
        logging.info("Saving final outputs")
        self._save_data(self.train_data, "train.csv", {}, display=False)
        self._save_data(self.eval_data, "eval.csv", {}, display=False)

        if not self.eval_data.empty:
            plt.plot_eval(
                self.eval_data,
                f"Evaluation-{self.algorithm}-{self.task}",
                f"{self.algorithm}",
                self.current_sub_directory,
                "eval",
            )
        if not self.train_data.empty:
            plt.plot_train(
                self.train_data,
                f"Training-{self.algorithm}-{self.task}",
                f"{self.algorithm}",
                self.current_sub_directory,
                "train",
                20,
            )

        self.save_agent(f"{self.algorithm}", "final")

    def __initialise_base_directory(self) -> None:
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

    def __initialise_sub_directory(self) -> None:
        if not os.path.exists(f"{self.current_sub_directory}/data"):
            os.makedirs(f"{self.current_sub_directory}/data")

        if not os.path.exists(f"{self.current_sub_directory}/models"):
            os.makedirs(f"{self.current_sub_directory}/models")

        if not os.path.exists(f"{self.current_sub_directory}/figures"):
            os.makedirs(f"{self.current_sub_directory}/figures")

        if not os.path.exists(f"{self.current_sub_directory}/videos"):
            os.makedirs(f"{self.current_sub_directory}/videos")

        if not os.path.exists(f"{self.current_sub_directory}/memory"):
            os.makedirs(f"{self.current_sub_directory}/memory")

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

        run_name_with_hypen_or_empty = f"{run_name}-" if run_name != "" else ""

        log_dir = format_str.format(
            algorithm=algorithm,
            domain=domain,
            task=task,
            gym=gym,
            run_name=run_name_with_hypen_or_empty,
            domain_task=domain_task,
            date=date,
            **names,
        )
        return f"{base_dir}/{log_dir}"
