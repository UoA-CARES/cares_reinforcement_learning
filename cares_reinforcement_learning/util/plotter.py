import argparse
import ast
import json
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)


def plot_data(
    plot_frame: pd.DataFrame,
    title: str,
    label: str,
    x_label: str,
    y_label: str,
    directory: str,
    filename: str,
    label_fontsize: int = 15,
    title_fontsize: int = 20,
    ticks_fontsize: int = 10,
    close_figure: bool = True,
) -> None:
    matplotlib.use("agg")

    # Plot Styles
    plt.style.use("seaborn-v0_8")

    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel(y_label, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    sns.lineplot(
        data=plot_frame,
        x="x_data",
        y="y_data",
        label=label,
    )

    plt.fill_between(
        plot_frame["x_data"],
        plot_frame["y_data"] - plot_frame["std_dev"],
        plot_frame["y_data"] + plot_frame["std_dev"],
        alpha=0.3,
    )

    plt.legend(loc="best").set_draggable(True)

    plt.tight_layout(pad=0.5)

    if not os.path.exists(f"{directory}/figures"):
        os.makedirs(f"{directory}/figures")

    plt.savefig(f"{directory}/figures/{filename}.png")

    if close_figure:
        plt.close()


def plot_comparisons(
    plot_frames: list[pd.DataFrame],
    title: str,
    labels: list[str],
    x_label: str,
    y_label: str,
    directory: str,
    filename: str,
    label_fontsize: int = 15,
    title_fontsize: int = 20,
    ticks_fontsize: int = 10,
) -> None:
    for plot_frame, label in zip(plot_frames, labels):
        plot_data(
            plot_frame,
            title,
            label,
            x_label,
            y_label,
            directory,
            filename,
            label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
            ticks_fontsize=ticks_fontsize,
            close_figure=False,
        )

    plt.close()


def perpare_average_plot_frame(
    data_frame: pd.DataFrame,
    x_data: str = "total_steps",
    y_data: str = "episode_reward",
) -> pd.DataFrame:

    average_data = data_frame.groupby(x_data).agg(["mean", "std"]).reset_index()

    plot_frame: pd.DataFrame = pd.DataFrame()
    plot_frame["x_data"] = average_data[x_data]
    plot_frame["y_data"] = average_data[y_data]["mean"]
    plot_frame["std_dev"] = average_data[y_data]["std"]

    return plot_frame


def prepare_plot_frame(
    data_frame: pd.DataFrame,
    window_size: int = 1,
    x_data: str = "total_steps",
    y_data: str = "episode_reward",
) -> pd.DataFrame:

    plot_frame: pd.DataFrame = pd.DataFrame()
    plot_frame["x_data"] = data_frame[x_data]
    plot_frame["y_data"] = (
        data_frame[y_data]
        .rolling(window_size, step=1, min_periods=1, center=True)
        .mean()
    )
    plot_frame["std_dev"] = (
        data_frame[y_data]
        .rolling(window_size, step=1, min_periods=1, center=True)
        .std()
    )

    return plot_frame


def plot_eval(
    eval_data: pd.DataFrame,
    title: str,
    label: str,
    directory: str,
    filename: str,
) -> None:
    eval_plot_frame = prepare_plot_frame(eval_data, window_size=1)

    x_label: str = "Steps"
    y_label: str = "Average Reward"

    plot_data(
        eval_plot_frame,
        title,
        label,
        x_label,
        y_label,
        directory,
        filename,
    )


def plot_train(
    train_data: pd.DataFrame,
    title: str,
    label: str,
    directory: str,
    filename: str,
    window_size: int,
    y_data: str = "episode_reward",
) -> None:
    x_label: str = "Steps"
    y_label: str = "Reward"

    train_plot_frame = prepare_plot_frame(train_data, window_size, y_data)
    plot_data(
        train_plot_frame,
        title,
        label,
        x_label,
        y_label,
        directory,
        filename,
    )


def read_environmnet_config(result_directory: str) -> dict:
    with open(f"{result_directory}/env_config.json", "r", encoding="utf-8") as file:
        env_config = json.load(file)
    return env_config


def read_algorithm_config(result_directory: str) -> dict:
    with open(f"{result_directory}/alg_config.json", "r", encoding="utf-8") as file:
        alg_config = json.load(file)
    return alg_config


def read_train_config(result_directory: str) -> dict:
    with open(f"{result_directory}/train_config.json", "r", encoding="utf-8") as file:
        train_config = json.load(file)
    return train_config


def get_param_value(param_tag: str, config: dict) -> str | None:
    if param_tag in config:
        return config[param_tag]
    return None


def get_param_tag(param_tags: dict, alg_config: dict, train_config: dict) -> str:
    if len(param_tags) == 0:
        return ""

    param_tag = ""
    for key, tags in param_tags.items():

        value = get_param_value(key, alg_config)
        if value is None:
            value = get_param_value(key, train_config)

        if isinstance(value, dict):
            tags = tags.split(",")

            for tag in tags:
                tag = tag.strip()
                secondary_value = get_param_value(tag, value)
                if secondary_value is not None:
                    param_tag += f"_{tag}_{secondary_value}"

        elif value is not None:
            param_tag += f"_{key}_{value}"

    return param_tag


def generate_labels(
    args, title: str, result_directory: str
) -> tuple[str, str, str, str]:
    env_config = read_environmnet_config(result_directory)
    train_config = read_train_config(result_directory)
    alg_config = read_algorithm_config(result_directory)

    algorithm = alg_config["algorithm"]
    domain = env_config["domain"]
    task = env_config["task"]
    task = task if domain == "" else f"{domain}-{task}"

    param_tag = get_param_tag(args["param_tag"], alg_config, train_config)
    label = algorithm + param_tag

    title = task if title == "" else title
    return title, algorithm, task, label


def parse_args() -> dict:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-d",
        "--data_directories",
        type=str,
        nargs="+",
        help="List of Specific Directories with data you want to compare",
    )

    # group.add_argument(
    #     "-a",
    #     "--algorithm_directories",
    #     type=str,
    #     nargs="+",
    #     help="List of Algorithm Directories with data you want to compare",
    # )

    # group.add_argument(
    #     "-t",
    #     "--task_directories",
    #     type=str,
    #     nargs="+",
    #     help="List of Task Directories with data you want to compare",
    # )

    parser.add_argument(
        "-s",
        "--save_directory",
        type=str,
        required=True,
        help="Save directory for the plots",
    )

    parser.add_argument(
        "--x_data",
        type=str,
        default="total_steps",
        help="Data you want to plot in x_axis - default is steps",
    )

    parser.add_argument(
        "--y_data",
        type=str,
        default="episode_reward",
        help="Data you want to plot in y_axis - default is episode_reward",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Title for the plot - default is task name",
    )

    parser.add_argument(
        "--x_label",
        type=str,
        default=None,
        help="X Axis Label for the plot - default is x_data",
    )

    parser.add_argument(
        "--y_label",
        type=str,
        default=None,
        help="Y Axis Label for the plot - default is y_data",
    )

    parser.add_argument(
        "--label_fontsize", type=int, default=15, help="Fontsize for the x-axis label"
    )

    parser.add_argument(
        "--title_fontsize", type=int, default=20, help="Fontsize for the title"
    )

    parser.add_argument(
        "--ticks_fontsize", type=int, default=10, help="Fontsize for the ticks"
    )

    parser.add_argument(
        "--window_size", type=int, default=20, help="Window Size for the training plot"
    )

    parser.add_argument(
        "--param_tag",
        type=ast.literal_eval,
        default="{}",
        help="Tag to add to labels based on algorithm or training parameter",
    )

    parser.add_argument(
        "--plot_seeds",
        action=argparse.BooleanOptionalAction,
        help="Plot Individual Seeds for each algorithm in addition to the average of all seeds",
    )

    # converts into a dictionary
    args = vars(parser.parse_args())
    return args


def plot_evaluations():
    args = parse_args()

    x_data = args["x_data"]
    y_data = args["y_data"]

    title = args["title"]
    x_label = x_data if args["x_label"] is None else args["x_label"]
    y_label = y_data if args["y_label"] is None else args["y_label"]

    window_size = args["window_size"]

    save_directory = args["save_directory"]

    eval_plot_frames = []
    train_plot_frames = []
    labels = []

    for index, data_directory in enumerate(args["data_directories"]):
        logging.info(
            f"Processing {index+1}/{len(args['data_directories'])} Data for {data_directory}"
        )
        model_path = Path(f"{data_directory}")
        directory = model_path.glob("*")

        result_directories = [x for x in directory if x.is_dir()]

        title, algorithm, task, label = generate_labels(args, title, model_path)
        labels.append(label)

        average_train_data = pd.DataFrame()
        average_eval_data = pd.DataFrame()

        seed_train_plot_frames = []
        seed_eval_plot_frames = []
        seed_label = []

        for i, result_directory in enumerate(result_directories):
            logging.info(
                f"Processing Data for {algorithm}: {i+1}/{len(result_directories)} on task {task}"
            )

            if "eval.csv" not in os.listdir(
                f"{result_directory}/data"
            ) or "train.csv" not in os.listdir(f"{result_directory}/data"):
                logging.warning(
                    f"Skipping {result_directory} as it does not have train.csv or eval.csv"
                )
                continue

            eval_data = pd.read_csv(f"{result_directory}/data/eval.csv")
            train_data = pd.read_csv(f"{result_directory}/data/train.csv")

            # Concat the eval seed data into a single data frame
            average_eval_data = pd.concat(
                [average_eval_data, eval_data], ignore_index=True
            )

            if args["plot_seeds"]:
                seed_label.append(f"{label}_{i}")
                seed_plot_frame = prepare_plot_frame(
                    eval_data, window_size=1, x_data=x_data, y_data=y_data
                )
                seed_eval_plot_frames.append(seed_plot_frame)

            train_data = prepare_plot_frame(
                train_data,
                window_size=window_size,
                x_data=x_data,
                y_data=y_data,
            )

            average_train_data = pd.concat(
                [average_train_data, train_data], ignore_index=True
            )

            if args["plot_seeds"]:
                seed_train_plot_frames.append(train_data)

        if args["plot_seeds"]:
            plot_comparisons(
                seed_eval_plot_frames,
                f"{title}",
                seed_label,
                x_label,
                y_label,
                save_directory,
                f"{title}-{algorithm}-eval",
                label_fontsize=args["label_fontsize"],
                title_fontsize=args["title_fontsize"],
                ticks_fontsize=args["ticks_fontsize"],
            )

            plot_comparisons(
                seed_train_plot_frames,
                f"{title}",
                seed_label,
                x_label,
                y_label,
                save_directory,
                f"{title}-{algorithm}-train",
                label_fontsize=args["label_fontsize"],
                title_fontsize=args["title_fontsize"],
                ticks_fontsize=args["ticks_fontsize"],
            )

        eval_plot_frame = perpare_average_plot_frame(average_eval_data, x_data, y_data)
        eval_plot_frames.append(eval_plot_frame)

        train_plot_frame = perpare_average_plot_frame(
            average_train_data, "x_data", "y_data"
        )
        train_plot_frames.append(train_plot_frame)

    plot_comparisons(
        train_plot_frames,
        f"{title}",
        labels,
        x_label,
        y_label,
        save_directory,
        f"{title}-compare-train",
        label_fontsize=args["label_fontsize"],
        title_fontsize=args["title_fontsize"],
        ticks_fontsize=args["ticks_fontsize"],
    )

    plot_comparisons(
        eval_plot_frames,
        f"{title}",
        labels,
        x_label,
        y_label,
        save_directory,
        f"{title}-compare-eval",
        label_fontsize=args["label_fontsize"],
        title_fontsize=args["title_fontsize"],
        ticks_fontsize=args["ticks_fontsize"],
    )


def main():
    plot_evaluations()


if __name__ == "__main__":
    main()
