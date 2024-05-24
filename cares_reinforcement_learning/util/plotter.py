import argparse
import json
import logging
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)


# TODO make the plots look how people want them too. This is just a basic example
import matplotlib.pyplot as plt


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
    display: bool = True,
    close_figure: bool = True,
) -> None:

    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel(y_label, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    sns.lineplot(
        data=plot_frame,
        x=plot_frame["steps"],
        y="avg",
        label=label,
        errorbar="sd",
    )

    plt.legend(loc="best").set_draggable(True)

    plt.tight_layout(pad=0.5)

    if display:
        plt.show()

    if close_figure:
        plt.close()

    if not os.path.exists(f"{directory}/figures"):
        os.makedirs(f"{directory}/figures")

    plt.savefig(f"{directory}/figures/{filename}.png")


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
    display: bool = True,
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
            display=False,
            close_figure=False,
        )

    if display:
        plt.show()

    plt.close()


def prepare_eval_plot_frame(eval_data: pd.DataFrame) -> pd.DataFrame:
    x_data: str = "total_steps"
    y_data: str = "episode_reward"

    plot_frame: pd.DataFrame = pd.DataFrame()
    plot_frame["steps"] = eval_data[x_data]
    plot_frame["avg"] = eval_data[y_data]

    return plot_frame


def plot_eval(
    eval_data: pd.DataFrame,
    title: str,
    label: str,
    directory: str,
    filename: str,
    display: bool = False,
) -> None:
    eval_plot_frame = prepare_eval_plot_frame(eval_data)
    plot_data(
        eval_plot_frame,
        title,
        label,
        "Steps",
        "Average Reward",
        directory,
        filename,
        display,
    )


def prepare_train_plot_frame(
    train_data: pd.DataFrame, window_size: int
) -> pd.DataFrame:
    x_data: str = "total_steps"
    y_data: str = "episode_reward"

    plot_frame: pd.DataFrame = pd.DataFrame()
    plot_frame["steps"] = train_data[x_data]
    plot_frame["avg"] = (
        train_data[y_data].rolling(window_size, step=1, min_periods=1).mean()
    )
    plot_frame["std_dev"] = (
        train_data[y_data].rolling(window_size, step=1, min_periods=1).std()
    )

    return plot_frame


def plot_train(
    train_data: pd.DataFrame,
    title: str,
    label: str,
    directory: str,
    filename: str,
    window_size: int,
    display: bool = False,
) -> None:
    train_plot_frame = prepare_train_plot_frame(train_data, window_size)
    plot_data(
        train_plot_frame,
        title,
        label,
        "Steps",
        "Average Reward",
        directory,
        filename,
        display,
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

    param_tag = args["param_tag"]
    param = str(alg_config[param_tag]) if param_tag in alg_config else ""
    param += str(train_config[param_tag]) if param_tag in train_config else ""
    param = f"_{param_tag}_{param}" if param else ""
    label = algorithm + param

    title = task if title == "" else title
    return title, algorithm, task, label


def parse_args() -> dict:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_directory",
        type=str,
        required=True,
        help="Directory you want to save the data into",
    )

    parser.add_argument(
        "-d",
        "--data_directories",
        type=str,
        nargs="+",
        help="List of Directories with data you want to compare",
        required=True,
    )

    parser.add_argument(
        "-t",
        "--title",
        type=str,
        default="",
        help="Title for the plot - default will be taken from the task name",
    )

    parser.add_argument(
        "-x",
        "--x_axis",
        type=str,
        default="Steps",
        help="X Axis Label for the plot",
    )

    parser.add_argument(
        "-y",
        "--y_axis",
        type=str,
        default="Average Reward",
        help="Y Axis Label for the plot",
    )

    parser.add_argument(
        "-p",
        "--param_tag",
        type=str,
        default="",
        help="Tag to add to labels based on algorithm or training parameter",
    )

    parser.add_argument(
        "-ps",
        "--plot_seeds",
        action=argparse.BooleanOptionalAction,
        help="Plot Individual Seeds for each algorithm in addition to the average of all seeds",
    )

    parser.add_argument(
        "--label_fontsize",
        type=int,
        default=15,
        help="Fontsize for the x-axis label",
    )

    parser.add_argument(
        "--title_fontsize",
        type=int,
        default=20,
        help="Fontsize for the title",
    )

    parser.add_argument(
        "--ticks_fontsize",
        type=int,
        default=10,
        help="Fontsize for the ticks",
    )

    # converts into a dictionary
    args = vars(parser.parse_args())
    return args


def main():
    args = parse_args()

    title = args["title"]
    label_x = args["x_axis"]
    label_y = args["y_axis"]

    directory = args["save_directory"]

    eval_plot_frames = []
    labels = []

    for d, data_directory in enumerate(args["data_directories"]):
        logging.info(
            f"Processing {d+1}/{len(args['data_directories'])} Data for {data_directory}"
        )
        result_directories = glob(f"{data_directory}/*")

        title, algorithm, task, label = generate_labels(
            args, title, result_directories[0]
        )
        labels.append(label)

        average_eval_data = pd.DataFrame()

        seed_plot_frames = []
        seed_label = []

        for i, result_directory in enumerate(result_directories):
            logging.info(
                f"Processing Data for {algorithm}: {i+1}/{len(result_directories)} on task {task}"
            )

            if "train.csv" not in os.listdir(f"{result_directory}/data"):
                logging.warning(
                    f"Skipping {result_directory} as it does not have train.csv"
                )
                continue

            eval_data = pd.read_csv(f"{result_directory}/data/eval.csv")
            average_eval_data = pd.concat(
                [average_eval_data, eval_data], ignore_index=True
            )

            if args["plot_seeds"]:
                seed_label.append(f"{label}_{i}")
                seed_plot_frame = prepare_eval_plot_frame(eval_data)
                seed_plot_frames.append(seed_plot_frame)

        if args["plot_seeds"]:
            plot_comparisons(
                seed_plot_frames,
                f"{title}",
                seed_label,
                label_x,
                label_y,
                directory,
                f"{title}-{algorithm}-eval",
                label_fontsize=args["label_fontsize"],
                title_fontsize=args["title_fontsize"],
                ticks_fontsize=args["ticks_fontsize"],
                display=False,
            )

        eval_plot_frame = prepare_eval_plot_frame(average_eval_data)

        eval_plot_frames.append(eval_plot_frame)

    plot_comparisons(
        eval_plot_frames,
        f"{title}",
        labels,
        label_x,
        label_y,
        directory,
        f"{title}-compare-eval",
        label_fontsize=args["label_fontsize"],
        title_fontsize=args["title_fontsize"],
        ticks_fontsize=args["ticks_fontsize"],
        display=True,
    )


if __name__ == "__main__":
    main()
