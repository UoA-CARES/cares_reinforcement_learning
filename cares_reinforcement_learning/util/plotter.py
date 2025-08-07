import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)


def _create_plot(
    title: str,
    x_label: str,
    y_label: str,
    x_label_two: str = "",
    y_label_two: str = "",
    label_fontsize: int = 15,
    title_fontsize: int = 20,
    ticks_fontsize: int = 10,
) -> tuple[plt.Axes, plt.Axes | None]:
    matplotlib.use("agg")

    # Plot Styles
    plt.style.use("seaborn-v0_8")

    _, axis_one = plt.subplots()

    axis_one.set_xlabel(x_label, fontsize=label_fontsize)
    axis_one.set_ylabel(y_label, fontsize=label_fontsize)
    axis_one.tick_params(axis="y")
    axis_one.tick_params(axis="x", labelsize=ticks_fontsize)
    axis_one.tick_params(axis="y", labelsize=ticks_fontsize)
    axis_one.set_title(title, fontsize=title_fontsize)

    axis_two = None
    if x_label_two != "" and y_label_two != "":
        axis_two = axis_one.twinx()
        axis_two.set_xlabel(x_label_two, fontsize=label_fontsize)
        axis_two.set_ylabel(y_label_two, fontsize=label_fontsize)
        axis_two.tick_params(axis="y", labelsize=ticks_fontsize)
        axis_two.tick_params(axis="x", labelsize=ticks_fontsize)

    return axis_one, axis_two


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

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1)).set_draggable(True)

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
    x_label_two: str = "",
    y_label_two: str = "",
) -> None:

    axis_one, axis_two = _create_plot(
        title,
        x_label,
        y_label,
        x_label_two=x_label_two,
        y_label_two=y_label_two,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
        ticks_fontsize=ticks_fontsize,
    )

    for (plot_frame_one, plot_frame_two), label in zip(plot_frames, labels):
        sns.lineplot(
            data=plot_frame_one,
            x="x_data",
            y="y_data",
            ax=axis_one,
            label=label,
            legend=False,
        )

        axis_one.fill_between(
            plot_frame_one["x_data"],
            plot_frame_one["y_data"] - plot_frame_one["std_dev"],
            plot_frame_one["y_data"] + plot_frame_one["std_dev"],
            alpha=0.3,
        )

        if plot_frame_two is not None and axis_two is not None:
            sns.lineplot(
                data=plot_frame_two,
                x="x_data",
                y="y_data",
                ax=axis_two,
                label=label,
                linestyle="--",
                legend=False,
            )

            axis_two.fill_between(
                plot_frame_two["x_data"],
                plot_frame_two["y_data"] - plot_frame_two["std_dev"],
                plot_frame_two["y_data"] + plot_frame_two["std_dev"],
                alpha=0.3,
            )

    lines, labels = axis_one.get_legend_handles_labels()

    if axis_two is not None:
        lines_two, labels_two = (
            axis_two.get_legend_handles_labels() if axis_two else ([], [])
        )

        labels = [f"{label} ({y_label})" for label in labels]
        labels_two = [f"{label} ({y_label_two})" for label in labels_two]

        lines += lines_two
        labels += labels_two

    axis_one.legend(lines, labels, loc="best", bbox_to_anchor=(1.05, 1))

    plt.tight_layout(pad=0.5)

    if not os.path.exists(f"{directory}/figures"):
        os.makedirs(f"{directory}/figures")

    plt.savefig(f"{directory}/figures/{filename}.png")

    plt.close()


def _perpare_average_plot_frame(
    data_frame: pd.DataFrame,
    x_data: str = "x_data",
    y_data: str = "y_data",
) -> pd.DataFrame:

    average_data = data_frame.groupby(x_data).agg(["mean", "std"]).reset_index()

    plot_frame: pd.DataFrame = pd.DataFrame()
    plot_frame["x_data"] = average_data[x_data]
    plot_frame["y_data"] = average_data[y_data]["mean"]
    plot_frame["std_dev"] = average_data[y_data]["std"]

    return plot_frame


def _prepare_plot_frame(
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
    x_data: str = "total_steps",
    y_data: str = "episode_reward",
) -> None:

    eval_plot_frame = _prepare_plot_frame(
        eval_data, window_size=1, x_data=x_data, y_data=y_data
    )

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
    x_data: str = "total_steps",
    y_data: str = "episode_reward",
) -> None:
    x_label: str = "Steps"
    y_label: str = "Reward"

    train_plot_frame = _prepare_plot_frame(
        train_data, window_size=window_size, x_data=x_data, y_data=y_data
    )

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


def get_param_label(param_tag: str, alg_config: dict, train_config: dict) -> str:
    label = ""
    if param_tag == "":
        return label

    def get_param_value(param_tag: str, config: dict) -> str | None:
        if param_tag in config:
            return config[param_tag]
        return None

    value = get_param_value(param_tag, alg_config)
    if value is None:
        value = get_param_value(param_tag, train_config)

    if value is not None:
        label += f"_{param_tag}_{value}"

    return label


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

    param_tag = get_param_label(args["param_tag"], alg_config, train_config)
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

    group.add_argument(
        "-t",
        "--task_directory",
        type=str,
        help="Task Directory with algorithm results you want to compare",
    )

    parser.add_argument(
        "-s",
        "--save_directory",
        type=str,
        required=True,
        help="Save directory for the plots",
    )

    parser.add_argument(
        "--x_train",
        type=str,
        default="total_steps",
        help="Data you want to plot in x_axis for train - default is steps",
    )

    parser.add_argument(
        "--y_train",
        type=str,
        default="episode_reward",
        help="Data you want to plot in y_axis for train graphs - default is episode_reward",
    )

    parser.add_argument(
        "--y_train_two",
        type=str,
        default=None,
        help="Data you want to plot in second y_axis for train graphs - default is None",
    )

    parser.add_argument(
        "--x_eval",
        type=str,
        default="total_steps",
        help="Data you want to plot in x_axis for eval - default is steps",
    )

    parser.add_argument(
        "--y_eval",
        type=str,
        default="episode_reward",
        help="Data you want to plot in y_axis for eval graphs - default is episode_reward",
    )

    parser.add_argument(
        "--y_eval_two",
        type=str,
        default=None,
        help="Data you want to plot in a second y_axis for eval graphs - default is None",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Title for the plot - default is task name",
    )

    parser.add_argument(
        "--x_label_train",
        type=str,
        default=None,
        help="X Axis Label for the plot - default is x_data",
    )

    parser.add_argument(
        "--x_label_eval",
        type=str,
        default=None,
        help="X Axis Label for the plot - default is x_eval",
    )

    parser.add_argument(
        "--y_label_train",
        type=str,
        default=None,
        help="Y Axis Label for the plot - default is y_data",
    )

    parser.add_argument(
        "--y_label_train_two",
        type=str,
        default=None,
        help="Y Axis Label for the plot - default is y_data_two",
    )

    parser.add_argument(
        "--y_label_eval",
        type=str,
        default=None,
        help="Y Axis Label for the plot - default is y_eval",
    )

    parser.add_argument(
        "--y_label_eval_two",
        type=str,
        default=None,
        help="Y Axis Label for the plot - default is y_eval_two",
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
        type=str,
        default="",
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


def _get_task_directoies(task_directory: str) -> list[str]:

    model_path = Path(f"{task_directory}")
    directory = model_path.glob("*")

    result_directories: list[str] = [str(x) for x in directory if x.is_dir()]

    return result_directories


def _read_data(
    result_directory: str, file_name: str, window_size: int, x_data: str, y_data: str
) -> pd.DataFrame:
    data = pd.read_csv(f"{result_directory}/data/{file_name}.csv")

    plot_frame = _prepare_plot_frame(
        data, window_size=window_size, x_data=x_data, y_data=y_data
    )

    return plot_frame


def plot_evaluations():
    args = parse_args()

    x_train = args["x_train"]
    y_train = args["y_train"]
    y_train_two = args["y_train_two"]

    x_eval = args["x_eval"]
    y_eval = args["y_eval"]
    y_eval_two = args["y_eval_two"]

    # TODO eval_y and train_y should be different and based on their train/eval data
    title = args["title"]
    x_label_train = x_train if args["x_label_train"] is None else args["x_label_train"]
    y_label_train = y_train if args["y_label_train"] is None else args["y_label_train"]

    y_label_train_two = ""
    if y_train_two is not None:
        y_label_train_two = (
            y_train_two
            if args["y_label_train_two"] is None
            else args["y_label_train_two"]
        )

    x_label_eval = x_eval if args["x_label_eval"] is None else args["x_label_eval"]
    y_label_eval = y_eval if args["y_label_eval"] is None else args["y_label_eval"]

    y_label_eval_two = ""
    if y_eval_two is not None:
        y_label_eval_two = (
            y_eval_two if args["y_label_eval_two"] is None else args["y_label_eval_two"]
        )

    window_size = args["window_size"]

    save_directory = args["save_directory"]

    eval_plot_frames = []
    train_plot_frames = []
    labels = []

    directories: list[str] = []

    if args.get("data_directories") is not None:
        directories = args["data_directories"]
    elif args.get("task_directory") is not None:
        directories = _get_task_directoies(args["task_directory"])

    # plot list of directories against each other
    for index, data_directory in enumerate(directories):
        logging.info(
            f"Processing {index+1}/{len(directories)} Data for {data_directory}"
        )
        model_path = Path(f"{data_directory}")
        directory = model_path.glob("*")

        result_directories = [x for x in directory if x.is_dir()]

        title, algorithm, task, label = generate_labels(args, title, model_path)

        labels.append(label)

        average_train_data = pd.DataFrame()
        average_train_data_two = pd.DataFrame() if y_train_two is not None else None

        average_eval_data = pd.DataFrame()
        average_eval_data_two = pd.DataFrame() if y_eval_two is not None else None

        seed_train_plot_frames = []
        seed_eval_plot_frames = []
        seed_label = []

        # Single Seed Data for result
        for i, result_directory in enumerate(result_directories):
            logging.info(
                f"Processing Data for {algorithm}: {i+1}/{len(result_directories)} on task {task}"
            )

            if "train.csv" not in os.listdir(f"{result_directory}/data"):
                logging.warning(
                    f"Skipping {result_directory} as it does not have train.csv"
                )
                continue

            train_data = _read_data(
                result_directory, "train", window_size, x_train, y_train
            )

            if y_train_two is not None:
                train_data_two = _read_data(
                    result_directory, "train", window_size, x_train, y_train_two
                )
                average_train_data_two = pd.concat(
                    [average_train_data_two, train_data_two], ignore_index=True
                )

            average_train_data = pd.concat(
                [average_train_data, train_data], ignore_index=True
            )

            if "eval.csv" not in os.listdir(f"{result_directory}/data"):
                logging.warning(
                    f"Skipping {result_directory} as it does not have eval.csv"
                )
                continue

            eval_data = _read_data(
                result_directory, "eval", window_size, x_eval, y_eval
            )

            if y_eval_two is not None:
                eval_data_two = _read_data(
                    result_directory, "eval", window_size, x_eval, y_eval_two
                )
                average_eval_data_two = pd.concat(
                    [average_eval_data_two, eval_data_two], ignore_index=True
                )

            average_eval_data = pd.concat(
                [average_eval_data, eval_data], ignore_index=True
            )

            if args["plot_seeds"]:
                seed_label.append(f"{label}_{i}")
                seed_eval_plot_frames.append(eval_data)
                seed_train_plot_frames.append(train_data)

        if args["plot_seeds"]:
            # Plot the individual training seeds
            plot_comparisons(
                seed_train_plot_frames,
                f"{title}",
                seed_label,
                x_label_train,
                y_label_train,
                save_directory,
                f"{title}-{algorithm}-train",
                label_fontsize=args["label_fontsize"],
                title_fontsize=args["title_fontsize"],
                ticks_fontsize=args["ticks_fontsize"],
            )

            # Plot the individual evaluation seeds
            plot_comparisons(
                seed_eval_plot_frames,
                f"{title}",
                seed_label,
                x_label_eval,
                y_label_eval,
                save_directory,
                f"{title}-{algorithm}-eval",
                label_fontsize=args["label_fontsize"],
                title_fontsize=args["title_fontsize"],
                ticks_fontsize=args["ticks_fontsize"],
            )

        train_plot_frame = _perpare_average_plot_frame(average_train_data)

        train_plot_frame_two = None
        if y_train_two is not None:
            train_plot_frame_two = _perpare_average_plot_frame(average_train_data_two)

        train_plot_frames.append([train_plot_frame, train_plot_frame_two])

        eval_plot_frame = _perpare_average_plot_frame(average_eval_data)

        eval_plot_frame_two = None
        if y_eval_two is not None:
            eval_plot_frame_two = _perpare_average_plot_frame(average_eval_data_two)

        eval_plot_frames.append([eval_plot_frame, eval_plot_frame_two])

    # Plot the training comparisons
    plot_comparisons(
        train_plot_frames,
        f"{title}",
        labels,
        x_label_train,
        y_label_train,
        save_directory,
        f"{title}-compare-train",
        label_fontsize=args["label_fontsize"],
        title_fontsize=args["title_fontsize"],
        ticks_fontsize=args["ticks_fontsize"],
        x_label_two=x_label_train,
        y_label_two=y_label_train_two,
    )

    # Plot the evaluation comparisons
    plot_comparisons(
        eval_plot_frames,
        f"{title}",
        labels,
        x_label_eval,
        y_label_eval,
        save_directory,
        f"{title}-compare-eval",
        label_fontsize=args["label_fontsize"],
        title_fontsize=args["title_fontsize"],
        ticks_fontsize=args["ticks_fontsize"],
        x_label_two=x_label_eval,
        y_label_two=y_label_eval_two,
    )


def main():
    plot_evaluations()


if __name__ == "__main__":
    main()
