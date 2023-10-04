import os

import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import uuid

def plot_data(plot_frame, title, label, x_label, y_label, directory, filename, display=True, close_figure=True):
    window_size = plot_frame["window_size"]

    ax = sns.lineplot(data=plot_frame, x=plot_frame["steps"], y="avg", label=label)
    ax.set(xlabel=x_label, ylabel=y_label)
    
    Z = 1.960 # 95% confidence interval
    confidence_interval = Z * plot_frame["std_dev"] / np.sqrt(window_size)

    plt.fill_between(plot_frame["steps"], plot_frame["avg"] - confidence_interval, plot_frame["avg"] + confidence_interval, alpha=0.4)

    plt.title(title) 

    plt.savefig(f"{directory}/figures/{filename}.png")

    if display:
        plt.show()

    if close_figure:
        plt.close()
    
def plot_comparisons(plot_frames, title, labels, x_label, y_label, directory, filename, display=True):
    for plot_frame, label in zip(plot_frames, labels):
        plot_data(plot_frame, title, label, x_label, y_label, directory, filename, display=False, close_figure=False)
    
    if display:
        plt.show()

    plt.close()

def prepare_eval_plot_frame(eval_data):
    x_data = "total_steps"
    y_data = "episode_reward"

    window_size = eval_data['episode'].max()

    plot_frame = pd.DataFrame()

    frame_average = eval_data.groupby([x_data], as_index=False).mean()
    frame_std = eval_data.groupby([x_data], as_index=False).std()

    plot_frame["steps"]   = frame_average[x_data]
    plot_frame["avg"]     = frame_average[y_data]
    plot_frame["std_dev"] = frame_std[y_data]
    plot_frame["window_size"] = window_size

    return plot_frame

def plot_eval(eval_data, title, label, directory, filename, display=False):
    eval_plot_frame = prepare_eval_plot_frame(eval_data)
    plot_data(eval_plot_frame, title, label, "Steps", "Average Reward", directory, filename, display)

def prepare_train_plot_frame(train_data, window_size):
    x_data = "total_steps"
    y_data = "episode_reward"

    plot_frame = pd.DataFrame()
    plot_frame["steps"] = train_data[x_data]
    plot_frame["avg"]  = train_data[y_data].rolling(window_size, step=1, min_periods=1).mean()
    plot_frame["std_dev"] = train_data[y_data].rolling(window_size, step=1, min_periods=1).std()
    plot_frame["window_size"] = window_size

    return plot_frame

def plot_train(train_data, title, label, directory, filename, window_size, display=False):
    train_plot_frame = prepare_train_plot_frame(train_data, window_size)
    plot_data(train_plot_frame, title, label, "Steps", "Average Reward", directory, filename, display)

def parse_args():
    parser = argparse.ArgumentParser()  # Add an argument

    parser.add_argument('-d','--data_path', type=str, required=True)

    return vars(parser.parse_args())  # converts into a dictionary

def main():
    args = parse_args()

    directory = args["data_path"]
    train_data = pd.read_csv(f"{args['data_path']}/data/train.csv")
    eval_data  = pd.read_csv(f"{args['data_path']}/data/eval.csv")

    train_plot_frame = prepare_train_plot_frame(train_data, window_size=1)
    rolling_train_plot_frame = prepare_train_plot_frame(train_data, window_size=2)
    eval_plot_frame = prepare_eval_plot_frame(eval_data)

    plot_comparisons([train_plot_frame, rolling_train_plot_frame, eval_plot_frame], "Comparison", ['train', 'train-rolling', 'eval'], "Steps", "Average Reward", directory, "compare", True)
    
if __name__ == '__main__':
    main()