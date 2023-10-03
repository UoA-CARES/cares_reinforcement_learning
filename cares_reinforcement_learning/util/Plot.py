import os

import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import uuid

# TODO make this more easy and simple, plot and store checkpoints

def plot_average(x, y, x_label='x_value',y_label='y_value', title='Title', window_size=10, file_path='figures/figure.png', display=False):
    
    figure = plt.figure()
    figure.set_figwidth(8) 
    sns.set_theme(style="darkgrid")
    
    data_dict = {x_label: x, y_label: y}
    df = pd.DataFrame(data=data_dict)

    df["avg"] = df[y_label].rolling(window_size, min_periods=1).mean()
    df["std_dev"] = df[y_label].rolling(window_size, min_periods=1).std()
    
    ax = sns.lineplot(data=df, x=x_label, y="avg", label='Average')
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.fill_between(df[x_label], df["avg"] - df["std_dev"], df["avg"] +
                    df["std_dev"], alpha=0.4)

    sns.move_legend(ax, "lower right")

    plt.title(title) 
    
    plt.savefig(file_path)
    plt.close(figure)
 
class Plot:
    def __init__(self, title='Training', x_label='Episode', y_label='Reward', x_data=[], y_data=[], plot_freq=1, checkpoint_freq=1):
        plt.ion()

        self.title = title

        self.x_label = x_label
        self.y_label = y_label

        self.figure = plt.figure()
        self.figure.set_figwidth(8)

        self.x_data = x_data
        self.y_data = y_data

        self.plot_num = 0
        self.plot_freq = plot_freq
        self.checkpoint_freq = checkpoint_freq

        sns.set_theme(style="darkgrid")

    def post(self, reward):
        self.plot_num += 1

        self.x_data.append(len(self.x_data))
        self.y_data.append(reward)

        if self.plot_num % self.plot_freq == 0:
            self.__create_plot()
            plt.pause(10e-10)

        if self.plot_num % self.checkpoint_freq == 0:
            pass
            #self.save_csv(f'{self.title}.csv')

    def plot(self):
        plt.ioff()
        self.__create_plot()
        plt.show()

    def __create_plot(self):
        data_dict = {self.x_label: self.x_data, self.y_label: self.y_data}
        df = pd.DataFrame(data=data_dict)

        sns.lineplot(data=df, x=self.x_label, y=self.y_label)
        plt.title(self.title)

    def save_plot(self, file_name=str(uuid.uuid4().hex)):
        self.__create_plot()

        dir_exists = os.path.exists("figures")

        if not dir_exists:
            os.makedirs("figures")

        plt.savefig(f"figures/{file_name}")

    def plot_average(self, window_size=10, file_name=str(uuid.uuid4().hex)):

        plt.ioff()
        
        data_dict = {"Episode": self.x_data, "Reward": self.y_data}
        df = pd.DataFrame(data=data_dict)

        df["Average Reward"] = df["Reward"].rolling(window_size).mean()
        df["Standard Deviation"] = df["Reward"].rolling(window_size).std()

        ax = sns.lineplot(data=df, x="Episode", y="Average Reward", label="Average Reward")
        ax.set(xlabel="Episode", ylabel="Reward")
        plt.fill_between(df["Reward"], df["Average Reward"] - df["Standard Deviation"], df["Average Reward"] +
                        df["Standard Deviation"], alpha=0.4)

        sns.move_legend(ax, "lower right")

        plt.title(self.title)

        dir_exists = os.path.exists("figures")

        if not dir_exists:
            os.makedirs("figures")

        plt.savefig(f"figures/{file_name}")

        plt.show()

def parse_eval_data(eval_data):
    window_size = eval_data['episode'].max()
    print(f'Window Size: {window_size}')

    x_label = "steps"
    y_label = "episode_reward"
    
    eval_data[x_label] = eval_data["train_step"].rolling(window_size, min_periods=1).mean()
    eval_data["avg"] = eval_data[y_label].rolling(window_size, min_periods=1).mean()
    eval_data["std_dev"] = eval_data[y_label].rolling(window_size, min_periods=1).std()
    
    ax = sns.lineplot(data=eval_data, x=x_label, y="avg", label='Average')
    ax.set(xlabel=x_label, ylabel=y_label)
    
    Z = 1.960 # 95% confidence interval
    confidence_interval = Z * eval_data["std_dev"] / np.sqrt(window_size)

    plt.fill_between(eval_data[x_label], eval_data["avg"] - confidence_interval, eval_data["avg"] + confidence_interval, alpha=0.4)

    sns.move_legend(ax, "lower right")

    plt.title("Title") 
    
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()  # Add an argument

    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--eval',  type=str, required=True)

    return vars(parser.parse_args())  # converts into a dictionary

def main():
    args = parse_args()

    train_data = pd.read_csv(args['train'])
    eval_data  = pd.read_csv(args['eval'])

    parse_eval_data(eval_data)

    sns.lineplot(data=train_data, x='total_steps', y='reward')
    plt.show()


if __name__ == '__main__':
    main()