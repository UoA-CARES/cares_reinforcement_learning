import os
import csv

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:

    def write_to_file(self, column_title: str, file_name: str, *args):
        """
        Write arrays of data to file. Data written to raw_data directory

        Parameters:
            column_title: a string containing column headers separated by space
                e.g. `column_title` = "episode reward"`
            file_name: the name of the file to be written into
            *args: any number of arrays that you want saved to file. Pass the arrays into the function as arguments

        Returns:
            Nothing

        Example Usage:

        write_to_file("Episode Reward", "results.csv", episode_array, reward_array)
        """

        dir_exists = os.path.exists("raw_data")

        if not dir_exists:
            os.makedirs("raw_data")

        file_out = open(f"raw_data/{file_name}", "w")
        csv_out = csv.writer(file_out)

        csv_out.writerow(column_title.split(" "))
        for row in zip(*args):
            csv_out.writerow(row)

    def plot_learning(self, title: str, reward, file_name: str = "figure.png"):
        """
        Plot the learning of the agent. Saves the figure to figures directory

        Parameters:
            title: title of the plot
            reward: the array of rewards to be plot
            file_name: the name of the figure when saved to disc
        """
        y = reward
        x = range(1, len(reward) + 1)

        print(reward)
        print(x)

        data_dict = {"Episode": x, "Reward": y}
        df = pd.DataFrame(data=data_dict)

        sns.set_theme(style="darkgrid")
        plt.figure().set_figwidth(8)

        sns.lineplot(data=df, x="Episode", y="Reward")
        plt.title(title)

        dir_exists = os.path.exists("figures")

        if not dir_exists:
            os.makedirs("figures")

        plt.savefig(f"figures/{file_name}")
        plt.show()

    def plot_learning_average(self, title: str, reward, file_name: str = "figure.png"):
        """
        Plot the rolling average and the actual learning. Saves the figure to figures directory

        Parameters:
            title: title of the plot
            reward: the array of rewards to be plot
            file_name: the name of the figure when saved to disc
        """
        y = reward
        x = range(1, len(reward) + 1)

        data_dict = {"Episode": x, "Reward": y}
        df = pd.DataFrame(data=data_dict)

        df["Average Reward"] = df["Reward"].rolling(100).mean()

        sns.set_theme(style="darkgrid")
        plt.figure().set_figwidth(8)

        sns.lineplot(data=df, x="Episode", y="Reward", alpha=0.4)
        sns.lineplot(data=df, x="Episode", y="Average Reward")

        plt.fill_between(df["Episode"], df["Reward"], df["Average Reward"], alpha=0.4)
        plt.title(title)

        dir_exists = os.path.exists("figures")

        if not dir_exists:
            os.makedirs("figures")

        plt.savefig(f"figures/{file_name}")

        plt.show()

    def plot_average_std(self, title: str, reward, file_name: str = "figure.png"):
        """
        Plot the rolling average and the standard deviation. Saves the figure to figures directory

        Parameters:
            title: title of the plot
            reward: the array of rewards to be plot
            file_name: the name of the figure when saved to disc
        """
        y = reward
        x = range(1, len(reward) + 1)

        data_dict = {"Episode": x, "Reward": y}
        df = pd.DataFrame(data=data_dict)

        df["Average Reward"] = df["Reward"].rolling(500).mean()
        df["Standard Deviation"] = df["Reward"].rolling(500).std()

        sns.set_theme(style="darkgrid")
        plt.figure().set_figwidth(8)

        ax = sns.lineplot(data=df, x="Episode", y="Average Reward", label="Average Reward")
        ax.set(xlabel="Episode", ylabel="Reward")
        plt.fill_between(df["Episode"], df["Average Reward"] - df["Standard Deviation"], df["Average Reward"] +
                         df["Standard Deviation"], alpha=0.4)

        sns.move_legend(ax, "lower right")

        plt.title(title)

        dir_exists = os.path.exists("figures")

        if not dir_exists:
            os.makedirs("figures")

        plt.savefig(f"figures/{file_name}")

        plt.show()
