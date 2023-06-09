import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import uuid

# TODO make this more easy and simple, plot and store checkpoints

def plot_average(x, y, x_label='x_value',y_label='y_value', title='Title', window_size=10, file_path='figures/figure.png', display=False):
    plt.ioff()
    plt.cla()
    plt.clf()
    
    figure = plt.figure()
    figure.set_figwidth(8)
     
    sns.set_theme(style="darkgrid")
    
    data_dict = {x_label: x, y_label: y}
    df = pd.DataFrame(data=data_dict)

    df["Average"] = df[y_label].rolling(window_size).mean()
    df["Standard Deviation"] = df[y_label].rolling(window_size).std()

    ax = sns.lineplot(data=df, x=x_label, y=y_label, label='Average')
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.fill_between(df[x_label], df["Average"] - df["Standard Deviation"], df["Average"] +
                    df["Standard Deviation"], alpha=0.4)

    sns.move_legend(ax, "lower right")

    plt.title(title) 
    
    plt.savefig(file_path)
    
class Plot:
    def __init__(self, title='Training', x_label='Episode', y_label='Reward', x_data=None, y_data=None, plot_freq=1, checkpoint_freq=1):
        if x_data is None:
            x_data = []
        if y_data is None:
            y_data = []

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
            self.save_csv(f'{self.title}.csv')

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
        plt.fill_between(df["Episode"], df["Average Reward"] - df["Standard Deviation"], df["Average Reward"] +
                        df["Standard Deviation"], alpha=0.4)

        sns.move_legend(ax, "lower right")

        plt.title(self.title)

        dir_exists = os.path.exists("figures")

        if not dir_exists:
            os.makedirs("figures")

        plt.savefig(f"figures/{file_name}")

        plt.show()

#
#
# def read_file(file_path: str):
#     """
#     Reads a file that contains rewards separated by new line
#
#     Parameters:
#         file_path: a string path to the data file
#     """
#     file = open(file_path, "r")
#     strings = file.readlines()
#     floats = [float(x) for x in strings]
#     return floats
#
#
# def plot_learning(title: str, reward, file_name: str = "figure.png"):
#     """
#     Plot the learning of the agent. Saves the figure to figures directory
#
#     Parameters:
#         title: title of the plot
#         reward: the array of rewards to be plot
#         file_name: the name of the figure when saved to disc
#     """
#     y = reward
#     x = range(1, len(reward) + 1)
#
#     print(reward)
#     print(x)
#
#     data_dict = {"Episode": x, "Reward": y}
#     df = pd.DataFrame(data=data_dict)
#
#     sns.set_theme(style="darkgrid")
#     plt.figure().set_figwidth(8)
#
#     sns.lineplot(data=df, x="Episode", y="Reward")
#     plt.title(title)
#
#     dir_exists = os.path.exists("figures")
#
#     if not dir_exists:
#         os.makedirs("figures")
#
#     plt.savefig(f"figures/{file_name}")
#     plt.show()
#
#
# def plot_learning_vs_average(title: str, reward, file_name: str = "figure.png", window_size: int = 10):
#     """
#     Plot the rolling average and the actual learning. Saves the figure to figures directory
#
#     Parameters:
#         title: title of the plot
#         reward: the array of rewards to be plot
#         file_name: the name of the figure when saved to disc
#         window_size: the size of the rolling average window
#     """
#     y = reward
#     x = range(1, len(reward) + 1)
#
#     data_dict = {"Episode": x, "Reward": y}
#     df = pd.DataFrame(data=data_dict)
#
#     df["Average Reward"] = df["Reward"].rolling(window_size).mean()
#
#     sns.set_theme(style="darkgrid")
#     plt.figure().set_figwidth(8)
#
#     sns.lineplot(data=df, x="Episode", y="Reward", alpha=0.4)
#     sns.lineplot(data=df, x="Episode", y="Average Reward")
#
#     plt.fill_between(df["Episode"], df["Reward"], df["Average Reward"], alpha=0.4)
#     plt.title(title)
#
#     dir_exists = os.path.exists("figures")
#
#     if not dir_exists:
#         os.makedirs("figures")
#
#     plt.savefig(f"figures/{file_name}")
#
#     plt.show()
#
#
# def plot_average_with_std(reward,
#                           title: str = "Cool Graph",
#                           file_name: str = "figure.png",
#                           window_size: int = 10):
#     """
#     Plot the rolling average and standard deviation. Saves the figure to figures directory
#
#     Parameters:
#         title: title of the plot
#         reward: the array of rewards to be plot
#         file_name: the name of the figure when saved to disc
#         window_size: the size of the rolling average window
#     """
#     y = reward
#     x = range(1, len(reward) + 1)
#
#     data_dict = {"Episode": x, "Reward": y}
#     df = pd.DataFrame(data=data_dict)
#
#     df["Average Reward"] = df["Reward"].rolling(window_size).mean()
#     df["Standard Deviation"] = df["Reward"].rolling(window_size).std()
#
#     sns.set_theme(style="darkgrid")
#     plt.figure().set_figwidth(8)
#
#     ax = sns.lineplot(data=df, x="Episode", y="Average Reward", label="Average Reward")
#     ax.set(xlabel="Episode", ylabel="Reward")
#     plt.fill_between(df["Episode"], df["Average Reward"] - df["Standard Deviation"], df["Average Reward"] +
#                      df["Standard Deviation"], alpha=0.4)
#
#     sns.move_legend(ax, "lower right")
#
#     plt.title(title)
#
#     dir_exists = os.path.exists("figures")
#
#     if not dir_exists:
#         os.makedirs("figures")
#
#     plt.savefig(f"figures/{file_name}")
#
#     plt.show()
#
