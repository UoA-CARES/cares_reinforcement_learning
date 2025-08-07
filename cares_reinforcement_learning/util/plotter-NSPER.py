import argparse
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# Constants
WINDOW_SIZE = 10
Z = 1.960  # 95% confidence interval (Z-score for normal distribution)

# Seaborn style for dark background
# plt.style.use("dark_background")
# Set a soft grey background with stylish grid
sns.set_style("whitegrid")  
plt.rcParams["axes.facecolor"] = "#F0F0F0"  # Light grey background

# Define color mapping with high-contrast colors
palette = palette = ["#E41A1C","#66C2A5", "#8DA0CB", "#FC8D62", "#A6D854", "#E78AC3", "#FFD92F"] 

#sns.color_palette("coolwarm",10)  # Gradient-based palette
color_mapping = {
    "NSPER": palette[0],
    "NSPER+R": palette[0],
    "NOVELTY": palette[1],
    "NOVELTY+R": palette[1],
    "SURPRISE": palette[2],
    "SURPRISE+R": palette[2],
}

def load_and_process_csv(file_path):
    """Load CSV and compute confidence interval."""
    try:
        df = pd.read_csv(file_path)

        if 'step' not in df.columns or 'avg_episode_reward' not in df.columns:
            logging.warning(f"Skipping {file_path}: Missing required columns.")
            return None

        df = df.sort_values("step")  # Ensure sorted steps

        # Compute rolling mean & standard error
        df["rolling_mean"] = df["avg_episode_reward"].rolling(WINDOW_SIZE, min_periods=1).mean()
        df["rolling_sem"] = df["avg_episode_reward"].rolling(WINDOW_SIZE, min_periods=1).sem()  # Standard Error
        df["conf_int_pos"] = df["rolling_mean"] + Z * df["rolling_sem"]  # Upper Bound
        df["conf_int_neg"] = df["rolling_mean"] - Z * df["rolling_sem"]  # Lower Bound

        return df

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

def plot_results(folder_path, save_fig=True, show_fig=False):
    """Plot CSV files from a folder using confidence intervals with advanced styling."""
    if not os.path.exists(folder_path):
        logging.error(f"Folder '{folder_path}' does not exist.")
        return

    file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not file_names:
        logging.warning(f"No CSV files found in '{folder_path}'.")
        return

    plt.figure(figsize=(12, 7), dpi=300)

    for name, color in color_mapping.items():
        file_path = os.path.join(folder_path, f"{name}.csv")

        if os.path.exists(file_path):
            df = load_and_process_csv(file_path)
            if df is None:
                continue

            linestyle = '-' if "+R" in name else '--'

            # Plot line
            sns.lineplot(
                x=df["step"], y=df["rolling_mean"], label=name, color=color, linestyle=linestyle, linewidth=2
            )

            # Gradient-filled confidence bands
            plt.fill_between(df["step"], df["conf_int_neg"], df["conf_int_pos"], color=color, alpha=0.3)

    # Aesthetics
    plt.xlabel("Million Steps", fontsize=25, fontweight="bold", color="white")
    plt.ylabel("Average Return", fontsize=25, fontweight="bold", color="white")
    folder_name = os.path.basename(folder_path)
    plt.title(f'{folder_name}', fontsize=40, weight='bold')

    #plt.title(os.path.basename(folder_path), fontsize=16, fontweight="bold", color="white")

    # Improve legend
    plt.legend(frameon=True, fontsize=12, loc="best", fancybox=True, edgecolor="white")

    # Improve grid appearance
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')

    # Better layout
    #plt.tight_layout()

    # Save figure
    if save_fig:

        save_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}.png")
     
        folder_name = os.path.basename(folder_path)

        plt.savefig(f"{folder_path}/figures/{folder_name}.png", dpi=300, bbox_inches='tight')


    if show_fig:
        plt.show()

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RL results with confidence intervals from CSV files.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing CSV files.")
    parser.add_argument("--show", action="store_true", help="Display the plot instead of saving it.")
    args = parser.parse_args()

    plot_results(args.folder_path, save_fig=not args.show, show_fig=args.show)
