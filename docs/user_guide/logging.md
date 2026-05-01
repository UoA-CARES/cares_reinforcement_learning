--8<-- "include/glossary.md"
# Logging

CARES RL provides comprehensive logging for all experiments, ensuring reproducibility, transparency, and ease of debugging. All key training, evaluation, and environment parameters, as well as algorithm-specific metrics, are automatically recorded and organized for every run.

## What Gets Logged

All configuration files: `env_config.json`, `alg_config.json`, `train_config.json` (and others as needed) are stored with all parameters required to replicate the experiment. This folder will contain the following directories and information saved during the training session:

- For each seed:
    - `data/train.csv` and `data/eval.csv`: step-by-step logs of training and evaluation, including rewards, losses, and all algorithm-specific metrics
    - `models/`: all checkpoints, final, and best models
    - `figures/`: plots of training/evaluation curves
    - `videos/`: (if enabled) videos of agent performance

!!! example "Training output structure"

    ```text
    <log_path>/
    ├── env_config.json
    ├── alg_config.json
    ├── train_config.json
    ├── *_config.json
    ├── SEED_N/
    │   ├── data/
    │   │   ├── train.csv
    │   │   └── eval.csv
    │   ├── figures/
    │   │   ├── eval.png
    │   │   └── train.png
    │   ├── models/
    │   │   ├── model.pht
    │   │   ├── CHECKPOINT_N.pht
    │   │   └── ...
    │   └── videos/
    │       ├── STEP.mp4
    │       └── ...
    └── ...
    ```

## Log Locations and Formatting

By default logs are saved into a folder under `~/cares_rl_logs/` with the naming convention of `{algorithm}/{algorithm}-{domain_task}-{date}`. You can control where and how logs are saved using environment variables:

- `CARES_LOG_BASE_DIR`: Sets the root directory for all logs. Example:
    ```sh
    export CARES_LOG_BASE_DIR=~/my_custom_logs
    ```
    If unset, defaults to `~/cares_rl_logs`.
- `CARES_LOG_PATH_TEMPLATE`: Controls the folder structure and naming. Supports variables like `{algorithm}`, `{domain_task}`, `{date}`. Example:
    ```sh
    export CARES_LOG_PATH_TEMPLATE="{algorithm}/{algorithm}-{domain_task}-{date}"
    ```
    These variables are replaced automatically for each run.

## Debugging with Logs

The logging system captures far more than just reward curves. Each run records environment configurations, algorithm settings, and a wide range of training metrics, providing a complete view of how an experiment was executed. This information is essential for reproducing results, comparing runs, and diagnosing unexpected behaviour.

During training, logs include both general performance metrics (e.g. rewards, episode length) and algorithm-specific learning signals (e.g. Q-values, entropy, exploration rates). These metrics help explain *why* or *how* well an algorithm is or is not learning, rather than just relying on the reward curve.

For example:

- If DQN is not improving, check whether epsilon is decaying as expected and whether Q-values are stable or diverging.
- If SAC appears unstable, inspect entropy and alpha values to understand the exploration–exploitation balance.
- If results vary significantly between runs, compare configuration files and seed-specific outputs to identify inconsistencies.

!!! note "Algorithm-Specific Metrics"
    Refer to the corresponding algorithm documentation for detailed guidance on how to interpret training log values for each algorithm.

## Plotting

The `cares-rl-plot` utility can plot the data from one or multiple training sessions. Features include:

- Plot multiple runs for comparison
- Support for dual y-axes (e.g., reward and loss on the same plot)
- Customizable axis labels, title, font sizes, and legend
- Plots are saved as PNGs in the given folder `-s <PATH_TO_OUTPUT>` directory

**Example** Plot and compare the results of two or more training instances:
```sh
cares-rl-plot -s ~/cares_rl_logs -d <PATH_TO_ONE> <PATH_TO_TWO>
```

!!! tip "Full Plotting Options"
    For all plotting configuration options run:
    ```sh
    cares-rl-plot -h
    ```

---8<-- "include/links.md"