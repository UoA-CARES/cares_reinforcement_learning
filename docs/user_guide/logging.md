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

Logs include not just rewards and losses, but also internal metrics (e.g., Q-values, entropies, exploration rates) to help diagnose why an algorithm may not be learning. For example:

- If DQN is not learning, check if epsilon is decaying as expected, or if Q-values are diverging.
- If SAC is unstable, check entropy and alpha values.

!!! note "Algorithm-Specific Metrics"
        Some metrics are algorithm-specific (e.g. entropy in SAC, epsilon in DQN). See the individual algorithm documentation for detailed explanations of how to interpret these values.

## Plotting

The `cares-rl-plot` utility can plot the data from one or multiple training sessions. Features include:

- Plot single or multiple runs for comparison
- Support for dual y-axes (e.g., reward and loss on the same plot)
- Customizable axis labels, title, font sizes, and legend
- Plots are saved as PNGs in the `figures/` directory

Example usage:

```sh
cares-rl-plot -h
```

Plot the results of a single training instance:
```sh
cares-rl-plot -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS
```

Plot and compare the results of two or more training instances:
```sh
cares-rl-plot -s ~/cares_rl_logs -d <run1> -d <run2> --y2 loss --y2_label "Loss"
```

## Best Practices

- Always keep logs for reproducibility
- Use environment variables to organize experiments
- Use plotting to compare across seeds, algorithms, or hyperparameters

```sh
cares-rl-plot -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM_A/ALGORITHM_A-TASK-YY_MM_DD:HH:MM:SS ~/cares_rl_logs/ALGORITHM_B/ALGORITHM_B-TASK-YY_MM_DD:HH:MM:SS
```

---8<-- "include/links.md"