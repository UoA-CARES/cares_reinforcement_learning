--8<-- "include/glossary.md"
# Usage Guide

This section provides a focused overview of how to use the CARES Reinforcement Learning framework from the command line. Here you'll find practical examples and explanations for running training, evaluation, testing, and plotting commands, as well as tips for customizing your experiments. Whether you're running quick tests or large-scale experiments, this guide will help you make the most of the CLI tools and configuration options.

!!! tip "Usage Guide"
    The library is configuration-driven, meaning most behaviour can be controlled through configuration files and command-line overrides rather than modifying code.

## Running Training and Evaluation

The main entry point is the `cares-rl` CLI command with the specific commands: 

- `train` - start a training sessions
- `resume` - resume a training session from a checkpoint (not deterministic)
- `evaluate` - re-run the evaluation of a training sessions
- `test` - test the trained model against a different seed over a given number of episodes

!!! tip "Command Help"
    Use `cares-rl -h` for help on all commands and options.

## Training an Agent (train)

The `train` command is used to start a new training session for a reinforcement learning agent in a specified environment. You can use either direct command line inputs (cli) overrides for quick experiments or configuration files (config) for full reproducibility. Below are examples for both approaches, along with tips for customizing your training runs.

The commands to choose whether to use the command line interface or configuration file approach:

- `cli` - read parameter overrides from the command line using the custom [rl_parser][rl-parser] arguement praser 
- `config` loads everything from config files for full reproducibility.

!!! tip "cli versus config"
    For quick experiments, use `cli` to change simple configuration parameters to key parameters and the defaults for the rest.
    
    For reproducibility, use the `config` and the configuration files which allow full control over all parameters for consistency between runs and allow you to configure an aglorithm's network architecture. 

### Command Line Input (cli)
The `cli` (command line input) mode lets you specify all training, environment, and algorithm parameters directly as command-line flags. This is ideal for quick experiments, prototyping, or when you want to override just a few settings without creating configuration files. 

!!! warning "cli limitations"
    The network architectures for a given algorithm can't be changed via command line - at least not in a nice convenient style. Use the `config` command for that instead.  

**Gym Envrionment**: Choose the Gym Envrionment and Task

```bash
cares-rl train cli --gym <GYM> --task <TASK>
```

**Algorithm**: Choose the algorithm and optionally override the default training parameters.

```bash
cares-rl train cli --gym <GYM> --task <TASK> <Algorithm> --<PARAMETER> <VALUE>
```

!!! tip "Algorithm Configurations"
    To find all configuration parameters for a given Algorithm you can find them in the [configurations file][config-file] under `<Algorithm>Config`.


**Example**: Train a DQN agent on CartPole-v1

The command below runs DQN setting the learning rate to 0.001, gamma to 0.99, and batch size to 32. 
```bash
cares-rl train cli --gym openai --task CartPole-v1 DQN --lr 0.001 --gamma 0.99 --batch_size 32
```

!!! tip "Configurations Saved"
    The full training configurations are saved into the training logs for future reference - default location `~/cares_rl_logs/<ALGORITHM>`

!!! tip "Enable future Resume"
    In order to enable the resume feature for a given experiment you need to manually enable `--save_train_checkpoints 1`

### Configuration Files (config)

The `config` mode allows you to specify all experiment parameters in structured JSON configuration files, rather than on the command line. This approach is ideal for reproducible research, large-scale experiments, or when you want to precisely control every aspect of your environment, algorithm, and training setup.

All parameters, including advanced options like network architectures, optimizer settings, and environment details, can be set in the config files. The CLI will load these files and use them for the entire run, ensuring consistency and repeatability.

**Data Path structure:** The command will read the three files from the provided path: 
```
~/my_experiment/
    train_config.json
    env_config.json
    alg_config.json
```

**Example: Running with config files**

The command below runs the training as defined by the configuration files detailed above. 
```bash
cares-rl train config --data_path ~/my_experiment/
```

!!! tip "Generating Configuration Files"
    You can generate JSON files manually, or use a previous run's output as a template for new experiments.

!!! tip "Defaults Override Unspecificed Parameters"
    Not all parameters need to be defined in the config files - defaults will be automatically used where parameters are not defined in the config file. 

    The custom recording tool will generate the complete configuration files in the logging directory.

***Example train_config.json:** provides the configuration parameters for the runners. 

The `train_config.json` provides the configuration parameters that setup the training runners.

```json
{
    // The independent seeds to train on - in series or parallel
    "seeds": [10,20,30,40,50], 
    // The number of seeds to run in parallel at a given time
    "max_workers": 3,

    // The number of training steps between evaluations
    "number_steps_per_evaluation": 10000, 
    // The number of episodes to run per evaluation
    "number_eval_episodes": 10, 
    // Whether to record a video of the first evaluation episode
    "record_eval_video": 1, 

    // Whether to record training checkpoints in order to resume training
    "save_train_checkpoints": 1,
    // How frequently to save checkpoint data (in episodes)
    "checkpoint_interval": 1, 

    // Whether to display the environment during training
    "display": 0 
}
```

!!! tip "Training Configuration"
    The training configuration is found in the [Algorithm Configurations file][config-file]

***Example env_config.json:** provides the configuration parameters for the training envrionment.

The `env_config.json` can  be used to configure any parameters that are exposed by the gym environment - e.g. change reward functions, state space settings, or other internal modes. The standard parameters for all gym envrionments configure the image observation if it is required by an Algorithm. 

```json
{
    // The gym environemnt to load
    "gym": "dmcs",
    
    // DMCS specific parameter to set the task domain
    "domain": "ball_in_cup", 
    // The name of the task to train on
    "task": "catch",  
    
    // Image observation parameters
    "frames_to_stack": 3, 
    "frame_width": 84, "frame_height": 84, "grey_scale": 0, 

    // fps controls the output for the evaluation recording
    "record_video_fps": 30,
}
```

!!! tip "Environment Configurations"
    All Gym configurations can be found under the [Envrionment configuration file][envs-config].

***Example alg_config.json:** provides the configuration parameters for the algroithm. 

The `alg_config.json` can be used to override the desired algoithm parameters and the architecture of its networks through the [MLPConfig][config-file] interface. For full instructions on how to configure the `MLPConfig` please read the instructions under [MLP Configuration](./mlp_configuration.md).  

```json
{
  // Configuration for SAC
  "algorithm": "SAC",

  // Parameter Overrides for SAC
  "actor_lr": 0.0003,
  "critic_lr": 0.0003,
  "alpha_lr": 0.0003,

  "tau": 0.005,
  "reward_scale": 1.0,
  "log_std_bounds": [-20.0, 2.0],
  
  // Modifications to the networks can be placed here
  // using the custom MLPConfig definitions
  
  // Actor configuration for SAC here we define the equivalent of:
  // actor = nn.Sequential(
  //         nn.Linear(observation_size, 256),
  //         nn.ReLU(),
  //         nn.Linear(256, 256),
  //         nn.ReLU(),
  //         nn.Linear(256, num_actions),
  //     )
  "actor_config": {
    "layers": [
      {
        "layer_category": "trainable",
        "layer_type": "Linear",
        "out_features": 256,
        "params": {}
      },
      {
        "layer_category": "function",
        "layer_type": "ReLU",
        "params": {}
      },
      {
        "layer_category": "trainable",
        "layer_type": "Linear",
        "in_features": 256,
        "out_features": 256,
        "params": {}
      },
      {
        "layer_category": "function",
        "layer_type": "ReLU",
        "params": {}
      }
    ]
  },

  // Critic configuration for SAC here we define the equivalent of:
  // critic = nn.Sequential(
  //         nn.Linear(observation_size, 256),
  //         nn.ReLU(),
  //         nn.Linear(256, 256),
  //         nn.ReLU(),
  //         nn.Linear(256, num_actions),
  //     )

  "critic_config": {
    "layers": [
      {
        "layer_category": "trainable",
        "layer_type": "Linear",
        "out_features": 256,
        "params": {}
      },
      {
        "layer_category": "function",
        "layer_type": "ReLU",
        "params": {}
      },
      {
        "layer_category": "trainable",
        "layer_type": "Linear",
        "in_features": 256,
        "out_features": 256,
        "params": {}
      },
      {
        "layer_category": "function",
        "layer_type": "ReLU",
        "params": {}
      },
      {
        "layer_category": "trainable",
        "layer_type": "Linear",
        "in_features": 256,
        "out_features": 1,
        "params": {}
      }
    ]
  }
}
```

!!! tip "Algorithm Configurations"
    The algorithm configurations are found in the [Algorithm Configurations file][config-file]

### Resuming Training (resume)

The `resume` command allows you to continue training from a previously saved checkpoint. This is useful if training was interrupted, or if you want to further improve a model without starting from scratch. The resume function loads all configuration and model state from the specified training directory. For resuming to work, you must have enabled checkpoint saving during the original training run (e.g., `--save_train_checkpoints 1`) - the defualt setting is `0` as it increases the log storage significantly.

!!! tip "Extending Training"
    You can adjust the number of additional steps or other parameters in the config files before resuming if desired - for example increase the max training steps in the configuration files to extend the training on a previously finished experiment.

**Example: Resume training from a checkpoint**

Training will continue from the last saved checkpoint - `--save_train_checkpoints 1` must have been set on the experiment for this to work.

```bash
cares-rl resume --data_path my_experiment/
```

!!! warning "Resume is not Deterministic"
    The resume command is not guaranteed to be fully deterministic, but it will restore the model, optimizer, and training state as closely as possible.

## Evaluating Models (evaluate)
The `evaluate` command is used to re-run the evaluation phase of a completed or in-progress training run. This is useful for generating updated evaluation metrics, plots, or logs without re-running the entire training process. Evaluation uses the saved model evaluation checkpoints and configuration files from a previous run.

**Example: Evaluate a trained model**
```bash
cares-rl evaluate --data_path <PATH_TO_EXPERIMENT>
```

**What evaluation does:**

- Runs the evaluation loop as defined in your configuration (e.g., number of episodes, evaluation seeds, metrics).
- Produces updated evaluation logs, plots, and summary statistics in the output directory.
- Does not modify or continue training—evaluation is read-only and safe to run multiple times.

!!! tip "Post Training Evaluations"
    Use evaluation after training or resuming to generate consistent metrics and plots for comparison or publication.

    You can adjust evaluation parameters in the config files before running evaluation if you want to change the number of episodes or other settings.

## Test Models (test)

The `test` command is used to evaluate the final trained model on a new evaluation seed and for a specified number of episodes. This is essential for assessing the generalization and robustness of your agent, as it tests the model on data it has not seen during training or evaluation.

**Example: Test a trained model**
```bash
cares-rl test --data_path <PATH_TO_TRAINING_DATA> --episodes <NUM_EPISODES_TO_RUN> --eval_seed <SEED>
```

**What testing does:**

- Loads the final checkpoint for each training seed from the experiment directory.
- Runs the agent for the specified number of episodes using the provided evaluation seed (ensuring reproducibility and fair comparison).
- Produces test logs, summary statistics, and plots in the output directory.
- Does not alter the model or training state — testing is read-only and safe to repeat.

!!! tip "Best Practices for Testing"
    Always use a different `--eval_seed` for testing than for training or evaluation to avoid overfitting to a particular random seed.
  
    Run multiple test seeds and average results for robust performance estimates.
  
    Use the same number of episodes for each test run to ensure fair comparison.

!!! note "Testing vs. Evaluation"
    **Evaluation** runs on all checkpoints (e.g., for plotting learning curves or tracking progress during training) on the original training seed.
  
    **Testing** runs only on the final model, with a new evaluation seed, to measure generalization and final performance.

---8<-- "include/links.md"