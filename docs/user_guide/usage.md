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
    For quick experiments, use CLI flags to change simple configuration parameters to key parameters and the defaults for the rest.
    
    For reproducibility, use the configuration files which allow full control over all parameters for consistency between runs and allow you to configure an aglorithm's network architecture. 

### Command Line Input (cli)
The `cli` (command line input) mode lets you specify all training, environment, and algorithm parameters directly as command-line flags. This is ideal for quick experiments, prototyping, or when you want to override just a few settings without creating configuration files. 

!!! warning "cli limitations"
    The network architectures for a given algorithm can't be changed via command line - at least not in a nice convenient style. Use the `config` command for that instead.  

**Gym Envrionment**: Choose the Gym Envrionment and Task
The first piece is choosing which gym envrionment and task you are seeking to train the algorithm against. 

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

### Configuration Files (config)

The `config` mode allows you to specify all experiment parameters in structured JSON configuration files, rather than on the command line. This approach is ideal for reproducible research, large-scale experiments, or when you want to precisely control every aspect of your environment, algorithm, and training setup.

All parameters, including advanced options like network architectures, optimizer settings, and environment details, can be set in the config files. The CLI will load these files and use them for the entire run, ensuring consistency and repeatability.

**Data Path structure:** The command will read the three files from the provided path: 
```
~/my_experiment/
  env_config.json
  train_config.json
  alg_config.json
```

***Example env_config.json:** provides the configuration parameters for the training envrionment.

The `env_config.json` can  be used to configure any parameters that are exposed by the gym environment - e.g. change reward functions, state space settings, or other internal modes.

```json
{
    "gym": "dmcs",
    "domain": "ball_in_cup", 
    "task": "catch", 
    "display": 0, 
    "save_train_checkpoints": 0, 
    "state_std": 0.0, 
    "action_std": 0.0, 
    "frames_to_stack": 3, 
    "frame_width": 84, "frame_height": 84, "grey_scale": 0, 
    "record_video_fps": 30
}
```

***Example train_config.json:** provides the configuration parameters for the runners. 

```json
{
    "seeds": [10], 
    "number_steps_per_evaluation": 10000, 
    "number_eval_episodes": 10, 
    "record_eval_video": 1, 
    "checkpoint_interval": 1, 
    "max_workers": 1
}
```

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

**Example: Running with config files**

The command below runs the training as defined by the configuration files detailed above. 
```bash
cares-rl train config --data_path ~/my_experiment/
```

!!! tip "Generating Configuration Files"
    You can generate or edit these JSON files manually, or use a previous run's output as a template for new experiments.

!!! tip "Defaults Override Unspecificed Parameters"
    Not all parameters need to be configured in the config files - defaults will be used where parameters are left out. 

    The custom recording tool will auotmatically fill in the default values in the log outputs too. 

## Evaluating Models (evaluate)
Evaluate a run (reproduce evaluation metrics/plots):
```bash
cares-rl evaluate --data_path <PATH_TO_TRAINING_DATA>
```

## Test Models (test)
Test a trained model:
```bash
cares-rl test --data_path <PATH_TO_TRAINING_DATA> --episodes 10 --eval_seed SEED
```

## Resuming Training (resume)
Resume from a checkpoint (if `--save_train_checkpoints 1` was enabled):
```bash
cares-rl resume --data_path <PATH_TO_TRAINING_DATA>
```

--8<-- "include/links.md"