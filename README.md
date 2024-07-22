<img src="./media/logo.png" alt="CARES reinforcement learning package logo" />

The CARES reinforcement learning bed used as the foundation for RL related projects.

# Motivation

**Reinforcement Learning Algorithms** (that is to say, *how* the Neural Networks are updated) stay the same no matter the application. This package is designed so that these algorithms are only programmed **once** and can be *"plugged & played"* into different environments.

# Usage

Consult the repository [wiki](https://github.com/UoA-CARES/cares_reinforcement_learning/wiki) for a guide on how to use the package

# Installation Instructions

If you want to utilise the GPU with Pytorch install CUDA first - https://developer.nvidia.com/cuda-toolkit

Install Pytorch following the instructions here - https://pytorch.org/get-started/locally/

`git clone` the repository into your desired directory on your local machine

Run `pip3 install -r requirements.txt` in the **root directory** of the package

To make the module **globally accessible** in your working environment run `pip3 install --editable .` in the **project root**

# Running an Example

This package serves as a library of specific RL algorithms and utility functions being used by the CARES RL team. For an example of how to use this package in your own environments see the example gym packages below that use these algorithms for training agents on a variety of simulated and real-world tasks.

## Gym Environments

We have created a standardised general purpose gym that wraps the most common simulated environments used in reinforcement learning into a single easy to use place:  https://github.com/UoA-CARES/gymnasium_envrionments

This package contains wrappers for the following gym environments:

### Deep Mind Control Suite

The standard Deep Mind Control suite: https://github.com/google-deepmind/dm_control

<p align="center">
    <img alt="deep mind control suite" src="./media/dmcs.png" style="width: 80%;"/>
</p>

### OpenAI Gymnasium

The standard OpenAI Gymnasium: https://github.com/Farama-Foundation/Gymnasium

<p align="center">
    <img alt="open ai" src="./media/openai.jpg" style="width: 80%;" />
</p>

### Game Boy Emulator

Environment running Gameboy games utilising the pyboy wrapper: https://github.com/UoA-CARES/pyboy_environment

<p align="center">
    <img alt="game boy mario" src="./media/mario.png" style="width: 40%;" />
    <img alt="game boy pokemon" src="./media/pokemon.png" style="width: 40%;"/>
</p>

## Gripper Gym

The gripper gym contains all the code for training our dexterous robotic manipulators: https://github.com/UoA-CARES/gripper_gym

<p align="center">
    <img alt="rotation task" src="./media/rotation_task-min.jpg" style="width: 40%;"/>
    <img alt="translation task" src="./media/translation_task-min.jpg" style="width: 40%;"/>
</p>

## F1Tenth Autonomous Racing

The Autonomous F1Tenth package contains all the code for training our F1Tenth platforms to autonomously race: https://github.com/UoA-CARES/autonomous_f1tenth

<p align="center">
    <img alt="f one tenth" src="./media/f1tenth-min.png" style="width: 80%;"/>
</p>

# Utilities

CARES RL provides a number of useful utility functions and classes for generating consistent results across the team. These utilities should be utilised in the new environments we build to test our approaches.

## Record.py

The Record class allows data to be saved into a consistent format during training. This allows all data to be consistently formatted for plotting against each other for fair and consistent evaluation.

All data from a training run is saved into the directory specified in the `CARES_LOG_DIR` environment variable. If not specified, this will default to `'~/cares_rl_logs'`.

You may specify a custom log directory format using the `log_path` config option. This path supports variable interpolation such as the algorithm used, seed, date etc. This defaults to `"{algorithm}/{algorithm}-{domain_task}-{date}/{seed}"` so that each run is saved as a new seed under the algorithm and domain-task pair for that algorithm.

The following variables are supported for `log_path` variable interpolation:

- `algorithm`
- `domain`
- `task`
- `domain_task`: The domain and task or just task if domain does not exist
- `gym`
- `seed`
- `date`: The current date in the `YY_MM_DD-HH-MM-SS` format
- `run_name`: The run name if it is provided, otherwise "unnamed"
- `run_name_else_date`: The run name if it is provided, otherwise the date

This folder will contain the following directories and information saved during the training session:

```text
├─ <log_path>
|  ├─ env_config.py
|  ├─ alg_config.py
|  ├─ train_config.py
|  ├─ data
|  |  ├─ train.csv
|  |  ├─ eval.csv
|  ├─ figures
|  |  ├─ eval.png
|  |  ├─ train.png
|  ├─ models
|  |  ├─ model.pht
|  |  ├─ CHECKPOINT_N.pht
|  |  ├─ ...
|  ├─ videos
|  |  ├─ STEP.mp4
├─ ...
```

## plotting.py

The plotting utility will plot the data contained in the training data based on the format created by the Record class. An example of how to plot the data from one or multiple training sessions together is shown below.

Plot the results of a single training instance

```sh
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS
```

Plot and compare the results of two or more training instances

```sh
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM_A/ALGORITHM_A-TASK-YY_MM_DD:HH:MM:SS ~/cares_rl_logs/ALGORITHM_B/ALGORITHM_B-TASK-YY_MM_DD:HH:MM:SS
```

Running 'python3 plotter.py -h' will provide details on the plotting parameters and control arguments.

```sh
python3 plotter.py -h
```

## configurations.py

Provides baseline data classes for environment, training, and algorithm configurations to allow for consistent recording of training parameters.

## RLParser.py

Provides a means of loading environment, training, and algorithm configurations through command line or configuration files. Enables consistent tracking of parameters when running training on various algorithms.

## NetworkFactory.py

A factory class for creating a baseline RL algorithm that has been implemented into the CARES RL package.

## MemoryFactory.py

A factory class for creating a memory buffer that has been implemented into the CARES RL package.

# Package Structure

```text
cares_reinforcement_learning/
├─ algorithm/
├─ policy/
│  │  ├─ TD3.py
│  │  ├─ ...
│  ├─ value/
│  │  ├─ DQN.py
│  │  ├─ ...
├─ networks/
│  ├─ DQN/
│  │  ├─ network.py
│  ├─ TD3.py/
│  │  ├─ actor.py
│  │  ├─ critic.py
│  ├─ ...
├─ memory/
│  ├─ prioritised_replay_buffer.py
├─ util/
│  ├─ network_factory.py
│  ├─ ...
```

`algorithm`: contains update mechanisms for neural networks as defined by the algorithm.

`networks`: contains standard neural networks that can be used with each algorithm

`memory`: contains the implementation of various memory buffers - e.g. Prioritised Experience Replay

`util`: contains common utility classes

# Supported Algorithms

| Algorithm   | Observation Space          | Action Space | Paper Reference |
| ----------- | -------------------------- | ------------ | --------------- |
| DQN         | Vector                     | Discrete     | [DQN Paper](https://arxiv.org/abs/1312.5602) |
| DoubleDQN   | Vector                     | Discrete     | [DoubleDQN Paper](https://arxiv.org/abs/1509.06461) |
| DuelingDQN  | Vector                     | Discrete     | [DuelingDQN Paper](https://arxiv.org/abs/1511.06581) |
| ----------- | -------------------------- | ------------ | --------------- |
| PPO         | Vector                     | Continuous   | [PPO Paper](https://arxiv.org/abs/1707.06347) |
| DDPG        | Vector                     | Continuous   | [DDPG Paper](https://arxiv.org/pdf/1509.02971v5.pdf) |
| TD3         | Vector                     | Continuous   | [TD3 Paper](https://arxiv.org/abs/1802.09477v3) |
| SAC         | Vector                     | Continuous   | [SAC Paper](https://arxiv.org/abs/1812.05905) |
| PERTD3      | Vector                     | Continuous   | [PERTD3 Paper](https://arxiv.org/abs/1511.05952) |
| PERSAC      | Vector                     | Continuous   | [PERSAC Paper](https://arxiv.org/abs/1511.05952) |
| PALTD3      | Vector                     | Continuous   | [PALTD3 Paper](https://arxiv.org/abs/2007.06049) |
| LAPTD3      | Vector                     | Continuous   | [LAPTD3 Paper](https://arxiv.org/abs/2007.06049) |
| LAPSAC      | Vector                     | Continuous   | [LAPSAC Paper](https://arxiv.org/abs/2007.06049) |
| LA3PTD3     | Vector                     | Continuous   | [LA3PTD3 Paper](https://arxiv.org/abs/2209.00532) |
| LA3PSAC     | Vector                     | Continuous   | [LA3PSAC Paper](https://arxiv.org/abs/2209.00532) |
| MAPERTD3    | Vector                     | Continuous   | [MAPERTD3 Paper](https://openreview.net/pdf?id=WuEiafqdy9H) |
| MAPERSAC    | Vector                     | Continuous   | [MAPERSAC Paper](https://openreview.net/pdf?id=WuEiafqdy9H) |
| RDTD3       | Vector                     | Continuous   | WIP |
| RDSAC       | Vector                     | Continuous   | WIP |
| REDQ        | Vector                     | Continuous   | [REDQ Paper](https://arxiv.org/pdf/2101.05982.pdf) |
| TQC         | Vector                     | Continuous   | [TQC Paper](https://arxiv.org/abs/1812.05905) |
| CTD4        | Vector                     | Continuous   | [CTD4 Paper](https://arxiv.org/abs/2405.02576) |
| ----------- | -------------------------- | ------------ | --------------- |
| NaSATD3     | Image                      | Continuous   | In Submission |
| TD3AE       | Image                      | Continuous   | [TD3AE Paper](https://arxiv.org/abs/1910.01741) |
| SACAE       | Image                      | Continuous   | [SACAE Paper](https://arxiv.org/abs/1910.01741) |
