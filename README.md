![CARES reinforcement learning package logo](./media/logo.png)
The CARES reinforcement learning bed used as the foundation for RL related projects.

## Motivation
**Reinforcement Learning Algorithms** (that is to say, *how* the Neural Networks are updated) stay the same no matter the application. This package is designed so that these algorithms are only programmed **once** and can be *"plugged & played"* into different environments.

## Usage
Consult the repository [wiki](https://github.com/UoA-CARES/cares_reinforcement_learning/wiki) for a guide on how to use the package

## Installation Instructions
If you want to utilise the GPU with Pytorch install CUDA first - https://developer.nvidia.com/cuda-toolkit

Install Pytorch following the instructions here - https://pytorch.org/get-started/locally/

`git clone` the repository into your desired directory on your local machine

Run `pip3 install -r requirements.txt` in the **root directory** of the package

To make the module **globally accessible** in your working environment run `pip3 install --editable .` in the **project root**

## Running an Example
This package serves as a library of specific RL algorithms and utility functions being used by the CARES RL team. For an example of how to use this package in your own envrionments see this package which uses these algorithms on the Deep Mind Control suite and OpenAI gym envrionments - https://github.com/UoA-CARES/gymnasium_envrionments 

## Utilities
CARES RL provides a number of useful utility functions and classes for generating consistent results across the team. These utilities should be utilised in the new envrionments we build to test our approaches.

### configurations.py
Provides baseline dataclasses for environment, training, and algorithm configurations to allow for consistent recording of training parameters. 

### RLParser.py
Provides a means of loading enevironment, training, and algorithm configurations through command line or configuration files. Enables consistent tracking of parameters when running training on various algorithms.

### Record.py
The Record class allows data to be saved into a consistent format during training. This allows all data to be consistently formatted for plotting against each other for fair and consistent evaluation.

All data from a training run is saved into '~/cares_rl_logs'. A folder will be created for each training run named as 'seed/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS', e.g. '10/TD3-HalfCheetah-v4-23_10_11_08:47:22'. This folder will contain the following directories and information saved during the training session:

```
ALGORITHM-TASK-YY_MM_DD:HH:MM:SS/
├─ SEED
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
|  |  ├─ ...
├─ SEED...
├─ ...
```

### plotting.py
The plotting utility will plot the data contained in the training data. An example of how to plot the data from one or multiple training sessions together is shown below. Running 'python3 plotter.py -h' will provide details on the plotting parameters.

```
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS
```

### NetworkFactory.py
A factory class for creating a baseline RL algorithm that has been implemented into the CARES RL package. 

### MemoryFactory.py
A factory class for creating a memory buffer (prioritised or not) that has been implemented into the CARES RL package.

## Package Structure

```
cares_reinforcement_learning/
├─ algorithm/
natsort  ├─ policy/
│  │  ├─ TD3.py
│  │  ├─ ...
│  ├─ value/
│  │  ├─ DQN.py
│  │  ├─ ...
├─ networks/
│  ├─ DQN/
│  │  ├─ Network.py
│  ├─ TD3.py/
│  │  ├─ Actor.py
│  │  ├─ Critic.py
│  ├─ ...
├─ memory/
│  ├─ MemoryBuffer.py
├─ util/
│  ├─ NetworkFactory.py
│  ├─ ...

```
`algorithm`: contains update mechanisms for neural networks as defined by the algorithm.

`networks`: contains standard neural networks that can be used with each algortihm

`util/`: contains common utility classes

## Supported Algorithms
| Algorithm      | Action Space |
| ----------- | ----------- |
| DQN      | Discrete              | 
| DoubleDQN   | Discrete         |
| DuelingDQN   | Discrete         |
| DDPG   | Continuous         |
| PPO      | Continuous       |
| TD3   | Continuous         |
| PALTD3   | Continuous               |
| PERTD3   | Continuous               |
| LAPTD3   | Continuous               |
| LA3PTD3   | Continuous               |
| MAPERTD3   | Continuous               |
| RDTD3   | Continuous               |
| NaSATD3   | Continuous               |
| CTD4   | Continuous               |
| SAC   | Continuous               |
| REDQ   | Continuous               |
| TQC   | Continuous               |

## In progress
| Algorithm      | Action Space |  On/Off Policy |
| ----------- | ----------- | ----------- |
| D4PG   | Continuous               |
 



