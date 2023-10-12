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
This repository includes a script that allows you to run any OpenAI Gymnasium (https://github.com/Farama-Foundation/Gymnasium) or Deep Mind Control Suite (https://github.com/google-deepmind/dm_control) environment – provided you comply with all the dependencies for that environment. These examples make use of the package, and can provide an example on how one might use the package in their own environments.

`example_training_loops.py` takes in hyperparameters that allow you to customise the training run enviromment – OpenAI or DMCS Environment - or RL algorithm. Use `python3 example_training_loops.py -h` for help on what parameters are available for customisation.

An example is found below for running on the OpenAI and DMCS environments with TD3:
```
python3 example_training_loops.py openai --task HalfCheetah-v4 TD3


python3 example_training_loops.py dmcs --domain ball_in_cup --task catch TD3
```

### Data Outputs
All data from a training run is saved into '~/cares_rl_logs'. A folder will be created for each training run named as 'ALGORITHM-TASK-YY_MM_DD:HH:MM:SS', e.g. 'TD3-HalfCheetah-v4-23_10_11_08:47:22'. This folder will contain the following directories and information saved during the training session:

```
ALGORITHM-TASK-YY_MM_DD:HH:MM:SS/
├─ SEED
|  ├─ config.py
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

### Plotting
The plotting utility will plot the data contained in the training data. An example of how to plot the data from one or multiple training sessions together is shown below. Running 'python3 plotter.py -h' will provide details on the plotting parameters.

```
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS
```

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
| Algorithm      | Action Space | On/Off Policy |
| ----------- | ----------- |----------- |
| DQN      | Discrete       | Off Policy       | 
| DoubleDQN   | Discrete        | Off Policy |
| DuelingDQN   | Discrete        | Off Policy |
| DDPG   | Continuous        | Off Policy |
| TD3   | Continuous        | Off Policy |
| SAC   | Continuous        | Off Policy       |
| PPO      | Continuous       | On Policy       | 

## In progress
| Algorithm      | Action Space |  On/Off Policy |
| ----------- | ----------- | ----------- |
| D4PG   | Continuous        | Off Policy       |
 



