<img src="https://drive.google.com/uc?export=view&id=1NpXB7lFONA2epIxdZFv5vfCTRRqt8A_9" />
The CARES reinforcement learning bed used as the foundation for RL related projects.

## Motivation
**Reinforcement Learning Algorithms** (that is to say, *how* the Neural Networks are updated) stay the same no matter the application. This package is designed so that these algorithms are only programmed **once** and can be *"plugged & played"* into different environments.

## Usage
Consult the repository [wiki](https://github.com/UoA-CARES/cares_reinforcement_learning/wiki) for a guide on how to use the package

## Installation Instructions
`git clone` the repository

If you would like to leverage your machine's GPU, uncomment the optional dependencies in the `requirements.txt` before moving on.

Run `pip3 install -r requirements.txt` in the **root directory** of the package

To make the module **globally accessible** in your working environment run `pip3 install --editable .` in the **project root**

## Running an Example
This repository includes a script that allows you to run any OpenAI environment – provided you comply with all the dependencies for that environment. These examples make use of the package, and can provide an example on how one might use the package in their own environments.

`example_training_loops.py` takes in hyperparameters that allow you to customise the training run – OpenAI Environment, training steps, gamma... Use `python3 example_training_loops.py -h` for help on what hyperparameters are available for customisation.

An example is found below:
```
python3 example_training_loops.py --task 'Pendulum-v1' --algorithm PPO --max_steps_training 1000000 --seed 571 --gamma 0.99 --actor_lr 0.0001 --critic_lr 0.001
```


## Package Structure

```
cares_reinforcement_learning/
├─ algorithm/
│  ├─ DQN.py
│  ├─ TD3.py
│  ├─ PPO.py
│  ├─ ...
├─ networks/
│  ├─ DQN/
│  │  ├─ Network.py
│  ├─ TD3.py/
│  │  ├─ Actor.py
│  │  ├─ Critic.py
│  │  ├─ ...
├─ util/
│  ├─ MemoryBuffer.py
│  ├─ Plot.py
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
| DDPG   | Continuous        | Off Policy |
| TD3   | Continuous        | Off Policy |
| SAC   | Continuous        | Off Policy       |
| PPO      | Continuous       | On Policy       | 

## In progress
| Algorithm      | Action Space |  On/Off Policy |
| ----------- | ----------- | ----------- |
| D4PG   | Continuous        | Off Policy       |
 



