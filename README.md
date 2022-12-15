# CARES Reinforcement Learning Package
The cares reinforcement learning bed used as the foundation for RL related projects.


## Usage
Consult the repository [wiki](https://github.com/UoA-CARES/summer_reinforcement_learning/wiki) for a guide on how to use the package

## Installation Instructions
`git clone` the repository

We recommend creating a `conda` environment to run the package.

To **create** a conda environment with the **necessary dependencies**, run the following in the root of the package:

  ```python3
  conda create --name <env> --file requirements_conda.txt
  ```

## Package Structure

```
reinforcement_learning_summer/
├─ networks/
│  ├─ DQN.py
│  ├─ DDPG.py
│  ├─ ...
├─ util/
   ├─ MemoryBuffer.py
   ├─ PlotingUtil.py
   ├─ ...
```
`networks/`: contains neural network (NN) wrappers that are responsible for housing and updating the NN according to RL algorithms

`util/`: contains common utility classes

