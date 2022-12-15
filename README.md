# CARES Reinforcement Learning Package
The cares reinforcement learning bed used as the foundation for RL related projects.


## Usage
Consult the repository [wiki](https://github.com/UoA-CARES/summer_reinforcement_learning/wiki) for a guide on how to use the package

## Installation Instructions
`git clone` the repository

Run `pip install -r requirements.txt` in the **root directory** of the package

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

