<img src="https://drive.google.com/uc?export=view&id=1NpXB7lFONA2epIxdZFv5vfCTRRqt8A_9" />
The CARES reinforcement learning bed used as the foundation for RL related projects.

## Motivation
**Reinforcement Learning Algorithms** (that is to say, *how* the Neural Networks are updated) stay the same no matter the application. This package is designed so that these algorithms are only programmed **once** and can be *"plugged & played"* into different environments.

## Usage
Consult the repository [wiki](https://github.com/UoA-CARES/cares_reinforcement_learning/wiki) for a guide on how to use the package

## Installation Instructions
`git clone` the repository

If you would like to leverage your machine's GPU, uncomment the optional dependencies in the `requirements.txt` before moving on.

Run `pip install -r requirements.txt` in the **root directory** of the package

To make the module **globally accessible** in your environment run `python3 setup.py install` in the **project root**
If problems arise, try `python3 setup.py install --user`

## Package Structure

```
cares_reinforcement_learning/
├─ algorithm/
│  ├─ DQN.py
│  ├─ DDPG.py
│  ├─ ...
├─ networks/
│  ├─ DQN.py
│  ├─ DDPG.py
│  ├─ ...
├─ util/
   ├─ MemoryBuffer.py
   ├─ PlotingUtil.py
   ├─ ...
```
`Algorithms/`: contains the code that is responsible for housing and updating the NN according to RL algorithms

`Networks/`: contains....

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
 



