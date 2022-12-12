# Summer Reinforcement Learning Package
A python package that allows developers to build and train reinforcement learning (RL) models quickly and efficiently.

## Package Dependencies
`cd` into the root folder and run `pip install -r requirements.txt` to install dependencies


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

