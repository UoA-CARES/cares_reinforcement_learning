<img src="https://drive.google.com/uc?export=view&id=1NpXB7lFONA2epIxdZFv5vfCTRRqt8A_9" />
The CARES reinforcement learning bed used as the foundation for RL related projects.


## Usage
Consult the repository [wiki](https://github.com/UoA-CARES/cares_reinforcement_learning/wiki/%F0%9F%8F%A0-Home) for a guide on how to use the package

## Installation Instructions
`git clone` the repository

If you would like to leverage your machine's GPU, uncomment the optional dependencies in the `requirements.txt` before moving on.

Run `pip install -r requirements.txt` in the **root directory** of the package

To make the module **globally accessible** on your environment run `python3 setup.py install`.

## Package Structure

```
cares_reinforcement_learning/
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

