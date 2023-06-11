from cares_reinforcement_learning.util import Record 
from cares_reinforcement_learning. networks.TD3 import Actor
import numpy as np

actor = Actor(3, 2, 0.02)

# Put all our Neural Networks into a dictionary with the name we would like to call them
networks = {'actor': actor}

# Create an instance of Logger: This will create the necessary directories for logging
record = Record(networks=networks, config={'hello': 'person'})

# Simulating the Steps
for i in range(200):
    record.log(
        S=i,
        R=np.random.randint(100),
        AL=np.random.randint(-20, 0),
    )

# Simulating end of Episode Logging
# Note: if we reuse R for episode reward we will concatenate the R to the same R as above
# Recommended to create Episode and Step Rewards (ER and R)
record.log(
    E=100,
    ER=200
)

# Simulating end of training final save
record.save()
