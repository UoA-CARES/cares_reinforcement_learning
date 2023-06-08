from cares_reinforcement_learning.util import Logger
from cares_reinforcement_learning. networks.TD3 import Actor

actor = Actor(3, 2, 0.02)

# Put all our Neural Networks into a dictionary with the name we would like to call them
networks = {'actor': actor}

# Create an instance of Logger: This will create the necessary directories for logging
logger = Logger(networks=networks, checkpoint_freq=5)

# Simulating the Steps
for _ in range(10):
    logger.log(
        S=1000,
        R=100.0,
        AL=-23
    )

# Simulating end of Episode Logging
# Note: if we reuse R for episode reward we will concatenate the R to the same R as above
# Recommended to create Episode and Step Rewards (ER and R)
logger.log(
    E=100,
    ER=200
)

# Simulating end of training final save
logger.save()
