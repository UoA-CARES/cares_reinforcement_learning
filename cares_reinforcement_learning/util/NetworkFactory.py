
def create_DQN(args):
  from cares_reinforcement_learning.algorithm import DQN
  from cares_reinforcement_learning.networks.DQN import Network
  
  network = Network(args["observation_size"], args["action_num"], args["lr"])

  agent = DQN(
      network=network,
      gamma=args["gamma"],
      device=args["device"]
  )
  return agent

def create_DuelingDQN(args):
  from cares_reinforcement_learning.algorithm import DQN
  from cares_reinforcement_learning.networks.DuelingDQN import DuelingNetwork
  
  network = DuelingNetwork(args["observation_size"], args["action_num"], args["lr"])

  agent = DQN(
      network=network,
      gamma=args["gamma"],
      device=args["device"]
  )
  return agent

def create_DDQN(args):
  from cares_reinforcement_learning.algorithm import DoubleDQN
  from cares_reinforcement_learning.networks.DoubleDQN import Network

  network = Network(args["observation_size"], args["action_num"], args["lr"])

  agent = DoubleDQN(
      network=network,
      gamma=args["gamma"],
      tau=args["tau"],
      device=args["device"]
  )
  return agent

def create_PPO(args):
  from cares_reinforcement_learning.algorithm import PPO
  from cares_reinforcement_learning.networks.PPO import Actor
  from cares_reinforcement_learning.networks.PPO import Critic

  actor  = Actor(args["observation_size"], args["action_num"], args["actor_lr"])
  critic = Critic(args["observation_size"], args["critic_lr"])

  agent = PPO(
      actor_network=actor,
      critic_network=critic,
      gamma=args["gamma"],
      action_num=args["action_num"],
      device=args["device"]
  )
  return agent

def create_SAC(args):
  from cares_reinforcement_learning.algorithm import SAC
  from cares_reinforcement_learning.networks.SAC import Actor
  from cares_reinforcement_learning.networks.SAC import Critic

  actor  = Actor(args["observation_size"], args["action_num"], args["actor_lr"])
  critic = Critic(args["observation_size"], args["action_num"], args["critic_lr"])

  agent = SAC(
      actor_network=actor,
      critic_network=critic,
      gamma=args["gamma"],
      tau=args["tau"],
      action_num=args["action_num"],
      device=args["device"],
  )
  return agent

def create_DDPG(args):
  from cares_reinforcement_learning.algorithm import DDPG
  from cares_reinforcement_learning.networks.DDPG import Actor
  from cares_reinforcement_learning.networks.DDPG import Critic

  actor  = Actor(args["observation_size"], args["action_num"], args["actor_lr"])
  critic = Critic(args["observation_size"], args["action_num"], args["critic_lr"])

  agent = DDPG(
      actor_network=actor,
      critic_network=critic,
      gamma=args["gamma"],
      tau=args["tau"],
      action_num=args["action_num"],
      device=args["device"],
  )
  return agent

def create_TD3(args):
    from cares_reinforcement_learning.algorithm import TD3
    from cares_reinforcement_learning.networks.TD3 import Actor
    from cares_reinforcement_learning.networks.TD3 import Critic
    
    actor  = Actor(args["observation_size"], args["action_num"], args["actor_lr"])
    critic = Critic(args["observation_size"], args["action_num"], args["critic_lr"])

    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=args["gamma"],
        tau=args["tau"],
        action_num=args["action_num"],
        device=args["device"],
    )
    return agent

class NetworkFactory:
  def create_network(self, algorithm, args):
    if algorithm == "DQN":
      return create_DQN(args)
    elif algorithm == "DDQN":
      return create_DDQN(args)
    elif algorithm == "DuelingDQN":
      return create_DuelingDQN(args)
    elif algorithm == "PPO":
      return create_PPO(args)
    elif algorithm == "DDPG":
      return create_DDPG(args)
    elif algorithm == "SAC":
      return create_DDPG(args)
    elif algorithm == "TD3":
      return create_TD3(args)
    raise ValueError(f"Unkown algorithm: {algorithm}")