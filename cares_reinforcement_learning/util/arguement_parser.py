"""
Example of using sub-parser, sub-commands and sub-sub-commands :-)
"""

import argparse

def environment_args(parent_parser):
    env_parser = argparse.ArgumentParser()
    env_parsers = env_parser.add_subparsers(help='sub-command help', dest='gym_environment', required=True)

    # create the parser for the DMCS sub-command
    parser_dmcs = env_parsers.add_parser('dmcs', help='DMCS', parents=[parent_parser])
    required = parser_dmcs.add_argument_group('required arguments')
    required.add_argument('--domain', type=str, required=True)
    required.add_argument('--task', type=str, required=True)
    
    # create the parser for the OpenAI sub-command
    parser_openai = env_parsers.add_parser('openai', help='openai', parents=[parent_parser])
    required = parser_openai.add_argument_group('required arguments')
    required.add_argument('--task', type=str, required=True)
    return env_parser

def algorithm_args(parent_parser):
    alg_parser = argparse.ArgumentParser(add_help=False)
    alg_parsers = alg_parser.add_subparsers(help='sub-command help', dest='algorithm', required=True)

    # create the parser for TD3 with default parameters
    parser_TD3 = alg_parsers.add_parser('TD3', help='TD3', parents=[parent_parser])
    parser_TD3.add_argument('--actor_lr', type=float, default=1e-4)
    parser_TD3.add_argument('--critic_lr', type=float, default=1e-3)
    parser_TD3.add_argument('--gamma', type=float, default=0.99)
    parser_TD3.add_argument('--tau', type=float, default=0.005)
    
    # create the parser for DDPG with default parameters
    parser_DDPG = alg_parsers.add_parser('DDPG', help='DDPG', parents=[parent_parser])
    parser_DDPG.add_argument('--actor_lr', type=float, default=1e-4)
    parser_DDPG.add_argument('--critic_lr', type=float, default=1e-3)
    parser_DDPG.add_argument('--gamma', type=float, default=0.99)
    parser_DDPG.add_argument('--tau', type=float, default=0.005)

    # create the parser for SAC with default parameters
    parser_SAC = alg_parsers.add_parser('SAC', help='SAC', parents=[parent_parser])
    parser_SAC.add_argument('--actor_lr', type=float, default=1e-4)
    parser_SAC.add_argument('--critic_lr', type=float, default=1e-3)
    parser_SAC.add_argument('--gamma', type=float, default=0.99)
    parser_SAC.add_argument('--tau', type=float, default=0.005)

    # create the parser for PPO with default parameters
    parser_PPO = alg_parsers.add_parser('PPO', help='SAC', parents=[parent_parser])
    parser_PPO.add_argument('--actor_lr', type=float, default=1e-4)
    parser_PPO.add_argument('--critic_lr', type=float, required=1e-3)
    parser_PPO.add_argument('--gamma', type=float, required=0.99)

    # create the parser for DQN with default parameters
    parser_DQN = alg_parsers.add_parser('DQN', help='DQN', parents=[parent_parser])
    parser_DQN.add_argument('--lr', type=float, default=1e-3)
    parser_DQN.add_argument('--gamma', type=float, required=0.99)

    # create the parser for DuelingDQN with default parameters
    parser_DuelingDQN = alg_parsers.add_parser('DuelingDQN', help='DuelingDQN', parents=[parent_parser])
    parser_DuelingDQN.add_argument('--lr', type=float, default=1e-3)
    parser_DuelingDQN.add_argument('--gamma', type=float, required=0.99)

    # create the parser for DoubleDQN with default parameters
    parser_DoubleDQN = alg_parsers.add_parser('DoubleDQN', help='DoubleDQN', parents=[parent_parser])
    parser_DoubleDQN.add_argument('--lr', type=float, default=1e-3)
    parser_DoubleDQN.add_argument('--gamma', type=float, required=0.99)

    return alg_parser

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)  # Add an argument
    
    parser.add_argument('--memory', type=str, default="MemoryBuffer")
    parser.add_argument('--image_observation', type=bool, default=False)

    parser.add_argument('--G', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--max_steps_exploration', type=int, default=10000)
    parser.add_argument('--max_steps_training', type=int, default=100000)

    parser.add_argument('--number_steps_per_evaluation', type=int, default=10000)
    parser.add_argument('--number_eval_episodes', type=int, default=10)

    parser.add_argument('--seed', type=int, default=571)
    parser.add_argument('--evaluation_seed', type=int, default=152)

    parser.add_argument('--max_steps_per_batch', type=float, default=5000)

    parser.add_argument('--plot_frequency', type=int, default=100)
    parser.add_argument('--checkpoint_frequency', type=int, default=100)

    parser = algorithm_args(parent_parser=parser)
    parser = environment_args(parent_parser=parser)
    
    return vars(parser.parse_args()) # converts to a dictionary
  
if __name__ == '__main__':
    parse_args()
