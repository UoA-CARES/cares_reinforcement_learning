#!/bin/bash 

shopt -s expand_aliases

alias dqn='echo test dqn'
alias ppo='echo test ppo'
alias td3='echo test td3'
# alias dqn='python3 example/example_training_loops.py --task "CartPole-v1"    --algorithm DQN --max_steps_training 20000 --max_steps_evaluation 0'
# alias ppo='python3 example/example_training_loops.py --task "HalfCheetah-v4" --algorithm PPO --max_steps_training 50000 --max_steps_evaluation 0'
# alias td3='python3 example/example_training_loops.py --task "HalfCheetah-v4" --algorithm TD3 --max_steps_training 50000 --max_steps_evaluation 0'

(trap 'kill 0' SIGINT; dqn & ppo & td3 & wait)



