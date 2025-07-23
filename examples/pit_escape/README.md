# Pit Escape

This example is the [Pit Escape](https://robotbenchmark.net/benchmark/pit_escape/) problem included in 
[Webots](https://cyberbotics.com). This problem is an extended 
[MountainCar](https://gym.openai.com/envs/MountainCar-v0/) problem, in 3 dimensions and more complicated 
observation and action spaces.

One solution of the problem is provided, using a discrete action space and utilizing the 
[emitter - receiver scheme](https://github.com/aidudezzz/deepbots#emitter---receiver-scheme).
It is solved with the 
[Proximal Policy Optimization Reinforcement (PPO) Learning (RL) algorithm](https://openai.com/blog/openai-baselines-ppo/).
It uses [PyTorch](https://pytorch.org/) as the backend neural network library.

More Pit Escape examples can be added in the future, using other algorithms/backends and a continuous action space.

## Contents
- [Pit Escape discrete](https://github.com/aidudezzz/deepworlds/tree/dev/examples/pit_escape/pit_escape_discrete)

## Solved PitEscape - PPO discrete

Trained agent showcase:

![image](https://github.com/aidudezzz/deepworlds/blob/dev/examples/pit_escape/doc/gif/pitEscapeSolved.gif)

Reward per episode plot:

![image](https://github.com/aidudezzz/deepworlds/blob/dev/examples/pit_escape/doc/img/rewardPlot.png)
