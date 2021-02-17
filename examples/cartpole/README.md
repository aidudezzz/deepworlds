# CartPole

This example is a recreation of the classic [CartPole](https://gym.openai.com/envs/CartPole-v0/)
problem in [Webots](https://cyberbotics.com) as seen in [OpenAI's gym](https://gym.openai.com/).

Two solutions of the problem are provided, one with discrete action space and one with continuous action space.
The discrete one is solved with the 
[Proximal Policy Optimization Reinforcement (PPO) Learning (RL) algorithm](https://openai.com/blog/openai-baselines-ppo/).
The continuous one is solved with the 
[Deep Deterministic Policy Gradient RL algorithm](https://arxiv.org/abs/1509.02971).
Both use [PyTorch](https://pytorch.org/) as their backend neural network library.

Switching between discrete and continuous action spaces requires several modifications on both the robot and the 
supervisor controllers.

More CartPole examples can be added in the future, using other algorithms/backends.
Both examples utilize the [emitter - receiver scheme](https://github.com/aidudezzz/deepbots#emitter---receiver-scheme),
with plans of providing the same examples using the 
[robot - supervisor scheme](https://github.com/aidudezzz/deepbots#combined-robot-supervisor-scheme) in the future.

## Contents
- [CartPole discrete](https://github.com/tsampazk/deepworlds/tree/readme-fixes/examples/cartpole/cartpole_discrete)
- [CartPole continuous](https://github.com/tsampazk/deepworlds/tree/readme-fixes/examples/cartpole/cartpole_continous)


## Solved CartPole - PPO discrete

Trained agent showcase:

TODO fix link to gif
<p align="center">
    <img src="https://raw.githubusercontent.com/tsampazk/deepworlds/fixes-updates/examples/cartpole/doc/gif/cartpoleSolved.gif">
</p>

Reward per episode plot:

TODO fix link to plot
<p align="center">
    <img src="https://raw.githubusercontent.com/tsampazk/deepworlds/fixes-updates/examples/cartpole/doc/img/rewardPlot.png">
</p>
