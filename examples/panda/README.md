# Franka Emika Panda with Deepbots and Reinforcement Learning
> [deepbots](https://github.com/aidudezzz/deepbots)\
> [Franka Emika Panda](https://www.franka.de/technology)

The Panda robot model was made from [these files](https://github.com/mkrizmancic/franka_gazebo/tree/master/meshes).\
We are trying to solve some interesting problems with reinforcement learning and finally deploy the models in the real world.

## Installation
1. [Install Webots R2021a](https://www.cyberbotics.com/)
2. Install Python versions 3.8
    * Follow the Using Python guide provided by Webots
3. Install deepbots 0.1.3.dev2 through pip running the following command:\
<code>pip install -i https://test.pypi.org/simple/ deepbots</code>
4. Install PyTorch via pip
* If you want to solve this task with IKPY, please [install IKPY](https://pypi.org/project/ikpy/) and switch to IKPY controller.

## Goal reaching with a 7-DoF Panda robotic arm
The goal is to train an agent to reach the randomly selected goal within limited steps.\
Here, the problem is solved with the [Deep Deterministic Policy Gradient RL algorithm](https://arxiv.org/abs/1509.02971). The agent observes its seven motor positions and the Cartesian coordinates of the selected goal, and then controls the seven motor positions. 
|Trained Agent Showcase|Reward Per Episode Plot|
|----------------------|-----------------------|
|![image](https://github.com/KelvinYang0320/deepworlds/blob/dev/examples/panda/doc/demo.gif)|![image](https://github.com/KelvinYang0320/deepworlds/blob/dev/examples/panda/doc/trend.png)|



## Acknowledgments
This project is part of the System Integration Implementation teamwork, an undergraduate course supervised by [Prof. Chih-Tsun Huang](http://www.cs.nthu.edu.tw/~cthuang) of [Dept. of Computer Science, National Tsing Hua University](http://dcs.site.nthu.edu.tw/).\
\
The Panda robot model was contributed by all the team members, [Yung Tai Shih](https://github.com/garystone1), [Tsu Hsiang Chen](https://github.com/Truman-Sean), [Chun Kai Yang](https://github.com/yckai2679), [Yan Feng Su](https://github.com/YenFengSu), [Hsuan Yu Liao](https://github.com/GuluLingpo), and the author of this deepworlds Panda example, [Jiun Kai Yang](https://github.com/KelvinYang0320).
