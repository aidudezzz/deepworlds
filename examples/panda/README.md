# Franka Emika Panda with Deepbots and Reinforcement Learning
> [deepbots](https://github.com/aidudezzz/deepbots)\
> [Franka Emika Panda](https://www.franka.de/technology)

The Panda robot model was made from [these files](https://github.com/mkrizmancic/franka_gazebo/tree/master/meshes).\
We are trying to solve some interesting problems with reinforcement learning and finally deploy the models in the real world.

## Goal reaching with a 7-DoF Panda robotic arm
The goal is to train an agent to reach the randomly selected goal within limited steps.\
Here, the problem is solved with the [Deep Deterministic Policy Gradient RL algorithm](https://arxiv.org/abs/1509.02971). The agent observes its seven motor positions and the Cartesian coordinates of the selected goal, and then controls the seven motor positions. 
|Trained Agent Showcase|Reward Per Episode Plot|
|----------------------|-----------------------|
|![image](https://github.com/KelvinYang0320/deepworlds/blob/dev/examples/panda/doc/demo.gif)|![image](https://github.com/KelvinYang0320/deepworlds/blob/dev/examples/panda/doc/trend.png)|

If you want to solve this task with [IKPY](https://pypi.org/project/ikpy/), you can switch to [IKPY controller](./panda_goal_reaching/controllers/IKPY/).

## Acknowledgments
This project is part of the System Integration Implementation teamwork, an undergraduate course supervised by [Prof. Chih-Tsun Huang](http://www.cs.nthu.edu.tw/~cthuang) of [Dept. of Computer Science, National Tsing Hua University](http://dcs.site.nthu.edu.tw/).\
\
The Panda robot model was contributed by all the team members, [Yung Tai Shih](https://github.com/garystone1), [Tsu Hsiang Chen](https://github.com/Truman-Sean), [Chun Kai Yang](https://github.com/yckai2679), [Yan Feng Su](https://github.com/YenFengSu), [Hsuan Yu Liao](https://github.com/GuluLingpo), and the author of this deepworlds Panda example, [Jiun Kai Yang](https://github.com/KelvinYang0320).
