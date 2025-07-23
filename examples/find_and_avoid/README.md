# Find target & avoid obstacles

:warning: NOTE: This agent do not converge well

This is a typical find target and avoid obstacles task with a simple world
configuration. For this task the E-puck robot is used, which is a compact mobile
robot developed by GCtronic and EPFL and is included in Webots. The world
configuration contains an obstacle and a target ball. The E-puck robot uses 8 IR
proximity distance sensors and it has two motors for moving. The agent, apart
from the distance sensor values, also receives the Euclidean distance and angle
from the target. Consequently, the observation the agent gets is an
one-dimensional vector with 10 values. On the other hand, the actuators are
motors, which means that the outputs of the agent are two values controlling the
forward/backward movement and left/right turning respectively (referred to as
gas and wheel).

## Agents 
    
+ Deep Deterministic Policy Gradience (DDPG)
  + Current implementation based on [Phil Tabor](https://github.com/philtabor)
    [implementation](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/DDPG)
    which is presented in [Youtube
    video](https://www.youtube.com/watch?v=6Yd5WnYls_Y).
 
|Trained Agent Showcase|Reward Per Episode Plot|
|----------------------|-----------------------|
|![image](https://github.com/KelvinYang0320/deepworlds/blob/dev/examples/find_and_avoid/doc/demo.gif)|![image](https://github.com/KelvinYang0320/deepworlds/blob/dev/examples/find_and_avoid/doc/trend.png)|
    
