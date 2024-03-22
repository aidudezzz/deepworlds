# Find target & avoid obstacles v2

## Showcase

Trained agent showcase:

![image](./doc/find_avoid_v2_trained.gif)

[Online interactive animation of trained agent](https://webots.cloud/AcNXp_5)

## General
This is the updated and expanded second version of find and avoid.
For this example, a simple custom differential drive robot is used, equipped with
forward-facing distance and touch sensors. The goal is to navigate to a target by modifying
the left and right motor speeds while avoiding obstacles using the distance and touch sensors.
The agent observes its distance to the target, relative facing angle, motor speeds, touch
and distance sensor values, and latest action. The observation can also be augmented with observations 
from earlier steps.

## World
The world consists of an arena that can be populated using procedural generation with various obstacles, 
boxes of various shapes, chairs, and jars. With the help of a grid map, the map can be randomized in two ways. The first one
places the robot and the target at random positions, with a variable distance between them to control difficulty,
and a variable number of obstacles are scattered randomly in the arena. It is made sure that there is 
a free path on the grid map between the robot and the target using a simple BFS path-finding algorithm.
The second way of randomizing the map creates a corridor with several rows and three columns. The robot is
placed on the first row, and the target is placed on one of the other rows, which can be controlled to modify difficulty.
Two obstacles are placed in each of the in-between rows, making sure there is a path between the robot and the target.


## Training
The training happens in a curriculum of increasing difficulty. First, a random map is used with a few scattered 
obstacles and the target placed at a large distance. This way, paths between the robot and the target 
are mostly free of obstacles, providing an easy start. Next, using the corridor randomization, training continues
starting with one row of obstacles and increasing gradually between training sessions to two, three, and so on, 
with more and more obstacles added for each row. 
Finally, the agent is trained on a random map, with all 25 available obstacles placed within and a varying distance 
between the robot and the target.

## Testing
Testing first takes place in random maps set up similar to the last training session using stable-baselines3's 
`evaluate_policy` for 100 episodes (and consequently maps). 

Then a custom evaluation procedure takes place that tests the agent in various corridor setups of increasing 
difficulties and a random map setup for 100 episodes each.

## Logging
Tensorboard is used for logging various aspects of the training procedure. To watch the tensorboard logs, navigate to 
`/deepworlds/examples/find_and_avoid_v2/controllers/robot_supervisor_manager` and run 
`tensorboard --logdir ./experiments/`.

For testing, a CSV file is produced that includes the reward per episode, how the episode ended (collision, timeout 
or reached target), as well as the steps taken until the end of the episode. Moreover, the average reward and the 
standard deviation of the reward are recorded, as returned by sb3's `evaluate_policy`.

## Agents 
    
+ [Stable Baselines3 - Contrib Maskable Proximal Policy Optimization (Maskable-PPO)](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)

## Cite

- [Deep Reinforcement Learning With Action Masking for Differential-Drive Robot Navigation Using Low-Cost Sensors, K Tsampazis, et al., MLSP2023, IEEE](https://ieeexplore.ieee.org/abstract/document/10285997)
