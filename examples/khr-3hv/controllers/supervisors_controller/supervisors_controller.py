from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from RobotUtils import RobotFunc

from gym.spaces import Box, Discrete
import numpy as np
import math

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune import grid_search
import ray.rllib.agents.ppo as ppo

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback

from ray.tune.integration.wandb import wandb_mixin

import wandb
MOTOR_VELOCITY = 7

MAX_DISTANCE_WALKED = 0

PROJECT_NAME = "Deepbots-Custom-Reward-1term-Cuda"


class khr3hvRobotSupervisor(RobotSupervisor):
    
    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        Observation:
            Type: Box(9)
            Num	Observation             Min(rad)      Max(rad)
            0   Robot position x-axis   -inf          +inf      
            1   Robot position y-axis   -inf          +inf
            2   Robot position z-axis   -inf          +inf 
            3	LeftAnkle               -inf          +inf
            4	LeftCrus                -inf          +inf
            5	LeftFemur               -inf          +inf
            6	RightAnkle              -inf          +inf
            7	RightCrus               -inf          +inf
            8	RightFemur              -inf          +inf
            
            
        Actions:
            Type: Continuous
            Num	BodyPost     Min      Max      Desc
            0	LeftAnkle   -2.35    +2.35    Set the motor position from -2.35 to +2.35 
            1	LeftCrus    -2.35    +2.35    Set the motor position from -2.35 to +2.35 
            2	LeftFemur   -2.35    +2.35    Set the motor position from -2.35 to +2.35 
            3	RightAnkle  -2.35    +2.35    Set the motor position from -2.35 to +2.35 
            4	RightCrus   -2.35    +2.35    Set the motor position from -2.35 to +2.35 
            5	RightFemur  -2.35    +2.35    Set the motor position from -2.35 to +2.35 
          
        Reward: 
            bonus for moving forward fast
            bonues for being alive
            malus for battery usage
        Starting State:
            [0, 0, ..., 0]
        
        Episode Termination:

            Obs = [0.0 for _ in range(self.observation_space.shape[0])]
        """

        super().__init__()
        lowObs = -np.inf * np.ones(9)
        maxObs = np.inf * np.ones(9)
        self.observation_space = Box(low=lowObs, high=maxObs, dtype=np.float64)
        
        lowAct = -2.356 * np.ones(6)
        maxAct = 2.356 * np.ones(6)
        # NOTE Ray is using np.float32 webots is need on the motor np.float64
        # Stable baseline could work with np.float64
        self.action_space = Box(low=lowAct, high=maxAct, dtype=np.float64)

        # Set up various robot components
        self.robot = self.getSelf() 

        self.maxIter = 200
        self.curentIter = 0
        self.setup_agent()
        self.actuators = np.zeros(6)
        
        self.motorPositionArr = np.zeros(6)
        
        self.counterLoging = 0
        # Set up misc
        self.stepsPerEpisode = 200  # How many steps to run each episode (changing this messes up the solved condition)
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  

   
    def get_observations(self):

       
        motorPos = self.robot.getPosition()
        global MAX_DISTANCE_WALKED
        
        
        MAX_DISTANCE_WALKED = max(MAX_DISTANCE_WALKED, motorPos[2])

        self.counterLoging +=1

        #if self.counterLoging % 100 == 0:
        #    wandb.init(project=PROJECT_NAME, reinit=False)
        #    wandb.log({"MAX_DISTANCE_WALKED": MAX_DISTANCE_WALKED})
        
       
        
        motorPos.extend([i for i in self.motorPositionArr])
        return np.array(motorPos)
            
        

    def get_reward(self, action):
        
        #movingForward = 1.25 * self.robot.getPosition()[2] #  get the velocity of agent or self.robot.getPosition()[2]
        self.curentIter += 1
        
        reward = ((self.maxIter - self.curentIter) / self.maxIter) * self.robot.getPosition()[2] #- \
        #        0.02 * np.sum(np.power(self.actuators, 2)) - \
        #        0.05  *  math.pow(self.robot.getPosition()[0],2)
        # Mujoco reward
        #reward = min(7, self.robot.getVelocity()[2]) - \
        #    0.005 * (np.power(self.robot.getVelocity()[2], 2) + np.power(self.robot.getVelocity()[0], 2)) - \
        #    0.05 *  self.robot.getPosition()[0] - \
        #    0.02 * np.power(np.linalg.norm(self.robot.getVelocity()),2) + 0.02   
        
        # Video reward missing the accutators
        #reward = self.robot.getVelocity()[2] + 0.0625 - \
        #          50 * math.pow(self.robot.getPosition()[1],2) - \
        #          0.02 * np.sum(np.power(self.actuators, 2)) - \
        #          3 *  math.pow(self.robot.getPosition()[0],2)
           
        return reward 

    def is_done(self):
        # Agent fall
        if self.robot.getPosition()[1] < 0.54:
            return True     
        # Agent walked out of the box
        #if self.robot.getPosition()[2] > 25: 
        #    return True  

        return False


    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        self.curentIter = 0
        #self.actuators = np.zeros(6)
        return np.zeros(self.observation_space.shape[0])

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be executed by the robot.
        It contains an integer denoting the action, either 0 or 1, with 0 being forward and
        1 being backward movement. The corresponding motorSpeed value is applied to the wheels.

        :param action: The list that contains the action integer
        :type action: list of int
        """
        motorIndexes = [0, 1, 2, 3, 4, 5]

        for i, ac in zip(motorIndexes, action):
            #ac = np.clip(ac, -2.35, 2.35)
            #self.actuators[i] = ac
            self.motorPositionArr[i] += ac
            self.motorList[i].setVelocity(MOTOR_VELOCITY)
            self.motorList[i].setPosition(ac)

    def setup_agent(self):
        """
        This method initializes the 17 motors, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.motorList = RobotFunc.getAllMotors(self)

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        :param mode:
        :return:
        """
        print("render() is not used")

        
          
            
            
            
useRay = False
           

output = 'results/'
 


if useRay:

    register_env("khr3hv", lambda config: khr3hvRobotSupervisor())

    model_config = {
            "env": "khr3hv",
            "model":{
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
            "lr": 1e-4,  # try different lrs
            "framework": "torch",
            "gamma": 0.995,
            "lambda": 0.95,
            "clip_param": 0.2,
            "kl_coeff": 1.0,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 64,
            "horizon": 200,
            "train_batch_size": 64,
            "num_workers": 1,
            "num_gpus": 1,
            "batch_mode": "complete_episodes",
            "observation_filter": "MeanStdFilter"
            }

    stop = {
            "timesteps_total": 30e6,
    }

    tune.run(ppo.PPOTrainer, 
                        config=model_config, 
                        stop=stop,
                        callbacks=[WandbLoggerCallback(
                                project=PROJECT_NAME,
                                api_key_file="../../../wandb.txt",
                                log_config=True)],
                        local_dir=output, 
                        checkpoint_freq=200,
                        checkpoint_at_end=True)

    ray.shutdown()


else:

    khr3_env = khr3hvRobotSupervisor()

    #check_env(khr3_env)
    model = PPO("MlpPolicy", khr3_env, verbose=1)
    model.learn(total_timesteps=4000000)
    model.save("khr3hvRobotSupervisor")




print("MAX_DISTANCE_WALKED = ", MAX_DISTANCE_WALKED)
""" 
obs = khr3_env.reset()
khr3_env.episodeScore = 0
while True:
    action, _states = model.predict(obs)
    obs, reward, done, _ = khr3_env.step(action)
    khr3_env.episodeScore += reward  # Accumulate episode reward

    if done:
        print("Reward accumulated =", khr3_env.episodeScore)
        khr3_env.episodeScore = 0
        obs = khr3_env.reset()
        break

"""





















