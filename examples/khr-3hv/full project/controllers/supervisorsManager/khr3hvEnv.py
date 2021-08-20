import numpy as np
from scipy.ndimage.interpolation import shift

from RobotUtils import RobotFunc
from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from gym.spaces import Box, Discrete

import wandb 

GLOBAL_DISTANCE = 0

class khr3hvRobotSupervisor(RobotSupervisor):
    
    def __init__(self, projectName, wandbSaveFreq, train, useRay):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here. 
        For the observation we are using a time window of 5, to store the last 5 episode observations.
        Observation:
            Type: Box(10)
            Num	Observation             Min(rad)      Max(rad)
            0   Robot position x-axis   -inf          +inf      
            1   Robot position y-axis   -inf          +inf
            2   Robot position z-axis   -inf          +inf
            3   Robot velocity z-axis    0            +7  
            4	LeftAnkle               -inf          +inf
            5	LeftCrus                -inf          +inf
            6	LeftFemur               -inf          +inf
            7	RightAnkle              -inf          +inf
            8	RightCrus               -inf          +inf
            9	RightFemur              -inf          +inf
             
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
            bonus for moving forward 
            difference of current and pre vious positions

        Starting State:
            [0, 0, ..., 0]
        
        Episode Termination:
            Robot y axis smaller than 0.50 cm
            Robot walked more that 15 m
        """

        super().__init__()
        # Time window of 5
        self.t = 5
        # Number of observations
        self.numObs = 10
        # Total observations
        self.obs = np.zeros(self.t * self.numObs)
        # Lower and maximum values on observation space
        lowObs = -np.inf * np.ones(self.t * self.numObs)
        maxObs = np.inf * np.ones(self.t * self.numObs)
        self.observation_space = Box(low=lowObs, high=maxObs, dtype=np.float64)
        # Lower and maximum values on action space
        lowAct = -2.356 * np.ones(6)
        maxAct = 2.356 * np.ones(6)
        self.action_space = Box(low=lowAct, high=maxAct, dtype=np.float64)

        # Set up various robot components
        self.motor_velocity = 2
        self.robot = self.getSelf() 
        self.setup_agent()
        self.motorPositionArr = np.zeros(6)
        self.episodeScore = 0  # Score accumulated during an episode
        self.prev_pos = self.robot.getPosition()[2]

        # Logging parameters
        self.distance_walk = 0
        self.save_freq = wandbSaveFreq
        self.counterLoging = 0
        self.train = train
        self.projectName = projectName
        self.useRay = useRay
   
    def get_observations(self):
        """
        Create the observation vector on each time step. 
        The observations are the robot position (x,z,y), robot velocity, 
        and the state of each motor.
        :return: Observation vector
        :rtype: numpy array
        """
        motorPos = self.robot.getPosition()
        motorPos.append(self.robot.getVelocity()[2])

        self.distance_walk = max(self.distance_walk, self.robot.getPosition()[2])
        self.counterLoging +=1
        global GLOBAL_DISTANCE
        GLOBAL_DISTANCE = max(GLOBAL_DISTANCE, self.robot.getPosition()[2])
        # Logging on the wandb. When using Ray we don't use the custom logging because Ray has it own 
        # logging and produce compatiblity errors of two instaces of wandb.
        if self.counterLoging % self.save_freq == 0 and self.train and not self.useRay:
            #wandb.init(project=self.projectName, reinit=False)
            wandb.log({"Episode distance walked": self.distance_walk, 
                       "Current z position": self.robot.getPosition()[2],
                       "Globan distance walekd": GLOBAL_DISTANCE
                       })           
        
        motorPos.extend([i for i in self.motorPositionArr])
        # Shift the last 10 observations by one, on the time window of 5
        self.obs = shift(self.obs, -self.numObs, cval=np.NaN)
        self.obs[-self.numObs:] = motorPos

        return np.array(self.obs)

    def get_reward(self, action):
        """
        Calculate the reward of the agent. The reward award agent when moving forward on the z-axis.
        Also, we are using a second term so the agent moves further away from his previous position.
        :return: Reward value
        :rtype: float

        """
        reward = 2.5 * self.robot.getPosition()[2] + self.robot.getPosition()[2] - self.prev_pos 
        
        if self.counterLoging % self.save_freq == 0 and self.train and not self.useRay:
            wandb.log({"reward": reward,
                       "reward-1term-weight-pos": 2.5 * self.robot.getPosition()[2],
                        "reward-2term-diff-possition": self.robot.getPosition()[2] - self.prev_pos
                    })

        self.prev_pos = self.robot.getPosition()[2]
        return reward 

    def is_done(self):
        """
        This method checks the termination criteria for each episode. 
        If the criteria are satisfied return True otherwise it returns False.
        :return: Termination criteria
        :rtype: Boolean

        """
        # Agent fall.
        if self.robot.getPosition()[1] < 0.5: 
            return True     
        # Agent walked out of the box
        if self.robot.getPosition()[2] > 15: 
            return True  
        return False

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero 
        vector in the shape of the observation space.
        :return: Starting observation zero vector
        :rtype: numpy array
        """
        self.prev_pos = 0
        self.obs = np.zeros(self.t * self.numObs)
        self.distance_walk = 0 

        return np.zeros(self.observation_space.shape[0])

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be 
        executed by the robot.It contains a float number denoting the action. 
        The corresponding motorPosition value is applied at each motor.
        :param action: The list that contains the action value
        :type action: list of float
        """
        motorIndexes = [0, 1, 2, 3, 4, 5]

        for i, ac in zip(motorIndexes, action):
            self.motorPositionArr[i] += ac
            self.motorList[i].setVelocity(self.motor_velocity)
            self.motorList[i].setPosition(ac)

    def setup_agent(self):
        """
        This method initializes the 17 (all) or 6 (leg)  motors, 
        storing the references inside a list and setting the starting
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