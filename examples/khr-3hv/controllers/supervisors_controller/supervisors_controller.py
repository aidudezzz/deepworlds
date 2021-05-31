from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from RobotUtils import RobotFunc

from gym.spaces import Box, Discrete
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env

MOTOR_VELOCITY = 7



class khr3hvRobotSupervisor(RobotSupervisor):
    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        Observation:
            Type: Box(22)
            Num	Observation             Min(rad)      Max(rad)
            0   Robot position x-axis  
            1   Robot position y-axis
            2   Robot position z-axis
            3	Head                    -2.36          2.36
            4	LeftAnkle               -2.36          2.36
            5	LeftArm                 -2.36          2.36
            6	LeftCrus                -2.36          2.36
            7	LeftElbow               -2.36          2.36
            8	LeftFemur               -2.36          2.36
            9	LeftFemurHead1          -2.36          2.36
            10	LeftFemurHead2          -2.36          2.36
            11	LeftFoot                -2.36          2.36
            12	LeftForearm             -2.36          2.36
            13	LeftShoulder            -2.36          2.36
            14	RightAnkle              -2.36          2.36
            15	RightArm                -2.36          2.36
            16	RightCrus               -2.36          2.36
            17	RightElbow              -2.36          2.36
            18	RightFemur              -2.36          2.36
            19	RightFemurHead1         -2.36          2.36
            20	RightFemurHead2         -2.36          2.36
            21	RightFoot               -2.36          2.36
            22	RightForearm            -2.36          2.36
            23	RightShoulder           -2.36          2.36
            24	Waist                   -2.36          2.36
            
        Actions:
            Type: Continuous
            Num	BodyPost    Min   Max   Desc
            0	LeftAnkle   -1    +1    Set the motor position from θ to θ + (action 0)*0.032
            1	LeftCrus    -1    +1    Set the motor position from θ to θ + (action 1)*0.032
            2	LeftFemur   -1    +1    Set the motor position from θ to θ + (action 2)*0.032
            3	LeftFoot    -1    +1    Set the motor position from θ to θ + (action 3)*0.032
            4	RightAnkle  -1    +1    Set the motor position from θ to θ + (action 4)*0.032
            5	RightCrus   -1    +1    Set the motor position from θ to θ + (action 5)*0.032
            6	RightFemur  -1    +1    Set the motor position from θ to θ + (action 6)*0.032
            7	RightFoot   -1    +1    Set the motor position from θ to θ + (action 7)*0.032
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
        lowObs = -2.36 * np.ones(22)
        maxObs = 2.36 * np.ones(22)
        self.observation_space = Box(low=lowObs, high=maxObs, dtype=np.float64)
        
        lowAct = -1 * np.ones(22)
        maxAct = np.ones(22)
        self.action_space = Box(low=lowAct, high=maxAct, dtype=np.float64)

        # Set up various robot components
        self.robot = self.getSelf() 
    
        self.setup_agent()
        
        self.motorPositionArr = np.zeros(22)
        
        # Set up misc
        self.stepsPerEpisode = 200  # How many steps to run each episode (changing this messes up the solved condition)
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  

    def get_observations(self):

        #RobotPositionX = RobotFunc.normalizeToRange(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        #RobotPositionY = RobotFunc.normalizeToRange(self.robot.getPosition()[1], -0.4, 0.4, -1.0, 1.0)
        #RobotPositionZ = RobotFunc.normalizeToRange(self.robot.getPosition()[2], -0.4, 0.4, -1.0, 1.0)
        motorPos = []
        
        motorPos.extend([i for i in self.motorPositionArr])
        return np.array(motorPos)
            
        

    def get_reward(self, action):
        alive = 1
        movingForward = 1.25 * self.robot.getPosition()[2] #  get the velocity of agent or self.robot.getPosition()[2]
        #battery =  
        reward = alive + movingForward
        print(reward)
        return reward 

    def is_done(self):
        if self.episodeScore > 300:
            return True
        # agent fall
        if self.robot.getPosition()[1] < 0.5:
            return True         

        return False

    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires thself.robot.getPosition()[2]at the average episode score of last 100 episodes is over 195.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 300:  # Last 100 episode scores average value
                return True
        return False

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return np.zeros(self.observation_space.shape[0])

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be executed by the robot.
        It contains an integer denoting the action, either 0 or 1, with 0 being forward and
        1 being backward movement. The corresponding motorSpeed value is applied to the wheels.

        :param action: The list that contains the action integer
        :type action: list of int
        """
        #print(action)
        #action = int(action[0])
        for i in range(self.action_space.shape[0]):
            motorPosition = self.motorPositionArr[i] + action[i]
            motorPosition = np.clip(motorPosition, -2.36, 2.36)
            self.motorList[i].setVelocity(MOTOR_VELOCITY)
            self.motorList[i].setPosition(motorPosition)

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

    def Movehands(self,motion,counter):
        for _ in range(20):
            self.motorList[2].setVelocity(MOTOR_VELOCITY)
            self.motorList[12].setVelocity(MOTOR_VELOCITY)
            if motion:
                counter += 0.1
            else:
                counter -= 0.1
            print(counter)
            self.motorList[2].setPosition(counter)
            self.motorList[12].setPosition(counter)
        return counter
        
          
            
            
            

           



            
        





env = khr3hvRobotSupervisor()

check_env(env)

"""
motion = True
counter = 0

for i in range(1):
    counter = env.Movehands(motion, counter)
    motion = not motion

"""

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=40000)
model.save("ppo1_cartpole")

#del model # remove to demonstrate saving and loading

#model = PPO.load("ppo1_cartpole")

obs = env.reset()
env.episodeScore = 0
while True:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.episodeScore += reward  # Accumulate episode reward

    if done:
        print("Reward accumulated =", env.episodeScore)
        env.episodeScore = 0
        obs = env.reset()























