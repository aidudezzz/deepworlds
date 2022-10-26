from deepbots.supervisor import RobotSupervisorEnv
from gym.spaces import Box, Discrete
import numpy as np
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # 
from ArmUtil import Func, ToArmCoord
from robot_supervisor_manager import STEPS_PER_EPISODE, MOTOR_VELOCITY

class PandaRobotSupervisor(RobotSupervisorEnv):
    """
    Observation:
        Type: Box(10)
        Num	Observation                Min(rad)      Max(rad)
        0	Target x                   -Inf           Inf
        1	Target y                   -Inf           Inf
        2	Target z                   -Inf           Inf
        3	Position Sensor on A1      -2.8972        2.8972
        4	Position Sensor on A2      -1.7628        1.7628
        5	Position Sensor on A3      -2.8972        2.8972
        6	Position Sensor on A4      -3.0718       -0.0698
        7	Position Sensor on A5      -2.8972        2.8972
        8   Position Sensor on A6      -0.0175        3.7525
        9	Position Sensor on A7      -2.8972        2.8972
        
    Actions:
        Type: Continuous
        Num	  Min   Max   Desc
        0	  -1    +1    Set the motor position from θ to θ + (action 0)*0.032
        ...
        6     -1    +1    Set the motor position from θ to θ + (action 6)*0.032
    Reward:
        Reward is - 2-norm for every step taken (extra points for getting close enough to the target)
    Starting State:
        [Target x, Target y, Target z, 0, 0, 0, -0.0698, 0, 0, 0]
    Episode Termination:
        distance between "endEffector" and "TARGET" < 0.005 or reached step limit
        Episode length is greater than 300
        Solved Requirements (average episode score in last 100 episodes > -100.0)
    """

    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        """

        super().__init__()

        # Set up gym spaces
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -2.8972, -1.7628, -2.8972, -3.0718, -2.8972, -0.0175, -2.8972]),
                                     high=np.array([np.inf,  np.inf,  np.inf, 2.8972,  1.7628,  2.8972, -0.0698,  2.8972,  3.7525,  2.8972]),
                                     dtype=np.float64)
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float64)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.position_sensors = Func.get_all_position_sensors(self, self.timestep)
        self.end_effector = self.getFromDef("endEffector")

        # Select one of the targets
        self.target = self.getFromDef("TARGET%s"%(np.random.randint(1, 10, 1)[0]))

        self.setup_motors()

        # Set up misc
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        # Set these to ensure that the robot stops moving
        self.motor_position_values = np.zeros(7)
        self.motor_position_values_target = np.zeros(7)
        self.distance = float("inf")
        
        # handshaking limit
        self.cnt_handshaking = 0

    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the Panda goal reaching problem.
        All values apart are gathered here from the robot and TARGET objects.

        :return: Observation: [Target x, Target y, Target z, Value of Position Sensor on A1, ..., Value of Position Sensor on A7]
        :rtype: list
        """
        # process of negotiation
        prec = 0.0001
        err = np.absolute(np.array(self.motor_position_values)-np.array(self.motor_position_values_target)) < prec
        if not np.all(err) and self.cnt_handshaking<20:
            self.cnt_handshaking = self.cnt_handshaking + 1
            return ["StillMoving"]
        else:
            self.cnt_handshaking = 0
        # ----------------------
        
        target_position = ToArmCoord.convert(self.target.getPosition())
        message = [i for i in target_position]
        message.extend([i for i in self.motor_position_values])
        return message

    def get_reward(self, action):
        """
        Reward is - 2-norm for every step taken (extra points for getting close enough to the target)

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: - 2-norm (+ extra points)
        :rtype: float
        """
        target_position = self.target.getPosition()
        target_position = ToArmCoord.convert(target_position)

        end_effector_position = self.end_effector.getPosition()
        end_effector_position = ToArmCoord.convert(end_effector_position)

        self.distance = np.linalg.norm([target_position[0]-end_effector_position[0],target_position[1]-end_effector_position[1],target_position[2]-end_effector_position[2]])
        reward = -self.distance # - 2-norm
        
        # Extra points
        if self.distance < 0.01:
            reward = reward + 1.5
        elif self.distance < 0.015:
            reward = reward + 1.0
        elif self.distance < 0.03:
            reward = reward + 0.5
        return reward

    def is_done(self):
        """
        An episode is done if the distance between "end_effector" and "TARGET" < 0.005 
        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if(self.distance < 0.005):
            done = True
        else:
            done = False
        return done

    def solved(self):
        """
        This method checks whether the Panda goal reaching task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over -100.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episode_score_list) > 500:  # Over 500 trials thus far
            if np.mean(self.episode_score_list[-500:]) > 120.0:  # Last 500 episode scores average value
                return True
        return False

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        obs = [0.0 for _ in range(self.observation_space.shape[0])]
        obs[3] = -0.0698
        return obs

    def motor_to_range(self, motor_position, i):
        if(i==0):
            motor_position = np.clip(motor_position, -2.8972, 2.8972)
        elif(i==1):
            motor_position = np.clip(motor_position, -1.7628, 1.7628)
        elif(i==2):
            motor_position = np.clip(motor_position, -2.8972, 2.8972)
        elif(i==3):
            motor_position = np.clip(motor_position, -3.0718, -0.0698)
        elif(i==4):
            motor_position = np.clip(motor_position, -2.8972, 2.8972)
        elif(i==5):
            motor_position = np.clip(motor_position, -0.0175, 3.7525)
        elif(i==6):
            motor_position = np.clip(motor_position, -2.8972, 2.8972)
        else:
            pass
        return motor_position

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be executed by the robot.
        The message contains 7 float values that are applied on each motor as position.

        :param action: The message the supervisor sent containing the next action.
        :type action: list of float
        """
        # ignore this action and keep moving
        if action[0]==-1 and len(action)==1:
            for i in range(7):
                self.motor_position_values[i] = self.position_sensors[i].getValue()
                self.motor_list[i].setVelocity(MOTOR_VELOCITY)
                self.motor_list[i].setPosition(self.motor_position_values_target[i])
            return
        
        self.motor_position_values = np.array(Func.get_value(self.position_sensors))
        for i in range(7):
            motor_position = self.motor_position_values[i] + action[i]
            motor_position = self.motor_to_range(motor_position, i)
            self.motor_list[i].setVelocity(MOTOR_VELOCITY)
            self.motor_list[i].setPosition(motor_position)
            self.motor_position_values_target[i]=motor_position # Update motor_position_values_target 

    def setup_motors(self):
        """
        This method initializes the seven motors, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.motor_list = Func.get_all_motors(self)

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
