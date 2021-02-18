from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from utilities import normalizeToRange

from gym.spaces import Box, Discrete
import numpy as np


class CartPoleRobotSupervisor(RobotSupervisor):
    def __init__(self):
        super().__init__()

        # Set up gym spaces
        self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                     high=np.array([0.4, np.inf, 1.3, np.inf]),
                                     dtype=np.float64)
        self.action_space = Discrete(2)

        # Set up various robot components
        self.positionSensor = self.getDevice("polePosSensor")
        self.positionSensor.enable(self.timestep)

        self.poleEndpoint = self.getFromDef("POLE_ENDPOINT")

        self.wheels = [None for _ in range(4)]
        self.setup_motors()

        # Set up misc
        self.stepsPerEpisode = 200  # How many steps to run each episode (changing this messes up the solved condition)
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        # Position on z axis
        cartPosition = normalizeToRange(self.getSelf().getPosition()[2], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on z axis
        cartVelocity = normalizeToRange(self.getSelf().getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)

        poleAngle = normalizeToRange(self.positionSensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)

        # Angular velocity x of endpoint
        endpointVelocity = normalizeToRange(self.poleEndpoint.getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True)

        return [cartPosition, cartVelocity, poleAngle, endpointVelocity]

    def get_reward(self, action):
        return 1

    def is_done(self):
        if self.episodeScore > 195.0:
            return True

        poleAngle = round(self.positionSensor.getValue(), 2)
        if abs(poleAngle) > 0.261799388:  # 15 degrees off vertical
            return True

        cartPosition = round(self.getSelf().getPosition()[2], 2)  # Position on z axis
        if abs(cartPosition) > 0.39:
            return True

        return False

    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over 195.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episode scores average value
                return True
        return False

    def get_default_observation(self):
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def apply_action(self, action):
        action = int(action[0])

        assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(action)

        if action == 0:
            motorSpeed = 5.0
        else:
            motorSpeed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motorSpeed)

    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.wheels[0] = self.getDevice('wheel1')
        self.wheels[1] = self.getDevice('wheel2')
        self.wheels[2] = self.getDevice('wheel3')
        self.wheels[3] = self.getDevice('wheel4')
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0.0)

    def get_info(self):
        return {}

    def render(self, mode='human'):
        print("render() is not used")
