from deepbots.supervisor import RobotSupervisorEnv
from utilities import normalize_to_range

from gym.spaces import Box, Discrete
import numpy as np


class CartPoleRobotSupervisor(RobotSupervisorEnv):
    """
    CartPoleRobotSupervisor acts as an environment having all the appropriate methods such as get_reward().
    This class utilizes the robot-supervisor scheme combining both the robot controls and the environment
    in the same class. Moreover, the reset procedure used is the default implemented reset.
    This class is made with the new release of deepbots in mind that fully integrates gym.Env, using gym.spaces.

    Taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py and modified
    for Webots.
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves forwards and backwards. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described
        by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position x axis      -0.4            0.4
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -1.3 rad        1.3 rad
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Move cart forward
        1	Move cart backward

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        [0.0, 0.0, 0.0, 0.0]
    Episode Termination:
        Pole Angle is more than 0.261799388 rad (15 degrees)
        Cart Position is more than 0.39 on x axis (cart has reached arena edge)
        Episode length is greater than 200
        Solved Requirements (average episode score in last 100 episodes > 195.0)
    """

    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        """

        super().__init__()

        # Set up gym spaces
        self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                     high=np.array([0.4, np.inf, 1.3, np.inf]),
                                     dtype=np.float64)
        self.action_space = Discrete(2)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)

        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")

        self.wheels = [None for _ in range(4)]
        self.setup_motors()

        # Set up misc
        self.steps_per_episode = 200  # How many steps to run each episode (changing this messes up the solved condition)
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the CartPole problem.
        All values apart are gathered here from the robot and pole_endpoint objects.
        All values are normalized appropriately to [-1, 1], according to their original ranges.

        :return: Observation: [cart_position, cart_velocity, pole_angle, poleTipVelocity]
        :rtype: list
        """
        # Position on x axis
        cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on x axis
        cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Pole angle off vertical
        pole_angle = normalize_to_range(self.position_sensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
        # Angular velocity y of endpoint
        endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

        return np.array([cart_position, cart_velocity, pole_angle, endpoint_velocity])

    def get_reward(self, action):
        """
        Reward is +1 for each step taken, including the termination step.

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: Always 1
        :rtype: int
        """
        return 1

    def is_done(self):
        """
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is on the
        arena's edges.

        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if self.episode_score > 195.0:
            return True

        pole_angle = round(self.position_sensor.getValue(), 2)
        if abs(pole_angle) > 0.261799388:  # 15 degrees off vertical
            return True

        cart_position = round(self.robot.getPosition()[0], 2)  # Position on x axis
        if abs(cart_position) > 0.39:
            return True

        return False

    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over 195.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episode scores average value
                return True
        return False

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return np.array([0.0 for _ in range(self.observation_space.shape[0])])

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be executed by the robot.
        It contains an integer denoting the action, either 0 or 1, with 0 being forward and
        1 being backward movement. The corresponding motor_speed value is applied to the wheels.

        :param action: The list that contains the action integer
        :type action: list of int
        """
        #print(action)
        #action = int(action[0])

        assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(action)

        if action == 0:
            motor_speed = 5.0
        else:
            motor_speed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)

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
