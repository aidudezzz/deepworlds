import numpy as np

from deepbots.supervisor import CSVSupervisorEnv
from utilities import normalize_to_range


class CartPoleSupervisor(CSVSupervisorEnv):
    """
    CartPoleSupervisor acts as an environment having all the appropriate methods such as get_reward().

    Taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py and modified for Webots.
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
        Type: Continuous(1)
        Num     Min     Max     Desc
        0       -inf    inf     Value is directly tied to set motors speed

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
        In the constructor, the agent object is created, the robot is spawned in the world via respawnRobot().
        References to robot and the pole endpoint are initialized here, used for building the observation.
        When in test mode (self.test = True) the agent stops being trained and picks actions in a non-stochastic way.
        """
        print("Robot is spawned in code, if you want to inspect it pause the simulation.")
        super().__init__()
        # observation and action spaces are set as tuples, because that's how DDPG agent expects them
        self.observation_space = 4
        self.action_space = 1
        self.robot = self.getFromDef("ROBOT")

        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.message_received = None  # Variable to save the messages received from the robot

        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.test = False  # Whether the agent is in test mode

    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the CartPole problem.
        All values apart from pole angle are gathered here from the robot and pole_endpoint objects.
        The pole angle value is taken from the message sent by the robot.
        All values are normalized appropriately to [-1, 1], according to their original ranges.

        :return: Observation: [cart_position, cart_velocity, pole_angle, poleTipVelocity]
        :rtype: list
        """
        # Position on x axis
        cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on x axis
        cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)

        self.message_received = self.handle_receiver()  # update message received from robot, which contains pole angle
        if self.message_received is not None:
            pole_angle = normalize_to_range(float(self.message_received[0]), -0.23, 0.23, -1.0, 1.0, clip=True)
        else:
            # method is called before message_received is initialized
            pole_angle = 0.0

        # Angular velocity y of endpoint
        endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

        return [cart_position, cart_velocity, pole_angle, endpoint_velocity]

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space)]

    def get_reward(self, action=None):
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

        if self.message_received is not None:
            pole_angle = round(float(self.message_received[0]), 2)
        else:
            # method is called before message_received is initialized
            pole_angle = 0.0
        if abs(pole_angle) > 0.261799388:  # 15 degrees off vertical
            return True

        cart_position = round(self.robot.getPosition()[0], 2)  # Position on x axis
        if abs(cart_position) > 0.39:
            return True

        return False

    def get_info(self):
        """
        Dummy implementation of get_info.

        :return: None
        :rtype: None
        """
        return None

    def render(self, mode="human"):
        """
        Dummy implementation of render.
        """
        pass

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
