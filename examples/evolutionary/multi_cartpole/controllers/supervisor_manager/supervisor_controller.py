import numpy as np

from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange, plotData


class CartPoleSupervisor(SupervisorCSV):
    """
    TODO Add description of multicartpole problem

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
        0	Cart Position z axis      -0.4            0.4
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
        Cart Position is more than absolute 0.89 on z axis (cart has reached arena edge)
        Episode length is greater than 200
        Solved Requirements (average episode score in last 100 episodes > 195.0)
    """

    def __init__(self):
        """
        In the constructor, the agent object is created.
        References to robots and the pole endpoints are initialized here, used for building the observation.
        When in test mode (self.test = True) the agent stops being trained and picks actions in a non-stochastic way.
        """
        super().__init__()
        self.observationSpace = 4
        self.actionSpace = 2
        self.robot = [self.supervisor.getFromDef("ROBOT" + str(i)) for i in range(9)]
        self.initPositions = [self.robot[i].getField("translation").getSFVec3f() for i in range(9)]
        self.poleEndpoint = [self.supervisor.getFromDef("POLE_ENDPOINT_" + str(i)) for i in range(9)]

        self.messageReceived = None  # Variable to save the messages received from the robots

        self.stepsPerEpisode = 200  # How many steps to run each episode (changing this messes up the solved condition)
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.test = False  # Whether the agent is in test mode

    def get_observations(self):
        """
        This get_observation implementation builds the required observations for the MultiCartPole problem.
        All values apart from pole angle are gathered here from the robots and poleEndpoint objects.
        The pole angle value is taken from the messages sent by the robots.
        All values are normalized appropriately to [-1, 1], according to their original ranges.

        :return: Observation: [cartPosition1, cartVelocity1, poleAngle1, poleTipVelocity1],
                                [cartPosition2, cartVelocity2, poleAngle2, poleTipVelocity2]
        :rtype: tuple(list, list)
        """
        # Position on z axis
        cartPosition = [normalizeToRange(self.robot[i].getPosition()[2], -0.4, 0.4, -1.0, 1.0) for i in range(9)]

        # Linear velocity on z axis
        cartVelocity = [normalizeToRange(self.robot[i].getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True) for i in range(9)]

        self.messageReceived = []
        for _ in range(9):
            self.messageReceived.append(self.handle_receiver())  # Message contains pole angle of the first robot
        
        poleAngle = [None for _ in range(9)]

        if None not in self.messageReceived:
            for message in self.messageReceived:
                robot_no = int(message[0][5])
                poleAngle[robot_no] = normalizeToRange(float(message[1]), -0.23, 0.23, -1.0, 1.0, clip=True)
        else:
            # method is called before messageReceived is initialized
            poleAngle = [0.0 for _ in range(9)]

        # Angular velocity x of endpoint
        endpointVelocity = [normalizeToRange(self.poleEndpoint[i].getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True) for i in range(9)]


        messages = [None for _ in range(9)]
        for i in range(9):
            messages[i] = [cartPosition[i], cartVelocity[i], poleAngle[i], endpointVelocity[i]]
        
        return messages[0], messages[1], messages[2], messages[3], messages[4], messages[5], messages[6], messages[7], messages[8]

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

        if self.episodeScore > 195.0:
            return True

        if None not in self.messageReceived:
            # Identify which part of the message list comes from which robot
            poleAngle = [None for i in range(9)]
            for message in self.messageReceived:
                robot_no = int(message[0][5])
                poleAngle[robot_no] = round(float(message[1]), 2)

        else:
            # method is called before messageReceived is initialized
            poleAngle = [0.0 for _ in range(9)]
        
        if all(abs(x) > 0.261799388 for x in poleAngle):  # 15 degrees off vertical
            return True

        cartPosition = [round(self.robot[i].getPosition()[2] - self.initPositions[i][2], 2) for i in range(9)]      
        if all(abs(x) > 0.89 for x in cartPosition):
            return True

        return False

    def get_default_observation(self):
        """
        Returns the default observation of zeros.

        :return: Default observation zero vector
        :rtype: list
        """
        observation = []

        for _ in range(9):
            robot_obs = [0.0 for _ in range(self.observationSpace)]
            observation.append(robot_obs)

        return observation

    def get_info(self):
        """
        Dummy implementation of get_info.

        :return: None
        :rtype: None
        """
        return None

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



'''import numpy as np

from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange, plotData


class CartPoleSupervisor(SupervisorCSV):
    """
    TODO Add description of multicartpole problem

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
        0	Cart Position z axis      -0.4            0.4
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
        Cart Position is more than absolute 0.89 on z axis (cart has reached arena edge)
        Episode length is greater than 200
        Solved Requirements (average episode score in last 100 episodes > 195.0)
    """

    def __init__(self):
        """
        In the constructor, the agent object is created.
        References to robots and the pole endpoints are initialized here, used for building the observation.
        When in test mode (self.test = True) the agent stops being trained and picks actions in a non-stochastic way.
        """
        super().__init__()
        self.observationSpace = 4
        self.actionSpace = 2
        self.robot1 = self.supervisor.getFromDef("ROBOT1")
        self.robot2 = self.supervisor.getFromDef("ROBOT2")

        self.poleEndpoint1 = self.supervisor.getFromDef("POLE_ENDPOINT_1")
        self.poleEndpoint2 = self.supervisor.getFromDef("POLE_ENDPOINT_2")
        self.messageReceived = None  # Variable to save the messages received from the robots

        self.stepsPerEpisode = 200  # How many steps to run each episode (changing this messes up the solved condition)
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.test = False  # Whether the agent is in test mode

    def get_observations(self):
        """
        This get_observation implementation builds the required observations for the MultiCartPole problem.
        All values apart from pole angle are gathered here from the robots and poleEndpoint objects.
        The pole angle value is taken from the messages sent by the robots.
        All values are normalized appropriately to [-1, 1], according to their original ranges.

        :return: Observation: [cartPosition1, cartVelocity1, poleAngle1, poleTipVelocity1],
                                [cartPosition2, cartVelocity2, poleAngle2, poleTipVelocity2]
        :rtype: tuple(list, list)
        """
        # Position on z axis
        cartPosition1 = normalizeToRange(self.robot1.getPosition()[2], -0.4, 0.4, -1.0, 1.0)
        cartPosition2 = normalizeToRange(self.robot2.getPosition()[2], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on z axis
        cartVelocity1 = normalizeToRange(self.robot1.getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)
        cartVelocity2 = normalizeToRange(self.robot2.getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)

        self.messageReceived = []
        self.messageReceived.append(self.handle_receiver())  # Message contains pole angle of one robot
        self.messageReceived.append(self.handle_receiver())  # Message contains pole angle of the second robot

        if None not in self.messageReceived:
            # Identify which part of the message list comes from which robot
            if self.messageReceived[0][0] == "robot1":
                poleAngle1 = normalizeToRange(float(self.messageReceived[0][1]), -0.23, 0.23, -1.0, 1.0, clip=True)
                poleAngle2 = normalizeToRange(float(self.messageReceived[1][1]), -0.23, 0.23, -1.0, 1.0, clip=True)
            else:
                poleAngle1 = normalizeToRange(float(self.messageReceived[1][1]), -0.23, 0.23, -1.0, 1.0, clip=True)
                poleAngle2 = normalizeToRange(float(self.messageReceived[0][1]), -0.23, 0.23, -1.0, 1.0, clip=True)
        else:
            # method is called before messageReceived is initialized
            poleAngle1 = 0.0
            poleAngle2 = 0.0

        # Angular velocity x of endpoint
        endpointVelocity1 = normalizeToRange(self.poleEndpoint1.getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True)
        endpointVelocity2 = normalizeToRange(self.poleEndpoint2.getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True)

        return [cartPosition1, cartVelocity1, poleAngle1, endpointVelocity1],\
               [cartPosition2, cartVelocity2, poleAngle2, endpointVelocity2]

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
        if self.episodeScore > 195.0:
            return True

        if None not in self.messageReceived:
            # Identify which part of the message list comes from which robot
            if self.messageReceived[0][0] == "robot1":
                poleAngle1 = round(float(self.messageReceived[0][1]), 2)
                poleAngle2 = round(float(self.messageReceived[1][1]), 2)
            else:
                poleAngle1 = round(float(self.messageReceived[1][1]), 2)
                poleAngle2 = round(float(self.messageReceived[0][1]), 2)
        else:
            # method is called before messageReceived is initialized
            poleAngle1 = 0.0
            poleAngle2 = 0.0

        if abs(poleAngle1) > 0.261799388 or abs(poleAngle2) > 0.261799388:  # 15 degrees off vertical
            return True

        cartPosition1 = round(self.robot1.getPosition()[2], 2)  # Position on z axis
        cartPosition2 = round(self.robot2.getPosition()[2], 2)  # Position on z axis
        if abs(cartPosition1) > 0.89 or abs(cartPosition2) > 0.89:
            return True

        return False

    def get_default_observation(self):
        """
        Returns the default observation of zeros.

        :return: Default observation zero vector
        :rtype: list
        """
        return [[0.0 for _ in range(self.observationSpace)], [0.0 for _ in range(self.observationSpace)]]

    def get_info(self):
        """
        Dummy implementation of get_info.

        :return: None
        :rtype: None
        """
        return None

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
        return False'''
