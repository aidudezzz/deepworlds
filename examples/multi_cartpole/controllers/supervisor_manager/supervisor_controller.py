import numpy as np
from deepbots.supervisor import EmitterReceiverSupervisorEnv

from utilities import normalizeToRange


class CartPoleSupervisor(EmitterReceiverSupervisorEnv):
    """
    CartPoleSupervisor acts as an environment having all the appropriate methods such as get_reward().

    Taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py and modified for Webots.
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves forwards and backwards. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's
        velocity. 
        Multicartpole consists 9 cartpole robots training simultaneously. Works using the Emitter Receiver Scheme.
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
        Cart Position is more than absolute 0.89 on x axis (cart has reached arena edge)
        Episode length is greater than 200
        Solved Requirements (average episode score in last 100 episodes > 195.0)
    """

    def __init__(self, num_robots=9):
        """
        In the constructor, the agent object is created.
        References to robots and the pole endpoints are initialized here, used for building the observation.
        When in test mode (self.test = True) the agent stops being trained and picks actions in a non-stochastic way.
        
        :param num_robots: Number of robots in the environment
        :type num_robots: int
        """
        self.num_robots = num_robots
        super(CartPoleSupervisor, self).__init__()
        self.observationSpace = 4
        self.actionSpace = 2
        self.robot = [
            self.getFromDef("ROBOT" + str(i)) for i in range(self.num_robots)
        ]

        self.poleEndpoint = [
            self.getFromDef("POLE_ENDPOINT_" + str(i))
            for i in range(self.num_robots)
        ]

        # self.pole_sensor = [
        #     self.getFromDevice('POLE_POS_SENSOR_' + str(i))
        #     for i in range(self.num_robots)
        # ]

        self.messageReceived = None  # Variable to save the messages received from the robots
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = [
        ]  # A list to save all the episode scores, used to check if task is solved
        self.test = False  # Whether the agent is in test mode

        self.init_positions = [
            self.robot[i].getPosition()[1] for i in range(self.num_robots)
        ]

        print('Init positions:', self.init_positions)

    def initialize_comms(self, emitter_name=None, receiver_name=None):
        """
        Initializes emitter and receiver channels for each robot
        :return communication: A list of dictionaries, each corresponding to 1 robot with keys "emitter" and "receiver" and values
                                as emitter and receiver instances of the robot
        :rtype: list(dict)
        """
        emitter_list = []
        receiver_list = []
        for i in range(self.num_robots):
            emitter = self.getDevice(f'emitter{i}')
            receiver = self.getDevice(f'receiver{i}')

            emitter.setChannel(i)
            receiver.setChannel(i)

            receiver.enable(self.timestep)

            emitter_list.append(emitter)
            receiver_list.append(receiver)

        return emitter_list, receiver_list

    def handle_emitter(self, actions):
        """
        Emits actions to the robots through robot specific emitter channels
        """
        for i, action in enumerate(actions):
            message = str(action).encode("utf-8")
            self.emitter[i].send(message)

    def handle_receiver(self):
        """
        Receives actions sent by the robots in the environment through robot specific receiver channels
        :return messages: list of messages sent by the robots
        :rtype: list(int)
        """
        messages = []
        for receiver in self.receiver:
            if receiver.getQueueLength() > 0:
                messages.append(receiver.getString())
                receiver.nextPacket()
            else:
                messages.append(None)

        return messages

    def get_observations(self):
        """
        This get_observation implementation builds the required observations for the MultiCartPole problem.
        All values apart from pole angle are gathered here from the robots and poleEndpoint objects.
        The pole angle value is taken from the messages sent by the robots.
        All values are normalized appropriately to [-1, 1], according to their original ranges.

        :return: Observation:[[cartPosition0, cartVelocity0, poleAngle0, poleTipVelocity0],
                              [cartPosition1, cartVelocity1, poleAngle1, poleTipVelocity1], ...
                              [cartPosition9, cartVelocity9, poleAngle9, poleTipVelocity9]]
        :rtype: list(list(list(float), list(float), float, float))   
        """
        message_received = self.handle_receiver()
        if None not in message_received:
            self.messageReceived = np.array(
                list(map(float, message_received)))
        else:
            self.messageReceived = np.array([0.0 for _ in range(self.num_robots)])

        # Position on y-axis
        relative_positions = [
            self.robot[i].getPosition()[1] - self.init_positions[i]
            for i in range(self.num_robots)
        ]

        cart_positions, cart_velocities, pole_angles, endpoint_velocities = [], [], [], []
        for i in range(self.num_robots):
            # Position on y-axis
            cart_positions.append(
                normalizeToRange(relative_positions[i], -0.35, 0.35, -1.0, 1.0))

            # Linear velocity of cart on y-axis
            cart_velocities.append(
                normalizeToRange(self.robot[i].getVelocity()[1], -0.2, 0.2, -1.0, 1.0, clip=True))

            # Pole angle off vertical
            pole_angles.append(
                normalizeToRange(self.messageReceived[i],
                                 -0.261799388,
                                 0.261799388,
                                 -1.0,
                                 1.0,
                                 clip=True))

            # Angular velocity x of endpoint
            endpoint_velocities.append(np.clip(self.poleEndpoint[i].getVelocity()[3], -1.0, 1.0))

        return np.array([
            cart_positions, cart_velocities, pole_angles, endpoint_velocities
        ]).T

    def get_reward(self, action=None):
        """
        Reward is +1 for each step taken, including the termination step.

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: Always 1
        :rtype: int
        """

        return (np.abs(self.messageReceived) < 0.261799388).astype(int)

    def is_done(self):
        """
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is a certain distance 
        away from the initial position for either of the carts

        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """

        if self.episodeScore > 195.0:
            return True

        if not all(np.abs(self.messageReceived) < 0.261799388):  # 15 degrees off vertical
            return True

        relative_positions = np.array([
            self.robot[i].getPosition()[1] - self.init_positions[i]
            for i in range(self.num_robots)
        ])
        if not any(np.abs(relative_positions) < 0.35):
            return True

        return False

    def get_default_observation(self):
        """
        Returns the default observation of zeros.

        :return: Default observation zero vector
        :rtype: list
        """
        observation = []

        for _ in range(self.num_robots):
            robot_obs = [0.0 for _ in range(self.observationSpace)]
            observation.append(robot_obs)

        return observation

    def get_info(self):
        """
        Dummy implementation of get_info.
        """
        pass

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

    def reset(self):
        """
        Used to reset the world to an initial state. Default, problem-agnostic, implementation of reset method,
        using Webots-provided methods.

        *Note that this works properly only with Webots versions >R2020b and must be overridden with a custom reset method when using
        earlier versions. It is backwards compatible due to the fact that the new reset method gets overridden by whatever the user
        has previously implemented, so an old supervisor can be migrated easily to use this class.

        :return: default observation provided by get_default_observation()
        """
        default_observation = super().reset()

        for receiver in self.receiver:
            receiver.disable()
            receiver.enable(self.timestep)
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        return default_observation
