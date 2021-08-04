from deepbots.supervisor.controllers.supervisor_env import SupervisorEnv
import numpy as np
from controller import Supervisor
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange


class CartPoleSupervisor(SupervisorEnv):
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
    TODO:
    - Improve docstrings
    - Make it agnostic to number of robots
    """

    def __init__(self):
        """
        In the constructor, the agent object is created.
        References to robots and the pole endpoints are initialized here, used for building the observation.
        When in test mode (self.test = True) the agent stops being trained and picks actions in a non-stochastic way.
        """
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())
        self.communication = self.initialize_comms(9)

        self.observationSpace = 4
        self.actionSpace = 2
        self.robot = [self.getFromDef("ROBOT" + str(i)) for i in range(9)]
        self.initPositions = [self.robot[i].getField("translation").getSFVec3f() for i in range(9)]
        self.poleEndpoint = [self.getFromDef("POLE_ENDPOINT_" + str(i)) for i in range(9)]

        self.messageReceived = None                     # Variable to save the messages received from the robots

        self.stepsPerEpisode = 200                      # Number of steps per episode
        self.episodeScore = 0                           # Score accumulated during an episode
        self.episodeScoreList = []                      # A list to save all the episode scores, used to check if task is solved
        self.test = False                               # Whether the agent is in test mode

    def initialize_comms(self, robots_num):
        communication = []
        for i in range(robots_num):
            emitter = self.getDevice(f'emitter{i}')
            receiver = self.getDevice(f'receiver{i}')

            emitter.setChannel(i)
            receiver.setChannel(i)

            receiver.enable(self.timestep)

            communication.append({
                'emitter': emitter,
                'receiver': receiver,
            })

        return communication

    def step(self, action):
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        self.handle_emitter(action)

        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    def handle_emitter(self, actions):
        for i, action in enumerate(actions):
            message = str(action).encode("utf-8")
            self.communication[i]['emitter'].send(message)

    def handle_receiver(self):
        messages = []
        for com in self.communication:
            receiver = com['receiver']
            if receiver.getQueueLength() > 0:
                messages.append(receiver.getData().decode("utf-8"))
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

        :return: Observation: [cartPosition0, cartVelocity0, poleAngle0, poleTipVelocity0],
                              [cartPosition1, cartVelocity1, poleAngle1, poleTipVelocity1], ...
                              [cartPosition9, cartVelocity9, poleAngle9, poleTipVelocity9]
        :rtype: tuple(list, list)
        """
        # Position on z axis
        cartPosition = [normalizeToRange(self.robot[i].getPosition()[2], -0.4, 0.4, -1.0, 1.0) for i in range(9)]

        # Linear velocity on z axis
        cartVelocity = [normalizeToRange(self.robot[i].getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True) for i in range(9)]

        self.messageReceived = self.handle_receiver()

        poleAngle = [None for _ in range(9)]

        for i, message in enumerate(self.messageReceived):    
            if message is not None:
                poleAngle[i] = normalizeToRange(message, -0.23, 0.23, -1.0, 1.0, clip=True)
            else:
                # method is called before messageReceived is initialized
                poleAngle = [0.0 for _ in range(9)]

        # Angular velocity x of endpoint
        endpointVelocity = [normalizeToRange(self.poleEndpoint[i].getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True) for i in range(9)]


        messages = [None for _ in range(9)]
        for i in range(9):
            messages[i] = [cartPosition[i], cartVelocity[i], poleAngle[i], endpointVelocity[i]]
        
        return messages

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
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is a certain distance 
        awat from the initial position for either of the carts

        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """

        if self.episodeScore > 195.0:
            return True

        if None not in self.messageReceived:                                    # Identify which part of the message list comes from which robot
            poleAngle = [None for _ in range(9)]
            for message in self.messageReceived:
                print(message)
                robot_no = int(message[0][5])
                poleAngle[robot_no] = round(float(message[1]), 2)

        else:                                                                   # if method is called before messageReceived is initialized
            poleAngle = [0.0 for _ in range(9)]
        
        if not all(abs(x) < 0.261799388 for x in poleAngle):                    # 15 degrees off vertical
            return True

        cartPosition = [round(self.robot[i].getPosition()[2] - self.initPositions[i][2], 2) for i in range(9)]      
        if not all(abs(x) < 0.89 for x in cartPosition):
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
        pass

    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over 195.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episodeScoreList) > 100:                         # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 195.0:        # Last 100 episode scores average value
                return True
        return False

    def reset(self):
        """
        Used to reset the world to an initial state.
        Default, problem-agnostic, implementation of reset method,
        using Webots-provided methods.
        *Note that this works properly only with Webots versions >R2020b
        and must be overridden with a custom reset method when using
        earlier versions. It is backwards compatible due to the fact
        that the new reset method gets overridden by whatever the user
        has previously implemented, so an old supervisor can be migrated
        easily to use this class.
        :return: default observation provided by get_default_observation()
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        super(Supervisor, self).step(int(self.getBasicTimeStep()))

        for i in range(9):
            self.communication[i]['receiver'].disable()
            self.communication[i]['receiver'].enable(self.timestep)

            receiver = self.communication[i]['receiver']
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        return self.get_default_observation()