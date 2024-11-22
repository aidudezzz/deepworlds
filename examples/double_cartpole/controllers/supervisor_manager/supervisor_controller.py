import numpy as np
from deepbots.supervisor import EmitterReceiverSupervisorEnv
from gym.spaces import Box, Discrete

from utilities import normalize_to_range


class DoubleCartPoleSupervisor(EmitterReceiverSupervisorEnv):
    """
    DoubleCartPoleSupervisor serves as an environment for a double cart-pole system, 
    with a setup including two cart-poles that are joint on the tips on a shared platform. It provides all necessary 
    methods, such as get_reward(), to interact with and manage the environment.

    Description:
        Two poles are attached by un-actuated joints to two separate carts, and the top 
        tips of the two poles are connected through an extra pole. The carts can move 
        forward and backward along the x-axis. Each pole starts upright, and the goal is 
        to prevent both poles from falling over by adjusting the velocities of the carts. 
        This setup uses the Emitter-Receiver communication scheme for interaction.
    
    Observation:
        Type: Box(8)
        Num Observation                Min         Max
        0   Cart1 Position x axis      -0.4        0.4
        1   Cart1 Velocity             -Inf        Inf
        2   Pole1 Angle                -1.3 rad    1.3 rad
        3   Pole1 Velocity At Tip      -Inf        Inf
        4   Cart2 Position x axis      -0.4        0.4
        5   Cart2 Velocity             -Inf        Inf
        6   Pole2 Angle                -1.3 rad    1.3 rad
        7   Pole2 Velocity At Tip      -Inf        Inf
        
        Each observation provides the position, velocity, angle, and angular velocity of 
        both carts and poles, allowing each agent to make informed decisions to balance 
        the poles.

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Move cart forward
        1	Move cart backward

        Note: The precise velocity change for each action depends on the pole's angle, as 
        the pole’s center of gravity affects the force required to stabilize it.

    Reward:
        The agent receives a reward of 1 for every step the poles are balanced, including 
        the termination step. This encourages agents to maintain balance for as long as possible.

    Starting State:
        All observations are initialized to zero:
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Episode Termination:
        The episode terminates if any of the following conditions are met:
        - Either pole’s angle exceeds ±15 degrees (0.2618 rad).
        - Either cart’s position reaches ±0.35 on the x-axis, indicating the cart has 
          reached the platform’s edge.
        - The episode length exceeds 200 steps.

    Solved Requirements:
        The task is considered solved if the average episode score over the last 100 
        episodes exceeds 195.0, indicating consistent performance in balancing both poles.
    """

    def __init__(self, num_robots=2):
        """
        In the constructor, the agent object is created.
        References to robots and the pole endpoints are initialized here, used for building the observation.
  
        :param num_robots: Number of robots in the environment
        :type num_robots: int
        """
        self.num_robots = num_robots
        super(DoubleCartPoleSupervisor, self).__init__()
        # Observation space contains info about both cartpoles
        self.observation_space = Box(
            low=np.array([-0.4, -np.inf, -1.3, -np.inf, -0.4, -np.inf, -1.3, -np.inf]),
            high=np.array([0.4, np.inf, 1.3, np.inf, 0.4, np.inf, 1.3, np.inf]),
            dtype=np.float64
        )
        # Action space represents the available actions for each cart seperately
        self.action_space = Discrete(2)

        self.robot = [
            self.getFromDef("ROBOT" + str(i+1)) for i in range(self.num_robots)
        ]
        self.pole_endpoint = [
            self.getFromDef("POLE_ENDPOINT_" + str(i+1))
            for i in range(self.num_robots)
        ]

        self.message_received = None  # Variable to save the messages received from the robots
        self.episode_score = np.array(num_robots)
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

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

            emitter.setChannel(i+1)
            receiver.setChannel(i+1)

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
        All values apart from pole angle are gathered here from the robots and pole_endpoint objects.
        The pole angle value is taken from the messages sent by the robots.
        All values are normalized appropriately to -1, 1], according to their original ranges.
        :return: Observation:[cart_position1, cart_velocity1, pole_angle1, pole_tip_velocity1,
                              cart_position2, cart_velocity2, pole_angle2, pole_tip_velocity2]
        :rtype: list(float)   
        """
        message_received = self.handle_receiver()
        if None not in message_received:
            self.message_received = np.array(
                list(map(float, message_received)))
        else:
            self.message_received = np.array([0.0 for _ in range(self.num_robots)])

        # Position on y-axis
        positions = [
            self.robot[i].getPosition()[1]
            for i in range(self.num_robots)
        ]

        cart_positions, cart_velocities, pole_angles, endpoint_velocities = [], [], [], []
        for i in range(self.num_robots):
            # Position on x-axis
            cart_positions.append(
                normalize_to_range(positions[i], -0.35, 0.35, -1.0, 1.0)
            )

            # Linear velocity of cart on x-axis
            cart_velocities.append(
                normalize_to_range(self.robot[i].getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
            )

            # Pole angle off vertical
            pole_angles.append(
                normalize_to_range(
                    self.message_received[i],
                    -0.261799388,
                    0.261799388,
                    -1.0,
                    1.0,
                    clip=True
                )
            )

            # Angular velocity y of endpoint
            endpoint_velocities.append(np.clip(self.pole_endpoint[i].getVelocity()[4], -1.0, 1.0))

        return (np.array([
            cart_positions, cart_velocities, pole_angles, endpoint_velocities
        ]).T).flatten()

    def get_reward(self, action=None):
        """
        Reward is +1 for each step taken, including the termination step.
        :param action: Not used, defaults to None
        :type action: None, optional
        :return: Always 1
        :rtype: int
        """
        return (np.abs(self.message_received) < 0.261799388).astype(int)

    def is_done(self):
        """
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is a certain distance 
        away from the initial position for either of the carts
        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if np.all(self.episode_score > 195.0):
            return True

        if not all(np.abs(self.message_received) < 0.261799388):  # 15 degrees off vertical
            return True

        positions = np.array([
            self.robot[i].getPosition()[1]
            for i in range(self.num_robots)
        ])
        if any(np.abs(positions) >= 0.35):
            return True

        return False

    def get_default_observation(self):
        """
        Returns the default observation of zeros.
        :return: Default observation zero vector
        :rtype: list(float)
        """
        observation = [0.0 for _ in range(self.observation_space.shape[0])]
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
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.all(np.mean(self.episode_score_list[-100:], axis=0) >= 195.0):  # Last 100 episode scores average value
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
