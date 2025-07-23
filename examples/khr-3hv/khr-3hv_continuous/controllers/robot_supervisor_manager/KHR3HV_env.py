import numpy as np
from deepbots.supervisor import RobotSupervisorEnv
from gym.spaces import Box
import wandb


class KHR3HVRobotSupervisor(RobotSupervisorEnv):
    """
    This example is similar to Webots https://robotbenchmark.net/benchmark/humanoid_sprint/ example and
    Gym https://www.gymlibrary.dev/environments/mujoco/humanoid/#humanoid environment.

    Program a Konda KHR-3HV humanoid robot to walk as far as possible.

    Description:
        The robot has to walk as far as possible. This example stops if the robot falls or walks out of the box.
    Observation:
        Type: Box(10)
        Num	Observation             Min(rad)      Max(rad)
        0   Robot position x-axis   -inf          +inf
        1   Robot position y-axis   -inf          +inf
        2   Robot position z-axis   -inf          +inf
        3   Robot velocity x-axis    0            +7
        4	LeftAnkle               -inf          +inf
        5	LeftCrus                -inf          +inf
        6	LeftFemur               -inf          +inf
        7	RightAnkle              -inf          +inf
        8	RightCrus               -inf          +inf
        9	RightFemur              -inf          +inf

    Actions:
        Type: Continuous
        Num	BodyPost      Min       Max
        0	LeftAnkle   -2.356    +2.356
        1	LeftCrus    -2.356    +2.356
        2	LeftFemur   -2.356    +2.356
        3	RightAnkle  -2.356    +2.356
        4	RightCrus   -2.356    +2.356
        5	RightFemur  -2.356    +2.356

    Reward:
        The metric is based on the robot's distance from the previous position.
        Also, a bonus, 2.5*current_position, is for encouraging it to walk as far as it can.
    Starting State:
        [0, 0, ..., 0]
    Episode Termination:
        Robot z axis smaller than 0.50 cm
        Robot walked more than 15 m
    """

    def __init__(self, project_name, wandb_save_period, train, use_ray):
        """
        In the constructor, the observation_space and action_space are set and references to the various components of
        the robot required are initialized here.
        For the observation, we are using a time window of 5 to store the last 5 episode observations.
        """
        super().__init__()
        self.max_distance_walked = 0
        self.time_window = 5  # Time window of 5
        self.num_obs = 10  # Number of observations
        self.obs = np.zeros(self.time_window * self.num_obs)  # Total observations
        # Lower and maximum values on observation space
        low_obs = -np.inf * np.ones(self.time_window * self.num_obs)
        max_obs = np.inf * np.ones(self.time_window * self.num_obs)
        self.observation_space = Box(low=low_obs, high=max_obs, dtype=np.float64)
        # Lower and maximum values on action space
        low_act = -2.356 * np.ones(6)
        max_act = 2.356 * np.ones(6)
        self.action_space = Box(low=np.array(low_act, dtype=np.float32),
                                high=np.array(max_act, dtype=np.float32),
                                dtype=np.float32)

        # Set up various robot components
        self.motor_velocity = 2
        self.robot = self.getSelf()
        self.setup_agent()
        self.motor_position_arr = np.zeros(6)
        self.episode_score = 0  # Score accumulated during an episode
        self.prev_pos = self.robot.getPosition()[0]

        # Logging parameters
        self.distance_walked = 0
        self.save_period = wandb_save_period
        self.counter_logging = 0
        self.time_window_train = train
        self.project_name = project_name
        self.use_ray = use_ray

    def get_observations(self):
        """
        Create the observation vector on each time step.
        The observations are the robot position (x,z,y), robot velocity,
        and the state of each motor.
        :return: Observation vector
        :rtype: numpy array
        """
        motor_pos = self.robot.getPosition()
        motor_pos.append(self.robot.getVelocity()[0])

        self.distance_walked = max(self.distance_walked, self.robot.getPosition()[0])
        self.counter_logging += 1
        self.max_distance_walked = max(self.max_distance_walked, self.robot.getPosition()[0])
        # Logging on wandb. When using Ray we don't use the custom logging because Ray has it own
        # logging and produces compatibility errors of two instances of wandb.
        if self.counter_logging % self.save_period == 0 and self.time_window_train and not self.use_ray:
            wandb.log({"Episode distance walked": self.distance_walked,
                       "Current x position": self.robot.getPosition()[0],
                       "Global distance walked": self.max_distance_walked
                       })

        motor_pos.extend([i for i in self.motor_position_arr])
        # Shift the last 10 observations by one, on the time window of 5
        self.obs[:-self.num_obs] = self.obs[self.num_obs:]
        self.obs[-self.num_obs:] = motor_pos

        return np.array(self.obs)

    def get_reward(self, action):
        """
        Calculate the reward of the agent. Reward the agent when moving forward on the x-axis, but
        reward the agent based on the distance it moved from its previous position.
        :return: Reward value
        :rtype: float
        """
        reward = 2.5 * self.robot.getPosition()[0] + self.robot.getPosition()[0] - self.prev_pos

        if self.counter_logging % self.save_period == 0 and self.time_window_train and not self.use_ray:
            wandb.log({"reward": reward,
                       "reward-1term-weight-pos": 2.5 * self.robot.getPosition()[0],
                       "reward-2term-diff-possition": self.robot.getPosition()[0] - self.prev_pos
                       })

        self.prev_pos = self.robot.getPosition()[0]
        return reward

    def is_done(self):
        """
        This method checks the termination criteria for each episode.
        If the criteria are satisfied it returns True otherwise it returns False.
        :return: Whether the termination criteria have been met.
        :rtype: bool
        """
        # Robot has fallen
        if self.robot.getPosition()[2] < 0.5:
            return True
        # Robot has walked out of the box
        if self.robot.getPosition()[0] > 15:
            return True
        return False

    def reset(self):
        """
        This method overrides reset in SupervisorEnv to reset a few variables.
        :return: observation provided by the following get_default_observation()
        """
        self.prev_pos = 0
        self.obs = np.zeros(self.time_window * self.num_obs)
        self.distance_walked = 0
        return super().reset()

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero
        vector in the shape of the observation space.
        :return: Starting observation zero vector
        :rtype: ndarray
        """
        return np.zeros(self.observation_space.shape[0])

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be
        executed as float numbers denoting the action for each motor.
        The corresponding motor_position value is applied at each motor.
        :param action: The list that contains the action values
        :type action: list of floats
        """
        motor_indexes = [0, 1, 2, 3, 4, 5]

        for i, j in zip(motor_indexes, action):
            self.motor_position_arr[i] += j
            self.motor_list[i].setVelocity(self.motor_velocity)
            self.motor_list[i].setPosition(float(j))

    def setup_agent(self):
        """
        This method initializes the 6 (leg) motors,
        storing the references inside a list and setting the starting
        positions and velocities.
        """
        # Get the motors names
        # We can uncomment the following list and make some modifications to control all motors.
        # ['Head', 'LeftAnkle', 'LeftArm', 'LeftCrus', 'LeftElbow', 'LeftFemur', 'LeftFemurHead1', 'LeftFemurHead2',
        #  'LeftFoot', 'LeftForearm', 'LeftShoulder', 'RightAnkle', 'RightArm', 'RightCrus', 'RightElbow', 'RightFemur',
        #  'RightFemurHead1', 'RightFemurHead2', 'RightFoot', 'RightForearm', 'RightShoulder', 'Waist']
        motor_names = ['LeftAnkle', 'LeftCrus', 'LeftFemur', 'RightAnkle', 'RightCrus', 'RightFemur']

        self.motor_list = []
        for motor_name in motor_names:
            motor = self.getDevice(motor_name)	 # Get the motor handle
            motor.setPosition(float('inf'))  # Set starting position
            motor.setVelocity(0.0)  # Zero out starting velocity
            self.motor_list.append(motor)  # Append motor to motor_list

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
