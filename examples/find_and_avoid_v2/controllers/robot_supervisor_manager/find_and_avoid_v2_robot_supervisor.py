import random
from warnings import warn
import numpy as np
from gym.spaces import Box, Discrete
from deepbots.supervisor import RobotSupervisorEnv
from utilities import normalize_to_range, get_distance_from_target, get_angle_from_target
from controller import Supervisor, Keyboard


class FindAndAvoidV2RobotSupervisor(RobotSupervisorEnv):
    """
    This is the updated and expanded second version of find and avoid.
    For this example, a simple custom differential drive robot is used, equipped with
    forward-facing distance and touch sensors. The goal is to navigate to a target by modifying
    the speeds of the left and right motor, while avoiding obstacles using the distance and touch sensors.
    The agent observes its distance to the target as well as its relative facing angle, its motor speeds, the touch
    and distance sensor values and its latest action. The observation can also be augmented with observations
    from earlier steps.

    Distance sensors scheme
    Sensor indices/positions:   [0,   1,    2,    3,     4,    5,     6,      7,     8,    9,     10,   11,   12]
    Frontal sensors slice:                              [4:         frontal         :9]
    Left/right sensors slices:  [0:         left        :5]                         [8:        right         :13]

    :param description: A description that can be saved in an exported file
    :type description: str
    :param maximum_episode_steps: The maximum steps per episode before timeout reset
    :type maximum_episode_steps: int
    :param step_window: How many steps of observations to add in the observation window, defaults to 1
    :type step_window: int, optional
    :param seconds_window: How many seconds of observations to add in the observation window, defaults to 1
    :type seconds_window: int, optional
    :param add_action_to_obs: Whether to add the latest action one-hot vector to the observation, defaults to True
    :type add_action_to_obs: bool, optional
    :param reset_on_collisions: On how many steps of collisions to reset, defaults to 0
    :type reset_on_collisions: int, optional
    :param manual_control: Whether to override agent actions with user keyboard control, defaults to False
    :type manual_control: bool, optional
    :param on_target_threshold: The threshold under which the robot is considered on target, defaults to 0.1
    :type on_target_threshold: float, optional
    :param max_ds_range: The maximum range of the distance sensors in cm, defaults to 100.0
    :type max_ds_range: float, optional
    :param ds_type: The type of distance sensors to use, can be either "generic" or "sonar", defaults to "generic"
    :type ds_type: str, optional
    :param ds_n_rays: The number of rays per sensor, defaults to 1
    :type ds_n_rays: int, optional
    :param ds_aperture: The angle at which the rays of each sensor are spread, defaults to 0.1 radians
    :type ds_aperture: float, optional
    :param ds_resolution: Minimum resolution of the sensors, the minimum change it can read, defaults to -1 (infinite)
    :type ds_resolution: float, optional
    :param ds_noise: The percentage of gaussian noise to add to the distance sensors, defaults to 0.0
    :type ds_noise: float, optional
    :param ds_denial_list: The list of distance sensor indices to disable, defaults to None
    :type ds_denial_list: list, optional
    :param target_distance_weight: The target distance reward weight, defaults to 1.0
    :type target_distance_weight: float, optional
    :param target_angle_weight: The target angle reward weight, defaults to 1.0
    :type target_angle_weight: float, optional
    :param dist_sensors_weight: The distance sensors reward weight, defaults to 1.0
    :type dist_sensors_weight: float, optional
    :param smoothness_weight: The smoothness reward weight, defaults to 1.0
    :type smoothness_weight: float, optional
    :param speed_weight: The speed reward weight, defaults to 1.0
    :type speed_weight: float, optional
    :param target_reach_weight: The target reach reward weight, defaults to 1.0
    :type target_reach_weight: float, optional
    :param collision_weight: The collision reward weight, defaults to 1.0
    :type collision_weight: float, optional
    :param map_width: The map width, defaults to 7
    :type map_width: int, optional
    :param map_height: The map height, defaults to 7
    :type map_height: int, optional
    :param cell_size: The cell size, defaults to None, [0.5, 0.5]
    :type cell_size: list, optional
    :param seed: The random seed, defaults to None
    :type seed: int, optional
    """

    def __init__(self, description, maximum_episode_steps, step_window=1, seconds_window=0, add_action_to_obs=True,
                 reset_on_collisions=0, manual_control=False, on_target_threshold=0.1,
                 max_ds_range=100.0, ds_type="generic", ds_n_rays=1, ds_aperture=0.1,
                 ds_resolution=-1, ds_noise=0.0, ds_denial_list=None,
                 target_distance_weight=1.0, target_angle_weight=1.0, dist_sensors_weight=1.0,
                 target_reach_weight=1.0, collision_weight=1.0, smoothness_weight=1.0, speed_weight=1.0,
                 map_width=7, map_height=7, cell_size=None, seed=None):
        super().__init__()

        ################################################################################################################
        # General
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.experiment_desc = description
        self.manual_control = manual_control

        # Viewpoint stuff used to reset camera position
        self.viewpoint = self.getFromDef("VIEWPOINT")
        self.viewpoint_position = self.viewpoint.getField("position").getSFVec3f()
        self.viewpoint_orientation = self.viewpoint.getField("orientation").getSFRotation()

        # Keyboard control
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        if ds_denial_list is None:
            self.ds_denial_list = []
        else:
            self.ds_denial_list = ds_denial_list
        ################################################################################################################
        # Robot setup

        # Set up various robot components
        self.robot = self.getSelf()
        self.number_of_distance_sensors = 13  # Fixed according to ds that exist on robot

        # Set up gym observation and action spaces
        # The action mapping is as follows:
        # - 0: Increase both motor speeds, forward action
        # - 1: Decrease both motor speeds, backward action
        # - 2: Increase right motor speed, decrease left motor speed, turn left
        # - 3: Increase left motor speed, decrease right motor speed, turn right
        # - 4: No change in motor speeds, no action
        self.action_space = Discrete(5)

        self.add_action_to_obs = add_action_to_obs
        self.step_window = step_window
        self.seconds_window = seconds_window
        self.obs_list = []
        # Set up observation low values
        # Distance to target, angle to target, motor speed left, motor speed right, touch left, touch right
        single_obs_low = [0.0, -1.0, -1.0, -1.0, 0.0, 0.0]
        # Add action one-hot vector
        if self.add_action_to_obs:
            single_obs_low.extend([0.0 for _ in range(self.action_space.n)])
        # Append distance sensor values
        single_obs_low.extend([0.0 for _ in range(self.number_of_distance_sensors)])

        # Set up corresponding observation high values
        single_obs_high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if self.add_action_to_obs:
            single_obs_high.extend([1.0 for _ in range(self.action_space.n)])
        single_obs_high.extend([1.0 for _ in range(self.number_of_distance_sensors)])

        # Expand sizes depending on step window and seconds window
        self.single_obs_size = len(single_obs_low)
        obs_low = []
        obs_high = []
        for _ in range(self.step_window + self.seconds_window):
            obs_low.extend(single_obs_low)
            obs_high.extend(single_obs_high)
            self.obs_list.extend([0.0 for _ in range(self.single_obs_size)])
        # Memory is used for creating the windows in get_observation()
        self.obs_memory = [[0.0 for _ in range(self.single_obs_size)]
                           for _ in range((self.seconds_window * int(np.ceil(1000 / self.timestep))) +
                                          self.step_window)]
        self.observation_counter_limit = int(np.ceil(1000 / self.timestep))
        self.observation_counter = self.observation_counter_limit

        # Finally initialize space
        self.observation_space = Box(low=np.array(obs_low),
                                     high=np.array(obs_high),
                                     dtype=np.float64)

        # Set up sensors
        self.distance_sensors = []
        self.ds_max = []
        self.ds_type = ds_type
        self.ds_n_rays = ds_n_rays
        self.ds_aperture = ds_aperture
        self.ds_resolution = ds_resolution
        self.ds_noise = ds_noise
        # The minimum distance sensor thresholds, under which there is an obstacle obstructing forward movement
        # Note that these values are highly dependent on how the sensors are placed on the robot
        self.ds_thresholds = [8.0, 8.0, 8.0, 10.15, 14.7, 13.15,
                              12.7,
                              13.15, 14.7, 10.15, 8.0, 8.0, 8.0]
        # Loop through the ds_group node to set max sensor values, initialize the devices, set the type, etc.
        robot_children = self.robot.getField("children")
        for childNodeIndex in range(robot_children.getCount()):
            robot_child = robot_children.getMFNode(childNodeIndex)  # NOQA
            if robot_child.getTypeName() == "Group":
                ds_group = robot_child.getField("children")
                for i in range(self.number_of_distance_sensors):
                    self.distance_sensors.append(self.getDevice(f"distance sensor({str(i)})"))
                    self.distance_sensors[-1].enable(self.timestep)  # NOQA
                    ds_node = ds_group.getMFNode(i)
                    ds_node.getField("lookupTable").setMFVec3f(4, [max_ds_range / 100.0, max_ds_range])
                    ds_node.getField("lookupTable").setMFVec3f(3, [0.75 * max_ds_range / 100.0, 0.75 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("lookupTable").setMFVec3f(2, [0.5 * max_ds_range / 100.0, 0.5 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("lookupTable").setMFVec3f(1, [0.25 * max_ds_range / 100.0, 0.25 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("type").setSFString(self.ds_type)
                    ds_node.getField("numberOfRays").setSFInt32(self.ds_n_rays)
                    ds_node.getField("aperture").setSFFloat(self.ds_aperture)
                    ds_node.getField("resolution").setSFFloat(self.ds_resolution)
                    self.ds_max.append(max_ds_range)  # NOQA

        # Touch sensors are used to determine when the robot collides with an obstacle
        self.touch_sensor_left = self.getDevice("touch sensor left")
        self.touch_sensor_left.enable(self.timestep)  # NOQA
        self.touch_sensor_right = self.getDevice("touch sensor right")
        self.touch_sensor_right.enable(self.timestep)  # NOQA

        # Set up motors
        self.left_motor = self.getDevice("left_wheel")
        self.right_motor = self.getDevice("right_wheel")
        self.motor_speeds = [0.0, 0.0]
        self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])

        # Grab target node
        self.target = self.getFromDef("TARGET")
        self.target.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])

        ################################################################################################################
        # Set up miscellaneous
        # Various robot and target metrics, for the current step and the previous step
        self.on_target_threshold = on_target_threshold  # Threshold that defines whether robot is considered "on target"
        self.initial_target_distance = 0.0
        self.initial_target_angle = 0.0
        self.current_tar_d = 0.0  # Distance to target
        self.previous_tar_d = 0.0
        self.current_tar_a = 0.0  # Angle to target in respect to the facing angle of the robot
        self.previous_tar_a = 0.0
        self.current_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]  # Latest distance sensor values
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.current_touch_sensors = [0.0, 0.0]
        self.current_position = [0, 0]  # World position
        self.previous_position = [0, 0]
        self.current_rotation = 0.0  # Facing angle
        self.previous_rotation = 0.0
        self.current_rotation_change = 0.0  # Latest facing angle change
        self.previous_rotation_change = 0.0

        # Various episode/training metrics, etc.
        self.current_timestep = 0
        self.collisions_counter = 0  # Counter of collisions during the episode
        self.reset_on_collisions = reset_on_collisions  # Upper limit of number of collisions before reset
        self.maximum_episode_steps = maximum_episode_steps  # Steps before timeout
        self.done_reason = ""  # Used to terminate episode and print the reason the episode is done
        self.reset_count = -1  # How many resets of the env overall, -1 to disregard the very first reset
        self.reach_target_count = 0  # How many times the target was reached
        self.collision_termination_count = 0  # How many times an episode was terminated due to collisions
        self.timeout_count = 0  # How many times an episode timed out
        self.min_distance_reached = float("inf")  # The current episode minimum distance to target reached
        self.min_dist_reached_list = []  # Used to store latest minimum distances reached, used as training metric
        self.smoothness_list = []  # Used to store the episode smoothness rewards, used as training metric
        self.episode_accumulated_reward = 0.0  # The reward accumulated in the current episode
        self.touched_obstacle_left = False
        self.touched_obstacle_right = False
        self.mask = [True for _ in range(self.action_space.n)]  # The action mask
        self.trigger_done = False  # Used to trigger the done condition
        self.just_reset = True  # Whether the episode was just reset

        # Dictionary holding the weights for the various reward components
        self.reward_weight_dict = {"dist_tar": target_distance_weight, "ang_tar": target_angle_weight,
                                   "dist_sensors": dist_sensors_weight, "tar_reach": target_reach_weight,
                                   "collision": collision_weight, "smoothness_weight": smoothness_weight,
                                   "speed_weight": speed_weight}

        ################################################################################################################
        # Map stuff
        self.map_width, self.map_height = map_width, map_height
        if cell_size is None:
            self.cell_size = [0.5, 0.5]
        # Center map to (0, 0)
        origin = [-(self.map_width // 2) * self.cell_size[0], (self.map_height // 2) * self.cell_size[1]]
        self.map = Grid(self.map_width, self.map_height, origin, self.cell_size)

        # Obstacle references and starting positions used to reset them
        self.all_obstacles = []
        self.all_obstacles_starting_positions = []
        for childNodeIndex in range(self.getFromDef("OBSTACLES").getField("children").getCount()):
            child = self.getFromDef("OBSTACLES").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_obstacles.append(child)
            self.all_obstacles_starting_positions.append(child.getField("translation").getSFVec3f())

        # Wall references
        self.walls = [self.getFromDef("WALL_1"), self.getFromDef("WALL_2")]
        self.walls_starting_positions = [self.getFromDef("WALL_1").getField("translation").getSFVec3f(),
                                         self.getFromDef("WALL_2").getField("translation").getSFVec3f()]

        # Path node references and starting positions used to reset them
        self.all_path_nodes = []
        self.all_path_nodes_starting_positions = []
        for childNodeIndex in range(self.getFromDef("PATH").getField("children").getCount()):
            child = self.getFromDef("PATH").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_path_nodes.append(child)
            self.all_path_nodes_starting_positions.append(child.getField("translation").getSFVec3f())

        self.current_difficulty = {}
        self.number_of_obstacles = 0  # The number of obstacles to use, set from set_difficulty method
        if self.number_of_obstacles > len(self.all_obstacles):
            warn(f"\n \nNumber of obstacles set is greater than the number of obstacles that exist in the "
                 f"world ({self.number_of_obstacles} > {len(self.all_obstacles)}).\n"
                 f"Number of obstacles is set to {len(self.all_obstacles)}.\n ")
            self.number_of_obstacles = len(self.all_obstacles)

        self.path_to_target = []  # The map cells of the path
        # The min and max (manhattan) distances of the target length allowed, set from set_difficulty method
        self.min_target_dist = 1
        self.max_target_dist = 1

    def set_reward_weight_dict(self, target_distance_weight, target_angle_weight, dist_sensors_weight,
                               target_reach_weight, collision_weight, smoothness_weight, speed_weight):
        """
        Utility function to help change the reward weight dictionary on runtime.
        """
        self.reward_weight_dict = {"dist_tar": target_distance_weight, "ang_tar": target_angle_weight,
                                   "dist_sensors": dist_sensors_weight, "tar_reach": target_reach_weight,
                                   "collision": collision_weight, "smoothness_weight": smoothness_weight,
                                   "speed_weight": speed_weight}

    def set_maximum_episode_steps(self, new_value):
        """
        This is required to be able to change the value of timeout steps for sb3 to register it.
        """
        self.maximum_episode_steps = new_value

    def set_difficulty(self, difficulty_dict, key=None):
        """
        Sets the difficulty and corresponding variables with a difficulty dictionary provided.

        :param difficulty_dict: A dictionary containing the difficulty information, e.g.
            {"type": "t", "number_of_obstacles": X, "min_target_dist": Y, "max_target_dist": Z}, where type can be
            "random" or "corridor".
        :type difficulty_dict: dict
        :param key: The key of the difficulty dictionary, if provided, the method prints it along with the
            difficulty dict, defaults to None
        :type key: str, optional
        """
        self.current_difficulty = difficulty_dict
        self.number_of_obstacles = difficulty_dict["number_of_obstacles"]
        self.min_target_dist = difficulty_dict["min_target_dist"]
        self.max_target_dist = difficulty_dict["max_target_dist"]
        if key is not None:
            print(f"Changed difficulty to: {key}, {difficulty_dict}")
        else:
            print("Changed difficulty to:", difficulty_dict)

    def get_action_mask(self):
        """
        Returns the mask for the current state. The mask is a list of bools where each element corresponds to an
        action, and if the bool is False the corresponding action is masked, i.e. disallowed.
        Action masking allows the agent to perform certain actions under certain conditions, disallowing illogical
        decisions.

        Mask is modified first by the touch sensors, and if no collisions are detected, secondly by
        the distance sensors.

        The action mapping is as follows:
        - 0: Increase both motor speeds, forward action
        - 1: Decrease both motor speeds, backward action
        - 2: Increase right motor speed, decrease left motor speed, turn left
        - 3: Increase left motor speed, decrease right motor speed, turn right
        - 4: No change in motor speeds, no action

        :return: The action mask list of bools
        :rtype: list of booleans
        """
        self.mask = [True for _ in range(self.action_space.n)]
        # Mask backward action that will cause the agent to move backwards by default
        if self.motor_speeds[0] <= 0.0 and self.motor_speeds[1] <= 0.0:
            self.mask[1] = False

        # Create various flag lists for the distance sensors
        # Whether any sensor is reading under its minimum threshold, and calculate and store how much
        reading_under_threshold = [0.0 for _ in range(self.number_of_distance_sensors)]
        # Whether there is any obstacle under half the max range of the distance sensors
        detecting_obstacle = [False for _ in range(self.number_of_distance_sensors)]
        # Whether there is an obstacle really close, i.e. under half the minimum threshold, in front
        front_under_half_threshold = False
        for i in range(len(self.current_dist_sensors)):
            if self.current_dist_sensors[i] <= self.ds_max[i] / 2:
                detecting_obstacle[i] = True
            # Sensor is reading under threshold, store how much under threshold it reads
            if self.current_dist_sensors[i] < self.ds_thresholds[i]:
                reading_under_threshold[i] = self.ds_thresholds[i] - self.current_dist_sensors[i]
                # Frontal sensor (index 4 to 8) is reading under half threshold
                if i in [4, 5, 6, 7, 8] and self.current_dist_sensors[i] < (self.ds_thresholds[i] / 2):
                    front_under_half_threshold = True
        # Split left and right slices to use later
        reading_under_threshold_left = reading_under_threshold[0:5]
        reading_under_threshold_right = reading_under_threshold[8:13]

        # First modify mask using the touch sensors as they are more important than distance sensors
        # Unmask backward and mask forward if a touch sensor is detecting collision
        if any(self.current_touch_sensors):
            self.mask[0] = False
            self.mask[1] = True
            # Set flags to keep masking/unmasking until robot is clear of obstacles
            # Distinguish between left and right touch
            if self.current_touch_sensors[0]:
                self.touched_obstacle_left = True
            if self.current_touch_sensors[1]:
                self.touched_obstacle_right = True
        # Not touching obstacles and can't detect obstacles with distance sensors,
        # can stop masking forward and unmasking backwards
        elif not any(reading_under_threshold):
            self.touched_obstacle_left = False
            self.touched_obstacle_right = False

        # Keep masking forward and unmasking backwards as long as a touched_obstacle flag is True
        if self.touched_obstacle_left or self.touched_obstacle_right:
            self.mask[0] = False
            self.mask[1] = True

            if self.touched_obstacle_left and not self.touched_obstacle_right:
                # Touched on left, mask left action, unmask right action
                self.mask[2] = False
                self.mask[3] = True
            if self.touched_obstacle_right and not self.touched_obstacle_left:
                # Touched on right, mask right action, unmask left action
                self.mask[3] = False
                self.mask[2] = True
        # If there are no touching obstacles, modify mask by distance sensors
        else:
            # Obstacles very close in front
            if front_under_half_threshold:
                # Mask forward
                self.mask[0] = False

            # Target is straight ahead, no obstacles close-by
            if not any(detecting_obstacle) and abs(self.current_tar_a) < 0.1:
                # Mask left and right turning
                self.mask[2] = self.mask[3] = False

            angle_threshold = 0.1
            # No obstacles on the right and target is on the right or
            # no obstacles on the right and obstacles on the left regardless of target direction
            if not any(reading_under_threshold_right):
                if self.current_tar_a <= - angle_threshold or any(reading_under_threshold_left):
                    self.mask[2] = False  # Mask left

            # No obstacles on the left and target is on the left or
            # no obstacles on the left and obstacles on the right regardless of target direction
            if not any(reading_under_threshold_left):
                if self.current_tar_a >= angle_threshold or any(reading_under_threshold_right):
                    self.mask[3] = False  # Mask right

            # Both left and right sensors are reading under threshold
            if any(reading_under_threshold_left) and any(reading_under_threshold_right):
                # Calculate the sum of how much each sensor's threshold is surpassed
                sum_left = sum(reading_under_threshold_left)
                sum_right = sum(reading_under_threshold_right)
                # If left side has obstacles closer than right
                if sum_left - sum_right < -5.0:
                    self.mask[2] = True  # Unmask left
                # If right side has obstacles closer than left
                elif sum_left - sum_right > 5.0:
                    self.mask[3] = True  # Unmask right
                # If left and right side have obstacles on roughly equal distances
                else:
                    # Enable touched condition
                    self.touched_obstacle_right = self.touched_obstacle_left = True
        return self.mask

    def get_observations(self, action=None):
        """
        This method returns the observation list of the agent.
        A single observation consists of the distance and angle to the target, the current motor speeds,
        the touch sensor values, the latest action represented by a one-hot vector,
        and finally the distance sensor values.

        All values are normalized in their respective ranges, where appropriate:
        - Distance is normalized to [0.0, 1.0]
        - Angle is normalized to [-1.0, 1.0]
        - Motor speeds are already constrained within [-1.0, 1.0]
        - Touch sensor values can only been 0 or 1
        - Distance sensor values are normalized to [1.0, 0.0]
          This is done so the input gets a large activation value when the sensor returns
          small values, i.e. an obstacle is close.

        All observations are held in a memory (self.obs_memory) and the current observation is augmented with
        self.step_window steps of the latest single observations and with self.seconds_window seconds of
        observations. This means that for self.step_window=2 and self.seconds_window=2, the observation
        is the latest two single observations plus an observation from 1 second in the past and an observation
        from 2 seconds in the past.

        :param action: The latest action, defaults to None to match signature of parent method
        :type action: int, optional
        :return: Observation list
        :rtype: list
        """
        if self.just_reset:
            self.previous_tar_d = self.current_tar_d
            self.previous_tar_a = self.current_tar_a
        # Add distance, angle, motor speeds
        obs = [normalize_to_range(self.current_tar_d, 0.0, self.initial_target_distance, 0.0, 1.0, clip=True),
               normalize_to_range(self.current_tar_a, -np.pi, np.pi, -1.0, 1.0, clip=True),
               self.motor_speeds[0], self.motor_speeds[1]]
        # Add touch sensor values
        obs.extend(self.current_touch_sensors)

        if self.add_action_to_obs:
            # Add action one-hot
            action_one_hot = [0.0 for _ in range(self.action_space.n)]
            try:
                action_one_hot[action] = 1.0
            except IndexError:
                pass
            obs.extend(action_one_hot)

        # Add distance sensor values
        ds_values = []
        for i in range(len(self.distance_sensors)):
            ds_values.append(normalize_to_range(self.current_dist_sensors[i], 0, self.ds_max[i], 1.0, 0.0))
        obs.extend(ds_values)

        self.obs_memory = self.obs_memory[1:]  # Drop oldest
        self.obs_memory.append(obs)  # Add the latest observation

        # Add the latest observations based on self.step_window
        dense_obs = ([self.obs_memory[i] for i in range(len(self.obs_memory) - 1,
                                                        len(self.obs_memory) - 1 - self.step_window, -1)])

        diluted_obs = []
        counter = 0
        for j in range(len(self.obs_memory) - 1 - self.step_window, 0, -1):
            counter += 1
            if counter >= self.observation_counter_limit - 1:
                diluted_obs.append(self.obs_memory[j])
                counter = 0
        self.obs_list = []
        for single_obs in diluted_obs:
            for item in single_obs:
                self.obs_list.append(item)
        for single_obs in dense_obs:
            for item in single_obs:
                self.obs_list.append(item)

        return self.obs_list

    def get_reward(self, action):
        """
        This method calculates the reward. The reward consists of various components that get weighted and added into
        the final reward that gets returned.

        -  The distance to target reward is firstly the negative normalized current to target creating a continuous
        reward based on how far the agent is from the target. Bonus reward is also added everytime a new minimum
        distance is achieved in the episode, giving an additional incentive moving along a path that closes the
        distance. This bonus reward is similar to the pit escape problem reward.
        - Reach target reward is 0.0, unless the current distance to target is under the on_target_threshold when it
        becomes 1.0 - 0.5 * (current_timestep/max_timesteps), and reset is triggered (done). This way when the target
        is reached, half the reward is scaled based on the time it took for the robot to reach the target.
        - If the current angle to target is over a certain threshold, the angle to target reward is calculated based
        on the latest change of the angle to target, if positive and over a small threshold, the reward is 1.0,
        if negative and under a small threshold, the reward is -1.0. If the absolute change is under the threshold, the
        reward is 0.0. If the current angle to target is under a certain threshold (i.e. facing the target),
        the agent is reward with 1.0.
        The threshold is pi/4 scaled by the normalized distance to target, i.e. the closer the robot is to the target,
        the more it is rewarded for turning towards it and the stricter is the threshold under which the robot is
        considered facing the target.
        - The distance sensor reward is calculated by first calculating a sum. For each sensor reading over the minimum
        ds threshold, +1 is added to the sum, otherwise -1 is added. Then the average is taken and normalized
        to [-1.0, 0.0], with different initial ranges depending on what type of sensor is used (generic or sonar).
        The final reward is scaled by the normalized distance to target. The closer the robot is to the target the less
        important this reward is.
        - Collision reward is 0.0, unless a or both the touch sensors detects a collision and it becomes -1.0.
        It also counts the number of collisions and triggers a reset (done) when the set limit
        reset_on_collisions is reached.
        - Smoothness reward is calculated based on the rotational speed of the robot. The higher the latest absolute
        facing angle change the closer the reward is to -1.0. The closer the value is to 0.0 (no turning at all), the
        closer the reward is to 1.0. The final reward is scaled by the normalized distance to target.
        The closer the robot is to the target the less important this reward is. Note the custom value for maximum
        rotational speed used for normalizing.
        - Speed reward is calculated based on the translational speed of the robot. The higher the latest distance moved
        is, the closer the reward is to 1.0. The closer the value is to 0.0 (no distance moved), the closer the reward
        is to -1.0. The final reward is scaled by the normalized distance to target. The closer the robot is to the
        target the less important this reward is. Note the custom value for maximum distance moved used for normalizing.

        Finally, the angle to target reward is zeroed out if there is a non-negative distance sensor reward or
        non-negative collision reward, to allow the agent to turn and move around obstacles without penalizing.

        All these rewards are multiplied with their corresponding weights taken from reward_weight_dict and summed into
        the final reward.

        :param action: The latest action
        :type action: int
        :return: The total step reward
        :rtype: float
        """
        # If episode was just reset, set previous and current target distance and angle
        if self.just_reset:
            self.previous_tar_d = self.current_tar_d = self.initial_target_distance
            self.previous_tar_a = self.current_tar_a = self.initial_target_angle
        ################################################################################################################
        # Distance to target rewards
        # Reward for decreasing distance to the target

        # The normalized current distance to target is used to scale rewards
        normalized_current_tar_d = normalize_to_range(self.current_tar_d,
                                                      0.0, self.initial_target_distance, 0.0, 1.0, clip=True)

        # Initial distance reward is minus the normalized distance to the target
        dist_tar_reward = -normalized_current_tar_d
        # If min distance is decreased, add +1 reward
        if round(self.current_tar_d, 4) - round(self.min_distance_reached, 4) < 0.0:
            dist_tar_reward += 1.0
            self.min_distance_reached = self.current_tar_d

        # Reward for reaching the target, i.e. decreasing the real distance under the threshold
        # Final reward is modified by the time it took to reach the target
        reach_tar_reward = 0.0
        if self.current_tar_d < self.on_target_threshold:
            reach_tar_reward = 1.0 - 0.5 * self.current_timestep / self.maximum_episode_steps
            self.done_reason = "reached target"  # This triggers termination of episode

        ################################################################################################################
        # Angle to target reward
        # Reward for decreasing angle to the target
        # If turning towards the target apply +1.0, if turning away apply -1.0
        if abs(self.current_tar_a) > (np.pi / 4) * normalized_current_tar_d:
            if round(abs(self.previous_tar_a) - abs(self.current_tar_a), 3) > 0.001:
                ang_tar_reward = 1.0
            elif round(abs(self.previous_tar_a) - abs(self.current_tar_a), 3) < -0.001:
                ang_tar_reward = -1.0
            else:
                ang_tar_reward = 0.0
        else:
            ang_tar_reward = 1.0
        ################################################################################################################
        # Obstacle avoidance rewards
        # Reward for distance sensors values
        dist_sensors_reward = 0
        for i in range(len(self.distance_sensors)):
            if self.current_dist_sensors[i] < self.ds_thresholds[i]:
                # If any sensor is under threshold add penalty
                dist_sensors_reward -= 1.0
            else:
                # Otherwise add reward
                dist_sensors_reward += 1.0
        dist_sensors_reward /= self.number_of_distance_sensors
        # Realistically not all sonar sensors can read under the thresholds set, if robot is boxed in
        #  with a wall in front, wall on the left and right, only the two left and two right-most sensors as
        #  well as the three frontal ones will read low values, resulting in an average reward of -0.077,
        #  which is normalized to -1.0.
        if self.ds_type == "sonar":
            dist_sensors_reward = round(normalize_to_range(dist_sensors_reward, -0.077, 1.0, -1.0, 0.0, clip=True), 4)
        elif self.ds_type == "generic":
            dist_sensors_reward = round(normalize_to_range(dist_sensors_reward, -1.0, 1.0, -1.0, 0.0, clip=True), 4)
        #  Final value is multiplied by current distance, so this reward gets less important, the closer
        #  the robot is to the target.
        dist_sensors_reward *= normalized_current_tar_d

        # Penalty for collisions
        # Check if the robot has collided with anything, assign negative reward
        collision_reward = 0.0
        if any(self.current_touch_sensors):
            self.collisions_counter += 1
            if self.collisions_counter >= self.reset_on_collisions - 1 and self.reset_on_collisions != -1:
                self.done_reason = "collision"  # This triggers termination of episode
            collision_reward = -1.0

        ################################################################################################################
        # Rewards for driving smoothly and at speed
        # Smoothness reward based on angular velocity, -1.0 for fast turning, 1.0 for no turning
        # Multiplied by the current distance to target, means that the farther away the robot is the more important
        #  it is to move smoothly. Near the target, violent turning maneuvers might be needed.
        smoothness_reward = round(
            -abs(normalize_to_range(self.current_rotation_change, -0.0183, 0.0183, -1.0, 1.0, clip=True)), 2)
        if not self.just_reset:
            self.smoothness_list.append(smoothness_reward)  # To use as metric
        smoothness_reward *= normalized_current_tar_d

        # Speed reward based on distance moved on last step. Obviously, straight movement produces better reward.
        # This also means that neutral turns are also penalized because the position is not changing.
        # Similar to smoothness reward, moving at speed is less important near the target.
        dist_moved = np.linalg.norm([self.current_position[0] - self.previous_position[0],
                                     self.current_position[1] - self.previous_position[1]])
        speed_reward = normalize_to_range(dist_moved, 0.0, 0.0012798, -1.0, 1.0)
        speed_reward *= normalized_current_tar_d

        ################################################################################################################
        # Reward modification based on whether there are obstacles detected nearby
        if dist_sensors_reward != 0.0 or any(self.current_touch_sensors):
            ang_tar_reward = 0.0

        ################################################################################################################
        # Total reward calculation
        weighted_dist_tar_reward = self.reward_weight_dict["dist_tar"] * dist_tar_reward
        weighted_ang_tar_reward = self.reward_weight_dict["ang_tar"] * ang_tar_reward
        weighted_dist_sensors_reward = self.reward_weight_dict["dist_sensors"] * dist_sensors_reward
        weighted_reach_tar_reward = self.reward_weight_dict["tar_reach"] * reach_tar_reward
        weighted_collision_reward = self.reward_weight_dict["collision"] * collision_reward
        weighted_smoothness_reward = self.reward_weight_dict["smoothness_weight"] * smoothness_reward
        weighted_speed_reward = self.reward_weight_dict["speed_weight"] * speed_reward

        # Add various weighted rewards together
        reward = (weighted_dist_tar_reward + weighted_ang_tar_reward + weighted_dist_sensors_reward +
                  weighted_collision_reward + weighted_reach_tar_reward + weighted_smoothness_reward +
                  weighted_speed_reward)

        self.episode_accumulated_reward += reward

        if self.just_reset:
            # For the first step in each episode (just reset), return 0.0 reward.
            return 0.0
        else:
            return reward

    def is_done(self):
        """
        Episode done triggers when the done_reason string is set which happens in the reward function, when the maximum
        number of collisions is reached or the target is reached. This method handles the episode termination on
        timeout and sets the done_reason string appropriately.

        :return: Whether the episode is done
        :rtype: bool
        """
        if self.done_reason != "":
            return True
        # Timeout
        if self.current_timestep >= self.maximum_episode_steps:
            self.done_reason = "timeout"
            return True
        return False

    def reset(self):
        """
        Resets the simulation physics and objects and re-initializes robot and target positions,
        along any other variables that need to be reset to their original values.

        The new map is created depending on difficulty, and viewpoint is reset.
        """
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))  # NOQA
        self.obs_memory = [[0.0 for _ in range(self.single_obs_size)]
                           for _ in range((self.seconds_window * int(np.ceil(1000 / self.timestep))) +
                                          self.step_window)]
        self.observation_counter = self.observation_counter_limit
        # Reset path and various values
        self.trigger_done = False
        self.path_to_target = None
        self.motor_speeds = [0.0, 0.0]
        self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])
        self.collisions_counter = 0

        # Set robot random rotation
        self.robot.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])

        # Randomize obstacles and target
        if self.current_difficulty["type"] == "random":
            while True:
                # Randomize robot and obstacle positions
                self.randomize_map("random")
                self.simulationResetPhysics()
                # Set the target in a valid position and find a path to it
                # and repeat until a reachable position has been found for the target
                self.path_to_target = self.get_random_path(add_target=True)
                if self.path_to_target is not None:
                    self.path_to_target = self.path_to_target[1:]  # Remove starting node
                    break
        elif self.current_difficulty["type"] == "corridor":
            while True:
                max_distance_allowed = 1
                # Randomize robot and obstacle positions
                self.randomize_map("corridor")
                self.simulationResetPhysics()
                # Set the target in a valid position and find a path to it
                # and repeat until a reachable position has been found for the target
                self.path_to_target = self.get_random_path(add_target=False)
                if self.path_to_target is not None:
                    self.path_to_target = self.path_to_target[1:]  # Remove starting node
                    break
                max_distance_allowed += 1
        self.place_path(self.path_to_target)
        self.just_reset = True

        # Reset viewpoint so it plays nice
        self.viewpoint.getField("position").setSFVec3f(self.viewpoint_position)
        self.viewpoint.getField("orientation").setSFRotation(self.viewpoint_orientation)

        # Finally, reset any other values and count any metrics
        self.reset_count += 1
        if self.done_reason != "":
            print(f"Reward: {self.episode_accumulated_reward}, steps: {self.current_timestep}, "
                  f"done reason:{self.done_reason}")
        if self.done_reason == "collision":
            self.collision_termination_count += 1
        elif self.done_reason == "reached target":
            self.reach_target_count += 1
        elif self.done_reason == "timeout":
            self.timeout_count += 1
        self.done_reason = ""
        self.current_timestep = 0
        self.initial_target_distance = get_distance_from_target(self.robot, self.target)
        self.initial_target_angle = get_angle_from_target(self.robot, self.target)
        self.min_dist_reached_list.append(self.min_distance_reached)
        self.min_distance_reached = self.initial_target_distance - 0.01
        self.episode_accumulated_reward = 0.0
        self.current_dist_sensors = [self.ds_max[i] for i in range(len(self.distance_sensors))]
        self.previous_dist_sensors = [self.ds_max[i] for i in range(len(self.distance_sensors))]
        self.current_touch_sensors = [0.0, 0.0]
        self.current_position = list(self.robot.getPosition()[:2])
        self.previous_position = list(self.robot.getPosition()[:2])
        self.current_rotation = self.get_robot_rotation()
        self.previous_rotation = self.get_robot_rotation()
        self.current_rotation_change = 0.0
        self.previous_rotation_change = 0.0
        self.current_tar_d = 0.0
        self.previous_tar_d = 0.0
        self.current_tar_a = 0.0
        self.previous_tar_a = 0.0
        self.touched_obstacle_left = False
        self.touched_obstacle_right = False
        self.mask = [True for _ in range(self.action_space.n)]
        return self.get_default_observation()

    def clear_smoothness_list(self):
        """
        Method used to trigger the cleaning of the smoothness_list on demand.
        """
        self.smoothness_list = []

    def clear_min_dist_reached_list(self):
        """
        Method used to trigger the cleaning of the min_dist_reached_list on demand.
        """
        self.min_dist_reached_list = []

    def get_default_observation(self):
        """
        Basic get_default_observation implementation that returns a zero vector
        in the shape of the observation space.
        :return: A list of zeros in shape of the observation space
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_robot_rotation(self):
        # Fix rotation vector, because Webots randomly flips Z
        temp_rot = self.robot.getField("rotation").getSFRotation()
        if temp_rot[2] < 0.0:
            return -temp_rot[3]
        else:
            return temp_rot[3]

    def update_current_metrics(self):
        """
        Updates any metric that needs to be updated in each step. It serves as a unified place for updating metrics
        used in various methods. This runs after each simulation step.
        """
        # Save previous values
        self.previous_tar_d = self.current_tar_d
        self.previous_tar_a = self.current_tar_a
        self.previous_dist_sensors = self.current_dist_sensors
        self.previous_position = self.current_position
        self.previous_rotation = self.current_rotation
        self.previous_rotation_change = self.current_rotation_change

        # Target distance and angle
        self.current_tar_d = get_distance_from_target(self.robot, self.target)
        self.current_tar_a = get_angle_from_target(self.robot, self.target)

        # Get current position
        self.current_position = list(self.robot.getPosition()[:2])

        # Get current rotation
        self.current_rotation = self.get_robot_rotation()
        # To get rotation change we need to make sure there's not a big change from -pi to pi
        if self.current_rotation * self.previous_rotation < 0.0:
            self.current_rotation_change = self.previous_rotation_change
        else:
            self.current_rotation_change = self.current_rotation - self.previous_rotation

        # Get all distance sensor values
        self.current_dist_sensors = []  # Values are in range [0, self.ds_max]
        for ds in self.distance_sensors:
            self.current_dist_sensors.append(ds.getValue())  # NOQA

        # Deprive robot of distance sensors
        # Distance sensors whose index is in the denial list get their value overwritten with the max value
        for i in self.ds_denial_list:
            self.current_dist_sensors[i] = self.ds_max[i]

        # Get both touch sensor values
        self.current_touch_sensors = [self.touch_sensor_left.getValue(), self.touch_sensor_right.getValue()]  # NOQA

    def step(self, action):
        """
        Step override method which slightly modifies the parent.
        It applies the previous action, steps the simulation, updates the metrics with new values and then
        gets the new observation, reward, done flag and info and returns them.

        :param action: The action to perform
        :type action: int
        :return: new observation, reward, done flag, info
        :rtype: tuple
        """
        action = self.apply_action(action)

        if super(Supervisor, self).step(self.timestep) == -1:  # NOQA
            exit()

        self.update_current_metrics()
        self.current_timestep += 1

        obs = self.get_observations(action)
        rew = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()

        if self.just_reset:
            self.just_reset = False

        return (
            obs,
            rew,
            done,
            info
        )

    def apply_action(self, action):
        """
        This method gets an integer action value [0, 1, ...] where each value corresponds to an action.

        The integer-action mapping is as follows:
        - 0: Increase both motor speeds, forward action
        - 1: Decrease both motor speeds, backward action
        - 2: Increase right motor speed, decrease left motor speed, turn left
        - 3: Increase left motor speed, decrease right motor speed, turn right
        - 4: No change in motor speeds, no action

        This method also incorporates the keyboard control and if the user presses any of the
        control buttons that correspond to the aforementioned actions.

        The key-action mapping is as follows:
        - W: Increase both motor speeds, forward action
        - S: Decrease both motor speeds, backward action
        - A: Increase right motor speed, decrease left motor speed, turn left
        - D: Increase left motor speed, decrease right motor speed, turn right
        - X: Set motor speeds to zero, stop

        More keys are used to print helpful information:
        - O: Print latest observation
        - R: Print latest reward
        - M: Print latest action mask and current motor speeds

        Finally, the motor speeds are clipped to [-1, 1] and applied to the motors.

        :param action: The action to execute
        :type action: int
        :return: The action executed
        :rtype: int
        """
        key = self.keyboard.getKey()
        if key == ord("O"):  # Print latest observation
            print(self.obs_memory[-1])
        if key == ord("R"):  # Print latest reward
            print(self.get_reward(action))
        if key == ord("M"):  # Print current mask
            names = ["Forward", "Backward", "Left", "Right", "No action"]
            print([names[i] for i in range(len(self.mask)) if self.mask[i]])
            print(self.motor_speeds)

        if self.manual_control:
            action = 4
        if key == ord("W") and self.mask[0]:  # Increase both motor speeds
            action = 0
        if key == ord("S") and self.mask[1]:  # Decrease both motor speeds
            action = 1
        if key == ord("A") and self.mask[2]:  # Increase right motor speed, decrease left motor speed, turn left
            action = 2
        if key == ord("D") and self.mask[3]:  # Decrease right motor speed, increase left motor speed, turn right
            action = 3
        if key == ord("X"):  # No action
            action = 4
            self.motor_speeds = [0.0, 0.0]

        if action == 0:  # Increase both motor speeds
            if self.motor_speeds[0] < 1.0:
                self.motor_speeds[0] += 0.25
            if self.motor_speeds[1] < 1.0:
                self.motor_speeds[1] += 0.25
        elif action == 1:  # Decrease both motor speeds
            if self.motor_speeds[0] > -1.0:
                self.motor_speeds[0] -= 0.25
            if self.motor_speeds[1] > -1.0:
                self.motor_speeds[1] -= 0.25
        elif action == 2:  # Increase right motor speed, decrease left motor speed, turn left
            if self.motor_speeds[0] > -1.0:
                self.motor_speeds[0] -= 0.25
            if self.motor_speeds[1] < 1.0:
                self.motor_speeds[1] += 0.25
        elif action == 3:  # Decrease right motor speed, increase left motor speed, turn right
            if self.motor_speeds[0] < 1.0:
                self.motor_speeds[0] += 0.25
            if self.motor_speeds[1] > -1.0:
                self.motor_speeds[1] -= 0.25
        elif action == 4:  # No action
            pass

        self.motor_speeds = np.clip(self.motor_speeds, -1.0, 1.0)
        # Apply motor speeds
        self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])
        return action

    def set_velocity(self, v_left, v_right):
        """
        Sets the two motor velocities.
        :param v_left: velocity value for left motor
        :type v_left: float
        :param v_right: velocity value for right motor
        :type v_right: float
        """
        self.left_motor.setPosition(float('inf'))  # NOQA
        self.right_motor.setPosition(float('inf'))  # NOQA
        self.left_motor.setVelocity(v_left)  # NOQA
        self.right_motor.setVelocity(v_right)  # NOQA

    def get_info(self):
        """
        Returns the reason the episode is done when the episode terminates.

        :return: Dictionary containing information
        :rtype: dict
        """
        if self.done_reason != "":
            return {"done_reason": self.done_reason}
        else:
            return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render.

        :param mode: defaults to 'human'
        :type mode: str, optional
        """
        print("render() is not used")

    def export_parameters(self, path,
                          net_arch, gamma, gae_lambda, target_kl, vf_coef, ent_coef, n_steps, batch_size):
        """
        Exports all parameters that define the environment/experiment setup.

        :param path: The path to save the export
        :type path: str
        :param net_arch: The network architectures, e.g. dict(pi=[1024, 512, 256], vf=[2048, 1024, 512])
        :type net_arch: dict with two lists
        :param gamma: The gamma value
        :type gamma: float
        :param gae_lambda: The GAE lambda value
        :type gae_lambda: float
        :param target_kl: The target_kl value
        :type target_kl: float
        :param vf_coef: The vf_coef value
        :type vf_coef: float
        :param ent_coef: The ent_coef value
        :type ent_coef: float
        :param n_steps: Number of steps between each training session for sb3
        :type n_steps: int
        :param batch_size: The batch size used during training
        :type batch_size: int
        """
        import json
        param_dict = {"experiment_description": self.experiment_desc,
                      "seed": self.seed,
                      "n_steps:": n_steps,
                      "batch_size": batch_size,
                      "maximum_episode_steps": self.maximum_episode_steps,
                      "add_action_to_obs": self.add_action_to_obs,
                      "step_window": self.step_window,
                      "seconds_window": self.seconds_window,
                      "ds_params": {
                          "max range": self.ds_max,
                          "type": self.ds_type,
                          "rays": self.ds_n_rays,
                          "aperture": self.ds_aperture,
                          "resolution": self.ds_resolution,
                          "noise": self.ds_noise,
                          "minimum thresholds": self.ds_thresholds},
                      "rewards_weights": self.reward_weight_dict,
                      "map_width": self.map_width, "map_height": self.map_height, "cell_size": self.cell_size,
                      "difficulty": self.current_difficulty,
                      "ppo_params": {
                          "net_arch": net_arch,
                          "gamma": gamma,
                          "gae_lambda": gae_lambda,
                          "target_kl": target_kl,
                          "vf_coef": vf_coef,
                          "ent_coef": ent_coef,
                      }
                      }
        with open(path, 'w') as fp:
            json.dump(param_dict, fp, indent=4)

    ####################################################################################################################
    # Map-related methods below

    def remove_objects(self):
        """
        Removes objects from arena, by setting their original translations and rotations.
        """
        for object_node, starting_pos in zip(self.all_obstacles, self.all_obstacles_starting_positions):
            object_node.getField("translation").setSFVec3f(starting_pos)
            object_node.getField("rotation").setSFRotation([0, 0, 1, 0])
        for path_node, starting_pos in zip(self.all_path_nodes, self.all_path_nodes_starting_positions):
            path_node.getField("translation").setSFVec3f(starting_pos)
            path_node.getField("rotation").setSFRotation([0, 0, 1, 0])
        for wall_node, starting_pos in zip(self.walls, self.walls_starting_positions):
            wall_node.getField("translation").setSFVec3f(starting_pos)
            wall_node.getField("rotation").setSFRotation([0, 0, 1, -1.5708])

    def randomize_map(self, type_="random"):
        """
        Randomizes the obstacles on the map, by first removing all the objects and emptying the grid map.
        Then, based on the type_ argument provided, places the set number of obstacles in various random configurations.

        - "random": places the number_of_obstacles in free positions on the map and randomizes their rotation
        - "corridor": creates a corridor placing the robot at the start and the target at a distance along the corridor.
            It then places the obstacles on each row along the corridor between the target and the robot, making sure
            there is a valid path, i.e. consecutive rows should have free cells either diagonally or in the same column

        :param type_: The type of randomization, either "random" or "corridor", defaults to "random"
        :type type_: str, optional
        """
        self.remove_objects()
        self.map.empty()
        robot_z = 0.0399261  # Custom z value for the robot to avoid physics issues

        if type_ == "random":
            self.map.add_random(self.robot, robot_z)  # Add robot in a random position
            for obs_node in random.sample(self.all_obstacles, self.number_of_obstacles):
                self.map.add_random(obs_node)
                obs_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])
        elif type_ == "corridor":
            # Add robot to starting position
            self.map.add_cell((self.map_width - 1) // 2, self.map_height - 1, self.robot, robot_z)
            robot_coordinates = [(self.map_width - 1) // 2, self.map_height - 1]
            # Limit the provided min, max target distances
            if self.max_target_dist > self.map_height - 1:
                print(f"max_target_dist set out of range, setting to: {min(self.max_target_dist, self.map_height - 1)}")
            if self.min_target_dist > self.map_height - 1:
                print(f"min_target_dist set out of range, setting to: {min(self.min_target_dist, self.map_height - 1)}")
            # Get true min max target positions
            min_target_pos = self.map_height - 1 - min(self.max_target_dist, self.map_height - 1)
            max_target_pos = self.map_height - 1 - min(self.min_target_dist, self.map_height - 1)
            if min_target_pos == max_target_pos:
                target_y = min_target_pos
            else:
                target_y = random.randint(min_target_pos, max_target_pos)
            # Finally add target
            self.map.add_cell(robot_coordinates[0], target_y, self.target)

            # If there is space between target and robot, add obstacles
            if abs(robot_coordinates[1] - target_y) > 1:
                # We add two obstacles on each row between the target and robot so there is one free cell for the path
                # To generate the obstacle placements within the corridor, we need to make sure that there is
                # a free path within the corridor that leads from one row to the next.
                # This means we need to avoid the case where there's a free place in the first column and on the next
                # row the free place is in the third row.
                def add_two_obstacles():
                    col_choices = [robot_coordinates[0] + i for i in range(-1, 2, 1)]
                    random_col_1_ = random.choice(col_choices)
                    col_choices.remove(random_col_1_)
                    random_col_2_ = random.choice(col_choices)
                    col_choices.remove(random_col_2_)
                    return col_choices[0], random_col_1_, random_col_2_

                max_obstacles = (abs(robot_coordinates[1] - target_y) - 1) * 2
                random_sample = random.sample(self.all_obstacles, min(max_obstacles, self.number_of_obstacles))
                prev_free_col = 0
                for row_coord, obs_node_index in \
                        zip(range(target_y + 1, robot_coordinates[1]), range(0, len(random_sample), 2)):
                    # For each row between the robot and the target, add 2 obstacles
                    if prev_free_col == 0:
                        # If previous free column is the center one, any positions for the new row are valid
                        prev_free_col, random_col_1, random_col_2 = add_two_obstacles()
                    else:
                        # If previous free column is not the center one, then the new free one cannot be
                        # on the other side
                        current_free_col, random_col_1, random_col_2 = add_two_obstacles()
                        while abs(prev_free_col - current_free_col) == 2:
                            current_free_col, random_col_1, random_col_2 = add_two_obstacles()
                        prev_free_col = current_free_col
                    self.map.add_cell(random_col_1, row_coord, random_sample[obs_node_index])
                    random_sample[obs_node_index].getField("rotation"). \
                        setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])
                    self.map.add_cell(random_col_2, row_coord, random_sample[obs_node_index + 1])
                    random_sample[obs_node_index + 1].getField("rotation"). \
                        setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])

            # Abuse the grid map and add wall objects as placeholder to limit path finding within the corridor
            for row_coord in range(target_y + 1, robot_coordinates[1]):
                self.map.add_cell(robot_coordinates[0] - 2, row_coord, self.walls[0])
                self.map.add_cell(robot_coordinates[0] + 2, row_coord, self.walls[1])
            new_position = [-0.75,
                            self.walls_starting_positions[0][1],
                            self.walls_starting_positions[0][2]]
            self.walls[0].getField("translation").setSFVec3f(new_position)
            new_position = [0.75,
                            self.walls_starting_positions[1][1],
                            self.walls_starting_positions[1][2]]
            self.walls[1].getField("translation").setSFVec3f(new_position)

    def get_random_path(self, add_target=True):
        """
        Returns a path to the target or None if path is not found. Based on the add_target flag it also places the
        target randomly at a certain manhattan min/max distance to the robot.

        :param add_target: Whether to also add the target before returning the path, defaults to True
        :type add_target: bool, optional
        """
        robot_coordinates = self.map.find_by_name("robot")
        if add_target:
            if not self.map.add_near(robot_coordinates[0], robot_coordinates[1],
                                     self.target,
                                     min_distance=self.min_target_dist, max_distance=self.max_target_dist):
                return None  # Need to re-randomize obstacles as add_near failed
        return self.map.bfs_path(robot_coordinates, self.map.find_by_name("target"))

    def place_path(self, path):
        """
        Place the path nodes (the small deepbots logos) on their proper places depending on the path generated.

        :param path: The path list
        :type path: list
        """
        for p, l in zip(path, self.all_path_nodes):
            self.map.add_cell(p[0], p[1], l)

    def find_dist_to_path(self):
        """
        This method is not currently used. It calculates the closest point and distance to the path,
        returning both.

        :return: The minimum distance to the path and the corresponding closest point on the path
        :rtype: tuple
        """

        def dist_to_line_segm(p, l1, l2):
            v = l2 - l1
            w = p - l1
            c1 = np.dot(w, v)
            if c1 <= 0:
                return np.linalg.norm(p - l1), l1
            c2 = np.dot(v, v)
            if c2 <= c1:
                return np.linalg.norm(p - l2), l2
            b = c1 / c2
            pb = l1 + b * v
            return np.linalg.norm(p - pb), pb

        np_path = np.array([self.map.get_world_coordinates(self.path_to_target[i][0], self.path_to_target[i][1])
                            for i in range(len(self.path_to_target))])
        robot_pos = np.array(self.robot.getPosition()[:2])

        if len(np_path) == 1:
            return np.linalg.norm(np_path[0] - robot_pos), np_path[0]

        min_distance = float('inf')
        closest_point = None
        for i in range(np_path.shape[0] - 1):
            edge = np.array([np_path[i], np_path[i + 1]])
            distance, point_on_line = dist_to_line_segm(robot_pos, edge[0], edge[1])
            min_distance = min(min_distance, distance)
            closest_point = point_on_line
        return min_distance, closest_point


####################################################################################################################
# Grid class used to create the random obstacle map


class Grid:
    """
    The grid map used to place all objects in the arena and find the paths.

    Partially coded by OpenAI's ChatGPT.
    """

    def __init__(self, width, height, origin, cell_size):
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.origin = origin
        self.cell_size = cell_size

    def size(self):
        return len(self.grid[0]), len(self.grid)

    def add_cell(self, x, y, node, z=None):
        if self.grid[y][x] is None and self.is_in_range(x, y):
            self.grid[y][x] = node
            if z is None:
                node.getField("translation").setSFVec3f(
                    [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], node.getPosition()[2]])
            else:
                node.getField("translation").setSFVec3f(
                    [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], z])
            return True
        return False

    def remove_cell(self, x, y):
        if self.is_in_range(x, y):
            self.grid[y][x] = None
        else:
            warn("Can't remove cell outside grid range.")

    def get_cell(self, x, y):
        if self.is_in_range(x, y):
            return self.grid[y][x]
        else:
            warn("Can't return cell outside grid range.")
            return None

    def get_neighbourhood(self, x, y):
        if self.is_in_range(x, y):
            neighbourhood_coords = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                                    (x + 1, y + 1), (x - 1, y - 1),
                                    (x - 1, y + 1), (x + 1, y - 1)]
            neighbourhood_nodes = []
            for nc in neighbourhood_coords:
                if self.is_in_range(nc[0], nc[1]):
                    neighbourhood_nodes.append(self.get_cell(nc[0], nc[1]))
            return neighbourhood_nodes
        else:
            warn("Can't get neighbourhood of cell outside grid range.")
            return None

    def is_empty(self, x, y):
        if self.is_in_range(x, y):
            if self.grid[y][x]:
                return False
            else:
                return True
        else:
            warn("Coordinates provided are outside grid range.")
            return None

    def empty(self):
        self.grid = [[None for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

    def add_random(self, node, z=None):
        x = random.randint(0, len(self.grid[0]) - 1)
        y = random.randint(0, len(self.grid) - 1)
        if self.grid[y][x] is None:
            return self.add_cell(x, y, node, z=z)
        else:
            self.add_random(node, z=z)

    def add_near(self, x, y, node, min_distance=1, max_distance=1):
        # Make sure the randomly selected cell is not occupied
        for tries in range(self.size()[0] * self.size()[1]):
            coords = self.get_random_neighbor(x, y, min_distance, max_distance)
            if coords and self.add_cell(coords[0], coords[1], node):
                return True  # Return success, the node was added
        return False  # Failed to insert near

    def get_random_neighbor(self, x, y, d_min, d_max):
        neighbors = []
        rows = self.size()[0]
        cols = self.size()[1]
        for i in range(-d_max, d_max + 1):
            for j in range(-d_max, d_max + 1):
                if i == 0 and j == 0:
                    continue
                if 0 <= x + i < rows and 0 <= y + j < cols:
                    distance = abs(x + i - x) + abs(y + j - y)
                    if d_min <= distance <= d_max:
                        neighbors.append((x + i, y + j))
        if len(neighbors) == 0:
            return None
        return random.choice(neighbors)

    def get_world_coordinates(self, x, y):
        if self.is_in_range(x, y):
            world_x = self.origin[0] + x * self.cell_size[0]
            world_y = self.origin[1] - y * self.cell_size[1]
            return world_x, world_y
        else:
            return None, None

    def get_grid_coordinates(self, world_x, world_y):
        x = round((world_x - self.origin[0]) / self.cell_size[0])
        y = -round((world_y - self.origin[1]) / self.cell_size[1])
        if self.is_in_range(x, y):
            return x, y
        else:
            return None, None

    def find_by_name(self, name):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] and self.grid[y][x].getField("name").getSFString() == name:  # NOQA
                    return x, y
        return None

    def is_in_range(self, x, y):
        if (0 <= x < len(self.grid[0])) and (0 <= y < len(self.grid)):
            return True
        return False

    def bfs_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        queue = [(start, [start])]  # (coordinates, path to coordinates)
        visited = set()
        visited.add(start)
        while queue:
            coords, path = queue.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)]:  # neighbors
                x, y = coords
                x2, y2 = x + dx, y + dy
                if self.is_in_range(x2, y2) and (x2, y2) not in visited:
                    if self.grid[y2][x2] is not None and (x2, y2) == goal:
                        return path + [(x2, y2)]
                    elif self.grid[y2][x2] is None:
                        visited.add((x2, y2))
                        queue.append(((x2, y2), path + [(x2, y2)]))
        return None
