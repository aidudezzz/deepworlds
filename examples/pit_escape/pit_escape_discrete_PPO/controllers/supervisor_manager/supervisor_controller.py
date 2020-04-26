import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from agent.PPOAgent import PPOAgent, Transition
from utilities import normalizeToRange, getDistanceFromCenter, plotData
from keyboard_controller_pit_escape import KeyboardControllerPitEscape


class PitEscapeSupervisor(SupervisorCSV):
    """
    This example is taken from Webots https://robotbenchmark.net/benchmark/pit_escape/ example.

    Program a BB-8 robot lost in a sand desert to climb out of a pit as quickly as possible.
    This benchmark aims at developing a program that controls a BB-8 robot to escape from a pit.

    Metrics:
    The robot has to get out of the pit as fast as possible. The benchmark stops if the robot takes more than one
    minute to escape. If the robot is able to get out of the pit, the metric will be based on how fast the robot was
    to get out. Otherwise, the metric will measure how close it was from escaping. In the first case the metric ranges
    from 50% to 100% and is linearly correlated with the time result. A value of 100% is awarded for an instantaneous
    escape, while a value of 50% is awarded for a last-second escape. In the second case the metric ranges from 0% to
    50% and is linearly correlated with the distance from the top of the pit.

    How to improve the performance?
    The slope is too steep for the robot to simply go forward. Instead, it should go back and forth to accumulate
    momentum until it has enough to climb out of the pit.

    Observation:
    Num	Observation                   Min         Max
    0	BB-8 Gyro X axis            -Inf            Inf
    1	BB-8 Gyro Y axis            -Inf            Inf
    2	BB-8 Gyro Z axis            -Inf            Inf
    3	BB-8 Accelerometer X axis   -Inf            Inf
    4	BB-8 Accelerometer Y axis   -Inf            Inf
    5	BB-8 Accelerometer Z axis   -Inf            Inf

    Actions:
        Type: Discrete(4)
        Num Action
        0   Set pitch motor speed to maxSpeed
        1   Set pitch motor speed to -maxSpeed
        2   Set yaw motor speed to maxSpeed
        3   Set yaw motor speed to -maxSpeed

        Note: maxSpeed is set in the robot controller.
    Reward:
        Reward method implementation works based on https://robotbenchmark.net/benchmark/pit_escape/ metric.
        The metric is based on the robot's distance from the pit center. It gets updated when the maximum
        distance from pit center achieved increases. If the robot escapes the pit, the metric is set to 0.5 plus
        a term based on episode time elapsed.

        Thus, the metric monotonically increases and moves from 0.0 to 1.0 on instantaneous escape. Unsolved
        episode's metric varies between 0.0 and 0.5 and solved episode's metric varies between 0.5 and 1.0.

        The step reward return is the difference between the previous step's metric and the current metric. Due to
        the fact that the metric increases monotonically, the difference returned is always >= 0, and > 0 at a step
        where a higher distance from center is achieved.
    Starting State:
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Episode Termination:
        Episode ends after 60.0 seconds or if the robot escapes from the pit, which is calculated based on its
        distance from the pit center and the pit radius.
    """

    def __init__(self, episodeLimit=10000):
        """
        In the constructor, the agent object is created, the robot is spawned in the world via respawnRobot().
        Reference to robot is initialized here.

        :param episodeLimit: int, upper limit of how many episodes to run
        """
        print("Robot is spawned in code, if you want to inspect it pause the simulation.")
        super().__init__()
        self.observationSpace = 6
        self.actionSpace = 4
        self.agent = PPOAgent(self.observationSpace, self.actionSpace)

        self.robot = None
        self.robotDef = "ROBOT_BB-8"
        self.respawnRobot()

        self.episodeCount = 0  # counter for episodes
        self.episodeLimit = episodeLimit
        self.episodeScore = 0  # score accumulated during an episode
        self.episodeScoreList = []  # a list to save all the episode scores, used to check if task is solved
        self.test = False  # whether the agent is in test mode

        self.longestDistance = 0.0
        self.oldMetric = 0.0
        self.metric = 0.0
        self.time = self.supervisor.getTime()  # Current time
        self.startTime = 0.0
        self.episodeTime = 0.0
        self.maxTime = 60.0  # Time in seconds that each episode lasts
        self.pitRadius = self.supervisor.getFromDef("PIT").getField("pitRadius").getSFFloat()

    def get_observations(self):
        """
        Observation gets the message sent by the robot through the receiver.
        The values are extracted from the message, converted to float, normalized and clipped.
        If no message is received, it returns zeros.

        :return: list, observation: [gyro x, gyro y, gyro z, accelerometer x, accelerometer y, accelerometer z]
        """
        messageReceived = self.handle_receiver()
        if messageReceived is not None:
            return [normalizeToRange(float(messageReceived[i]), -5.0, 5.0, -1.0, 1.0, clip=True) for i in
                    range(len(messageReceived))]
        else:
            return [0.0 for _ in range(self.observationSpace)]

    def get_reward(self, action=None):
        """
        Reward method implementation works based on https://robotbenchmark.net/benchmark/pit_escape/ metric.
        Calculates max distance achieved during the episode and based on that updates the metric.
        If the max distance achieved is over the pitRadius, the episode is considered solved and
        the metric updates with that in mind.
        This method updates the metric, but returns the difference with the previous recorded metric.

        :param action: None, get_reward doesn't need the action to calculate the reward
        :return: float, step reward
        """
        self.oldMetric = self.metric

        distance = getDistanceFromCenter(self.robot)  # Calculate current distance from center
        if distance > self.longestDistance:
            self.longestDistance = distance  # Update max
            self.metric = 0.5 * self.longestDistance / self.pitRadius  # Update metric

        # Escaping increases metric over 0.5 based on time elapsed in episode
        if self.longestDistance > self.pitRadius:
            self.metric = 0.5 + 0.5 * (self.maxTime - self.episodeTime) / self.maxTime

        # Step reward is how much the metric changed
        return self.metric - self.oldMetric

    def is_done(self):
        """
        Pit escape implementation counts time elapsed in current episode, based on episode's start time, and current
        time taken from Webots supervisor. The episode is terminated after maxTime seconds, or when the episode is
        solved, the robot is out of the pit.

        :return: bool, True if termination conditions are met, False otherwise
        """
        doneFlag = False
        self.episodeTime = self.time - self.startTime  # Update episode time

        # Time's not up
        if self.episodeTime < self.maxTime:
            self.time = self.supervisor.getTime()  # Update current time
        # Episode time run out
        else:
            doneFlag = True

        # Episode solved
        if self.longestDistance > self.pitRadius:
            doneFlag = True

        if doneFlag:
            self.startTime = self.time  # Update next episode's start time
            # Reset reward related variables
            self.longestDistance = 0.0
            self.metric = 0.0
            self.oldMetric = 0.0
            return True

        return False

    def reset(self):
        """
        Reset calls respawnRobot() method and returns starting observation.
        :return: list, starting observation filled with zeros
        """
        self.respawnRobot()
        return [0.0 for _ in range(self.observationSpace)]

    def respawnRobot(self):
        """
        This method reloads the saved BB-8 robot in its initial state from the disk.
        """
        if self.robot is not None:
            # Despawn existing robot
            self.robot.remove()

        # Respawn robot in starting position and state
        rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
        childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
        childrenField.importMFNode(-2, "BB-8.wbo")  # Load robot from file and add to second-to-last position

        # Get the new robot reference
        self.robot = self.supervisor.getFromDef(self.robotDef)
        # Reset the simulation physics to start over
        self.supervisor.simulationResetPhysics()

    def get_info(self):
        return None

    def solved(self):
        """
        This method checks whether the Pit Escape task is solved, so training terminates.
        Task is considered solved when average score of last 100 episodes is > 0.85.
        :return: bool, True if task is solved, False otherwise
        """
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 0.85:  # Last 100 episodes' scores average value
                return True
        return False

    def step(self, action, repeatSteps=None):
        """
        This custom implementation of step incorporates a repeat step feature. By setting the repeatSteps
        value, the supervisor is stepped and the selected action is emitted to the robot repeatedly.
        :param action: iterable, that contains the action value(s)
        :param repeatSteps: int, number of steps to repeatedly do the same action before returning
        :return: observation, reward, done, info
        """
        if repeatSteps is not None and repeatSteps != 0:
            for _ in range(repeatSteps):
                self.supervisor.step(self.get_timestep())
                self.handle_emitter(action)
        else:
            self.supervisor.step(self.get_timestep())
            self.handle_emitter(action)

        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )
