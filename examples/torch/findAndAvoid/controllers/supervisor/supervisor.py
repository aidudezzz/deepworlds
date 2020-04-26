import math

import numpy as np

import utilities as utils
from deepbots.supervisor.controllers.supervisor_emitter_receiver import \
    SupervisorCSV
from deepbots.supervisor.wrappers.keyboard_printer import KeyboardPrinter
from deepbots.supervisor.wrappers.tensorboard_wrapper import TensorboardLogger
from models.default_networks import DefaultDDPG
from models.networks import DDPG

OBSERVATION_SPACE = 10
ACTION_SPACE = 2

DIST_SENSORS_MM = {'min': 0, 'max': 1023}
EUCL_MM = {'min': 0, 'max': 1.5}
ACTION_MM = {'min': -1, 'max': 1}
ANGLE_MM = {'min': -math.pi, 'max': math.pi}


class FindTargetSupervisor(SupervisorCSV):
    def __init__(self, robot, target, observation_space):
        super(FindTargetSupervisor, self).__init__(
            emitter_name='emitter', receiver_name='receiver')
        self.robot_name = robot
        self.target_name = target
        self.robot = self.supervisor.getFromDef(robot)
        self.target = self.supervisor.getFromDef(target)
        self.observation = [0 for i in range(observation_space)]
        self.findThreshold = 0.12
        self.steps = 0
        self.steps_threshold = 6000
        self.message = []
        self.should_done = False

    def get_observations(self):
        message = super().handle_receiver()

        observation = []
        self.message = []
        if message is not None:
            for i in range(len(message)):

                self.message.append(float(message[i]))

                observation.append(
                    utils.normalize_to_range(
                        float(message[i]), DIST_SENSORS_MM['min'],
                        DIST_SENSORS_MM['max'], 0, 1))

            distanceFromTarget = utils.get_distance_from_target(
                self.robot, self.target)
            self.message.append(distanceFromTarget)
            distanceFromTarget = utils.normalize_to_range(
                distanceFromTarget, EUCL_MM['min'], EUCL_MM['max'], 0, 1)
            observation.append(distanceFromTarget)

            angleFromTarget = utils.get_angle_from_target(
                self.robot, self.target, is_true_angle=True, is_abs=False)
            self.message.append(angleFromTarget)
            angleFromTarget = utils.normalize_to_range(
                angleFromTarget, ANGLE_MM['min'], ANGLE_MM['max'], 0, 1)
            observation.append(angleFromTarget)

        else:
            observation = [0 for i in range(OBSERVATION_SPACE)]

        self.observation = observation

        return self.observation

    def empty_queue(self):
        self.message = None
        self.observation = None
        while self.supervisor.step(self.timestep) != -1:
            if self.receiver.getQueueLength() > 0:
                self.receiver.nextPacket()
            else:
                break

    def get_reward(self, action):
        if (self.message is None or len(self.message) == 0
                or self.observation is None):
            return 0

        rf_values = np.array(self.message[:8])
        distance = self.message[8]

        reward = 0

        if self.steps > self.steps_threshold:
            return -10

        if utils.get_distance_from_target(self.robot,
                                          self.target) < self.findThreshold:
            return +10

        if np.abs(action[1]) > 1.5 or np.abs(action[0]) > 1.5:
            if self.steps > 10:
                self.should_done = True
            return -1

        if np.max(rf_values) > 500:
            if self.steps > 10:
                self.should_done = True
            return -1
        elif np.max(rf_values) > 100:
            return -0.5

        if (distance != 0):
            reward = (0.6 / distance)

        reward -= (self.steps / self.steps_threshold)
        return reward

    def is_done(self):
        self.steps += 1
        distance = utils.get_distance_from_target(self.robot, self.target)

        if distance < self.findThreshold:
            print("======== + Solved + ========")
            return True

        if self.steps > self.steps_threshold or self.should_done:
            return True

        return False

    def reset(self):
        print("Reset simulation")
        self.respawnRobot()
        self.steps = 0
        self.should_done = False
        self.message = None
        return self.observation

    def get_info(self):
        pass

    def respawnRobot(self):
        """
        This method reloads the saved CartPole robot in its initial state from the disk.
        """
        if self.robot is not None:
            # Despawn existing robot
            self.robot.remove()

        # Respawn robot in starting position and state
        rootNode = self.supervisor.getRoot(
        )  # This gets the root of the scene tree
        childrenField = rootNode.getField(
            'children'
        )  # This gets a list of all the children, ie. objects of the scene
        childrenField.importMFNode(
            -2, "Robot.wbo"
        )  # Load robot from file and add to second-to-last position

        # Get the new robot and pole endpoint references
        self.robot = self.supervisor.getFromDef(self.robot_name)
        self.target = self.supervisor.getFromDef(self.target_name)
        # Reset the simulation physics to start over
        self.supervisor.simulationResetPhysics()

        self._last_message = None


supervisor_pre = FindTargetSupervisor('robot', 'target', observation_space=10)
supervisor_env = KeyboardPrinter(supervisor_pre)
supervisor_env = TensorboardLogger(
    supervisor_env,
    log_dir="logs/results/ddpg",
    v_action=1,
    v_observation=1,
    v_reward=1,
    windows=[10, 100, 200])

agent = DDPG(
    lr_actor=0.000025,
    lr_critic=0.00025,
    input_dims=[10],
    gamma=0.99,
    tau=0.001,
    env=supervisor_env,
    batch_size=64,
    layer1_size=400,
    layer2_size=300,
    n_actions=2,
    load_models=False,
    save_dir='./models/saved/ddpg/')

score_history = []

np.random.seed(0)

for i in range(1, 500):
    done = False
    score = 0
    obs = list(map(float, supervisor_env.reset()))
    supervisor_pre.empty_queue()
    first_iter = True
    if i % 250 == 0:
        print("================= TESTING =================")
        while not done:
            act = agent.choose_action_test(obs).tolist()
            new_state, _, done, _ = supervisor_env.step(act)
            obs = list(map(float, new_state))
    else:
        print("================= TRAINING =================")
        while not done:
            if (not first_iter):
                act = agent.choose_action_train(obs).tolist()
            else:
                first_iter = False
                act = [0, 0]

            new_state, reward, done, info = supervisor_env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward

            obs = list(map(float, new_state))

    score_history.append(score)
    print("===== Episode", i, "score %.2f" % score,
          "100 game average %.2f" % np.mean(score_history[-100:]))

    if i % 100 == 0:
        agent.save_models()
