import gym
import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import \
    SupervisorCSV
from deepbots.supervisor.wrappers.keyboard_printer import KeyboardPrinter

import utilities as utils
from models.networks import DDPG

OBSERVATION_SPACE = 10
ACTION_SPACE = 2

DIST_SENSORS_MM = {'min': 0, 'max': 1023}
EUCL_MM = {'min': 0, 'max': 1.5}
ACTION_MM = {'min': -1, 'max': 1}
ANGLE_MM = {'min': -np.pi, 'max': np.pi}


class FindTargetSupervisor(SupervisorCSV):
    def __init__(self, robot, target):
        super(FindTargetSupervisor, self).__init__(emitter_name='emitter',
                                                   receiver_name='receiver')
        self.observations = OBSERVATION_SPACE

        self.robot_name = robot
        self.target_name = target
        self.robot = self.getFromDef(robot)
        self.target = self.getFromDef(target)
        self.findThreshold = 0.12
        self.steps = 0
        self.steps_threshold = 600
        self.message = []
        self.should_done = False

    def get_default_observation(self):
        return [0 for i in range(OBSERVATION_SPACE)]

    def get_observations(self):
        message = self.handle_receiver()

        observation = []
        self.message = []
        if message is not None:
            for i in range(len(message)):

                self.message.append(float(message[i]))

                observation.append(
                    utils.normalize_to_range(float(message[i]),
                                             DIST_SENSORS_MM['min'],
                                             DIST_SENSORS_MM['max'], 0, 1))

            distanceFromTarget = utils.get_distance_from_target(
                self.robot, self.target)
            self.message.append(distanceFromTarget)
            distanceFromTarget = utils.normalize_to_range(
                distanceFromTarget, EUCL_MM['min'], EUCL_MM['max'], 0, 1)
            observation.append(distanceFromTarget)

            angleFromTarget = utils.get_angle_from_target(self.robot,
                                                          self.target,
                                                          is_true_angle=True,
                                                          is_abs=False)
            self.message.append(angleFromTarget)
            angleFromTarget = utils.normalize_to_range(angleFromTarget,
                                                       ANGLE_MM['min'],
                                                       ANGLE_MM['max'], 0, 1)
            observation.append(angleFromTarget)

        else:
            observation = [0 for i in range(OBSERVATION_SPACE)]

        self.observation = observation

        return self.observation

    def get_reward(self, action):
        if (self.message is None or len(self.message) == 0
                or self.observation is None):
            return 0

        rf_values = np.array(self.message[:8])

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
        elif np.max(rf_values) > 200:
            return -0.5

        # if (distance != 0):
        #     reward = 0.1 * np.round((0.6 / distance), 1)

        # reward -= (self.steps / self.steps_threshold)
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
        self.steps = 0
        self.should_done = False

        return super().reset()

    def get_info(self):
        pass


supervisor_pre = FindTargetSupervisor('robot', 'target')
supervisor_env = KeyboardPrinter(supervisor_pre)
agent = DDPG(lr_actor=0.00025,
             lr_critic=0.00025,
             input_dims=[10],
             gamma=0.99,
             tau=0.001,
             env=supervisor_env,
             batch_size=256,
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

    # if i % 100 == 0:
    #     agent.save_models()
