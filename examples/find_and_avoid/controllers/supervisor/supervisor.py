import gym
import numpy as np
from deepbots.supervisor import CSVSupervisorEnv
from deepbots.supervisor.wrappers import KeyboardPrinter

import utilities as utils
from models.networks import DDPG
import os
OBSERVATION_SPACE = 10
ACTION_SPACE = 2

DIST_SENSORS_MM = {'min': 0, 'max': 1023}
EUCL_MM = {'min': 0, 'max': 1.5}
ACTION_MM = {'min': -1, 'max': 1}
ANGLE_MM = {'min': -np.pi, 'max': np.pi}


class FindTargetSupervisor(CSVSupervisorEnv):
    def __init__(self, robot, target):
        super(FindTargetSupervisor, self).__init__(emitter_name='emitter',
                                                   receiver_name='receiver')
        self.observations = OBSERVATION_SPACE

        self.robot_name = robot
        self.target_name = target
        self.robot = self.getFromDef(robot)
        self.target = self.getFromDef(target)
        self.find_threshold = 0.05
        self.steps = 0
        self.steps_threshold = 500
        self.message = []
        self.should_done = False

        self.pre_distance = None
        '''
        Get other 2 intermediate targets when training the robot in small_world.wbt instead of small_world_easy.wbt.
        
        self.mid1 = self.getFromDef("mid1")
        self.mid2 = self.getFromDef("mid2")
        '''
        self.is_solved = False

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

            distance_from_target = utils.get_distance_from_target(
                self.robot, self.target)
            self.message.append(distance_from_target)
            distance_from_target = utils.normalize_to_range(
                distance_from_target, EUCL_MM['min'], EUCL_MM['max'], 0, 1)
            observation.append(distance_from_target)

            angle_from_target = utils.get_angle_from_target(self.robot, self.target, is_abs=False)
            self.message.append(angle_from_target)
            angle_from_target = utils.normalize_to_range(angle_from_target,
                                                       ANGLE_MM['min'],
                                                       ANGLE_MM['max'], 0, 1)
            observation.append(angle_from_target)

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

        # # (1) Take too many steps
        # if self.steps > self.steps_threshold:
        #     return -10
        # reward -= (self.steps / self.steps_threshold)

        # # (2) Reward according to distance
        target_ = self.target
            
        if self.pre_distance == None:
            self.pre_distance = utils.get_distance_from_target(self.robot, target_)
        else:
            cur_distance = utils.get_distance_from_target(self.robot, target_)
            reward = self.pre_distance - cur_distance
            self.pre_distance = cur_distance
            
        # # (3) Find the target
        # if utils.get_distance_from_target(self.robot, self.target) < self.find_threshold:
        #     reward += 5

        # # (4) Action 1 (gas) or Action 0 (turning) should <= 1.5
        # if np.abs(action[1]) > 1.5 or np.abs(action[0]) > 1.5:
        #     if self.steps > 10:
        #         self.should_done = True
        #     return -1

        # # (5) Stop or Punish the agent when the robot is getting to close to obstacle
        # if np.max(rf_values) > 500:
        #     if self.steps > 10:
        #         self.should_done = True
        #     return -1
        # elif np.max(rf_values) > 200:
        #     return -0.5
        
        return reward

    def is_done(self):
        self.steps += 1
        distance = utils.get_distance_from_target(self.robot, self.target)

        if distance < self.find_threshold:
            print("======== + Solved + ========")
            self.is_solved = True
            return True

        if self.steps > self.steps_threshold or self.should_done:
            return True

        return False

    def reset(self):
        self.steps = 0
        self.should_done = False
        self.pre_distance = None
        self.is_solved = False

        return super().reset()

    def get_info(self):
        pass


def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)     
    else:
        print ("Successfully created the directory %s " % path)

if __name__ == '__main__':
    create_path("./models/saved/ddpg/")
    create_path("./exports/")

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
                layer3_size=200,
                n_actions=2,
                load_models=False,
                save_dir='./models/saved/ddpg/')
    # # Load from checkpoint
    # agent.load_models(lr_critic=0.00025, lr_actor=0.00025, 
    #                 input_dims=[10], 
    #                 layer1_size=400,
    #                 layer2_size=300, 
    #                 layer3_size=200, 
    #                 n_actions=2, 
    #                 load_dir='./models/saved/ddpg/')
    score_history = []

    np.random.seed(0)
    n_episode = 600
    for i in range(n_episode+1):
        done = False
        score = 0
        obs = list(map(float, supervisor_env.reset()))
        
        first_iter = True

        if score_history == [] or np.mean(score_history[-50:])<0.5 or score_history[-1]<0.65:
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
        else:
            print("================= TESTING =================")
            while not done:
                if (not first_iter):
                    act = agent.choose_action_test(obs).tolist()
                else:
                    first_iter = False
                    act = [0, 0]
                
                new_state, _, done, _ = supervisor_env.step(act)
                obs = list(map(float, new_state))
            

        score_history.append(score)
        fp = open("./exports/Episode-score.txt","a")
        fp.write(str(score)+'\n')
        fp.close()
        print("===== Episode", i, "score %.2f" % score,
            "50 game average %.2f" % np.mean(score_history[-50:]))

        if supervisor_pre.is_solved == True:
            agent.save_models()