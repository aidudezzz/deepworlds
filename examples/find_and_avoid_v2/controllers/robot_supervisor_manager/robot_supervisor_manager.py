"""
More runners for discrete RL algorithms can be added here.
"""
import PPO_trainer
import PPO_testing

experiment_name = "trained_agent"
experiment_description = """The baseline agent trained on default parameters."""
only_test = False  # If true, the trained agent from "experiment_name" will be loaded and evaluated
test_results_filename = "testing_results"

# Testing setup
deterministic = False
use_masking = True
manual_control = False

# Distance sensors setup
ds_params = {"ds_type": "sonar",  # generic, sonar
             "ds_n_rays": 4,  # 1, 4
             "ds_aperture": 0.1,  # 1.57, 0.1
             "ds_resolution": -1.0,  # -1.0, 1.0
             "ds_noise": 0.025,  # 0.0, 0.025
             "max_ds_range": 100  # in cm
             }
"""
- diff_0: Random map with a few obstacles, mostly easy clear or almost clear paths to the target for initial learning.
- diff_1: Corridor map with one row of two obstacles. Forces the agent to move around obstacles more.
- diff_2: Corridor map with two rows of four obstacles. Same as before but more difficult.
- diff_3: Corridor map with three rows of six obstacles. Same as before but more difficult.
- diff_4: Corridor map with four or five rows of eight or ten obstacles. Same as before but more difficult.
- diff_5: Random map with all available obstacles. Final complex difficulty in a more general environment.
"""
difficulty_dict = {"diff_0": {"type": "random", "number_of_obstacles": 10,
                              "min_target_dist": 10, "max_target_dist": 10, "total_timesteps": 262144},
                   "diff_1": {"type": "corridor", "number_of_obstacles": 2,
                              "min_target_dist": 2, "max_target_dist": 2, "total_timesteps": 524288},
                   "diff_2": {"type": "corridor", "number_of_obstacles": 4,
                              "min_target_dist": 3, "max_target_dist": 3, "total_timesteps": 524288},
                   "diff_3": {"type": "corridor", "number_of_obstacles": 6,
                              "min_target_dist": 4, "max_target_dist": 4, "total_timesteps": 524288},
                   "diff_4": {"type": "corridor", "number_of_obstacles": 10,
                              "min_target_dist": 5, "max_target_dist": 6, "total_timesteps": 524288},
                   "diff_5": {"type": "random", "number_of_obstacles": 25,
                              "min_target_dist": 10, "max_target_dist": 12, "total_timesteps": 1048576}}

# Environment setup
maximum_episode_steps = 16_384  # Steps for episode timeout

step_window = 1  # Latest steps of observations
seconds_window = 1  # How many latest seconds of observations
add_action_to_obs = True
reset_on_collisions = 4096  # Reset on number of collisions
on_tar_threshold = 0.1  # The distance under which the robot is considered "on target"

# How many test episodes to run on sb3 evaluation and then for each difficulty in the custom
# evaluation, see PPO_testing.py
tests_per_difficulty = 100

# Distance sensor denial list
# Filled with integers that correspond to indices of distance sensors, e.g. [1, 2, 3, 5, 7, 9, 10, 11]
#  Distance sensors from left to right:
#  Sensor indices/positions:   [0,   1,    2,    3,     4,    5,     6,      7,     8,    9,     10,   11,   12]
#  Frontal sensors:                                    [           frontal           ]
#  Left/right sensors:         [           left          ]                         [           right           ]
# Sensors whose index is contained in the list have their value overwritten with the max value, disabling them.
# Used only in testing, to train with sensor denial you need to provide it to the ctor of the environment or entirely
# remove unused sensors from the world as well as the observation of the agent and modify the environment internally.
ds_denial_list = []

# Reward weights
target_dist_weight = 1.0
target_angle_weight = 1.0
dist_sensors_weight = 10.0
target_reach_weight = 1000.0
collision_weight = 100.0
smoothness_weight = 0.0
speed_weight = 0.0

# Training setup
n_steps = 2048
batch_size = 64
gamma = 0.999
gae_lambda = 0.95
target_kl = None
vf_coef = 0.5
ent_coef = 0.001
lr_rate = lambda f: f * 3e-4  # NOQA Linear decreasing schedule

net_arch = dict(pi=[1024, 512, 256], vf=[2048, 1024, 512])  # Actor-critic layers
# Map setup
map_w, map_h = 7, 7
cell_size = None

seed = 1
env = PPO_trainer.run(experiment_name=experiment_name, experiment_description=experiment_description,
                      manual_control=manual_control, only_test=only_test, maximum_episode_steps=maximum_episode_steps,
                      step_window=step_window, seconds_window=seconds_window, add_action_to_obs=add_action_to_obs,
                      ds_params=ds_params, reset_on_collisions=reset_on_collisions, on_tar_threshold=on_tar_threshold,
                      target_dist_weight=target_dist_weight, target_angle_weight=target_angle_weight,
                      dist_sensors_weight=dist_sensors_weight, target_reach_weight=target_reach_weight,
                      collision_weight=collision_weight, smoothness_weight=smoothness_weight, speed_weight=speed_weight,
                      net_arch=net_arch, n_steps=n_steps, batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda,
                      target_kl=target_kl, vf_coef=vf_coef, ent_coef=ent_coef, lr_rate=lr_rate,
                      difficulty_dict=difficulty_dict, seed=seed)
seed = 2
env.ds_denial_list = ds_denial_list  # Distance sensor denial is added only for testing
PPO_testing.run(experiment_name, env, deterministic, use_masking, testing_results_filename=test_results_filename,
                tests_per_difficulty=tests_per_difficulty, seed=seed)
