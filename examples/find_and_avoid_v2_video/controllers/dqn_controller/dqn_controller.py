'''
DQN Controller for the mobile robot
'''
import DQN_trainer
import DQN_testing

experiment_name = "trained_agent"
experiment_description = """The baseline agent trained on default parameters."""
only_test = False  # If true, the trained agent from "experiment_name" will be loaded and evaluated
test_results_filename = "testing_results"

# Testing setup
deterministic = False
use_masking = False
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
maximum_episode_steps = 4096  # Steps for episode timeout

step_window = 1           # Latest steps of observations
seconds_window = 1        # How many latest seconds of observations
add_action_to_obs = True
reset_on_collisions = 1   # Reset on number of collisions
on_tar_threshold = 1      # The distance under which the robot is considered "on target"

# How many test episodes to run on sb3 evaluation and then for each difficulty in the custom
# evaluation, see DQN_testing.py
tests_per_difficulty = 10 

# Distance sensor denial list
ds_denial_list = []

# Reward weights
target_dist_weight = 1.0
target_angle_weight = 1.0
dist_sensors_weight = 1.0
target_reach_weight = 1.0
collision_weight = 1.0
smoothness_weight = 0.0
speed_weight = 0.0

# Training setup
batch_size = 64
gamma = 0.999
lr_rate = 3e-4
tau=1e-3
num_steps_for_update = 4 
exp_frac = 0.2 # fraction of training time for decaying exploration rate
eps_init = 1.0
eps_final = 0.05
reward_avg_pts = 100
log_interval=4
render_interval=100 # number of episodes before redering video to tensorboard 
net_arch = [64, 64] # Quality network architecture (hidden layers)

# Map setup
map_w, map_h = 7, 7
cell_size = None

seed = 1
env = DQN_trainer.run(experiment_name=experiment_name,
                      experiment_description=experiment_description,
                      manual_control=manual_control,
                      only_test=only_test,
                      maximum_episode_steps=maximum_episode_steps,
                      step_window=step_window, 
                      seconds_window=seconds_window, 
                      add_action_to_obs=add_action_to_obs,
                      ds_params=ds_params, 
                      reset_on_collisions=reset_on_collisions, 
                      on_tar_threshold=on_tar_threshold,
                      target_dist_weight=target_dist_weight, 
                      target_angle_weight=target_angle_weight,
                      dist_sensors_weight=dist_sensors_weight, 
                      target_reach_weight=target_reach_weight,
                      collision_weight=collision_weight, 
                      smoothness_weight=smoothness_weight, 
                      speed_weight=speed_weight,
                      net_arch=net_arch, 
                      n_steps=None, 
                      batch_size=batch_size, 
                      gamma=gamma, 
                      map_w=map_w,
                      map_h=map_h,
                      cell_size=cell_size,
                      lr_rate=lr_rate,
                      difficulty_dict=difficulty_dict, 
                      seed=seed, 
                      tau=tau, 
                      num_steps_for_update=num_steps_for_update,
                      exp_frac=exp_frac, 
                      eps_init=eps_init, 
                      eps_final=eps_final, 
                      reward_avg_pts=reward_avg_pts,
                      log_interval=log_interval,
                      render_interval=render_interval
                    )
seed = 2
env.ds_denial_list = ds_denial_list  # Distance sensor denial is added only for testing
DQN_testing.run(experiment_name, env, deterministic, use_masking, testing_results_filename=test_results_filename,
                tests_per_difficulty=tests_per_difficulty, seed=seed)
