"""
More runners for continuous RL algorithms can be added here.
"""
import DDPG_runner

# Modify these constants if needed.
EPISODE_LIMIT = 10000
STEPS_PER_EPISODE = 200  # How many steps to run each episode (changing this messes up the solved condition)

if __name__ == '__main__':
    DDPG_runner.run()

# from robot_supervisor import CartPoleRobotSupervisor
# from stable_baselines3.ppo import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import CheckpointCallback
#
#
# env = Monitor(CartPoleRobotSupervisor(), filename="")
# checkpoint_callback = CheckpointCallback(
#     save_freq=int(1e4),
#     save_path='./checkpoints/',
#     name_prefix='rl_model'
# )
#
# # Train
# model = PPO(
#     'MlpPolicy',
#     env,
#     tensorboard_log='./tb_logs/',
#     verbose=1
# )
# model.learn(total_timesteps=int(1e6), callback=checkpoint_callback)
