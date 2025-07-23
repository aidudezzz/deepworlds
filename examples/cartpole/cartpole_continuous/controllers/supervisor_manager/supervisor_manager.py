"""
More runners for continuous RL algorithms can be added here.
"""
import DDPG_runner

# Modify these constants if needed.
EPISODE_LIMIT = 10000
STEPS_PER_EPISODE = 200  # How many steps to run each episode (changing this messes up the solved condition)

if __name__ == '__main__':
    DDPG_runner.run()
