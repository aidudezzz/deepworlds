"""
More runners for discrete RL algorithms can be added here.
"""
import PPO_runner

# Modify these constants if needed.
EPISODE_LIMIT = 10000
MAX_TIME = 60.0  # Time in seconds that each episode lasts

if __name__ == '__main__':
    PPO_runner.run()
