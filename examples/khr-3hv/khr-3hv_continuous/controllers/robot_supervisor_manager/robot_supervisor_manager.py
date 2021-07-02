import PPO_runner
from dataclasses import dataclass

"""
Modify these parameters if needed.
"""
PROJECT_NAME = "KHR-3HV Humanoid"  # The project name on wandb
WANDB_SAVE_PERIOD = 100  # Logging period (steps) on wandb
TRAIN = True  # True: Train the agent; False: Go straight into testing
USE_RAY = False  # Use Ray or Stable Baselines 3
OUTPUT_DIR = 'results/'


@dataclass
class ModelHP:
    """
    Common model hyperparameters for both Ray and Stable Baselines 3, modify if needed.
    """
    learning_rate: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_param: float = 0.2

    """
    Ray-only model hyperparameters, modify if needed.
    """
    kl_coeff: float = 1.0
    num_sgd_iter: int = 10
    sgd_minibatch_size: int = 64
    train_batch_size: int = 64
    num_workers: int = 1
    num_gpus: int = 0
    batch_mode: str = "complete_episodes"
    observation_filter: str = "MeanStdFilter"


if __name__ == '__main__':
    args = ModelHP()
    PPO_runner.run(PROJECT_NAME, WANDB_SAVE_PERIOD, TRAIN, USE_RAY, OUTPUT_DIR, args)
