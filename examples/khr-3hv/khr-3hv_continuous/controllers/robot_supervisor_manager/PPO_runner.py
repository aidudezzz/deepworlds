from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb

import ray
from ray import tune
from ray.tune.registry import register_env
try:
    from ray.rllib.agents.ppo import ppo
except ImportError:
    from ray.rllib.algorithms.ppo import ppo
from ray.tune.integration.wandb import WandbLoggerCallback

from KHR3HV_env import KHR3HVRobotSupervisor


# Scheduler for the learning rate when using the stable baselines
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate scheduler for when using stable baselines.
    :param initial_value: The learning rate initial value.
    :type initial_value: float
    :return: Current learning rate depending on remaining progress.
    :rtype: float
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    # TODO  decrease the learning_rate by .10
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: The remaining progress.
        :type progress_remaining: float
        :return: Current learning rate depending on remaining progress.
        :rtype: float
        """
        return progress_remaining * initial_value

    return func


def run(project_name, wandb_save_period, train, use_ray, output, args):
    # Train with Ray
    if use_ray:
        # Register the environment to be used by Ray
        register_env("khr3hv", lambda config: KHR3HVRobotSupervisor(project_name, wandb_save_period, train, use_ray))
        # Model hyperparameters
        model_config = {
            "env": "khr3hv",
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",  # tanh or relu
            },
            "lr": args.learning_rate,
            "framework": "torch",
            "gamma": args.gamma,
            "lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "kl_coeff": args.kl_coeff,
            "num_sgd_iter": args.num_sgd_iter,
            "sgd_minibatch_size": args.sgd_minibatch_size,
            "horizon": 200,
            "train_batch_size": args.train_batch_size,
            "num_workers": args.num_workers,
            "num_gpus": args.num_gpus,
            "batch_mode": args.batch_mode,
            "observation_filter": args.observation_filter
        }
        # Stop criteria for training
        stop = {
            "timesteps_total": 100e6,
        }
        if train:
            # Initialize wandb providing the project name
            # wandb.init(project=PROJECT_NAME)
            # Train the model using Ray tune
            tune.run(ppo.PPOTrainer,
                     config=model_config,
                     stop=stop,
                     callbacks=[WandbLoggerCallback(
                                project=project_name,
                                log_config=True)],
                     local_dir=output,
                     checkpoint_freq=wandb_save_period,
                     checkpoint_at_end=True
                     )
            ray.shutdown()
        else:
            # Create an agent having the predefined model hyperparameters
            agent = ppo.PPOTrainer(config=model_config)
            # Restore the train weights
            agent.restore(
                "./results/Deepbots-khr3hv-Ray-reward-weight,prev_pos,obs-x,y,z,"
                "vel-done-0.5-RNN-5,clip_range-0.2,learning_rate-0.0001_6778500_steps")

            # Use the environment of khr3hv Robot
            khr3hv_env = KHR3HVRobotSupervisor(project_name, wandb_save_period, train, use_ray)
            # Reset the environment and test the trained agent
            obs = khr3hv_env.reset()
            khr3hv_env.episode_score = 0
            while True:
                action, _states = agent.compute_action(obs)
                obs, reward, done, _ = khr3hv_env.step(action)
                khr3hv_env.episode_score += reward  # Accumulate episode reward

                if done:
                    print("Reward accumulated =", khr3hv_env.episode_score)
                    khr3hv_env.episode_score = 0
                    obs = khr3hv_env.reset()

    # Train with Stable Baselines 3 framework
    else:
        # Use the environment of khr3hv Robot
        khr3hv_env = KHR3HVRobotSupervisor(project_name, wandb_save_period, train, use_ray)

        if train:
            # Initialize wandb providing the project name
            wandb.init(project=project_name)

            # Save a checkpoint every wandb_save_period steps
            checkpoint_callback = CheckpointCallback(
                save_freq=wandb_save_period, save_path=output, name_prefix=project_name)

            # Set the neural network size and/or the activation function
            # policy_kwargs = dict(activation_fn=th.nn.ReLU, dict(pi=[64, 64], vf=[64, 64])])
            policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])

            # Check if the khr3hv_env is compatible with Stable Baseline 3
            check_env(khr3hv_env)
            # Declare PPO agent with predefined hyperparameters
            model = PPO("MlpPolicy", khr3hv_env, verbose=1, policy_kwargs=policy_kwargs,
                        clip_range=args.clip_param, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        learning_rate=args.learning_rate)
            # Train the agent and use the callback function to save the agent weights at a given time
            model.learn(total_timesteps=100000000, callback=checkpoint_callback)
        else:
            # Load the weights of a trained Agent
            model = PPO.load(
                "./results/Deepbots-khr3hv-SB-reward-weight,prev_pos,obs-x,y,z,"
                "vel-done-0.5-RNN-5,clip_range-0.2,learning_rate-0.0001_6778500_steps")

            # Reset the environment and test the trained agent
            obs = khr3hv_env.reset()
            khr3hv_env.episode_score = 0
            while True:
                print("Testing...")
                action, _states = model.predict(obs)
                obs, reward, done, _ = khr3hv_env.step(action)
                khr3hv_env.episode_score += reward  # Accumulate episode reward

                if done:
                    print("Reward accumulated =", khr3hv_env.episode_score)
                    khr3hv_env.episode_score = 0
                    obs = khr3hv_env.reset()
