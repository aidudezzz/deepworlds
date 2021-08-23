import numpy as np
from scipy.ndimage.interpolation import shift
import wandb

from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune import grid_search
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.integration.wandb import wandb_mixin

from NaoEnv import NaoRobotSupervisor


# Scheduler for the learning rate when using the stable baselines
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value:
    :return: current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    # TODO  decrease the lr by .10
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func         
            
def run(projectName, wandbSaveFreq, train, useRay, output, args):

    if useRay:
        # Register the enviroment to be used by Ray
        register_env("Nao", lambda config: NaoRobotSupervisor(projectName, 
                                              wandbSaveFreq, train, useRay))
        # Model hyperparameters
        model_config = {
                "env": "Nao",
                "model":{
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu", # tanh or relu
                },
                "lr": args.lr,
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
            # Init the wandb providing the project name
            #wandb.init(project=PROJECT_NAME)
            # Train the model using Ray tune
            tune.run(ppo.PPOTrainer, 
                            config=model_config, 
                            stop=stop,
                            callbacks=[WandbLoggerCallback(
                                    project=projectName,
                                    api_key_file="../../../wandb.txt",
                                    log_config=True)],
                            local_dir=output, 
                            checkpoint_freq=wandbSaveFreq,
                            checkpoint_at_end=True
                            )
            ray.shutdown()
        else:
            # Crate an agent having the predifined model hyperparameters
            agent = ppo.PPOTrainer(config=model_config)
            # Restore the train weights
            agent.restore("./results/Deepbots-Nao-Ray-reward-weight,prev_pos,obs-x,y,z,vel-done-0.5-RNN-5,clip_range-0.2,lr-0.0001_6778500_steps")
            
            # Use the enviroment of nao Robot
            nao_env = NaoRobotSupervisor(projectName, wandbSaveFreq, train, useRay) 
            # Reset the enviroment and test the trained agent
            obs = nao_env.reset()
            nao_env.episodeScore = 0
            while True:
                action, _states = agent.compute_action(obs)
                obs, reward, done, _ = nao_env.step(action)
                nao_env.episodeScore += reward  # Accumulate episode reward

                if done:
                    print("Reward accumulated =", nao_env.episodeScore)
                    nao_env.episodeScore = 0
                    obs = nao_env.reset()
    else:
        # Use of the Stable-Baseline framework
        
        # Use the enviroment of nao Robot
        nao_env = NaoRobotSupervisor(projectName, wandbSaveFreq, train, useRay) 

        if train:
            # Init the wandb providing the project name
            wandb.init(project=projectName)
            
            # Save a checkpoint
            checkpoint_callback = CheckpointCallback(save_freq=wandbSaveFreq, save_path=output,
                                            name_prefix=projectName)
            
            # Change the neural net size or the activation function
            policy_kwargs = dict(#activation_fn=th.nn.ReLU,
                        net_arch=[dict(pi=[64, 64], vf=[64, 64])])
            
            # Check if the nao_env is compatible with Stable Baseline 
            check_env(nao_env) 
            # Declare PPO agent with predifined hyperparameters
            model = PPO("MlpPolicy", nao_env, verbose=1, 
                        clip_range=args.clip_param, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        learning_rate=args.lr)
            # Train the agent and use the callback function to save the agent weights at a given time
            model.learn(total_timesteps=100000000, callback=checkpoint_callback)
        else:
            # Load the weights of a trained Agent
            model = PPO.load("./results/Deepbots-Nao-SB-reward-weight,prev_pos,obs-x,y,z,vel-done-0.5-RNN-5,clip_range-0.2,lr-0.0001_6778500_steps")

            # Reset the enviroment and test the trained agent
            obs = nao_env.reset()
            nao_env.episodeScore = 0
            while True:
                action, _states = model.predict(obs)
                obs, reward, done, _ = nao_env.step(action)
                nao_env.episodeScore += reward  # Accumulate episode reward

                if done:
                    print("Reward accumulated =", nao_env.episodeScore)
                    nao_env.episodeScore = 0
                    obs = nao_env.reset()