from numpy import convolve, ones, mean
import gym

from robot_supervisor import CartPoleRobotSupervisor
from utilities import plotData

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env


def run():
    # Initialize supervisor object
    env = CartPoleRobotSupervisor()
    
    # Verify that the environment is working as a gym-style env
    #check_env(env)
    
    #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
    model = PPO("MlpPolicy", env, verbose=1)
    # Indicate the total timmepstes that the agent should be trained.
    model.learn(total_timesteps=40000)
    # Save the model
    model.save("ppo_cartpole")

    #del model # remove to demonstrate saving and loading

    # If is needed to load the trained model
    #model = PPO.load("ppo1_cartpole")

    ################################################################
    # End of the training period and now we evaluate the trained agent
    ################################################################
    
    # Initialize the environment
    obs = env.reset()
    env.episodeScore = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episodeScore)
            env.episodeScore = 0
            obs = env.reset()

