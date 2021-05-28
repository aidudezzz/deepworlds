from numpy import convolve, ones, mean
import gym

from robot_supervisor import CartPoleRobotSupervisor
from utilities import plotData

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env


def run():
    # Initialize supervisor object
    env = CartPoleRobotSupervisor()
    
    # verify that the environment is working as a gym-style env
    #check_env(env)
    
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=40000)
    model.save("ppo1_cartpole")

    #del model # remove to demonstrate saving and loading

    #model = PPO.load("ppo1_cartpole")

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

