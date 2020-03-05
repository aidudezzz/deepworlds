import numpy as np
from time import time

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.DDPG_agent import DDPGAgent
from utilities import plotData


def run():
    # Initialize supervisor object
    supervisorPre = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard controller
    supervisorEnv = KeyboardControllerCartPole(supervisorPre)

    # The agent used here is trained with the DDPG algorithm (https://arxiv.org/abs/1509.02971).
    agent = DDPGAgent(supervisorPre.observationSpace, supervisorPre.actionSpace, lr_actor=0.000025, lr_critic=0.00025,
                      layer1_size=30, layer2_size=50, layer3_size=30, batch_size=64)

    episodeCount = 0
    episodeLimit = 10000
    solved = False  # Whether the solved requirement is met

    startTime = time()  # Start counting real training time
    simTime = 0  # Accumulator for simulated time elapsed

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        # Get time elapsed in episode before resetting it and add it to simTime
        simTime += supervisorPre.supervisor.getTime()

        state = supervisorEnv.reset()  # Reset robot and get starting observation
        supervisorPre.episodeScore = 0

        # Inner loop is the episode loop
        for step in range(supervisorPre.stepsPerEpisode):
            # In training mode the agent returns the action plus OU noise for exploration
            selectedAction = agent.choose_action_train(state)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached
            # the done condition
            newState, reward, done, info = supervisorEnv.step(selectedAction)

            # Save the current state transition in agent's memory
            agent.remember(state, selectedAction, reward, newState, int(done))
            # Perform a learning step

            supervisorPre.episodeScore += reward  # Accumulate episode reward
            agent.learn()
            if done or step == supervisorPre.stepsPerEpisode - 1:
                # Save the episode's score
                supervisorPre.episodeScoreList.append(supervisorPre.episodeScore)
                # agent.learn(batch_size=step + 1)
                solved = supervisorPre.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        if supervisorPre.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episodeCount, "score:", supervisorPre.episodeScore)
        episodeCount += 1  # Increment episode counter

    # Calculate and print training wall clock time
    trainingTime = time() - startTime
    if trainingTime < 60.0:
        print("Training finished after:", trainingTime, "seconds, real time.")
    else:
        rem = round(trainingTime % 60, 2)
        mins = int(trainingTime / 60)
        print("Training finished after:", mins, "minutes,", rem, "seconds, real time.")

    # Print simulated time
    if simTime < 60.0:
        print("Training finished after:", simTime, "seconds, simulated time.")
    else:
        rem = round(simTime % 60, 2)
        mins = int(simTime / 60)
        print("Training finished after:", mins, "minutes,", rem, "seconds, simulated time.")

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    # this is done to smooth out the plots
    movingAvgN = 10
    plotData(np.convolve(supervisorPre.episodeScoreList, np.ones((movingAvgN,)) / movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")

    if not solved and not supervisorPre.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")
    state = supervisorEnv.reset()
    supervisorPre.test = True
    while True:
        selectedAction = agent.choose_action_test(state)
        state, _, _, _ = supervisorEnv.step(selectedAction)
