from numpy import convolve, ones, mean

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.PPO_agent import PPOAgent, Transition
from utilities import plotData


def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisorPre.
    supervisorEnv = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard printer
    # supervisorEnv = KeyboardControllerCartPole(supervisorPre)

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent = PPOAgent(supervisorEnv.observationSpace, supervisorEnv.actionSpace)

    episodeCount = 0
    episodeLimit = 2000
    solved = False  # Whether the solved requirement is met
    averageEpisodeActionProbs = []

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        state = supervisorEnv.reset()  # Reset robots and get starting observation")
        supervisorEnv.episodeScore = 0
        actionProbs = [[] for i in range(9)]

        # Inner loop is the episode loop
        for step in range(supervisorEnv.stepsPerEpisode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            #selectedAction1, actionProb1 = agent.work(state[0], type_="selectAction")  # Robot 1
            #selectedAction2, actionProb2 = agent.work(state[1], type_="selectAction")  # Robot 2
            # Save the current selectedAction's probabilities
            #actionProbs1.append(actionProb1)
            #actionProbs2.append(actionProb2)
            selectedActions = []
            for i in range(9):
                selectedAction, actionProb = agent.work(state[i], type_="selectAction")
                actionProbs[i].append(actionProb)
                selectedActions.append(selectedAction)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            newState, reward, done, info = supervisorEnv.step(["actions", *selectedActions])

            # Save the current state transitions from both robots in agent's memory
            for i in range(9):
                trans = Transition(state[i], selectedActions[i], actionProbs[i][-1], reward, newState[i])
                agent.storeTransition(trans)
            #trans = Transition(state[0], selectedAction1, actionProb1, reward, newState[0])
            #agent.storeTransition(trans)
            #trans = Transition(state[1], selectedAction2, actionProb2, reward, newState[1])
            #agent.storeTransition(trans)

            supervisorEnv.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                supervisorEnv.episodeScoreList.append(supervisorEnv.episodeScore)
                agent.trainStep(batchSize=step + 1)
                solved = supervisorEnv.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        if supervisorEnv.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episodeCount, "score:", supervisorEnv.episodeScore)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avgActionProb = [mean(actionProbs[i]) for i in range(9)]

        #avgActionProb1 = mean(actionProbs1)
        #avgActionProb2 = mean(actionProbs2)
        averageEpisodeActionProbs = [avgActionProb[i] for i in range(9)]
        #averageEpisodeActionProbsR1.append(avgActionProb1)
        #averageEpisodeActionProbsR2.append(avgActionProb2)
        print("Avg action prob 1:", avgActionProb[0])
        print("Avg action prob 2:", avgActionProb[1])

        episodeCount += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    movingAvgN = 10
    #plotData(convolve(supervisorEnv.episodeScoreList, ones((movingAvgN,))/movingAvgN, mode='valid'),
    #         "episode", "episode score", "Episode scores over episodes")
    #plotData(convolve(averageEpisodeActionProbsR1, ones((movingAvgN,))/movingAvgN, mode='valid'),
    #         "episode", "average episode action probability 1", "Average episode action probability over episodes 1")
    #plotData(convolve(averageEpisodeActionProbsR2, ones((movingAvgN,))/movingAvgN, mode='valid'),
    #         "episode", "average episode action probability 2", "Average episode action probability over episodes 2")

    if not solved and not supervisorEnv.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")
    print("Press R to reset.")
    state = supervisorEnv.reset()
    supervisorEnv.test = True
    supervisorEnv.episodeScore = 0
    while True:
        selectedAction1, actionProb1 = agent.work(state[0], type_="selectActionMax")  # Robot 1
        selectedAction2, actionProb2 = agent.work(state[1], type_="selectActionMax")  # Robot 2

        state, reward, done, _ = supervisorEnv.step([selectedAction1, selectedAction2])
        supervisorEnv.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisorEnv.episodeScore)
            supervisorEnv.episodeScore = 0
            state = supervisorEnv.reset()

'''
from numpy import convolve, ones, mean

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.PPO_agent import PPOAgent, Transition
from utilities import plotData


def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisorPre.
    supervisorEnv = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard printer
    # supervisorEnv = KeyboardControllerCartPole(supervisorPre)

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent = PPOAgent(supervisorEnv.observationSpace, supervisorEnv.actionSpace)

    episodeCount = 0
    episodeLimit = 2000
    solved = False  # Whether the solved requirement is met
    averageEpisodeActionProbsR1 = []  # Save average episode taken actions probability to plot later
    averageEpisodeActionProbsR2 = []

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        state = supervisorEnv.reset()  # Reset robots and get starting observation")
        supervisorEnv.episodeScore = 0
        actionProbs1 = []  # These lists hold the probabilities of each chosen action
        actionProbs2 = []

        # Inner loop is the episode loop
        for step in range(supervisorEnv.stepsPerEpisode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selectedAction1, actionProb1 = agent.work(state[0], type_="selectAction")  # Robot 1
            selectedAction2, actionProb2 = agent.work(state[1], type_="selectAction")  # Robot 2
            # Save the current selectedAction's probabilities
            actionProbs1.append(actionProb1)
            actionProbs2.append(actionProb2)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            newState, reward, done, info = supervisorEnv.step(["actions", selectedAction1, selectedAction2])

            # Save the current state transitions from both robots in agent's memory
            trans = Transition(state[0], selectedAction1, actionProb1, reward, newState[0])
            agent.storeTransition(trans)
            trans = Transition(state[1], selectedAction2, actionProb2, reward, newState[1])
            agent.storeTransition(trans)

            supervisorEnv.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                supervisorEnv.episodeScoreList.append(supervisorEnv.episodeScore)
                agent.trainStep(batchSize=step + 1)
                solved = supervisorEnv.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        if supervisorEnv.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episodeCount, "score:", supervisorEnv.episodeScore)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avgActionProb1 = mean(actionProbs1)
        avgActionProb2 = mean(actionProbs2)
        averageEpisodeActionProbsR1.append(avgActionProb1)
        averageEpisodeActionProbsR2.append(avgActionProb2)
        print("Avg action prob 1:", avgActionProb1)
        print("Avg action prob 2:", avgActionProb2)

        episodeCount += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    movingAvgN = 10
    plotData(convolve(supervisorEnv.episodeScoreList, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")
    plotData(convolve(averageEpisodeActionProbsR1, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "average episode action probability 1", "Average episode action probability over episodes 1")
    plotData(convolve(averageEpisodeActionProbsR2, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "average episode action probability 2", "Average episode action probability over episodes 2")

    if not solved and not supervisorEnv.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")
    print("Press R to reset.")
    state = supervisorEnv.reset()
    supervisorEnv.test = True
    supervisorEnv.episodeScore = 0
    while True:
        selectedAction1, actionProb1 = agent.work(state[0], type_="selectActionMax")  # Robot 1
        selectedAction2, actionProb2 = agent.work(state[1], type_="selectActionMax")  # Robot 2

        state, reward, done, _ = supervisorEnv.step([selectedAction1, selectedAction2])
        supervisorEnv.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisorEnv.episodeScore)
            supervisorEnv.episodeScore = 0
            state = supervisorEnv.reset()
'''
