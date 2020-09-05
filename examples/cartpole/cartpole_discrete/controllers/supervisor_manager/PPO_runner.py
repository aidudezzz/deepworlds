from numpy import convolve, ones, mean

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.PPO_agent import PPOAgent, Transition
from utilities import plotData


def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisorPre.
    supervisorPre = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard printer
    supervisorEnv = KeyboardControllerCartPole(supervisorPre)

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent = PPOAgent(supervisorPre.observationSpace, supervisorPre.actionSpace)

    episodeCount = 0
    episodeLimit = 2000
    solved = False  # Whether the solved requirement is met
    averageEpisodeActionProbs = []  # Save average episode taken actions probability to plot later

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        state = supervisorEnv.reset()  # Reset robot and get starting observation
        supervisorPre.episodeScore = 0
        actionProbs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        for step in range(supervisorPre.stepsPerEpisode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selectedAction, actionProb = agent.work(state, type_="selectAction")
            # Save the current selectedAction's probability
            actionProbs.append(actionProb)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            newState, reward, done, info = supervisorEnv.step([selectedAction])

            # Save the current state transition in agent's memory
            trans = Transition(state, selectedAction, actionProb, reward, newState)
            agent.storeTransition(trans)

            supervisorPre.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                supervisorPre.episodeScoreList.append(supervisorPre.episodeScore)
                agent.trainStep(batchSize=step + 1)
                solved = supervisorPre.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        if supervisorPre.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episodeCount, "score:", supervisorPre.episodeScore)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avgActionProb = mean(actionProbs)
        averageEpisodeActionProbs.append(avgActionProb)
        print("Avg action prob:", avgActionProb)

        episodeCount += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    movingAvgN = 10
    plotData(convolve(supervisorPre.episodeScoreList, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")
    plotData(convolve(averageEpisodeActionProbs, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "average episode action probability", "Average episode action probability over episodes")

    if not solved and not supervisorPre.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")
    print("Press R to reset.")
    state = supervisorEnv.reset()
    supervisorPre.test = True
    supervisorPre.episodeScore = 0
    while True:
        selectedAction, actionProb = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = supervisorEnv.step([selectedAction])
        supervisorPre.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisorPre.episodeScore)
            supervisorPre.episodeScore = 0
            state = supervisorEnv.reset()
