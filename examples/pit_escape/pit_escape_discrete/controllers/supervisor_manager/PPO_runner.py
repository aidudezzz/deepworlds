from numpy import convolve, ones, mean

from supervisor_controller import PitEscapeSupervisor
from keyboard_controller_pit_escape import KeyboardControllerPitEscape
from agent.PPOAgent import PPOAgent, Transition
from utilities import plotData
from supervisor_manager import EPISODE_LIMIT


def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisorPre.
    supervisorPre = PitEscapeSupervisor()
    # Wrap the Pit Escape supervisor in the custom keyboard printer
    supervisorEnv = KeyboardControllerPitEscape(supervisorPre)

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent = PPOAgent(supervisorPre.observationSpace, supervisorPre.actionSpace)

    episodeCount = 0
    solved = False  # Whether the solved requirement is met
    repeatActionSteps = 1  # Amount of steps for which to repeat a certain action
    averageEpisodeActionProbs = []  # Save average episode taken actions probability to plot later

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < EPISODE_LIMIT:
        state = supervisorEnv.reset()  # Reset robot and get starting observation
        supervisorPre.episodeScore = 0
        actionProbs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        step = 0
        # Episode is terminated based on time elapsed and not on number of steps
        while True:
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            actionValues, actionProb = agent.work(state, type_="selectAction")
            # Save the current selectedAction's probability
            actionProbs.append(actionProb)

            # Step the supervisor to get the current action's reward, the new state and whether we reached the done
            # condition
            newState, reward, done, info = supervisorEnv.step([actionValues], repeatActionSteps)

            # Save the current state transition in agent's memory
            trans = Transition(state, actionValues, actionProb, reward, newState)
            agent.storeTransition(trans)

            supervisorPre.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                supervisorPre.episodeScoreList.append(supervisorPre.episodeScore)
                agent.trainStep(batchSize=step + 1)
                solved = supervisorPre.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState
            step += 1

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
        actionValues, _ = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = supervisorEnv.step([actionValues], repeatActionSteps)
        supervisorPre.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisorPre.episodeScore)
            supervisorPre.episodeScore = 0
            state = supervisorEnv.reset()
