from numpy import convolve, ones, mean

from robot_supervisor import CartPoleRobotSupervisor
from agent.PPO_agent import PPOAgent, Transition
from utilities import plotData


def run():
    # Initialize supervisor object
    env = CartPoleRobotSupervisor()

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    # We pass 4 as numberOfInputs and 2 as numberOfOutputs, taken from the gym spaces
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)

    episodeCount = 0
    episodeLimit = 2000
    solved = False  # Whether the solved requirement is met
    averageEpisodeActionProbs = []  # Save average episode taken actions probability to plot later

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        state = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0
        actionProbs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        for step in range(env.stepsPerEpisode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selectedAction, actionProb = agent.work(state, type_="selectAction")
            # Save the current selectedAction's probability
            actionProbs.append(actionProb)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            newState, reward, done, info = env.step([selectedAction])

            # Save the current state transition in agent's memory
            trans = Transition(state, selectedAction, actionProb, reward, newState)
            agent.storeTransition(trans)

            env.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                agent.trainStep(batchSize=step + 1)
                solved = env.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avgActionProb = mean(actionProbs)
        averageEpisodeActionProbs.append(avgActionProb)
        print("Avg action prob:", avgActionProb)

        episodeCount += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    movingAvgN = 10
    plotData(convolve(env.episodeScoreList, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")
    plotData(convolve(averageEpisodeActionProbs, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "average episode action probability", "Average episode action probability over episodes")

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episodeScore = 0
    while True:
        selectedAction, actionProb = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = env.step([selectedAction])
        env.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episodeScore)
            env.episodeScore = 0
            state = env.reset()
