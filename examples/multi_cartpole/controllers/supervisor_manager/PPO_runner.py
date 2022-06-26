from numpy import convolve, ones, mean

from supervisor_controller import CartPoleSupervisor
from agent.PPO_agent import PPOAgent, Transition
from utilities import plotData

#Change these variables if needed
EPISODE_LIMIT = 2000
STEPS_PER_EPISODE = 200
NUM_ROBOTS = 9

def run():
    # Initialize supervisor object
    supervisorEnv = CartPoleSupervisor()

    episodeCount = 0

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent = PPOAgent(supervisorEnv.observationSpace, supervisorEnv.actionSpace)

    solved = False                                    # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < EPISODE_LIMIT:
        state = supervisorEnv.reset()                  # Reset robots and get starting observation)
        supervisorEnv.episodeScore = 0
        actionProbs = [[] for i in range(NUM_ROBOTS)]

        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selectedActions = []
            for i in range(NUM_ROBOTS):
                selectedAction, actionProb = agent.work(state[i], type_="selectAction")
                actionProbs[i].append(actionProb)
                selectedActions.append(selectedAction)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            newState, reward, done, info = supervisorEnv.step([*selectedActions])

            # Save the current state transitions from all robots in agent's memory
            for i in range(NUM_ROBOTS):
                trans = Transition(state[i], selectedActions[i], actionProbs[i][-1], reward, newState[i])
                agent.storeTransition(trans)

            supervisorEnv.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                supervisorEnv.episodeScoreList.append(supervisorEnv.episodeScore)
                agent.trainStep(batchSize=step + 1)
                solved = supervisorEnv.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        avgActionProb = [round(mean(actionProbs[i]), 4) for i in range(NUM_ROBOTS)]
        
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        print(f"Episode: {episodeCount} Score = {supervisorEnv.episodeScore} | Average Action Probabilities = {avgActionProb}")

        episodeCount += 1  # Increment episode counter
    
    movingAvgN = 10
    plotData(convolve(supervisorEnv.episodeScoreList, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")

    if not solved and not supervisorEnv.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")

    state = supervisorEnv.reset()
    supervisorEnv.test = True
    supervisorEnv.episodeScore = 0
    while True:
        selectedActions = []
        for i in range(NUM_ROBOTS):
            selectedAction, actionProb = agent.work(state[i], type_="selectAction")
            actionProbs[i].append(actionProb)
            selectedActions.append(selectedAction)

        state, reward, done, _ = supervisorEnv.step(selectedActions)
        supervisorEnv.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisorEnv.episodeScore)
            supervisorEnv.episodeScore = 0
            state = supervisorEnv.reset()
