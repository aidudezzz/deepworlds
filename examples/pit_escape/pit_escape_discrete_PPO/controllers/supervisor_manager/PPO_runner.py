from numpy import convolve, ones, mean
from supervisor_controller import PitEscapeSupervisor
from keyboard_controller_pit_escape import KeyboardControllerPitEscape
from agent.PPOAgent import Transition
from utilities import plotData


# Initialize supervisor object
supervisor = PitEscapeSupervisor()
# Wrap the CartPole supervisor in the custom keyboard printer
supervisor = KeyboardControllerPitEscape(supervisor)

solved = False  # Whether the solved requirement is met
repeatActionSteps = 1  # Amount of steps for which to repeat a certain action
averageEpisodeActionProbs = []  # Save average episode taken actions probability to plot later

# Run outer loop until the episodes limit is reached or the task is solved
while not solved and supervisor.controller.episodeCount < supervisor.controller.episodeLimit:
    state = supervisor.controller.reset()  # Reset robot and get starting observation
    supervisor.controller.episodeScore = 0
    actionProbs = []  # This list holds the probability of each chosen action

    # Inner loop is the episode loop
    step = 0
    # Episode is terminated based on time elapsed
    while True:
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        actionValues, actionProb = supervisor.controller.agent.work(state, type_="selectAction")
        # Save the current selectedAction's probability
        actionProbs.append(actionProb)

        # Step the supervisor to get the current action's reward, the new state and whether we reached the done
        # condition
        newState, reward, done, info = supervisor.step([actionValues], repeatActionSteps)

        # Save the current state transition in agent's memory
        trans = Transition(state, actionValues, actionProb, reward, newState)
        supervisor.controller.agent.storeTransition(trans)

        supervisor.controller.episodeScore += reward  # Accumulate episode reward
        if done:
            # Save the episode's score
            supervisor.controller.episodeScoreList.append(supervisor.controller.episodeScore)
            supervisor.controller.agent.trainStep(batchSize=step + 1)
            solved = supervisor.controller.solved()  # Check whether the task is solved
            break

        state = newState  # state for next step is current step's newState
        step += 1

    if supervisor.controller.test:  # If test flag is externally set to True, agent is deployed
        break

    print("Episode #", supervisor.controller.episodeCount, "score:", supervisor.controller.episodeScore)
    # The average action probability tells us how confident the agent was of its actions.
    # By looking at this we can check whether the agent is converging to a certain policy.
    avgActionProb = mean(actionProbs)
    averageEpisodeActionProbs.append(avgActionProb)
    print("Avg action prob:", avgActionProb)

    supervisor.controller.episodeCount += 1  # Increment episode counter

# np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
movingAvgN = 10
plotData(convolve(supervisor.controller.episodeScoreList, ones((movingAvgN,)) / movingAvgN, mode='valid'),
         "episode", "episode score", "Episode scores over episodes")
plotData(convolve(averageEpisodeActionProbs, ones((movingAvgN,)) / movingAvgN, mode='valid'),
         "episode", "average episode action probability", "Average episode action probability over episodes")

if not solved and not supervisor.controller.test:
    print("Reached episode limit and task was not solved.")
else:
    if not solved:
        print("Task is not solved, deploying agent for testing...")
    elif solved:
        print("Task is solved, deploying agent for testing...")
    state = supervisor.controller.reset()
    supervisor.controller.test = True
    while True:
        # print(state)
        actionValues, _ = supervisor.controller.agent.work(state, type_="selectActionMax")
        # print(actionValues)
        state, _, done, _ = supervisor.step([actionValues], repeatActionSteps)

        if done:
            supervisor.controller.reset()
