from numpy import convolve, ones

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.DDPG_agent import DDPGAgent
from utilities import plotData
from supervisor_manager import EPISODE_LIMIT, STEPS_PER_EPISODE

def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisorPre.
    supervisorPre = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard controller
    supervisorEnv = KeyboardControllerCartPole(supervisorPre)

    # The agent used here is trained with the DDPG algorithm (https://arxiv.org/abs/1509.02971).
    agent = DDPGAgent(supervisorPre.observationSpace, supervisorPre.actionSpace, lr_actor=0.000025, lr_critic=0.00025,
                      layer1_size=30, layer2_size=50, layer3_size=30, batch_size=64)

    episodeCount = 0
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < EPISODE_LIMIT:
        state = supervisorEnv.reset()  # Reset robot and get starting observation
        supervisorPre.episodeScore = 0

        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            # In training mode the agent returns the action plus OU noise for exploration
            selectedAction = agent.choose_action_train(state)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached
            # the done condition
            newState, reward, done, info = supervisorEnv.step(selectedAction)

            # Save the current state transition in agent's memory
            agent.remember(state, selectedAction, reward, newState, int(done))

            supervisorPre.episodeScore += reward  # Accumulate episode reward
            # Perform a learning step
            agent.learn()
            if done or step == STEPS_PER_EPISODE - 1:
                # Save the episode's score
                supervisorPre.episodeScoreList.append(supervisorPre.episodeScore)
                solved = supervisorPre.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        if supervisorPre.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episodeCount, "score:", supervisorPre.episodeScore)
        episodeCount += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    # this is done to smooth out the plots
    movingAvgN = 10
    plotData(convolve(supervisorPre.episodeScoreList, ones((movingAvgN,)) / movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")

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
        selectedAction = agent.choose_action_test(state)
        state, reward, done, _ = supervisorEnv.step(selectedAction)
        supervisorPre.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisorPre.episodeScore)
            supervisorPre.episodeScore = 0
            state = supervisorEnv.reset()
