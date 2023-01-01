from numpy import convolve, ones

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.DDPG_agent import DDPGAgent
from utilities import plot_data
from supervisor_manager import EPISODE_LIMIT, STEPS_PER_EPISODE


def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisor_pre.
    supervisor_pre = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard controller
    supervisor_env = KeyboardControllerCartPole(supervisor_pre)

    # The agent used here is trained with the DDPG algorithm (https://arxiv.org/abs/1509.02971).
    agent = DDPGAgent((supervisor_pre.observation_space,), (supervisor_pre.action_space, ),
                      lr_actor=0.000025, lr_critic=0.00025,
                      layer1_size=30, layer2_size=50, layer3_size=30, batch_size=64)

    episode_count = 0
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < EPISODE_LIMIT:
        state = supervisor_env.reset()  # Reset robot and get starting observation
        supervisor_pre.episodeScore = 0

        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            # In training mode the agent returns the action plus OU noise for exploration
            selected_action = agent.choose_action_train(state)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached
            # the done condition
            new_state, reward, done, info = supervisor_env.step(selected_action)

            # Save the current state transition in agent's memory
            agent.remember(state, selected_action, reward, new_state, int(done))

            supervisor_pre.episodeScore += reward  # Accumulate episode reward
            # Perform a learning step
            agent.learn()
            if done or step == STEPS_PER_EPISODE - 1:
                # Save the episode's score
                supervisor_pre.episode_score_list.append(supervisor_pre.episodeScore)
                solved = supervisor_pre.solved()  # Check whether the task is solved
                break

            state = new_state  # state for next step is current step's new_state

        if supervisor_pre.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episode_count, "score:", supervisor_pre.episodeScore)
        episode_count += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    # this is done to smooth out the plots
    moving_avg_n = 10
    plot_data(convolve(supervisor_pre.episode_score_list, ones((moving_avg_n,)) / moving_avg_n, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")

    if not solved and not supervisor_pre.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")
    print("Press R to reset.")
    state = supervisor_env.reset()
    supervisor_pre.test = True
    supervisor_pre.episodeScore = 0
    while True:
        selected_action = agent.choose_action_test(state)
        state, reward, done, _ = supervisor_env.step(selected_action)
        supervisor_pre.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisor_pre.episodeScore)
            supervisor_pre.episodeScore = 0
            state = supervisor_env.reset()
