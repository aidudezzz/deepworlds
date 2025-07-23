from numpy import convolve, ones

from robot_supervisor import CartPoleRobotSupervisor
from agent.DDPG_agent import DDPGAgent
from utilities import plot_data

from robot_supervisor_manager import EPISODE_LIMIT, STEPS_PER_EPISODE

def run():
    # Initialize supervisor object
    env = CartPoleRobotSupervisor()

    # The agent used here is trained with the DDPG algorithm (https://arxiv.org/abs/1509.02971).
    # We pass (4, ) as numberOfInputs and (2, ) as numberOfOutputs, taken from the gym spaces
    agent = DDPGAgent(env.observation_space.shape, env.action_space.shape, lr_actor=0.000025, lr_critic=0.00025,
                      layer1_size=30, layer2_size=50, layer3_size=30, batch_size=64)

    episode_count = 0
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < EPISODE_LIMIT:
        state = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0

        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            # In training mode the agent returns the action plus OU noise for exploration
            selected_action = agent.choose_action_train(state)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached
            # the done condition
            new_state, reward, done, info = env.step(selected_action)

            # Save the current state transition in agent's memory
            agent.remember(state, selected_action, reward, new_state, int(done))

            env.episode_score += reward  # Accumulate episode reward
            # Perform a learning step
            agent.learn()
            if done or step == STEPS_PER_EPISODE - 1:
                # Save the episode's score
                env.episode_score_list.append(env.episode_score)
                solved = env.solved()  # Check whether the task is solved
                break

            state = new_state  # state for next step is current step's new_state

        print("Episode #", episode_count, "score:", env.episode_score)
        episode_count += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    # this is done to smooth out the plots
    moving_avg_n = 10
    plot_data(convolve(env.episode_score_list, ones((moving_avg_n,)) / moving_avg_n, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episode_score = 0
    while True:
        selected_action = agent.choose_action_test(state)
        state, reward, done, _ = env.step(selected_action)
        env.episode_score += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episode_score)
            env.episode_score = 0
            state = env.reset()
