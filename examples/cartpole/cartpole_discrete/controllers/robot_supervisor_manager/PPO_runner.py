from numpy import convolve, ones, mean

from robot_supervisor import CartPoleRobotSupervisor
from agent.PPO_agent import PPOAgent, Transition
from utilities import plot_data


def run():
    # Initialize supervisor object
    env = CartPoleRobotSupervisor()

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    # We pass 4 as numberOfInputs and 2 as numberOfOutputs, taken from the gym spaces
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)

    episode_count = 0
    episode_limit = 2000
    solved = False  # Whether the solved requirement is met
    average_episode_action_probs = []  # Save average episode taken actions probability to plot later

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < episode_limit:
        state = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0
        action_probs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        for step in range(env.steps_per_episode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selected_action, action_prob = agent.work(state, type_="selectAction")
            # Save the current selected_action's probability
            action_probs.append(action_prob)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached the
            # done condition
            new_state, reward, done, info = env.step([selected_action])

            # Save the current state transition in agent's memory
            trans = Transition(state, selected_action, action_prob, reward, new_state)
            agent.store_transition(trans)

            env.episode_score += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                env.episode_score_list.append(env.episode_score)
                agent.train_step(batch_size=step + 1)
                solved = env.solved()  # Check whether the task is solved
                break

            state = new_state  # state for next step is current step's new_state

        print("Episode #", episode_count, "score:", env.episode_score)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avg_action_prob = mean(action_probs)
        average_episode_action_probs.append(avg_action_prob)
        print("Avg action prob:", avg_action_prob)

        episode_count += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    moving_avg_n = 10
    plot_data(convolve(env.episode_score_list, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")
    plot_data(convolve(average_episode_action_probs, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             "episode", "average episode action probability", "Average episode action probability over episodes")

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episode_score = 0
    while True:
        selected_action, action_prob = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = env.step([selected_action])
        env.episode_score += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episode_score)
            env.episode_score = 0
            state = env.reset()
