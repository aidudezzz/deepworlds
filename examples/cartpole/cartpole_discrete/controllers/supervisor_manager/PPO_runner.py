from numpy import convolve, ones, mean

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.PPO_agent import PPOAgent, Transition
from utilities import plot_data


def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisor_pre.
    supervisor_pre = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard printer
    supervisor_env = KeyboardControllerCartPole(supervisor_pre)

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent = PPOAgent(supervisor_pre.observation_space, supervisor_pre.action_space)

    episode_count = 0
    episod_limit = 2000
    solved = False  # Whether the solved requirement is met
    average_episode_action_probs = []  # Save average episode taken actions probability to plot later

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < episod_limit:
        state = supervisor_env.reset()  # Reset robot and get starting observation
        supervisor_pre.episodeScore = 0
        action_probs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        for step in range(supervisor_pre.steps_per_episode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selected_action, action_prob = agent.work(state, type_="selectAction")
            # Save the current selected_action's probability
            action_probs.append(action_prob)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached the
            # done condition
            new_state, reward, done, info = supervisor_env.step([selected_action])

            # Save the current state transition in agent's memory
            trans = Transition(state, selected_action, action_prob, reward, new_state)
            agent.store_transition(trans)

            supervisor_pre.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                supervisor_pre.episode_score_list.append(supervisor_pre.episodeScore)
                agent.train_step(batch_size=step + 1)
                solved = supervisor_pre.solved()  # Check whether the task is solved
                break

            state = new_state  # state for next step is current step's new_state

        if supervisor_pre.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episode_count, "score:", supervisor_pre.episodeScore)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avgaction_prob = mean(action_probs)
        average_episode_action_probs.append(avgaction_prob)
        print("Avg action prob:", avgaction_prob)

        episode_count += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    moving_avg_n = 10
    plot_data(convolve(supervisor_pre.episode_score_list, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")
    plot_data(convolve(average_episode_action_probs, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             "episode", "average episode action probability", "Average episode action probability over episodes")

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
        selected_action, action_prob = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = supervisor_env.step([selected_action])
        supervisor_pre.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisor_pre.episodeScore)
            supervisor_pre.episodeScore = 0
            state = supervisor_env.reset()
