from numpy import convolve, ones, mean

from supervisor_controller import PitEscapeSupervisor
from keyboard_controller_pit_escape import KeyboardControllerPitEscape
from agent.PPOAgent import PPOAgent, Transition
from utilities import plot_data
from supervisor_manager import EPISODE_LIMIT


def run():
    # Initialize supervisor object
    # Whenever we want to access attributes, etc., from the supervisor controller we use
    # supervisor_pre.
    supervisor_pre = PitEscapeSupervisor()
    # Wrap the Pit Escape supervisor in the custom keyboard printer
    supervisor_env = KeyboardControllerPitEscape(supervisor_pre)

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent = PPOAgent(supervisor_pre.observation_space, supervisor_pre.action_space)

    episode_count = 0
    solved = False  # Whether the solved requirement is met
    repeat_action_steps = 1  # Amount of steps for which to repeat a certain action
    average_episode_action_probs = []  # Save average episode taken actions probability to plot later

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < EPISODE_LIMIT:
        state = supervisor_env.reset()  # Reset robot and get starting observation
        supervisor_pre.episodeScore = 0
        action_probs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        step = 0
        # Episode is terminated based on time elapsed and not on number of steps
        while True:
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            action_values, action_prob = agent.work(state, type_="selectAction")
            # Save the current selectedAction's probability
            action_probs.append(action_prob)

            # Step the supervisor to get the current action's reward, the new state and whether we reached the done
            # condition
            new_state, reward, done, info = supervisor_env.step([action_values], repeat_action_steps)

            # Save the current state transition in agent's memory
            trans = Transition(state, action_values, action_prob, reward, new_state)
            agent.store_transition(trans)

            supervisor_pre.episodeScore += reward  # Accumulate episode reward
            if done:
                # Save the episode's score
                supervisor_pre.episode_score_list.append(supervisor_pre.episodeScore)
                agent.train_step(batch_size=step + 1)
                solved = supervisor_pre.solved()  # Check whether the task is solved
                break

            state = new_state  # state for next step is current step's new_state
            step += 1

        if supervisor_pre.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episode_count, "score:", supervisor_pre.episodeScore)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avg_action_prob = mean(action_probs)
        average_episode_action_probs.append(avg_action_prob)
        print("Avg action prob:", avg_action_prob)

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
        action_values, _ = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = supervisor_env.step([action_values], repeat_action_steps)
        supervisor_pre.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", supervisor_pre.episodeScore)
            supervisor_pre.episodeScore = 0
            state = supervisor_env.reset()
