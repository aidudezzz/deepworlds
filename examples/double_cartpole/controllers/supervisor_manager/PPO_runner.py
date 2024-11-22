import numpy as np

from agent.PPO_agent import PPOAgent, Transition
from supervisor_controller import DoubleCartPoleSupervisor
from utilities import plot_data

# Change these variables if needed
EPISODE_LIMIT = 10000
STEPS_PER_EPISODE = 200
NUM_ROBOTS = 2

def run(full_space=True):
    """
    Performs the training of the PPO agants and then deployes the trained agents to run in an infinite loop.

    Also plots the training results and prints progress during training.

    :param full_space: Toggle between providing each agent with the full observation space or only its own cart's data.

        - When True, each agent receives the full observation space, including the other cart's data: 
            [x_cart1, v_cart1, theta_pole1, v_pole1, x_cart2, v_cart2, theta_pole2, v_pole2].
        - When False, each agent receives only its own cart's data: [x_cart, v_cart, theta_pole, v_pole]. Deafults to True.
    :type full_space: bool
    """
    # Initialize supervisor object
    supervisor = DoubleCartPoleSupervisor(num_robots=NUM_ROBOTS)
    # Determine the dimensonality of the observation space each agent will be fed based on the `full_space` parameter
    agent_obs_dim = (
        supervisor.observation_space.shape[0] if full_space 
        else supervisor.observation_space.shape[0] // supervisor.num_robots
    )
    # The agents used here are trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    agent_1 = PPOAgent(
        agent_obs_dim,
        supervisor.action_space.n,
        clip_param=0.2,
        max_grad_norm=0.5,
        ppo_update_iters=5,
        batch_size=8,
        gamma=0.99,
        use_cuda=False,
        actor_lr=0.001,
        critic_lr=0.003,
    )
    agent_2 = PPOAgent(
        agent_obs_dim,
        supervisor.action_space.n,
        clip_param=0.2,
        max_grad_norm=0.5,
        ppo_update_iters=5,
        batch_size=8,
        gamma=0.99,
        use_cuda=False,
        actor_lr=0.001,
        critic_lr=0.003,
    )
    agents = [agent_1, agent_2]

    episode_count = 0
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < EPISODE_LIMIT:
        state = supervisor.reset()  # Reset robots and get starting observation
        supervisor.episode_score = 0
        action_probs = []
        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selected_actions, action_probs = [], []
            for i in range(NUM_ROBOTS):
                if full_space:
                    agent_state = state
                else:
                    agent_state = state[(i * agent_obs_dim): ((i + 1) * agent_obs_dim)]
                selected_action, action_prob = agents[i].work(
                    agent_state,
                    type_="selectAction"
                )
                action_probs.append(action_prob)
                selected_actions.append(selected_action)

            # Step the supervisor to get the current selected_action reward, the new state and whether we reached the
            # done condition
            new_state, reward, done, info = supervisor.step(
                [*selected_actions]
            )

            # Save the current state transitions from all robots in agent's memory
            for i in range(NUM_ROBOTS):
                if full_space:
                    agent_state = state
                    agent_new_state = new_state
                else:
                    agent_state = state[(i * agent_obs_dim): ((i + 1) * agent_obs_dim)]
                    agent_new_state = new_state[(i * agent_obs_dim): ((i + 1) * agent_obs_dim)]
                agents[i].store_transition(
                    Transition(
                        agent_state,
                        selected_actions[i],
                        action_probs[i],
                        reward[i],
                        agent_new_state,
                    )
                )
            # Accumulate episode reward
            supervisor.episode_score += np.array(reward)

            if done:
                # Save the episode's score
                supervisor.episode_score_list.append(
                    supervisor.episode_score
                )
                # Perform a training step
                for i in range(NUM_ROBOTS):
                    agents[i].train_step(batch_size=step + 1)
                # Check whether the task is solved
                solved = supervisor.solved()
                break

            state = new_state  # state for next step is current step's new_state

        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        avg_action_prob = [
            round(np.mean(action_probs[i]), 4) for i in range(NUM_ROBOTS)
        ]
        print(
            f"Episode: {episode_count} Score = {supervisor.episode_score} | Average Action Probabilities = {avg_action_prob}"
        )

        episode_count += 1  # Increment episode counter

    moving_avg_n = 10
    plot_data(
        np.convolve(
            np.array(supervisor.episode_score_list).T[0],
            np.ones((moving_avg_n,)) / moving_avg_n,
            mode='valid',
        ),
        "episode",
        "episode score",
        "Episode scores over episodes", save=True, save_name="reward.png"
    )

    if not solved:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")

    state = supervisor.reset()
    supervisor.episode_score = 0
    while True:
        selected_actions = []
        action_probs = []
        for i in range(NUM_ROBOTS):
            if full_space:
                agent_state = state
            else:
                agent_state = state[(i * agent_obs_dim): ((i + 1) * agent_obs_dim)]
            selected_action, action_prob = agents[i].work(
                agent_state, # state[0:4] for 1st robot, state[4:8] for 2nd robot
                type_="selectAction"
            )
            action_probs.append(action_prob)
            selected_actions.append(selected_action)

        state, reward, done, _ = supervisor.step(selected_actions)
        supervisor.episode_score += np.array(reward)  # Accumulate episode reward

        if done:
            print(f"Reward accumulated = {supervisor.episode_score}")
            supervisor.episode_score = 0
            state = supervisor.reset()
