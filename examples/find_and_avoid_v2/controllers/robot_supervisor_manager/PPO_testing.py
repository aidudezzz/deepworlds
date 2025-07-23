import numpy as np
import torch
import random
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy


def mask_fn(env):
    return env.get_action_mask()


def run(experiment_name, env, deterministic, use_masking, testing_results_filename=None, tests_per_difficulty=100,
        seed=None):
    # Difficulty setup
    """
    - diff_0: Corridor without obstacles, varied distance, trivial
    - diff_1: Corridor map with one row of two obstacles. Forces the agent to move around obstacles.
    - diff_2: Corridor map with two rows of four obstacles. Same as before but more difficult.
    - diff_3: Corridor map with three rows of six obstacles. Same as before but more difficult.
    - diff_4: Corridor map with four of eight obstacles. Same as before but more difficult.
    - diff_5: Random map with all available obstacles. Final complex difficulty in a more general environment.
    """
    difficulty_dict = {"diff_0": {"type": "corridor", "number_of_obstacles": 0,
                                  "min_target_dist": 1, "max_target_dist": 3},
                       "diff_1": {"type": "corridor", "number_of_obstacles": 2,
                                  "min_target_dist": 2, "max_target_dist": 2},
                       "diff_2": {"type": "corridor", "number_of_obstacles": 4,
                                  "min_target_dist": 3, "max_target_dist": 3},
                       "diff_3": {"type": "corridor", "number_of_obstacles": 6,
                                  "min_target_dist": 4, "max_target_dist": 4},
                       "diff_4": {"type": "corridor", "number_of_obstacles": 8,
                                  "min_target_dist": 5, "max_target_dist": 5},
                       "diff_5": {"type": "random", "number_of_obstacles": 25,
                                  "min_target_dist": 5, "max_target_dist": 12}}
    difficulty_keys = list(difficulty_dict.keys())

    env.reset_on_collisions = -1  # Never terminate on collisions
    env.set_maximum_episode_steps(env.maximum_episode_steps * 2)  # Double episode timeout limit
    # Set default reward weights to make rewards comparable on testing
    env.set_reward_weight_dict(target_distance_weight=0.0, target_angle_weight=0.0, dist_sensors_weight=0.0,
                               target_reach_weight=1000.0, collision_weight=1.0,
                               smoothness_weight=0.0, speed_weight=0.0)
    experiment_dir = f"./experiments/{experiment_name}"
    load_path = experiment_dir + f"/{experiment_name}_diff_5_agent"

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    try:
        model = MaskablePPO.load(load_path)  # NOQA
    except FileNotFoundError:
        load_path += ".zip"
        model = MaskablePPO.load(load_path)  # NOQA

    env.set_difficulty(difficulty_dict["diff_5"], "diff_5")
    print("################### SB3 EVALUATION STARTED ###################")
    print("Evaluating for 100 episodes in diff_5 (random map).")
    print(f"Experiment name: {experiment_name}, deterministic: {deterministic}")
    # Evaluate the agent with sb3
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=tests_per_difficulty,
                                              deterministic=deterministic, use_masking=use_masking)
    print("################### SB3 EVALUATION FINISHED ###################")
    print(f"Experiment name: {experiment_name}, deterministic: {deterministic}")
    print(f"Mean reward: {mean_reward} \n"
          f"STD reward:  {std_reward}")

    print("################### CUSTOM EVALUATION STARTED ###################")
    print(f"Experiment name: {experiment_name}, deterministic: {deterministic}")
    diff_ind = 0
    env.set_difficulty(difficulty_dict[difficulty_keys[diff_ind]], difficulty_keys[diff_ind])

    import csv
    header = [experiment_name]
    for i in range(len(difficulty_keys)):
        for j in range(tests_per_difficulty):
            header.append(f"{difficulty_keys[i]}")

    episode_rewards = ["reward"]
    done_reasons = ["done_reason"]
    steps_row = ["steps"]
    if testing_results_filename is None:
        file_name = "/testing_results.csv" if not deterministic else "/testing_results_det.csv"
    else:
        file_name = f"/{testing_results_filename}.csv" if not deterministic else f"/{testing_results_filename}_det.csv"
    with open(experiment_dir + file_name, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        steps = 0
        cumulative_rew = 0.0
        tests_count = 0
        obs = env.reset()
        successes_counter = 0
        while True:
            if use_masking:
                action_masks = mask_fn(env)
            else:
                action_masks = None
            action, _states = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            obs, rewards, done, info = env.step(action)
            steps += 1
            cumulative_rew += rewards
            if done:
                episode_rewards.append(cumulative_rew)
                steps_row.append(steps)
                if info["done_reason"] == "reached target":
                    successes_counter += 1
                print(f"{experiment_name} - Episode reward: {cumulative_rew}, steps: {steps}")
                full_test_count = (tests_per_difficulty * diff_ind) + (tests_count + 1)
                max_test_count = tests_per_difficulty * len(difficulty_keys)
                try:
                    print(f"Reached target percentage: {100 * successes_counter / full_test_count} %")
                except ZeroDivisionError:
                    print(f"Reached target percentage: {100 * successes_counter} %")
                print(f"Testing progress: "
                      f"{round((full_test_count / max_test_count) * 100.0, 2)}"
                      f"%, {full_test_count} / {max_test_count}")
                done_reasons.append(info['done_reason'])
                cumulative_rew = 0.0
                steps = 0

                tests_count += 1
                if tests_count == tests_per_difficulty:
                    diff_ind += 1
                    try:
                        env.set_difficulty(difficulty_dict[difficulty_keys[diff_ind]], key=difficulty_keys[diff_ind])
                    except IndexError:
                        print("Testing complete.")
                        break
                    tests_count = 0
                obs = env.reset()

        writer.writerow(episode_rewards)
        writer.writerow(done_reasons)
        writer.writerow(steps_row)
        writer.writerow(["sb3 rew:", mean_reward, "sb3 std:", std_reward])
    print("################### CUSTOM EVALUATION FINISHED ###################")
    # Continue running until user shuts the simulation down
    steps = 0
    cumulative_rew = 0.0
    obs = env.reset()
    while True:
        action_masks = mask_fn(env)
        action, _states = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        obs, rewards, done, info = env.step(action)
        cumulative_rew += rewards
        steps += 1
        if done:
            print(f"{experiment_name} - Episode reward: {cumulative_rew}, steps: {steps}")
            cumulative_rew = 0.0
            steps = 0
            obs = env.reset()
