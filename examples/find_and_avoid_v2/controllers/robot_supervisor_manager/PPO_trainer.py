import os
import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from find_and_avoid_v2_robot_supervisor import FindAndAvoidV2RobotSupervisor


class AdditionalInfoCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, experiment_name, env, current_difficulty=None, verbose=1):
        super(AdditionalInfoCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.experiment_name = experiment_name
        self.current_difficulty = current_difficulty
        self.env = env

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.logger.record("experiment_name", self.experiment_name)
        self.logger.record("difficulty", self.current_difficulty)
        hparam_dict = {
            "experiment_name": self.experiment_name,
            "difficulty": self.current_difficulty,
            "algorithm": self.model.__class__.__name__,
            "gamma": self.model.gamma,  # NOQA
            "gae_lambda": self.model.gae_lambda,  # NOQA
            "target_kl": self.model.target_kl,  # NOQA
            "vf_coef": self.model.vf_coef,  # NOQA
            "ent_coef": self.model.ent_coef,  # NOQA
            "n_steps": self.model.n_steps,  # NOQA
            "batch_size": self.model.batch_size,  # NOQA
            "maximum_episode_steps": self.env.maximum_episode_steps,
            "step_window": self.env.step_window,
            "seconds_window": self.env.seconds_window,
            "add_action_to_obs": self.env.add_action_to_obs,
            "reset_on_collisions": self.env.reset_on_collisions,
            "on_target_threshold": self.env.on_target_threshold,
            "ds_type": self.env.ds_type,
            "ds_noise": self.env.ds_noise,
            "target_distance_weight": self.env.reward_weight_dict["dist_tar"],
            "tar_angle_weight": self.env.reward_weight_dict["ang_tar"],
            "dist_sensors_weight": self.env.reward_weight_dict["dist_sensors"],
            "tar_reach_weight": self.env.reward_weight_dict["tar_reach"],
            "collision_weight": self.env.reward_weight_dict["collision"],
            "smoothness_weight": self.env.reward_weight_dict["smoothness_weight"],
            "speed_weight": self.env.reward_weight_dict["speed_weight"],

        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorboard will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/collision termination count": 0.0,
            "rollout/reach target count": 0.0,
            "rollout/timeout count": 0.0,
            "rollout/reset count": 0.0,
            "rollout/success percentage": 0.0,
            "rollout/smoothness": 0.0,
            "rollout/min_dist_reached": 0.0,
            "learning rate": self.model.learning_rate,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.logger.record("general/experiment_name", self.experiment_name)
        self.logger.record("general/difficulty", self.current_difficulty)
        self.logger.record("rollout/reset count", self.env.reset_count)
        self.logger.record("rollout/reach target count", self.env.reach_target_count)
        self.logger.record("rollout/collision termination count", self.env.collision_termination_count)
        self.logger.record("rollout/timeout count", self.env.timeout_count)
        if len(self.env.smoothness_list) >= 1:
            self.logger.record("rollout/smoothness", float(np.mean(self.env.smoothness_list)))
        else:
            self.logger.record("rollout/smoothness", -1.0)
        if len(self.env.min_dist_reached_list) > 1:
            self.logger.record("rollout/min_dist_reached", np.mean(self.env.min_dist_reached_list))
        else:
            self.logger.record("rollout/min_dist_reached", 0.0)
        if self.env.reach_target_count == 0 or self.env.reset_count == 0:
            self.logger.record("rollout/success percentage", 0.0)
        else:
            self.logger.record("rollout/success percentage",
                               self.env.reach_target_count / self.env.reset_count)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.env.clear_smoothness_list()
        self.env.clear_min_dist_reached_list()
        pass


def mask_fn(env):
    return env.get_action_mask()


def run(experiment_name="experiment", experiment_description="", manual_control=False, only_test=False,
        maximum_episode_steps=16_384, step_window=1, seconds_window=1, add_action_to_obs=True, ds_params=None,
        reset_on_collisions=4_096, on_tar_threshold=0.1,
        target_dist_weight=1.0, target_angle_weight=1.0, dist_sensors_weight=10.0, target_reach_weight=1000.0,
        collision_weight=100.0, smoothness_weight=0.0, speed_weight=0.0,
        map_w=7, map_h=7, cell_size=None, seed=None,
        net_arch=None, gamma=0.999, gae_lambda=0.95, target_kl=None, vf_coef=0.5, ent_coef=0.001, difficulty_dict=None,
        n_steps=2048, batch_size=64, lr_rate=None, ds_denial_list=None):

    experiment_dir = f"./experiments/{experiment_name}"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if lr_rate is None:
        lr_rate = lambda f: f * 3e-4  # NOQA Linear decreasing schedule

    if net_arch is None:
        net_arch = dict(pi=[1024, 512, 256], vf=[2048, 1024, 512])  # Actor-critic layers

    if ds_params is None:
        ds_type = "generic"
        ds_n_rays = 1
        ds_aperture = 1.57
        ds_resolution = -1.0
        ds_noise = 0.0
        max_ds_range = 100.0
    else:
        ds_type = ds_params["ds_type"]
        ds_n_rays = ds_params["ds_n_rays"]
        ds_aperture = ds_params["ds_aperture"]
        ds_resolution = ds_params["ds_resolution"]
        ds_noise = ds_params["ds_noise"]
        max_ds_range = ds_params["max_ds_range"]

    env = \
        Monitor(
            ActionMasker(
                FindAndAvoidV2RobotSupervisor(experiment_description, maximum_episode_steps, step_window=step_window,
                                              seconds_window=seconds_window,
                                              add_action_to_obs=add_action_to_obs, max_ds_range=max_ds_range,
                                              reset_on_collisions=reset_on_collisions, manual_control=manual_control,
                                              on_target_threshold=on_tar_threshold,
                                              ds_type=ds_type, ds_n_rays=ds_n_rays, ds_aperture=ds_aperture,
                                              ds_resolution=ds_resolution, ds_noise=ds_noise,
                                              ds_denial_list=ds_denial_list,
                                              target_distance_weight=target_dist_weight,
                                              target_angle_weight=target_angle_weight,
                                              dist_sensors_weight=dist_sensors_weight,
                                              target_reach_weight=target_reach_weight,
                                              collision_weight=collision_weight,
                                              smoothness_weight=smoothness_weight, speed_weight=speed_weight,
                                              map_width=map_w, map_height=map_h, cell_size=cell_size, seed=seed),
                action_mask_fn=mask_fn  # NOQA
            )
        )
    if not only_test:
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        env.export_parameters(experiment_dir + f"/{experiment_name}_parameters.json",
                              net_arch, gamma, gae_lambda, target_kl, vf_coef, ent_coef, n_steps, batch_size)

        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=net_arch)
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=n_steps, batch_size=batch_size,
                            gamma=gamma, gae_lambda=gae_lambda, target_kl=target_kl,
                            vf_coef=vf_coef, ent_coef=ent_coef, learning_rate=lr_rate,
                            verbose=1, tensorboard_log=experiment_dir)

        difficulty_keys = list(difficulty_dict.keys())
        printing_callback = AdditionalInfoCallback(experiment_name, env, verbose=1,
                                                   current_difficulty=difficulty_keys[0])

        for diff_key in difficulty_keys:
            env.set_difficulty(difficulty_dict[diff_key])
            printing_callback.current_difficulty = diff_key
            model.learn(total_timesteps=difficulty_dict[diff_key]["total_timesteps"], tb_log_name=diff_key,
                        reset_num_timesteps=False, callback=printing_callback)
            model.save(experiment_dir + f"/{experiment_name}_{diff_key}_agent")
        print("################### TRAINING FINISHED ###################")
    else:
        print(f"Training skipped, only testing for experiment: {experiment_name}.")
    return env
