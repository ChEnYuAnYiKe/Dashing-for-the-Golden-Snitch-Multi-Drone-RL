import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardCheckCallback(BaseCallback):
    def __init__(
        self,
        verbose=0,
        num_env=None,
    ):
        super(RewardCheckCallback, self).__init__(verbose)
        # for cal dones, get episode
        if num_env is None:
            raise ValueError("num_env cannot be None. Please specify the number of environments.")

        # init
        self.num_nev = num_env
        self.n_episodes = 0
        self.episode_prog_rewards = np.zeros(self.num_nev)
        self.episode_command_rewards = np.zeros(self.num_nev)
        self.episode_crash_rewards = np.zeros(self.num_nev)
        self.episode_drone_safe_rewards = np.zeros(self.num_nev)
        self.episode_prog_done_rewards = 0
        self.episode_command_done_rewards = 0
        self.episode_crash_done_rewards = 0
        self.episode_drone_safe_done_rewards = 0
        self.episode_finish_rate = 0
        self.episode_success_rate = 0

    def _on_step(self) -> bool:
        assert (
            "dones" in self.locals
        ), "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.check_finish_rate = "finish_rate" in self.locals["infos"][0]
        self.check_success_rate = "success_rate" in self.locals["infos"][0]
        # add prog rewards
        for i in range(self.num_nev):
            # add reward per step
            self.episode_prog_rewards[i] += self.locals["infos"][i]["prog_reward"]
            self.episode_command_rewards[i] += self.locals["infos"][i]["command_reward"]
            self.episode_crash_rewards[i] += self.locals["infos"][i]["crash_reward"]
            self.episode_drone_safe_rewards[i] += self.locals["infos"][i]["drone_safe_reward"]

        # add reward per step for each env
        self.episode_prog_done_rewards += np.sum(self.episode_prog_rewards[self.locals["dones"]])
        self.episode_command_done_rewards += np.sum(self.episode_command_rewards[self.locals["dones"]])
        self.episode_crash_done_rewards += np.sum(self.episode_crash_rewards[self.locals["dones"]])
        self.episode_drone_safe_done_rewards += np.sum(self.episode_drone_safe_rewards[self.locals["dones"]])
        if self.check_finish_rate:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self.episode_finish_rate += self.locals["infos"][i]["finish_rate"]
        if self.check_success_rate:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self.episode_success_rate += self.locals["infos"][i]["success_rate"]

        # clear episode reward for each done env
        self.episode_prog_rewards[self.locals["dones"]] = 0
        self.episode_command_rewards[self.locals["dones"]] = 0
        self.episode_crash_rewards[self.locals["dones"]] = 0
        self.episode_drone_safe_rewards[self.locals["dones"]] = 0

        # cal total episodes per rollout
        self.n_episodes += np.sum(self.locals["dones"]).item()
        return True

    def _on_rollout_end(self) -> None:
        # get each part of the reward (mean)
        ep_prog_rew_mean = self.episode_prog_done_rewards / self.n_episodes
        ep_command_rew_mean = self.episode_command_done_rewards / self.n_episodes
        ep_crash_rew_mean = self.episode_crash_done_rewards / self.n_episodes
        ep_drone_safe_rew_mean = self.episode_drone_safe_done_rewards / self.n_episodes
        self.logger.record("rew_part/ep_prog_rew_mean", ep_prog_rew_mean)
        self.logger.record("rew_part/ep_command_rew_mean", ep_command_rew_mean)
        self.logger.record("rew_part/ep_crash_rew_mean", ep_crash_rew_mean)
        self.logger.record("rew_part/ep_drone_safe_rew_mean", ep_drone_safe_rew_mean)
        if self.check_finish_rate:
            episode_finish_rate_mean = self.episode_finish_rate / self.n_episodes
            self.logger.record("episode_info/finish_rate", episode_finish_rate_mean)
        if self.check_success_rate:
            episode_success_rate_mean = self.episode_success_rate / self.n_episodes
            self.logger.record("episode_info/success_rate", episode_success_rate_mean)

        # reset
        self.n_episodes = 0
        self.episode_prog_done_rewards = 0
        self.episode_command_done_rewards = 0
        self.episode_crash_done_rewards = 0
        self.episode_drone_safe_done_rewards = 0
        if self.check_finish_rate:
            self.episode_finish_rate = 0
        if self.check_success_rate:
            self.episode_success_rate = 0

    def _on_training_end(self) -> None:
        pass
