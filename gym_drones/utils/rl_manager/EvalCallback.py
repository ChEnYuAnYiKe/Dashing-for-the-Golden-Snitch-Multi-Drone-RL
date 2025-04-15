import json, os
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING, Optional, Callable, Union

if TYPE_CHECKING:
    from gym_drones import envs


class EvalBaseCallback(ABC):
    """Base class for evaluation callbacks.

    This class is intended to be inherited by other callback classes
    that implement specific evaluation logic.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (default is 0).
    num_drones : int, optional
        Number of drones (default is 1).

    """

    def __init__(self, verbose: int = 0, num_drones: int = 1):
        super().__init__()
        self.verbose = verbose
        self.num_drones = num_drones
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}

    def init_callback(self, env: "envs.DroneBase") -> None:
        """
        Initialize the callback by saving references to the environment for convenience.
        """
        self.env = env
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_eval_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        self._on_eval_start()

    def _on_eval_start(self) -> None:
        pass

    def on_episode_start(self) -> None:
        self._on_episode_start()

    def _on_episode_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> None:
        pass

    def on_step(self) -> None:
        return self._on_step()

    def on_eval_end(self) -> None:
        self._on_eval_end()

    def _on_eval_end(self) -> None:
        pass

    def on_episode_end(self) -> None:
        self._on_episode_end()

    def _on_episode_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        Parameters
        ----------
        locals_ : dict
            The local variables during rollout collection.
            This is a dictionary containing the local variables
            from the current scope.

        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        Parameters
        ----------
        locals_ : dict
            The local variables during rollout collection.
            This is a dictionary containing the local variables
            from the current scope.

        """
        pass

    def save_results(self, save_dir: Union[str, os.PathLike], comment: str = "") -> None:
        """
        Save the results to a file.

        Parameters
        ----------
        save_dir : str
            The folder where the results will be saved.
        comment : str, optional
            A comment to be added to the filename (default is empty).
        save_timestamps : bool, optional
            Whether to save the results with timestamps (default is False).

        """
        pass


class ConvertCallback(EvalBaseCallback):
    """
    Convert functional callback (old-style) to object.

    Parameters
    ----------
    callback : Callable[[Dict[str, Any], Dict[str, Any]], bool]
        The callback function to be converted.
    verbose : int, optional
        Verbosity level (default is 0).

    """

    def __init__(self, callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]], verbose: int = 0):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)


class EvalCallbackList(EvalBaseCallback):
    """
    Class for chaining callbacks.

    Parameters
    ----------
    callbacks : list
        List of callbacks to be chained together.

    """

    def __init__(self, callbacks: List[EvalBaseCallback]):
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.env)

    def _on_eval_start(self) -> None:
        for callback in self.callbacks:
            callback.on_eval_start(self.locals, self.globals)

    def _on_episode_start(self) -> None:
        for callback in self.callbacks:
            callback.on_episode_start()

    def _on_step(self) -> None:
        for callback in self.callbacks:
            callback.on_step()

    def _on_episode_end(self) -> None:
        for callback in self.callbacks:
            callback.on_episode_end()

    def _on_eval_end(self) -> None:
        for callback in self.callbacks:
            callback.on_eval_end()

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)

    def save_results(self, save_dir: Union[str, os.PathLike], comment: str = "") -> None:
        """
        Save the results to a file.

        Parameters
        ----------
        save_dir : str
            The folder where the results will be saved.
        comment : str, optional
            A comment to be added to the filename (default is empty).
        save_timestamps : bool, optional
            Whether to save the results with timestamps (default is False).

        """
        for callback in self.callbacks:
            callback.save_results(save_dir=save_dir, comment=comment)


class EvalRewardCallback(EvalBaseCallback):
    """Class for collecting rewards during evaluation.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (default is 0).

    """

    def __init__(
        self,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        # Initialize episode rewards
        self.episode_prog_rewards = 0
        self.episode_command_rewards = 0
        self.episode_crash_rewards = 0
        self.episode_drone_safe_rewards = 0
        # Initialize episode done rewards
        self.episode_prog_done_rewards = []
        self.episode_command_done_rewards = []
        self.episode_crash_done_rewards = []
        self.episode_drone_safe_done_rewards = []

    def _on_episode_start(self):
        """Called at the start of each episode."""
        # clear episode reward for each done env
        self.episode_prog_rewards = 0
        self.episode_command_rewards = 0
        self.episode_crash_rewards = 0
        self.episode_drone_safe_rewards = 0

    def _on_step(self):
        # add reward per step
        self.episode_prog_rewards += self.locals["infos"]["prog_reward"]
        self.episode_command_rewards += self.locals["infos"]["command_reward"]
        self.episode_crash_rewards += self.locals["infos"]["crash_reward"]
        self.episode_drone_safe_rewards += self.locals["infos"]["drone_safe_reward"]

    def _on_episode_end(self):
        """Called at the end of each episode."""
        # add reward per step for each env
        self.episode_prog_done_rewards.append(self.episode_prog_rewards)
        self.episode_command_done_rewards.append(self.episode_command_rewards)
        self.episode_crash_done_rewards.append(self.episode_crash_rewards)
        self.episode_drone_safe_done_rewards.append(self.episode_drone_safe_rewards)
        total_reward = (
            self.episode_prog_rewards
            + self.episode_command_rewards
            + self.episode_crash_rewards
            + self.episode_drone_safe_rewards
        )
        if self.verbose > 0:
            print(f"    Total reward:       {total_reward:8.4f}")
        if self.verbose > 1:
            print(f"    Reward components:")
            print(f"        - progress:     {self.episode_prog_rewards:8.4f}")
            print(f"        - command:      {self.episode_command_rewards:8.4f}")
            print(f"        - crash:        {self.episode_crash_rewards:8.4f}")
            print(f"        - drone safety: {self.episode_drone_safe_rewards:8.4f}")

    def _on_eval_end(self) -> None:
        # get each part of the reward (mean)
        self.ep_prog_rew_mean = np.array(self.episode_prog_done_rewards).mean()
        self.ep_command_rew_mean = np.array(self.episode_command_done_rewards).mean()
        self.ep_crash_rew_mean = np.array(self.episode_crash_done_rewards).mean()
        self.ep_drone_safe_rew_mean = np.array(self.episode_drone_safe_done_rewards).mean()

        # Calculate standard deviations
        self.ep_prog_rew_std = np.array(self.episode_prog_done_rewards).std()
        self.ep_command_rew_std = np.array(self.episode_command_done_rewards).std()
        self.ep_crash_rew_std = np.array(self.episode_crash_done_rewards).std()
        self.ep_drone_safe_rew_std = np.array(self.episode_drone_safe_done_rewards).std()

        # Calculate total reward mean and std
        self.total_reward_mean = (
            self.ep_prog_rew_mean + self.ep_command_rew_mean + self.ep_crash_rew_mean + self.ep_drone_safe_rew_mean
        )
        self.total_reward_std = np.sqrt(
            self.ep_prog_rew_std**2
            + self.ep_command_rew_std**2
            + self.ep_crash_rew_std**2
            + self.ep_drone_safe_rew_std**2
        )

        if self.verbose > 0:
            print(f"[Reward Info]")
            print(f"Total reward:       {self.total_reward_mean:8.4f} ± {self.total_reward_std:.4f}")
            print("Reward components:")
            print(f"    - progress:     {self.ep_prog_rew_mean:8.4f} ± {self.ep_prog_rew_std:.4f}")
            print(f"    - command:      {self.ep_command_rew_mean:8.4f} ± {self.ep_command_rew_std:.4f}")
            print(f"    - crash:        {self.ep_crash_rew_mean:8.4f} ± {self.ep_crash_rew_std:.4f}")
            print(f"    - drone safety: {self.ep_drone_safe_rew_mean:8.4f} ± {self.ep_drone_safe_rew_std:.4f}")

    def save_results(self, save_dir: Union[str, os.PathLike], comment: str = "") -> None:
        """Save the results to a file.

        Parameters
        ----------
        save_dir : str
            The folder where the results will be saved.
        comment : str, optional
            A comment to be added to the filename (default is empty).

        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir + "/")
        # Save the results to a file
        with open(f"{save_dir}/{comment}_eval_rewards.json", "w") as f:
            json.dump(
                {
                    "total_reward_mean": float(self.total_reward_mean),
                    "total_reward_std": float(self.total_reward_std),
                    "reward_components": {
                        "prog_reward_mean": float(self.ep_prog_rew_mean),
                        "prog_reward_std": float(self.ep_prog_rew_std),
                        "command_reward_mean": float(self.ep_command_rew_mean),
                        "command_reward_std": float(self.ep_command_rew_std),
                        "crash_reward_mean": float(self.ep_crash_rew_mean),
                        "crash_reward_std": float(self.ep_command_rew_std),
                        "drone_safe_reward_mean": float(self.ep_drone_safe_rew_mean),
                        "drone_safe_reward_std": float(self.ep_drone_safe_rew_std),
                    },
                },
                f,
                indent=4,
            )


class EvalTimeCallback(EvalBaseCallback):
    """Class for collecting time statistics during race track evaluation.

    Parameters
    ----------
    data : np.ndarray
        Track data to be used for evaluation.
    verbose : int, optional
        Verbosity level (default is 0).

    """

    def __init__(
        self,
        data: np.ndarray,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.data = data
        self.num_drones = self.data.get("num_drones", 1)
        self.start_index = 0
        self.end_index = 0
        self.start_time = np.zeros((self.num_drones,))
        self.end_time = np.zeros((self.num_drones,))
        self.num_waypoints = np.zeros((self.num_drones,))
        self.finished = np.full((self.num_drones,), False)
        self.start_flag = np.full((self.num_drones,), False)
        self.end_flag = np.full((self.num_drones,), False)
        self.lap_time_list = []
        self.finished_list = []
        self._get_lap_points()

    def _get_lap_points(self) -> None:
        """Get the lap points from the data."""
        # read the dict data (skip the valid check)
        same_track = self.data.get("same_track", True)
        repeat_lap = self.data.get("repeat_lap", 1)
        # track waypoints
        start_points = np.array(self.data["start_points"]).reshape((self.num_drones, -1, 3))
        if same_track:
            waypoints = np.repeat(np.array(self.data["waypoints"]), self.num_drones, axis=0).reshape(
                (self.num_drones, -1, 3)
            )
        else:
            waypoints = np.array(self.data["waypoints"]).reshape((self.num_drones, -1, 3))

        start_len = start_points.shape[1]
        main_seg_len = waypoints.shape[1]

        nth_seg = round((repeat_lap + 1) / 2)
        self.start_index = start_len + (nth_seg - 1) * main_seg_len
        self.end_index = self.start_index + main_seg_len

    def _print_drone_lap_info(self) -> None:
        print("    Lap infos:")
        print(f"        - Lap points: Pt.{self.start_index + 1} - Pt.{self.end_index + 1}")
        print("        " + "-" * 41)
        print("        | Drone | Start(s) |  End(s)  | Time(s) | Status")
        print("        " + "-" * 41)
        for i in range(self.num_drones):
            start_time = self.start_time[i]
            end_time = self.end_time[i]

            # if the drone never passed the start point
            if start_time == 0:
                status = "NEVER_STARTED"
                time_str = "---"
                start_str = "---"
                end_str = "---"
            # if the drone never passed the end point
            elif end_time == 0 or not self.finished[i]:
                status = "DNF (CRASH)"
                time_str = "---"
                start_str = f"{start_time:.2f}"
                end_str = "---"
            # if the drone passed the end point
            else:
                status = "FINISHED" if self.finished[i] else "UNFINISHED"
                time_str = f"{(end_time - start_time):.2f}"
                start_str = f"{start_time:.2f}"
                end_str = f"{end_time:.2f}"

            print(f"        |  {(i + 1):2d}   |   {start_str:6s} |   {end_str:6s} |   {time_str:5s} | {status}")
        print("        " + "-" * 41)

    def _on_episode_start(self):
        """Called at the start of each episode."""
        self.start_time = np.zeros((self.num_drones,))
        self.end_time = np.zeros((self.num_drones,))
        self.num_waypoints = np.zeros((self.num_drones,))
        self.finished = np.full((self.num_drones,), False)
        self.start_flag = np.full((self.num_drones,), False)
        self.end_flag = np.full((self.num_drones,), False)

    def _on_step(self):
        self.num_waypoints = self.env.num_waypoints.copy().reshape((self.num_drones,))
        self.finished = np.array(self.env.finished).copy().reshape((self.num_drones,))
        # set the start time for each drone
        start_indices = np.where((~self.start_flag) & (self.num_waypoints >= self.start_index))[0]
        if len(start_indices) > 0:
            self.start_time[start_indices] = self.locals["step_counter"] / self.env.CTRL_FREQ
            self.start_flag[start_indices] = True
        # set the end time for each drone
        end_indices = np.where((~self.end_flag) & (self.num_waypoints >= self.end_index))[0]
        if len(end_indices) > 0:
            self.end_time[end_indices] = self.locals["step_counter"] / self.env.CTRL_FREQ
            self.end_flag[end_indices] = True

    def _on_episode_end(self):
        """Called at the end of each episode."""
        lap_time = self.end_time - self.start_time
        self.finished_list.append(self.finished.copy())
        self.lap_time_list.append(lap_time.copy())
        if self.verbose > 0:
            self._print_drone_lap_info()

    def _on_eval_end(self) -> None:
        self.lap_time_mean = np.array(self.lap_time_list).mean()
        self.lap_time_std = np.array(self.lap_time_list).std()
        self.finished_mean = np.array(self.finished_list).mean()
        if self.verbose > 0:
            print(f"[Time Info]")
            print(f"Lap time: {self.lap_time_mean} ± {self.lap_time_std} s")
            print(f"Finished Rate: {(self.finished_mean * 100):.2f}%")

    def save_results(self, save_dir: Union[str, os.PathLike], comment: str = "") -> None:
        """Save the results to a file.

        Parameters
        ----------
        save_dir : str
            The folder where the results will be saved.
        comment : str, optional
            A comment to be added to the filename (default is empty).

        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir + "/")
        # Save the results to a file
        with open(f"{save_dir}/{comment}_eval_lap_times.json", "w") as f:
            json.dump(
                {
                    "Lap time": float(self.lap_time_mean),
                    "Lap time std": float(self.lap_time_std),
                    "Finished Rate": float(self.finished_mean),
                },
                f,
                indent=4,
            )
