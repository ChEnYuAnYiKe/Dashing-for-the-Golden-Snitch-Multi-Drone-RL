import os
import numpy as np

from gym_drones.utils.enums import DroneModel
from gym_drones.envs.single_agent.SingleDroneAgentBase import (
    ActionType,
    ObservationType,
    SimulationDim,
    SingleDroneAgentBase,
)

from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, Union


class HoverEnv(SingleDroneAgentBase):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.SINGLE_TEST_DRONE,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        sim_freq: int = 100,
        ctrl_freq: int = 100,
        obs: ObservationType = ObservationType.KIN_REL,
        act: ActionType = ActionType.RATE,
        dim: SimulationDim = SimulationDim.DIM_3,
        episode_len_sec: int = 15,
        eval_mode: bool = False,
        sensor_sigma: float = 0.01,
        domain_rand: bool = False,
        drone_model_path: Optional[Union[str, os.PathLike]] = None,
    ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        sim_freq : int, optional
            The frequency at which simulator steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        dim : SimulationDim, optional
            The dimension of the simulation (2D or 3D).
        episode_len_sec : float, optional
            The length of the episode in seconds.
        eval_mode : bool, optional
            Whether the environment is in evaluation mode.
        sensor_sigma : float, optional
            The standard deviation of the sensor noise.
        domain_rand : bool, optional
            Whether to use domain randomization.
        drone_model_path : str | os.PathLike, optional
            The path to the drone model file.

        """
        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            obs=obs,
            act=act,
            dim=dim,
            episode_len_sec=episode_len_sec,
            eval_mode=eval_mode,
            sensor_sigma=sensor_sigma,
            domain_rand=domain_rand,
            drone_model_path=drone_model_path,
        )

    ################################################################################

    def _computeReward(self, terminated: bool, truncated: bool) -> float:
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        last_state = self.last_states[0, :]
        target = self._getDroneTarget(0)
        action = state[17:21]
        rate = action[1:4]

        if truncated:
            self.crash_reward = -5
            return -5  # crash
        else:
            weights = np.array([1, 1, 3])  # ensure z is more important
            last_prog_rew = np.sqrt(np.sum(np.square((target - last_state[0:3]) * weights)))
            current_prog_rew = np.sqrt(np.sum(np.square((target - state[0:3]) * weights)))

            prog_reward = last_prog_rew - current_prog_rew

            command_reward = (
                -2e-4 * np.linalg.norm(rate)
                - 1e-4 * np.linalg.norm(self.clipped_action[0, :] - self.last_clipped_action[0, :]) ** 2
            )
            total_reward = prog_reward + command_reward
            self.prog_reward = prog_reward
            self.command_reward = command_reward
            return total_reward

    ################################################################################

    def _computeTerminated(self) -> bool:
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    ################################################################################

    def _computeTruncated(self) -> bool:
        """Computes the current truncated value(s).

        Unused in this implementation.

        Returns
        -------
        bool
            Always false.

        """
        MAX_Z = self.MAX_POS_Z * 1.5

        state = self._getDroneStateVector(0)
        target = self._getDroneTarget(0)
        truncated = abs(target[2] - state[2]) >= MAX_Z
        self.crashed = truncated
        return truncated

    ################################################################################

    def _computeInfo(self) -> dict:
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "target": self.TARGET,
            "prog_reward": self.prog_reward,
            "command_reward": self.command_reward,
            "crash_reward": self.crash_reward,
            "drone_safe_reward": 0,
        }

    ################################################################################

    def _clipAndNormalizeState(self, state: np.ndarray) -> np.ndarray:
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (29,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (30,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_XY = self.MAX_POS_XY
        MAX_Z = self.MAX_POS_Z
        MAX_LIN_VEL_XY = self.MAX_LIN_VEL_XY
        MAX_LIN_VEL_Z = self.MAX_LIN_VEL_Z
        MAX_RPY = np.pi  # Full range

        target = self._getDroneTarget(0)
        next_target = self._getDroneNextTarget(0)
        rel_pos = target - state[0:3]
        distance = np.linalg.norm(rel_pos)
        rel_gate = next_target - target

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_RPY, MAX_RPY)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_distance = np.clip(distance, 0, 3 * MAX_XY)
        clipped_rel_xy = np.clip(rel_pos[0:2], -MAX_XY, MAX_XY)
        clipped_rel_z = np.clip(rel_pos[2], -MAX_Z, MAX_Z)
        clipped_rel_xy_next = np.clip(rel_gate[0:2], -MAX_XY, MAX_XY)
        clipped_rel_z_next = np.clip(rel_gate[2], -MAX_Z, MAX_Z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_RPY
        normalized_y = state[9] / MAX_RPY  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        )
        normalized_distance = clipped_distance / (MAX_XY * 3)
        normalized_direction = rel_pos / distance if distance != 0 else rel_pos
        normalized_rel_xy = clipped_rel_xy / MAX_XY
        normalized_rel_z = clipped_rel_z / MAX_Z
        normalized_rel_xy_next = clipped_rel_xy_next / (MAX_XY)
        normalized_rel_z_next = clipped_rel_z_next / (MAX_Z)

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,  # 0, 1
                normalized_pos_z,  # 2
                state[3:7],  # 3, 4, 5, 6
                normalized_rp,  # 7, 8
                normalized_y,  # 9
                normalized_vel_xy,  # 10, 11
                normalized_vel_z,  # 12
                normalized_ang_vel,  # 13, 14, 15
                state[16:20],  # 16, 17, 18, 19
                normalized_distance,  # 20
                normalized_direction,  # 21, 22, 23
                normalized_rel_xy,  # 24, 25
                normalized_rel_z,  # 26
                normalized_rel_xy_next,  # 27, 28
                normalized_rel_z_next,  # 29
            ]
        ).reshape(
            30,
        )

        return norm_and_clipped

    ################################################################################

    def _stepNextTarget(self) -> None:
        """Change the random targets"""
        state = self._getDroneStateVector(0)
        target = self._getDroneTarget(0)
        if np.linalg.norm(target - state[0:3]) <= self.WAYPOINT_R:
            self.num_waypoints[0] = self.num_waypoints[0] + 1
            self.TARGET[0] = self.next_TARGET[0]
            if not self.run_track:
                self.next_TARGET[0] = self.waypoints[(self.num_waypoints[0] + 1) % len(self.waypoints)]
            else:
                if self.num_waypoints[0] + 1 >= len(self.waypoints):
                    self.next_TARGET[0] = self.waypoints[-1]
                    if self.num_waypoints[0] >= len(self.waypoints):
                        self.finished = True
                else:
                    self.next_TARGET[0] = self.waypoints[self.num_waypoints[0] + 1]

    ################################################################################

    def _randomInit(self) -> None:
        """Random initialization of the environment."""
        self.num_waypoints = np.zeros(self.NUM_DRONES, dtype=int)
        # set initial position
        self.INIT_XYZS[:, :] = np.array([(np.hstack([0, 0, self.INIT_HEIGHT])) for i in range(self.NUM_DRONES)])
        # generate random waypoints
        waypoints = []
        if self.DIM == SimulationDim.DIM_2 or self.DIM == SimulationDim.DIM_3:
            for _ in range(30):
                if self.DIM == SimulationDim.DIM_2:
                    temp_waypoint = np.hstack(
                        [
                            0.0,
                            self.np_random.uniform(low=-self.MAX_POS_XY / 2, high=self.MAX_POS_XY / 2, size=(1,)),
                            self.np_random.uniform(
                                low=self.INIT_HEIGHT - self.MAX_POS_Z / 2,
                                high=self.INIT_HEIGHT + self.MAX_POS_Z / 2,
                                size=(1,),
                            ),
                        ]
                    )
                elif self.DIM == SimulationDim.DIM_3:
                    temp_waypoint = np.hstack(
                        [
                            self.np_random.uniform(low=-self.MAX_POS_XY / 2, high=self.MAX_POS_XY / 2, size=(2,)),
                            self.np_random.uniform(
                                low=self.INIT_HEIGHT - self.MAX_POS_Z / 2,
                                high=self.INIT_HEIGHT + self.MAX_POS_Z / 2,
                                size=(1,),
                            ),
                        ]
                    )
                waypoints.append(temp_waypoint)
            self.waypoints = np.array(waypoints)
            # set initial target
            self.TARGET[0] = self.waypoints[0]
            self.next_TARGET[0] = self.waypoints[1]
        else:
            print("[ERROR] in HoverEnv._randomInit(): Wrong self.DIM!")

    ################################################################################

    def _setWaypoints_reset(
        self,
        waypoints: np.ndarray,
        waypoints_radius: Optional[float] = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """reset method for setting specific waypoints

        parameters
        ----------
        waypoints : ndarray
            (num_waypoints, 3)-shaped array containing the desired XYZ position of the drone.
        waypoints_radius : float, optional
            The radius of the waypoints.
        seed : int, optional
            The seed for the random number generator.
        options : dict, optional
            Dictionary containing the options for the environment.

        returns
        -------
        initial_obs : ndarray
            (NUM_DRONES, OBS_DIM)-shaped array containing the initial observation of the drone.
        initial_info : dict
            Dictionary containing the initial info of the drone.

        """
        super().reset(seed=seed, options=options)

        self.run_track = True
        self.waypoints = waypoints
        if waypoints_radius is not None:
            self.WAYPOINT_R = waypoints_radius
        self.num_waypoints = np.zeros(self.NUM_DRONES, dtype=int)
        # set initial position
        self.INIT_XYZS[0] = self.waypoints[0]
        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        self.INIT_VELS = np.zeros((self.NUM_DRONES, 3))
        # set initial target
        self.TARGET[0] = self.waypoints[0]
        self.next_TARGET[0] = self.waypoints[1]
        #### Housekeeping ##########################################
        self._housekeeping()
        #### initial reward check callback #########################
        self.prog_reward = 0
        self.command_reward = 0
        self.crash_reward = 0
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        #### Return the initial info ########################
        initial_info = self._computeInfo()
        return initial_obs, initial_info
