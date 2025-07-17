import os
import numpy as np
from gymnasium import spaces

from gym_drones.envs.DroneBase import DroneBase
from gym_drones.utils.enums import (
    DroneModel,
    ActionType,
    ObservationType,
    SimulationDim,
)
from typing import Optional, Tuple, Union


class SingleDroneAgentBase(DroneBase):
    """Base single drone environment class for reinforcement learning."""

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
        """Initialization of a generic single agent RL environment.

        Attribute `num_drones` is automatically set to 1;

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone param (detailed in an .yaml file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        sim_freq : int, optional
            The frequency at which simulator steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision etc.)
        act : ActionType, optional
            The type of action space (2D or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)
        dim : SimulationDim, optional
            The dimension of the simulation (2D or 3D).
        episode_len_sec : float, optional
            The length of the episode in seconds.
        eval_mode : bool, optional
            Whether the environment is in evaluation mode.
        sensor_sigma : float, optional
            The standard deviation of the sensor noise.
        domain_rand : bool, optional
            Whether to randomize the environment.
        drone_model_path : str | os.PathLike, optional
            The path to the drone model file (in .yaml format) to be used for simulation.
            If None, the default path is used.

        """
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.DIM = dim
        #### Set max sim time (per episode) ########################
        self.EPISODE_LEN_SEC = episode_len_sec
        self.EVAL_MODE = eval_mode
        self.sensor_sigma = sensor_sigma
        #### Initialize DroneBase class ############################
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            domain_rand=domain_rand,
            drone_model_path=drone_model_path,
        )

    ################################################################################

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into real inputs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        # NOTE: This step() function is capable for stable-baselines 3, with no
        # share obs ouputs. For share obs, check the specific implementation
        # in MultiDroneAgentBase.step()

        #### Save, preprocess, and clip the action to the max value #
        self.clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the physics steps #####
        for _ in range(self.SIM_STEPS_PER_CTRL):
            self._dynamics_vectorized(self.clipped_action)
        #### Prepare the return values #############################
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        if truncated:
            terminated = False
        reward = self._computeReward(terminated, truncated)
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + self.SIM_STEPS_PER_CTRL
        #### Change to next waypoint ###############################
        self._stepNextTarget()
        #### compute observation ###################################
        obs = self._computeObs()
        #### Save the last step (e.g. to compute drag) #############
        self._saveLastStates()
        self._saveLastAction(action)
        self.last_clipped_action = self.clipped_action
        return obs, reward, terminated, truncated, info

    ################################################################################

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        # NOTE: This reset() function is capable for stable-baselines 3, with no
        # share obs ouputs. For share obs, check the specific implementation
        # in MultiDroneAgentBase.computeShareObs() and MultiDroneAgentBase.reset()

        #### Set initial states ####################################
        super(DroneBase, self).reset(seed=seed)
        self.step_counter = 0

        # Targets
        self.TARGET = (
            np.vstack(
                [
                    np.array([x * 4 * self.L for x in range(self.NUM_DRONES)]),
                    np.array([y * 4 * self.L for y in range(self.NUM_DRONES)]),
                    np.ones(self.NUM_DRONES),
                ]
            )
            .transpose()
            .reshape(self.NUM_DRONES, 3)
        )  # init z at 1 m
        self.next_TARGET = (
            np.vstack(
                [
                    np.array([x * 4 * self.L for x in range(self.NUM_DRONES)]),
                    np.array([y * 4 * self.L for y in range(self.NUM_DRONES)]),
                    np.ones(self.NUM_DRONES),
                ]
            )
            .transpose()
            .reshape(self.NUM_DRONES, 3)
        )  # init z at 1 m
        # Positions
        self.INIT_XYZS = (
            np.vstack(
                [
                    np.array([x * 4 * self.L for x in range(self.NUM_DRONES)]),
                    np.array([y * 4 * self.L for y in range(self.NUM_DRONES)]),
                    np.ones(self.NUM_DRONES),
                ]
            )
            .transpose()
            .reshape(self.NUM_DRONES, 3)
        )  # init z at 1 m
        # Attitudes
        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        # Velocities
        self.INIT_VELS = np.zeros((self.NUM_DRONES, 3))

        #### Random Init ###########################################
        self._randomInit()
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
        # Crashed
        self.crashed = np.full(self.NUM_DRONES, False)
        # Finished
        self.finished = np.full(self.NUM_DRONES, False)
        # run the track
        self.run_track = False
        return initial_obs, initial_info

    ################################################################################

    def _actionSpace(self) -> spaces.Box:
        """Returns the action space of the environment.

        Returns
        -------
        ndarray
            A Box() of size depending on the action type.

        """
        if self.ACT_TYPE == ActionType.RATE:
            if self.DIM == SimulationDim.DIM_2:
                size = 2
            elif self.DIM == SimulationDim.DIM_3:
                size = 4
                # size = 3
            else:
                print("[ERROR] in SingleDroneAgentBase._actionSpace(): Wrong self.DIM!")
        else:
            print("[ERROR] in SingleDroneAgentBase._actionSpace(): Wrong self.ACT_TYPE!")
            exit()
        return spaces.Box(
            low=np.float32(-1 * np.ones(size)),
            high=np.float32(np.ones(size)),
            dtype=np.float32,
        )

    ################################################################################

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """Pre-processes the action passed to `.step()` into real inputs.

        Parameter `action` is processed differenly for each of the different
        action types.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into real inputs.

        Returns
        -------
        ndarray
            array of inputs (Thrust and Rates etc.)

        """
        if self.ACT_TYPE == ActionType.RATE:
            if self.DIM == SimulationDim.DIM_2:
                return np.hstack(
                    [
                        self.TWR_MAX * self.GRAVITY * (1 + action[0]) / 2,
                        self.MAX_RATE_XY * action[1],
                        0,
                        0,
                    ]
                )
            elif self.DIM == SimulationDim.DIM_3:
                return np.hstack(
                    [
                        self.TWR_MAX * self.GRAVITY * (1 + action[0]) / 2,
                        self.MAX_RATE_XY * action[1:3],
                        self.MAX_RATE_Z * action[3],
                    ]
                )
                # return np.hstack([self.TWR_MAX*self.GRAVITY*(1 + action[0])/2, self.MAX_RATE_XY*action[1:3], 0])
            else:
                print("[ERROR] in SingleDroneAgentBase._preprocessAction(): Wrong self.DIM!")
        else:
            print("[ERROR] in SingleDroneAgentBase._preprocessAction(): Wrong self.ACT_TYPE!")

    ################################################################################

    def _observationSpace(self) -> spaces.Box:
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_2:
                    ############################################################
                    #### OBS SPACE OF SIZE 5
                    #### Observation vector ###  Y   Z   R   VY  VZ
                    obs_lower_bound = np.array([-1, 0, -1, -1, -1])
                    obs_upper_bound = np.array([1, 1, 1, 1, 1])
                    return spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        dtype=np.float32,
                    )
                    ############################################################
                elif self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 9
                    #### Observation vector ###  X   Y   Z   R   P   Y   VX  VY  VZ
                    obs_lower_bound = np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1])
                    obs_upper_bound = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
                    return spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.KIN_REL:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_2:
                    ############################################################
                    #### OBS SPACE OF SIZE 6 (Distance & Direction)
                    #### Observation vector ###  D   DY  DZ  R   VY  VZ
                    obs_lower_bound = np.array([0, -1, -1, -1, -1, -1])
                    obs_upper_bound = np.array([1, 1, 1, 1, 1, 1])
                    return spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        dtype=np.float32,
                    )
                    ############################################################
                elif self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 10 (Distance & Direction)
                    #### Observation vector ###  D   DX  DY  DZ  R   P   Y   VX  VY  VZ
                    obs_lower_bound = np.array([0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
                    obs_upper_bound = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                    return spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.POS_REL:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 13 (Relative pos + absolute vel) ####
                    #### Observation vector ### X Y Z R P Y Vx Vy Vz at ap aq ar
                    obs_lower_bound = -1
                    obs_upper_bound = 1
                    return spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        shape=(13,),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.ROT_REL:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 19 (Relative pos + absolute vel #####
                    ########################## + rot matrix + last action) #####
                    #### Observation vector ### X Y Z Vx Vy Vz
                    #### Observation vector ### rot_x rot_y rot_z at ap aq ar
                    obs_lower_bound = -1
                    obs_upper_bound = 1
                    return spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        shape=(19,),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.RACE:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 22 (Relative pos 1,2 + absolute vel #
                    ########################## + rot matrix + last action) #####
                    #### Observation vector ### X1 Y1 Z1 X2 Y2 Z2 Vx Vy Vz
                    #### Observation vector ### rot_x rot_y rot_z at ap aq ar
                    obs_lower_bound = -1
                    obs_upper_bound = 1
                    return spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        shape=(22,),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        else:
            print("[ERROR] in SingleDroneAgentBase._observationSpace(): Wrong self.OBS_TYPE!")

    ################################################################################

    def _computeObs(self) -> np.ndarray:
        """Returns the current observation of the environment.

        Parameters
        ----------
        nth_drone : int
            The index of the drone for which to compute the observation.

        Returns
        -------
        ndarray
            A Box() of shape depending on the observation type.

        """
        obs_uncliped = self._getDroneStateVector(0)
        if self.EVAL_MODE:
            obs_uncliped = self._add_sensor_noise(obs_uncliped, self.sensor_sigma)
        obs = self._clipAndNormalizeState(obs_uncliped)
        if self.OBS_TYPE == ObservationType.KIN:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_2:
                    ############################################################
                    #### OBS SPACE OF SIZE 5
                    #### vector ###  pos       rpy        vel
                    ret = np.hstack([obs[1:3], obs[7], obs[11:13]]).reshape(
                        5,
                    )
                    return ret.astype("float32")
                    ############################################################
                elif self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 9
                    #### vector ###  pos       rpy        vel
                    ret = np.hstack([obs[0:3], obs[7:10], obs[10:13]]).reshape(
                        9,
                    )
                    return ret.astype("float32")
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.KIN_REL:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_2:
                    ############################################################
                    #### OBS SPACE OF SIZE 6
                    #### vector ###  distance  direction  rpy     vel
                    ret = np.hstack([obs[20], obs[22:24], obs[7], obs[11:13]]).reshape(
                        6,
                    )
                    return ret.astype("float32")
                    ############################################################
                elif self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 10
                    #### vector ###  distance  direction  rpy        vel
                    ret = np.hstack([obs[20], obs[21:24], obs[7:10], obs[10:13]]).reshape(
                        10,
                    )
                    return ret.astype("float32")
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.POS_REL:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 13
                    #### vector ###  rel_pos     rpy        vel         last_action
                    ret = np.hstack([obs[24:27], obs[7:10], obs[10:13], obs_uncliped[25:29]]).reshape(
                        13,
                    )
                    return ret.astype("float32")
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.ROT_REL:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 19
                    #### vector ###  rel_pos     vel
                    ret = np.hstack(
                        [
                            obs[24:27],
                            obs[10:13],
                            #### vector ###  rotation
                            self.rot[0, :, :].flatten(),
                            #### vector ###  last_action
                            obs_uncliped[25:29],
                        ]
                    ).reshape(
                        19,
                    )
                    return ret.astype("float32")
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.RACE:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 22
                    #### vector ###  rel_pos_1   rel_pos_2   vel
                    ret = np.hstack(
                        [
                            obs[24:27],
                            obs[27:30],
                            obs[10:13],
                            #### vector ###  rotation
                            self.rot[0, :, :].flatten(),
                            #### vector ###  last_action
                            obs_uncliped[25:29],
                        ]
                    ).reshape(
                        22,
                    )
                    return ret.astype("float32")
                    ############################################################
                else:
                    print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.DIM!")
            else:
                print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.ACT_TYPE!")
        else:
            print("[ERROR] in SingleDroneAgentBase._computeObs(): Wrong self.OBS_TYPE!")

    ################################################################################

    def _clipAndNormalizeState(self, state: np.ndarray) -> None:
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError

    ################################################################################

    def _stepNextTarget(self) -> None:
        """Change the goal, if env is not terminated

        example: generate random targets or gates

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _randomInit(self) -> None:
        """Change the initial state & target etc.

        example: generate random initial state

        Must be implemented in a subclass.

        """
        raise NotImplementedError
