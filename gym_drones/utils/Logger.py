import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and inputs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(
        self,
        logging_freq_hz: int,
        output_folder: str = "results",
        num_drones: int = 1,
        duration_sec: int = 0,
    ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """

        self.OUTPUT_FOLDER = output_folder
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec * self.LOGGING_FREQ_HZ))
        #### Note: this is the suggest information to log ##############################
        self.states = np.zeros((num_drones, 20, duration_sec * self.LOGGING_FREQ_HZ))  #### 20 states: pos (3,), 0-2
        # vel (3,), 3-5
        # euler (3,), 6-8
        # rate (3,), 9-11
        # quat (4,), 12-15
        # target (3,), 16-18
        # thrust (1,), 19
        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros(
            (num_drones, 12, duration_sec * self.LOGGING_FREQ_HZ)
        )  #### 12 control targets: thrust (1,), 0
        # rate (3,), 1-3
        # zeros(8),

    ################################################################################

    def log(self, drone: int, timestamp, state, control=np.zeros(12)):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 20, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1] - 1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = state
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1

    ################################################################################

    def save(self):
        """Save the logs to file."""
        with open(
            os.path.join(
                self.OUTPUT_FOLDER,
                "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + ".npy",
            ),
            "wb",
        ) as out_file:
            np.savez(
                out_file,
                timestamps=self.timestamps,
                states=self.states,
                controls=self.controls,
            )

    ################################################################################

    def save_as_csv(self, comment: str = "", save_timestamps: bool = False) -> str:
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        current_time = ("-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")) if save_timestamps else ""
        csv_dir = os.path.join(
            self.OUTPUT_FOLDER,
            "save-flight-" + comment + current_time,
        )
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir + "/")
        t = np.arange(0, self.timestamps.shape[1]) / self.LOGGING_FREQ_HZ
        data_dict = {"time": t}
        #### 20 states: pos (3,), 0-2, vel (3,), 3-5, euler (3,), 6-8, rate (3,), 9-11,
        #### quat (4,), 12-15, target (3,), 16-18, thrust (1,), 19
        for i in range(self.NUM_DRONES):
            data_dict[f"x_{i}"] = self.states[i, 0, :]
            data_dict[f"y_{i}"] = self.states[i, 1, :]
            data_dict[f"z_{i}"] = self.states[i, 2, :]
            data_dict[f"vx_{i}"] = self.states[i, 3, :]
            data_dict[f"vy_{i}"] = self.states[i, 4, :]
            data_dict[f"vz_{i}"] = self.states[i, 5, :]
            data_dict[f"roll_{i}"] = self.states[i, 6, :]
            data_dict[f"pitch_{i}"] = self.states[i, 7, :]
            data_dict[f"yaw_{i}"] = self.states[i, 8, :]
            data_dict[f"wx_{i}"] = self.states[i, 9, :]
            data_dict[f"wy_{i}"] = self.states[i, 10, :]
            data_dict[f"wz_{i}"] = self.states[i, 11, :]
            data_dict[f"qw_{i}"] = self.states[i, 15, :]
            data_dict[f"qx_{i}"] = self.states[i, 12, :]
            data_dict[f"qy_{i}"] = self.states[i, 13, :]
            data_dict[f"qz_{i}"] = self.states[i, 14, :]
            data_dict[f"target_x_{i}"] = self.states[i, 16, :]
            data_dict[f"target_y_{i}"] = self.states[i, 17, :]
            data_dict[f"target_z_{i}"] = self.states[i, 18, :]
            data_dict[f"thrust_{i}"] = self.states[i, 19, :]
            data_dict[f"input_T_{i}"] = self.controls[i, 0, :]
            data_dict[f"input_p_{i}"] = self.controls[i, 1, :]
            data_dict[f"input_q_{i}"] = self.controls[i, 2, :]
            data_dict[f"input_r_{i}"] = self.controls[i, 3, :]

        # create a DataFrame
        df = pd.DataFrame(data_dict)

        # save the dataframe to csv
        df.to_csv(f"{csv_dir}/{comment}_flight_all_data.csv", index=False)
        return csv_dir

    ################################################################################

    def plot(self):
        """Logs entries for a single simulation step, of a single drone."""
        #### Loop over colors and line styles ######################
        plt.rc(
            "axes",
            prop_cycle=(cycler("color", ["r", "g", "b", "y"]) + cycler("linestyle", ["-", "--", ":", "-."])),
        )
        fig, axs = plt.subplots(10, 2, figsize=(12, 8))
        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)
        if len(t) > self.states.shape[2]:
            t = t[: self.states.shape[2]]

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_" + str(j))
            axs[row, col].plot(t, self.states[j, 16, :], label="target_x")
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("x (m)")

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_" + str(j))
            axs[row, col].plot(t, self.states[j, 17, :], label="target_y")
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("y (m)")

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_" + str(j))
            axs[row, col].plot(t, self.states[j, 18, :], label="target_z")
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("z (m)")

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("r (rad)")
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("p (rad)")
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("y (rad)")

        #### Quat ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("q w")
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("q x")
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("q y")
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("q z")

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("vx (m/s)")
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("vy (m/s)")
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("vz (m/s)")

        #### RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("wx")
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("wy")
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("wz")

        #### Inputs ##################################################
        # thrust
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.controls[j, 0, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("input T")
        # rate p
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.controls[j, 1, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("input p")
        # rate q
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.controls[j, 2, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("input q")
        # rate r
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.controls[j, 3, :], label="drone_" + str(j))
        axs[row, col].set_xlabel("time")
        axs[row, col].set_ylabel("input r")

        #### Drawing options #######################################
        for i in range(10):
            for j in range(2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc="upper right", frameon=True)
        fig.subplots_adjust(left=0.06, bottom=0.05, right=0.99, top=0.98, wspace=0.15, hspace=0.0)

        plt.show()
