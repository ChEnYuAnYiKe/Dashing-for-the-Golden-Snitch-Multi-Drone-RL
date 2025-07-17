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
        self.crashed_step = np.zeros(num_drones, dtype=int)  # crashed step for each drone
        self.finished_step = np.zeros(num_drones, dtype=int)  # finished step for each drone
        self.end_step = np.zeros(num_drones, dtype=int)  # end step for each drone
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

    def update_info(self, **kwargs):
        """Updates the logger's internal attributes.

        This method provides a flexible way to update any attribute of the Logger instance
        at runtime. It is typically used at the end of a simulation or evaluation to record
        final state information, such as the number of steps each drone completed.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments where the key is the attribute name (e.g., "finished_step")
            and the value is the new value for that attribute.

        """
        # This method can be extended to log additional information if needed.
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[WARNING] in Logger.update_info(), attribute '{key}' not found.")

        # Post-process to determine the definitive end step for plotting
        self._finalize_steps()

    ################################################################################

    def _finalize_steps(self):
        """Computes the definitive end step for each drone based on its status.

        This internal method processes the `crashed_step` and `finished_step`
        attributes. For drones that did not crash or finish (where the step is 0),
        their step count is treated as the full simulation duration for comparison.
        The final `end_step` is then calculated as the earliest of these events.
        """
        max_steps = self.timestamps.shape[1]

        # A step of 0 means the event didn't happen; use max_steps as a placeholder for comparison.
        finished = np.where(self.finished_step == 0, max_steps, self.finished_step)
        crashed = np.where(self.crashed_step == 0, max_steps, self.crashed_step)

        # The actual end step is the minimum of the two events (finish or crash).
        self.end_step = np.minimum(finished, crashed)

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
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        style_cycle = ["-", "--", ":", "-."]

        fig, axs = plt.subplots(10, 2, figsize=(30, 27))
        fig.suptitle("Drone Simulation Logs", fontsize=16, fontweight="bold")
        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)
        if len(t) > self.states.shape[2]:
            t = t[: self.states.shape[2]]

        # Define plot configurations: (y_label, data_source, data_indices, target_indices)
        # data_source: 0 for self.states, 1 for self.controls
        plot_configs = [
            # Column 0: States
            [
                ("x (m)", 0, [0], [16]),
                ("y (m)", 0, [1], [17]),
                ("z (m)", 0, [2], [18]),
                ("Roll (rad)", 0, [6], None),
                ("Pitch (rad)", 0, [7], None),
                ("Yaw (rad)", 0, [8], None),
                ("q w", 0, [15], None),
                ("q x", 0, [12], None),
                ("q y", 0, [13], None),
                ("q z", 0, [14], None),
            ],
            # Column 1: Velocities, Rates, and Inputs
            [
                ("vx (m/s)", 0, [3], None),
                ("vy (m/s)", 0, [4], None),
                ("vz (m/s)", 0, [5], None),
                ("wx (rad/s)", 0, [9], None),
                ("wy (rad/s)", 0, [10], None),
                ("wz (rad/s)", 0, [11], None),
                ("Input Thrust", 1, [0], None),
                ("Input p", 1, [1], None),
                ("Input q", 1, [2], None),
                ("Input r", 1, [3], None),
            ],
        ]

        for col_idx, col_configs in enumerate(plot_configs):
            for row_idx, config in enumerate(col_configs):
                ax = axs[row_idx, col_idx]
                y_label, data_source_idx, data_indices, target_indices = config

                data_source = self.states if data_source_idx == 0 else self.controls

                for drone_idx in range(self.NUM_DRONES):
                    color = color_cycle[drone_idx % len(color_cycle)]
                    style = style_cycle[drone_idx % len(style_cycle)]

                    end_step = self.end_step[drone_idx]
                    if end_step > 0:
                        t_plot = np.arange(0, end_step / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)
                        if len(t_plot) > end_step:
                            t_plot = t_plot[:end_step]
                    else:
                        t_plot = t
                        end_step = self.timestamps.shape[1]

                    # Plot target data if specified
                    if target_indices:
                        for target_idx in target_indices:
                            # Use a distinct style for targets
                            ax.plot(
                                t_plot,
                                self.states[drone_idx, target_idx, :end_step],
                                color="k",
                                linestyle=":",
                                label=f"Target",
                            )

                    # Plot main data
                    for data_idx in data_indices:
                        ax.plot(
                            t_plot,
                            data_source[drone_idx, data_idx, :end_step],
                            color=color,
                            linestyle=style,
                            label=f"Drone {drone_idx}",
                        )

                    # Add markers for finished or crashed events
                    if row_idx < 3 and col_idx == 0:
                        # Marker for finishing
                        finish_step = self.finished_step[drone_idx]
                        if finish_step > 0 and finish_step <= len(t_plot):
                            finish_time = finish_step / self.LOGGING_FREQ_HZ
                            finish_val = self.states[drone_idx, row_idx, finish_step - 1]
                            ax.plot(finish_time, finish_val, "g*", markersize=12, label=f"Drone {drone_idx} Finished")

                        # Marker for crashing
                        crash_step = self.crashed_step[drone_idx]
                        if crash_step > 0 and crash_step <= len(t_plot):
                            crash_time = crash_step / self.LOGGING_FREQ_HZ
                            crash_val = self.states[drone_idx, row_idx, crash_step - 1]
                            ax.plot(crash_time, crash_val, "rX", markersize=10, label=f"Drone {drone_idx} Crashed")

                # Only show x-axis label on the bottom-most plot in each column
                if row_idx == len(col_configs) - 1:
                    ax.set_xlabel("Time (s)", fontsize=10)
                    ax.tick_params(axis="x", labelsize=8)
                else:
                    ax.tick_params(axis="x", labelbottom=False)  # Hide x-tick labels for non-bottom plots

                ax.set_ylabel(y_label, fontsize=10)
                ax.tick_params(axis="y", labelsize=8)
                ax.grid(True)

        #### Drawing options #######################################
        # Create a single, shared legend for the entire figure
        handles, labels = [], []
        for ax in axs.flat:
            h, l = ax.get_legend_handles_labels()
            for i, label in enumerate(l):
                if label not in labels:
                    labels.append(label)
                    handles.append(h[i])

        # Place legend at the bottom of the figure
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(self.NUM_DRONES + 1, 6),
                bbox_to_anchor=(0.5, 0.01),
                fontsize=16,
            )

        # Adjust layout to prevent overlap and make space for legend and title
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])  # rect=[left, bottom, right, top]

        plt.show()
