import numpy as np
from typing import Dict, Union, List, Tuple, Any, Callable


def _prepare_inputs(t: Union[float, np.ndarray], initial_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to ensure t and initial_pos are correctly shaped for broadcasting."""
    t = np.asarray(t)
    initial_pos = np.asarray(initial_pos)

    # Reshape t to (num_frames, 1, 1) for broadcasting against (drones, gates, xyz)
    if t.ndim == 0:  # if t is a scalar
        t = t.reshape(1, 1, 1)
    elif t.ndim == 1:  # if t is a 1D array of time points
        t = t.reshape(-1, 1, 1)

    # Ensure initial_pos is at least 3D
    if initial_pos.ndim == 2:
        initial_pos = initial_pos[np.newaxis, :, :]  # Add a drone dimension

    return t, initial_pos


def sin(t: Union[float, np.ndarray], initial_pos: np.ndarray, params: Dict[str, Union[np.ndarray, str]]) -> np.ndarray:
    """
    Fully vectorized function for sinusoidal motion on multiple drones and gates.

    Parameters:
    - t: Scalar or 1D array of time points, shape (num_frames,).
    - initial_pos: 3D array of initial positions, shape (num_drones, num_gates, 3).
    - params: Dictionary of parameters.
        - 'phase', 'amplitude', 'frequency': 1D arrays, shape (num_gates,).
        - 'axis': A single character 'x', 'y', or 'z'.

    Returns:
    - 3D array of new positions, shape (num_frames, num_drones, num_gates, 3).
    """
    t, initial_pos = _prepare_inputs(t, initial_pos)

    # Reshape per-gate params to (1, num_gates, 1) for broadcasting
    phase = np.asarray(params["phase"])[np.newaxis, :, np.newaxis]
    amplitude = np.asarray(params["amplitude"])[np.newaxis, :, np.newaxis]
    frequency = np.asarray(params["frequency"])[np.newaxis, :, np.newaxis]

    offset_values = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    offsets = np.zeros_like(initial_pos, dtype=float)
    axis_map = {"x": 0, "y": 1, "z": 2}
    offsets[..., axis_map[params["axis"]]] = offset_values.squeeze(-1)

    return initial_pos + offsets


def circle(
    t: Union[float, np.ndarray], initial_pos: np.ndarray, params: Dict[str, Union[np.ndarray, str, bool]]
) -> np.ndarray:
    """
    Fully vectorized function for circular motion on multiple drones and gates.

    Parameters:
    - t: Scalar or 1D array of time points, shape (num_frames,).
    - initial_pos: 3D array of initial positions, shape (num_drones, num_gates, 3).
    - params: Dictionary of parameters.
        - 'phase', 'radius', 'speed': 1D arrays, shape (num_gates,).
        - 'axis': A single string 'xy', 'xz', or 'yz'.
        - 'clockwise': A single boolean.

    Returns:
    - 3D array of new positions, shape (num_frames, num_drones, num_gates, 3).
    """
    t, initial_pos = _prepare_inputs(t, initial_pos)

    # Reshape per-gate params for broadcasting
    phase = np.asarray(params["phase"])[np.newaxis, :, np.newaxis]
    radius = np.asarray(params["radius"])[np.newaxis, :, np.newaxis]
    speed = np.asarray(params["speed"])[np.newaxis, :, np.newaxis]

    angle = speed * t + phase
    direction = -1.0 if params.get("clockwise", True) else 1.0

    cos_vals = radius * np.cos(angle)
    sin_vals = radius * np.sin(angle) * direction

    offsets = np.zeros_like(initial_pos, dtype=float)
    plane_map = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
    axis_indices = plane_map[params["axis"]]

    offsets[..., axis_indices[0]] = cos_vals.squeeze(-1)
    offsets[..., axis_indices[1]] = sin_vals.squeeze(-1)

    return initial_pos + offsets


def linear(
    t: Union[float, np.ndarray], initial_pos: np.ndarray, params: Dict[str, Union[np.ndarray, str]]
) -> np.ndarray:
    """
    Fully vectorized function for linear motion on multiple drones and gates.

    Parameters:
    - t: Scalar or 1D array of time points, shape (num_frames,).
    - initial_pos: 3D array of initial positions, shape (num_drones, num_gates, 3).
    - params: Dictionary of parameters.
        - 'start_val', 'end_val', 'duration': 1D arrays, shape (num_gates,).
        - 'phase': 1D array of time delays before motion starts, shape (num_gates,).
        - 'axis': A single character 'x', 'y', or 'z'.

    Returns:
    - 3D array of new positions, shape (num_frames, num_drones, num_gates, 3).
    """
    t, initial_pos = _prepare_inputs(t, initial_pos)

    # Reshape per-gate params for broadcasting
    start_val = np.asarray(params["start_val"])[np.newaxis, :, np.newaxis]
    end_val = np.asarray(params["end_val"])[np.newaxis, :, np.newaxis]
    duration = np.asarray(params["duration"])[np.newaxis, :, np.newaxis]
    # Add phase parameter, defaulting to 0 if not provided
    phase = np.asarray(params.get("phase", 0.0))[np.newaxis, :, np.newaxis]

    # Calculate the effective time for motion, starting only after the phase delay
    effective_t = np.maximum(0, t - phase)

    # Calculate progress based on the effective time
    progress = np.clip(effective_t / duration, 0, 1)
    new_coords = start_val + (end_val - start_val) * progress

    new_pos = initial_pos.copy()
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[params["axis"]]

    # Broadcast assignment
    new_pos[..., axis_idx] = new_coords.squeeze(-1)

    return new_pos


# --- The Motion Function Registry ---
MOTION_FUNCTIONS = {
    "sin": sin,
    "circle": circle,
    "linear": linear,
}


class GateMotionPlanner:
    """
    Pre-compiles a moving gate configuration into an efficient motion plan
    and computes gate positions at any given time.
    """

    def __init__(
        self,
        num_waypoints_per_lap: int,
        num_laps: int,
        moving_gate_config: List[Dict[str, Any]],
        motion_library: Dict[str, Callable] = MOTION_FUNCTIONS,
    ):
        """
        Initializes the planner by parsing the configuration.

        Parameters:
        - num_waypoints_per_lap: The number of waypoints (gates) in a single lap.
        - num_laps: The total number of laps the track is repeated for.
        - moving_gate_config: The 'moving_gate' list from the YAML file.
        - motion_library: The MOTION_FUNCTIONS dictionary mapping names to functions.
        """
        self.num_waypoints_per_lap = num_waypoints_per_lap
        self.num_laps = num_laps
        self.motion_plan = self._compile_plan(moving_gate_config, motion_library)

    def _compile_plan(
        self, config: List[Dict[str, Any]], library: Dict[str, Callable] = MOTION_FUNCTIONS
    ) -> List[Dict[str, Any]]:
        """Parses the raw config into a structured, ready-to-use motion plan."""
        plan = []
        for drone_config in config:
            drone_id = drone_config["id"]

            # Iterate over motion functions specified for the drone (e.g., 'sin', 'circle')
            for func_name, gate_specs in drone_config.items():
                if func_name == "id":
                    continue

                if func_name not in library:
                    print(f"Warning: Motion function '{func_name}' not found in library. Skipping.")
                    continue

                # Expand lap-relative indices to global absolute indices
                all_gate_indices = []
                numeric_params = {
                    key: []
                    for key in gate_specs[0]["params"].keys()
                    if isinstance(gate_specs[0]["params"][key], (int, float))
                }
                config_params = {
                    key: gate_specs[0]["params"][key]
                    for key in gate_specs[0]["params"].keys()
                    if not isinstance(gate_specs[0]["params"][key], (int, float))
                }
                all_params_lists = {key: [] for key in gate_specs[0]["params"].keys()}

                for spec in gate_specs:
                    relative_index = spec["index"]
                    # Generate absolute indices for all laps
                    absolute_indices_for_this_spec = [
                        (lap * self.num_waypoints_per_lap) + relative_index for lap in range(self.num_laps)
                    ]
                    all_gate_indices.extend(absolute_indices_for_this_spec)

                    # Duplicate the numeric parameters for each lap
                    for key in numeric_params:
                        numeric_params[key].extend([spec["params"][key]] * self.num_laps)

                # Convert only numeric parameter lists to numpy arrays
                final_params = {key: np.asarray(val) for key, val in numeric_params.items()}
                # Add the non-numeric config parameters back in
                final_params.update(config_params)

                # Add the compiled task to the plan
                plan.append(
                    {
                        "drone_id": drone_id,
                        "gate_indices": np.array(all_gate_indices),
                        "function": library[func_name],
                        "params": final_params,
                    }
                )
        return plan

    def compute_positions(self, t: float, initial_positions: np.ndarray) -> np.ndarray:
        """
        Computes the new positions of all gates for all drones at a given time.

        Parameters:
        - t: The current time (scalar).
        - initial_positions: A 3D array of initial gate positions,
                             shape (num_drones, num_gates, 3).

        Returns:
        - A 3D array of the new gate positions, same shape as input.
        """
        # Start with the static positions; we will modify them based on the plan.
        new_positions = initial_positions.copy()

        for task in self.motion_plan:
            drone_id = task["drone_id"]
            gate_indices = task["gate_indices"]
            motion_func = task["function"]
            params = task["params"]

            # Select the specific slice of initial positions for this task
            # Shape will be (num_affected_gates, 3)
            initial_slice = initial_positions[drone_id, gate_indices]

            # Call the vectorized motion function from the library
            # It will return an array of shape (num_affected_gates, 3)
            updated_slice = motion_func(t, initial_slice, params)

            # Use advanced indexing to place the results back into the main array
            new_positions[drone_id, gate_indices] = updated_slice

        return new_positions
