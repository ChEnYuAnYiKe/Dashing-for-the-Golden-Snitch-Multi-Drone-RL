import pytest
import numpy as np
from gym_drones.utils.motion_library import sin, circle, linear, GateMotionPlanner


@pytest.fixture
def motion_params():
    """Fixture to provide motion parameters for testing."""
    # This fixture provides a common set of parameters for motion tests
    return {
        "t": np.array([0.25]),
        "initial_pos": np.array([[[10, 20, 30]]]),  # 1 drone, 1 gate
        "num_drones": 1,
        "num_gates": 1,
    }


def test_sin_motion(motion_params):
    """Test sine motion."""
    params = {"phase": np.array([0]), "amplitude": np.array([5]), "frequency": np.array([1]), "axis": "x"}
    new_pos = sin(motion_params["t"], motion_params["initial_pos"], params)
    expected_offsets = np.array([5, 0, 0])  # sin(pi/2), 0, 0
    expected_pos_x = motion_params["initial_pos"] + expected_offsets
    assert new_pos.shape == (1, 1, 3)
    np.testing.assert_allclose(new_pos, expected_pos_x, atol=1e-6)
    assert np.all(new_pos[0, 0, 1:] == [20, 30])  # Y, Z axes remain unchanged


def test_circle_motion(motion_params):
    """Test circular motion."""
    params = {
        "phase": np.array([0]),
        "radius": np.array([10]),
        "speed": np.array([np.pi]),  # radians per second
        "axis": "xy",
        "clockwise": False,
    }
    new_pos = circle(motion_params["t"], motion_params["initial_pos"], params)
    # t=0.25 -> angle=pi/4; t=0.5 -> angle=pi/2
    expected_offsets_x = 10 * np.cos(np.array([np.pi / 4, np.pi / 4, np.pi / 2]))
    expected_offsets_y = 10 * np.sin(np.array([np.pi / 4, np.pi / 4, 0]))
    print(f"Expected offsets X: {expected_offsets_x}, Y: {expected_offsets_y}")
    np.testing.assert_allclose(new_pos, motion_params["initial_pos"] + expected_offsets_x, atol=1e-6)
    np.testing.assert_allclose(new_pos, motion_params["initial_pos"] + expected_offsets_y, atol=1e-6)
    assert np.all(new_pos[0, 0, 2] == 30)  # Z axis remains unchanged


def test_linear_motion(motion_params):
    """Test linear motion."""
    params = {
        "start_val": np.array([30]),
        "end_val": np.array([40]),
        "duration": np.array([0.5]),
        "phase": np.array([0]),
        "axis": "z",
    }
    new_pos = linear(motion_params["t"], motion_params["initial_pos"], params)
    # progress: 0, 0.5, 1.0
    expected_pos_z = np.array([0, 0, 5])
    assert np.all(new_pos[0, 0, :2] == [10, 20])  # X, Y axes remain unchanged
    np.testing.assert_allclose(new_pos, motion_params["initial_pos"] + expected_pos_z, atol=1e-6)


def test_gate_motion_planner():
    """Test GateMotionPlanner's compilation and computation capabilities."""
    config = [{"id": 0, "sin": [{"index": 0, "params": {"phase": 0, "amplitude": 5, "frequency": 1, "axis": "x"}}]}]
    planner = GateMotionPlanner(num_waypoints_per_lap=1, num_laps=1, moving_gate_config=config)
    initial_pos = np.array([[[10, 20, 30]]])  # 1 drone, 1 gate

    # t=0.25, sin(2*pi*1*0.25) = sin(pi/2) = 1. Offset should be 5.
    new_pos = planner.compute_positions(0.25, initial_pos)

    assert new_pos.shape == (1, 1, 3)
    np.testing.assert_allclose(new_pos[0, 0, :], [15, 20, 30], atol=1e-6)
