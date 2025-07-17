import pytest
import numpy as np
from gym_drones.utils.Logger import Logger
from gym_drones.utils.vis_utils import _logger_to_traj_data, _data_to_racetrack


@pytest.fixture
def sample_logger():
    """Create a Logger instance with some data."""
    logger = Logger(logging_freq_hz=10, num_drones=1, duration_sec=1)
    for i in range(5):
        timestamp = i / 10.0
        state = np.arange(20) + i
        logger.log(drone=0, timestamp=timestamp, state=state)
    logger.update_info(finished_step=np.array([5]))
    return logger


def test_logger_log_and_finalize(sample_logger):
    """Test basic logging and final step calculation of Logger."""
    assert sample_logger.counters[0] == 5
    assert sample_logger.end_step[0] == 5
    np.testing.assert_allclose(sample_logger.states[0, 0, 4], 4)  # x pos at last step
    np.testing.assert_allclose(sample_logger.states[0, 19, 4], 23)  # thrust at last step


def test_logger_to_traj_data(sample_logger):
    """Test conversion from Logger to trajectory data."""
    data_list, end_time, crash_effect = _logger_to_traj_data(sample_logger)

    assert len(data_list) == 1
    traj_data = data_list[0]

    assert traj_data.dtype.names == ("t", "p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w", "v_x", "v_y", "v_z")
    assert len(traj_data) == 5  # 因为 end_step 是 5
    assert end_time == 0.5
    assert not crash_effect[0]
    np.testing.assert_allclose(traj_data["t"][-1], 0.4)
    np.testing.assert_allclose(traj_data["p_x"][-1], 4)  # state[0] + i


def test_data_to_racetrack():
    """Test conversion from dictionary data to RaceTrack object."""
    track_data = {
        "comment": "test_track",
        "num_drones": 1,
        "start_points": [[[0, 0, 1]]],
        "end_points": [[[2, 0, 1]]],
        "waypoints": [[[1, 0, 1]]],
    }
    noise_matrix = np.zeros((1, 1, 3))
    racetrack_list = _data_to_racetrack(
        track_data, shape_kwargs={"radius": 1.0, "margin": 0.0}, noise_matrix=noise_matrix
    )

    assert len(racetrack_list) == 1
    track = racetrack_list[0]
    assert track.race_name == "test_track_drone1"
    assert len(track.gate_sequence) == 1
    np.testing.assert_allclose(track.gate_sequence[0][1].position, [1, 0, 1])
