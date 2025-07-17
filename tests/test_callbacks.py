import pytest
import numpy as np
from gym_drones.utils.rl_manager.EvalCallback import EvalRewardCallback, EvalTimeCallback


@pytest.fixture
def mock_env():
    """create a mock environment for testing."""

    class MockEnv:
        def __init__(self):
            self.CTRL_FREQ = 100
            self.num_waypoints = np.array([0])
            self.crashed = np.array([False])
            self.finished = np.array([False])

    return MockEnv()


def test_eval_reward_callback(mock_env):
    """test reward callback functionality."""
    callback = EvalRewardCallback(verbose=0)
    callback.init_callback(mock_env)

    # simulate an episode
    callback.on_episode_start()
    for i in range(10):
        # simulate drone passing through waypoints
        callback.locals = {
            "infos": {"prog_reward": 1.0, "command_reward": -0.1, "crash_reward": 0.0, "drone_safe_reward": 0.0}
        }
        callback.on_step()
    callback.on_episode_end()

    # simulate evaluation end
    callback.on_eval_end()

    assert len(callback.episode_prog_done_rewards) == 1
    np.testing.assert_allclose(callback.ep_prog_rew_mean, 10.0)
    np.testing.assert_allclose(callback.ep_command_rew_mean, -1.0)
    np.testing.assert_allclose(callback.total_reward_mean, 9.0)


def test_eval_time_callback(mock_env):
    """test time callback functionality."""
    track_data = {
        "num_drones": 1,
        "start_points": [[[0, 0, 1]]],
        "waypoints": [[i, 0, 1] for i in range(1, 10)],
        "end_points": [[[10, 0, 1]]],
        "repeat_lap": 1,
    }
    callback = EvalTimeCallback(data=track_data, verbose=0)
    callback.init_callback(mock_env)

    # simulate an episode
    callback.on_episode_start()
    # simulate drone starting at first waypoint
    mock_env.num_waypoints = np.array([callback.start_index])
    callback.locals = {"step_counter": 100}
    callback.on_step()
    assert callback.start_flag[0]
    np.testing.assert_allclose(callback.start_time[0], 1.0)  # 100 / 100Hz

    # simulate drone reaching the end
    mock_env.num_waypoints = np.array([callback.end_index])
    mock_env.finished = np.array([True])
    callback.locals = {"step_counter": 350}
    callback.on_step()
    assert callback.end_flag[0]
    np.testing.assert_allclose(callback.end_time[0], 3.5)  # 350 / 100Hz

    callback.on_episode_end()
    callback.on_eval_end()

    np.testing.assert_allclose(callback.lap_time_mean, 2.5)  # 3.5 - 1.0
    np.testing.assert_allclose(callback.finished_mean, 1.0)
