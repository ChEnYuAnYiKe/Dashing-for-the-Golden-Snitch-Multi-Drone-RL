import pytest
import numpy as np
import yaml
import os
import gymnasium as gym

from gym_drones.envs.single_agent.HoverEnv import HoverEnv
from gym_drones.envs.multi_agent.RaceEnv import RaceEnv
from gym_drones.utils.enums import ObservationType, ActionType

# ===============================================================================
# Fixtures for Environment Instances
# ===============================================================================


@pytest.fixture
def hover_env():
    """create a HoverEnv instance."""
    return HoverEnv(obs=ObservationType.RACE, act=ActionType.RATE)


@pytest.fixture
def race_env():
    """create a RaceEnv instance with default parameters."""
    return RaceEnv(num_drones=2, obs=ObservationType.RACE_MULTI, act=ActionType.RATE)


# ===============================================================================
# Tests for Environment API Compliance
# ===============================================================================


@pytest.mark.parametrize("env_fixture", ["hover_env", "race_env"])
def test_env_api_compliance(env_fixture, request):
    """
    test the environment API compliance for reset and step methods.
    This will ensure that the environments follow the OpenAI Gym API.
    - reset() should return observation and info
    - step(action) should return observation, reward, terminated, truncated, info
    """
    env = request.getfixturevalue(env_fixture)

    # test reset()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    assert (obs[0].shape if obs.ndim > 1 else obs.shape) == env.observation_space.shape, "Observation shape mismatch"
    assert isinstance(info, dict), "Info should be a dictionary"

    # test step(action)
    action = env.action_space.sample()
    action = np.tile(action, (env.NUM_DRONES, 1)) if env.NUM_DRONES > 1 else action
    print(action.shape)  # Debugging line to check action shape
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray), "Step observation should be a numpy array"
    assert (
        obs[0].shape if obs.ndim > 1 else obs.shape
    ) == env.observation_space.shape, "Step observation shape mismatch"

    if isinstance(env, (HoverEnv)):
        assert isinstance(reward, float), "Reward should be a float for single-agent env"
        assert isinstance(terminated, bool), "Terminated flag should be a bool for single-agent env"
        assert truncated in [True, False], "Truncated flag should be a bool for single-agent env"
    else:  # Multi-agent
        assert isinstance(reward, np.ndarray), "Reward should be a numpy array for multi-agent env"
        assert reward.shape == (env.NUM_DRONES,), "Reward shape mismatch for multi-agent env"
        # In RaceEnv, terminated is bool, truncated is array
        assert isinstance(terminated, bool), "Terminated flag should be a bool for multi-agent env"
        assert truncated in [True, False], "Truncated flag should be a numpy array for multi-agent env"

    assert isinstance(info, dict), "Step info should be a dictionary"
    env.close()


# ===============================================================================
# Basic Functionality Tests
# ===============================================================================


@pytest.mark.parametrize("env_fixture", ["hover_env", "race_env"])
def test_env_run(env_fixture, request):
    """
    simple test to run the environment for a few steps.
    """
    env = request.getfixturevalue(env_fixture)
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        action = np.tile(action, (env.NUM_DRONES, 1)) if env.NUM_DRONES > 1 else action
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset()
    env.close()


# ===============================================================================
# RaceEnv Specific Tests
# ===============================================================================


def test_race_env_set_waypoints(race_env):
    """
    test the _setWaypoints_reset method in RaceEnv.
    """
    num_drones = race_env.NUM_DRONES
    num_waypoints = 5
    waypoints = np.random.rand(num_drones, num_waypoints, 3)

    # set waypoints and check the returned observation and info
    obs, info = race_env._setWaypoints_reset(waypoints)

    assert isinstance(obs, np.ndarray)
    assert (obs[0].shape if obs.ndim > 1 else obs.shape) == race_env.observation_space.shape
    assert isinstance(info, dict)
    assert "target" in info

    # check if the target matches the waypoints
    assert np.allclose(info["target"], waypoints[:, 0, :])

    race_env.close()
