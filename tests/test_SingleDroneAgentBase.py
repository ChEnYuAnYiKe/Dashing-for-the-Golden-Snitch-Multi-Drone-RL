import numpy as np
from gym_drones.envs.single_agent import SingleDroneAgentBase
from gym_drones.utils.enums import DroneModel
from gym_drones.utils.enums import ObservationType, SimulationDim


def test_SingleDroneAgentBase_initialization():
    # test default initialization
    env = SingleDroneAgentBase()
    assert env.DRONE_MODEL == DroneModel.SINGLE_TEST_DRONE
    assert env.NUM_DRONES == 1
    assert env.SIM_FREQ == 100
    assert env.CTRL_FREQ == 100

    # test custom initialization
    obs = ObservationType.RACE
    dim = SimulationDim.DIM_3
    initial_xyzs = np.array([[0, 0, 1]])
    initial_rpys = np.array([[0, 0, 0]])
    env = SingleDroneAgentBase(
        obs=obs,
        dim=dim,
        drone_model=DroneModel.SINGLE_TEST_DRONE,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        sim_freq=200,
        ctrl_freq=50,
    )
    assert env.DRONE_MODEL == DroneModel.SINGLE_TEST_DRONE
    assert np.array_equal(env.INIT_XYZS, initial_xyzs)
    assert np.array_equal(env.INIT_RPYS, initial_rpys)
    assert env.SIM_FREQ == 200
    assert env.CTRL_FREQ == 50
