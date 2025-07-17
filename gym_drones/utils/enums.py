from enum import Enum


class DroneModel(Enum):
    """Drone models enumeration class."""

    SINGLE_TEST_DRONE = "single_test_drone"  # single params
    MULTI_TEST_DRONE = "multi_test_drone"  # multi params
    MULTI_5_UAV = "multi_test_drone_5uav"  # multi params


################################################################################


class ActionType(Enum):
    """Action type enumeration class."""

    RATE = "rate"  # Rate input (using PID control etc.)


################################################################################


class ObservationType(Enum):
    """Observation type enumeration class."""

    KIN = "kin"  # Kinematic information (pose, linear and angular velocities, acceleration etc .)
    KIN_REL = "kin_rel"  # Kinematic relative information (distance, direction to target, linear and angular velocities, acceleration etc .)
    POS_REL = "pos_rel"  # Kinematic relative information (position to target, linear and angular velocities, acceleration etc .)
    ROT_REL = "rot_rel"  # Kinematic relative information, use rot matrix
    RACE = "race"  # Kinematic relative information, use rot matrix, obs 2 gates
    RACE_MULTI = "race_multi"  # Kinematic relative information, use rot matrix, obs 2 gates, multi drones
    GATE = "gate"  # obs next 2 gates
    GATE_TRACK = "gate_track"  # obs next 2 gates, and the last gate


class SimulationDim(Enum):
    """Dimension of simulation enumeration class."""

    DIM_2 = "2D"  # 2-D space
    DIM_3 = "3D"  # 3-D space


class SimulationType(Enum):
    """Task of simulation enumeration class."""

    RANDOM = "random"  # random episode
    STAY = "stay"  # stay hover
    STEP_NEXT = "step_next"  # step next target
