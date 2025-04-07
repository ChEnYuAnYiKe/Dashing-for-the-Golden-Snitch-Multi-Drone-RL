from gymnasium.envs.registration import register

register(
    id="hover-env-v0",
    entry_point="gym_drones.envs.single_agent:HoverEnv",
)

register(
    id="race-env-v0",
    entry_point="gym_drones.envs.multi_agent:RaceEnv",
    disable_env_checker=True,
)
register(
    id="track-env-v0",
    entry_point="gym_drones.envs.single_agent:TrackEnv",
    disable_env_checker=True,
)
