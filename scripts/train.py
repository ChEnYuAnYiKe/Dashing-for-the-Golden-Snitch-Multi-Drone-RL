import os, argparse
import torch as th

from gym_drones.utils.enums import ObservationType, SimulationDim, ActionType, DroneModel
from gym_drones.utils.rl_manager.config import process_config
from gym_drones.utils.rl_manager.runner import pre_runner, build_env, load_model, train_model

#### Set Constants #######################################
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

envkey = {"hover_race": "hover_race"}
algkey = {"hover_race": "ppo"}
runkey = {"hover_race": "hover_race"}

enum_mapping = {
    "drone_model": DroneModel,
    "obs": ObservationType,
    "act": ActionType,
    "dim": SimulationDim,
    "activation_fn": th.nn,
}


def run():
    #### Get the Training Parameters ########################
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Add arguments
    parser.add_argument("-e", "--env", type=str, required=True, help="Specify the environment.")
    parser.add_argument("-n", "--exp_name", type=str, required=False, help="Specify the experiment name.")
    parser.add_argument("-B", "--num_envs", type=int, required=False, help="Specify the number of environments.")
    parser.add_argument("-M", "--max_steps", type=int, required=False, help="Specify the max steps.")
    parser.add_argument("-S", "--save_num", type=int, required=False, help="Specify the save frequency.")
    parser.add_argument("-m", "--load_model", type=str, required=False, help="Specify the model path.")
    parser.add_argument("-c", "--load_ckpt", type=str, required=False, help="Specify the checkpoint path.")
    parser.add_argument("-s", "--load_step", type=int, required=False, help="Specify the load step.")
    parser.add_argument("-f", "--config", type=str, required=False, help="Specify the config file.")
    parser.add_argument("-v", "--verbose", type=int, required=False, help="Specify the verbosity level.")
    parser.add_argument("--seed", type=int, required=False, help="Specify the random seed.")
    parser.add_argument(
        "--no_reset_t",
        action="store_false",
        help="Do not reset the number of timesteps (default: use the value from config.yaml).",
    )

    # Use the arguments
    args = parser.parse_args()
    if args.load_model is not None and args.load_ckpt is not None:
        Warning("Both model and checkpoint paths are specified. The checkpoint path will be used.")

    # Read the config file
    config_dict = process_config(args, current_dir, envkey, algkey, runkey, enum_mapping)

    #### Start the Training ###############################
    # Prepare the runner
    device = pre_runner(config_dict)

    # Build the env
    env = build_env(config_dict, current_dir)

    # Load the model
    model = load_model(config_dict=config_dict, current_dir=current_dir, env=env, device=device)

    # train the model
    train_model(model=model, config_dict=config_dict, current_dir=current_dir)


if __name__ == "__main__":
    run()
