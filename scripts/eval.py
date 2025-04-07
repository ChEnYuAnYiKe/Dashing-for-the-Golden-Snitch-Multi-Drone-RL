import os, argparse
import torch as th

from gym_drones.utils.enums import ObservationType, SimulationDim, ActionType, DroneModel
from gym_drones.utils.rl_manager.config import process_config
from gym_drones.utils.rl_manager.runner import pre_runner, build_env, load_model
from gym_drones.utils.rl_manager.eval_utils import eval_model

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
    parser.add_argument("-r", "--eval_name", type=str, required=False, help="Specify the evaluation name.")
    parser.add_argument("-m", "--load_model", type=str, required=False, help="Specify the model path.")
    parser.add_argument("-c", "--load_ckpt", type=str, required=False, help="Specify the checkpoint path.")
    parser.add_argument("-s", "--load_step", type=int, required=False, help="Specify the load step.")
    parser.add_argument("-S", "--save_eval", type=str, required=False, help="Specify the results save path.")
    parser.add_argument("-f", "--config", type=str, required=False, help="Specify the config file.")
    parser.add_argument("-v", "--verbose", type=int, required=False, help="Specify the verbosity level.")
    parser.add_argument("-k", "--no_ow", action="store_false", help="Do not overwrite the results.")
    parser.add_argument("--seed", type=int, required=False, help="Specify the random seed.")
    parser.add_argument("--comment", type=str, default="results", help="Specify the comment for the results.")

    # Use the arguments
    args = parser.parse_args()
    if args.load_model is None and args.load_ckpt is None:
        raise ValueError("Please specify the model path using --load_model or --load_ckpt.")
    if args.load_model is not None and args.load_ckpt is not None:
        Warning("Both model and checkpoint paths are specified. The checkpoint path will be used.")
    if args.config is None:
        print(
            "No config file specified. Using default config, which may not be suitable for your environment or model."
        )

    # Read the config file
    config_dict = process_config(
        args=args,
        current_dir=current_dir,
        envkey=envkey,
        algkey=algkey,
        runkey=runkey,
        enum_mapping=enum_mapping,
        eval_mode=True,
    )

    #### Start the Training ###############################
    # Prepare the runner
    pre_runner(config_dict=config_dict, eval_mode=True)

    # Build the env
    env = build_env(config_dict=config_dict, current_dir=current_dir, eval_mode=True)

    # Load the model
    model = load_model(config_dict=config_dict, current_dir=current_dir, eval_mode=True)

    # evaluate the model and get the logger
    logger = eval_model(
        model=model, env=env, config_dict=config_dict, current_dir=current_dir, save_results=True, comment=args.comment
    )

    # visualize the results
    logger.plot()


if __name__ == "__main__":
    run()
