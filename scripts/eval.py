import os, argparse
import torch as th

from gym_drones.utils.enums import ObservationType, SimulationDim, ActionType, DroneModel
from gym_drones.utils.rl_manager.config import process_config
from gym_drones.utils.rl_manager.runner import pre_runner, build_env, load_model
from gym_drones.utils.rl_manager.eval_utils import eval_model
from gym_drones.utils.vis_utils import create_raceplotter

import matplotlib.pyplot as plt

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
    parser.add_argument("-l", "--loop", type=int, required=False, help="Specify the number of evaluation loops.")
    parser.add_argument("--seed", type=int, required=False, help="Specify the random seed.")
    parser.add_argument("--comment", type=str, required=False, help="Specify the comment for the results.")
    parser.add_argument("--track", type=str, required=False, help="Specify the track name.")
    parser.add_argument("--track_sigma", type=float, default=0.1, help="Specify the track sigma.")
    parser.add_argument("--save_timestamps", action="store_true", help="Save timestamps.")
    parser.add_argument("--radius", type=float, default=1.0, help="Specify the radius for the waypoints.")
    parser.add_argument("--margin", type=float, default=0.0, help="Specify the margin for the waypoints.")
    parser.add_argument("--headless", action="store_true", help="Use headless mode for 3D visualization.")

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
    logger, track_raw_data, noise_matrix, save_dir, comment = eval_model(
        model=model,
        env=env,
        config_dict=config_dict,
        current_dir=current_dir,
        save_results=True,
        save_timestamps=args.save_timestamps,
        comment=args.comment,
        track_name=args.track,
        track_sigma=args.track_sigma,
    )

    # create the raceplotter list
    shape_kwargs = {
        "radius": args.radius,
        "margin": args.margin,
    }
    raceplotter_list = create_raceplotter(
        logger=logger, track_data=track_raw_data, shape_kwargs=shape_kwargs, noise_matrix=noise_matrix
    )

    # visualize the results
    for i, raceplotter in enumerate(raceplotter_list):
        raceplotter.plot(
            cmap=plt.cm.cool_r,
            save_fig=True,
            save_path=save_dir,
            fig_name=f"{comment}_drone{i + 1}_2d",
            fig_title=f"{comment} (drone {i + 1})",
            **shape_kwargs,
        )
        raceplotter.plot3d(
            cmap=plt.cm.cool_r,
            save_fig=True,
            save_path=save_dir,
            fig_name=f"{comment}_drone{i + 1}_3d",
            fig_title=f"{comment} (drone {i + 1})",
            gate_color="gray",
            gate_alpha=0.06,
            **shape_kwargs,
        )
        if args.headless:
            raceplotter.save_3d_fig(
                save_path=save_dir, fig_name=f"{comment}_drone{i + 1}_cover", hide_background=True, hide_ground=True
            )
    logger.plot()


if __name__ == "__main__":
    run()
