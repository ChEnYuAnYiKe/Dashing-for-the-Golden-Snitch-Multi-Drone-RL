import os
import numpy as np
import torch as th
import gymnasium as gym
from typing import Union, Optional, Callable

from stable_baselines3 import PPO

from gym_drones.utils.Logger import Logger


def get_predict_fn(
    model: Union[PPO, th.nn.Module], config_dict: dict
) -> Callable[[np.ndarray], Union[np.ndarray, tuple]]:
    """Get the prediction function for the model.

    Parameters
    ----------
    model : Union[PPO, th.nn.Module]
        The model to be evaluated.
    config_dict : dict
        The configuration dictionary containing the evaluation parameters.

    Returns
    -------
    Callable[[np.ndarray], Union[np.ndarray, tuple]]
        The prediction function for the model.
        In the case of Stable-Baselines3 PPO, it returns a tuple (action, state).
        In the case of PyTorch model, it returns the output of the model.
    """
    if isinstance(model, PPO):
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("Evaluating a Stable-Baselines3 PPO model. Starting evaluation...")
        predict_fn = lambda obs: model.predict(obs, deterministic=True)
    elif isinstance(model, th.nn.Module):
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("Evaluating a PyTorch model. Starting evaluation...")
        device = th.device("cuda" if config_dict["pyrl"]["use_cuda"] else "cpu")
        predict_fn = lambda obs: model(th.from_numpy(obs).to(device)).detach().cpu().numpy()
    else:
        raise TypeError("Unsupported model type. Expected PPO or th.nn.Module.")
    return predict_fn


def get_save_path(config_dict: dict, current_dir: Union[str, os.PathLike]) -> os.PathLike:
    """Get the save path for the evaluation results.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary containing the evaluation parameters.
    current_dir : str or os.PathLike
        The current directory where the results will be saved.

    Returns
    -------
    os.PathLike
        The path where the evaluation results will be saved.

    """
    # get the save path
    model_name = config_dict["logging"]["save_model_name"]
    model_id = config_dict["logging"]["run_id"]
    eval_save_dirname = config_dict["logging"].get("save_eval_path", config_dict["logging"]["save_config_dirname"])
    eval_save_path = os.path.join(
        current_dir,
        eval_save_dirname,
        config_dict["pyrl"]["exp_name"],
        f"{model_name}_{model_id + 1}",
        "evals",
    )
    eval_name = config_dict["pyrl"].get("eval_name", config_dict["pyrl"]["exp_name"])
    eval_id = config_dict["logging"]["eval_id"]
    output_folder = os.path.join(
        eval_save_path,
        f"{eval_name}_{eval_id + 1}",
    )

    # show the evaluation details
    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("-" * 50)
        print("[EVALUATION DETAILS]")
        print(f"Evaluation name: {eval_name}_{eval_id + 1}")
        print(f"Evaluating {model_name}_{model_id + 1} on {config_dict['env']['env_id']}")
    return output_folder


def eval_model(
    model: Union[PPO, th.nn.Module],
    env: gym.Env,
    config_dict: dict,
    current_dir: Union[str, os.PathLike],
    eval_loop: int = 3,
    save_results: bool = True,
    comment: str = "results",
) -> Logger:
    """Evaluate the model.

    Parameters
    ----------
    model : PPO
        The PPO model to be evaluated.
    env : gym.Env
        The environment to be used for evaluation.
    config_dict : dict
        The configuration dictionary containing the evaluation parameters.
    current_dir : str or os.PathLike
        The current directory where the results will be saved.
    eval_loop : int, optional
        The number of evaluation loops, by default 3
    save_results : bool, optional
        Whether to save the results or not, by default True
    comment : str, optional
        A comment to be added to the saved results, by default "results"

    Returns
    -------
    Logger
        The logger object containing the evaluation results.

    """
    # unwrap the environment
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    # get the save path
    output_folder = get_save_path(config_dict, current_dir)

    # Check the model type
    predict_fn = get_predict_fn(model, config_dict)

    for j in range(eval_loop):
        # create the logger and reset the environment
        logger = Logger(
            logging_freq_hz=int(env.CTRL_FREQ),
            num_drones=env.NUM_DRONES,
            output_folder=output_folder,
        )
        obs, info = env.reset(options={})
        total_reward = 0

        # Sim loop
        for i in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ + 1):
            # predict
            outputs = predict_fn(obs)
            action = outputs[0] if isinstance(outputs, tuple) else outputs

            # step
            obs, reward, terminated, truncated, info = env.step(action)
            target = info["target"]
            total_reward += reward

            # record state
            for nth_drone in range(env.NUM_DRONES):
                log_state = np.hstack(
                    [
                        env.pos[nth_drone],
                        env.vel[nth_drone],
                        env.rpy[nth_drone],
                        env.rate[nth_drone],
                        env.quat[nth_drone],
                        target[nth_drone],
                        env.thrust[nth_drone],
                    ]
                )
                action_to_log = action[nth_drone] if action.ndim > 1 else action
                log_control = np.hstack([action_to_log, np.zeros(8)])

                logger.log(
                    drone=nth_drone,
                    timestamp=i / env.CTRL_FREQ,
                    state=log_state,
                    control=log_control,
                )

            # reset
            if terminated:
                if config_dict["rl_hyperparams"]["verbose"] > 0:
                    print(f"    [Episode {j + 1}] Terminated after {i} timesteps")
                    print(f"        Total reward: {total_reward}")
                break
                # obs, info = env.reset(options={})
            elif truncated:
                if config_dict["rl_hyperparams"]["verbose"] > 0:
                    print(f"    [Episode {j + 1}] Truncated after {i} timesteps")
                    print(f"        Total reward: {total_reward}")
                break
                # obs, info = env.reset(options={})

    # close the environment
    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("Evaluation finished!")
        print("-" * 50)
    env.close()

    if save_results:
        # save the logger
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print(f"Saving results to {output_folder}")
        logger.save_as_csv(comment=comment)
    return logger
