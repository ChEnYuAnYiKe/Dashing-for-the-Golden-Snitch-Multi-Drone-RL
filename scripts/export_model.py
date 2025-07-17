import os
import argparse
import torch as th
from stable_baselines3 import PPO


class TorchScriptPolicy(th.nn.Module):
    """
    A wrapper for the policy network to make it compatible with TorchScript.
    """

    def __init__(self, extractor, action_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this.
        action_hidden = self.extractor(observation)
        return self.action_net(action_hidden)


def export_model(model_path: str, save_path: str, device: th.device):
    """
    Loads a PPO model, extracts its policy network, and saves it as a
    TorchScript module.

    :param model_path: Path to the trained PPO model (.zip).
    :param save_path: Path to save the exported TorchScript model (.pt).
    :param device: The device to use for loading the model.
    """
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device=device)
    print("Model policy architecture:")
    print(model.policy)

    # Create the TorchScript policy
    onnxable_model = TorchScriptPolicy(model.policy.mlp_extractor.policy_net, model.policy.action_net)

    observation_size = model.observation_space.shape
    dummy_input = th.randn(1, *observation_size).to(device)

    print(f"Using device '{device}' for tracing.")

    # Trace the model
    traced_module = th.jit.trace(onnxable_model.eval(), dummy_input)

    # Freeze and optimize for inference
    frozen_module = th.jit.freeze(traced_module)
    frozen_module = th.jit.optimize_for_inference(frozen_module)

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the TorchScript model
    th.jit.save(frozen_module, save_path)
    print(f"Model successfully exported to: {save_path}")


def main():
    """
    Main function to parse arguments and run the model export process.
    """
    envkey = {
        "hover_race": "hover_race",
        "race_multi_2": "race_multi_2",
        "race_multi_3": "race_multi_3",
        "race_multi_5": "race_multi_5",
        "kin_2d": "kin_2d",
        "kin_3d": "kin_3d",
        "kin_rel_2d": "kin_rel_2d",
        "kin_rel_3d": "kin_rel_3d",
        "pos_rel": "pos_rel",
        "rot_rel": "rot_rel",
    }

    parser = argparse.ArgumentParser(description="Export a Stable-Baselines3 PPO model to a TorchScript file.")
    parser.add_argument(
        "--model_dir", "-m", type=str, default="Learning_log", help="Directory containing the trained PPO model."
    )
    parser.add_argument(
        "--model_path", "-p", type=str, required=True, help="Path to the trained PPO model (.zip file)."
    )
    parser.add_argument(
        "--model_name",
        "-n",
        type=str,
        default=None,
        help="Name of the model file to export. If not provided, " "it will be derived from the model_path.",
    )
    parser.add_argument(
        "--env", "-e", type=str, required=True, help="Environment name for which the model was trained."
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        default="Model",
        help="Path to save the exported TorchScript model (.pt file). "
        "If not provided, it will be saved next to the input model with a '_exported.pt' suffix.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for loading and exporting the model ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    # Validate environment
    if args.env not in envkey:
        raise ValueError(f"Invalid environment '{args.env}'. " f"Valid options are: {list(envkey.keys())}")

    # If save_path is not provided, create a default one
    if args.model_name is None:
        args.model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    model_path = os.path.join(args.model_dir, args.env, args.model_path, args.model_name)
    save_path = os.path.join(args.save_path, args.env, f"{args.model_name}.pt")

    device = th.device(args.device)
    export_model(model_path, save_path, device)


if __name__ == "__main__":
    main()
