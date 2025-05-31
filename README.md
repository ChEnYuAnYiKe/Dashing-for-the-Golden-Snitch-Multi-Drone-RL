# Dashing for the Golden Snitch: Multi-Drone RL

## Table of Contents

1. [Introduction](#introduction)
   - [News](#news)
   - [Demonstration Video](#demonstration-video)
   - [Related Papers](#related-papers)
2. [Quick Installation](#quick-installation)
3. [Usage](#usage)
   - [Start training (Beta)](#start-training-beta)
   - [Evaluation (Beta)](#evaluation-beta)
4. [Citation](#citation)
5. [License](#license)

## Introduction

- **A multi-agent environment for time-optimal motion planning**. This repository uses multi-agent reinforcement learning to present a decentralized policy network for time-optimal multi-drone flight.
- This project is **a reimplementation of** [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) **optimized for multi-agent scenarios**. We have adjusted the code to make it more suitable for handling many agents simultaneously.
- We **customize PPO** in *a centralized training, decentralized execution* (CTDE) fashion, based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and inspired by the [on-policy(MAPPO)](https://github.com/marlbenchmark/on-policy) repository.

### News

- **Coming Soon**: 📢 Full multi-agent version will be released shortly. Stay tuned!
- **May 31, 2025**: 🚀 Full Single-agent version released!
- **April 7, 2025**: 🚀 Single-agent version released! Now available for training and testing (beta) in single-drone scenarios.
- **March 5, 2025**: 📄 The camera-ready version of our paper has been updated on [arXiv](https://arxiv.org/abs/2409.16720).
- **January 27, 2025**: 🎉 Our paper has been accepted to **ICRA 2025**!
- **September 25, 2024**: 📝 Paper preprint available on arXiv.

### Demonstration Video

[![Demonstration](https://img.youtube.com/vi/KACuFMtGGpo/maxresdefault.jpg)](https://youtu.be/KACuFMtGGpo)

Real-world experiments with two quadrotors using the same network achieve **a maximum speed of 13.65 m/s** and **a maximum body rate of 13.4 rad/s** in a *5.5 m x 5.5 m x 2.0 m* space across various tracks, **relying entirely on onboard computation**.

### Related Papers
- [**Dashing for the Golden Snitch: Multi-Drone Time-Optimal Motion Planning with Multi-Agent Reinforcement Learning**](https://arxiv.org/abs/2409.16720),  Wang, X., Zhou, J., Feng, Y., Mei, J., Chen, J., & Li, S. (2024), arXiv preprint arXiv:2409.16720. **Accepted at ICRA 2025**.

## Quick Installation

It's recommended to use a virtual environment, such as conda:

```bash
conda create -n marl_drones python=3.11 # Requires Python >= 3.10
conda activate marl_drones
```

1. For the latest version, clone the repository and install it locally:

   ```bash
   # clone the repository
   git clone https://github.com/KafuuChikai/Dashing-for-the-Golden-Snitch-Multi-Drone-RL.git
   cd Dashing-for-the-Golden-Snitch-Multi-Drone-RL
   
   # update the submodule
   git submodule update --init --recursive
   
   # install the package and dependencies
   pip install -e .
   ```

> [!NOTE]
> **PyTorch** is excluded from the dependencies because its version depends on your system setup. Install the correct version manually based on your hardware and CUDA configuration.

2. Install the correct version of **PyTorch** following the [official instructions](https://pytorch.org/get-started/locally/).

## Usage

### Start Training (Beta)

To start training, run the following command:

```bash
python scripts/train.py -e hover_race -B 16
```

Below are the available drone reinforcement learning environments, including practical scenarios and several examples:

| Env Code     | Name                | Description                                                  |
| ------------ | ------------------- | ------------------------------------------------------------ |
| `hover_race` | Race (single drone) | Real drone control scenario, training drones to navigate through tracks |

| Tutorial Code | Name                                    | Description                                                  |
| ------------- | --------------------------------------- | ------------------------------------------------------------ |
| `kin_2d`      | 2D Kinematics ⚠️ **Non-working example** | Simplified 2D movement control, focusing on Y and Z axes only |
| `kin_3d`      | 3D Kinematics ⚠️ **Non-working example** | Complete 3D space movement control                           |
| `kin_rel_2d`  | 2D Relative Kinematics                  | 2D motion control based on relative positions                |
| `kin_rel_3d`  | 3D Relative Kinematics                  | 3D motion control based on relative positions                |
| `pos_rel`     | Position Relative Control               | observe previous step actions                                |
| `rot_rel`     | Rotation Relative Control               | Using rotation matrices for continue orientation observation |

The script supports the following command-line arguments:

| Arguments             | Short | Type   | Required | Description                                                |
|-----------------------|-------|--------|----------|------------------------------------------------------------|
| `--env`               | `-e`  | `str`  | Yes      | Specify the environment (e.g., `hover_race`).              |
| `--exp_name`          | `-n`  | `str`  | No       | Specify the experiment name.                               |
| `--num_envs`          | `-B`  | `int`  | No       | Specify the number of environments.                        |
| `--max_steps`         | `-M`  | `int`  | No       | Specify the maximum number of steps.                       |
| `--save_num`          | `-S`  | `int`  | No       | Specify the number of checkpoints to save.                 |
| `--load_model`        | `-m`  | `str`  | No       | Specify the path to load a pre-trained model.              |
| `--load_ckpt`         | `-c`  | `str`  | No       | Specify the path to load a checkpoint.                     |
| `--load_step`         | `-s`  | `int`  | No       | Specify the step to load from a checkpoint.                |
| `--config`            | `-f`  | `str`  | No       | Specify the path to the configuration file.                |
| `--verbose`           | `-v`  | `int`  | No       | Specify the verbosity level (e.g., `0`, `1` or `2`).       |
| `--seed`              | None  | `int`  | No	    | Specify the random seed.                                   |
| `--no_reset_t`        | None  | `bool` | No	    |	Do not reset the timestep. default: use `config.yaml`.     |

**Examples**

- Run the following command to train in the `hover_race` environment with 16 parallel environments:
```bash
python scripts/train.py -e hover_race -B 16
```

- To load a pre-trained model and continue training:
```bash
python scripts/train.py -e hover_race -m path/to/model
```

- To use a configuration file for training parameters:
```bash
python scripts/train.py -e hover_race -f path/to/config
```

- To load a checkpoint from step `500,000`:
```bash
python scripts/train.py -e hover_race -c path/to/checkpoints -s 500000
```

### Evaluation (Beta)

|                                                              |                                                              |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/cc6415c5-8db9-4b05-b87e-c753c4c8fb65" /> | <video src="https://github.com/user-attachments/assets/328413a2-fbcf-425c-a349-8c2891fac08b" /> |

To evaluate the demo, run the following:

```bash
python scripts/eval.py -e hover_race -m demo_model/Race_single.pt -v 2 --track single_drone/UZH_single.yaml --track_sigma 0 --vis_config UZH_gate
```

```bash
python scripts/eval.py -e hover_race -m demo_model/Race_single.pt -v 2 --track single_drone/Star_5_single.yaml --track_sigma 0 --vis_config star_tracks
```

The script supports the following command-line arguments:

| Arguments           | Short | Type    | Required | Description                                             |
| ------------------- | ----- | ------- | -------- | ------------------------------------------------------- |
| `--env`             | `-e`  | `str`   | Yes      | Specify the environment (e.g., `hover_race`).           |
| `--exp_name`        | `-n`  | `str`   | No       | Specify the experiment name.                            |
| `--eval_name`       | `-r`  | `str`   | No       | Specify the evaluation name.                            |
| `--load_model`      | `-m`  | `str`   | No       | Specify the path to load a pre-trained model.           |
| `--load_ckpt`       | `-c`  | `str`   | No       | Specify the path to load a checkpoint.                  |
| `--load_step`       | `-s`  | `int`   | No       | Specify the step to load from a checkpoint.             |
| `--save_eval`       | `-S`  | `str`   | No       | Specify the path to save evaluation results.            |
| `--config`          | `-f`  | `str`   | No       | Specify the path to the configuration file.             |
| `--verbose`         | `-v`  | `int`   | No       | Specify the verbosity level (e.g., `0`, `1` or `2`).    |
| `--no_ow`           | `-k`  | `bool`  | No       | Do not overwrite the results.                           |
| `--seed`            | None  | `int`   | No       | Specify the random seed.                                |
| `--comment`         | None  | `str`   | No       | Specify a comment for the results (default: `results`). |
| `--track`           | None  | `str`   | No       | Specify the track name.                                 |
| `--track_sigma`     | None  | `float` | No       | Specify the track noise.                                |
| `--save_timestamps` | None  | `bool`  | No       | Save files with timestamps (flag argument).             |
| `--radius`          | None  | `float` | No       | Specify the radius for waypoints (default: `1.0`).      |
| `--margin`          | None  | `float` | No       | Specify the margin for waypoints (default: `0.0`).      |
| `--vis_config`      | None  | `str`   | No       | Specify the visualization config file.                  |

**Examples**

- Run the following command to evaluate in the hover_race environment using a pre-trained model:
```bash
python scripts/eval.py -e hover_race -m path/to/model
```

- To specify a configuration file for evaluation:
```bash
python scripts/eval.py -e hover_race -m path/to/model -f path/to/config
```

- To save evaluation results to a specific directory:
```bash
python scripts/eval.py -e hover_race -m path/to/model -S path/to/save_results
```

- To add a comment to the evaluation results:
```bash
python scripts/eval.py -e hover_race -m path/to/model --comment your_comment
```

## Citation

If you use this repository in your research, please consider citing:

```bibtex
@article{Wang2024Dashing,
  author = {Wang, X. and Zhou, J. and Feng, Y. and Mei, J. and Chen, J. and Li, S.},
  title = {Dashing for the Golden Snitch: Multi-Drone Time-Optimal Motion Planning with Multi-Agent Reinforcement Learning},
  journal = {arXiv preprint arXiv:2409.16720},
  year = {2024},
  url = {https://arxiv.org/abs/2409.16720}
}
```

## License
This project is released under the MIT License. Please review the [License file](LICENSE) for more details.