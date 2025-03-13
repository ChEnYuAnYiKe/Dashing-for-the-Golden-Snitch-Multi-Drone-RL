# Dashing for the Golden Snitch: Multi-Drone RL

## Table of Contents

1. [Introduction](#introduction)
   - [News](#news)
   - [Demonstration Video](#demonstration-video)
   - [Related Papers](#related-papers)
2. [Quick Installation](#quick-installation)
3. [Usage](#usage)
4. [Citation](#citation)
5. [License](#license)

## Introduction

- **A multi-agent environment for time-optimal motion planning**. This repository uses multi-agent reinforcement learning to present a decentralized policy network for time-optimal multi-drone flight.
- This project is **a reimplementation of** [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) **optimized for multi-agent scenarios**. We have adjusted the code to make it more suitable for handling many agents simultaneously.
- We **customize PPO** in *a centralized training, decentralized execution* (CTDE) fashion, based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and inspired by the [on-policy(MAPPO)](https://github.com/marlbenchmark/on-policy) repository.

### News

- **March 5, 2025**: The camera-ready version of our paper has been updated on [arXiv](https://arxiv.org/abs/2409.16720).
- **January 27, 2025**: Our paper has been accepted to **ICRA 2025**!
- **September 25, 2024**: Paper preprint available on arXiv.
- **Coming Soon**: The public and release version is coming soon...

### Demonstration Video

[![Demonstration](https://img.youtube.com/vi/KACuFMtGGpo/maxresdefault.jpg)](https://youtu.be/KACuFMtGGpo)

Real-world experiments with two quadrotors using the same network achieve **a maximum speed of 13.65 m/s** and **a maximum body rate of 13.4 rad/s** in a *5.5 m x 5.5 m x 2.0 m* space across various tracks, **relying entirely on onboard computation**.

### Related Papers
- [**Dashing for the Golden Snitch: Multi-Drone Time-Optimal Motion Planning with Multi-Agent Reinforcement Learning**](https://arxiv.org/abs/2409.16720),  Wang, X., Zhou, J., Feng, Y., Mei, J., Chen, J., & Li, S. (2024), arXiv preprint arXiv:2409.16720. **Accepted at ICRA 2025**.

## Quick Installation

It's recommended to use a virtual environment, such as conda:

coming soon...

## Usage

coming soon...

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