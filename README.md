# Dashing for the Golden Snitch: Multi-Drone RL
## 0. Introduction

- **A multi-agent environment for time-optimal motion planning**. This repository presents a **decentralized policy network** for time-optimal multi-drone flight using multi-agent reinforcement learning.
- This project is **a reimplementation of** [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones), **optimized for multi-agent scenarios**. We have adjusted the code to make it more suitable for handling a large number of agents simultaneously.
- We **customize PPO** in *a centralized training, decentralized execution* (CTDE) fashion, based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and inspired by the [on-policy(MAPPO)](https://github.com/marlbenchmark/on-policy) repository.

### Demonstration Video

[![Demonstration](https://img.youtube.com/vi/KACuFMtGGpo/maxresdefault.jpg)](https://youtu.be/KACuFMtGGpo)

Real-world experiments with two quadrotors using the same network achieve **a maximum speed of 13.65 m/s** and **a maximum body rate of 13.4 rad/s** in a *5.5 m x 5.5 m x 2.0 m* space across various tracks, **relying entirely on onboard computation**.

### Related Papers
- [**Dashing for the Golden Snitch: Multi-Drone Time-Optimal Motion Planning with Multi-Agent Reinforcement Learning**](https://arxiv.org/abs/2409.16720),  Wang, X., Zhou, J., Feng, Y., Mei, J., Chen, J., & Li, S. (2024), arXiv preprint arXiv:2409.16720. **Accepted at ICRA 2025**.

# The public and release version is coming soon...