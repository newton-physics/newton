# Newton Physics Tutorials

[![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-35QaaoiXx6VDmVNBOEtmpLs9VBs)

This directory contains a set of tutorials for learning **Newton**, a GPU-accelerated physics simulation engine built upon NVIDIA Warp. These tutorials cover everything from basic concepts to advanced techniques like differentiable simulation and reinforcement learning.

## Tutorial Overview

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 00 | [Introduction](00_introduction.ipynb) | Introduction to Newton's core concepts and architecture | ModelBuilder, solvers, ViewerRerun, GPU acceleration |
| 01 | [Articulations](01_articulations.ipynb) | Creating multi-body systems connected by joints | Joint types (revolute, prismatic, ball, fixed), URDF/MJCF/USD import |
| 02 | [Inverse Kinematics](02_inverse_kinematics.ipynb) | IK solving with the Franka FR3 robot arm | IK objectives, Levenberg-Marquardt optimization, end-effector control |
| 03 | [Joint Control](03_joint_control.ipynb) | Joint target control with the UR10 robot arm | PD controllers, trajectory generation, ArticulationView |
| 04 | [Domain Randomization](04_domain_randomization.ipynb) | Multi-world simulation for reinforcement learning | Parallel simulation, domain randomization, batch operations |
| 05 | [Robot Policy](05_robot_policy.ipynb) | Running trained RL policies on the Unitree Go2 quadruped | Policy loading, observation computation, closed-loop control |
| 06 | [Differentiable Simulation](06_diffsim.ipynb) | Gradient-based optimization through physics | Automatic differentiation, trajectory optimization, gradient descent |

## Getting Started

### Prerequisites

See [Newton documentation](https://newton-physics.github.io/newton/guide/installation.html).

### Running the Tutorials

You can run these tutorials in several ways:

1. **Brev Cloud**: Click the deploy button above to launch a pre-configured environment
2. **Local Jupyter**: Open the notebooks in Jupyter Lab or Jupyter Notebook
3. **VS Code**: Use the Jupyter extension in Visual Studio Code

## Additional Resources

- [Newton GitHub Repository](https://github.com/NVIDIA/newton)
- [Newton Documentation](https://newton-physics.github.io)
- [NVIDIA Warp Documentation](https://github.com/NVIDIA/warp)

## Support

If you encounter any issues or have questions, please open an issue on the [Newton GitHub repository](https://github.com/NVIDIA/newton/issues).

