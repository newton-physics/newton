
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push Events - AWS GPU Tests](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu_tests.yml)

# Newton

**⚠️ Prerelease Software ⚠️**

**This project is in active alpha development.** This means the API is unstable, features may be added or removed, and
breaking changes are likely to occur frequently and without notice as the design is refined.

Newton is a GPU-accelerated physics simulation engine built upon [NVIDIA Warp](https://github.com/NVIDIA/warp),
specifically targeting roboticists and simulation researchers.
It extends and generalizes Warp's existing `warp.sim` module, integrating
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as a primary backend.
Newton emphasizes GPU-based computation, differentiability, and user-defined extensibility, facilitating rapid iteration
and scalable robotics simulation.

Newton is maintained by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/),
and [NVIDIA](https://www.nvidia.com/).

## Development

See the [development guide](https://newton-physics.github.io/newton/development-guide.html) for instructions on how to
get started.
