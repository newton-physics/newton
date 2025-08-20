
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

## Examples

## Basic Examples


| Example | Run Command | Thumbnail |
|---------|-------------|-----------|
| [example_basic_pendulum.py](newton/examples/basic/example_basic_pendulum.py) | `python -m newton.examples basic_pendulum` | ![Pendulum](docs/images/examples/example_basic_pendulum.png) |
| [example_basic_urdf.py](newton/examples/basic/example_basic_urdf.py) | `python -m newton.examples basic_urdf` | ![URDF](docs/images/examples/example_basic_urdf.png) |
| [example_basic_viewer.py](newton/examples/basic/example_basic_viewer.py) | `python -m newton.examples basic_viewer` | ![Viewer](docs/images/examples/example_basic_viewer.png) |

## Example Options

The examples support the following common line arguments:

| Argument                | Description                                                                                | Default         |
|-------------------------|--------------------------------------------------------------------------------------------|-----------------|
| `--viewer`              | Viewer type: `gl` (OpenGL window), `usd` (USD file output), `rerun` (ReRun), or `null` (no viewer).         | `gl`            |
| `--device`              | Compute device to use, e.g., `cpu`, `cuda:0`, etc.                                         | `None` (default Warp device) |
| `--num-frames`          | Number of frames to simulate (for USD output).                                             | `100`          |
| `--output-path`         | Output path for USD files (required if `--viewer usd` is used).                            | `None`          |

Some examples may add additional arguments (see their respective source files for details).

## Example Usage

    # Basic usage
    python -m newton.examples basic_pendulum

    # With uv
    uv run python -m newton.examples basic_pendulum

    # With viewer options  
    python -m newton.examples basic_viewer --viewer usd --output-path my_output.usd

    # With device selection
    python -m newton.examples basic_urdf --device cuda:0

    # Multiple arguments
    python -m newton.examples basic_viewer --viewer gl --num-frames 500 --device cpu




