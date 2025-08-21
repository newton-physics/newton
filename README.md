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

<table>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_pendulum.py">
        <img src="docs/images/examples/example_basic_pendulum.jpg" alt="Pendulum">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_urdf.py">
        <img src="docs/images/examples/example_basic_urdf.jpg" alt="URDF">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_viewer.py">
        <img src="docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples basic_pendulum</code>
    </td>
    <td align="center">
      <code>python -m newton.examples basic_urdf</code>
    </td>
    <td align="center">
      <code>python -m newton.examples basic_viewer</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_shapes.py">
        <img src="docs/images/examples/example_basic_shapes.jpg" alt="Shapes">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/basic/example_basic_joints.py">
        <img src="docs/images/examples/example_basic_joints.jpg" alt="Joints">
      </a>
    </td>
    <td align="center" width="33%">
      <!-- <a href="newton/examples/basic/example_basic_viewer.py">
        <img src="docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a> -->
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples basic_shapes</code>
    </td>
    <td align="center">
      <code>python -m newton.examples basic_joints</code>
    </td>
    <td align="center">
      <!-- <code>python -m newton.examples basic_viewer</code> -->
    </td>
  </tr>
</table>

## Cloth Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_bending.py">
        <img src="docs/images/examples/example_cloth_bending.jpg" alt="Cloth Bending">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_hanging.py">
        <img src="docs/images/examples/example_cloth_hanging.jpg" alt="Cloth Hanging">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_style3d.py">
        <img src="docs/images/examples/example_cloth_style3d.jpg" alt="Cloth Style3D">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples cloth_bending</code>
    </td>
    <td align="center">
      <code>python -m newton.examples cloth_hanging</code>
    </td>
    <td align="center">
      <code>python -m newton.examples cloth_style3d</code>
    </td>
  </tr>
</table>

## MPM Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/mpm/example_mpm_granular.py">
        <img src="docs/images/examples/example_mpm_granular.jpg" alt="MPM Granular">
      </a>
    </td>
    <td align="center" width="33%">
      <!-- Future MPM example -->
    </td>
    <td align="center" width="33%">
      <!-- Future MPM example -->
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples mpm_granular</code>
    </td>
    <td align="center">
      <!-- Future MPM example -->
    </td>
    <td align="center">
      <!-- Future MPM example -->
    </td>
  </tr>
</table>

## Example Options

The examples support the following common line arguments:

| Argument        | Description                                                                                         | Default                      |
| --------------- | --------------------------------------------------------------------------------------------------- | ---------------------------- |
| `--viewer`      | Viewer type: `gl` (OpenGL window), `usd` (USD file output), `rerun` (ReRun), or `null` (no viewer). | `gl`                         |
| `--device`      | Compute device to use, e.g., `cpu`, `cuda:0`, etc.                                                  | `None` (default Warp device) |
| `--num-frames`  | Number of frames to simulate (for USD output).                                                      | `100`                        |
| `--output-path` | Output path for USD files (required if `--viewer usd` is used).                                     | `None`                       |

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
