[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push - AWS GPU](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml)

**This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

# Newton

Newton is a GPU-accelerated physics simulation engine built upon [NVIDIA Warp](https://github.com/NVIDIA/warp), specifically targeting roboticists and simulation researchers.

Newton extends and generalizes Warp's ([deprecated](https://github.com/NVIDIA/warp/discussions/735)) `warp.sim` module, and integrates
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as its primary backend. Newton emphasizes GPU-based computation, [OpenUSD](https://openusd.org/) support, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

Newton is a [Linux Foundation](https://www.linuxfoundation.org/) project that is community-built and maintained. It is permissively licensed under the [Apache-2.0 license](https://github.com/newton-physics/newton/blob/main/LICENSE.md).

Newton was initiated by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Quickstart

During the alpha development phase, we recommend using the [uv](https://docs.astral.sh/uv/) Python package and project manager. You may find uv installation instructions in the [Newton Installation Guide](https://newton-physics.github.io/newton/guide/installation.html#method-1-using-uv-recommended).

Once uv is installed, running Newton examples is straightforward:

```bash
# Clone the repository
git clone git@github.com:newton-physics/newton.git
cd newton

# set up the uv environment for running Newton examples
uv sync --extra examples

# run an example
uv run -m newton.examples basic_pendulum
```

See the [installation guide](https://newton-physics.github.io/newton/guide/installation.html) for detailed instructions that include steps for setting up a Python environment for use with Newton.

## Examples

Before running the examples below, set up the uv environment with:

```bash
uv sync --extra examples
```

### Basic Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_pendulum.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_pendulum.jpg" alt="Pendulum">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_urdf.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_urdf.jpg" alt="URDF">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_viewer.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples basic_pendulum</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples basic_urdf</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples basic_viewer</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_shapes.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_shapes.jpg" alt="Shapes">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_joints.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_joints.jpg" alt="Joints">
      </a>
    </td>
    <td align="center" width="33%">
      <!-- <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_viewer.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a> -->
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples basic_shapes</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples basic_joints</code>
    </td>
    <td align="center">
      <!-- <code>uv run -m newton.examples basic_viewer</code> -->
    </td>
  </tr>
</table>

### Robot Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_cartpole.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_cartpole.jpg" alt="Cartpole">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_humanoid.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_humanoid.jpg" alt="Humanoid">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_g1.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_g1.jpg" alt="G1">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples robot_cartpole</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples robot_humanoid</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples robot_g1</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_h1.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_h1.jpg" alt="H1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_d.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_anymal_d.jpg" alt="Anymal D">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_c_walk.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_anymal_c_walk.jpg" alt="Anymal C Walk">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples robot_h1</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples robot_anymal_d</code>
    </td>
    <td align="center">
      <code>uv run --extra torch-cu12 -m newton.examples robot_anymal_c_walk</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_policy.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_policy.jpg" alt="Policy">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_ur10.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_ur10.jpg" alt="UR10">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run --extra torch-cu12 -m newton.examples robot_policy</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples robot_ur10</code>
    </td>
    <td align="center">
    </td>
  </tr>
</table>

### Cloth Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_bending.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_bending.jpg" alt="Cloth Bending">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_hanging.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_hanging.jpg" alt="Cloth Hanging">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_style3d.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_style3d.jpg" alt="Cloth Style3D">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples cloth_bending</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples cloth_hanging</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples cloth_style3d</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_h1.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_h1.jpg" alt="Cloth H1">
      </a>
    </td>
  </tr>
 <tr>
    <td align="center">
      <code>uv run -m newton.examples cloth_h1</code>
    </td>
 </tr>
</table>

### Inverse Kinematics Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_franka.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_franka.jpg" alt="IK Franka">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_h1.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_h1.jpg" alt="IK H1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_benchmark.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_benchmark.jpg" alt="IK Benchmark">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples ik_franka</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples ik_h1</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples ik_benchmark</code>
    </td>

  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_custom.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_custom.jpg" alt="IK Custom">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_franka.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_franka.jpg" alt="Cloth Franka">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_twist.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_twist.jpg" alt="Cloth Twist">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples ik_custom</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples cloth_franka</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples cloth_twist</code>
    </td>
  </tr>

</table>

### MPM Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_granular.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_granular.jpg" alt="MPM Granular">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_anymal.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_anymal.jpg" alt="MPM Anymal">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_twoway_coupling.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_twoway_coupling.jpg" alt="MPM two-way coupling">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples mpm_granular</code>
    </td>
    <td align="center">
      <code>uv run --extra torch-cu12 -m newton.examples mpm_anymal</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples mpm_twoway_coupling</code>
    </td>
  </tr>
</table>

### Sensor Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_contact.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_contact.jpg" alt="Sensor Contact">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_tiled_camera.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_tiled_camera.jpg" alt="Sensor Contact">
      </a>
    </td>
    <td align="center" width="33%">
      <!-- Future Sensor example -->
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples sensor_contact</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples sensor_tiled_camera</code>
    </td>
    <td align="center">
      <!-- Future Sensor example -->
    </td>
  </tr>
</table>

### Selection Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_cartpole.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_cartpole.jpg" alt="Selection Cartpole">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_materials.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_materials.jpg" alt="Selection Materials">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_articulations.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_articulations.jpg" alt="Selection Articulations">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples selection_cartpole</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples selection_materials</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples selection_articulations</code>
    </td>
  </tr>
</table>

### DiffSim Examples

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_ball.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_ball.jpg" alt="DiffSim Ball">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_cloth.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_cloth.jpg" alt="DiffSim Cloth">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_drone.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_drone.jpg" alt="DiffSim Drone">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples diffsim_ball</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples diffsim_cloth</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples diffsim_drone</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_spring_cage.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_spring_cage.jpg" alt="DiffSim Spring Cage">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_soft_body.jpg" alt="DiffSim Soft Body">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_bear.py">
        <img src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_bear.jpg" alt="DiffSim Quadruped">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>uv run -m newton.examples diffsim_spring_cage</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples diffsim_soft_body</code>
    </td>
    <td align="center">
      <code>uv run -m newton.examples diffsim_bear</code>
    </td>
  </tr>
</table>

### Example Options

The examples support the following command-line arguments:

| Argument        | Description                                                                                         | Default                      |
| --------------- | --------------------------------------------------------------------------------------------------- | ---------------------------- |
| `--viewer`      | Viewer type: `gl` (OpenGL window), `usd` (USD file output), `rerun` (ReRun), or `null` (no viewer). | `gl`                         |
| `--device`      | Compute device to use, e.g., `cpu`, `cuda:0`, etc.                                                  | `None` (default Warp device) |
| `--num-frames`  | Number of frames to simulate (for USD output).                                                      | `100`                        |
| `--output-path` | Output path for USD files (required if `--viewer usd` is used).                                     | `None`                       |

Some examples may add additional arguments (see their respective source files for details).

### Example Usage

```bash
# List available examples
uv run -m newton.examples

# Run with the USD viewer and save to my_output.usd
uv run -m newton.examples basic_viewer --viewer usd --output-path my_output.usd

# Run on a selected device
uv run -m newton.examples basic_urdf --device cuda:0

# Combine options
uv run -m newton.examples basic_viewer --viewer gl --num-frames 500 --device cpu
```

## Contributing and Development

See the [contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md) and the [development guide](https://newton-physics.github.io/newton/guide/development.html) for instructions on how to contribute to Newton.

## Support and Community Discussion

For questions, please consult the [Newton documentation](https://newton-physics.github.io/newton/guide/overview.html) first before creating [a discussion in the main repository](https://github.com/newton-physics/newton/discussions).

## Code of Conduct

By participating in this community, you agree to abide by the Linux Foundation [Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

## Project Governance, Legal, and Members

Please see the [newton-governance repository](https://github.com/newton-physics/newton-governance) for more information about project governance.
