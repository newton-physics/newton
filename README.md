[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push - AWS GPU](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml)

# Newton

Newton is a GPU-accelerated physics simulation engine built upon [NVIDIA Warp](https://github.com/NVIDIA/warp), specifically targeting roboticists and simulation researchers.

Newton extends and generalizes Warp's ([deprecated](https://github.com/NVIDIA/warp/discussions/735)) `warp.sim` module, and integrates
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as its primary backend. Newton emphasizes GPU-based computation, [OpenUSD](https://openusd.org/) support, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

Newton is a [Linux Foundation](https://www.linuxfoundation.org/) project that is community-built and maintained. Code is licensed under [Apache-2.0](https://github.com/newton-physics/newton/blob/main/LICENSE.md). Documentation is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Additional and third-party license texts are available in [`newton/licenses`](https://github.com/newton-physics/newton/tree/main/newton/licenses).

Newton was initiated by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Requirements

- **Python** 3.10+
- **OS:** Linux (x86-64, aarch64), Windows (x86-64), or macOS (CPU only)
- **GPU:** NVIDIA GPU (Maxwell or newer), driver 545 or newer (CUDA 12). No local CUDA Toolkit installation required. macOS runs on CPU.

For detailed system requirements and tested configurations, see the [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html).

## Quickstart

```bash
pip install "newton[examples]"
python -m newton.examples
```

To install from source with [uv](https://docs.astral.sh/uv/), see the [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html).

## Examples

Before running the examples below, install Newton with the examples extra:

```bash
pip install "newton[examples]"
```

If you installed from source with uv, substitute `uv run` for `python` in the commands below.

<table>
  <tr>
    <td colspan="3"><h3>Basic Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_pendulum.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_pendulum.jpg" alt="Pendulum">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_urdf.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_urdf.jpg" alt="URDF">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_viewer.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_pendulum</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_urdf</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_viewer</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_shapes.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_shapes.jpg" alt="Shapes">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_joints.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_joints.jpg" alt="Joints">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_conveyor.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_conveyor.jpg" alt="Conveyor">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_shapes</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_joints</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_conveyor</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_heightfield.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_heightfield.jpg" alt="Heightfield">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_recording.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_recording.jpg" alt="Recording">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_replay_viewer.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_replay_viewer.jpg" alt="Replay Viewer">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_heightfield</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples recording</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples replay_viewer</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_plotting.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_plotting.jpg" alt="Plotting">
      </a>
    </td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_plotting</code>
    </td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="3"><h3>Robot Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_cartpole.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_cartpole.jpg" alt="Cartpole">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_g1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_g1.jpg" alt="G1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_h1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_h1.jpg" alt="H1">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_cartpole</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_g1</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_h1</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_d.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_anymal_d.jpg" alt="Anymal D">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_c_walk.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_anymal_c_walk.jpg" alt="Anymal C Walk">
      </a>
    </td>
    <td></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_anymal_d</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_anymal_c_walk</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_policy.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_policy.jpg" alt="Policy">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_ur10.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_ur10.jpg" alt="UR10">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_panda_hydro.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_panda_hydro.jpg" alt="Panda Hydro">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_policy</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_ur10</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_panda_hydro</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_allegro_hand.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_allegro_hand.jpg" alt="Allegro Hand">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_allegro_hand</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Cable Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_twist.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_twist.jpg" alt="Cable Twist">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_y_junction.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_y_junction.jpg" alt="Cable Y-Junction">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_bundle_hysteresis.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_bundle_hysteresis.jpg" alt="Cable Bundle Hysteresis">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_twist</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_y_junction</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_bundle_hysteresis</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_pile.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_pile.jpg" alt="Cable Pile">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_pile</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Cloth Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_bending.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_bending.jpg" alt="Cloth Bending">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_hanging.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_hanging.jpg" alt="Cloth Hanging">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_style3d.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_style3d.jpg" alt="Cloth Style3D">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_bending</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_hanging</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_style3d</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_h1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_h1.jpg" alt="Cloth H1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_twist.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_twist.jpg" alt="Cloth Twist">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_franka.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_franka.jpg" alt="Cloth Franka">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_h1</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_twist</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_franka</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_rollers.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_rollers.jpg" alt="Cloth Rollers">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_poker_cards.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_poker_cards.jpg" alt="Cloth Poker Cards">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_rollers</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_poker_cards</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Inverse Kinematics Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_franka.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_franka.jpg" alt="IK Franka">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_h1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_h1.jpg" alt="IK H1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_custom.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_custom.jpg" alt="IK Custom">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_franka</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_h1</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_custom</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_cube_stacking.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_cube_stacking.jpg" alt="IK Cube Stacking">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_cube_stacking</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>MPM Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_granular.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_granular.jpg" alt="MPM Granular">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_anymal.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_anymal.jpg" alt="MPM Anymal">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_twoway_coupling.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_twoway_coupling.jpg" alt="MPM Two-Way Coupling">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_granular</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_anymal</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_twoway_coupling</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_grain_rendering.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_grain_rendering.jpg" alt="MPM Grain Rendering">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_multi_material.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_multi_material.jpg" alt="MPM Multi Material">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_viscous.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_viscous.jpg" alt="MPM Viscous">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_grain_rendering</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_multi_material</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_viscous</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_beam_twist.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_beam_twist.jpg" alt="MPM Beam Twist">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_snow_ball.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_snow_ball.jpg" alt="MPM Snow Ball">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_beam_twist</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_snow_ball</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Sensor Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_contact.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_contact.jpg" alt="Sensor Contact">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_tiled_camera.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_tiled_camera.jpg" alt="Sensor Tiled Camera">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_imu.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_imu.jpg" alt="Sensor IMU">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples sensor_contact</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples sensor_tiled_camera</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples sensor_imu</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Selection Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_cartpole.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_cartpole.jpg" alt="Selection Cartpole">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_materials.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_materials.jpg" alt="Selection Materials">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_articulations.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_articulations.jpg" alt="Selection Articulations">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples selection_cartpole</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples selection_materials</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples selection_articulations</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_multiple.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_multiple.jpg" alt="Selection Multiple">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples selection_multiple</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>DiffSim Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_ball.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_ball.jpg" alt="DiffSim Ball">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_cloth.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_cloth.jpg" alt="DiffSim Cloth">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_drone.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_drone.jpg" alt="DiffSim Drone">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_ball</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_cloth</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_drone</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_spring_cage.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_spring_cage.jpg" alt="DiffSim Spring Cage">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_soft_body.jpg" alt="DiffSim Soft Body">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_bear.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_bear.jpg" alt="DiffSim Quadruped">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_spring_cage</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_soft_body</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_bear</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Multi-Physics Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_softbody_gift.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_gift.jpg" alt="Softbody Gift">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_softbody_dropping_to_cloth.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_dropping_to_cloth.jpg" alt="Softbody Dropping to Cloth">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_gift</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_dropping_to_cloth</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Contacts Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_nut_bolt_hydro.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_nut_bolt_hydro.jpg" alt="Nut Bolt Hydro">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_nut_bolt_sdf.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_nut_bolt_sdf.jpg" alt="Nut Bolt SDF">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_brick_stacking.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_brick_stacking.jpg" alt="Brick Stacking">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples nut_bolt_hydro</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples nut_bolt_sdf</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples brick_stacking</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_pyramid.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_pyramid.jpg" alt="Pyramid">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_contacts_rj45_plug.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_contacts_rj45_plug.jpg" alt="RJ45 Plug">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples pyramid</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples contacts_rj45_plug</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Softbody Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_hanging.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_hanging.jpg" alt="Softbody Hanging">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_franka.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_franka.jpg" alt="Softbody Franka">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_hanging</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_franka</code>
    </td>
    <td align="center" width="33%">
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
python -m newton.examples --list

# Run with the USD viewer and save to my_output.usd
python -m newton.examples basic_viewer --viewer usd --output-path my_output.usd

# Run on a selected device
python -m newton.examples basic_urdf --device cuda:0

# Combine options
python -m newton.examples basic_viewer --viewer gl --num-frames 500 --device cpu
```

## Contributing and Development

See the [contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md) and the [development guide](https://newton-physics.github.io/newton/latest/guide/development.html) for instructions on how to contribute to Newton.

## Support and Community Discussion

For questions, please consult the [Newton documentation](https://newton-physics.github.io/newton/latest/guide/overview.html) first before creating [a discussion in the main repository](https://github.com/newton-physics/newton/discussions).

## Code of Conduct

By participating in this community, you agree to abide by the Linux Foundation [Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

## Project Governance, Legal, and Members

Please see the [newton-governance repository](https://github.com/newton-physics/newton-governance) for more information about project governance.
