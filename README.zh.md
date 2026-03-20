[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push - AWS GPU](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml)

# Newton

<p align="center">
  <a href="README.md">English</a> ·
  <strong>中文</strong>
</p>

Newton 是一个基于 [NVIDIA Warp](https://github.com/NVIDIA/warp) 的 GPU 加速物理仿真引擎，专为机器人专家和仿真研究人员设计。

Newton 扩展并泛化了 Warp 的（[已弃用](https://github.com/NVIDIA/warp/discussions/735)）`warp.sim` 模块，并集成了 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 作为其主要后端。Newton 强调基于 GPU 的计算、[OpenUSD](https://openusd.org/) 支持、可微性以及用户定义的可扩展性，从而促进快速迭代和可扩展的机器人仿真。

Newton 是一个由社区构建和维护的 [Linux 基金会 (Linux Foundation)](https://www.linuxfoundation.org/) 项目。代码采用 [Apache-2.0](https://github.com/newton-physics/newton/blob/main/LICENSE.md) 协议授权。文档采用 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) 协议授权。

Newton 由 [迪士尼研究中心 (Disney Research)](https://www.disneyresearch.com/)、[Google DeepMind](https://deepmind.google/) 和 [NVIDIA](https://www.nvidia.com/) 发起。

## 要求 (Requirements)

- **Python:** 3.10+
- **操作系统 (OS):** Linux (x86-64, aarch64), Windows (x86-64), 或 macOS (仅 CPU)
- **显卡 (GPU):** NVIDIA GPU (Maxwell 或更新版本), 驱动 545 或更新版本 (CUDA 12)。无需本地安装 CUDA Toolkit。macOS 在 CPU 上运行。

有关详细的系统要求和测试过的配置，请参阅[安装指南](https://newton-physics.github.io/newton/latest/guide/installation.html)。

## 快速开始 (Quickstart)

```bash
pip install "newton[examples]"
python -m newton.examples basic_pendulum
```

要使用 [uv](https://docs.astral.sh/uv/) 从源码安装，请参阅[安装指南](https://newton-physics.github.io/newton/latest/guide/installation.html)。

## 示例 (Examples)

在运行以下示例之前，请安装带有示例额外选项的 Newton：

```bash
pip install "newton[examples]"
```

如果您使用 uv 从源码安装，请在以下命令中用 `uv run` 代替 `python`。

<table>
  <tr>
    <td colspan="3"><h3>基础示例 (Basic Examples)</h3></td>
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
    <td colspan="3"><h3>机器人示例 (Robot Examples)</h3></td>
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
    <td colspan="3"><h3>线缆示例 (Cable Examples)</h3></td>
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
    <td colspan="3"><h3>布料示例 (Cloth Examples)</h3></td>
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
    <td colspan="3"><h3>逆运动学示例 (Inverse Kinematics Examples)</h3></td>
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
    <td colspan="3"><h3>MPM 示例 (MPM Examples)</h3></td>
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
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>传感器示例 (Sensor Examples)</h3></td>
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
    <td colspan="3"><h3>选择示例 (Selection Examples)</h3></td>
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
    <td colspan="3"><h3>可微仿真示例 (DiffSim Examples)</h3></td>
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
    <td colspan="3"><h3>多物理场示例 (Multi-Physics Examples)</h3></td>
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
    <td colspan="3"><h3>碰撞示例 (Contacts Examples)</h3></td>
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
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples pyramid</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>软体示例 (Softbody Examples)</h3></td>
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

### 示例选项 (Example Options)

这些示例支持以下命令行参数：

| 参数 (Argument) | 描述 (Description)                                                                                          | 默认值 (Default)             |
| --------------- | --------------------------------------------------------------------------------------------------- | ---------------------------- |
| `--viewer`      | 查看器类型：`gl` (OpenGL 窗口), `usd` (USD 文件输出), `rerun` (ReRun), 或 `null` (无查看器)。 | `gl`                         |
| `--device`      | 要使用的计算设备，例如 `cpu`, `cuda:0` 等。                                                  | `None` (默认 Warp 设备) |
| `--num-frames`  | 要仿真的帧数（用于 USD 输出）。                                                      | `100`                        |
| `--output-path` | USD 文件的输出路径（如果使用 `--viewer usd` 则为必填项）。                                     | `None`                       |

某些示例可能会添加额外的参数（详情请参阅其各自的源文件）。

### 示例用法 (Example Usage)

```bash
# 列出可用示例
python -m newton.examples

# 使用 USD 查看器运行并保存到 my_output.usd
python -m newton.examples basic_viewer --viewer usd --output-path my_output.usd

# 在选定设备上运行
python -m newton.examples basic_urdf --device cuda:0

# 组合选项
python -m newton.examples basic_viewer --viewer gl --num-frames 500 --device cpu
```

## 贡献与开发 (Contributing and Development)

有关如何为 Newton 做出贡献的说明，请参阅[贡献指南](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md)和[开发指南](https://newton-physics.github.io/newton/latest/guide/development.html)。

## 支持与社区讨论 (Support and Community Discussion)

如有疑问，在[主仓库发起讨论](https://github.com/newton-physics/newton/discussions)之前，请先查阅 [Newton 文档](https://newton-physics.github.io/newton/latest/guide/overview.html)。

## 行为准则 (Code of Conduct)

参与本社区即表示您同意遵守 Linux 基金会的[行为准则](https://lfprojects.org/policies/code-of-conduct/)。

## 项目治理、法律和成员 (Project Governance, Legal, and Members)

有关项目治理的更多信息，请参阅 [newton-governance 仓库](https://github.com/newton-physics/newton-governance)。
