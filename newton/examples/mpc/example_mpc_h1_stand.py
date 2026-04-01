# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example MPC Go2 Turn
#
# DIAL-MPC Go2 quadruped turning in place.
# The reward encourages yaw rotation while maintaining height and
# upright posture — demonstrates MPC with angular velocity targets.
#
# Command: python -m newton.examples mpc_h1_stand
#
###########################################################################

from __future__ import annotations

import sys

import numpy as np
import yaml

import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode

from newton.examples.mpc.dial_mpc import (
    DIALMPCController,
    RolloutSim,
    add_mpc_args,
    compute_pitch,
    compute_up_in_body,
    mppi_config_from_args,
    quat_rotate_inv,
    quat_to_yaw,
    render_video_from_trajectory,
)

N_JOINTS = 12


# ============================================================
# Go2 Robot Builder (shared with walk example)
# ============================================================


def _load_go2_config():
    asset_path = newton.utils.download_asset("unitree_go2")
    yaml_path = str(asset_path / "rl_policies" / "go2.yaml")
    usd_path = str(asset_path / "usd" / "go2.usda")
    with open(yaml_path, encoding="utf-8") as f:
        go2_config = yaml.safe_load(f)
    return usd_path, go2_config


def _make_go2_builder(usd_path, go2_config, load_visual_shapes=True):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0,
    )
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    builder.add_usd(
        usd_path,
        xform=wp.transform(wp.vec3(0, 0, 0.4)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=load_visual_shapes,
        load_visual_shapes=load_visual_shapes,
    )
    builder.approximate_meshes("convex_hull")

    builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
    builder.joint_q[7:] = go2_config["mjw_joint_pos"]
    for i in range(go2_config["num_dofs"]):
        builder.joint_target_ke[i + 6] = 100.0
        builder.joint_target_kd[i + 6] = 2.0
        builder.joint_armature[i + 6] = go2_config["mjw_joint_armature"][i]
        builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)
        builder.joint_target_pos[6 + i] = go2_config["mjw_joint_pos"][i]

    return builder, go2_config


def build_go2_visual():
    usd_path, go2_config = _load_go2_config()
    builder, _ = _make_go2_builder(usd_path, go2_config, load_visual_shapes=True)
    return builder, builder.default_shape_cfg


# ============================================================
# Turn Reward Function
# ============================================================


def turn_reward(rollout_sim, config, t, actions=None, n_actual=None):
    """Reward for turning in place: yaw rate tracking + stability."""
    bp, bq, bv, ba = rollout_sim.get_base_states()
    N = n_actual or bp.shape[0]
    bp, bq, bv, ba = bp[:N], bq[:N], bv[:N], ba[:N]

    ramp = min(1.0, t / config.ramp_up_time) if config.ramp_up_time > 0 else 1.0
    ab = quat_rotate_inv(bq, ba)

    # Yaw rate tracking (primary objective: turn at target rate)
    target_vyaw = config.target_vyaw * ramp
    reward_yaw_rate = -3.0 * ((ab[:, 2] - target_vyaw) ** 2)

    # Penalize linear velocity (stay in place while turning)
    reward_vel = -1.0 * (bv[:, 0] ** 2 + bv[:, 1] ** 2)

    # Height maintenance
    h_err = bp[:, 2] - config.target_height
    reward_height = -2.0 * np.where(h_err < 0, 10.0 * h_err ** 2, h_err ** 2)

    # Upright + pitch
    up_body = compute_up_in_body(bq)
    up_world = np.zeros_like(up_body)
    up_world[:, 2] = 1.0
    reward_upright = -2.0 * np.sum((up_body - up_world) ** 2, axis=1)

    pitch = compute_pitch(bq)
    reward_pitch = -3.0 * (pitch ** 2)

    # Body pitch/roll rate smoothness
    reward_body_rate = -0.3 * (ab[:, 0] ** 2 + ab[:, 1] ** 2)

    # Action regularization
    reward_action = np.zeros(N, dtype=np.float32)
    if actions is not None:
        reward_action = -0.01 * np.mean(actions ** 2, axis=1)

    reward = (reward_yaw_rate + reward_vel + reward_height + reward_upright +
              reward_pitch + reward_body_rate + reward_action)

    terminated = (bp[:, 2] < 0.08) | (up_body[:, 2] < -0.3)
    return np.where(terminated, -100.0, reward)


# ============================================================
# Newton Example
# ============================================================


class Example:
    """DIAL-MPC Go2 turning in place example."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()

        self.mpc_config = mppi_config_from_args(args)
        self.mpc_config.target_vyaw = getattr(args, "target_vyaw", 1.0)  # rad/s
        self.mpc_config.target_height = 0.30

        usd_path, self.go2_config = _load_go2_config()
        robot_builder, _ = _make_go2_builder(usd_path, self.go2_config, load_visual_shapes=True)

        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(robot_builder, world_count=1)
        scene.add_ground_plane(cfg=robot_builder.default_shape_cfg)

        self.model = scene.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=True, solver="newton",
            ls_iterations=10, nconmax=30, njmax=100,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        home_pos = np.array(self.go2_config["mjw_joint_pos"], dtype=np.float32)
        self.joint_range_low = home_pos - 0.2
        self.joint_range_high = home_pos + 0.2

        np.random.seed(42)

        self.sim_time = 0.0
        self.sim_step = 0
        self._settle()

        settled_z = self.state_0.joint_q.numpy()[2]
        self.mpc_config.target_height = float(settled_z)
        print(f"  Settled height: {settled_z:.4f}m")

        physics_builder, _ = _make_go2_builder(usd_path, self.go2_config, load_visual_shapes=False)
        n_rollout = self.mpc_config.n_samples + 1
        self.rollout_sim = RolloutSim(
            physics_builder, n_rollout,
            ctrl_dt=self.mpc_config.ctrl_dt,
            sim_substeps=self.mpc_config.sim_substeps,
            device=self.device,
        )

        self.controller = DIALMPCController(
            self.mpc_config, self.rollout_sim, N_JOINTS,
            self.joint_range_low, self.joint_range_high,
            reward_fn=turn_reward,
        )

        self.trajectory = {
            "body_q": [], "body_qd": [], "time": [], "reward": [],
            "yaw": [], "yaw_rate": [], "height": [], "action": [],
        }
        self.frame_dt = self.mpc_config.ctrl_dt

    def _settle(self):
        home = np.array(self.go2_config["mjw_joint_pos"], dtype=np.float32)
        full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        full[6:] = home
        wp.copy(self.control.joint_target_pos,
                wp.array(full, dtype=wp.float32, device=self.device))
        sim_dt = self.mpc_config.ctrl_dt / self.mpc_config.sim_substeps
        for _ in range(100):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_step >= self.mpc_config.n_steps:
            return

        jq = self.state_0.joint_q.numpy().copy()
        jqd = self.state_0.joint_qd.numpy().copy()

        action, reward = self.controller.plan(jq, jqd, self.sim_time)

        joint_targets = self.controller.act_to_joint(action)
        full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        full[6:] = joint_targets
        wp.copy(self.control.joint_target_pos,
                wp.array(full, dtype=wp.float32, device=self.device))

        sim_dt = self.mpc_config.ctrl_dt / self.mpc_config.sim_substeps
        for _ in range(self.mpc_config.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        bq = self.state_0.body_q.numpy().copy()
        bqd = self.state_0.body_qd.numpy().copy()
        base_quat = bq[0, 3:7]
        yaw = quat_to_yaw(base_quat.reshape(1, 4))[0]
        ab = quat_rotate_inv(base_quat.reshape(1, 4), bqd[0, 3:6].reshape(1, 3))[0]

        self.trajectory["body_q"].append(bq)
        self.trajectory["body_qd"].append(bqd)
        self.trajectory["time"].append(self.sim_time)
        self.trajectory["reward"].append(reward)
        self.trajectory["yaw"].append(float(yaw))
        self.trajectory["yaw_rate"].append(float(ab[2]))
        self.trajectory["height"].append(bq[0, 2])
        self.trajectory["action"].append(action.copy())

        if self.sim_step % 20 == 0 or self.sim_step == self.mpc_config.n_steps - 1:
            print(
                f"  Step {self.sim_step:4d}/{self.mpc_config.n_steps}  "
                f"t={self.sim_time:.2f}s  yaw={np.degrees(yaw):+.1f}°  "
                f"yaw_rate={ab[2]:+.2f}rad/s  z={bq[0, 2]:.3f}m  rew={reward:+.3f}"
            )

        self.sim_time += self.frame_dt
        self.sim_step += 1

    def render(self):
        bq = self.state_0.body_q.numpy()
        base_pos = bq[0, :3]
        # Top-down view for turning
        cam = wp.vec3(float(base_pos[0]), float(base_pos[1]) + 2.0, 1.5)
        self.viewer.set_camera(pos=cam, pitch=-30.0, yaw=-90.0)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        traj = self.trajectory
        n = len(traj["body_q"])
        if n == 0:
            raise ValueError("No trajectory data")

        heights = np.array(traj["height"])
        yaws = np.array(traj["yaw"])
        yaw_rates = np.array(traj["yaw_rate"])

        ramp_steps = int(self.mpc_config.ramp_up_time / self.mpc_config.ctrl_dt)
        post_ramp = slice(min(ramp_steps + 20, n), n)

        print(f"\n{'='*60}")
        print("Go2 Turn Verification")
        print(f"{'='*60}")
        n_pass = 0
        n_fail = 0

        def check(name, cond, msg=""):
            nonlocal n_pass, n_fail
            if cond:
                print(f"  PASS: {name}")
                n_pass += 1
            else:
                print(f"  FAIL: {name} — {msg}")
                n_fail += 1

        check("height > 0.15m always", heights.min() > 0.15, f"min={heights.min():.3f}m")

        final_quat = traj["body_q"][-1][0, 3:7]
        up_z = compute_up_in_body(final_quat.reshape(1, 4))[0, 2]
        check("stays upright (up_z > 0.7)", up_z > 0.7, f"up_z={up_z:.3f}")

        # Check that yaw changed significantly (turned)
        total_yaw = np.abs(yaws[-1] - yaws[0])
        check("total yaw change > 30°",
              total_yaw > np.radians(30),
              f"total_yaw={np.degrees(total_yaw):.1f}°")

        mean_yaw_rate = np.abs(yaw_rates[post_ramp]).mean()
        check("mean yaw rate > 0.3 rad/s",
              mean_yaw_rate > 0.3,
              f"mean={mean_yaw_rate:.2f} rad/s")

        print(f"\n  Summary: {n_pass} passed, {n_fail} failed")
        print(f"  Height: {heights.mean():.3f}m ± {heights.std():.3f}m")
        print(f"  Total yaw: {np.degrees(yaws[-1] - yaws[0]):.1f}°")
        print(f"  Mean yaw rate: {mean_yaw_rate:.2f} rad/s")
        print(f"{'='*60}")

        if n_fail > 0:
            raise ValueError(f"{n_fail} verification checks failed")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_mpc_args(parser)
        parser.add_argument("--target-vyaw", type=float, default=1.0, help="Target yaw rate (rad/s)")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    sys.argv = [a for a in sys.argv if a not in ("--headless",)]
    if "--viewer" not in " ".join(sys.argv):
        sys.argv.extend(["--viewer", "null"])

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

    output = getattr(args, "output", "")
    if output and example.trajectory["body_q"]:
        render_video_from_trajectory(example.trajectory, output, build_go2_visual)
