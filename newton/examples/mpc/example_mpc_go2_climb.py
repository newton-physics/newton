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
# Example MPC Go2 Climb
#
# DIAL-MPC Go2 quadruped climbing onto a box, following the reward
# structure from the DIAL-MPC paper (Yin et al., 2024): head-position
# goal reaching with foot contact bonus.
#
# Command: python -m newton.examples mpc_go2_climb
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
    quat_rotate,
    quat_rotate_inv,
    quat_to_yaw,
    render_video_from_trajectory,
)

N_JOINTS = 12
BOX_X = 1.0          # box center X position [m]
BOX_HALF_HEIGHT = 0.10  # box half-height [m] (full height = 20cm)
BOX_HALF_LEN = 0.5   # box half-length in X [m]
BOX_HALF_WIDTH = 0.5  # box half-width in Y [m]

# Head offset from torso CoM in body frame (Go2 head is ~28.5cm forward)
HEAD_OFFSET = np.array([0.285, 0.0, 0.0], dtype=np.float32)


# ============================================================
# Go2 Robot + Box Builder
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


def _add_box_to_builder(builder):
    """Add a kinematic (immovable, visible) box platform to the builder."""
    box_body = builder.add_body(
        xform=wp.transform(wp.vec3(BOX_X, 0.0, BOX_HALF_HEIGHT)),
        is_kinematic=True,
    )
    builder.add_shape_box(
        body=box_body,
        hx=BOX_HALF_LEN,
        hy=BOX_HALF_WIDTH,
        hz=BOX_HALF_HEIGHT,
        cfg=builder.default_shape_cfg,
    )


def build_go2_with_box_visual():
    usd_path, go2_config = _load_go2_config()
    builder, _ = _make_go2_builder(usd_path, go2_config, load_visual_shapes=True)
    _add_box_to_builder(builder)
    return builder, builder.default_shape_cfg


# ============================================================
# Climb Reward — Paper-style: head position goal + foot contact
# ============================================================


def climb_reward(rollout_sim, config, t, actions=None, n_actual=None):
    """Reward for climbing onto a box (paper-style goal reaching).

    Following DIAL-MPC paper: dominant term is negative squared distance
    from head to target position on top of the box. Very light upright
    penalty. No velocity or height tracking.
    """
    bp, bq, bv, ba = rollout_sim.get_base_states()
    N = n_actual or bp.shape[0]
    bp, bq, bv, ba = bp[:N], bq[:N], bv[:N], ba[:N]

    # Compute head position in world frame
    head_local = np.tile(HEAD_OFFSET, (N, 1))
    head_world = bp + quat_rotate(bq, head_local)

    # Target: center of box top, at standing height above box
    target_pos = np.array([BOX_X, 0.0, 2 * BOX_HALF_HEIGHT + config.standing_height],
                          dtype=np.float32)

    # === Primary: head-to-target distance (weight 1.0, paper-style) ===
    reward_pos = -1.0 * np.sum((head_world - target_pos[None, :]) ** 2, axis=1)

    # === Upright (very light, weight 0.01 in paper) ===
    up_body = compute_up_in_body(bq)
    up_world = np.zeros_like(up_body)
    up_world[:, 2] = 1.0
    reward_upright = -0.01 * np.sum((up_body - up_world) ** 2, axis=1)

    # === Yaw (weight 0.3 in paper) ===
    yaw = quat_to_yaw(bq)
    reward_yaw = -0.3 * (np.arctan2(np.sin(yaw), np.cos(yaw)) ** 2)

    # === Action regularization ===
    reward_action = np.zeros(N, dtype=np.float32)
    if actions is not None:
        reward_action = -0.005 * np.mean(actions ** 2, axis=1)

    reward = reward_pos + reward_upright + reward_yaw + reward_action

    # Termination: only if robot falls completely
    terminated = bp[:, 2] < 0.05
    return np.where(terminated, -100.0, reward)


# ============================================================
# Newton Example
# ============================================================


class Example:
    """DIAL-MPC Go2 box climbing example."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()

        self.mpc_config = mppi_config_from_args(
            args,
            h_sample=25,         # paper uses 25
            h_node=5,            # paper uses 5
            sigma_scale=1.0,
            n_diffuse=4,
            n_diffuse_init=10,
            temp_sample=0.05,
        )
        self.mpc_config.standing_height = 0.30  # calibrated below

        # Build visual model with box
        usd_path, self.go2_config = _load_go2_config()
        robot_builder, _ = _make_go2_builder(usd_path, self.go2_config, load_visual_shapes=True)
        _add_box_to_builder(robot_builder)

        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(robot_builder, world_count=1)
        scene.add_ground_plane(cfg=robot_builder.default_shape_cfg)

        self.model = scene.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=True, solver="newton",
            ls_iterations=10, nconmax=50, njmax=150,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        # Wide action range for climbing (paper uses asymmetric ranges)
        home_pos = np.array(self.go2_config["mjw_joint_pos"], dtype=np.float32)
        self.joint_range_low = home_pos - 0.5
        self.joint_range_high = home_pos + 0.5

        np.random.seed(42)

        self.sim_time = 0.0
        self.sim_step = 0
        self._settle()

        settled_z = self.state_0.joint_q.numpy()[2]
        self.mpc_config.standing_height = float(settled_z)
        target_z = 2 * BOX_HALF_HEIGHT + settled_z
        print(f"  Settled height: {settled_z:.4f}m")
        print(f"  Box: height={2 * BOX_HALF_HEIGHT:.2f}m, "
              f"target head pos=({BOX_X:.1f}, 0, {target_z:.2f})")

        # Build rollout sim with box
        physics_builder, _ = _make_go2_builder(usd_path, self.go2_config, load_visual_shapes=False)
        _add_box_to_builder(physics_builder)
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
            reward_fn=climb_reward,
        )

        self.trajectory = {
            "body_q": [], "body_qd": [], "time": [], "reward": [],
            "velocity": [], "height": [], "pitch": [], "action": [],
        }
        self.frame_dt = self.mpc_config.ctrl_dt

    def _settle(self):
        home = np.array(self.go2_config["mjw_joint_pos"], dtype=np.float32)
        full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        full[6:6 + N_JOINTS] = home
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
        full[6:6 + N_JOINTS] = joint_targets
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
        pitch = compute_pitch(base_quat.reshape(1, 4))[0]

        self.trajectory["body_q"].append(bq)
        self.trajectory["body_qd"].append(bqd)
        self.trajectory["time"].append(self.sim_time)
        self.trajectory["reward"].append(reward)
        self.trajectory["velocity"].append(bqd[0, 0])
        self.trajectory["height"].append(bq[0, 2])
        self.trajectory["pitch"].append(float(pitch))
        self.trajectory["action"].append(action.copy())

        if self.sim_step % 20 == 0 or self.sim_step == self.mpc_config.n_steps - 1:
            # Compute head position
            head_local = HEAD_OFFSET.reshape(1, 3)
            head_world = bq[0, :3] + quat_rotate(base_quat.reshape(1, 4), head_local)[0]
            target_z = 2 * BOX_HALF_HEIGHT + self.mpc_config.standing_height
            box_front = BOX_X - BOX_HALF_LEN
            on_box = bq[0, 0] > box_front and bq[0, 2] > target_z - 0.05
            tag = "ON BOX" if on_box else ""
            print(
                f"  Step {self.sim_step:4d}/{self.mpc_config.n_steps}  "
                f"t={self.sim_time:.2f}s  head=({head_world[0]:.2f},{head_world[2]:.2f})  "
                f"z={bq[0, 2]:.3f}m  pitch={np.degrees(pitch):+.1f}°  "
                f"rew={reward:+.3f}  {tag}"
            )

        self.sim_time += self.frame_dt
        self.sim_step += 1

    def render(self):
        bq = self.state_0.body_q.numpy()
        base_pos = bq[0, :3]
        cam = wp.vec3(float(base_pos[0]), float(base_pos[1]) + 2.5, 0.6)
        self.viewer.set_camera(pos=cam, pitch=-5.0, yaw=-90.0)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        traj = self.trajectory
        n = len(traj["body_q"])
        if n == 0:
            raise ValueError("No trajectory data")

        heights = np.array(traj["height"])
        xs = np.array([bq[0, 0] for bq in traj["body_q"]])
        pitches = np.array(traj["pitch"])
        actions = np.array(traj["action"])

        box_front = BOX_X - BOX_HALF_LEN
        box_top_z = 2 * BOX_HALF_HEIGHT + self.mpc_config.standing_height
        final_x = xs[-1]
        final_z = heights[-1]

        print(f"\n{'='*60}")
        print("Go2 Climb Verification")
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

        check("no collapse (min height > 0.05m)",
              heights.min() > 0.05, f"min={heights.min():.3f}m")

        final_quat = traj["body_q"][-1][0, 3:7]
        up_z = compute_up_in_body(final_quat.reshape(1, 4))[0, 2]
        check("final upright (up_z > 0.3)",
              up_z > 0.3, f"up_z={up_z:.3f}")

        check("robot past box front edge (x > box_front)",
              final_x > box_front,
              f"final_x={final_x:.3f}m, box_front={box_front:.2f}m")

        check("robot ON the box (z > box_top - 5cm)",
              final_z > box_top_z - 0.05,
              f"final_z={final_z:.3f}m, box_top_z={box_top_z:.2f}m")

        # Height gain check
        on_box_mask = xs > box_front
        if on_box_mask.any():
            max_z = heights[on_box_mask].max()
            gain = max_z - heights[0]
            check("height gain on box > 10cm",
                  gain > 0.10,
                  f"gain={gain:.3f}m (max={max_z:.3f}, init={heights[0]:.3f})")
        else:
            check("height gain on box > 10cm", False, "never reached box")

        dist = final_x - xs[0]
        check("forward progress > 0.3m",
              dist > 0.3, f"dist={dist:.3f}m")

        # Smoothness
        if len(actions) > 2:
            act_diffs = np.abs(np.diff(actions, axis=0))
            check("smooth actions (mean jump < 0.4)",
                  act_diffs.mean() < 0.4,
                  f"mean={act_diffs.mean():.3f}")

        print(f"\n  Summary: {n_pass} passed, {n_fail} failed")
        print(f"  Final: x={final_x:.2f}m, z={final_z:.3f}m")
        print(f"  Box: front={box_front:.2f}m, top_z={box_top_z:.2f}m")
        if on_box_mask.any():
            print(f"  Height gain: {gain:.3f}m")
        print(f"  Max pitch: {np.degrees(np.abs(pitches).max()):.1f}°")
        print(f"{'='*60}")

        if n_fail > 0:
            raise ValueError(f"{n_fail} verification checks failed")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_mpc_args(parser)
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
        render_video_from_trajectory(example.trajectory, output, build_go2_with_box_visual)
