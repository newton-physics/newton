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

"""
DIAL-MPC: Reusable framework for Diffusion-Inspired Annealing MPC.

Provides robot-agnostic building blocks for sampling-based model-predictive
control using diffusion-style annealing (MPPI variant).

Reference: "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control
via Diffusion-Style Annealing" (Yin et al., 2024, ICRA 2025)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

import warp as wp

import newton
import newton.examples


# ============================================================
# Quaternion Utilities (xyzw convention, matching Warp)
# ============================================================


def quat_rotate(q, v):
    """Rotate vector(s) v by quaternion(s) q. q: (...,4) xyzw, v: (...,3)."""
    qv, qw = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def quat_rotate_inv(q, v):
    """Rotate vector(s) v by inverse of quaternion(s) q."""
    qv, qw = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v - qw * t + np.cross(qv, t)


def quat_to_yaw(q):
    """Extract yaw angle from quaternion(s). q: (...,4) xyzw."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_pitch(base_quat):
    """Compute pitch angle (forward lean) from quaternion(s). Returns radians."""
    N = base_quat.shape[0] if base_quat.ndim > 1 else 1
    fwd_local = np.zeros((N, 3), dtype=np.float32)
    fwd_local[:, 0] = 1.0
    q = base_quat.reshape(N, 4)
    fwd_in_world = quat_rotate(q, fwd_local)
    return np.arctan2(-fwd_in_world[:, 2], np.sqrt(fwd_in_world[:, 0] ** 2 + fwd_in_world[:, 1] ** 2))


def compute_up_in_body(base_quat):
    """Compute world up vector in body frame. Returns (N, 3)."""
    N = base_quat.shape[0] if base_quat.ndim > 1 else 1
    up_world = np.zeros((N, 3), dtype=np.float32)
    up_world[:, 2] = 1.0
    return quat_rotate_inv(base_quat.reshape(N, 4), up_world)


# ============================================================
# Spline Interpolation
# ============================================================


def make_time_grids(h_sample, h_node, ctrl_dt):
    step_us = np.linspace(0, ctrl_dt * h_sample, h_sample + 1)
    step_nodes = np.linspace(0, ctrl_dt * h_sample, h_node + 1)
    return step_us, step_nodes


def _build_interp_matrix(src_t, dst_t):
    """Precompute quadratic Lagrange interpolation weights: dst = W @ src."""
    n_src = len(src_t)
    n_dst = len(dst_t)
    W = np.zeros((n_dst, n_src), dtype=np.float64)
    for i, t in enumerate(dst_t):
        idx = np.searchsorted(src_t, t, side="right") - 1
        idx = np.clip(idx, 0, n_src - 2)
        if idx == 0:
            j0, j1, j2 = 0, 1, 2
        elif idx >= n_src - 2:
            j0, j1, j2 = n_src - 3, n_src - 2, n_src - 1
        else:
            j0, j1, j2 = idx - 1, idx, idx + 1
        t0, t1, t2 = src_t[j0], src_t[j1], src_t[j2]
        if t0 != t1 and t0 != t2 and t1 != t2:
            W[i, j0] = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
            W[i, j1] = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
            W[i, j2] = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
        else:
            j = np.clip(idx, 0, n_src - 2)
            dt = src_t[j + 1] - src_t[j]
            alpha = (t - src_t[j]) / dt if dt != 0 else 0.0
            W[i, j] = 1.0 - alpha
            W[i, j + 1] = alpha
    return W.astype(np.float32)


_interp_cache = {}


def nodes_to_actions(Y, step_nodes, step_us):
    """Interpolate control nodes -> full horizon actions. Y: (h_node+1, nu)."""
    key = (tuple(step_nodes), tuple(step_us))
    if key not in _interp_cache:
        _interp_cache[key] = _build_interp_matrix(step_nodes, step_us)
    return _interp_cache[key] @ Y


def actions_to_nodes(u, step_nodes, step_us):
    """Fit control nodes from actions. u: (h_sample+1, nu)."""
    key = (tuple(step_us), tuple(step_nodes))
    if key not in _interp_cache:
        _interp_cache[key] = _build_interp_matrix(step_us, step_nodes)
    return _interp_cache[key] @ u


def batch_nodes_to_actions(Y_batch, step_nodes, step_us):
    """Vectorized: (N, h_node+1, nu) -> (N, h_sample+1, nu). Uses matrix multiply."""
    key = (tuple(step_nodes), tuple(step_us))
    if key not in _interp_cache:
        _interp_cache[key] = _build_interp_matrix(step_nodes, step_us)
    W = _interp_cache[key]
    return np.einsum("sn,bnd->bsd", W, Y_batch)


# ============================================================
# Rollout Simulation (N parallel worlds for MPPI)
# ============================================================


class RolloutSim:
    """N parallel robot simulations for MPPI rollouts.

    Args:
        robot_builder: A :class:`newton.ModelBuilder` with a single robot (no ground plane).
        n_worlds: Number of parallel worlds (n_samples + 1).
        ctrl_dt: Control timestep [s].
        sim_substeps: Physics substeps per control step.
        device: Warp device.
        scene_setup_fn: Optional callable(scene_builder, robot_shape_cfg) to add
            extra geometry (boxes, ramps) to each world before finalization.
    """

    def __init__(self, robot_builder, n_worlds, ctrl_dt=0.02, sim_substeps=4,
                 device=None, scene_setup_fn=None):
        self.device = device or wp.get_device()
        self.n_worlds = n_worlds
        self.ctrl_dt = ctrl_dt
        self.sim_substeps = sim_substeps
        self.sim_dt = ctrl_dt / sim_substeps

        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(robot_builder, world_count=n_worlds)
        if scene_setup_fn is not None:
            scene_setup_fn(scene, robot_builder.default_shape_cfg)
        else:
            scene.add_ground_plane(cfg=robot_builder.default_shape_cfg)

        self.model = scene.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=True, solver="newton", ls_iterations=10,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.coords_per_world = self.model.joint_coord_count // n_worlds
        self.dofs_per_world = self.model.joint_dof_count // n_worlds
        self.bodies_per_world = self.model.body_count // n_worlds

        self._q_buf = wp.zeros(self.model.joint_coord_count, dtype=wp.float32, device=self.device)
        self._qd_buf = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.device)
        self._target_buf = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.device)

        self._graph = None
        if self.device.is_cuda:
            self._capture_graph()

    def _capture_graph(self):
        try:
            with wp.ScopedCapture() as capture:
                self._simulate_substeps()
            self._graph = capture.graph
            print(f"  CUDA graph captured ({self.sim_substeps} substeps, {self.n_worlds} worlds)")
        except Exception as e:
            print(f"  CUDA graph capture failed, using eager mode: {e}")
            self._graph = None

    def _simulate_substeps(self):
        need_copy = self.sim_substeps % 2 == 1
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if need_copy and i == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def reset_all(self, joint_q, joint_qd):
        """Reset all worlds to the same state. joint_q: (cpw,), joint_qd: (dpw,)."""
        q = np.tile(joint_q, self.n_worlds).astype(np.float32)
        qd = np.tile(joint_qd, self.n_worlds).astype(np.float32)
        wp.copy(self._q_buf, wp.array(q, dtype=wp.float32, device=self.device))
        wp.copy(self.state_0.joint_q, self._q_buf)
        wp.copy(self._qd_buf, wp.array(qd, dtype=wp.float32, device=self.device))
        wp.copy(self.state_0.joint_qd, self._qd_buf)
        self.state_0.clear_forces()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def set_targets(self, targets, n_actuated_dofs, free_joint_dofs=6):
        """Set joint targets for all worlds. targets: (n_worlds, n_actuated_dofs)."""
        full = np.zeros((self.n_worlds, self.dofs_per_world), dtype=np.float32)
        full[:, free_joint_dofs:free_joint_dofs + n_actuated_dofs] = targets
        wp.copy(self._target_buf, wp.array(full.reshape(-1), dtype=wp.float32, device=self.device))
        wp.copy(self.control.joint_target_pos, self._target_buf)

    def step_control(self):
        """Step all worlds by one control step (uses CUDA graph if available)."""
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate_substeps()

    def get_base_states(self):
        """Returns (base_pos, base_quat, base_vel, base_angvel) each (N, dim)."""
        bq = self.state_0.body_q.numpy().reshape(self.n_worlds, self.bodies_per_world, 7)
        bqd = self.state_0.body_qd.numpy().reshape(self.n_worlds, self.bodies_per_world, 6)
        return bq[:, 0, :3].copy(), bq[:, 0, 3:7].copy(), bqd[:, 0, :3].copy(), bqd[:, 0, 3:6].copy()

    def get_body_states(self, body_idx):
        """Returns (pos, quat, vel, angvel) for a specific body index, each (N, dim)."""
        bq = self.state_0.body_q.numpy().reshape(self.n_worlds, self.bodies_per_world, 7)
        bqd = self.state_0.body_qd.numpy().reshape(self.n_worlds, self.bodies_per_world, 6)
        return (bq[:, body_idx, :3].copy(), bq[:, body_idx, 3:7].copy(),
                bqd[:, body_idx, :3].copy(), bqd[:, body_idx, 3:6].copy())


# ============================================================
# DIAL-MPC Controller
# ============================================================


@dataclass
class MPPIConfig:
    """MPPI / DIAL-MPC configuration (robot-agnostic)."""
    n_samples: int = 2048
    h_sample: int = 12
    h_node: int = 4
    n_diffuse: int = 4
    n_diffuse_init: int = 10
    temp_sample: float = 0.05
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    sigma_scale: float = 1.0
    ctrl_dt: float = 0.02
    sim_substeps: int = 4
    n_steps: int = 200
    ramp_up_time: float = 1.0
    output: str = ""


class DIALMPCController:
    """DIAL-MPC: MPPI with diffusion-style annealing.

    Args:
        config: MPPI configuration.
        rollout_sim: Parallel rollout simulator.
        n_actuated_dofs: Number of actuated joints.
        joint_range_low: Lower joint limits for actuated DOFs.
        joint_range_high: Upper joint limits for actuated DOFs.
        reward_fn: Callable(rollout_sim, config, t, horizon_step, actions) -> rewards (N,).
    """

    def __init__(self, config, rollout_sim, n_actuated_dofs, joint_range_low, joint_range_high,
                 reward_fn):
        self.config = config
        self.rollout_sim = rollout_sim
        self.n_dofs = n_actuated_dofs
        self.joint_range_low = joint_range_low
        self.joint_range_high = joint_range_high
        self.reward_fn = reward_fn
        self.Y = np.zeros((config.h_node + 1, n_actuated_dofs), dtype=np.float32)
        self.step_us, self.step_nodes = make_time_grids(config.h_sample, config.h_node, config.ctrl_dt)
        idx = np.arange(config.h_node + 1)[::-1]
        self.sigma_control = config.horizon_diffuse_factor ** idx * config.sigma_scale
        self.step_count = 0

    def act_to_joint(self, act):
        """Map normalized action [-1,1] -> joint angles."""
        norm = (act + 1.0) / 2.0
        return self.joint_range_low + norm * (self.joint_range_high - self.joint_range_low)

    def _noise_schedule(self, n_diffuse):
        traj = self.config.traj_diffuse_factor ** np.arange(n_diffuse)
        return traj[:, None] * self.sigma_control[None, :]

    def reverse_once(self, jq, jqd, Y, noise_scale, t):
        """One MPPI denoising step."""
        cfg = self.config
        N = cfg.n_samples

        eps = np.random.randn(N, cfg.h_node + 1, self.n_dofs).astype(np.float32)
        Y_samples = eps * noise_scale[None, :, None] + Y[None, :, :]
        Y_samples[:, 0, :] = Y[0, :]
        Y_all = np.concatenate([Y_samples, Y[None, :, :]], axis=0)
        Y_all = np.clip(Y_all, -1.0, 1.0)

        u_all = batch_nodes_to_actions(Y_all, self.step_nodes, self.step_us)

        n_total = u_all.shape[0]
        batch_size = self.rollout_sim.n_worlds
        all_rewards = np.zeros(n_total, dtype=np.float32)

        for bs in range(0, n_total, batch_size):
            be = min(bs + batch_size, n_total)
            actual = be - bs
            self.rollout_sim.reset_all(jq, jqd)
            step_rew = np.zeros((actual, cfg.h_sample), dtype=np.float32)

            for h in range(cfg.h_sample):
                actions_h = u_all[bs:be, h, :]
                jt = self.act_to_joint(actions_h)
                if actual < batch_size:
                    jt_full = np.zeros((batch_size, self.n_dofs), dtype=np.float32)
                    jt_full[:actual] = jt
                else:
                    jt_full = jt

                self.rollout_sim.set_targets(jt_full, self.n_dofs)
                self.rollout_sim.step_control()

                rew = self.reward_fn(
                    self.rollout_sim, cfg, t + (h + 1) * cfg.ctrl_dt,
                    actions=actions_h[:actual], n_actual=actual,
                )
                step_rew[:, h] = rew

            all_rewards[bs:be] = step_rew.mean(axis=1)

        baseline = all_rewards[-1]
        std = all_rewards.std() + 1e-8
        logits = (all_rewards - baseline) / std / cfg.temp_sample
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        return np.einsum("n,nij->ij", weights, Y_all), float(all_rewards[-1])

    def shift(self, Y):
        u = nodes_to_actions(Y, self.step_nodes, self.step_us)
        u = np.roll(u, -1, axis=0)
        u[-1] = 0.0
        return actions_to_nodes(u, self.step_nodes, self.step_us)

    def plan(self, jq, jqd, t):
        """Plan next action. Returns (action, reward)."""
        n_diff = self.config.n_diffuse_init if self.step_count == 0 else self.config.n_diffuse
        schedule = self._noise_schedule(n_diff)
        rew = -np.inf
        for i in range(n_diff):
            self.Y, rew = self.reverse_once(jq, jqd, self.Y, schedule[i], t)
        action = self.Y[0].copy()
        self.Y = self.shift(self.Y)
        self.step_count += 1
        return action, rew


# ============================================================
# Video Renderer
# ============================================================


def render_video_from_trajectory(trajectory, output_path, build_visual_fn, fps=50):
    """Replay a recorded trajectory and encode to MP4.

    Args:
        trajectory: Dict with "body_q", "time" lists.
        output_path: Output MP4 file path.
        build_visual_fn: Callable() -> (builder, shape_cfg) for visual model.
        fps: Video frame rate.
    """
    body_qs = trajectory["body_q"]
    times = trajectory["time"]
    n_frames = len(body_qs)
    if n_frames == 0:
        print("No frames to render.")
        return

    print(f"\nRendering {n_frames} frames to {output_path}...")

    builder, shape_cfg = build_visual_fn()
    scene = newton.ModelBuilder()
    scene.replicate(builder, world_count=1)
    scene.add_ground_plane(cfg=shape_cfg)
    model = scene.finalize()

    viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=True)
    viewer.set_model(model)

    state = model.state()
    w = viewer.renderer._screen_width
    h = viewer.renderer._screen_height
    frame_buf = wp.zeros((h, w, 3), dtype=wp.uint8, device=wp.get_device())

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20", output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(n_frames):
        bq = body_qs[i]
        wp.copy(state.body_q, wp.array(bq, dtype=wp.transform, device=wp.get_device()))

        base_pos = bq[0, :3]
        cam = wp.vec3(float(base_pos[0]), float(base_pos[1]) + 2.5, 0.6)
        viewer.set_camera(pos=cam, pitch=-5.0, yaw=-90.0)

        viewer.begin_frame(times[i])
        viewer.log_state(state)
        viewer.end_frame()

        frame = viewer.get_frame(target_image=frame_buf)
        proc.stdin.write(frame.numpy().tobytes())

        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames}")

    proc.stdin.close()
    proc.wait()
    viewer.close()

    if proc.returncode == 0:
        sz = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Video saved: {output_path} ({sz:.1f} MB)")
    else:
        stderr = proc.stderr.read().decode()
        print(f"  ffmpeg error: {stderr[-500:]}")


# ============================================================
# Common argument parser additions for MPC examples
# ============================================================


def add_mpc_args(parser):
    """Add standard DIAL-MPC arguments to an argument parser."""
    parser.add_argument("--n-samples", type=int, default=2048, help="MPPI samples")
    parser.add_argument("--n-steps", type=int, default=200, help="MPC steps")
    parser.add_argument("--h-sample", type=int, default=12, help="Horizon length")
    parser.add_argument("--h-node", type=int, default=4, help="Spline nodes")
    parser.add_argument("--n-diffuse", type=int, default=4, help="Diffusion steps")
    parser.add_argument("--n-diffuse-init", type=int, default=10, help="Initial diffusion steps")
    parser.add_argument("--output", type=str, default="", help="Output video file (MP4)")
    return parser


def mppi_config_from_args(args, **overrides):
    """Create MPPIConfig from parsed arguments."""
    kwargs = dict(
        n_samples=getattr(args, "n_samples", 2048),
        n_steps=getattr(args, "n_steps", 200),
        h_sample=getattr(args, "h_sample", 12),
        h_node=getattr(args, "h_node", 4),
        n_diffuse=getattr(args, "n_diffuse", 4),
        n_diffuse_init=getattr(args, "n_diffuse_init", 10),
        output=getattr(args, "output", ""),
    )
    kwargs.update(overrides)
    return MPPIConfig(**kwargs)
