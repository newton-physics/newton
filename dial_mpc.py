"""
DIAL-MPC: Diffusion-Inspired Annealing for Legged MPC
======================================================

Reimplementation using NVIDIA Newton physics engine with full Go2 visual model.

Reference: "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control
via Diffusion-Style Annealing" (Yin et al., 2024, ICRA 2025)

Usage:
    uv run --extra examples python dial_mpc.py
    uv run --extra examples python dial_mpc.py --test
    uv run --extra examples python dial_mpc.py --n-steps 200 --n-samples 128
    uv run --extra examples python dial_mpc.py --headless --output dial_mpc_go2.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import yaml

import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode

# ============================================================
# Go2 Robot Configuration (from newton-assets YAML + DIAL-MPC paper)
# ============================================================

N_LEGS = 4
N_JOINTS_PER_LEG = 3
N_JOINTS = N_LEGS * N_JOINTS_PER_LEG  # 12
COORDS_PER_WORLD = 7 + N_JOINTS  # 19 (free joint 7 + revolute 12)
DOFS_PER_WORLD = 6 + N_JOINTS  # 18 (free joint 6 + revolute 12)


@dataclass
class Config:
    """DIAL-MPC + simulation configuration."""

    # MPPI parameters
    n_samples: int = 128
    h_sample: int = 12  # horizon in control steps
    h_node: int = 4  # spline control nodes
    n_diffuse: int = 3  # diffusion steps per MPC step
    n_diffuse_init: int = 8  # diffusion steps for first step
    temp_sample: float = 0.05  # MPPI temperature
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    sigma_scale: float = 1.0

    # Simulation
    ctrl_dt: float = 0.02  # 50 Hz control
    sim_substeps: int = 4
    n_steps: int = 200  # total MPC steps

    # Reward weights (balanced for forward velocity + stability)
    w_vel: float = 2.0          # primary objective: move forward
    w_height: float = 2.0       # maintain height
    w_upright: float = 2.0      # stay upright
    w_pitch: float = 3.0        # penalize forward lean specifically
    w_yaw: float = 0.3
    w_ang_vel: float = 0.3
    w_body_rate: float = 0.3    # smooth body rotation
    w_action_reg: float = 0.01  # light action regularization

    # Target velocities
    target_vx: float = 0.6  # moderate speed for stable gait
    target_vy: float = 0.0
    target_vyaw: float = 0.0

    # Gait: trot (diagonal pairs)
    gait_phases: list = field(default_factory=lambda: [0.0, 0.5, 0.5, 0.0])
    gait_duty_ratio: float = 0.45
    gait_cadence: float = 2.0
    gait_amplitude: float = 0.08

    # Velocity ramp
    ramp_up_time: float = 1.0

    # Video output
    output: str = ""

    @property
    def sim_dt(self) -> float:
        return self.ctrl_dt / self.sim_substeps


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


# ============================================================
# Spline Interpolation
# ============================================================


def make_time_grids(h_sample, h_node, ctrl_dt):
    step_us = np.linspace(0, ctrl_dt * h_sample, h_sample + 1)
    step_nodes = np.linspace(0, ctrl_dt * h_sample, h_node + 1)
    return step_us, step_nodes


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


def _build_interp_matrix(src_t, dst_t):
    """Precompute quadratic B-spline interpolation weights: dst = W @ src.

    Uses local quadratic (3-point) Lagrange interpolation matching scipy interp1d 'quadratic'.
    """
    n_src = len(src_t)
    n_dst = len(dst_t)
    W = np.zeros((n_dst, n_src), dtype=np.float64)
    for i, t in enumerate(dst_t):
        # Find the interval
        idx = np.searchsorted(src_t, t, side="right") - 1
        idx = np.clip(idx, 0, n_src - 2)
        # Pick 3-point stencil for quadratic interpolation
        if idx == 0:
            j0, j1, j2 = 0, 1, 2
        elif idx >= n_src - 2:
            j0, j1, j2 = n_src - 3, n_src - 2, n_src - 1
        else:
            j0, j1, j2 = idx - 1, idx, idx + 1
        t0, t1, t2 = src_t[j0], src_t[j1], src_t[j2]
        # Lagrange basis
        if t0 != t1 and t0 != t2 and t1 != t2:
            W[i, j0] = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
            W[i, j1] = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
            W[i, j2] = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
        else:
            # Degenerate case: linear fallback
            j = np.clip(idx, 0, n_src - 2)
            dt = src_t[j + 1] - src_t[j]
            alpha = (t - src_t[j]) / dt if dt != 0 else 0.0
            W[i, j] = 1.0 - alpha
            W[i, j + 1] = alpha
    return W.astype(np.float32)


# Cached interpolation matrix (populated on first use)
_interp_cache = {}


def batch_nodes_to_actions(Y_batch, step_nodes, step_us):
    """Vectorized: (N, h_node+1, nu) -> (N, h_sample+1, nu). Uses matrix multiply."""
    key = (tuple(step_nodes), tuple(step_us))
    if key not in _interp_cache:
        _interp_cache[key] = _build_interp_matrix(step_nodes, step_us)
    W = _interp_cache[key]
    # Y_batch: (N, h_node+1, nu), W: (h_sample+1, h_node+1)
    # result: (N, h_sample+1, nu)
    return np.einsum("sn,bnd->bsd", W, Y_batch)


# ============================================================
# Reward Function
# ============================================================


def compute_reward_batch(base_pos, base_quat, base_vel, base_angvel,
                         config, t, target_height, actions=None):
    """Compute reward for a batch of states. All inputs: (N, dim)."""
    N = base_pos.shape[0]
    ramp = min(1.0, t / config.ramp_up_time) if config.ramp_up_time > 0 else 1.0
    target_vx = config.target_vx * ramp

    # World-frame velocity for tracking (prevents "leaning forward = moving forward" exploit)
    ab = quat_rotate_inv(base_quat, base_angvel)

    reward_vel = -((base_vel[:, 0] - target_vx) ** 2 + (base_vel[:, 1] - config.target_vy) ** 2)
    reward_ang_vel = -((ab[:, 2] - config.target_vyaw) ** 2)
    # Asymmetric height penalty: harsh below target, mild above
    h_err = base_pos[:, 2] - target_height
    reward_height = np.where(h_err < 0, -10.0 * h_err ** 2, -h_err ** 2)

    up_world = np.zeros((N, 3), dtype=np.float32)
    up_world[:, 2] = 1.0
    up_body = quat_rotate_inv(base_quat, up_world)
    reward_upright = -np.sum((up_body - up_world) ** 2, axis=1)

    yaw = quat_to_yaw(base_quat)
    reward_yaw = -(np.arctan2(np.sin(yaw), np.cos(yaw)) ** 2)

    # Penalize forward pitch specifically (positive pitch = leaning forward)
    fwd_world = np.zeros((N, 3), dtype=np.float32)
    fwd_world[:, 0] = 1.0
    fwd_body = quat_rotate(base_quat, fwd_world)  # body forward in world frame
    pitch = np.arctan2(-fwd_body[:, 2], np.sqrt(fwd_body[:, 0]**2 + fwd_body[:, 1]**2))
    reward_pitch = -(pitch ** 2)

    # Penalize body pitch/roll angular velocity (smoothness)
    reward_body_rate = -(ab[:, 0] ** 2 + ab[:, 1] ** 2)

    # Action regularization (penalize deviation from home = action 0)
    reward_action = np.zeros(N, dtype=np.float32)
    if actions is not None:
        reward_action = -np.mean(actions ** 2, axis=1)

    reward = (config.w_vel * reward_vel + config.w_height * reward_height +
              config.w_upright * reward_upright + config.w_pitch * reward_pitch +
              config.w_yaw * reward_yaw + config.w_ang_vel * reward_ang_vel +
              config.w_body_rate * reward_body_rate + config.w_action_reg * reward_action)

    # Termination
    terminated = (base_pos[:, 2] < 0.08) | (up_body[:, 2] < -0.3)
    return np.where(terminated, -100.0, reward)


# ============================================================
# Model Builder
# ============================================================


def _load_go2_config():
    """Load Go2 asset paths and YAML configuration."""
    asset_path = newton.utils.download_asset("unitree_go2")
    yaml_path = str(asset_path / "rl_policies" / "go2.yaml")
    usd_path = str(asset_path / "usd" / "go2.usda")
    with open(yaml_path, encoding="utf-8") as f:
        go2_config = yaml.safe_load(f)
    return usd_path, go2_config


def _apply_go2_config(builder, go2_config):
    """Apply joint positions, PD gains, and armature from YAML config."""
    builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
    builder.joint_q[7:] = go2_config["mjw_joint_pos"]
    for i in range(go2_config["num_dofs"]):
        # Boost stiffness/damping above YAML defaults for stable MPC walking
        builder.joint_target_ke[i + 6] = 100.0  # stiffer than YAML's 50
        builder.joint_target_kd[i + 6] = 2.0    # more damping than YAML's 1
        builder.joint_armature[i + 6] = go2_config["mjw_joint_armature"][i]
        builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)
        builder.joint_target_pos[6 + i] = go2_config["mjw_joint_pos"][i]


def build_go2_visual(device=None):
    """Build Go2 ModelBuilder with full visual meshes (for rendering)."""
    usd_path, go2_config = _load_go2_config()

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
        hide_collision_shapes=True,
    )
    builder.approximate_meshes("convex_hull")
    _apply_go2_config(builder, go2_config)
    return builder, go2_config


def build_go2_physics(device=None):
    """Build Go2 ModelBuilder for physics only (no visual meshes, for MPPI rollouts)."""
    usd_path, go2_config = _load_go2_config()

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
        hide_collision_shapes=False,
        load_visual_shapes=False,  # no visual meshes for rollouts
    )
    builder.approximate_meshes("convex_hull")
    _apply_go2_config(builder, go2_config)
    return builder, go2_config


# ============================================================
# Rollout Simulation (N parallel worlds for MPPI)
# ============================================================


class RolloutSim:
    """N parallel Go2 simulations for MPPI rollouts (physics-only, no visual meshes)."""

    def __init__(self, n_worlds, config, device=None):
        self.device = device or wp.get_device()
        self.n_worlds = n_worlds
        self.config = config

        robot_builder, self.go2_config = build_go2_physics(self.device)

        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(robot_builder, world_count=n_worlds)
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

        # Determine per-world array sizes
        self.coords_per_world = self.model.joint_coord_count // n_worlds
        self.dofs_per_world = self.model.joint_dof_count // n_worlds
        self.bodies_per_world = self.model.body_count // n_worlds

        # Pre-allocate GPU buffers for state reset (avoid per-call allocations)
        self._q_buf = wp.zeros(self.model.joint_coord_count, dtype=wp.float32, device=self.device)
        self._qd_buf = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.device)
        self._target_buf = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.device)

        # CUDA graph capture for physics substeps
        self._graph = None
        if self.device.is_cuda:
            self._capture_graph()

    def _capture_graph(self):
        """Capture physics substeps as a CUDA graph for fast replay."""
        try:
            with wp.ScopedCapture() as capture:
                self._simulate_substeps()
            self._graph = capture.graph
            print(f"  CUDA graph captured ({self.config.sim_substeps} substeps, {self.n_worlds} worlds)")
        except Exception as e:
            print(f"  CUDA graph capture failed, using eager mode: {e}")
            self._graph = None

    def _simulate_substeps(self):
        """Run physics substeps (used for graph capture and eager mode)."""
        need_copy = self.config.sim_substeps % 2 == 1
        for i in range(self.config.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.config.sim_dt)
            if need_copy and i == self.config.sim_substeps - 1:
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

    def set_targets(self, targets):
        """Set joint targets for all worlds. targets: (n_worlds, 12)."""
        full = np.zeros((self.n_worlds, self.dofs_per_world), dtype=np.float32)
        full[:, 6:] = targets
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


# ============================================================
# DIAL-MPC Controller
# ============================================================


class DIALMPCController:
    """DIAL-MPC: MPPI with diffusion-style annealing."""

    def __init__(self, config, rollout_sim, joint_range_low, joint_range_high):
        self.config = config
        self.rollout_sim = rollout_sim
        self.joint_range_low = joint_range_low
        self.joint_range_high = joint_range_high
        self.Y = np.zeros((config.h_node + 1, N_JOINTS), dtype=np.float32)
        self.step_us, self.step_nodes = make_time_grids(config.h_sample, config.h_node, config.ctrl_dt)
        idx = np.arange(config.h_node + 1)[::-1]
        self.sigma_control = config.horizon_diffuse_factor ** idx * config.sigma_scale
        self.step_count = 0
        self.target_height = 0.29  # will be calibrated

    def act_to_joint(self, act):
        """Map normalized action [-1,1] -> joint angles. act: (..., 12)."""
        norm = (act + 1.0) / 2.0
        return self.joint_range_low + norm * (self.joint_range_high - self.joint_range_low)

    def _noise_schedule(self, n_diffuse):
        traj = self.config.traj_diffuse_factor ** np.arange(n_diffuse)
        return traj[:, None] * self.sigma_control[None, :]

    def reverse_once(self, jq, jqd, Y, noise_scale, t):
        """One MPPI denoising step."""
        cfg = self.config
        N = cfg.n_samples

        eps = np.random.randn(N, cfg.h_node + 1, N_JOINTS).astype(np.float32)
        Y_samples = eps * noise_scale[None, :, None] + Y[None, :, :]
        Y_samples[:, 0, :] = Y[0, :]
        Y_all = np.concatenate([Y_samples, Y[None, :, :]], axis=0)
        Y_all = np.clip(Y_all, -1.0, 1.0)

        u_all = batch_nodes_to_actions(Y_all, self.step_nodes, self.step_us)

        # Rollout
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
                    jt_full = np.zeros((batch_size, N_JOINTS), dtype=np.float32)
                    jt_full[:actual] = jt
                else:
                    jt_full = jt

                self.rollout_sim.set_targets(jt_full)
                self.rollout_sim.step_control()

                bp, bq, bv, ba = self.rollout_sim.get_base_states()
                rew = compute_reward_batch(
                    bp[:actual], bq[:actual], bv[:actual], ba[:actual],
                    cfg, t + (h + 1) * cfg.ctrl_dt, self.target_height,
                    actions=actions_h[:actual],
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
        """Plan next action. Returns (action (12,), reward)."""
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
# Newton Example Class
# ============================================================


class Example:
    """DIAL-MPC Go2 walking example using Newton viewer."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and getattr(args, "test", False)

        # Parse MPC config from args
        self.mpc_config = Config(
            n_samples=getattr(args, "n_samples", 128),
            n_steps=getattr(args, "n_steps", 200),
            h_sample=getattr(args, "h_sample", 12),
            h_node=getattr(args, "h_node", 4),
            n_diffuse=getattr(args, "n_diffuse", 3),
            n_diffuse_init=getattr(args, "n_diffuse_init", 8),
            target_vx=getattr(args, "target_vx", 0.8),
            output=getattr(args, "output", ""),
        )

        # Build real simulation model with full visual meshes for rendering
        robot_builder, self.go2_config = build_go2_visual(self.device)

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
        self.follow_cam = True

        # Tight action range (±0.2 rad) for stable walking
        home_pos = np.array(self.go2_config["mjw_joint_pos"], dtype=np.float32)
        self.joint_range_low = home_pos - 0.2
        self.joint_range_high = home_pos + 0.2

        # Seed for reproducibility
        np.random.seed(42)

        # Settle robot
        self.sim_time = 0.0
        self.sim_step = 0
        self._settle()

        # Calibrate target height from settled state
        jq = self.state_0.joint_q.numpy()
        settled_z = jq[2]
        print(f"  Settled height: {settled_z:.4f}m")

        # Build rollout simulation for MPPI
        n_rollout = self.mpc_config.n_samples + 1
        self.rollout_sim = RolloutSim(n_rollout, self.mpc_config, self.device)

        # Create controller
        self.controller = DIALMPCController(
            self.mpc_config, self.rollout_sim,
            self.joint_range_low, self.joint_range_high,
        )
        self.controller.target_height = float(settled_z)

        # Trajectory recording
        self.trajectory = {
            "body_q": [], "body_qd": [], "time": [], "reward": [],
            "velocity": [], "pitch": [], "ang_vel_body": [], "action": [],
        }
        self.frame_dt = self.mpc_config.ctrl_dt

        # Video recording state (deferred to post-processing to avoid GL/CUDA conflicts)
        self._ffmpeg_proc = None
        self._frame_buf = None
        self._video_active = False

    def _settle(self):
        """Let the robot settle at home pose."""
        home = np.array(self.go2_config["mjw_joint_pos"], dtype=np.float32)
        full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        full[6:] = home
        wp.copy(self.control.joint_target_pos,
                wp.array(full, dtype=wp.float32, device=self.device))

        for _ in range(100):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None,
                             self.mpc_config.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _start_video_encoder(self, filename, fps=50, width=1280, height=720):
        """Start ffmpeg subprocess for video encoding."""
        w = self.viewer.renderer._screen_width
        h = self.viewer.renderer._screen_height
        self._frame_buf = wp.zeros((h, w, 3), dtype=wp.uint8, device=self.device)
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "20", filename,
        ]
        self._ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._video_filename = filename
        print(f"  Recording video to {filename}")

    def step(self):
        if self.sim_step >= self.mpc_config.n_steps:
            return

        jq = self.state_0.joint_q.numpy().copy()
        jqd = self.state_0.joint_qd.numpy().copy()

        # Plan
        action, reward = self.controller.plan(jq, jqd, self.sim_time)

        # Apply to real sim
        joint_targets = self.controller.act_to_joint(action)
        full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        full[6:] = joint_targets
        wp.copy(self.control.joint_target_pos,
                wp.array(full, dtype=wp.float32, device=self.device))

        for _ in range(self.mpc_config.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None,
                             self.mpc_config.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Record
        bq = self.state_0.body_q.numpy().copy()
        bqd = self.state_0.body_qd.numpy().copy()
        base_vel_world = bqd[0, :3]  # world-frame linear velocity
        base_quat = bq[0, 3:7]
        base_angvel = bqd[0, 3:6]
        ab = quat_rotate_inv(base_quat.reshape(1, 4), base_angvel.reshape(1, 3))[0]

        # Compute pitch: angle of forward axis relative to horizontal
        fwd_local = np.array([1, 0, 0], dtype=np.float32)
        fwd_in_world = quat_rotate(base_quat.reshape(1, 4), fwd_local.reshape(1, 3))[0]
        pitch = np.arctan2(-fwd_in_world[2], np.sqrt(fwd_in_world[0]**2 + fwd_in_world[1]**2))

        self.trajectory["body_q"].append(bq)
        self.trajectory["body_qd"].append(bqd)
        self.trajectory["time"].append(self.sim_time)
        self.trajectory["reward"].append(reward)
        self.trajectory["velocity"].append(base_vel_world[0])  # world-frame forward vel
        self.trajectory["pitch"].append(float(pitch))
        self.trajectory["ang_vel_body"].append(ab.copy())
        self.trajectory["action"].append(action.copy())

        if self.sim_step % 20 == 0 or self.sim_step == self.mpc_config.n_steps - 1:
            print(
                f"  Step {self.sim_step:4d}/{self.mpc_config.n_steps}  "
                f"t={self.sim_time:.2f}s  vx={base_vel_world[0]:+.3f}m/s  "
                f"z={bq[0, 2]:.3f}m  rew={reward:+.3f}"
            )

        self.sim_time += self.frame_dt
        self.sim_step += 1

    def render(self):
        if self.follow_cam:
            bq = self.state_0.body_q.numpy()
            base_pos = bq[0, :3]
            # Side-on view: camera looks down -Y axis, tracking robot X
            cam = wp.vec3(float(base_pos[0]), float(base_pos[1]) + 2.0,
                          0.4)
            self.viewer.set_camera(pos=cam, pitch=-5.0, yaw=-90.0)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

        # Write frame to video encoder if active (live GL viewer mode)
        if self._video_active and self._ffmpeg_proc is not None:
            try:
                frame = self.viewer.get_frame(target_image=self._frame_buf)
                self._ffmpeg_proc.stdin.write(frame.numpy().tobytes())
            except Exception:
                pass

    def _finalize_video(self):
        """Close ffmpeg and report."""
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.stdin.close()
            self._ffmpeg_proc.wait()
            if self._ffmpeg_proc.returncode == 0:
                sz = os.path.getsize(self._video_filename) / 1024 / 1024
                print(f"\n  Video saved: {self._video_filename} ({sz:.1f} MB)")
            self._ffmpeg_proc = None

    def test_final(self):
        """Comprehensive verification of MPC trajectory quality."""
        traj = self.trajectory
        n = len(traj["body_q"])
        if n == 0:
            raise ValueError("No trajectory data")

        heights = np.array([bq[0, 2] for bq in traj["body_q"]])
        velocities = np.array(traj["velocity"])
        rewards = np.array(traj["reward"])
        pitches = np.array(traj["pitch"])
        actions = np.array(traj["action"])
        ang_vels = np.array(traj["ang_vel_body"])

        ramp_steps = int(self.mpc_config.ramp_up_time / self.mpc_config.ctrl_dt)
        post_ramp = slice(min(ramp_steps + 20, n), n)

        print(f"\n{'='*60}")
        print("Trajectory Verification")
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

        # --- Stability ---
        check("height > 0.12m always",
              heights.min() > 0.12,
              f"min={heights.min():.3f}m")

        check("height > 0.20m always",
              heights.min() > 0.20,
              f"min={heights.min():.3f}m")

        check("height std < 0.06m (steady torso)",
              heights[post_ramp].std() < 0.06,
              f"std={heights[post_ramp].std():.4f}m")

        final_quat = traj["body_q"][-1][0, 3:7]
        up = np.array([0, 0, 1], dtype=np.float32)
        up_body = quat_rotate_inv(final_quat.reshape(1, 4), up.reshape(1, 3))[0]
        check("stays upright (up_z > 0.7)",
              up_body[2] > 0.7,
              f"up_z={up_body[2]:.3f}")

        # --- Pitch ---
        mean_pitch_deg = np.degrees(pitches[post_ramp].mean())
        max_pitch_deg = np.degrees(np.abs(pitches[post_ramp]).max())
        check("mean pitch < 15° (not leaning forward)",
              abs(mean_pitch_deg) < 15,
              f"mean={mean_pitch_deg:.1f}°")

        check("max pitch < 30°",
              max_pitch_deg < 30,
              f"max={max_pitch_deg:.1f}°")

        # --- Velocity ---
        if n > post_ramp.start:
            mean_vel = velocities[post_ramp].mean()
            check("mean forward vel > 0.15 m/s",
                  mean_vel > 0.15,
                  f"mean={mean_vel:.3f} m/s")

            check("mean forward vel > 0.3 m/s",
                  mean_vel > 0.3,
                  f"mean={mean_vel:.3f} m/s")

            # Velocity should not drop to zero for extended periods
            vel_window = 20
            if len(velocities) > vel_window:
                rolling_mean = np.convolve(velocities, np.ones(vel_window)/vel_window, 'valid')
                check("no sustained stall (rolling vel > -0.1)",
                      rolling_mean[post_ramp.start:].min() > -0.1,
                      f"min rolling vel={rolling_mean[post_ramp.start:].min():.3f}")

        dist = traj["body_q"][-1][0, 0] - traj["body_q"][0][0, 0]
        check("forward distance > 0.5m",
              dist > 0.5,
              f"dist={dist:.3f}m")

        # --- Action smoothness ---
        if len(actions) > 2:
            act_diffs = np.diff(actions, axis=0)
            check("actions smooth (max jump < 1.5)",
                  np.abs(act_diffs).max() < 1.5,
                  f"max={np.abs(act_diffs).max():.3f}")

            check("actions smooth (mean jump < 0.2)",
                  np.abs(act_diffs).mean() < 0.2,
                  f"mean={np.abs(act_diffs).mean():.3f}")

        # --- Angular velocity smoothness ---
        pitch_rate = np.abs(ang_vels[post_ramp, 1])  # body pitch rate
        roll_rate = np.abs(ang_vels[post_ramp, 0])   # body roll rate
        check("pitch rate < 5 rad/s (mean)",
              pitch_rate.mean() < 5.0,
              f"mean={pitch_rate.mean():.2f} rad/s")

        check("roll rate < 3 rad/s (mean)",
              roll_rate.mean() < 3.0,
              f"mean={roll_rate.mean():.2f} rad/s")

        # --- Optimization convergence ---
        if n > 40:
            early_rew = rewards[10:30].mean()
            late_rew = rewards[-30:].mean()
            check("reward improves or stays stable",
                  late_rew > early_rew - 2.0,
                  f"early={early_rew:.3f} late={late_rew:.3f}")

            check("no catastrophic reward (> -50 always)",
                  rewards.min() > -50,
                  f"min reward={rewards.min():.1f}")

        # --- Summary ---
        print(f"\n  Summary: {n_pass} passed, {n_fail} failed")
        print(f"  Height: {heights.mean():.3f}m ± {heights.std():.3f}m  (min={heights.min():.3f})")
        print(f"  Velocity: {velocities[post_ramp].mean():.3f} m/s ± {velocities[post_ramp].std():.3f}")
        print(f"  Pitch: {mean_pitch_deg:.1f}° ± {np.degrees(pitches[post_ramp].std()):.1f}°")
        print(f"  Distance: {dist:.2f}m")
        print(f"  Reward: {rewards.mean():.3f} ± {rewards.std():.3f}")
        print(f"{'='*60}")

        if n_fail > 0:
            raise ValueError(f"{n_fail} verification checks failed")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--n-samples", type=int, default=2048, help="MPPI samples")
        parser.add_argument("--n-steps", type=int, default=200, help="MPC steps")
        parser.add_argument("--h-sample", type=int, default=12, help="Horizon length")
        parser.add_argument("--h-node", type=int, default=4, help="Spline nodes")
        parser.add_argument("--n-diffuse", type=int, default=3, help="Diffusion steps")
        parser.add_argument("--n-diffuse-init", type=int, default=8, help="Initial diffusion steps")
        parser.add_argument("--target-vx", type=float, default=0.8, help="Target forward velocity")
        parser.add_argument("--output", type=str, default="", help="Output video file (MP4)")
        return parser


# ============================================================
# Post-processing Video Renderer
# ============================================================


def render_video_from_trajectory(trajectory, output_path, fps=50, width=1280, height=720):
    """Replay a recorded trajectory through ViewerGL headless and encode to MP4."""
    body_qs = trajectory["body_q"]
    times = trajectory["time"]
    n_frames = len(body_qs)
    if n_frames == 0:
        print("No frames to render.")
        return

    print(f"\nRendering {n_frames} frames to {output_path}...")

    # Build a fresh visual model (separate CUDA context from MuJoCo)
    builder, _ = build_go2_visual()
    scene = newton.ModelBuilder()
    scene.replicate(builder, world_count=1)
    scene.add_ground_plane(cfg=builder.default_shape_cfg)
    model = scene.finalize()

    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
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
        # Side-on view: camera looks down -Y axis, tracking robot X
        cam = wp.vec3(float(base_pos[0]), float(base_pos[1]) + 2.0, 0.4)
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
# Entry Point
# ============================================================


if __name__ == "__main__":
    parser = Example.create_parser()
    # Force null viewer for MPC (avoids GL/CUDA conflicts with MuJoCo)
    # Video is rendered as a post-processing step
    sys.argv = [a for a in sys.argv if a not in ("--headless",)]
    if "--viewer" not in " ".join(sys.argv):
        sys.argv.extend(["--viewer", "null"])

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)

    # Post-process: render video if requested
    output = getattr(args, "output", "")
    if output and example.trajectory["body_q"]:
        render_video_from_trajectory(example.trajectory, output)
