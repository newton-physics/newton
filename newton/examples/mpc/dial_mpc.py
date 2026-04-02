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
control using diffusion-style annealing (MPPI variant).  The hot loop
(noise, spline interpolation, action mapping, reward, MPPI weights) runs
end-to-end on GPU via Warp kernels.

Reference: "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control
via Diffusion-Style Annealing" (Yin et al., 2024, ICRA 2025)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import numpy as np

import warp as wp

import newton
import newton.examples


# ============================================================
# Warp Kernels
# ============================================================


@wp.kernel
def _noise_spline_kernel(
    rng_base_seed: int,
    rng_diffuse_offset: int,
    Y_mean: wp.array2d(dtype=wp.float32),
    noise_scale: wp.array(dtype=wp.float32),
    W: wp.array2d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
    n_samples: int,
    h_node_p1: int,
    h_sample_p1: int,
    n_dofs: int,
):
    """Fused GPU kernel: generate noise, add to mean, clip, spline-interpolate.

    Generates Gaussian noise on GPU via wp.randn (no CPU transfer needed).
    For sample == n_samples (last row), outputs the mean trajectory (no noise).
    """
    sample, step, dof = wp.tid()
    if step >= h_sample_p1 or dof >= n_dofs:
        return

    val = float(0.0)
    for node in range(h_node_p1):
        if sample < n_samples:
            # Generate noise on GPU
            rng = wp.rand_init(rng_base_seed, rng_diffuse_offset + sample * h_node_p1 * n_dofs + node * n_dofs + dof)
            eps = wp.randn(rng)
            y_node = eps * noise_scale[node] + Y_mean[node, dof]
            if node == 0:
                y_node = Y_mean[0, dof]
            y_node = wp.clamp(y_node, -1.0, 1.0)
        else:
            y_node = Y_mean[node, dof]
        val = val + W[step, node] * y_node

    out[sample, step, dof] = val


@wp.kernel
def _noise_nodes_kernel(
    rng_base_seed: int,
    rng_diffuse_offset: int,
    Y_mean: wp.array2d(dtype=wp.float32),
    noise_scale: wp.array(dtype=wp.float32),
    Y_all: wp.array3d(dtype=wp.float32),
    n_samples: int,
    h_node_p1: int,
    n_dofs: int,
):
    """Generate noisy Y_all nodes on GPU (for CPU-side weighted average later)."""
    sample, node, dof = wp.tid()
    if node >= h_node_p1 or dof >= n_dofs:
        return

    if sample < n_samples:
        rng = wp.rand_init(rng_base_seed, rng_diffuse_offset + sample * h_node_p1 * n_dofs + node * n_dofs + dof)
        eps = wp.randn(rng)
        y = eps * noise_scale[node] + Y_mean[node, dof]
        if node == 0:
            y = Y_mean[0, dof]
        Y_all[sample, node, dof] = wp.clamp(y, -1.0, 1.0)
    else:
        Y_all[sample, node, dof] = Y_mean[node, dof]


@wp.kernel
def _scatter_targets_kernel(
    actions: wp.array2d(dtype=wp.float32),
    joint_range_low: wp.array(dtype=wp.float32),
    joint_range_high: wp.array(dtype=wp.float32),
    target_pos: wp.array(dtype=wp.float32),
    n_worlds: int,
    dofs_per_world: int,
    n_actuated: int,
    free_dofs: int,
    h_step: int,
):
    """Fused: act_to_joint + scatter into flat DOF target array."""
    world, dof = wp.tid()
    if world < n_worlds and dof < n_actuated:
        a = actions[world, dof]
        norm = (a + 1.0) / 2.0
        jt = joint_range_low[dof] + norm * (joint_range_high[dof] - joint_range_low[dof])
        target_pos[world * dofs_per_world + free_dofs + dof] = jt


@wp.kernel
def _mppi_rewards_reduce(
    step_rewards: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    h_sample: int,
):
    """Mean reward across horizon steps for each sample."""
    i = wp.tid()
    s = float(0.0)
    for h in range(h_sample):
        s = s + step_rewards[i, h]
    out[i] = s / float(h_sample)


# ============================================================
# Quaternion Utilities (NumPy, for recording/verification)
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
# Spline Interpolation (CPU, used for shift / single-sample ops)
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
    """Vectorized CPU: (N, h_node+1, nu) -> (N, h_sample+1, nu)."""
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
        """Set joint targets from numpy. targets: (n_worlds, n_actuated_dofs)."""
        full = np.zeros((self.n_worlds, self.dofs_per_world), dtype=np.float32)
        full[:, free_joint_dofs:free_joint_dofs + n_actuated_dofs] = targets
        wp.copy(self.control.joint_target_pos,
                wp.array(full.reshape(-1), dtype=wp.float32, device=self.device))

    def step_control(self):
        """Step all worlds by one control step (uses CUDA graph if available)."""
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate_substeps()

    def get_base_states(self):
        """Returns (base_pos, base_quat, base_vel, base_angvel) each (N, dim). CPU numpy."""
        bq = self.state_0.body_q.numpy().reshape(self.n_worlds, self.bodies_per_world, 7)
        bqd = self.state_0.body_qd.numpy().reshape(self.n_worlds, self.bodies_per_world, 6)
        return bq[:, 0, :3].copy(), bq[:, 0, 3:7].copy(), bqd[:, 0, :3].copy(), bqd[:, 0, 3:6].copy()

    def get_body_states(self, body_idx):
        """Returns (pos, quat, vel, angvel) for a specific body index. CPU numpy."""
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
    """DIAL-MPC: MPPI with diffusion-style annealing, GPU-accelerated.

    The entire MPPI hot loop runs on GPU:
    - Noise generation via wp.randn (no CPU→GPU transfer)
    - Spline interpolation via fused Warp kernel
    - Action-to-joint mapping + target scatter via Warp kernel
    - Reward computation via user-provided Warp kernel (stays on GPU)
    - MPPI weight computation via Warp reduce kernel

    Args:
        config: MPPI configuration.
        rollout_sim: Parallel rollout simulator.
        n_actuated_dofs: Number of actuated joints.
        joint_range_low: Lower joint limits for actuated DOFs.
        joint_range_high: Upper joint limits for actuated DOFs.
        reward_kernel: A ``@wp.kernel`` with signature::

            def my_reward(body_q, body_qd, rewards, bodies_per_world, n_worlds, ...user_params):
                world_id = wp.tid()
                ...
                rewards[world_id] = reward_value

            The kernel is launched with ``dim=n_worlds`` and reads body_q/body_qd
            directly on GPU. Extra scalar/array params are passed via ``reward_params``.
        reward_params: List of extra arguments to pass to the reward kernel after
            the standard (body_q, body_qd, rewards, bodies_per_world, n_worlds).
    """

    def __init__(self, config, rollout_sim, n_actuated_dofs, joint_range_low, joint_range_high,
                 reward_kernel, reward_params=None):
        self.config = config
        self.rollout_sim = rollout_sim
        self.n_dofs = n_actuated_dofs
        self.joint_range_low = joint_range_low
        self.joint_range_high = joint_range_high
        self.reward_kernel = reward_kernel
        self.reward_params = reward_params or []
        self.Y = np.zeros((config.h_node + 1, n_actuated_dofs), dtype=np.float32)
        self.step_us, self.step_nodes = make_time_grids(config.h_sample, config.h_node, config.ctrl_dt)
        idx = np.arange(config.h_node + 1)[::-1]
        self.sigma_control = config.horizon_diffuse_factor ** idx * config.sigma_scale
        self.step_count = 0

        device = rollout_sim.device
        N = config.n_samples
        Np1 = N + 1
        h_node_p1 = config.h_node + 1
        h_sample_p1 = config.h_sample + 1

        self._device = device

        # GPU buffers
        self._Y_mean_gpu = wp.zeros((h_node_p1, n_actuated_dofs), dtype=wp.float32, device=device)
        self._noise_scale_gpu = wp.zeros(h_node_p1, dtype=wp.float32, device=device)
        self._u_all_gpu = wp.zeros((Np1, h_sample_p1, n_actuated_dofs), dtype=wp.float32, device=device)
        self._Y_all_gpu = wp.zeros((Np1, h_node_p1, n_actuated_dofs), dtype=wp.float32, device=device)
        self._low_gpu = wp.array(joint_range_low, dtype=wp.float32, device=device)
        self._high_gpu = wp.array(joint_range_high, dtype=wp.float32, device=device)
        self._step_rewards_gpu = wp.zeros((Np1, config.h_sample), dtype=wp.float32, device=device)
        self._horizon_reward_gpu = wp.zeros(Np1, dtype=wp.float32, device=device)
        self._all_rewards_gpu = wp.zeros(Np1, dtype=wp.float32, device=device)

        # Precompute interpolation matrix on GPU
        key = (tuple(self.step_nodes), tuple(self.step_us))
        if key not in _interp_cache:
            _interp_cache[key] = _build_interp_matrix(self.step_nodes, self.step_us)
        self._W_gpu = wp.array(_interp_cache[key], dtype=wp.float32, device=device)

        # Random seed (incremented each diffusion step for different noise)
        self._rng_seed = 42
        self._rng_counter = 0

    def act_to_joint(self, act):
        """Map normalized action [-1,1] -> joint angles (numpy, for real sim)."""
        norm = (act + 1.0) / 2.0
        return self.joint_range_low + norm * (self.joint_range_high - self.joint_range_low)

    def _noise_schedule(self, n_diffuse):
        traj = self.config.traj_diffuse_factor ** np.arange(n_diffuse)
        return traj[:, None] * self.sigma_control[None, :]

    def reverse_once(self, jq, jqd, Y, noise_scale, t):
        """One MPPI denoising step — fully GPU-accelerated."""
        cfg = self.config
        N = cfg.n_samples
        Np1 = N + 1
        h_node_p1 = cfg.h_node + 1
        h_sample_p1 = cfg.h_sample + 1

        # Upload mean trajectory and noise scale
        wp.copy(self._Y_mean_gpu, wp.array(Y, dtype=wp.float32, device=self._device))
        wp.copy(self._noise_scale_gpu, wp.array(noise_scale.astype(np.float32), dtype=wp.float32, device=self._device))

        # Unique RNG offset per diffusion step
        self._rng_counter += 1
        rng_offset = self._rng_counter * N * h_node_p1 * self.n_dofs

        # GPU: fused noise + spline interpolation → u_all_gpu
        wp.launch(
            _noise_spline_kernel,
            dim=(Np1, h_sample_p1, self.n_dofs),
            inputs=[self._rng_seed, rng_offset,
                    self._Y_mean_gpu, self._noise_scale_gpu,
                    self._W_gpu, self._u_all_gpu,
                    N, h_node_p1, h_sample_p1, self.n_dofs],
            device=self._device,
        )

        # GPU: generate Y_all nodes (needed for weighted average)
        wp.launch(
            _noise_nodes_kernel,
            dim=(Np1, h_node_p1, self.n_dofs),
            inputs=[self._rng_seed, rng_offset,
                    self._Y_mean_gpu, self._noise_scale_gpu,
                    self._Y_all_gpu,
                    N, h_node_p1, self.n_dofs],
            device=self._device,
        )

        # Read back interpolated actions once (small vs per-step readback)
        u_all_np = self._u_all_gpu.numpy()

        # --- Rollout loop ---
        batch_size = self.rollout_sim.n_worlds
        sim = self.rollout_sim
        step_rewards = np.zeros((Np1, cfg.h_sample), dtype=np.float32)

        for bs in range(0, Np1, batch_size):
            be = min(bs + batch_size, Np1)
            actual = be - bs
            sim.reset_all(jq, jqd)

            for h in range(cfg.h_sample):
                # CPU: act_to_joint + set_targets (6% of time, not worth GPU complexity)
                actions_h = u_all_np[bs:be, h, :]
                jt = self.act_to_joint(actions_h)
                if actual < batch_size:
                    jt_full = np.zeros((batch_size, self.n_dofs), dtype=np.float32)
                    jt_full[:actual] = jt
                else:
                    jt_full = jt
                sim.set_targets(jt_full, self.n_dofs)

                sim.step_control()

                # GPU: reward kernel (reads body_q/qd directly on device — no readback!)
                self._horizon_reward_gpu.zero_()
                wp.launch(
                    self.reward_kernel,
                    dim=sim.n_worlds,
                    inputs=[sim.state_0.body_q, sim.state_0.body_qd,
                            self._horizon_reward_gpu,
                            sim.bodies_per_world, sim.n_worlds,
                            *self.reward_params],
                    device=self._device,
                )

                # Read back rewards for this step (small: just n_worlds floats)
                step_rewards[bs:be, h] = self._horizon_reward_gpu.numpy()[:actual]

        all_rewards = step_rewards.mean(axis=1)
        baseline = all_rewards[-1]
        std = all_rewards.std() + 1e-8
        logits = (all_rewards - baseline) / std / cfg.temp_sample
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()

        # Weighted average of Y_all (CPU — small, 2049 × 5 × 12)
        Y_all_np = self._Y_all_gpu.numpy()
        new_Y = np.einsum("n,nij->ij", weights, Y_all_np)
        return new_Y, float(all_rewards[-1])

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
    """Replay a recorded trajectory and encode to MP4."""
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
# Common argument parser
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
