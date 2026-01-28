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

"""Simulation entry point for the GPU Cosserat example."""

from __future__ import annotations

import math
import os
import time
from collections import deque

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.usd
from newton.examples.cosserat2.kernels.collision import (
    collide_particles_vs_triangles_bvh_kernel,
    compute_static_tri_aabbs_kernel,
)
from newton.examples.cosserat_codex.cli import build_rod_configs
from newton.examples.cosserat_codex.rod import DefKitDirectLibrary, DefKitDirectRodState
from newton.examples.cosserat_codex.constants import (
    DIRECT_SOLVE_WARP_BANDED_CHOLESKY,
    DIRECT_SOLVE_WARP_BLOCK_THOMAS,
    TILE,
)
from newton.examples.cosserat_codex.kernels import (
    _warp_apply_concentric_constraint,
    _warp_apply_track_sliding,
    _warp_build_segment_lines,
    _warp_copy_from_offset,
    _warp_copy_with_offset,
    _warp_set_root_on_track,
    _warp_update_velocities_from_positions,
)
from newton.examples.cosserat_codex.math_utils import (
    build_director_lines,
    compute_linear_offsets,
    quat_from_axis_angle,
    quat_multiply,
)
from newton.examples.cosserat_codex.rod import RodBatch
from newton.examples.cosserat_codex.solver import CosseratXPBDSolver


def _resolve_models_dir() -> str:
    """Resolve the models directory, following pointer files if needed."""
    base_dir = os.path.dirname(__file__)
    models_path = os.path.join(base_dir, "..", "gpu_warp", "models")
    if os.path.isdir(models_path):
        return models_path

    if os.path.isfile(models_path):
        with open(models_path, "r", encoding="utf-8") as handle:
            target = handle.read().strip()
        if target:
            resolved = os.path.abspath(os.path.join(base_dir, target))
            if os.path.isdir(resolved):
                return resolved

    fallback = os.path.abspath(os.path.join(base_dir, "..", "..", "cosserat", "models"))
    return fallback


class Example:
    """Full simulation example with Newton viewer integration."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.substeps = args.substeps
        self.linear_damping = args.linear_damping
        self.angular_damping = args.angular_damping
        self.bend_stiffness = args.bend_stiffness
        self.twist_stiffness = args.twist_stiffness
        self.rest_bend_d1 = args.rest_bend_d1
        self.rest_bend_d2 = args.rest_bend_d2
        self.rest_twist = args.rest_twist
        self.young_modulus_scale = args.young_modulus / 1.0e6
        self.torsion_modulus_scale = args.torsion_modulus / 1.0e6
        self.segment_length = args.segment_length
        self.use_banded = args.use_banded
        self.compare_offset = args.compare_offset
        self.rod_count = max(int(args.rod_count), 1)
        self.rod_spacing = float(args.rod_spacing)
        self.mesh_offset = np.zeros(3, dtype=np.float32)

        self.base_gravity = np.array(args.gravity, dtype=np.float32)
        self.gravity_enabled = False
        self.gravity_scale = 1.0
        self.floor_collision_enabled = True
        self.floor_height = 0.0
        self.floor_restitution = 0.0

        self.show_segments = True
        self.show_directors = False
        self.director_scale = 0.1

        # Track sliding constraint configuration
        self.track_enabled = True
        self.track_stiffness = 1.0
        self.track_ignore_tip_count = 0
        self.use_gauss_seidel = True
        self.use_two_sided = False

        # Per-rod insertion values (arclength from track start)
        self.insertion_speed = 0.5  # units per second
        self.rod_insertions = [0.0, 0.0]  # insertion for rod 0 and rod 1

        # Bendable tip configuration
        self.tip_bend_angle = 0.0  # radians
        self.tip_num_edges = 20  # number of edges affected by the bend
        self.tip_bend_speed = 0.5  # radians per second for +/- keys

        # Concentric constraint configuration (guidewire inside catheter)
        self.concentric_enabled = False
        self.concentric_stiffness = 1.0
        self.concentric_weight_inner = 0.5  # weight for inner rod (rod A / guidewire)
        self.concentric_weight_outer = 0.5  # weight for outer rod (rod B / catheter)
        self.concentric_use_inv_mass_sq = True  # use squared barycentric weights
        self.concentric_start_particle = 0  # first particle to apply constraint to

        self.root_move_speed = 0.3
        self.root_rotate_speed = 3.0
        self.root_rotation = 0.0

        self.simulate_reference = True
        self.simulate_gpu = True
        self.use_cuda_graph = args.use_cuda_graph
        self.use_iterative_refinement = False
        self.iterative_refinement_iters = 2
        self._force_sync_reference = True
        self._force_sync_gpu = True
        self._perf_window = 90
        self._frame_times = deque(maxlen=self._perf_window)
        self._ref_times = deque(maxlen=self._perf_window)
        self._gpu_times = deque(maxlen=self._perf_window)

        self._gravity_key_was_down = False
        self._reset_key_was_down = False
        self._banded_key_was_down = False
        self._lock_key_was_down = False
        self._track_key_was_down = False
        self._tip_bend_key_was_down = False
        self._concentric_key_was_down = False

        self.lib = DefKitDirectLibrary(args.dll_path, args.calling_convention)
        self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
        if not self.supports_non_banded:
            self.use_banded = True

        rod_radius = args.rod_radius if args.rod_radius is not None else args.particle_radius
        self.ref_rod = DefKitDirectRodState(
            lib=self.lib,
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            rod_radius=rod_radius,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            young_modulus=args.young_modulus,
            torsion_modulus=args.torsion_modulus,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
            use_banded=self.use_banded,
        )

        gpu_configs = build_rod_configs(args)
        self.gpu_batch = RodBatch(gpu_configs)
        self.gpu_state = self.gpu_batch.create_state(
            device=wp.get_device(),
            use_banded=self.use_banded,
            use_cuda_graph=self.use_cuda_graph,
        )
        self.use_cuda_graph = self.gpu_state.set_use_cuda_graph(self.use_cuda_graph)
        self.gpu_solver = CosseratXPBDSolver(
            self.gpu_batch,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
        )

        target_last_pos = np.array([-3.283308, -0.50000024, 1.6833224], dtype=np.float32)
        current_last_pos = self.ref_rod.positions[-1, 0:3].astype(np.float32)
        self.mesh_offset = target_last_pos - current_last_pos
        self._update_offsets()

        models_dir = _resolve_models_dir()
        usd_path = os.path.join(models_dir, "DynamicAorta.usdc")
        if not os.path.isfile(usd_path):
            raise FileNotFoundError(
                "Unable to find DynamicAorta.usdc. "
                f"Expected at '{usd_path}'. If you're using a pointer file at "
                "'cosserat_codex/gpu_warp/models', ensure it targets the cosserat models folder."
            )
        usd_stage = Usd.Stage.Open(usd_path)
        mesh_prim = usd_stage.GetPrimAtPath("/root/A4009/A4007/Xueguan_rudong/Dynamic_vessels/Mesh")
        vessel_mesh = newton.usd.get_mesh(mesh_prim)

        self.vessel_vertices_np = np.array(vessel_mesh.vertices, dtype=np.float32)
        self.vessel_indices_np = np.array(vessel_mesh.indices, dtype=np.int32).reshape(-1, 3)
        self.num_vessel_triangles = self.vessel_indices_np.shape[0]

        self.mesh_scale = 0.01
        self.mesh_xform = wp.transform(
            wp.vec3(0.0, 0.0, 1.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0),
        )

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        vessel_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,
            kd=1.0e2,
            mu=0.1,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=vessel_mesh,
            scale=(self.mesh_scale, self.mesh_scale, self.mesh_scale),
            xform=self.mesh_xform,
            cfg=vessel_cfg,
        )

        for i in range(self.ref_rod.num_points):
            mass = 0.0 if i == 0 else args.particle_mass
            ref_pos = tuple(self.ref_rod.positions[i, 0:3] + self.ref_offset)
            builder.add_particle(pos=ref_pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)

        self._gpu_point_starts = []
        self._gpu_edge_starts = []
        point_start = self.ref_rod.num_points
        edge_start = 0
        for rod_index, rod in enumerate(self.gpu_state.rods):
            config = self.gpu_batch.configs[rod_index]
            for i in range(rod.num_points):
                mass = 0.0 if i == 0 else config.particle_mass
                gpu_pos = tuple(rod.positions[i, 0:3] + self.gpu_offsets[rod_index])
                builder.add_particle(
                    pos=gpu_pos,
                    vel=(0.0, 0.0, 0.0),
                    mass=mass,
                    radius=args.particle_radius,
                )
            self._gpu_point_starts.append(point_start)
            self._gpu_edge_starts.append(edge_start)
            point_start += rod.num_points
            edge_start += max(0, rod.num_points - 1)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self._ref_segment_colors = np.tile(
            np.array([0.2, 0.6, 1.0], dtype=np.float32), (self.ref_rod.num_points - 1, 1)
        )
        gpu_edge_count = sum(max(0, rod.num_points - 1) for rod in self.gpu_state.rods)
        self._gpu_segment_colors = np.tile(np.array([1.0, 0.6, 0.2], dtype=np.float32), (gpu_edge_count, 1))

        device = self.model.device
        self._ref_positions_wp = wp.zeros(self.ref_rod.num_points, dtype=wp.vec3, device=device)
        self._ref_velocities_wp = wp.zeros(self.ref_rod.num_points, dtype=wp.vec3, device=device)
        self._ref_segment_starts_wp = wp.zeros(self.ref_rod.num_points - 1, dtype=wp.vec3, device=device)
        self._ref_segment_ends_wp = wp.zeros(self.ref_rod.num_points - 1, dtype=wp.vec3, device=device)
        self._gpu_segment_starts_wp = wp.zeros(gpu_edge_count, dtype=wp.vec3, device=device)
        self._gpu_segment_ends_wp = wp.zeros(gpu_edge_count, dtype=wp.vec3, device=device)
        self._ref_segment_colors_wp = wp.array(self._ref_segment_colors, dtype=wp.vec3, device=device)
        self._gpu_segment_colors_wp = wp.array(self._gpu_segment_colors, dtype=wp.vec3, device=device)

        scaled_vertices = self.vessel_vertices_np * self.mesh_scale
        transformed_vertices = np.zeros_like(scaled_vertices)
        for i in range(len(scaled_vertices)):
            v = wp.vec3(scaled_vertices[i, 0], scaled_vertices[i, 1], scaled_vertices[i, 2])
            v_transformed = wp.transform_point(self.mesh_xform, v)
            transformed_vertices[i] = [v_transformed[0], v_transformed[1], v_transformed[2]]

        self.vessel_vertices = wp.array(transformed_vertices, dtype=wp.vec3f, device=device)
        self.vessel_indices = wp.array(self.vessel_indices_np, dtype=wp.int32, device=device)
        self.tri_lower_bounds = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        self.tri_upper_bounds = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        wp.launch(
            kernel=compute_static_tri_aabbs_kernel,
            dim=self.num_vessel_triangles,
            inputs=[self.vessel_vertices, self.vessel_indices],
            outputs=[self.tri_lower_bounds, self.tri_upper_bounds],
            device=device,
        )
        self.vessel_bvh = wp.Bvh(self.tri_lower_bounds, self.tri_upper_bounds)

        total_particles = self.model.particle_count
        self._collision_radii_wp = wp.array(
            np.full(total_particles, args.particle_radius, dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )
        self._collision_inv_masses_np = np.zeros(total_particles, dtype=np.float32)
        self._collision_inv_masses_wp = wp.zeros(total_particles, dtype=wp.float32, device=device)
        self._ref_positions_prev = np.zeros((self.ref_rod.num_points, 3), dtype=np.float32)
        self._gpu_positions_prev = [
            wp.zeros(rod.num_points, dtype=wp.vec3, device=device) for rod in self.gpu_state.rods
        ]

        self._sync_state_from_rods(force=True)
        self._update_gravity()
        self._ref_root_base_orientation = self.ref_rod.orientations[0].copy()
        self._gpu_root_base_orientations = [
            rod.orientations[0].copy() for rod in self.gpu_state.rods
        ]

        # Initialize track start/end from rod first/last particle positions
        ref_first_pos = self.ref_rod.positions[0, 0:3].astype(np.float32)
        ref_last_pos = self.ref_rod.positions[-1, 0:3].astype(np.float32)
        self._ref_track_start = ref_first_pos.copy()
        self._ref_track_end = ref_last_pos.copy()

        # For GPU rods, store track per rod
        self._gpu_track_starts = []
        self._gpu_track_ends = []
        for rod in self.gpu_state.rods:
            gpu_positions = rod.positions_numpy()
            self._gpu_track_starts.append(gpu_positions[0].copy())
            self._gpu_track_ends.append(gpu_positions[-1].copy())

    def __del__(self):
        if hasattr(self, "ref_rod"):
            self.ref_rod.destroy()
        if hasattr(self, "gpu_state"):
            self.gpu_state.destroy()

    def _update_offsets(self):
        half_offset = 0.5 * self.compare_offset
        self.ref_offset = self.mesh_offset + np.array([0.0, -half_offset, 0.0], dtype=np.float32)
        self.gpu_base_offset = self.mesh_offset + np.array([0.0, half_offset, 0.0], dtype=np.float32)
        self.gpu_offsets = self._build_gpu_offsets()

    def _build_gpu_offsets(self):
        offsets = []
        for x_offset in compute_linear_offsets(self.rod_count, self.rod_spacing):
            offsets.append(self.gpu_base_offset + np.array([x_offset, 0.0, 0.0], dtype=np.float32))
        return offsets

    def _update_gravity(self):
        if self.gravity_enabled:
            gravity = self.base_gravity * self.gravity_scale
        else:
            gravity = np.zeros(3, dtype=np.float32)
        self.ref_rod.set_gravity(gravity)
        self.gpu_state.set_gravity(gravity)

    def _sync_state_from_rods(self, force: bool = False):
        sync_ref = force or self.simulate_reference or self._force_sync_reference
        sync_gpu = force or self.simulate_gpu or self._force_sync_gpu

        ref_offset = wp.vec3(float(self.ref_offset[0]), float(self.ref_offset[1]), float(self.ref_offset[2]))
        zero_offset = wp.vec3(0.0, 0.0, 0.0)

        if sync_ref:
            ref_positions = self.ref_rod.positions[:, 0:3].astype(np.float32)
            ref_velocities = self.ref_rod.velocities[:, 0:3].astype(np.float32)
            self._ref_positions_wp.assign(wp.array(ref_positions, dtype=wp.vec3, device=self.model.device))
            self._ref_velocities_wp.assign(wp.array(ref_velocities, dtype=wp.vec3, device=self.model.device))
            wp.launch(
                _warp_copy_with_offset,
                dim=self.ref_rod.num_points,
                inputs=[self._ref_positions_wp, ref_offset, 0, self.state.particle_q],
                device=self.model.device,
            )
            wp.launch(
                _warp_copy_with_offset,
                dim=self.ref_rod.num_points,
                inputs=[self._ref_velocities_wp, zero_offset, 0, self.state.particle_qd],
                device=self.model.device,
            )
            self._force_sync_reference = False

        if sync_gpu:
            for idx, rod in enumerate(self.gpu_state.rods):
                gpu_offset = self.gpu_offsets[idx]
                offset_wp = wp.vec3(float(gpu_offset[0]), float(gpu_offset[1]), float(gpu_offset[2]))
                start = self._gpu_point_starts[idx]
                wp.launch(
                    _warp_copy_with_offset,
                    dim=rod.num_points,
                    inputs=[rod.positions_wp, offset_wp, start, self.state.particle_q],
                    device=self.model.device,
                )
                wp.launch(
                    _warp_copy_with_offset,
                    dim=rod.num_points,
                    inputs=[rod.velocities_wp, zero_offset, start, self.state.particle_qd],
                    device=self.model.device,
                )
            self._force_sync_gpu = False

    def _update_collision_inv_masses(self):
        self._collision_inv_masses_np[: self.ref_rod.num_points] = self.ref_rod.inv_masses
        for idx, rod in enumerate(self.gpu_state.rods):
            start = self._gpu_point_starts[idx]
            end = start + rod.num_points
            self._collision_inv_masses_np[start:end] = rod.inv_masses
        self._collision_inv_masses_wp.assign(
            wp.array(self._collision_inv_masses_np, dtype=wp.float32, device=self.model.device)
        )

    def _apply_mesh_collisions(self, dt: float):
        if not hasattr(self, "vessel_bvh"):
            return

        self._ref_positions_prev[:] = self.ref_rod.positions[:, 0:3]
        for idx, rod in enumerate(self.gpu_state.rods):
            wp.copy(self._gpu_positions_prev[idx], rod.positions_wp)

        self._update_collision_inv_masses()
        self._sync_state_from_rods(force=True)

        wp.launch(
            kernel=collide_particles_vs_triangles_bvh_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.state.particle_q,
                self._collision_radii_wp,
                self._collision_inv_masses_wp,
                self.vessel_vertices,
                self.vessel_indices,
                self.vessel_bvh.id,
                self.use_gauss_seidel,
                self.use_two_sided,
            ],
            outputs=[self.state.particle_q],
            device=self.model.device,
        )

        ref_offset = wp.vec3(float(self.ref_offset[0]), float(self.ref_offset[1]), float(self.ref_offset[2]))
        wp.launch(
            _warp_copy_from_offset,
            dim=self.ref_rod.num_points,
            inputs=[self.state.particle_q, ref_offset, 0, self._ref_positions_wp],
            device=self.model.device,
        )
        ref_positions = self._ref_positions_wp.numpy()
        self.ref_rod.positions[:, 0:3] = ref_positions
        self.ref_rod.predicted_positions[:, 0:3] = ref_positions

        ref_vel = (ref_positions - self._ref_positions_prev) / float(dt)
        self.ref_rod.velocities[:, 0:3] = ref_vel
        self.ref_rod.velocities[self.ref_rod.inv_masses == 0.0, 0:3] = 0.0

        for idx, rod in enumerate(self.gpu_state.rods):
            gpu_offset = self.gpu_offsets[idx]
            offset_wp = wp.vec3(float(gpu_offset[0]), float(gpu_offset[1]), float(gpu_offset[2]))
            start = self._gpu_point_starts[idx]
            wp.launch(
                _warp_copy_from_offset,
                dim=rod.num_points,
                inputs=[self.state.particle_q, offset_wp, start, rod.positions_wp],
                device=self.model.device,
            )
            wp.launch(
                _warp_copy_from_offset,
                dim=rod.num_points,
                inputs=[self.state.particle_q, offset_wp, start, rod.predicted_positions_wp],
                device=self.model.device,
            )
            wp.launch(
                _warp_update_velocities_from_positions,
                dim=rod.num_points,
                inputs=[
                    self._gpu_positions_prev[idx],
                    rod.positions_wp,
                    rod.inv_masses_wp,
                    float(dt),
                    rod.velocities_wp,
                ],
                device=self.model.device,
            )

        self._force_sync_reference = True
        self._force_sync_gpu = True

    def _apply_track_constraint(self):
        """Apply track sliding constraint to keep particles on a line segment."""
        if not self.track_enabled:
            return

        self._update_collision_inv_masses()
        self._sync_state_from_rods(force=True)

        ref_num_constrained = self.ref_rod.num_points - self.track_ignore_tip_count
        if ref_num_constrained > 0:
            track_start = self._ref_track_start + self.ref_offset
            track_end = self._ref_track_end + self.ref_offset
            track_start_wp = wp.vec3(float(track_start[0]), float(track_start[1]), float(track_start[2]))
            track_end_wp = wp.vec3(float(track_end[0]), float(track_end[1]), float(track_end[2]))

            wp.launch(
                _warp_apply_track_sliding,
                dim=ref_num_constrained,
                inputs=[
                    self.state.particle_q,
                    self.state.particle_q,
                    self._collision_inv_masses_wp,
                    track_start_wp,
                    track_end_wp,
                    float(self.track_stiffness),
                    0,
                    ref_num_constrained,
                ],
                device=self.model.device,
            )

            ref_offset_wp = wp.vec3(
                float(self.ref_offset[0]), float(self.ref_offset[1]), float(self.ref_offset[2])
            )
            wp.launch(
                _warp_copy_from_offset,
                dim=self.ref_rod.num_points,
                inputs=[self.state.particle_q, ref_offset_wp, 0, self._ref_positions_wp],
                device=self.model.device,
            )
            ref_positions = self._ref_positions_wp.numpy()
            self.ref_rod.positions[:, 0:3] = ref_positions
            self.ref_rod.predicted_positions[:, 0:3] = ref_positions

        for idx, rod in enumerate(self.gpu_state.rods):
            gpu_num_constrained = rod.num_points - self.track_ignore_tip_count
            if gpu_num_constrained <= 0:
                continue

            track_start = self._gpu_track_starts[idx] + self.gpu_offsets[idx]
            track_end = self._gpu_track_ends[idx] + self.gpu_offsets[idx]
            track_start_wp = wp.vec3(float(track_start[0]), float(track_start[1]), float(track_start[2]))
            track_end_wp = wp.vec3(float(track_end[0]), float(track_end[1]), float(track_end[2]))

            start = self._gpu_point_starts[idx]
            end = start + gpu_num_constrained

            if idx < len(self.rod_insertions):
                insertion = self.rod_insertions[idx]
                wp.launch(
                    _warp_set_root_on_track,
                    dim=1,
                    inputs=[
                        rod.positions_wp,
                        rod.predicted_positions_wp,
                        rod.velocities_wp,
                        track_start_wp,
                        track_end_wp,
                        float(insertion),
                        0,
                    ],
                    device=self.model.device,
                )

            wp.launch(
                _warp_apply_track_sliding,
                dim=gpu_num_constrained,
                inputs=[
                    self.state.particle_q,
                    self.state.particle_q,
                    self._collision_inv_masses_wp,
                    track_start_wp,
                    track_end_wp,
                    float(self.track_stiffness),
                    start,
                    end,
                ],
                device=self.model.device,
            )

            gpu_offset = self.gpu_offsets[idx]
            offset_wp = wp.vec3(float(gpu_offset[0]), float(gpu_offset[1]), float(gpu_offset[2]))
            wp.launch(
                _warp_copy_from_offset,
                dim=rod.num_points,
                inputs=[self.state.particle_q, offset_wp, start, rod.positions_wp],
                device=self.model.device,
            )
            wp.launch(
                _warp_copy_from_offset,
                dim=rod.num_points,
                inputs=[self.state.particle_q, offset_wp, start, rod.predicted_positions_wp],
                device=self.model.device,
            )

        self._force_sync_reference = True
        self._force_sync_gpu = True

    def _apply_concentric_constraint(self):
        """Apply concentric constraint between GPU rods (guidewire inside catheter)."""
        if not self.concentric_enabled:
            return

        if len(self.gpu_state.rods) < 2:
            return

        rod_a = self.gpu_state.rods[0]
        rod_b = self.gpu_state.rods[1]

        insertion_a = self.rod_insertions[0] if len(self.rod_insertions) > 0 else 0.0
        insertion_b = self.rod_insertions[1] if len(self.rod_insertions) > 1 else 0.0
        insertion_diff = insertion_a - insertion_b

        if hasattr(rod_a, "rest_lengths_wp") and rod_a.rest_lengths_wp is not None:
            rest_lengths_a = rod_a.rest_lengths_wp
        else:
            if not hasattr(self, "_concentric_rest_lengths_a"):
                self._concentric_rest_lengths_a = wp.array(
                    rod_a.rest_lengths, dtype=wp.float32, device=self.model.device
                )
            rest_lengths_a = self._concentric_rest_lengths_a

        if hasattr(rod_b, "rest_lengths_wp") and rod_b.rest_lengths_wp is not None:
            rest_lengths_b = rod_b.rest_lengths_wp
        else:
            if not hasattr(self, "_concentric_rest_lengths_b"):
                self._concentric_rest_lengths_b = wp.array(
                    rod_b.rest_lengths, dtype=wp.float32, device=self.model.device
                )
            rest_lengths_b = self._concentric_rest_lengths_b

        wp.launch(
            _warp_apply_concentric_constraint,
            dim=rod_a.num_points,
            inputs=[
                rod_a.positions_wp,
                rod_a.predicted_positions_wp,
                rod_a.inv_masses_wp,
                rod_a.num_points,
                rod_b.positions_wp,
                rod_b.predicted_positions_wp,
                rod_b.inv_masses_wp,
                rod_b.num_points,
                rest_lengths_a,
                rest_lengths_b,
                float(insertion_diff),
                float(self.concentric_stiffness),
                float(self.concentric_weight_inner),
                float(self.concentric_weight_outer),
                int(1 if self.concentric_use_inv_mass_sq else 0),
                int(self.concentric_start_particle),
            ],
            device=self.model.device,
        )

        self._force_sync_gpu = True

    def _apply_tip_bend(self):
        """Apply bend to the tip edges by modifying rest Darboux vector."""
        num_edges = max(1, self.tip_num_edges)
        per_edge_bend = self.tip_bend_angle / num_edges

        ref_num_edges = self.ref_rod.num_points - 1
        tip_start_edge = max(0, ref_num_edges - num_edges)

        for e in range(tip_start_edge, ref_num_edges):
            self.ref_rod.rest_darboux[e, 0] = self.rest_bend_d1 + per_edge_bend
            self.ref_rod.rest_darboux[e, 1] = self.rest_bend_d2
            self.ref_rod.rest_darboux[e, 2] = self.rest_twist

        for e in range(tip_start_edge):
            self.ref_rod.rest_darboux[e, 0] = self.rest_bend_d1
            self.ref_rod.rest_darboux[e, 1] = self.rest_bend_d2
            self.ref_rod.rest_darboux[e, 2] = self.rest_twist

        for rod in self.gpu_state.rods:
            gpu_num_edges = rod.num_points - 1
            gpu_tip_start = max(0, gpu_num_edges - num_edges)

            for e in range(gpu_tip_start, gpu_num_edges):
                rod.rest_darboux[e, 0] = self.rest_bend_d1 + per_edge_bend
                rod.rest_darboux[e, 1] = self.rest_bend_d2
                rod.rest_darboux[e, 2] = self.rest_twist

            for e in range(gpu_tip_start):
                rod.rest_darboux[e, 0] = self.rest_bend_d1
                rod.rest_darboux[e, 1] = self.rest_bend_d2
                rod.rest_darboux[e, 2] = self.rest_twist

            if hasattr(rod, "rest_darboux_wp") and rod.num_edges > 0:
                rod.rest_darboux_wp.assign(
                    wp.array(rod.rest_darboux[:, 0:3], dtype=wp.vec3, device=rod.device)
                )

    def _handle_keyboard_input(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        g_down = self.viewer.is_key_down(key.G)
        if g_down and not self._gravity_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            self._update_gravity()
        self._gravity_key_was_down = g_down

        b_down = self.viewer.is_key_down(key.B)
        if b_down and not self._banded_key_was_down:
            gpu_backends = [DIRECT_SOLVE_WARP_BLOCK_THOMAS, DIRECT_SOLVE_WARP_BANDED_CHOLESKY]
            if self.gpu_state.rods:
                current_backend = self.gpu_state.rods[0].direct_solve_backend
                try:
                    current_idx = gpu_backends.index(current_backend)
                    next_idx = (current_idx + 1) % len(gpu_backends)
                except ValueError:
                    next_idx = 0
                for rod in self.gpu_state.rods:
                    rod.set_direct_solve_backend(gpu_backends[next_idx])
                    rod._graph = None
                    rod._graph_params = None
        self._banded_key_was_down = b_down

        l_down = self.viewer.is_key_down(key.L)
        if l_down and not self._lock_key_was_down:
            self.ref_rod.toggle_root_lock()
            for rod in self.gpu_state.rods:
                rod.toggle_root_lock()
        self._lock_key_was_down = l_down

        t_down = self.viewer.is_key_down(key.T)
        if t_down and not self._track_key_was_down:
            self.track_enabled = not self.track_enabled
        self._track_key_was_down = t_down

        c_down = self.viewer.is_key_down(key.C)
        if c_down and not self._concentric_key_was_down:
            if len(self.gpu_state.rods) >= 2:
                self.concentric_enabled = not self.concentric_enabled
        self._concentric_key_was_down = c_down

        if (self.viewer.is_key_down(key.PLUS) or self.viewer.is_key_down(key.EQUAL) or
                self.viewer.is_key_down(key.NUM_ADD)):
            self.tip_bend_angle += self.tip_bend_speed * self.frame_dt
            self._apply_tip_bend()

        if self.viewer.is_key_down(key.MINUS) or self.viewer.is_key_down(key.NUM_SUBTRACT):
            self.tip_bend_angle -= self.tip_bend_speed * self.frame_dt
            self._apply_tip_bend()

        if self.viewer.is_key_down(key.PAGEUP):
            self.rod_insertions[0] += self.insertion_speed * self.frame_dt
        if self.viewer.is_key_down(key.PAGEDOWN):
            self.rod_insertions[0] -= self.insertion_speed * self.frame_dt
            self.rod_insertions[0] = max(0.0, self.rod_insertions[0])

        if len(self.gpu_state.rods) >= 2:
            if self.viewer.is_key_down(key.HOME):
                self.rod_insertions[1] += self.insertion_speed * self.frame_dt
            if self.viewer.is_key_down(key.END):
                self.rod_insertions[1] -= self.insertion_speed * self.frame_dt
                self.rod_insertions[1] = max(0.0, self.rod_insertions[1])

        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._reset_key_was_down:
            self.ref_rod.reset()
            self.gpu_state.reset()
            self.root_rotation = 0.0
            self.tip_bend_angle = 0.0
            self.rod_insertions = [0.0, 0.0]
            self._apply_root_rotation()
            self._apply_tip_bend()
            self.sim_time = 0.0
            self._sync_state_from_rods()
        self._reset_key_was_down = r_down

        dx = 0.0
        dy = 0.0
        dz = 0.0

        if self.viewer.is_key_down(key.NUM_6):
            dx += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4):
            dx -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_8):
            dy += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2):
            dy -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_9):
            dz += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3):
            dz -= self.root_move_speed * self.frame_dt

        rotation_changed = False
        if self.viewer.is_key_down(key.NUM_7):
            self.root_rotation += self.root_rotate_speed * self.frame_dt
            rotation_changed = True
        if self.viewer.is_key_down(key.NUM_1):
            self.root_rotation -= self.root_rotate_speed * self.frame_dt
            rotation_changed = True

        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            self._apply_root_translation(dx, dy, dz)
        if rotation_changed:
            self._apply_root_rotation()

    def step(self):
        self._handle_keyboard_input()

        sub_dt = self.frame_dt / max(self.substeps, 1)
        frame_start = time.perf_counter()
        ref_time = 0.0
        gpu_time = 0.0
        self.gpu_solver.linear_damping = self.linear_damping
        self.gpu_solver.angular_damping = self.angular_damping
        for _ in range(self.substeps):
            if self.simulate_reference:
                t0 = time.perf_counter()
                self.ref_rod.step(sub_dt, self.linear_damping, self.angular_damping)
                if self.floor_collision_enabled:
                    self.ref_rod.apply_floor_collisions(self.floor_height, self.floor_restitution)
                ref_time += time.perf_counter() - t0
            if self.simulate_gpu:
                t0 = time.perf_counter()
                self.gpu_solver.step(self.gpu_state, self.gpu_state, None, None, sub_dt)
                if self.floor_collision_enabled:
                    self.gpu_state.apply_floor_collisions(self.floor_height, self.floor_restitution)
                gpu_time += time.perf_counter() - t0
            self._apply_mesh_collisions(sub_dt)
            self._apply_track_constraint()
            self._apply_concentric_constraint()

        self._sync_state_from_rods()
        self.sim_time += self.frame_dt
        frame_end = time.perf_counter()
        self._frame_times.append(frame_end - frame_start)
        if self.simulate_reference:
            self._ref_times.append(ref_time)
        if self.simulate_gpu:
            self._gpu_times.append(gpu_time)

    def _apply_root_translation(self, dx: float, dy: float, dz: float):
        delta = np.array([dx, dy, dz], dtype=np.float32)

        pos = self.ref_rod.positions[0, 0:3]
        new_pos = pos + delta
        self.ref_rod.positions[0, 0:3] = new_pos
        self.ref_rod.predicted_positions[0, 0:3] = new_pos
        self.ref_rod.velocities[0, 0:3] = 0.0
        self._force_sync_reference = True

        for rod in self.gpu_state.rods:
            rod.apply_root_translation(dx, dy, dz)
        self._force_sync_gpu = True

    def _apply_root_rotation(self):
        q_twist = quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.root_rotation)
        q_ref = quat_multiply(self._ref_root_base_orientation, q_twist)
        self.ref_rod.orientations[0] = q_ref
        self.ref_rod.predicted_orientations[0] = q_ref
        self.ref_rod.prev_orientations[0] = q_ref
        self._force_sync_reference = True

        for idx, rod in enumerate(self.gpu_state.rods):
            q_gpu = quat_multiply(self._gpu_root_base_orientations[idx], q_twist)
            rod.apply_root_rotation(q_gpu)
        self._force_sync_gpu = True

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        if self.show_segments:
            ref_offset = wp.vec3(float(self.ref_offset[0]), float(self.ref_offset[1]), float(self.ref_offset[2]))
            wp.launch(
                _warp_build_segment_lines,
                dim=self.ref_rod.num_points - 1,
                inputs=[
                    self._ref_positions_wp,
                    ref_offset,
                    0,
                    self._ref_segment_starts_wp,
                    self._ref_segment_ends_wp,
                ],
                device=self.model.device,
            )
            for idx, rod in enumerate(self.gpu_state.rods):
                gpu_offset = self.gpu_offsets[idx]
                offset_wp = wp.vec3(float(gpu_offset[0]), float(gpu_offset[1]), float(gpu_offset[2]))
                wp.launch(
                    _warp_build_segment_lines,
                    dim=rod.num_points - 1,
                    inputs=[
                        rod.positions_wp,
                        offset_wp,
                        int(self._gpu_edge_starts[idx]),
                        self._gpu_segment_starts_wp,
                        self._gpu_segment_ends_wp,
                    ],
                    device=self.model.device,
                )
            self.viewer.log_lines(
                "/rod_reference",
                self._ref_segment_starts_wp,
                self._ref_segment_ends_wp,
                self._ref_segment_colors_wp,
            )
            self.viewer.log_lines(
                "/rod_gpu",
                self._gpu_segment_starts_wp,
                self._gpu_segment_ends_wp,
                self._gpu_segment_colors_wp,
            )
        else:
            self.viewer.log_lines("/rod_reference", None, None, None)
            self.viewer.log_lines("/rod_gpu", None, None, None)

        if self.show_directors:
            ref_positions = self.ref_rod.positions[:, 0:3].astype(np.float32)
            ref_orientations = self.ref_rod.orientations.astype(np.float32)
            ref_starts, ref_ends, ref_colors = build_director_lines(
                ref_positions, ref_orientations, self.ref_offset, self.director_scale
            )
            self.viewer.log_lines(
                "/directors_reference",
                wp.array(ref_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_colors, dtype=wp.vec3, device=self.model.device),
            )

            for idx, rod in enumerate(self.gpu_state.rods):
                gpu_positions = rod.positions_numpy().astype(np.float32)
                gpu_orientations = rod.orientations_numpy().astype(np.float32)
                gpu_starts, gpu_ends, gpu_colors = build_director_lines(
                    gpu_positions, gpu_orientations, self.gpu_offsets[idx], self.director_scale
                )
                self.viewer.log_lines(
                    f"/directors_gpu_{idx}",
                    wp.array(gpu_starts, dtype=wp.vec3, device=self.model.device),
                    wp.array(gpu_ends, dtype=wp.vec3, device=self.model.device),
                    wp.array(gpu_colors, dtype=wp.vec3, device=self.model.device),
                )
        else:
            self.viewer.log_lines("/directors_reference", None, None, None)
            for idx in range(len(self.gpu_state.rods)):
                self.viewer.log_lines(f"/directors_gpu_{idx}", None, None, None)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Direct Cosserat Rod: Reference + GPU Warp")
        ui.text(f"Particles per rod: {self.ref_rod.num_points}")
        ui.text(f"GPU rods: {self.rod_count}")
        ui.text("Reference: blue, GPU: orange")
        ui.separator()

        changed_ref, self.simulate_reference = ui.checkbox("Simulate Reference (C++)", self.simulate_reference)
        changed_gpu, self.simulate_gpu = ui.checkbox("Simulate GPU (Warp)", self.simulate_gpu)
        changed_graph, self.use_cuda_graph = ui.checkbox("Use CUDA Graph (GPU rod)", self.use_cuda_graph)
        if changed_graph:
            self.use_cuda_graph = self.gpu_state.set_use_cuda_graph(self.use_cuda_graph)
        if changed_ref or changed_gpu:
            self._frame_times.clear()
            self._ref_times.clear()
            self._gpu_times.clear()

        _changed, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _changed, self.linear_damping = ui.slider_float("Linear Damping", self.linear_damping, 0.0, 0.05)
        _changed, self.angular_damping = ui.slider_float("Angular Damping", self.angular_damping, 0.0, 0.05)

        ui.separator()
        offset_changed, self.compare_offset = ui.slider_float("Compare Offset", self.compare_offset, 0.0, 5.0)
        if offset_changed:
            self._update_offsets()
            self._sync_state_from_rods(force=True)

        ui.separator()
        changed_bend_k, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float(
            "Twist Stiffness", self.twist_stiffness, 0.0, 1.0
        )
        if changed_bend_k or changed_twist_k:
            self.ref_rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)
            self.gpu_state.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)

        ui.separator()
        changed_rest_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_rest_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_rest_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_rest_d1 or changed_rest_d2 or changed_rest_twist:
            self.ref_rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)
            self.gpu_state.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)

        ui.separator()
        changed_E, self.young_modulus_scale = ui.slider_float("Young Modulus (1e6)", self.young_modulus_scale, 0.1, 1000.0)
        changed_G, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Modulus (1e6)", self.torsion_modulus_scale, 0.1, 1000.0
        )
        if changed_E or changed_G:
            self.ref_rod.young_modulus = self.young_modulus_scale * 1.0e6
            self.ref_rod.torsion_modulus = self.torsion_modulus_scale * 1.0e6
            for rod in self.gpu_state.rods:
                rod.young_modulus = self.young_modulus_scale * 1.0e6
                rod.torsion_modulus = self.torsion_modulus_scale * 1.0e6

        changed_seg, self.segment_length = ui.slider_float("Segment Length", self.segment_length, 0.01, 0.2)
        if changed_seg:
            self.ref_rod.rest_lengths[:] = self.segment_length
            for rod in self.gpu_state.rods:
                rod.rest_lengths[:] = self.segment_length
                if hasattr(rod, "rest_lengths_wp") and rod.num_edges > 0:
                    rod.rest_lengths_wp.assign(
                        wp.array(rod.rest_lengths, dtype=wp.float32, device=rod.device)
                    )

        ui.separator()
        _changed, self.gravity_scale = ui.slider_float("Gravity Scale", self.gravity_scale, 0.0, 2.0)
        _changed, self.floor_height = ui.slider_float("Floor Height", self.floor_height, -1.0, 1.0)
        _changed, self.floor_restitution = ui.slider_float("Floor Restitution", self.floor_restitution, 0.0, 1.0)

        ui.separator()
        ui.text("Track Sliding Constraint")
        _changed, self.track_enabled = ui.checkbox("Enable Track", self.track_enabled)
        _changed, self.track_stiffness = ui.slider_float("Track Stiffness", self.track_stiffness, 0.0, 1.0)
        _changed, self.track_ignore_tip_count = ui.slider_int(
            "Ignore Tip Particles", self.track_ignore_tip_count, 0, 10
        )

        ui.separator()
        ui.text("Rod Insertion (PgUp/PgDn: Rod0, Home/End: Rod1)")
        _changed, self.insertion_speed = ui.slider_float("Insertion Speed", self.insertion_speed, 0.1, 2.0)
        if len(self.gpu_state.rods) >= 1:
            _changed, self.rod_insertions[0] = ui.slider_float(
                "Rod 0 Insertion", self.rod_insertions[0], 0.0, 5.0
            )
        if len(self.gpu_state.rods) >= 2:
            _changed, self.rod_insertions[1] = ui.slider_float(
                "Rod 1 Insertion", self.rod_insertions[1], 0.0, 5.0
            )
            insertion_diff = self.rod_insertions[0] - self.rod_insertions[1]
            ui.text(f"Insertion Diff: {insertion_diff:.3f}")

        ui.separator()
        ui.text("Concentric Constraint (Guidewire/Catheter)")
        if len(self.gpu_state.rods) >= 2:
            _changed, self.concentric_enabled = ui.checkbox("Enable Concentric", self.concentric_enabled)
            _changed, self.concentric_stiffness = ui.slider_float(
                "Concentric Stiffness", self.concentric_stiffness, 0.0, 1.0
            )
            _changed, self.concentric_weight_inner = ui.slider_float(
                "Inner Rod Weight", self.concentric_weight_inner, 0.0, 1.0
            )
            _changed, self.concentric_weight_outer = ui.slider_float(
                "Outer Rod Weight", self.concentric_weight_outer, 0.0, 1.0
            )
            _changed, self.concentric_start_particle = ui.slider_int(
                "Start Particle", self.concentric_start_particle, 0, 20
            )
            _changed, self.concentric_use_inv_mass_sq = ui.checkbox(
                "Use Inv Mass Squared", self.concentric_use_inv_mass_sq
            )
        else:
            ui.text("(Requires 2+ GPU rods)")

        ui.separator()
        ui.text("Bendable Tip")
        
        changed_tip_angle, self.tip_bend_angle = ui.slider_float(
            "Tip Bend Angle (rad)", self.tip_bend_angle, -1.5, 1.5
        )
        changed_tip_edges, self.tip_num_edges = ui.slider_int(
            "Tip Edges Affected", self.tip_num_edges, 1, 10
        )
        _changed, self.tip_bend_speed = ui.slider_float("Tip Bend Speed", self.tip_bend_speed, 0.1, 2.0)
        if changed_tip_angle or changed_tip_edges:
                self._apply_tip_bend()

        ui.separator()
        ui.text("Collisions")
        _changed, self.use_gauss_seidel = ui.checkbox("Use Gauss-Seidel", self.use_gauss_seidel)
        _changed, self.use_two_sided = ui.checkbox("Use Two-Sided Collisions", self.use_two_sided)
        ui.separator()
        ui.text("Rod Solver")
        
        gpu_backend_labels = [
            "Block Thomas",
            "Banded Cholesky",
        ]
        gpu_backend_values = [
            DIRECT_SOLVE_WARP_BLOCK_THOMAS,
            DIRECT_SOLVE_WARP_BANDED_CHOLESKY,
        ]
        if self.gpu_state.rods:
            current_gpu_backend = self.gpu_state.rods[0].direct_solve_backend
            try:
                current_gpu_idx = gpu_backend_values.index(current_gpu_backend)
            except ValueError:
                current_gpu_idx = 0
            changed_gpu_backend, new_gpu_idx = ui.combo("GPU Solver", current_gpu_idx, gpu_backend_labels)
            if changed_gpu_backend:
                for rod in self.gpu_state.rods:
                    rod.set_direct_solve_backend(gpu_backend_values[new_gpu_idx])
                    rod._graph = None
                    rod._graph_params = None

            if self.gpu_state.rods:
                n_dofs = self.gpu_state.rods[0].n_dofs
                if current_gpu_backend == DIRECT_SOLVE_WARP_BANDED_CHOLESKY:
                    ui.text("  Using Warp banded Cholesky (spbsv_u11_1rhs-style)")
                    changed_iter_ref, self.use_iterative_refinement = ui.checkbox(
                        "Iterative Refinement", self.use_iterative_refinement
                    )
                    changed_iter_count, self.iterative_refinement_iters = ui.slider_int(
                        "Refinement Iterations", self.iterative_refinement_iters, 1, 10
                    )
                    if changed_iter_ref or changed_iter_count:
                        for rod in self.gpu_state.rods:
                            rod.set_iterative_refinement(
                                self.use_iterative_refinement, self.iterative_refinement_iters
                            )
                elif n_dofs <= TILE:
                    ui.text(f"  Using Warp dense tiled Cholesky (n_dofs={n_dofs} <= TILE={TILE})")
                else:
                    ui.text(f"  Using Warp block Thomas (n_dofs={n_dofs} > TILE={TILE})")

        if self.supports_non_banded:
            changed_banded, self.use_banded = ui.checkbox("Use Banded Solver (Ref)", self.use_banded)
            if changed_banded:
                self.ref_rod.set_solver_mode(self.use_banded)
                self.use_banded = self.ref_rod.use_banded
        else:
            ui.text("Non-banded solver not available in this DLL build.")

        ui.separator()
        ui.text("GPU Direct Stabilization")
        if self.gpu_state.rods:
            ui.text(f"GPU max |C|: {self.gpu_state.rods[0].last_constraint_max:.3e}")
            ui.text(f"GPU max |Δλ|: {self.gpu_state.rods[0].last_delta_lambda_max:.3e}")
            ui.text(f"GPU max correction: {self.gpu_state.rods[0].last_correction_max:.3e}")

        ui.separator()
        ui.text("Performance (avg over recent frames)")
        if self._frame_times:
            avg_frame = sum(self._frame_times) / len(self._frame_times)
            fps = 1.0 / avg_frame if avg_frame > 0.0 else 0.0
            ui.text(f"Frame step: {avg_frame * 1000.0:.2f} ms ({fps:.1f} FPS)")
        if self._ref_times:
            avg_ref = sum(self._ref_times) / len(self._ref_times)
            ui.text(f"Reference step: {avg_ref * 1000.0:.2f} ms")
        if self._gpu_times:
            avg_gpu = sum(self._gpu_times) / len(self._gpu_times)
            ui.text(f"GPU step: {avg_gpu * 1000.0:.2f} ms")

        ui.separator()
        _changed, self.show_segments = ui.checkbox("Show Rod Segments", self.show_segments)
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.3)

        ui.separator()
        ui.text("Root Control (Numpad, both rods)")
        _changed, self.root_move_speed = ui.slider_float("Move Speed", self.root_move_speed, 0.1, 5.0)
        _changed, self.root_rotate_speed = ui.slider_float("Rotate Speed", self.root_rotate_speed, 0.1, 3.0)
        ui.text(f"  Rotation: {self.root_rotation:.2f} rad")
        ui.text("  4/6: X-, X+  8/2: Y+, Y-  9/3: Z+, Z-")
        ui.text("  7/1: Rotate +Z/-Z")

        ui.separator()
        ui.text("Controls:")
        ui.text("  G: Toggle gravity")
        ui.text("  B: Cycle GPU solver (Thomas/Banded)")
        ui.text("  L: Toggle root lock (position + rotation)")
        ui.text("  T: Toggle track sliding constraint")
        ui.text("  C: Toggle concentric constraint")
        ui.text("  P: Toggle bendable tip")
        ui.text("  +/- or Numpad +/-: Adjust tip bend angle")
        ui.text("  PgUp/PgDn: Rod 0 insertion +/-")
        ui.text("  Home/End: Rod 1 insertion +/-")
        ui.text("  R: Reset")

    def test_final(self):
        ref_anchor = self.ref_rod.positions[0, 0:3]
        ref_initial = self.ref_rod._initial_positions[0, 0:3]
        ref_dist = float(np.linalg.norm(ref_anchor - ref_initial))
        assert ref_dist < 1.0e-3, f"Reference anchor moved too far: {ref_dist}"

        for rod in self.gpu_state.rods:
            gpu_positions = rod.positions_numpy()
            gpu_anchor = gpu_positions[0]
            gpu_initial = rod._initial_positions[0, 0:3]
            gpu_dist = float(np.linalg.norm(gpu_anchor - gpu_initial))
            assert gpu_dist < 1.0e-3, f"GPU anchor moved too far: {gpu_dist}"

        if not np.all(np.isfinite(self.ref_rod.positions[:, 0:3])):
            raise AssertionError("Non-finite reference positions detected")

        for rod in self.gpu_state.rods:
            if not np.all(np.isfinite(rod.positions_numpy())):
                raise AssertionError("Non-finite GPU positions detected")


__all__ = ["Example"]
