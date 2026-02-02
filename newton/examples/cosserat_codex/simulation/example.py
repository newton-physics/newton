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
from dataclasses import dataclass, field

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.usd
from newton.examples.cosserat2.kernels.collision import (
    collide_particles_vs_triangles_bvh_kernel,
    compute_static_tri_aabbs_kernel,
)
from newton.examples.cosserat_codex.cli import SolverType, build_rod_configs, parse_solver_types
from newton.examples.cosserat_codex.constants import (
    DIRECT_SOLVE_WARP_BANDED_CHOLESKY,
    DIRECT_SOLVE_WARP_BLOCK_JACOBI,
    DIRECT_SOLVE_WARP_BLOCK_THOMAS,
    DIRECT_SOLVE_WARP_SPLIT_THOMAS,
    TILE,
)
from newton.examples.cosserat_codex.kernels import (
    _warp_apply_track_sliding,
    _warp_build_segment_lines,
    _warp_copy_from_offset,
    _warp_copy_from_offset_batched,
    _warp_copy_with_offset,
    _warp_copy_with_offset_batched,
    _warp_set_root_on_track,
    _warp_update_velocities_from_positions,
    warp_concentric_constraint_direct,
)
from newton.examples.cosserat_codex.math_utils import (
    build_director_lines,
    compute_linear_offsets,
    quat_from_axis_angle,
    quat_multiply,
)
from newton.examples.cosserat_codex.meshing import RodMesher
from newton.examples.cosserat_codex.rod import (
    DefKitDirectLibrary,
    DefKitDirectRodState,
    NumpyDirectRodState,
    RodBatch,
    RodConfig,
)
from newton.examples.cosserat_codex.rod.base import RodStateBase
from newton.examples.cosserat_codex.solver import CosseratXPBDSolver


@dataclass
class RodInfo:
    """Container for per-rod information."""

    rod: RodStateBase  # The rod state instance (NumPy, Warp, or DLL)
    solver_type: SolverType  # Type of solver used
    config: RodConfig  # Rod configuration
    offset: np.ndarray  # World offset for rendering
    color: list  # RGB color for visualization
    mesh_radius: float  # Radius for tube mesh rendering
    particle_radius: float = 0.01  # Radius for particle visualization
    particle_color: list = field(default_factory=list)  # RGB color for particles

    def __post_init__(self):
        if not self.particle_color:
            self.particle_color = self.color.copy() if self.color else [0.7, 0.6, 0.4]


def _resolve_models_dir() -> str:
    """Resolve the models directory, following pointer files if needed."""
    base_dir = os.path.dirname(__file__)
    models_path = os.path.join(base_dir, "..", "gpu_warp", "models")
    if os.path.isdir(models_path):
        return models_path

    if os.path.isfile(models_path):
        with open(models_path, encoding="utf-8") as handle:
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
        self.collision_iterations = 2  # Number of collision iterations per substep
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
        self.show_rod_mesh = False
        self.mesh_resolution = 8
        self.mesh_smoothing = 3

        # Parse solver types from command line
        self.solver_types = parse_solver_types(args.rod_solvers, args.rod_count)

        # Color palette for rods (indexed by solver type and rod index)
        self._rod_colors = {
            SolverType.NUMPY: [0.2, 0.6, 1.0],  # Blue for NumPy
            SolverType.WARP: [1.0, 0.6, 0.2],  # Orange for Warp
            SolverType.DLL: [0.2, 0.8, 0.4],  # Green for DLL
        }
        # Additional colors for multiple rods of same type
        self._extra_colors = [
            [0.8, 0.2, 0.8],  # Purple
            [0.8, 0.8, 0.2],  # Yellow
            [0.2, 0.8, 0.8],  # Cyan
            [0.8, 0.4, 0.4],  # Coral
        ]

        # Will be populated after rod creation
        self.rod_infos: list[RodInfo] = []

        # Track sliding constraint configuration
        self.track_enabled = True
        self.track_stiffness = 1.0
        self.track_ignore_tip_count = 0
        self.use_gauss_seidel = True
        self.use_two_sided = True

        # Per-rod insertion values (arclength from track start)
        # Will be initialized after rods are created to have one entry per rod
        self.insertion_speed = 0.5  # units per second
        self.rod_insertions = []  # Populated after rod creation

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

        # Keyboard control mode: True = insertion mode (1/2/9/0), False = movement mode (IJKLUO)
        self.keyboard_insertion_mode = True

        self.simulate_reference = True
        self.simulate_gpu = True
        self.use_cuda_graph = args.use_cuda_graph
        self.use_parallel_kernels = True  # Toggle between parallel and sequential GPU kernels
        self.use_batched_step = True  # Toggle batched step for multiple rods
        self.sync_batched_arrays = True  # Sync between batched arrays and individual rods
        self.use_batched_cuda_graph = False  # CUDA graph for batched step (per substep)
        self.use_frame_cuda_graph = False  # CUDA graph for entire frame (all substeps)
        self.enable_batched_timers = False  # Timing instrumentation for batched step
        self._frame_graph = None
        self._frame_graph_params = None
        # Warp profiling
        self.enable_warp_profiling = False  # Enable wp.ScopedTimer profiling
        self.warp_profile_cuda = False  # Include CUDA kernel timing (cuda_filter)
        self._warp_profile_frame_count = 0  # Count frames for periodic reporting
        self.use_iterative_refinement = False
        self.iterative_refinement_iters = 2
        self._force_sync_reference = True
        self._force_sync_gpu = True
        # Dirty flags to avoid HtoD transfers when data hasn't changed
        self._collision_masses_dirty = True  # Force initial sync
        self._ref_rod_dirty = True  # Force initial sync
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
        self._mode_key_was_down = False

        # Only load DLL if needed (NUMPY or DLL solver types require it)
        self._needs_dll = any(st in (SolverType.NUMPY, SolverType.DLL) for st in self.solver_types)
        if self._needs_dll:
            self.lib = DefKitDirectLibrary(args.dll_path, args.calling_convention)
            self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
            if not self.supports_non_banded:
                self.use_banded = True
        else:
            self.lib = None
            self.supports_non_banded = True  # Warp solver supports both modes

        rod_radius = args.rod_radius if args.rod_radius is not None else args.particle_radius

        # Build configs for all rods
        all_configs = build_rod_configs(args)

        # Track color usage per solver type
        solver_type_counts = {SolverType.NUMPY: 0, SolverType.WARP: 0, SolverType.DLL: 0}

        # Create rods based on solver types
        self.numpy_rods: list[NumpyDirectRodState] = []
        self.dll_rods: list[DefKitDirectRodState] = []
        warp_configs: list[RodConfig] = []
        warp_indices: list[int] = []  # Track which indices are Warp rods

        for idx, (solver_type, config) in enumerate(zip(self.solver_types, all_configs, strict=True)):
            # Determine color for this rod
            type_count = solver_type_counts[solver_type]
            if type_count == 0:
                color = self._rod_colors[solver_type]
            else:
                color = self._extra_colors[(type_count - 1) % len(self._extra_colors)]
            solver_type_counts[solver_type] += 1

            if solver_type == SolverType.NUMPY:
                rod = NumpyDirectRodState(
                    lib=self.lib,
                    num_points=config.num_points,
                    segment_length=config.segment_length,
                    mass=config.particle_mass,
                    particle_height=config.particle_height,
                    rod_radius=config.rod_radius,
                    bend_stiffness=config.bend_stiffness,
                    twist_stiffness=config.twist_stiffness,
                    rest_bend_d1=config.rest_bend_d1,
                    rest_bend_d2=config.rest_bend_d2,
                    rest_twist=config.rest_twist,
                    young_modulus=config.young_modulus,
                    torsion_modulus=config.torsion_modulus,
                    gravity=config.gravity,
                    lock_root_rotation=config.lock_root_rotation,
                    use_banded=self.use_banded,
                )
                self.numpy_rods.append(rod)
                self.rod_infos.append(
                    RodInfo(
                        rod=rod,
                        solver_type=solver_type,
                        config=config,
                        offset=np.zeros(3, dtype=np.float32),  # Will be updated later
                        color=color,
                        mesh_radius=config.rod_radius,
                    )
                )
            elif solver_type == SolverType.DLL:
                rod = DefKitDirectRodState(
                    lib=self.lib,
                    num_points=config.num_points,
                    segment_length=config.segment_length,
                    mass=config.particle_mass,
                    particle_height=config.particle_height,
                    rod_radius=config.rod_radius,
                    bend_stiffness=config.bend_stiffness,
                    twist_stiffness=config.twist_stiffness,
                    rest_bend_d1=config.rest_bend_d1,
                    rest_bend_d2=config.rest_bend_d2,
                    rest_twist=config.rest_twist,
                    young_modulus=config.young_modulus,
                    torsion_modulus=config.torsion_modulus,
                    gravity=config.gravity,
                    lock_root_rotation=config.lock_root_rotation,
                    use_banded=self.use_banded,
                )
                self.dll_rods.append(rod)
                self.rod_infos.append(
                    RodInfo(
                        rod=rod,
                        solver_type=solver_type,
                        config=config,
                        offset=np.zeros(3, dtype=np.float32),
                        color=color,
                        mesh_radius=config.rod_radius,
                    )
                )
            else:  # SolverType.WARP
                warp_configs.append(config)
                warp_indices.append(idx)
                # Rod info will be added after GPU state creation
                self.rod_infos.append(None)  # Placeholder

        # Create Warp GPU rods if any
        self.gpu_batch = None
        self.gpu_state = None
        self.gpu_solver = None
        self.warp_rod_indices = warp_indices

        if warp_configs:
            self.gpu_batch = RodBatch(warp_configs)
            self.gpu_state = self.gpu_batch.create_state(
                device=wp.get_device(),
                use_banded=self.use_banded,
                use_cuda_graph=self.use_cuda_graph,
            )
            self.use_cuda_graph = self.gpu_state.set_use_cuda_graph(self.use_cuda_graph)
            self.use_batched_step = self.gpu_state.set_use_batched_step(self.use_batched_step)
            self.use_batched_cuda_graph = self.gpu_state.set_batched_cuda_graph(self.use_batched_cuda_graph)
            self.gpu_state.set_batched_timers(self.enable_batched_timers)
            self.gpu_solver = CosseratXPBDSolver(
                self.gpu_batch,
                linear_damping=self.linear_damping,
                angular_damping=self.angular_damping,
            )

            # Fill in Warp rod infos
            for gpu_idx, global_idx in enumerate(warp_indices):
                config = warp_configs[gpu_idx]
                rod = self.gpu_state.rods[gpu_idx]
                type_count = sum(1 for i in range(global_idx) if self.solver_types[i] == SolverType.WARP)
                if type_count == 0:
                    color = self._rod_colors[SolverType.WARP]
                else:
                    color = self._extra_colors[(type_count - 1) % len(self._extra_colors)]
                self.rod_infos[global_idx] = RodInfo(
                    rod=rod,
                    solver_type=SolverType.WARP,
                    config=config,
                    offset=np.zeros(3, dtype=np.float32),
                    color=color,
                    mesh_radius=config.rod_radius,
                )

        # For backward compatibility, set ref_rod to first rod
        if self.rod_infos:
            self.ref_rod = self.rod_infos[0].rod
        elif self._needs_dll:
            # Fallback: create a default NumPy rod (only when DLL is available)
            self.ref_rod = NumpyDirectRodState(
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
        else:
            raise RuntimeError("No rods configured. Specify at least one solver via --rod-solvers.")

        # Initialize per-rod insertion values (all start at 0)
        self.rod_insertions = [0.0] * len(self.rod_infos)

        target_last_pos = np.array([-3.283308, -0.50000024, 1.6833224], dtype=np.float32)
        current_last_pos = self.ref_rod.positions[-1, 0:3].astype(np.float32)
        self.mesh_offset = target_last_pos - current_last_pos
        self._update_offsets()

        models_dir = _resolve_models_dir()
        usd_path = os.path.join(models_dir, "AortaWithVesselsStatic.usdc")
        if not os.path.isfile(usd_path):
            raise FileNotFoundError(
                "Unable to find AortaWithVessels.usdc. "
                f"Expected at '{usd_path}'. If you're using a pointer file at "
                "'cosserat_codex/gpu_warp/models', ensure it targets the cosserat models folder."
            )
        usd_stage = Usd.Stage.Open(usd_path)
        # mesh_prim = usd_stage.GetPrimAtPath("/root/A4009/A4007/Xueguan_rudong/Dynamic_vessels/Mesh")
        mesh_prim = usd_stage.GetPrimAtPath("/root/Mesh/Mesh_004")
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

        # Add particles for all rods
        self._rod_point_starts = []
        self._rod_edge_starts = []
        point_start = 0
        edge_start = 0

        for rod_info in self.rod_infos:
            rod = rod_info.rod
            config = rod_info.config
            self._rod_point_starts.append(point_start)
            self._rod_edge_starts.append(edge_start)

            for i in range(rod.num_points):
                mass = 0.0 if i == 0 else config.particle_mass
                pos = tuple(rod.positions[i, 0:3] + rod_info.offset)
                builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)

            point_start += rod.num_points
            edge_start += max(0, rod.num_points - 1)

        # Backward compatibility: _gpu_point_starts and _gpu_edge_starts for Warp rods
        self._gpu_point_starts = []
        self._gpu_edge_starts = []
        for global_idx in self.warp_rod_indices:
            self._gpu_point_starts.append(self._rod_point_starts[global_idx])
            self._gpu_edge_starts.append(self._rod_edge_starts[global_idx])

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Segment colors per rod (using rod_info colors)
        self._segment_colors_per_rod = []
        for rod_info in self.rod_infos:
            num_edges = max(0, rod_info.rod.num_points - 1)
            colors = np.tile(np.array(rod_info.color, dtype=np.float32), (num_edges, 1))
            self._segment_colors_per_rod.append(colors)

        # Backward compatibility: ref and gpu segment colors
        if self.rod_infos:
            self._ref_segment_colors = self._segment_colors_per_rod[0]
        else:
            self._ref_segment_colors = np.zeros((0, 3), dtype=np.float32)

        if self.gpu_state is not None:
            gpu_edge_count = sum(max(0, rod.num_points - 1) for rod in self.gpu_state.rods)
            self._gpu_segment_colors = np.tile(np.array([1.0, 0.6, 0.2], dtype=np.float32), (gpu_edge_count, 1))
        else:
            self._gpu_segment_colors = np.zeros((0, 3), dtype=np.float32)

        # Initialize rod meshers for tube visualization (one per rod)
        self._rod_meshers = [
            RodMesher(
                num_points=rod_info.rod.num_points,
                radius=rod_info.mesh_radius,
                resolution=self.mesh_resolution,
                smoothing=self.mesh_smoothing,
                device=wp.get_device(),
            )
            for rod_info in self.rod_infos
        ]

        # Backward compatibility
        self._ref_mesher = self._rod_meshers[0] if self._rod_meshers else None
        self._gpu_meshers = [self._rod_meshers[idx] for idx in self.warp_rod_indices]
        self.gpu_mesh_radii = [self.rod_infos[idx].mesh_radius for idx in self.warp_rod_indices]
        self.ref_mesh_radius = self.rod_infos[0].mesh_radius if self.rod_infos else args.particle_radius
        self.ref_mesh_color = self.rod_infos[0].color if self.rod_infos else [0.2, 0.6, 1.0]

        device = self.model.device

        # Allocate per-rod warp arrays for positions and segments
        self._rod_positions_wp = []
        self._rod_velocities_wp = []
        self._rod_segment_starts_wp = []
        self._rod_segment_ends_wp = []
        self._rod_segment_colors_wp = []

        for idx, rod_info in enumerate(self.rod_infos):
            rod = rod_info.rod
            num_edges = max(0, rod.num_points - 1)
            self._rod_positions_wp.append(wp.zeros(rod.num_points, dtype=wp.vec3, device=device))
            self._rod_velocities_wp.append(wp.zeros(rod.num_points, dtype=wp.vec3, device=device))
            self._rod_segment_starts_wp.append(wp.zeros(num_edges, dtype=wp.vec3, device=device))
            self._rod_segment_ends_wp.append(wp.zeros(num_edges, dtype=wp.vec3, device=device))
            self._rod_segment_colors_wp.append(
                wp.array(self._segment_colors_per_rod[idx], dtype=wp.vec3, device=device)
            )

        # Backward compatibility
        self._ref_positions_wp = self._rod_positions_wp[0] if self._rod_positions_wp else None
        self._ref_velocities_wp = self._rod_velocities_wp[0] if self._rod_velocities_wp else None
        self._ref_segment_starts_wp = self._rod_segment_starts_wp[0] if self._rod_segment_starts_wp else None
        self._ref_segment_ends_wp = self._rod_segment_ends_wp[0] if self._rod_segment_ends_wp else None
        self._ref_segment_colors_wp = self._rod_segment_colors_wp[0] if self._rod_segment_colors_wp else None

        # Compute total GPU edge count for backward compatibility
        gpu_edge_count = 0
        if self.gpu_state is not None:
            gpu_edge_count = sum(max(0, rod.num_points - 1) for rod in self.gpu_state.rods)
        self._gpu_segment_starts_wp = wp.zeros(max(1, gpu_edge_count), dtype=wp.vec3, device=device)
        self._gpu_segment_ends_wp = wp.zeros(max(1, gpu_edge_count), dtype=wp.vec3, device=device)
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

        # Per-particle visualization colors and radii (updated each frame based on rod settings)
        self._particle_colors_np = np.zeros((total_particles, 3), dtype=np.float32)
        self._particle_radii_np = np.zeros(total_particles, dtype=np.float32)
        self._update_particle_visuals()

        # Previous positions for collision velocity update
        self._rod_positions_prev = [
            np.zeros((rod_info.rod.num_points, 3), dtype=np.float32) for rod_info in self.rod_infos
        ]
        self._ref_positions_prev = (
            self._rod_positions_prev[0] if self._rod_positions_prev else np.zeros((0, 3), dtype=np.float32)
        )
        self._gpu_positions_prev = []
        if self.gpu_state is not None:
            self._gpu_positions_prev = [
                wp.zeros(rod.num_points, dtype=wp.vec3, device=device) for rod in self.gpu_state.rods
            ]

        # Pre-allocate GPU arrays for batched sync operations (reduces kernel launch overhead)
        # These are used by _sync_state_from_rods and constraint copy-back when batched_arrays exist
        self._gpu_offsets_wp = None
        self._gpu_particle_start_indices_wp = None
        if self.gpu_state is not None and self.gpu_state.batched_arrays is not None:
            # Build offsets array for GPU rods (vec3 per rod)
            gpu_offsets_np = np.array(
                [[o[0], o[1], o[2]] for o in self.gpu_offsets], dtype=np.float32
            )
            self._gpu_offsets_wp = wp.array(gpu_offsets_np, dtype=wp.vec3, device=device)

            # Build particle start indices for mapping global particle index to rod index
            # This is stored in batched_arrays already as particle_rod_id_wp

        self._sync_state_from_rods(force=True)
        self._update_gravity()

        # Store root base orientations for all rods
        self._rod_root_base_orientations = [rod_info.rod.orientations[0].copy() for rod_info in self.rod_infos]
        self._ref_root_base_orientation = (
            self._rod_root_base_orientations[0] if self._rod_root_base_orientations else None
        )
        self._gpu_root_base_orientations = []
        if self.gpu_state is not None:
            self._gpu_root_base_orientations = [rod.orientations[0].copy() for rod in self.gpu_state.rods]

        # Initialize track start/end from rod first/last particle positions (for all rods)
        self._rod_track_starts = []
        self._rod_track_ends = []
        for rod_info in self.rod_infos:
            positions = rod_info.rod.positions[:, 0:3].astype(np.float32)
            self._rod_track_starts.append(positions[0].copy())
            self._rod_track_ends.append(positions[-1].copy())

        # Backward compatibility
        self._ref_track_start = self._rod_track_starts[0] if self._rod_track_starts else np.zeros(3, dtype=np.float32)
        self._ref_track_end = self._rod_track_ends[0] if self._rod_track_ends else np.zeros(3, dtype=np.float32)
        self._gpu_track_starts = [self._rod_track_starts[idx] for idx in self.warp_rod_indices]
        self._gpu_track_ends = [self._rod_track_ends[idx] for idx in self.warp_rod_indices]

    def __del__(self):
        # Clean up all rods
        if hasattr(self, "rod_infos"):
            for rod_info in self.rod_infos:
                if rod_info and hasattr(rod_info.rod, "destroy"):
                    rod_info.rod.destroy()
        if hasattr(self, "gpu_state") and self.gpu_state is not None:
            self.gpu_state.destroy()

    def _update_offsets(self):
        """Update world offsets for all rods based on spacing."""
        # Compute X offsets for spacing rods along X axis
        x_offsets = compute_linear_offsets(len(self.rod_infos), self.rod_spacing)

        for rod_info, x_offset in zip(self.rod_infos, x_offsets, strict=True):
            rod_info.offset = self.mesh_offset + np.array([x_offset, 0.0, 0.0], dtype=np.float32)

        # Backward compatibility: ref_offset is first rod's offset
        self.ref_offset = self.rod_infos[0].offset if self.rod_infos else self.mesh_offset.copy()

        # Backward compatibility: gpu_offsets for Warp rods only
        self.gpu_base_offset = self.mesh_offset.copy()
        self.gpu_offsets = self._build_gpu_offsets()

        # Update GPU offsets warp array if it exists (for batched sync)
        if hasattr(self, "_gpu_offsets_wp") and self._gpu_offsets_wp is not None and self.gpu_offsets:
            gpu_offsets_np = np.array(
                [[o[0], o[1], o[2]] for o in self.gpu_offsets], dtype=np.float32
            )
            self._gpu_offsets_wp.assign(wp.array(gpu_offsets_np, dtype=wp.vec3, device=self.model.device))

    def _build_gpu_offsets(self):
        """Build offsets list for Warp GPU rods (backward compatibility)."""
        offsets = []
        for idx in self.warp_rod_indices:
            if idx < len(self.rod_infos):
                offsets.append(self.rod_infos[idx].offset)
        return offsets

    def _update_particle_visuals(self):
        """Update per-particle colors and radii based on rod settings."""
        for idx, rod_info in enumerate(self.rod_infos):
            start = self._rod_point_starts[idx]
            end = start + rod_info.rod.num_points
            # Set particle colors for this rod
            self._particle_colors_np[start:end] = rod_info.particle_color
            # Set particle radii for this rod
            self._particle_radii_np[start:end] = rod_info.particle_radius

    def _update_gravity(self):
        if self.gravity_enabled:
            gravity = self.base_gravity * self.gravity_scale
        else:
            gravity = np.zeros(3, dtype=np.float32)

        # Update gravity for all rods
        for rod_info in self.rod_infos:
            rod_info.rod.set_gravity(gravity)

        # Also update GPU batch state if it exists
        if self.gpu_state is not None:
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

        if sync_gpu and self.gpu_state is not None:
            batched = self.gpu_state.batched_arrays
            # Use batched kernel if batched arrays exist and GPU offsets warp array is ready
            if batched is not None and self._gpu_offsets_wp is not None:
                # Ensure batched arrays are synced from individual rods first
                batched.sync_from_rods(self.gpu_state.rods)
                total_gpu_points = batched.total_points
                # Single kernel launch for all GPU rods - positions
                wp.launch(
                    _warp_copy_with_offset_batched,
                    dim=total_gpu_points,
                    inputs=[
                        batched.positions_wp,
                        self._gpu_offsets_wp,
                        batched.rod_offsets_wp,
                        batched.particle_rod_id_wp,
                        self.state.particle_q,
                    ],
                    device=self.model.device,
                )
                # Single kernel launch for all GPU rods - velocities (no offset)
                # Note: velocities don't need offset, so we use a zeroed offsets array
                if not hasattr(self, "_zero_offsets_wp") or self._zero_offsets_wp is None:
                    zero_offsets_np = np.zeros((len(self.gpu_state.rods), 3), dtype=np.float32)
                    self._zero_offsets_wp = wp.array(zero_offsets_np, dtype=wp.vec3, device=self.model.device)
                wp.launch(
                    _warp_copy_with_offset_batched,
                    dim=total_gpu_points,
                    inputs=[
                        batched.velocities_wp,
                        self._zero_offsets_wp,
                        batched.rod_offsets_wp,
                        batched.particle_rod_id_wp,
                        self.state.particle_qd,
                    ],
                    device=self.model.device,
                )
            else:
                # Fall back to per-rod kernel launches
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
        # Skip HtoD transfer if collision masses haven't changed
        if not self._collision_masses_dirty:
            return

        # Update collision inverse masses for all rods
        for idx, rod_info in enumerate(self.rod_infos):
            rod = rod_info.rod
            start = self._rod_point_starts[idx]
            end = start + rod.num_points
            self._collision_inv_masses_np[start:end] = rod.inv_masses
        self._collision_inv_masses_wp.assign(
            wp.array(self._collision_inv_masses_np, dtype=wp.float32, device=self.model.device)
        )
        self._collision_masses_dirty = False

    def _apply_mesh_collisions(self, dt: float, skip_initial_sync: bool = False):
        """Apply mesh collision constraints to all rods.

        Args:
            dt: Time step for velocity update.
            skip_initial_sync: If True, skip the initial sync from rods to state.
                Use this when a sync was already performed (e.g., in step() before
                the constraint loop) to avoid redundant kernel launches.
        """
        if not hasattr(self, "vessel_bvh"):
            return

        # Store previous positions for all rods
        for idx, rod_info in enumerate(self.rod_infos):
            rod = rod_info.rod
            if rod_info.solver_type == SolverType.WARP:
                wp.copy(self._gpu_positions_prev[self.warp_rod_indices.index(idx)], rod.positions_wp)
            else:
                self._rod_positions_prev[idx][:] = rod.positions[:, 0:3]

        self._update_collision_inv_masses()
        if not skip_initial_sync:
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

        # Copy back to all rods
        # Use batched kernel for GPU rods if batched arrays are available
        batched = self.gpu_state.batched_arrays if self.gpu_state is not None else None
        if batched is not None and self._gpu_offsets_wp is not None:
            # Single kernel launch for all GPU rods - copy back positions
            total_gpu_points = batched.total_points
            wp.launch(
                _warp_copy_from_offset_batched,
                dim=total_gpu_points,
                inputs=[
                    self.state.particle_q,
                    self._gpu_offsets_wp,
                    batched.rod_offsets_wp,
                    batched.particle_rod_id_wp,
                    batched.positions_wp,
                ],
                device=self.model.device,
            )
            # Also update predicted positions (same as positions after collision)
            wp.launch(
                _warp_copy_from_offset_batched,
                dim=total_gpu_points,
                inputs=[
                    self.state.particle_q,
                    self._gpu_offsets_wp,
                    batched.rod_offsets_wp,
                    batched.particle_rod_id_wp,
                    batched.predicted_positions_wp,
                ],
                device=self.model.device,
            )
            # Sync batched arrays back to individual rods
            batched.sync_to_rods(self.gpu_state.rods)

            # Update velocities for GPU rods (still per-rod for now due to prev positions tracking)
            for idx, rod_info in enumerate(self.rod_infos):
                if rod_info.solver_type == SolverType.WARP:
                    warp_idx = self.warp_rod_indices.index(idx)
                    rod = rod_info.rod
                    wp.launch(
                        _warp_update_velocities_from_positions,
                        dim=rod.num_points,
                        inputs=[
                            self._gpu_positions_prev[warp_idx],
                            rod.positions_wp,
                            rod.inv_masses_wp,
                            float(dt),
                            rod.velocities_wp,
                        ],
                        device=self.model.device,
                    )
        else:
            # Fall back to per-rod kernel launches for GPU rods
            for idx, rod_info in enumerate(self.rod_infos):
                if rod_info.solver_type == SolverType.WARP:
                    rod = rod_info.rod
                    offset = rod_info.offset
                    offset_wp = wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                    start = self._rod_point_starts[idx]
                    warp_idx = self.warp_rod_indices.index(idx)
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
                            self._gpu_positions_prev[warp_idx],
                            rod.positions_wp,
                            rod.inv_masses_wp,
                            float(dt),
                            rod.velocities_wp,
                        ],
                        device=self.model.device,
                    )

        # Handle NumPy/DLL rods (always per-rod)
        for idx, rod_info in enumerate(self.rod_infos):
            if rod_info.solver_type in (SolverType.NUMPY, SolverType.DLL):
                rod = rod_info.rod
                offset = rod_info.offset
                offset_wp = wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                start = self._rod_point_starts[idx]
                wp.launch(
                    _warp_copy_from_offset,
                    dim=rod.num_points,
                    inputs=[self.state.particle_q, offset_wp, start, self._rod_positions_wp[idx]],
                    device=self.model.device,
                )
                new_positions = self._rod_positions_wp[idx].numpy()
                rod.positions[:, 0:3] = new_positions
                rod.predicted_positions[:, 0:3] = new_positions

                # Update velocities
                vel = (new_positions - self._rod_positions_prev[idx]) / float(dt)
                rod.velocities[:, 0:3] = vel
                rod.velocities[rod.inv_masses == 0.0, 0:3] = 0.0

        self._force_sync_reference = True
        self._force_sync_gpu = True

    def _apply_track_constraint(self, skip_initial_sync: bool = False):
        """Apply track sliding constraint to keep particles on a line segment.

        This applies to all rods regardless of solver type. Each rod has:
        - A track (line segment from first to last initial position)
        - An insertion value (how far along the track the root is positioned)

        Args:
            skip_initial_sync: If True, skip the initial sync from rods to state.
                Use this when a sync was already performed (e.g., in step() before
                the constraint loop) to avoid redundant kernel launches.
        """
        if not self.track_enabled:
            return

        self._update_collision_inv_masses()
        if not skip_initial_sync:
            self._sync_state_from_rods(force=True)

        # Apply track constraint to all rods
        for idx, rod_info in enumerate(self.rod_infos):
            rod = rod_info.rod
            offset = rod_info.offset
            num_constrained = rod.num_points - self.track_ignore_tip_count
            if num_constrained <= 0:
                continue

            # Get track endpoints for this rod
            track_start = self._rod_track_starts[idx] + offset
            track_end = self._rod_track_ends[idx] + offset
            track_start_wp = wp.vec3(float(track_start[0]), float(track_start[1]), float(track_start[2]))
            track_end_wp = wp.vec3(float(track_end[0]), float(track_end[1]), float(track_end[2]))

            # Get particle range in the global state
            start = self._rod_point_starts[idx]
            end = start + num_constrained

            # Apply insertion (set root position on track)
            insertion = self.rod_insertions[idx] if idx < len(self.rod_insertions) else 0.0

            if rod_info.solver_type == SolverType.WARP:
                # For Warp rods, use GPU kernel
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
            else:
                # For NumPy/DLL rods, apply insertion on CPU
                track_dir = track_end - track_start
                track_len = np.linalg.norm(track_dir)
                if track_len > 1e-6:
                    track_dir = track_dir / track_len
                    # Clamp insertion to track length
                    clamped_insertion = min(insertion, track_len)
                    root_pos = track_start + track_dir * clamped_insertion
                    rod.positions[0, 0:3] = root_pos - offset  # Store without offset
                    rod.predicted_positions[0, 0:3] = root_pos - offset
                    rod.velocities[0, 0:3] = 0.0

            # Apply track sliding constraint via particle state
            wp.launch(
                _warp_apply_track_sliding,
                dim=num_constrained,
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

            # Copy back to rod state (NumPy/DLL rods only - GPU rods handled in batch below)
            if rod_info.solver_type in (SolverType.NUMPY, SolverType.DLL):
                offset_wp = wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                wp.launch(
                    _warp_copy_from_offset,
                    dim=rod.num_points,
                    inputs=[self.state.particle_q, offset_wp, start, self._rod_positions_wp[idx]],
                    device=self.model.device,
                )
                positions = self._rod_positions_wp[idx].numpy()
                rod.positions[:, 0:3] = positions
                rod.predicted_positions[:, 0:3] = positions

        # Batch copy-back for GPU rods
        batched = self.gpu_state.batched_arrays if self.gpu_state is not None else None
        if batched is not None and self._gpu_offsets_wp is not None:
            total_gpu_points = batched.total_points
            # Single kernel launch for all GPU rods - copy back positions
            wp.launch(
                _warp_copy_from_offset_batched,
                dim=total_gpu_points,
                inputs=[
                    self.state.particle_q,
                    self._gpu_offsets_wp,
                    batched.rod_offsets_wp,
                    batched.particle_rod_id_wp,
                    batched.positions_wp,
                ],
                device=self.model.device,
            )
            # Also update predicted positions
            wp.launch(
                _warp_copy_from_offset_batched,
                dim=total_gpu_points,
                inputs=[
                    self.state.particle_q,
                    self._gpu_offsets_wp,
                    batched.rod_offsets_wp,
                    batched.particle_rod_id_wp,
                    batched.predicted_positions_wp,
                ],
                device=self.model.device,
            )
            # Sync batched arrays back to individual rods
            batched.sync_to_rods(self.gpu_state.rods)
        else:
            # Fall back to per-rod kernel launches for GPU rods
            for idx, rod_info in enumerate(self.rod_infos):
                if rod_info.solver_type == SolverType.WARP:
                    rod = rod_info.rod
                    offset = rod_info.offset
                    offset_wp = wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                    start = self._rod_point_starts[idx]
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
        """Apply concentric constraint between GPU Warp rods (inner rod stays on outer rod centerline).

        Convention:
          - rods[0] = OUTER rod (catheter) - provides the centerline
          - rods[1] = INNER rod (guidewire) - constrained to stay on outer rod's centerline

        The constraint uses arc-length parametrization with insertion difference to compute
        the exact corresponding point on the outer rod's centerline for each inner rod particle.

        Note: This only works with Warp rods. The first two Warp rods are used.
        """
        if not self.concentric_enabled:
            return

        if self.gpu_state is None or len(self.gpu_state.rods) < 2:
            return

        # rod[0] = outer (catheter), rod[1] = inner (guidewire)
        outer_rod = self.gpu_state.rods[0]
        inner_rod = self.gpu_state.rods[1]

        # Get insertion values for the Warp rods (using their global indices)
        warp_idx_outer = self.warp_rod_indices[0] if len(self.warp_rod_indices) > 0 else 0
        warp_idx_inner = self.warp_rod_indices[1] if len(self.warp_rod_indices) > 1 else 1
        insertion_outer = self.rod_insertions[warp_idx_outer] if warp_idx_outer < len(self.rod_insertions) else 0.0
        insertion_inner = self.rod_insertions[warp_idx_inner] if warp_idx_inner < len(self.rod_insertions) else 0.0

        # insertion_diff = insertion_inner - insertion_outer
        # Positive when inner rod is more inserted (ahead of outer rod)
        insertion_diff = insertion_inner - insertion_outer

        # Get rest lengths arrays
        if hasattr(outer_rod, "rest_lengths_wp") and outer_rod.rest_lengths_wp is not None:
            outer_rest_lengths = outer_rod.rest_lengths_wp
        else:
            if not hasattr(self, "_concentric_outer_rest_lengths"):
                self._concentric_outer_rest_lengths = wp.array(
                    outer_rod.rest_lengths, dtype=wp.float32, device=self.model.device
                )
            outer_rest_lengths = self._concentric_outer_rest_lengths

        if hasattr(inner_rod, "rest_lengths_wp") and inner_rod.rest_lengths_wp is not None:
            inner_rest_lengths = inner_rod.rest_lengths_wp
        else:
            if not hasattr(self, "_concentric_inner_rest_lengths"):
                self._concentric_inner_rest_lengths = wp.array(
                    inner_rod.rest_lengths, dtype=wp.float32, device=self.model.device
                )
            inner_rest_lengths = self._concentric_inner_rest_lengths

        # Launch kernel - iterates over INNER rod particles
        # Uses new v3 implementation with cleaner arc-length parametrization
        wp.launch(
            warp_concentric_constraint_direct,
            dim=inner_rod.num_points,
            inputs=[
                # Outer rod (catheter) - provides the centerline
                outer_rod.positions_wp,
                outer_rod.predicted_positions_wp,
                outer_rod.inv_masses_wp,
                outer_rest_lengths,
                int(outer_rod.num_points),
                # Inner rod (guidewire) - constrained to stay on outer centerline
                inner_rod.positions_wp,
                inner_rod.predicted_positions_wp,
                inner_rod.inv_masses_wp,
                inner_rest_lengths,
                int(inner_rod.num_points),
                # Constraint parameters
                float(insertion_diff),
                float(self.concentric_stiffness),
                float(self.concentric_weight_inner),
                float(self.concentric_weight_outer),
                int(self.concentric_start_particle),
                int(-1),  # end_particle: -1 = all particles
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
                rod.rest_darboux_wp.assign(wp.array(rod.rest_darboux[:, 0:3], dtype=wp.vec3, device=rod.device))

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
            if self.gpu_state is not None and self.gpu_state.rods:
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

        f_down = self.viewer.is_key_down(key.F)
        if f_down and not self._lock_key_was_down:
            # Toggle root lock for all rods
            for rod_info in self.rod_infos:
                rod_info.rod.toggle_root_lock()
        self._lock_key_was_down = f_down

        t_down = self.viewer.is_key_down(key.T)
        if t_down and not self._track_key_was_down:
            self.track_enabled = not self.track_enabled
        self._track_key_was_down = t_down

        c_down = self.viewer.is_key_down(key.C)
        if c_down and not self._concentric_key_was_down:
            warp_rods = self.gpu_state.rods if self.gpu_state else []
            if len(warp_rods) >= 2:
                self.concentric_enabled = not self.concentric_enabled
        self._concentric_key_was_down = c_down

        # Toggle keyboard control mode (Tab key)
        tab_down = self.viewer.is_key_down(key.TAB)
        if tab_down and not self._mode_key_was_down:
            self.keyboard_insertion_mode = not self.keyboard_insertion_mode
        self._mode_key_was_down = tab_down

        if (
            self.viewer.is_key_down(key.PLUS)
            or self.viewer.is_key_down(key.EQUAL)
            or self.viewer.is_key_down(key.NUM_ADD)
        ):
            self.tip_bend_angle += self.tip_bend_speed * self.frame_dt
            self._apply_tip_bend()

        if self.viewer.is_key_down(key.MINUS) or self.viewer.is_key_down(key.NUM_SUBTRACT):
            self.tip_bend_angle -= self.tip_bend_speed * self.frame_dt
            self._apply_tip_bend()

        # Insertion controls for first two rods (any solver type)
        # 1/2/9/0 keys only work in insertion mode; PgUp/PgDn/Home/End always work
        # Insertion moves the root along the track direction
        if len(self.rod_insertions) >= 1:
            rod0_increase = self.viewer.is_key_down(key.PAGEUP)
            rod0_decrease = self.viewer.is_key_down(key.PAGEDOWN)
            if self.keyboard_insertion_mode:
                rod0_increase = rod0_increase or self.viewer.is_key_down(key._2)
                rod0_decrease = rod0_decrease or self.viewer.is_key_down(key._1)
            if rod0_increase:
                distance = self.insertion_speed * self.frame_dt
                self.rod_insertions[0] += distance
                self._apply_insertion_along_track(0, distance)
            if rod0_decrease:
                distance = self.insertion_speed * self.frame_dt
                self.rod_insertions[0] -= distance
                self.rod_insertions[0] = max(0.0, self.rod_insertions[0])
                self._apply_insertion_along_track(0, -distance)

        if len(self.rod_insertions) >= 2:
            rod1_increase = self.viewer.is_key_down(key.HOME)
            rod1_decrease = self.viewer.is_key_down(key.END)
            if self.keyboard_insertion_mode:
                rod1_increase = rod1_increase or self.viewer.is_key_down(key._0)
                rod1_decrease = rod1_decrease or self.viewer.is_key_down(key._9)
            if rod1_increase:
                distance = self.insertion_speed * self.frame_dt
                self.rod_insertions[1] += distance
                self._apply_insertion_along_track(1, distance)
            if rod1_decrease:
                distance = self.insertion_speed * self.frame_dt
                self.rod_insertions[1] -= distance
                self.rod_insertions[1] = max(0.0, self.rod_insertions[1])
                self._apply_insertion_along_track(1, -distance)

        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._reset_key_was_down:
            # Reset all rods
            for rod_info in self.rod_infos:
                rod_info.rod.reset()
            if self.gpu_state is not None:
                self.gpu_state.reset()
            self.root_rotation = 0.0
            self.tip_bend_angle = 0.0
            # Reset all insertions to 0
            self.rod_insertions = [0.0] * len(self.rod_infos)
            self._apply_root_rotation()
            self._apply_tip_bend()
            self.sim_time = 0.0
            self._sync_state_from_rods()
        self._reset_key_was_down = r_down

        dx = 0.0
        dy = 0.0
        dz = 0.0

        # Numpad controls
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

        # IJKLUO controls (only in movement mode, not insertion mode)
        if not self.keyboard_insertion_mode:
            if self.viewer.is_key_down(key.L):
                dx += self.root_move_speed * self.frame_dt
            if self.viewer.is_key_down(key.J):
                dx -= self.root_move_speed * self.frame_dt
            if self.viewer.is_key_down(key.I):
                dy += self.root_move_speed * self.frame_dt
            if self.viewer.is_key_down(key.K):
                dy -= self.root_move_speed * self.frame_dt
            if self.viewer.is_key_down(key.U):
                dz += self.root_move_speed * self.frame_dt
            if self.viewer.is_key_down(key.O):
                dz -= self.root_move_speed * self.frame_dt

        rotation_changed = False
        if self.viewer.is_key_down(key.NUM_7) or self.viewer.is_key_down(key.PERIOD):
            self.root_rotation += self.root_rotate_speed * self.frame_dt
            rotation_changed = True
        if self.viewer.is_key_down(key.NUM_1) or self.viewer.is_key_down(key.COMMA):
            self.root_rotation -= self.root_rotate_speed * self.frame_dt
            rotation_changed = True

        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            self._apply_root_translation(dx, dy, dz)
        if rotation_changed:
            self._apply_root_rotation()

    def _step_gpu_frame_inner(self, sub_dt: float) -> None:
        """Inner GPU frame step - all substeps, suitable for CUDA graph capture.

        This contains only GPU operations that can be captured in a CUDA graph.
        Track/concentric constraints and CPU operations are excluded.
        """
        for _ in range(self.substeps):
            self.gpu_state.step(sub_dt, self.linear_damping, self.angular_damping)
            if self.floor_collision_enabled:
                self.gpu_state.apply_floor_collisions(self.floor_height, self.floor_restitution)

    def _ensure_frame_cuda_graph(self, sub_dt: float) -> None:
        """Ensure frame-level CUDA graph is captured with current parameters."""
        params = (
            float(sub_dt),
            float(self.linear_damping),
            float(self.angular_damping),
            int(self.substeps),
            bool(self.floor_collision_enabled),
            float(self.floor_height),
            float(self.floor_restitution),
            bool(self.gpu_state.sync_batched_arrays) if self.gpu_state else False,
        )
        if self._frame_graph is not None and self._frame_graph_params == params:
            return

        if self.gpu_state is None:
            return

        device = wp.get_device()
        if not device.is_cuda:
            return

        # Temporarily disable per-substep timing during capture
        was_timers = self.gpu_state._enable_batched_timers
        self.gpu_state._enable_batched_timers = False

        try:
            with wp.ScopedCapture(device=device, force_module_load=True) as capture:
                self._step_gpu_frame_inner(sub_dt)
        finally:
            self.gpu_state._enable_batched_timers = was_timers

        self._frame_graph = capture.graph
        self._frame_graph_params = params

    def step(self):
        self._handle_keyboard_input()

        sub_dt = self.frame_dt / max(self.substeps, 1)
        frame_start = time.perf_counter()
        numpy_dll_time = 0.0
        gpu_time = 0.0

        # Update GPU solver damping if it exists
        if self.gpu_solver is not None:
            self.gpu_solver.linear_damping = self.linear_damping
            self.gpu_solver.angular_damping = self.angular_damping

        # Check if we can use frame-level CUDA graph (requires no CPU operations in loop)
        use_frame_graph = (
            self.use_frame_cuda_graph
            and self.simulate_gpu
            and self.gpu_state is not None
            and self.gpu_state.use_batched_step
            and not self.simulate_reference  # No CPU rod simulation
            and self.collision_iterations == 0  # No mesh collisions
            and not self.track_enabled  # No track constraint
            and not self.concentric_enabled  # No concentric constraint
        )

        # Warp profiling setup
        do_warp_profile = self.enable_warp_profiling
        cuda_filter = wp.TIMING_ALL if (do_warp_profile and self.warp_profile_cuda) else None

        with wp.ScopedTimer("frame", active=do_warp_profile, synchronize=do_warp_profile, cuda_filter=cuda_filter):
            if use_frame_graph:
                # Use frame-level CUDA graph for all substeps
                with wp.ScopedTimer("gpu_graph", active=do_warp_profile, synchronize=do_warp_profile):
                    t0 = time.perf_counter()
                    self._ensure_frame_cuda_graph(sub_dt)
                    wp.capture_launch(self._frame_graph)
                    gpu_time = time.perf_counter() - t0
            else:
                # Standard per-substep execution
                for substep_idx in range(self.substeps):
                    # Step NumPy and DLL rods individually
                    if self.simulate_reference:
                        with wp.ScopedTimer("ref_step", active=do_warp_profile):
                            t0 = time.perf_counter()
                            for rod_info in self.rod_infos:
                                if rod_info.solver_type in (SolverType.NUMPY, SolverType.DLL):
                                    rod_info.rod.step(sub_dt, self.linear_damping, self.angular_damping)
                                    if self.floor_collision_enabled:
                                        rod_info.rod.apply_floor_collisions(self.floor_height, self.floor_restitution)
                            numpy_dll_time += time.perf_counter() - t0

                    # Step Warp GPU rods together
                    if self.simulate_gpu and self.gpu_solver is not None and self.gpu_state is not None:
                        with wp.ScopedTimer("gpu_step", active=do_warp_profile, synchronize=do_warp_profile):
                            t0 = time.perf_counter()
                            self.gpu_solver.step(self.gpu_state, self.gpu_state, None, None, sub_dt)
                            if self.floor_collision_enabled:
                                self.gpu_state.apply_floor_collisions(self.floor_height, self.floor_restitution)
                            gpu_time += time.perf_counter() - t0

                    with wp.ScopedTimer("collisions", active=do_warp_profile, synchronize=do_warp_profile):
                        # Perform a single sync before all constraints to avoid redundant syncs
                        # Each constraint function would otherwise sync independently
                        needs_constraints = (
                            self.collision_iterations > 0 or self.track_enabled or self.concentric_enabled
                        )
                        if needs_constraints:
                            self._update_collision_inv_masses()
                            self._sync_state_from_rods(force=True)

                        for _ in range(self.collision_iterations):
                            self._apply_mesh_collisions(sub_dt, skip_initial_sync=True)
                        self._apply_track_constraint(skip_initial_sync=True)
                        self._apply_concentric_constraint()

            with wp.ScopedTimer("sync", active=do_warp_profile, synchronize=do_warp_profile):
                self._sync_state_from_rods()

        # Print CUDA activity report periodically when profiling
        if do_warp_profile and self.warp_profile_cuda:
            self._warp_profile_frame_count += 1
            if self._warp_profile_frame_count >= 60:  # Every ~1 second at 60fps
                self._warp_profile_frame_count = 0

        self.sim_time += self.frame_dt
        frame_end = time.perf_counter()
        self._frame_times.append(frame_end - frame_start)

        # Track timing for NumPy/DLL rods
        has_numpy_dll = any(ri.solver_type in (SolverType.NUMPY, SolverType.DLL) for ri in self.rod_infos)
        if self.simulate_reference and has_numpy_dll:
            self._ref_times.append(numpy_dll_time)

        # Track timing for Warp rods
        has_warp = any(ri.solver_type == SolverType.WARP for ri in self.rod_infos)
        if self.simulate_gpu and has_warp:
            self._gpu_times.append(gpu_time)

    def _apply_root_translation(self, dx: float, dy: float, dz: float):
        delta = np.array([dx, dy, dz], dtype=np.float32)

        # Apply translation to all rods
        for rod_info in self.rod_infos:
            rod = rod_info.rod
            if rod_info.solver_type in (SolverType.NUMPY, SolverType.DLL):
                pos = rod.positions[0, 0:3]
                new_pos = pos + delta
                rod.positions[0, 0:3] = new_pos
                rod.predicted_positions[0, 0:3] = new_pos
                rod.velocities[0, 0:3] = 0.0
            elif rod_info.solver_type == SolverType.WARP:
                rod.apply_root_translation(dx, dy, dz)

        self._force_sync_reference = True
        self._force_sync_gpu = True

    def _apply_insertion_along_track(self, rod_idx: int, distance: float):
        """Move the root of a specific rod along its track direction.

        Args:
            rod_idx: Index of the rod to move.
            distance: Distance to move along the track (positive = towards track_end).
        """
        if rod_idx >= len(self.rod_infos):
            return

        rod_info = self.rod_infos[rod_idx]
        rod = rod_info.rod

        # Get track direction for this rod
        track_start = self._rod_track_starts[rod_idx]
        track_end = self._rod_track_ends[rod_idx]
        track_dir = track_end - track_start
        track_len = np.linalg.norm(track_dir)
        if track_len < 1e-6:
            return
        track_dir = track_dir / track_len

        # Calculate delta in world space
        delta = track_dir * distance

        if rod_info.solver_type in (SolverType.NUMPY, SolverType.DLL):
            pos = rod.positions[0, 0:3]
            new_pos = pos + delta
            rod.positions[0, 0:3] = new_pos
            rod.predicted_positions[0, 0:3] = new_pos
            rod.velocities[0, 0:3] = 0.0
        elif rod_info.solver_type == SolverType.WARP:
            rod.apply_root_translation(float(delta[0]), float(delta[1]), float(delta[2]))

        self._force_sync_reference = True
        self._force_sync_gpu = True

    def _apply_root_rotation(self):
        q_twist = quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.root_rotation)

        # Apply rotation to all rods
        for idx, rod_info in enumerate(self.rod_infos):
            rod = rod_info.rod
            base_orientation = self._rod_root_base_orientations[idx]
            q_new = quat_multiply(base_orientation, q_twist)

            if rod_info.solver_type in (SolverType.NUMPY, SolverType.DLL):
                rod.orientations[0] = q_new
                rod.predicted_orientations[0] = q_new
                rod.prev_orientations[0] = q_new
            elif rod_info.solver_type == SolverType.WARP:
                rod.apply_root_rotation(q_new)

        self._force_sync_reference = True
        self._force_sync_gpu = True

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Disable default particle rendering and use custom per-rod particle colors/radii
        original_show_particles = self.viewer.show_particles
        self.viewer.show_particles = False
        self.viewer.log_state(self.state)
        self.viewer.show_particles = original_show_particles

        # Render particles with per-rod colors and radii
        if original_show_particles and self.model.particle_count > 0:
            self._update_particle_visuals()
            # Convert to warp arrays for GL viewer compatibility
            radii_wp = wp.array(self._particle_radii_np, dtype=wp.float32, device=self.model.device)
            colors_wp = wp.array(self._particle_colors_np, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_points(
                name="/model/particles",
                points=self.state.particle_q,
                radii=radii_wp,
                colors=colors_wp,
                hidden=False,
            )

        # Render rod segments for all rods
        if self.show_segments:
            for idx, rod_info in enumerate(self.rod_infos):
                rod = rod_info.rod
                offset = rod_info.offset
                offset_wp = wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))

                # Get positions array (NumPy/DLL use numpy arrays, Warp has positions_wp)
                if rod_info.solver_type == SolverType.WARP:
                    positions_wp = rod.positions_wp
                else:
                    # Update warp array from numpy
                    positions = rod.positions[:, 0:3].astype(np.float32)
                    self._rod_positions_wp[idx].assign(wp.array(positions, dtype=wp.vec3, device=self.model.device))
                    positions_wp = self._rod_positions_wp[idx]

                # Build segment lines
                wp.launch(
                    _warp_build_segment_lines,
                    dim=rod.num_points - 1,
                    inputs=[
                        positions_wp,
                        offset_wp,
                        0,
                        self._rod_segment_starts_wp[idx],
                        self._rod_segment_ends_wp[idx],
                    ],
                    device=self.model.device,
                )

                # Log lines for this rod
                self.viewer.log_lines(
                    f"/rod_{idx}",
                    self._rod_segment_starts_wp[idx],
                    self._rod_segment_ends_wp[idx],
                    self._rod_segment_colors_wp[idx],
                )
        else:
            for idx in range(len(self.rod_infos)):
                self.viewer.log_lines(f"/rod_{idx}", None, None, None)

        # Render directors for all rods
        if self.show_directors:
            for idx, rod_info in enumerate(self.rod_infos):
                rod = rod_info.rod
                offset = rod_info.offset

                # Get positions and orientations as numpy
                if rod_info.solver_type == SolverType.WARP:
                    positions = rod.positions_numpy().astype(np.float32)
                    orientations = rod.orientations_numpy().astype(np.float32)
                else:
                    positions = rod.positions[:, 0:3].astype(np.float32)
                    orientations = rod.orientations.astype(np.float32)

                starts, ends, colors = build_director_lines(positions, orientations, offset, self.director_scale)
                self.viewer.log_lines(
                    f"/directors_{idx}",
                    wp.array(starts, dtype=wp.vec3, device=self.model.device),
                    wp.array(ends, dtype=wp.vec3, device=self.model.device),
                    wp.array(colors, dtype=wp.vec3, device=self.model.device),
                )
        else:
            for idx in range(len(self.rod_infos)):
                self.viewer.log_lines(f"/directors_{idx}", None, None, None)

        # Render rod tube meshes (only update when visible)
        if self.show_rod_mesh:
            for idx, rod_info in enumerate(self.rod_infos):
                rod = rod_info.rod
                offset = rod_info.offset
                mesher = self._rod_meshers[idx]

                # Update mesher radius from rod_info
                mesher.set_radius(rod_info.mesh_radius)

                # Get positions as numpy
                if rod_info.solver_type == SolverType.WARP:
                    positions = rod.positions_numpy().astype(np.float32) + offset
                else:
                    positions = rod.positions[:, 0:3].astype(np.float32) + offset

                mesher.update_numpy(positions)
                verts_wp, indices_wp, normals_wp, uvs_wp = mesher.get_warp_arrays()
                self.viewer.log_mesh(
                    f"/rod_mesh_{idx}",
                    verts_wp,
                    indices_wp,
                    normals_wp,
                    uvs_wp,
                    hidden=False,
                )
        else:
            # Hide meshes without updating them (use existing arrays)
            for idx, mesher in enumerate(self._rod_meshers):
                verts_wp, indices_wp, normals_wp, uvs_wp = mesher.get_warp_arrays()
                self.viewer.log_mesh(
                    f"/rod_mesh_{idx}",
                    verts_wp,
                    indices_wp,
                    normals_wp,
                    uvs_wp,
                    hidden=True,
                )

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Direct Cosserat Rod Simulation")
        ui.text(f"Rods: {len(self.rod_infos)}")

        # Show solver types for each rod
        solver_counts = {SolverType.NUMPY: 0, SolverType.WARP: 0, SolverType.DLL: 0}
        for ri in self.rod_infos:
            solver_counts[ri.solver_type] += 1
        solver_summary = ", ".join(f"{st.value}: {cnt}" for st, cnt in solver_counts.items() if cnt > 0)
        ui.text(f"Solvers: {solver_summary}")
        ui.separator()

        changed_ref, self.simulate_reference = ui.checkbox("Simulate NumPy/DLL rods", self.simulate_reference)
        changed_gpu, self.simulate_gpu = ui.checkbox("Simulate Warp rods", self.simulate_gpu)

        # CUDA graph only available if we have Warp rods
        if self.gpu_state is not None:
            changed_graph, self.use_cuda_graph = ui.checkbox("Use CUDA Graph (Warp)", self.use_cuda_graph)
            if changed_graph:
                self.use_cuda_graph = self.gpu_state.set_use_cuda_graph(self.use_cuda_graph)
            changed_parallel, self.use_parallel_kernels = ui.checkbox(
                "Use Parallel Kernels (Warp)", self.use_parallel_kernels
            )
            if changed_parallel:
                self.gpu_state.set_parallel_kernels(self.use_parallel_kernels)
            changed_batched, self.use_batched_step = ui.checkbox("Use Batched Step (Warp)", self.use_batched_step)
            if changed_batched:
                self.use_batched_step = self.gpu_state.set_use_batched_step(self.use_batched_step)
            changed_sync, self.sync_batched_arrays = ui.checkbox("Sync Batched Arrays", self.sync_batched_arrays)
            if changed_sync:
                self.gpu_state.sync_batched_arrays = self.sync_batched_arrays
            changed_batched_graph, self.use_batched_cuda_graph = ui.checkbox(
                "Batched CUDA Graph", self.use_batched_cuda_graph
            )
            if changed_batched_graph:
                self.use_batched_cuda_graph = self.gpu_state.set_batched_cuda_graph(self.use_batched_cuda_graph)
            changed_batched_timers, self.enable_batched_timers = ui.checkbox(
                "Batched Timers", self.enable_batched_timers
            )
            if changed_batched_timers:
                self.gpu_state.set_batched_timers(self.enable_batched_timers)
            changed_frame_graph, self.use_frame_cuda_graph = ui.checkbox(
                "Frame CUDA Graph (all substeps)", self.use_frame_cuda_graph
            )
            if changed_frame_graph and self.use_frame_cuda_graph:
                # Invalidate the graph when enabled so it gets recaptured
                self._frame_graph = None
                self._frame_graph_params = None
        # Warp profiling options (always visible)
        ui.separator()
        ui.text("Warp Profiling")
        _changed, self.enable_warp_profiling = ui.checkbox("Enable Warp Profiling", self.enable_warp_profiling)
        if self.enable_warp_profiling:
            _changed, self.warp_profile_cuda = ui.checkbox("Profile CUDA Activities", self.warp_profile_cuda)
        if changed_ref or changed_gpu:
            self._frame_times.clear()
            self._ref_times.clear()
            self._gpu_times.clear()

        _changed, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _changed, self.collision_iterations = ui.slider_int("Collision Iterations", self.collision_iterations, 0, 8)
        _changed, self.linear_damping = ui.slider_float("Linear Damping", self.linear_damping, 0.0, 0.05)
        _changed, self.angular_damping = ui.slider_float("Angular Damping", self.angular_damping, 0.0, 0.05)

        ui.separator()
        offset_changed, self.compare_offset = ui.slider_float("Compare Offset", self.compare_offset, 0.0, 5.0)
        if offset_changed:
            self._update_offsets()
            self._sync_state_from_rods(force=True)

        ui.separator()
        changed_bend_k, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        if changed_bend_k or changed_twist_k:
            for rod_info in self.rod_infos:
                rod_info.rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)
            if self.gpu_state is not None:
                self.gpu_state.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)

        ui.separator()
        changed_rest_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_rest_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_rest_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_rest_d1 or changed_rest_d2 or changed_rest_twist:
            for rod_info in self.rod_infos:
                rod_info.rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)
            if self.gpu_state is not None:
                self.gpu_state.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)

        ui.separator()
        changed_E, self.young_modulus_scale = ui.slider_float(
            "Young Modulus (1e6)", self.young_modulus_scale, 0.1, 1000.0
        )
        changed_G, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Modulus (1e6)", self.torsion_modulus_scale, 0.1, 1000.0
        )
        if changed_E or changed_G:
            for rod_info in self.rod_infos:
                rod_info.rod.young_modulus = self.young_modulus_scale * 1.0e6
                rod_info.rod.torsion_modulus = self.torsion_modulus_scale * 1.0e6

        changed_seg, self.segment_length = ui.slider_float("Segment Length", self.segment_length, 0.01, 0.2)
        if changed_seg:
            for rod_info in self.rod_infos:
                rod = rod_info.rod
                rod.rest_lengths[:] = self.segment_length
                if hasattr(rod, "rest_lengths_wp") and rod.num_edges > 0:
                    rod.rest_lengths_wp.assign(wp.array(rod.rest_lengths, dtype=wp.float32, device=rod.device))

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

        # Show insertion controls for first two rods (any solver type)
        if len(self.rod_insertions) >= 1:
            solver_0 = self.rod_infos[0].solver_type.value.upper() if self.rod_infos else "?"
            _changed, self.rod_insertions[0] = ui.slider_float(
                f"Rod 0 ({solver_0}) Insertion", self.rod_insertions[0], 0.0, 5.0
            )
        if len(self.rod_insertions) >= 2:
            solver_1 = self.rod_infos[1].solver_type.value.upper() if len(self.rod_infos) > 1 else "?"
            _changed, self.rod_insertions[1] = ui.slider_float(
                f"Rod 1 ({solver_1}) Insertion", self.rod_insertions[1], 0.0, 5.0
            )
            insertion_diff = self.rod_insertions[0] - self.rod_insertions[1]
            ui.text(f"Insertion Diff: {insertion_diff:.3f}")

        ui.separator()
        ui.text("Concentric Constraint (Inner stays on Outer centerline)")
        ui.text("  Rod 0 = Outer (catheter), Rod 1 = Inner (guidewire)")
        warp_rods = self.gpu_state.rods if self.gpu_state else []
        if len(warp_rods) >= 2:
            _changed, self.concentric_enabled = ui.checkbox("Enable Concentric", self.concentric_enabled)
            _changed, self.concentric_stiffness = ui.slider_float(
                "Concentric Stiffness", self.concentric_stiffness, 0.0, 1.0
            )
            _changed, self.concentric_weight_inner = ui.slider_float(
                "Inner Weight (Rod 1)", self.concentric_weight_inner, 0.0, 1.0
            )
            _changed, self.concentric_weight_outer = ui.slider_float(
                "Outer Weight (Rod 0)", self.concentric_weight_outer, 0.0, 1.0
            )
            _changed, self.concentric_start_particle = ui.slider_int(
                "Start Particle (Inner)", self.concentric_start_particle, 0, 20
            )
            # Show insertion diff for debugging
            warp_idx_outer = self.warp_rod_indices[0] if len(self.warp_rod_indices) > 0 else 0
            warp_idx_inner = self.warp_rod_indices[1] if len(self.warp_rod_indices) > 1 else 1
            ins_outer = self.rod_insertions[warp_idx_outer] if warp_idx_outer < len(self.rod_insertions) else 0.0
            ins_inner = self.rod_insertions[warp_idx_inner] if warp_idx_inner < len(self.rod_insertions) else 0.0
            ui.text(f"  Insertion diff (inner-outer): {ins_inner - ins_outer:.3f}")
        else:
            ui.text("(Requires 2+ Warp rods)")

        ui.separator()
        ui.text("Bendable Tip")

        changed_tip_angle, self.tip_bend_angle = ui.slider_float("Tip Bend Angle (rad)", self.tip_bend_angle, -1.5, 1.5)
        changed_tip_edges, self.tip_num_edges = ui.slider_int("Tip Edges Affected", self.tip_num_edges, 1, 10)
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
            "Split Thomas",
            "Block Jacobi",
            "Banded Cholesky",
        ]
        gpu_backend_values = [
            DIRECT_SOLVE_WARP_BLOCK_THOMAS,
            DIRECT_SOLVE_WARP_SPLIT_THOMAS,
            DIRECT_SOLVE_WARP_BLOCK_JACOBI,
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
                            rod.set_iterative_refinement(self.use_iterative_refinement, self.iterative_refinement_iters)
                elif current_gpu_backend == DIRECT_SOLVE_WARP_SPLIT_THOMAS:
                    ui.text("  Using Warp split 3x3 Thomas (stretch + darboux)")
                elif current_gpu_backend == DIRECT_SOLVE_WARP_BLOCK_JACOBI:
                    ui.text("  Using Warp block Jacobi (parallel 6x6 blocks)")
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
        _changed, self.show_rod_mesh = ui.checkbox("Show Rod Mesh", self.show_rod_mesh)
        _changed, self.viewer.show_particles = ui.checkbox("Show Particles", self.viewer.show_particles)

        # Per-rod particle settings (always show when particles are visible)
        if self.viewer.show_particles:
            ui.separator()
            ui.text("Particle Settings (per rod)")
            for idx, rod_info in enumerate(self.rod_infos):
                solver_name = rod_info.solver_type.value.upper()
                ui.text(f"  Rod {idx} ({solver_name})")
                _changed, rod_info.particle_radius = ui.slider_float(
                    f"  Rod{idx} Particle Radius", rod_info.particle_radius, 0.001, 0.05
                )
                # Color sliders for R, G, B
                _changed_r, rod_info.particle_color[0] = ui.slider_float(
                    f"  Rod{idx} Particle R", rod_info.particle_color[0], 0.0, 1.0
                )
                _changed_g, rod_info.particle_color[1] = ui.slider_float(
                    f"  Rod{idx} Particle G", rod_info.particle_color[1], 0.0, 1.0
                )
                _changed_b, rod_info.particle_color[2] = ui.slider_float(
                    f"  Rod{idx} Particle B", rod_info.particle_color[2], 0.0, 1.0
                )

        # Per-rod mesh rendering settings (only show when mesh is enabled)
        if self.show_rod_mesh:
            ui.separator()
            ui.text("Rod Mesh Settings")

            # Per-rod settings
            for idx, rod_info in enumerate(self.rod_infos):
                solver_name = rod_info.solver_type.value.upper()
                ui.text(f"  Rod {idx} ({solver_name})")
                _changed, rod_info.mesh_radius = ui.slider_float(
                    f"  Rod{idx} Mesh Radius", rod_info.mesh_radius, 0.001, 0.1
                )
                c = rod_info.color
                ui.text(f"    Color: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")

        ui.separator()
        ui.text("Root Control (both rods)")
        _changed, self.root_move_speed = ui.slider_float("Move Speed", self.root_move_speed, 0.1, 5.0)
        _changed, self.root_rotate_speed = ui.slider_float("Rotate Speed", self.root_rotate_speed, 0.1, 3.0)
        ui.text(f"  Rotation: {self.root_rotation:.2f} rad")
        ui.text("  Numpad: 4/6 X-, X+  8/2 Y+, Y-  9/3 Z+, Z-")
        ui.text("  ,/. or 7/1: Rotate -Z/+Z")

        ui.separator()
        mode_name = "INSERTION" if self.keyboard_insertion_mode else "MOVEMENT"
        ui.text(f"Keyboard Mode: {mode_name} (Tab to toggle)")
        if self.keyboard_insertion_mode:
            ui.text("  2/1: Rod 0 insertion +/-")
            ui.text("  0/9: Rod 1 insertion +/-")
        else:
            ui.text("  IJKLUO: J/L X-, X+  I/K Y+, Y-  U/O Z+, Z-")

        ui.separator()
        ui.text("Controls:")
        ui.text("  Tab: Toggle keyboard mode (insertion/movement)")
        ui.text("  G: Toggle gravity")
        ui.text("  B: Cycle GPU solver (Thomas/Banded)")
        ui.text("  F: Toggle root lock (position + rotation)")
        ui.text("  T: Toggle track sliding constraint")
        ui.text("  C: Toggle concentric constraint")
        ui.text("  P: Toggle bendable tip")
        ui.text("  +/- or Numpad +/-: Adjust tip bend angle")
        ui.text("  PgUp/PgDn: Rod 0 insertion +/-")
        ui.text("  Home/End: Rod 1 insertion +/-")
        ui.text("  R: Reset")

    def test_final(self):
        # Check all rods
        for idx, rod_info in enumerate(self.rod_infos):
            rod = rod_info.rod
            solver_name = rod_info.solver_type.value

            # Get positions (Warp rods need numpy conversion)
            if rod_info.solver_type == SolverType.WARP:
                positions = rod.positions_numpy()
            else:
                positions = rod.positions[:, 0:3]

            anchor = positions[0]
            initial = rod._initial_positions[0, 0:3]
            dist = float(np.linalg.norm(anchor - initial))
            assert dist < 1.0e-3, f"Rod {idx} ({solver_name}) anchor moved too far: {dist}"

            if not np.all(np.isfinite(positions)):
                raise AssertionError(f"Non-finite positions detected in rod {idx} ({solver_name})")

        for rod in self.gpu_state.rods:
            if not np.all(np.isfinite(rod.positions_numpy())):
                raise AssertionError("Non-finite GPU positions detected")


__all__ = ["Example"]
