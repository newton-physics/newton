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

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import (
    Contacts,
    Control,
    JointType,
    Model,
    ModelAttributeAssignment,
    ModelAttributeFrequency,
    ModelBuilder,
    State,
)
from ..solver import SolverBase
from .particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
    ParticleForceElementAdjacencyInfo,
    # Topological filtering helper functions
    _set_to_csr,
    accumulate_particle_body_contact_force_and_hessian,
    accumulate_self_contact_force_and_hessian,
    accumulate_spring_force_and_hessian,
    # Planar DAT (Divide and Truncate) kernels
    apply_planar_truncation_parallel_by_collision,
    apply_truncation_ts,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    # Adjacency building kernels
    count_num_adjacent_edges,
    count_num_adjacent_faces,
    count_num_adjacent_springs,
    count_num_adjacent_tets,
    fill_adjacent_edges,
    fill_adjacent_faces,
    fill_adjacent_springs,
    fill_adjacent_tets,
    # Solver kernels (particle VBD)
    forward_step,
    solve_elasticity,
    solve_elasticity_tile,
    update_velocity,
)
from .rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    RigidForceElementAdjacencyInfo,
    # Iteration kernels
    accumulate_body_body_contacts_per_body,  # Body-body (rigid-rigid) contacts (Gauss-Seidel mode)
    accumulate_body_particle_contacts_per_body,  # Body-particle soft contacts (two-way coupling)
    build_body_body_contact_lists,  # Body-body (rigid-rigid) contact adjacency
    build_body_particle_contact_lists,  # Body-particle (rigid-particle) soft-contact adjacency
    compute_cable_dahl_parameters,  # Cable bending plasticity
    copy_rigid_body_transforms_back,
    # Adjacency building kernels
    count_num_adjacent_joints,
    fill_adjacent_joints,
    # Pre-iteration kernels (rigid AVBD)
    forward_step_rigid_bodies,
    solve_rigid_body,
    # Post-iteration kernels
    update_body_velocity,
    update_cable_dahl_state,
    update_duals_body_body_contacts,  # Body-body (rigid-rigid) contacts (AVBD penalty update)
    update_duals_body_particle_contacts,  # Body-particle soft contacts (AVBD penalty update)
    update_duals_joint,  # Cable joints (AVBD penalty update)
    warmstart_body_body_contacts,  # Body-body (rigid-rigid) contacts (penalty warmstart)
    warmstart_body_particle_contacts,  # Body-particle soft contacts (penalty warmstart)
    warmstart_joints,  # Cable joints (stretch & bend)
)
from .tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)


class SolverVBD(SolverBase):
    """An implicit solver using Vertex Block Descent (VBD) for particles and Augmented VBD (AVBD) for rigid bodies.

    This unified solver supports:
        - Particle simulation (cloth, soft bodies) using the VBD algorithm
        - Rigid body simulation (joints, contacts) using the AVBD algorithm
        - Coupled particle-rigid body systems

    For rigid bodies, the AVBD algorithm uses **soft constraints** with adaptive penalty parameters
    for joints and contacts. Hard constraints are not currently enforced.


    References:
        - Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block Descent. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages.
          https://doi.org/10.1145/3658179
    Note:
        `SolverVBD` requires coloring information for both particles and rigid bodies:

        - Particle coloring: :attr:`newton.Model.particle_color_groups` (required if particles are present)
        - Rigid body coloring: :attr:`newton.Model.body_color_groups` (required if rigid bodies are present)

        Call :meth:`newton.ModelBuilder.color` to automatically color both particles and rigid bodies.

    Example
    -------

    .. code-block:: python

        # Automatically color both particles and rigid bodies
        builder.color()

        model = builder.finalize()

        solver = newton.solvers.SolverVBD(model)

        # Initialize states and contacts
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.collide(state_in)

        # Simulation loop
        for i in range(100):
            contacts = model.collide(state_in)  # Update contacts
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    def __init__(
        self,
        model: Model,
        # Common parameters
        iterations: int = 10,
        friction_epsilon: float = 1e-2,
        integrate_with_external_rigid_solver: bool = False,
        # Particle parameters
        particle_enable_self_contact: bool = False,
        particle_self_contact_radius: float = 0.2,
        particle_self_contact_margin: float = 0.2,
        particle_conservative_bound_relaxation: float = 0.85,
        particle_vertex_contact_buffer_size: int = 32,
        particle_edge_contact_buffer_size: int = 64,
        particle_collision_detection_interval: int = 0,
        particle_edge_parallel_epsilon: float = 1e-5,
        particle_enable_tile_solve: bool = True,
        particle_topological_contact_filter_threshold: int = 2,
        particle_rest_shape_contact_exclusion_radius: float = 0.0,
        particle_external_vertex_contact_filtering_map: dict | None = None,
        particle_external_edge_contact_filtering_map: dict | None = None,
        # Rigid body parameters
        rigid_avbd_beta: float = 1.0e5,
        rigid_avbd_gamma: float = 0.99,
        rigid_contact_k_start: float = 1.0e2,  # AVBD: initial stiffness for all body contacts (body-body + body-particle)
        rigid_joint_linear_k_start: float = 1.0e4,  # AVBD: initial stiffness seed for linear joint constraints
        rigid_joint_angular_k_start: float = 1.0e1,  # AVBD: initial stiffness seed for angular joint constraints
        rigid_joint_linear_ke: float = 1.0e9,  # AVBD: stiffness cap for non-cable linear joint constraints (BALL/FIXED/etc.)
        rigid_joint_angular_ke: float = 1.0e9,  # AVBD: stiffness cap for non-cable angular joint constraints (FIXED/etc.)
        rigid_joint_linear_kd: float = 0.0,  # AVBD: Rayleigh damping coefficient for non-cable linear joint constraints
        rigid_joint_angular_kd: float = 0.0,  # AVBD: Rayleigh damping coefficient for non-cable angular joint constraints
        rigid_body_contact_buffer_size: int = 64,
        rigid_body_particle_contact_buffer_size: int = 256,
        rigid_enable_dahl_friction: bool = False,  # Cable bending plasticity/hysteresis
    ):
        """
        Args:
            model: The `Model` object used to initialize the integrator. Must be identical to the `Model` object passed
                to the `step` function.

            Common parameters:

            iterations: Number of VBD iterations per step.
            friction_epsilon: Threshold to smooth small relative velocities in friction computation (used for both particle
                and rigid body contacts).

            Particle parameters:

            particle_enable_self_contact: Whether to enable self-contact detection for particles.
            particle_self_contact_radius: The radius used for self-contact detection. This is the distance at which
                vertex-triangle pairs and edge-edge pairs will start to interact with each other.
            particle_self_contact_margin: The margin used for self-contact detection. This is the distance at which
                vertex-triangle pairs and edge-edge will be considered in contact generation. It should be larger than
                `particle_self_contact_radius` to avoid missing contacts.
            integrate_with_external_rigid_solver: Indicator for coupled rigid body-cloth simulation. When set to `True`,
                the solver assumes rigid bodies are integrated by an external solver (one-way coupling).
            particle_conservative_bound_relaxation: Relaxation factor for conservative penetration-free projection.
            particle_vertex_contact_buffer_size: Preallocation size for each vertex's vertex-triangle collision buffer.
            particle_edge_contact_buffer_size: Preallocation size for edge's edge-edge collision buffer.
            particle_collision_detection_interval: Controls how frequently particle self-contact detection is applied
                during the simulation. If set to a value < 0, collision detection is only performed once before the
                initialization step. If set to 0, collision detection is applied twice: once before and once immediately
                after initialization. If set to a value `n` >= 1, collision detection is applied before every `n` VBD
                iterations.
            particle_edge_parallel_epsilon: Threshold to detect near-parallel edges in edge-edge collision handling.
            particle_enable_tile_solve: Whether to accelerate the particle solver using tile API.
            particle_topological_contact_filter_threshold: Maximum topological distance (measured in rings) under which candidate
                self-contacts are discarded. Set to a higher value to tolerate contacts between more closely connected mesh
                elements. Only used when `particle_enable_self_contact` is `True`. Note that setting this to a value larger than 3 will
                result in a significant increase in computation time.
            particle_rest_shape_contact_exclusion_radius: Additional world-space distance threshold for filtering topologically close
                primitives. Candidate contacts with a rest separation shorter than this value are ignored. The distance is
                evaluated in the rest configuration conveyed by `model.particle_q`. Only used when `particle_enable_self_contact` is `True`.
            particle_external_vertex_contact_filtering_map: Optional dictionary used to exclude additional vertex-triangle pairs during
                contact generation. Keys must be vertex primitive ids (integers), and each value must be a `list` or
                `set` containing the triangle primitives to be filtered out. Only used when `particle_enable_self_contact` is `True`.
            particle_external_edge_contact_filtering_map: Optional dictionary used to exclude additional edge-edge pairs during contact
                generation. Keys must be edge primitive ids (integers), and each value must be a `list` or `set`
                containing the edges to be filtered out. Only used when `particle_enable_self_contact` is `True`.

            Rigid body parameters:

            rigid_avbd_beta: Penalty ramp rate for rigid body constraints (how fast k grows with constraint violation).
            rigid_avbd_gamma: Warmstart decay for penalty k (cross-step decay factor for rigid body constraints).
            rigid_contact_k_start: Initial penalty stiffness for all body contact constraints, including both body-body (rigid-rigid)
                and body-particle (rigid-particle) contacts (AVBD).
            rigid_joint_linear_k_start: Initial penalty seed for linear joint constraints (e.g., cable stretch, BALL linear).
                Used to seed the per-constraint adaptive penalties for all linear joint constraints.
            rigid_joint_angular_k_start: Initial penalty seed for angular joint constraints (e.g., cable bend, FIXED angular).
                Used to seed the per-constraint adaptive penalties for all angular joint constraints.
            rigid_joint_linear_ke: Stiffness cap used by AVBD for **non-cable** linear joint constraint scalars
                (e.g., BALL and the linear part of FIXED). Cable joints use the per-joint caps in
                ``model.joint_target_ke`` instead (cable interprets ``joint_target_ke/kd`` as constraint tuning).
            rigid_joint_angular_ke: Stiffness cap used by AVBD for **non-cable** angular joint constraint scalars
                (e.g., FIXED).
            rigid_joint_linear_kd: Rayleigh damping coefficient for non-cable linear joint constraints (paired with
                ``rigid_joint_linear_ke``).
            rigid_joint_angular_kd: Rayleigh damping coefficient for non-cable angular joint constraints (paired with
                ``rigid_joint_angular_ke``).
            rigid_body_contact_buffer_size: Max body-body (rigid-rigid) contacts per rigid body for per-body contact lists (tune based on expected body-body contact density).
            rigid_body_particle_contact_buffer_size: Max body-particle (rigid-particle) contacts per rigid body for per-body soft-contact lists (tune based on expected body-particle contact density).
            rigid_enable_dahl_friction: Enable Dahl hysteresis friction model for cable bending (default: False).
                Configure per-joint Dahl parameters via the solver-registered custom model attributes
                ``model.vbd.dahl_eps_max`` and ``model.vbd.dahl_tau``.

        Note:
            - The `integrate_with_external_rigid_solver` argument enables one-way coupling between rigid body and soft body
              solvers. If set to True, the rigid states should be integrated externally, with `state_in` passed to `step`
              representing the previous rigid state and `state_out` representing the current one. Frictional forces are
              computed accordingly.
            - `particle_vertex_contact_buffer_size`, `particle_edge_contact_buffer_size`, `rigid_body_contact_buffer_size`,
              and `rigid_body_particle_contact_buffer_size` are fixed and will not be dynamically resized during runtime.
              Setting them too small may result in undetected collisions (particles) or contact overflow (rigid body
              contacts).
              Setting them excessively large may increase memory usage and degrade performance.

        """
        super().__init__(model)

        # Common parameters
        self.iterations = iterations
        self.friction_epsilon = friction_epsilon

        # Rigid integration mode: when True, rigid bodies are integrated by an external
        # solver (one-way coupling). SolverVBD will not move rigid bodies, but can still
        # participate in particle-rigid interaction on the particle side.
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver

        # Initialize particle system
        self._init_particle_system(
            model,
            particle_enable_self_contact,
            particle_self_contact_radius,
            particle_self_contact_margin,
            particle_conservative_bound_relaxation,
            particle_vertex_contact_buffer_size,
            particle_edge_contact_buffer_size,
            particle_collision_detection_interval,
            particle_edge_parallel_epsilon,
            particle_enable_tile_solve,
            particle_topological_contact_filter_threshold,
            particle_rest_shape_contact_exclusion_radius,
            particle_external_vertex_contact_filtering_map,
            particle_external_edge_contact_filtering_map,
        )

        # Initialize rigid body system and rigid-particle (body-particle) interaction state
        self._init_rigid_system(
            model,
            rigid_avbd_beta,
            rigid_avbd_gamma,
            rigid_contact_k_start,
            rigid_joint_linear_k_start,
            rigid_joint_angular_k_start,
            rigid_joint_linear_ke,
            rigid_joint_angular_ke,
            rigid_joint_linear_kd,
            rigid_joint_angular_kd,
            rigid_body_contact_buffer_size,
            rigid_body_particle_contact_buffer_size,
            rigid_enable_dahl_friction,
        )

        # Rigid-only flag to control whether to update cross-step history
        # (rigid warmstart state such as contact/joint history).
        # Defaults to True. This setting applies only to the next call to :meth:`step` and is then
        # reset to ``True``. This is useful for substepping, where history update frequency might
        # differ from the simulation step frequency (e.g. updating only on the first substep).
        # This flag is automatically reset to True after each step().
        # Rigid warmstart update flag (contacts/joints).
        self.update_rigid_history = True

    def _init_particle_system(
        self,
        model: Model,
        particle_enable_self_contact: bool,
        particle_self_contact_radius: float,
        particle_self_contact_margin: float,
        particle_conservative_bound_relaxation: float,
        particle_vertex_contact_buffer_size: int,
        particle_edge_contact_buffer_size: int,
        particle_collision_detection_interval: int,
        particle_edge_parallel_epsilon: float,
        particle_enable_tile_solve: bool,
        particle_topological_contact_filter_threshold: int,
        particle_rest_shape_contact_exclusion_radius: float,
        particle_external_vertex_contact_filtering_map: dict | None,
        particle_external_edge_contact_filtering_map: dict | None,
    ):
        """Initialize particle-specific data structures and settings."""
        # Early exit if no particles
        if model.particle_count == 0:
            return

        self.particle_collision_detection_interval = particle_collision_detection_interval
        self.particle_topological_contact_filter_threshold = particle_topological_contact_filter_threshold
        self.particle_rest_shape_contact_exclusion_radius = particle_rest_shape_contact_exclusion_radius

        # Particle state storage
        self.particle_q_prev = wp.zeros_like(
            model.particle_q, device=self.device
        )  # per-substep previous q (for velocity)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)  # inertial target positions

        # Particle adjacency info
        self.particle_adjacency = self.compute_particle_force_element_adjacency().to(self.device)

        # Self-contact settings
        self.particle_enable_self_contact = particle_enable_self_contact
        self.particle_self_contact_radius = particle_self_contact_radius
        self.particle_self_contact_margin = particle_self_contact_margin
        self.particle_q_rest = model.particle_q

        # Tile solve settings
        if model.device.is_cpu and particle_enable_tile_solve:
            warnings.warn("Tiled solve requires model.device='cuda'. Tiled solve is disabled.", stacklevel=2)

        self.use_particle_tile_solve = particle_enable_tile_solve and model.device.is_cuda

        if particle_enable_self_contact:
            if particle_self_contact_margin < particle_self_contact_radius:
                raise ValueError(
                    "particle_self_contact_margin is smaller than particle_self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make particle_self_contact_margin 1.5-2 times larger than particle_self_contact_radius."
                )

            self.particle_conservative_bound_relaxation = particle_conservative_bound_relaxation
            self.particle_conservative_bounds = wp.zeros((model.particle_count,), dtype=float, device=self.device)
            # CCD safety margin: 2x the conservative bound relaxation (e.g., 0.42 -> 0.84)
            self.ccd_safety_margin = 2.0 * self.particle_conservative_bound_relaxation

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=particle_vertex_contact_buffer_size,
                edge_collision_buffer_pre_alloc=particle_edge_contact_buffer_size,
                edge_edge_parallel_epsilon=particle_edge_parallel_epsilon,
            )

            self.compute_particle_contact_filtering_list(
                particle_external_vertex_contact_filtering_map, particle_external_edge_contact_filtering_map
            )

            self.trimesh_collision_detector.set_collision_filter_list(
                self.particle_vertex_triangle_contact_filtering_list,
                self.particle_vertex_triangle_contact_filtering_list_offsets,
                self.particle_edge_edge_contact_filtering_list,
                self.particle_edge_edge_contact_filtering_list_offsets,
            )

            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
            )

            self.particle_self_contact_evaluation_kernel_launch_size = max(
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
            )
        else:
            self.particle_self_contact_evaluation_kernel_launch_size = (
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE
            )

        # Particle force and hessian storage
        self.particle_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.device)

        # Validation
        if len(self.model.particle_color_groups) == 0:
            raise ValueError(
                "model.particle_color_groups is empty! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

        self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
        self.particle_displacements = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.truncation_ts = wp.zeros(self.model.particle_count, dtype=float, device=self.device)

    def _init_rigid_system(
        self,
        model: Model,
        rigid_avbd_beta: float,
        rigid_avbd_gamma: float,
        rigid_contact_k_start: float,
        rigid_joint_linear_k_start: float,
        rigid_joint_angular_k_start: float,
        rigid_joint_linear_ke: float,
        rigid_joint_angular_ke: float,
        rigid_joint_linear_kd: float,
        rigid_joint_angular_kd: float,
        rigid_body_contact_buffer_size: int,
        rigid_body_particle_contact_buffer_size: int,
        rigid_enable_dahl_friction: bool,
    ):
        """Initialize rigid body-specific AVBD data structures and settings.

        This includes:
          - Rigid-only AVBD state (joints, body-body contacts, Dahl friction)
          - Shared interaction state for body-particle (rigid-particle) soft contacts
        """
        # AVBD penalty parameters
        self.avbd_beta = rigid_avbd_beta
        self.avbd_gamma = rigid_avbd_gamma

        # Common initial penalty seed / lower bound for body contacts (clamped to non-negative)
        self.k_start_body_contact = max(0.0, rigid_contact_k_start)

        # Joint constraint caps and damping for non-cable joints (constraint enforcement, not drives)
        self.rigid_joint_linear_ke = max(0.0, rigid_joint_linear_ke)
        self.rigid_joint_angular_ke = max(0.0, rigid_joint_angular_ke)
        self.rigid_joint_linear_kd = max(0.0, rigid_joint_linear_kd)
        self.rigid_joint_angular_kd = max(0.0, rigid_joint_angular_kd)

        # -------------------------------------------------------------
        # Rigid-only AVBD state (used when SolverVBD integrates bodies)
        # -------------------------------------------------------------
        if not self.integrate_with_external_rigid_solver and model.body_count > 0:
            # State storage
            # Initialize to the current poses for the first step to avoid spurious finite-difference
            # velocities/friction impulses.
            self.body_q_prev = wp.clone(model.body_q).to(self.device)
            self.body_inertia_q = wp.zeros_like(model.body_q, device=self.device)  # inertial target poses for AVBD

            # Adjacency and dimensions
            self.rigid_adjacency = self.compute_rigid_force_element_adjacency(model).to(self.device)

            # Force accumulation arrays
            self.body_torques = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)
            self.body_forces = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)

            # Hessian blocks (6x6 block structure: angular-angular, angular-linear, linear-linear)
            self.body_hessian_aa = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)
            self.body_hessian_al = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)
            self.body_hessian_ll = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)

            # Per-body contact lists
            # Body-body (rigid-rigid) contact adjacency (CSR-like: per-body counts and flat index array)
            self.body_body_contact_buffer_pre_alloc = rigid_body_contact_buffer_size
            self.body_body_contact_counts = wp.zeros(self.model.body_count, dtype=wp.int32, device=self.device)
            self.body_body_contact_indices = wp.zeros(
                self.model.body_count * self.body_body_contact_buffer_pre_alloc, dtype=wp.int32, device=self.device
            )

            # Body-particle (rigid-particle) contact adjacency (CSR-like: per-body counts and flat index array)
            self.body_particle_contact_buffer_pre_alloc = rigid_body_particle_contact_buffer_size
            self.body_particle_contact_counts = wp.zeros(self.model.body_count, dtype=wp.int32, device=self.device)
            self.body_particle_contact_indices = wp.zeros(
                self.model.body_count * self.body_particle_contact_buffer_pre_alloc,
                dtype=wp.int32,
                device=self.device,
            )

            # AVBD constraint penalties
            # Joint constraint layout + penalties (solver constraint scalars)
            self._init_joint_constraint_layout()
            self.joint_penalty_k = self._init_joint_penalty_k(rigid_joint_linear_k_start, rigid_joint_angular_k_start)

            # Contact penalties (adaptive penalties for body-body contacts)
            if model.shape_count > 0:
                if not hasattr(model, "rigid_contact_max") or model.rigid_contact_max is None:
                    raise ValueError(
                        "Model.rigid_contact_max is not set. Ensure the model was created via ModelBuilder.finalize()."
                    )
                max_contacts = model.rigid_contact_max
                # Per-contact AVBD penalty for body-body contacts
                self.body_body_contact_penalty_k = wp.full(
                    (max_contacts,), self.k_start_body_contact, dtype=float, device=self.device
                )

                # Pre-computed averaged body-body contact material properties (computed once per step in warmstart)
                self.body_body_contact_material_ke = wp.zeros(max_contacts, dtype=float, device=self.device)
                self.body_body_contact_material_kd = wp.zeros(max_contacts, dtype=float, device=self.device)
                self.body_body_contact_material_mu = wp.zeros(max_contacts, dtype=float, device=self.device)

            # Dahl friction model (cable bending plasticity)
            # State variables for Dahl hysteresis (persistent across timesteps)
            self.joint_sigma_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_kappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_dkappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Pre-computed Dahl parameters (frozen during iterations, updated per timestep)
            self.joint_sigma_start = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C_fric = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Dahl model configuration
            self.enable_dahl_friction = rigid_enable_dahl_friction
            self.joint_dahl_eps_max = wp.zeros(model.joint_count, dtype=float, device=self.device)
            self.joint_dahl_tau = wp.zeros(model.joint_count, dtype=float, device=self.device)

            if rigid_enable_dahl_friction:
                if model.joint_count == 0:
                    self.enable_dahl_friction = False
                else:
                    # Read per-joint Dahl parameters from model.vbd if present; otherwise use defaults (eps_max=0.5, tau=1.0).
                    # Recommended: call SolverVBD.register_custom_attributes(builder) before finalize() to allocate these arrays.
                    vbd_attrs: Any = getattr(model, "vbd", None)
                    if vbd_attrs is not None and hasattr(vbd_attrs, "dahl_eps_max") and hasattr(vbd_attrs, "dahl_tau"):
                        self.joint_dahl_eps_max = vbd_attrs.dahl_eps_max
                        self.joint_dahl_tau = vbd_attrs.dahl_tau
                    else:
                        self._init_dahl_params(0.5, 1.0, model)

        # -------------------------------------------------------------
        # Body-particle interaction - shared state
        # -------------------------------------------------------------
        # Soft contact penalties (adaptive penalties for body-particle contacts)
        # Use same initial penalty as body-body contacts
        max_soft_contacts = model.shape_count * model.particle_count
        # Per-contact AVBD penalty for body-particle soft contacts (same initial seed as body-body)
        self.body_particle_contact_penalty_k = wp.full(
            (max_soft_contacts,), self.k_start_body_contact, dtype=float, device=self.device
        )

        # Pre-computed averaged body-particle soft contact material properties (computed once per step in warmstart)
        # These correspond to body-particle soft contacts and are averaged between model.soft_contact_*
        # and shape material properties.
        self.body_particle_contact_material_ke = wp.zeros(max_soft_contacts, dtype=float, device=self.device)
        self.body_particle_contact_material_kd = wp.zeros(max_soft_contacts, dtype=float, device=self.device)
        self.body_particle_contact_material_mu = wp.zeros(max_soft_contacts, dtype=float, device=self.device)

        # Validation
        has_bodies = self.model.body_count > 0
        has_body_coloring = len(self.model.body_color_groups) > 0

        if has_bodies and not has_body_coloring:
            raise ValueError(
                "model.body_color_groups is empty but rigid bodies are present! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

    # =====================================================
    # Initialization Helper Methods
    # =====================================================

    def _init_joint_constraint_layout(self) -> None:
        """Initialize VBD-owned joint constraint indexing.

        VBD stores and adapts penalty stiffness values for *scalar constraint components*:
          - ``JointType.CABLE``: 2 scalars (stretch/linear, bend/angular)
          - ``JointType.BALL``: 1 scalar (isotropic linear anchor-coincidence)
          - ``JointType.FIXED``: 2 scalars (isotropic linear anchor-coincidence + isotropic angular)
          - ``JointType.FREE``: 0 scalars (not a constraint)

        Ordering (must match kernel indexing via ``joint_constraint_start``):
          - ``JointType.CABLE``: [stretch (linear), bend (angular)]
          - ``JointType.BALL``: [linear]
          - ``JointType.FIXED``: [linear, angular]

        Any other joint type will raise ``NotImplementedError``.
        """
        n_j = self.model.joint_count
        with wp.ScopedDevice("cpu"):
            jt_cpu = self.model.joint_type.to("cpu")
            jt = jt_cpu.numpy() if hasattr(jt_cpu, "numpy") else np.asarray(jt_cpu, dtype=int)

            dim_np = np.zeros((n_j,), dtype=np.int32)
            for j in range(n_j):
                if jt[j] == JointType.CABLE:
                    dim_np[j] = 2
                elif jt[j] == JointType.BALL:
                    dim_np[j] = 1
                elif jt[j] == JointType.FIXED:
                    dim_np[j] = 2
                else:
                    if jt[j] != JointType.FREE:
                        raise NotImplementedError(
                            f"SolverVBD rigid joints: JointType.{JointType(jt[j]).name} is not implemented yet "
                            "(only CABLE, BALL, and FIXED are supported)."
                        )
                    dim_np[j] = 0

            start_np = np.zeros((n_j,), dtype=np.int32)
            c = 0
            for j in range(n_j):
                start_np[j] = np.int32(c)
                c += int(dim_np[j])

            self.joint_constraint_count = int(c)
            self.joint_constraint_dim = wp.array(dim_np, dtype=wp.int32, device=self.device)
            self.joint_constraint_start = wp.array(start_np, dtype=wp.int32, device=self.device)

    def _init_joint_penalty_k(self, k_start_joint_linear: float, k_start_joint_angular: float):
        """
        Build initial joint penalty state on CPU and upload to solver device.

        This initializes the solver-owned joint constraint parameter arrays used by VBD.
        The arrays are sized by ``self.joint_constraint_count`` and indexed using
        ``self.joint_constraint_start`` (solver constraint indexing), not by model DOF indexing.

        Arrays:
          - ``k0``: initial penalty stiffness for each solver constraint scalar (stored as ``self.joint_penalty_k``)
          - ``k_min``: warmstart floor for each solver constraint scalar (stored as ``self.joint_penalty_k_min``)
          - ``k_max``: stiffness cap for each solver constraint scalar (stored as ``self.joint_penalty_k_max``)
          - ``kd``: damping coefficient for each solver constraint scalar (stored as ``self.joint_penalty_kd``)

        Supported rigid joint constraint types in SolverVBD:
          - ``JointType.CABLE`` (2 scalars: stretch + bend)
          - ``JointType.BALL`` (1 scalar: isotropic linear anchor-coincidence)
          - ``JointType.FIXED`` (2 scalars: isotropic linear anchor-coincidence + isotropic angular)

        ``JointType.FREE`` joints (created by :meth:`ModelBuilder.add_body`) are not constraints and are ignored.
        Any other joint types will raise ``NotImplementedError``.
        """
        if (
            not hasattr(self, "joint_constraint_start")
            or not hasattr(self, "joint_constraint_dim")
            or not hasattr(self, "joint_constraint_count")
        ):
            raise RuntimeError(
                "SolverVBD joint constraint layout is not initialized. "
                "Call SolverVBD._init_joint_constraint_layout() before _init_joint_penalty_k()."
            )

        if self.joint_constraint_count < 0:
            raise RuntimeError(
                f"SolverVBD joint constraint layout is invalid: joint_constraint_count={self.joint_constraint_count!r}"
            )

        constraint_count = self.joint_constraint_count
        with wp.ScopedDevice("cpu"):
            # Per-constraint AVBD penalty state:
            # - k0: initial penalty stiffness for this scalar constraint
            # - k_min: warmstart floor (so k doesn't decay below this across steps)
            # - k_max: stiffness cap (so k never exceeds the chosen target for this constraint)
            #
            # We start from solver-level seeds (k_start_*), but clamp to the per-constraint cap (k_max) so we always
            # satisfy k_min <= k0 <= k_max.
            stretch_k = max(0.0, k_start_joint_linear)
            bend_k = max(0.0, k_start_joint_angular)
            joint_k_min_np = np.zeros((constraint_count,), dtype=float)
            joint_k0_np = np.zeros((constraint_count,), dtype=float)
            # Per-constraint stiffness caps used for AVBD warmstart clamping and penalty growth limiting.
            # - Cable constraints: use model.joint_target_ke (cable material/constraint tuning; still model-DOF indexed)
            # - Ball constraints: use solver-level caps (rigid_joint_linear_ke)
            # Start from zeros and explicitly fill per joint/constraint-slot below for clarity.
            joint_k_max_np = np.zeros((constraint_count,), dtype=float)
            joint_kd_np = np.zeros((constraint_count,), dtype=float)

            jt_cpu = self.model.joint_type.to("cpu")
            jdofs_cpu = self.model.joint_qd_start.to("cpu")
            jtarget_ke_cpu = self.model.joint_target_ke.to("cpu")
            jtarget_kd_cpu = self.model.joint_target_kd.to("cpu")
            jc_start_cpu = self.joint_constraint_start.to("cpu")

            jt = jt_cpu.numpy() if hasattr(jt_cpu, "numpy") else np.asarray(jt_cpu, dtype=int)
            jdofs = jdofs_cpu.numpy() if hasattr(jdofs_cpu, "numpy") else np.asarray(jdofs_cpu, dtype=int)
            jc_start = (
                jc_start_cpu.numpy() if hasattr(jc_start_cpu, "numpy") else np.asarray(jc_start_cpu, dtype=np.int32)
            )
            jtarget_ke = (
                jtarget_ke_cpu.numpy() if hasattr(jtarget_ke_cpu, "numpy") else np.asarray(jtarget_ke_cpu, dtype=float)
            )
            jtarget_kd = (
                jtarget_kd_cpu.numpy() if hasattr(jtarget_kd_cpu, "numpy") else np.asarray(jtarget_kd_cpu, dtype=float)
            )

            n_j = self.model.joint_count
            for j in range(n_j):
                if jt[j] == JointType.CABLE:
                    c0 = int(jc_start[j])
                    dof0 = int(jdofs[j])
                    # CABLE requires 2 DOF entries in model.joint_target_ke/kd starting at joint_qd_start[j].
                    if dof0 < 0 or (dof0 + 1) >= len(jtarget_ke) or (dof0 + 1) >= len(jtarget_kd):
                        raise RuntimeError(
                            "SolverVBD _init_joint_penalty_k: JointType.CABLE requires 2 DOF entries in "
                            "model.joint_target_ke/kd starting at joint_qd_start[j]. "
                            f"Got joint_index={j}, joint_qd_start={dof0}, "
                            f"len(joint_target_ke)={len(jtarget_ke)}, len(joint_target_kd)={len(jtarget_kd)}."
                        )
                    # Constraint 0: cable stretch; constraint 1: cable bend
                    # Caps come from model.joint_target_ke (still model DOF indexed for cable material tuning).
                    joint_k_max_np[c0] = jtarget_ke[dof0]
                    joint_k_max_np[c0 + 1] = jtarget_ke[dof0 + 1]
                    # Per-slot warmstart lower bounds:
                    # - Use k_start_* as the floor, but clamp to the cap so k_min <= k_max always.
                    joint_k_min_np[c0] = min(stretch_k, joint_k_max_np[c0])
                    joint_k_min_np[c0 + 1] = min(bend_k, joint_k_max_np[c0 + 1])
                    # Initial seed: clamp to cap so k0 <= k_max
                    joint_k0_np[c0] = min(stretch_k, joint_k_max_np[c0])
                    joint_k0_np[c0 + 1] = min(bend_k, joint_k_max_np[c0 + 1])
                    # Damping comes from model.joint_target_kd (still model DOF indexed for cable tuning).
                    joint_kd_np[c0] = jtarget_kd[dof0]
                    joint_kd_np[c0 + 1] = jtarget_kd[dof0 + 1]
                elif jt[j] == JointType.BALL:
                    # BALL joints: isotropic linear anchor-coincidence constraint stored as a single scalar.
                    c0 = int(jc_start[j])
                    joint_k_max_np[c0] = self.rigid_joint_linear_ke
                    k_floor = min(stretch_k, self.rigid_joint_linear_ke)
                    joint_k_min_np[c0] = k_floor
                    joint_k0_np[c0] = k_floor
                    joint_kd_np[c0] = self.rigid_joint_linear_kd
                elif jt[j] == JointType.FIXED:
                    # FIXED joints are enforced as:
                    #   - 1 isotropic linear anchor-coincidence constraint (vector error, scalar penalty)
                    #   - 1 isotropic angular constraint (rotation-vector error, scalar penalty)
                    c0 = int(jc_start[j])

                    # Linear cap + floor (isotropic)
                    joint_k_max_np[c0 + 0] = self.rigid_joint_linear_ke
                    k_lin_floor = min(stretch_k, self.rigid_joint_linear_ke)
                    joint_k_min_np[c0 + 0] = k_lin_floor
                    joint_k0_np[c0 + 0] = k_lin_floor
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd

                    # Angular cap + floor (isotropic)
                    joint_k_max_np[c0 + 1] = self.rigid_joint_angular_ke
                    k_ang_floor = min(bend_k, self.rigid_joint_angular_ke)
                    joint_k_min_np[c0 + 1] = k_ang_floor
                    joint_k0_np[c0 + 1] = k_ang_floor
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                else:
                    # Layout builder already validated supported types; nothing to do for FREE.
                    pass

            # Upload to device: initial penalties, per-constraint caps, damping, and warmstart floors.
            self.joint_penalty_k_min = wp.array(joint_k_min_np, dtype=float, device=self.device)
            self.joint_penalty_k_max = wp.array(joint_k_max_np, dtype=float, device=self.device)
            self.joint_penalty_kd = wp.array(joint_kd_np, dtype=float, device=self.device)
            return wp.array(joint_k0_np, dtype=float, device=self.device)

    def _init_dahl_params(self, eps_max_input, tau_input, model):
        """
        Initialize per-joint Dahl friction parameters.

        Args:
            eps_max_input: float or array-like. Maximum strain (curvature) [rad].
                - Scalar: broadcast to all joints
                - Array-like (length = model.joint_count): per-joint values
                - Per-joint disable: set value to 0 for that joint
            tau_input: float or array-like. Memory decay length [rad].
                - Scalar: broadcast to all joints
                - Array-like (length = model.joint_count): per-joint values
                - Per-joint disable: set value to 0 for that joint
            model: Model object

        Notes:
            - This function validates shapes and converts to device arrays; it does not clamp or validate ranges.
              Kernels perform any necessary early-outs based on zero values.
            - To disable Dahl friction:
                - Globally: pass enable_dahl_friction=False to the constructor
                - Per-joint: set dahl_eps_max=0 or dahl_tau=0 for those joints
        """
        n = model.joint_count

        # eps_max
        if isinstance(eps_max_input, (int, float)):
            self.joint_dahl_eps_max = wp.full(n, eps_max_input, dtype=float, device=self.device)
        else:
            # Convert to numpy first
            x = eps_max_input.to("cpu") if hasattr(eps_max_input, "to") else eps_max_input
            eps_np = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=float)
            if eps_np.shape[0] != n:
                raise ValueError(f"dahl_eps_max length {eps_np.shape[0]} != joint_count {n}")
            # Direct host-to-device copy
            self.joint_dahl_eps_max = wp.array(eps_np, dtype=float, device=self.device)

        # tau
        if isinstance(tau_input, (int, float)):
            self.joint_dahl_tau = wp.full(n, tau_input, dtype=float, device=self.device)
        else:
            # Convert to numpy first
            x = tau_input.to("cpu") if hasattr(tau_input, "to") else tau_input
            tau_np = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=float)
            if tau_np.shape[0] != n:
                raise ValueError(f"dahl_tau length {tau_np.shape[0]} != joint_count {n}")
            # Direct host-to-device copy
            self.joint_dahl_tau = wp.array(tau_np, dtype=float, device=self.device)

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register solver-specific custom Model attributes for SolverVBD.

        Currently used for cable bending plasticity/hysteresis (Dahl friction model).

        Attributes are declared in the ``vbd`` namespace so they can be authored in scenes
        and in USD as ``newton:vbd:<attr>``.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_eps_max",
                frequency=ModelAttributeFrequency.JOINT,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.5,
                namespace="vbd",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_tau",
                frequency=ModelAttributeFrequency.JOINT,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="vbd",
            )
        )

    # =====================================================
    # Adjacency Building Methods
    # =====================================================

    def compute_particle_force_element_adjacency(self):
        particle_adjacency = ParticleForceElementAdjacencyInfo()

        with wp.ScopedDevice("cpu"):
            if self.model.edge_indices:
                edges_array = self.model.edge_indices.to("cpu")
                # build vertex-edge particle_adjacency data
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                )

                num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
                vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
                vertex_adjacent_edges_offsets[0] = 0
                particle_adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent edges has been filled to each vertex
                vertex_adjacent_edges_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                edge_particle_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
                # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
                particle_adjacency.v_adj_edges = wp.empty(shape=(edge_particle_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        particle_adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        particle_adjacency.v_adj_edges,
                    ],
                    dim=1,
                )
            else:
                particle_adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.tri_indices:
                face_indices = self.model.tri_indices.to("cpu")
                # compute adjacent triangles
                # count number of adjacent faces for each vertex
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(kernel=count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1)

                # preallocate memory based on counting results
                num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
                vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
                vertex_adjacent_faces_offsets[0] = 0
                particle_adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32)

                vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                face_particle_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
                # (face, vertex_order) * num_adj_faces * num_particles
                # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
                particle_adjacency.v_adj_faces = wp.empty(shape=(face_particle_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        particle_adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        particle_adjacency.v_adj_faces,
                    ],
                    dim=1,
                )
            else:
                particle_adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.tet_indices:
                tet_indices = self.model.tet_indices.to("cpu")
                num_vertex_adjacent_tets = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=count_num_adjacent_tets,
                    inputs=[tet_indices, num_vertex_adjacent_tets],
                    dim=1,
                )

                num_vertex_adjacent_tets = num_vertex_adjacent_tets.numpy()
                vertex_adjacent_tets_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_tets_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_tets)[:]
                vertex_adjacent_tets_offsets[0] = 0
                particle_adjacency.v_adj_tets_offsets = wp.array(vertex_adjacent_tets_offsets, dtype=wp.int32)

                vertex_adjacent_tets_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                tet_particle_adjacency_array_size = 2 * num_vertex_adjacent_tets.sum()
                particle_adjacency.v_adj_tets = wp.empty(shape=(tet_particle_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_tets,
                    inputs=[
                        tet_indices,
                        particle_adjacency.v_adj_tets_offsets,
                        vertex_adjacent_tets_fill_count,
                        particle_adjacency.v_adj_tets,
                    ],
                    dim=1,
                )
            else:
                particle_adjacency.v_adj_tets_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_tets = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.spring_indices:
                spring_array = self.model.spring_indices.to("cpu")
                # build vertex-springs particle_adjacency data
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                )

                num_vertex_adjacent_spring = num_vertex_adjacent_spring.numpy()
                vertex_adjacent_springs_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_springs_offsets[1:] = np.cumsum(num_vertex_adjacent_spring)[:]
                vertex_adjacent_springs_offsets[0] = 0
                particle_adjacency.v_adj_springs_offsets = wp.array(vertex_adjacent_springs_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent springs has been filled to each vertex
                vertex_adjacent_springs_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                particle_adjacency.v_adj_springs = wp.empty(shape=(num_vertex_adjacent_spring.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        particle_adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        particle_adjacency.v_adj_springs,
                    ],
                    dim=1,
                )

            else:
                particle_adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

        return particle_adjacency

    def compute_particle_contact_filtering_list(
        self, external_vertex_contact_filtering_map, external_edge_contact_filtering_map
    ):
        if self.model.tri_count:
            v_tri_filter_sets = None
            edge_edge_filter_sets = None
            if self.particle_topological_contact_filter_threshold >= 2:
                if self.particle_adjacency.v_adj_faces_offsets.size > 0:
                    v_tri_filter_sets = build_vertex_n_ring_tris_collision_filter(
                        self.particle_topological_contact_filter_threshold,
                        self.model.particle_count,
                        self.model.edge_indices.numpy(),
                        self.particle_adjacency.v_adj_edges.numpy(),
                        self.particle_adjacency.v_adj_edges_offsets.numpy(),
                        self.particle_adjacency.v_adj_faces.numpy(),
                        self.particle_adjacency.v_adj_faces_offsets.numpy(),
                    )
                if self.particle_adjacency.v_adj_edges_offsets.size > 0:
                    edge_edge_filter_sets = build_edge_n_ring_edge_collision_filter(
                        self.particle_topological_contact_filter_threshold,
                        self.model.edge_indices.numpy(),
                        self.particle_adjacency.v_adj_edges.numpy(),
                        self.particle_adjacency.v_adj_edges_offsets.numpy(),
                    )

            if external_vertex_contact_filtering_map is not None:
                if v_tri_filter_sets is None:
                    v_tri_filter_sets = [set() for _ in range(self.model.particle_count)]
                for vertex_id, filter_set in external_vertex_contact_filtering_map.items():
                    v_tri_filter_sets[vertex_id].update(filter_set)

            if external_edge_contact_filtering_map is not None:
                if edge_edge_filter_sets is None:
                    edge_edge_filter_sets = [set() for _ in range(self.model.edge_indices.shape[0])]
                for edge_id, filter_set in external_edge_contact_filtering_map.items():
                    edge_edge_filter_sets[edge_id].update(filter_set)

            if v_tri_filter_sets is None:
                self.particle_vertex_triangle_contact_filtering_list = None
                self.particle_vertex_triangle_contact_filtering_list_offsets = None
            else:
                (
                    self.particle_vertex_triangle_contact_filtering_list,
                    self.particle_vertex_triangle_contact_filtering_list_offsets,
                ) = _set_to_csr(v_tri_filter_sets)
                self.particle_vertex_triangle_contact_filtering_list = wp.array(
                    self.particle_vertex_triangle_contact_filtering_list, dtype=int, device=self.device
                )
                self.particle_vertex_triangle_contact_filtering_list_offsets = wp.array(
                    self.particle_vertex_triangle_contact_filtering_list_offsets, dtype=int, device=self.device
                )

            if edge_edge_filter_sets is None:
                self.particle_edge_edge_contact_filtering_list = None
                self.particle_edge_edge_contact_filtering_list_offsets = None
            else:
                (
                    self.particle_edge_edge_contact_filtering_list,
                    self.particle_edge_edge_contact_filtering_list_offsets,
                ) = _set_to_csr(edge_edge_filter_sets)
                self.particle_edge_edge_contact_filtering_list = wp.array(
                    self.particle_edge_edge_contact_filtering_list, dtype=int, device=self.device
                )
                self.particle_edge_edge_contact_filtering_list_offsets = wp.array(
                    self.particle_edge_edge_contact_filtering_list_offsets, dtype=int, device=self.device
                )

    def compute_rigid_force_element_adjacency(self, model):
        """
        Build CSR adjacency between rigid bodies and joints.

        Returns an instance of RigidForceElementAdjacencyInfo with:
          - body_adj_joints: flattened joint ids
          - body_adj_joints_offsets: CSR offsets of size body_count + 1

        Notes:
            - Runs on CPU to avoid GPU atomics; kernels iterate serially over joints (dim=1).
            - When there are no joints, offsets are an all-zero array of length body_count + 1.
        """
        adjacency = RigidForceElementAdjacencyInfo()

        with wp.ScopedDevice("cpu"):
            # Build body-joint adjacency data (rigid-only)
            if model.joint_count > 0:
                joint_parent_cpu = model.joint_parent.to("cpu")
                joint_child_cpu = model.joint_child.to("cpu")

                num_body_adjacent_joints = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                wp.launch(
                    kernel=count_num_adjacent_joints,
                    inputs=[joint_parent_cpu, joint_child_cpu, num_body_adjacent_joints],
                    dim=1,
                )

                num_body_adjacent_joints = num_body_adjacent_joints.numpy()
                body_adjacent_joints_offsets = np.empty(shape=(model.body_count + 1,), dtype=wp.int32)
                body_adjacent_joints_offsets[1:] = np.cumsum(num_body_adjacent_joints)[:]
                body_adjacent_joints_offsets[0] = 0
                adjacency.body_adj_joints_offsets = wp.array(body_adjacent_joints_offsets, dtype=wp.int32)

                body_adjacent_joints_fill_count = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(num_body_adjacent_joints.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_joints,
                    inputs=[
                        joint_parent_cpu,
                        joint_child_cpu,
                        adjacency.body_adj_joints_offsets,
                        body_adjacent_joints_fill_count,
                        adjacency.body_adj_joints,
                    ],
                    dim=1,
                )
            else:
                # No joints: create offset array of zeros (size body_count + 1) so indexing works
                adjacency.body_adj_joints_offsets = wp.zeros(shape=(model.body_count + 1,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

    # =====================================================
    # Main Solver Methods
    # =====================================================

    def set_rigid_history_update(self, update: bool):
        """Set whether the next step() should update rigid solver history (warmstarts).

        This setting applies only to the next call to :meth:`step` and is then reset to ``True``.
        This is useful for substepping, where history update frequency might differ from the
        simulation step frequency (e.g. updating only on the first substep).

        Args:
            update: If True, update rigid warmstart state. If False, reuse previous.
        """
        self.update_rigid_history = update

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ):
        """Execute one simulation timestep using VBD (particles) and AVBD (rigid bodies).

        The solver follows a 3-phase structure:
        1. Initialize: Forward integrate particles and rigid bodies, detect collisions, warmstart penalties
        2. Iterate: Interleave particle VBD iterations and rigid body AVBD iterations
        3. Finalize: Update velocities and persistent state (Dahl friction)

        To control rigid substepping behavior (warmstart history), call
        :meth:`set_rigid_history_update`
        before calling this method. It defaults to ``True`` and is reset to ``True`` after each call.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Control inputs.
            contacts: Contact data produced by :meth:`Model.collide` (rigid-rigid and rigid-particle contacts).
                If None, rigid contact handling is skipped. Note that particle self-contact (if enabled) does not
                depend on this argument.
            dt: Time step size.
        """
        # Use and reset the rigid history update flag (warmstarts).
        update_rigid_history = self.update_rigid_history
        self.update_rigid_history = True

        self.initialize_rigid_bodies(state_in, state_out, contacts, dt, update_rigid_history)
        self.initialize_particles(state_in, state_out, dt)

        for iter_num in range(self.iterations):
            self.solve_rigid_body_iteration(state_in, state_out, contacts, dt)
            self.solve_particle_iteration(state_in, state_out, contacts, dt, iter_num)

        self.finalize_rigid_bodies(state_out, dt)
        self.finalize_particles(state_out, dt)

    def penetration_free_truncation(self, particle_q_out=None):
        """
        Modify displacements_in in-place, also modify particle_q if its not None

        """
        if not self.particle_enable_self_contact:
            self.truncation_ts.fill_(1.0)
            wp.launch(
                kernel=apply_truncation_ts,
                dim=self.model.particle_count,
                inputs=[
                    self.pos_prev_collision_detection,  # pos: wp.array(dtype=wp.vec3),
                    self.particle_displacements,  # displacement_in: wp.array(dtype=wp.vec3),
                    self.truncation_ts,  # truncation_ts: wp.array(dtype=float),
                ],
                outputs=[
                    self.particle_displacements,  # displacement_out: wp.array(dtype=wp.vec3),
                    particle_q_out,  # pos_out: wp.array(dtype=wp.vec3),
                    wp.inf,  # max_displacement: float
                ],
                device=self.device,
            )

        else:
            ##  parallel by collision and atomic operation
            self.truncation_ts.fill_(1.0)
            wp.launch(
                kernel=apply_planar_truncation_parallel_by_collision,
                inputs=[
                    self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array(dtype=wp.vec3),
                    self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.trimesh_collision_info,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    self.particle_conservative_bound_relaxation * 2,
                ],
                outputs=[
                    self.truncation_ts,
                ],
                dim=self.particle_self_contact_evaluation_kernel_launch_size,
                device=self.device,
            )

            wp.launch(
                kernel=apply_truncation_ts,
                dim=self.model.particle_count,
                inputs=[
                    self.pos_prev_collision_detection,  # pos: wp.array(dtype=wp.vec3),
                    self.particle_displacements,  # displacement_in: wp.array(dtype=wp.vec3),
                    self.truncation_ts,  # truncation_ts: wp.array(dtype=float),
                    self.particle_displacements,  # displacement_out: wp.array(dtype=wp.vec3),
                    particle_q_out,  # pos_out: wp.array(dtype=wp.vec3),
                    self.particle_self_contact_margin
                    * self.particle_conservative_bound_relaxation,  # max_displacement: float
                ],
                device=self.device,
            )

    def initialize_particles(self, state_in: State, state_out: State, dt: float):
        """Initialize particle positions for the VBD iteration."""
        model = self.model

        # Early exit if no particles
        if model.particle_count == 0:
            return

        # Collision detection before initialization to compute conservative bounds
        self.collision_detection_penetration_free(state_in)

        model = self.model

        wp.launch(
            kernel=forward_step,
            inputs=[
                dt,
                model.gravity,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
            ],
            outputs=[
                self.inertia,
                self.particle_displacements,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        self.penetration_free_truncation(state_out.particle_q)

    def initialize_rigid_bodies(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
        update_rigid_history: bool,
    ):
        """Initialize rigid body states for AVBD solver (pre-iteration phase).

        Performs forward integration and initializes contact-related AVBD state when contacts are provided.

        If ``contacts`` is None, rigid contact-related work is skipped:
        no per-body contact adjacency is built, and no contact penalties are warmstarted.
        """
        model = self.model

        # ---------------------------
        # Rigid-only initialization
        # ---------------------------
        if model.body_count > 0 and not self.integrate_with_external_rigid_solver:
            # Forward integrate rigid bodies
            wp.launch(
                kernel=forward_step_rigid_bodies,
                inputs=[
                    dt,
                    model.gravity,
                    model.body_world,
                    state_in.body_f,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    state_in.body_q,  # input/output
                    state_in.body_qd,  # input/output
                ],
                outputs=[
                    self.body_q_prev,
                    self.body_inertia_q,
                ],
                dim=model.body_count,
                device=self.device,
            )

            if update_rigid_history:
                # Contact warmstarts / adjacency are optional: skip completely if contacts=None.
                if contacts is not None:
                    # Use the Contacts buffer capacity as launch dimension
                    contact_launch_dim = contacts.rigid_contact_max

                    # Build per-body contact lists once per step
                    # Build body-body (rigid-rigid) contact lists
                    self.body_body_contact_counts.zero_()
                    wp.launch(
                        kernel=build_body_body_contact_lists,
                        dim=contact_launch_dim,
                        inputs=[
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_body,
                            self.body_body_contact_buffer_pre_alloc,
                        ],
                        outputs=[
                            self.body_body_contact_counts,
                            self.body_body_contact_indices,
                        ],
                        device=self.device,
                    )

                    # Warmstart AVBD body-body contact penalties and pre-compute material properties
                    wp.launch(
                        kernel=warmstart_body_body_contacts,
                        inputs=[
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_ke,
                            model.shape_material_kd,
                            model.shape_material_mu,
                            self.k_start_body_contact,
                        ],
                        outputs=[
                            self.body_body_contact_penalty_k,
                            self.body_body_contact_material_ke,
                            self.body_body_contact_material_kd,
                            self.body_body_contact_material_mu,
                        ],
                        dim=contact_launch_dim,
                        device=self.device,
                    )

                # Warmstart AVBD penalty parameters for joints using the same cadence
                # as rigid history updates.
                if model.joint_count > 0:
                    wp.launch(
                        kernel=warmstart_joints,
                        inputs=[
                            self.joint_penalty_k_max,
                            self.joint_penalty_k_min,
                            self.avbd_gamma,
                            self.joint_penalty_k,  # input/output
                        ],
                        dim=self.joint_constraint_count,
                        device=self.device,
                    )

            # Compute Dahl hysteresis parameters for cable bending (once per timestep, frozen during iterations)
            if self.enable_dahl_friction and model.joint_count > 0:
                wp.launch(
                    kernel=compute_cable_dahl_parameters,
                    inputs=[
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.joint_constraint_start,
                        self.joint_penalty_k_max,
                        self.body_q_prev,  # Use previous body transforms (start of step) for linearization
                        model.body_q,  # rest body transforms
                        self.joint_sigma_prev,
                        self.joint_kappa_prev,
                        self.joint_dkappa_prev,
                        self.joint_dahl_eps_max,
                        self.joint_dahl_tau,
                    ],
                    outputs=[
                        self.joint_sigma_start,
                        self.joint_C_fric,
                    ],
                    dim=model.joint_count,
                    device=self.device,
                )

        # ---------------------------
        # Body-particle interaction
        # ---------------------------
        if model.particle_count > 0 and update_rigid_history and contacts is not None:
            # Build body-particle (rigid-particle) contact lists only when SolverVBD
            # is integrating rigid bodies itself; the external rigid solver path
            # does not use these per-body adjacency structures. Also skip if there
            # are no rigid bodies in the model.
            if not self.integrate_with_external_rigid_solver and model.body_count > 0:
                self.body_particle_contact_counts.zero_()
                wp.launch(
                    kernel=build_body_particle_contact_lists,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_shape,
                        model.shape_body,
                        self.body_particle_contact_buffer_pre_alloc,
                    ],
                    outputs=[
                        self.body_particle_contact_counts,
                        self.body_particle_contact_indices,
                    ],
                    device=self.device,
                )

            # Warmstart AVBD body-particle contact penalties and pre-compute material properties.
            # This is useful both when SolverVBD integrates rigid bodies and when an external
            # rigid solver is used, since cloth-rigid soft contacts still rely on these penalties.
            soft_contact_launch_dim = contacts.soft_contact_max
            wp.launch(
                kernel=warmstart_body_particle_contacts,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    model.soft_contact_ke,
                    model.soft_contact_kd,
                    model.soft_contact_mu,
                    model.shape_material_ke,
                    model.shape_material_kd,
                    model.shape_material_mu,
                    self.k_start_body_contact,
                ],
                outputs=[
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_ke,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                ],
                dim=soft_contact_launch_dim,
                device=self.device,
            )

    def solve_particle_iteration(
        self, state_in: State, state_out: State, contacts: Contacts | None, dt: float, iter_num: int
    ):
        """Solve one VBD iteration for particles."""
        model = self.model

        # Select rigid-body poses for particle-rigid contact evaluation
        if self.integrate_with_external_rigid_solver:
            body_q_for_particles = state_out.body_q
            body_q_prev_for_particles = state_in.body_q
            body_qd_for_particles = state_out.body_qd
        else:
            body_q_for_particles = state_in.body_q
            if model.body_count > 0:
                body_q_prev_for_particles = self.body_q_prev
            else:
                body_q_prev_for_particles = None
            body_qd_for_particles = state_in.body_qd

        # Early exit if no particles
        if model.particle_count == 0:
            return

        # Update collision detection if needed (penetration-free mode only)
        if self.particle_enable_self_contact:
            if (self.particle_collision_detection_interval == 0 and iter_num == 0) or (
                self.particle_collision_detection_interval >= 1
                and iter_num % self.particle_collision_detection_interval == 0
            ):
                self.collision_detection_penetration_free(state_out)

        # Zero out forces and hessians
        self.particle_forces.zero_()
        self.particle_hessians.zero_()

        # Iterate over color groups
        for color in range(len(self.model.particle_color_groups)):
            if contacts is not None:
                wp.launch(
                    kernel=accumulate_particle_body_contact_force_and_hessian,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_out.particle_q,
                        self.model.particle_colors,
                        # body-particle contact
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        self.model.particle_radius,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_count,
                        contacts.soft_contact_max,
                        self.model.shape_material_mu,
                        self.model.shape_body,
                        body_q_for_particles,
                        body_q_prev_for_particles,
                        body_qd_for_particles,
                        self.model.body_com,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device,
                )

            if model.spring_count:
                wp.launch(
                    kernel=accumulate_spring_force_and_hessian,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_out.particle_q,
                        self.model.particle_colors,
                        model.spring_count,
                        self.model.spring_indices,
                        self.model.spring_rest_length,
                        self.model.spring_stiffness,
                        self.model.spring_damping,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    dim=model.spring_count,
                    device=self.device,
                )

            if self.particle_enable_self_contact:
                wp.launch(
                    kernel=accumulate_self_contact_force_and_hessian,
                    dim=self.particle_self_contact_evaluation_kernel_launch_size,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_out.particle_q,
                        self.model.particle_colors,
                        self.model.tri_indices,
                        self.model.edge_indices,
                        # self-contact
                        self.trimesh_collision_info,
                        self.particle_self_contact_radius,
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device,
                    max_blocks=self.model.device.sm_count,
                )
            if self.use_particle_tile_solve:
                wp.launch(
                    kernel=solve_elasticity_tile,
                    dim=self.model.particle_color_groups[color].size * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.particle_q_prev,
                        state_out.particle_q,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.model.tet_indices,
                        self.model.tet_poses,
                        self.model.tet_materials,
                        self.particle_adjacency,
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    outputs=[
                        self.particle_displacements,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    kernel=solve_elasticity,
                    dim=self.model.particle_color_groups[color].size,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.particle_q_prev,
                        state_out.particle_q,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.model.tet_indices,
                        self.model.tet_poses,
                        self.model.tet_materials,
                        self.particle_adjacency,
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    outputs=[
                        self.particle_displacements,
                    ],
                    device=self.device,
                )
            self.penetration_free_truncation(state_out.particle_q)

    def solve_rigid_body_iteration(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float):
        """Solve one AVBD iteration for rigid bodies (per-iteration phase).

        Accumulates contact and joint forces/hessians, solves 6x6 rigid body systems per color,
        and updates AVBD penalty parameters (dual update).
        """
        model = self.model

        # Early exit if no rigid bodies
        if model.body_count == 0:
            return

        # If rigid bodies are integrated by an external solver, skip the AVBD rigid-body
        # solve but still update body-particle soft-contact penalties so that adaptive
        # AVBD stiffness is used for cloth-rigid interaction.
        if self.integrate_with_external_rigid_solver:
            if model.particle_count > 0 and contacts is not None:
                soft_contact_launch_dim = contacts.soft_contact_max
                wp.launch(
                    kernel=update_duals_body_particle_contacts,
                    dim=soft_contact_launch_dim,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        # Rigid poses come from the external solver when
                        # integrate_with_external_rigid_solver=True
                        state_out.body_q,
                        self.body_particle_contact_material_ke,
                        self.avbd_beta,
                        self.body_particle_contact_penalty_k,  # input/output
                    ],
                    device=self.device,
                )
            return

        # Zero out forces and hessians
        self.body_torques.zero_()
        self.body_forces.zero_()
        self.body_hessian_aa.zero_()
        self.body_hessian_al.zero_()
        self.body_hessian_ll.zero_()

        body_color_groups = model.body_color_groups

        # Gauss-Seidel-style per-color updates
        for color in range(len(body_color_groups)):
            color_group = body_color_groups[color]

            # Gauss-Seidel contact accumulation: evaluate contacts for bodies in this color
            # Accumulate body-particle forces and Hessians on bodies (per-body, per-color)
            if model.particle_count > 0 and contacts is not None:
                wp.launch(
                    kernel=accumulate_body_particle_contacts_per_body,
                    dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        # particle state
                        state_in.particle_q,
                        self.particle_q_prev,
                        model.particle_radius,
                        # rigid body state
                        self.body_q_prev,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        model.body_inv_mass,
                        # AVBD body-particle soft contact penalties and material properties
                        self.friction_epsilon,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        # soft contact data (body-particle contacts)
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        # shape/material data
                        model.shape_material_mu,
                        model.shape_body,
                        # per-body adjacency (body-particle contacts)
                        self.body_particle_contact_buffer_pre_alloc,
                        self.body_particle_contact_counts,
                        self.body_particle_contact_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

            # Accumulate body-body (rigid-rigid) contact forces and Hessians on bodies (per-body, per-color)
            if contacts is not None:
                wp.launch(
                    kernel=accumulate_body_body_contacts_per_body,
                    dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        self.body_q_prev,
                        state_in.body_q,
                        model.body_com,
                        model.body_inv_mass,
                        self.friction_epsilon,
                        self.body_body_contact_penalty_k,
                        self.body_body_contact_material_kd,
                        self.body_body_contact_material_mu,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_thickness0,
                        contacts.rigid_contact_thickness1,
                        model.shape_body,
                        self.body_body_contact_buffer_pre_alloc,
                        self.body_body_contact_counts,
                        self.body_body_contact_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

            wp.launch(
                kernel=solve_rigid_body,
                inputs=[
                    dt,
                    color_group,
                    state_in.body_q,
                    self.body_q_prev,
                    model.body_q,
                    model.body_mass,
                    model.body_inv_mass,
                    model.body_inertia,
                    self.body_inertia_q,
                    model.body_com,
                    self.rigid_adjacency,
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    self.joint_constraint_start,
                    self.joint_penalty_k,
                    self.joint_penalty_kd,
                    self.joint_sigma_start,
                    self.joint_C_fric,
                    self.body_forces,
                    self.body_torques,
                    self.body_hessian_ll,
                    self.body_hessian_al,
                    self.body_hessian_aa,
                ],
                outputs=[
                    state_out.body_q,
                ],
                dim=color_group.size,
                device=self.device,
            )

            wp.launch(
                kernel=copy_rigid_body_transforms_back,
                inputs=[color_group, state_out.body_q],
                outputs=[state_in.body_q],
                dim=color_group.size,
                device=self.device,
            )

        if contacts is not None:
            # AVBD dual update: update adaptive penalties based on constraint violation
            # Update body-body (rigid-rigid) contact penalties
            contact_launch_dim = contacts.rigid_contact_max
            wp.launch(
                kernel=update_duals_body_body_contacts,
                dim=contact_launch_dim,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_thickness0,
                    contacts.rigid_contact_thickness1,
                    model.shape_body,
                    state_out.body_q,
                    self.body_body_contact_material_ke,
                    self.avbd_beta,
                    self.body_body_contact_penalty_k,  # input/output
                ],
                device=self.device,
            )

            # Update body-particle contact penalties
            if model.particle_count > 0:
                soft_contact_launch_dim = contacts.soft_contact_max
                wp.launch(
                    kernel=update_duals_body_particle_contacts,
                    dim=soft_contact_launch_dim,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        # Rigid poses come from SolverVBD itself when
                        # integrate_with_external_rigid_solver=False
                        state_in.body_q,
                        self.body_particle_contact_material_ke,
                        self.avbd_beta,
                        self.body_particle_contact_penalty_k,  # input/output
                    ],
                    device=self.device,
                )

        # Update joint penalties at new positions
        wp.launch(
            kernel=update_duals_joint,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                self.joint_constraint_start,
                self.joint_penalty_k_max,
                state_out.body_q,
                model.body_q,
                self.avbd_beta,
                self.joint_penalty_k,  # input/output
            ],
            device=self.device,
        )

    def finalize_particles(self, state_out: State, dt: float):
        """Finalize particle velocities after VBD iterations."""
        # Early exit if no particles
        if self.model.particle_count == 0:
            return

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def finalize_rigid_bodies(self, state_out: State, dt: float):
        """Finalize rigid body velocities and Dahl friction state after AVBD iterations (post-iteration phase).

        Updates rigid body velocities using BDF1 and updates Dahl hysteresis state for cable bending.
        """
        model = self.model

        # Early exit if no rigid bodies or rigid bodies are driven by an external solver
        if model.body_count == 0 or self.integrate_with_external_rigid_solver:
            return

        # Velocity update (BDF1) after all iterations
        wp.launch(
            kernel=update_body_velocity,
            inputs=[
                dt,
                state_out.body_q,
                model.body_com,
                self.body_q_prev,  # input/output
            ],
            outputs=[state_out.body_qd],
            dim=model.body_count,
            device=self.device,
        )

        # Update Dahl hysteresis state after solver convergence (for next timestep's memory)
        if self.enable_dahl_friction and model.joint_count > 0:
            wp.launch(
                kernel=update_cable_dahl_state,
                inputs=[
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    self.joint_constraint_start,
                    self.joint_penalty_k_max,
                    state_out.body_q,
                    model.body_q,
                    self.joint_dahl_eps_max,
                    self.joint_dahl_tau,
                    self.joint_sigma_prev,  # input/output
                    self.joint_kappa_prev,  # input/output
                    self.joint_dkappa_prev,  # input/output
                ],
                dim=model.joint_count,
                device=self.device,
            )

    def collision_detection_penetration_free(self, current_state: State):
        self.pos_prev_collision_detection.assign(current_state.particle_q)
        self.particle_displacements.zero_()

        if self.particle_enable_self_contact:
            self.trimesh_collision_detector.refit(current_state.particle_q)
            self.trimesh_collision_detector.vertex_triangle_collision_detection(
                self.particle_self_contact_margin,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )
            self.trimesh_collision_detector.edge_edge_collision_detection(
                self.particle_self_contact_margin,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (newton.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.particle_enable_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)
