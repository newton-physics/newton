# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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
    ModelBuilder,
    State,
)
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from ..xpbd.kernels import apply_joint_forces
from .particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
    ParticleForceElementAdjacencyInfo,
    # Adjacency building kernels
    _count_num_adjacent_edges,
    _count_num_adjacent_faces,
    _count_num_adjacent_springs,
    _count_num_adjacent_tets,
    _fill_adjacent_edges,
    _fill_adjacent_faces,
    _fill_adjacent_springs,
    _fill_adjacent_tets,
    # Topological filtering helper functions
    accumulate_particle_body_contact_force_and_hessian,
    accumulate_self_contact_force_and_hessian,
    accumulate_spring_force_and_hessian,
    # Planar DAT (Divide and Truncate) kernels
    apply_planar_truncation_parallel_by_collision,
    apply_truncation_ts,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    # Solver kernels (particle VBD)
    forward_step,
    set_to_csr,
    solve_elasticity,
    solve_elasticity_tile,
    update_velocity,
)
from .rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    RigidContactHistory,
    RigidForceElementAdjacencyInfo,
    _count_num_adjacent_joints,
    _fill_adjacent_joints,
    accumulate_body_body_contacts_per_body,
    accumulate_body_particle_contacts_per_body,
    build_body_body_contact_lists,
    build_body_particle_contact_lists,
    check_contact_overflow,
    compute_cable_dahl_parameters,
    compute_rigid_contact_forces,
    forward_step_rigid_bodies,
    init_body_body_contact_materials,
    init_body_body_contacts_avbd,
    init_body_particle_contacts,
    snapshot_body_body_contact_history,
    solve_rigid_body,
    step_body_body_contact_C0_lambda,
    step_joint_C0_lambda,
    update_body_velocity,
    update_cable_dahl_state,
    update_duals_body_body_contacts,
    update_duals_body_particle_contacts,
    update_duals_joint,
)
from .tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)

__all__ = ["SolverVBD"]


class SolverVBD(SolverBase):
    """An implicit solver using Vertex Block Descent (VBD) for particles and Augmented VBD (AVBD) for rigid bodies.

    This unified solver supports:
        - Particle simulation (cloth, soft bodies) using the VBD algorithm
        - Rigid body simulation (joints, contacts) using the AVBD algorithm
        - Coupled particle-rigid body systems

    For rigid bodies, the AVBD algorithm uses penalty stiffness that is fixed by
    default (``rigid_avbd_beta=0``) or ramped per iteration from ``k_start`` seeds
    by the AVBD beta parameters. Hard contacts and hard joint slots additionally
    use augmented-Lagrangian state.

    Non-cable structural joint slots default to **hard mode** (augmented Lagrangian
    with persistent lambda and C0 stabilization). Cable stretch and bend default to
    **soft mode**. The hard/soft mode can be changed per slot via :meth:`set_joint_constraint_mode`.

    Joint limitations:
        - Supported joint types: BALL, FIXED, FREE, REVOLUTE, PRISMATIC, D6, CABLE.
          DISTANCE joints are not supported.
        - :attr:`~newton.Model.joint_enabled` is supported for all joint types.
        - :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd` are supported
          for REVOLUTE, PRISMATIC, D6 (as drives), and CABLE (as stretch/bend stiffness and damping).
          VBD interprets ``kd`` as a dimensionless Rayleigh coefficient (``D = kd * ke``).
        - :attr:`~newton.Model.joint_limit_lower`/:attr:`~newton.Model.joint_limit_upper` and
          :attr:`~newton.Model.joint_limit_ke`/:attr:`~newton.Model.joint_limit_kd` are supported
          for REVOLUTE, PRISMATIC, and D6 joints. The default ``limit_kd`` in
          :class:`~newton.ModelBuilder.JointDofConfig` is ``1e1``, which under VBD's Rayleigh
          convention (``D = kd * ke``) can produce excessive damping. When using joint limits
          with VBD, explicitly set ``limit_kd`` to a small value.
        - :attr:`~newton.Control.joint_f` (feedforward forces) is supported.
        - Not supported: :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          :attr:`~newton.Model.joint_target_mode`, equality constraints, mimic constraints.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Buffer sizing:
        SolverVBD pre-allocates contact state from capacities populated by
        ``CollisionPipeline`` when available; otherwise, the first ``step()``
        lazily sizes buffers from ``Contacts``. During CUDA graph recording, lazy
        resize is supported only when Warp mempool is enabled; otherwise the
        solver raises with guidance to pre-size before capture.

    References:
        - Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block Descent. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages.
          https://doi.org/10.1145/3658179
        - Chris Giles, Elie Diaz, and Cem Yuksel. 2025. Augmented Vertex Block Descent. ACM Trans. Graph. 44, 4, Article 90 (August 2025), 12 pages.
          https://doi.org/10.1145/3731195

    Note:
        `SolverVBD` requires coloring for each system it solves:

        - Particle coloring: :attr:`newton.Model.particle_color_groups` (required if particles are present)
        - Rigid body coloring: :attr:`newton.Model.body_color_groups` (required if rigid bodies are integrated by VBD)

        Call :meth:`newton.ModelBuilder.color` to automatically color both particles and rigid bodies.

        VBD uses ``model.body_q`` as the structural rest pose and reads
        ``model.joint_q`` for drive/limit rest-angle offsets. The body
        transforms must match the joint angles at solver creation time
        (see example below).

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
        contacts = model.contacts()

        # Simulation loop
        for i in range(100):
            model.collide(state_in, contacts)  # Update contacts
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    class JointSlot:
        """Named constraint slot indices for :meth:`set_joint_constraint_mode`.

        The first two solver constraint slots are structural where present:
          - CABLE: LINEAR/STRETCH -> stretch, ANGULAR/BEND -> bend
          - BALL: LINEAR only
          - FIXED/REVOLUTE/PRISMATIC/D6: LINEAR and ANGULAR

        Drive/limit slots start at slot 2 and are not represented here.
        STRETCH and BEND are cable-only aliases for LINEAR and ANGULAR.
        """

        LINEAR = 0
        ANGULAR = 1
        STRETCH = 0
        BEND = 1

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
        # Rigid body parameters — AVBD hyperparameters
        rigid_avbd_alpha: float = 0.95,  # C0 stabilization strength (C_stab = C - alpha * C0)
        rigid_avbd_joint_alpha: float | None = None,  # Joint alpha override; None uses rigid_avbd_alpha
        rigid_avbd_contact_alpha: float | None = None,  # Body-body contact alpha; None selects default
        rigid_avbd_beta: float = 0.0,  # Penalty ramp rate per iteration (0 = fixed-k)
        rigid_avbd_linear_beta: float | None = None,  # Linear beta override; None uses rigid_avbd_beta
        rigid_avbd_angular_beta: float | None = None,  # Angular beta override; None uses rigid_avbd_beta
        rigid_avbd_gamma: float = 0.999,  # Per-step decay for penalty k and persisted hard-mode lambda
        # Rigid body - contacts
        rigid_contact_hard: bool = True,  # Body-body contacts: hard=AL duals+C0, soft=penalty only
        rigid_contact_history: bool = False,  # Body-body contact warm-start (hard: k+duals+anchors; soft: k)
        rigid_contact_k_start: float = 1.0e2,  # Body-body/body-particle penalty seed when ramping is enabled
        rigid_body_contact_buffer_size: int = 64,  # Per-body body-body contact list capacity
        rigid_body_particle_contact_buffer_size: int = 256,  # Per-body particle-contact list capacity
        # Rigid body - joints
        rigid_joint_linear_ke: float = 1.0e5,  # Penalty stiffness ceiling for structural linear joint constraints
        rigid_joint_angular_ke: float = 1.0e5,  # Penalty stiffness ceiling for structural angular joint constraints
        rigid_joint_linear_k_start: float = 1.0e2,  # Linear penalty seed (used when linear beta > 0)
        rigid_joint_angular_k_start: float = 1.0e1,  # Angular penalty seed (used when angular beta > 0)
        rigid_joint_linear_kd: float = 0.0,  # Rayleigh damping for non-cable linear joint constraints
        rigid_joint_angular_kd: float = 0.0,  # Rayleigh damping for non-cable angular joint constraints
        rigid_enable_dahl_friction: bool | None = None,  # Deprecated: auto-detected from model attributes
    ):
        """
        Args:
            model: The `Model` object used to initialize the integrator. Must be identical to the `Model` object passed
                to the `step` function.

            Common parameters:

            iterations: Number of VBD iterations per step.
            friction_epsilon: Threshold to smooth small relative velocities in friction computation (used for both particle
                and rigid body contacts).
            integrate_with_external_rigid_solver: Indicator for coupled rigid body-cloth simulation. When set to `True`,
                the solver assumes rigid bodies are integrated by an external solver (one-way coupling).

            Particle parameters:

            particle_enable_self_contact: Whether to enable self-contact detection for particles.
            particle_self_contact_radius: The radius used for self-contact detection. This is the distance at which
                vertex-triangle pairs and edge-edge pairs will start to interact with each other.
            particle_self_contact_margin: The margin used for self-contact detection. This is the distance at which
                vertex-triangle pairs and edge-edge will be considered in contact generation. It should be larger than
                `particle_self_contact_radius` to avoid missing contacts.
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

            rigid_avbd_alpha: C0 stabilization strength (C_stab = C - alpha * C0). Range: [0, 1].
                Used as the default joint alpha. Also used as the default body-body contact alpha
                when both ``rigid_contact_hard`` and ``rigid_contact_history`` are enabled; otherwise
                body-body contacts default to zero alpha.
            rigid_avbd_joint_alpha: Joint-specific alpha override. ``None`` (default)
                uses ``rigid_avbd_alpha``.
            rigid_avbd_contact_alpha: Body-body contact alpha override. ``None`` (default)
                uses ``rigid_avbd_alpha`` when both ``rigid_contact_hard`` and
                ``rigid_contact_history`` are enabled, ``0.0`` otherwise.
            rigid_avbd_beta: Penalty ramp rate per AVBD iteration. ``0`` (default) disables
                ramping (fixed-k). Set to e.g. ``1e5`` for ramping. Used for both linear and
                angular constraints unless overridden. Note: linear (meters) and angular
                (radians) constraints have different units, so the overrides should be used
                for production tuning.
            rigid_avbd_linear_beta: Linear beta override for linear constraints (meters).
                ``None`` (default) uses ``rigid_avbd_beta``.
            rigid_avbd_angular_beta: Angular beta override for angular constraints (radians).
                ``None`` (default) uses ``rigid_avbd_beta``.
            rigid_avbd_gamma: Per-step decay factor for penalty k and persisted hard-mode lambda. Lower values
                decay faster, improving stability at the cost of slower convergence.
            rigid_contact_hard: Whether body-body rigid contacts use hard mode (augmented Lagrangian with
                persistent lambda and C0 stabilization) or soft mode (penalty only).
            rigid_contact_history: Whether to persist body-body contact state across steps using
                ``Contacts.rigid_contact_match_index`` from the collision pipeline. For hard contacts,
                restores lambda, penalty k, and sticky contact anchors; C0 is recomputed each step.
                For soft contacts, only restored penalty k affects the solve (useful with ramping).
                Requires contacts with ``rigid_contact_match_index`` populated; use
                ``CollisionPipeline(contact_matching="latest")`` for VBD warm-starting. Ignored
                when ``integrate_with_external_rigid_solver=True`` or ``model.body_count == 0``.
            rigid_contact_k_start: Body-body and body-particle contact penalty seed for AVBD ramping. Used when
                ``rigid_avbd_linear_beta`` (or ``rigid_avbd_beta`` fallback) is greater than zero.
                When the linear beta is 0, k is fixed at the contact stiffness regardless of this value.
            rigid_body_contact_buffer_size: Max body-body contacts per rigid body for per-body contact lists.
            rigid_body_particle_contact_buffer_size: Max body-particle contacts tracked per rigid body.
            rigid_joint_linear_ke: Penalty stiffness ceiling for non-cable structural linear joint slots.
            rigid_joint_angular_ke: Penalty stiffness ceiling for non-cable structural angular joint slots.
            rigid_joint_linear_k_start: Linear penalty seed for AVBD ramping. Used when
                ``rigid_avbd_linear_beta`` (or ``rigid_avbd_beta`` fallback) is greater than zero.
                When the linear beta is 0, k is fixed at the joint stiffness regardless of this value.
            rigid_joint_angular_k_start: Angular penalty seed for AVBD ramping. Used when
                ``rigid_avbd_angular_beta`` (or ``rigid_avbd_beta`` fallback) is greater than zero.
                When the angular beta is 0, k is fixed at the joint stiffness regardless of this value.
            rigid_joint_linear_kd: Rayleigh damping coefficient for non-cable linear joint constraints.
                Negative values are clamped to 0.
            rigid_joint_angular_kd: Rayleigh damping coefficient for non-cable angular joint constraints.
                Negative values are clamped to 0.
            rigid_enable_dahl_friction: Deprecated and ignored. Dahl friction is auto-detected
                from ``model.vbd.dahl_eps_max`` / ``model.vbd.dahl_tau``.

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
            - Dahl hysteresis friction for cable bending is auto-detected from custom model attributes
              ``model.vbd.dahl_eps_max`` and ``model.vbd.dahl_tau`` (set via
              ``SolverVBD.register_custom_attributes``).

        """
        if rigid_enable_dahl_friction is not None:
            warnings.warn(
                "rigid_enable_dahl_friction is deprecated and ignored. "
                "Dahl friction is now auto-detected from model attributes "
                "(model.vbd.dahl_eps_max / model.vbd.dahl_tau). "
                "To disable for a joint, set dahl_eps_max=0.",
                DeprecationWarning,
                stacklevel=2,
            )

        if rigid_avbd_beta < 0:
            raise ValueError(f"rigid_avbd_beta must be >= 0, got {rigid_avbd_beta}")
        rigid_avbd_linear_beta = rigid_avbd_linear_beta if rigid_avbd_linear_beta is not None else rigid_avbd_beta
        rigid_avbd_angular_beta = rigid_avbd_angular_beta if rigid_avbd_angular_beta is not None else rigid_avbd_beta

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
            rigid_avbd_alpha,
            rigid_avbd_linear_beta,
            rigid_avbd_angular_beta,
            rigid_avbd_gamma,
            rigid_avbd_joint_alpha,
            rigid_avbd_contact_alpha,
            rigid_contact_hard,
            rigid_contact_history,
            rigid_contact_k_start,
            rigid_body_contact_buffer_size,
            rigid_body_particle_contact_buffer_size,
            rigid_joint_linear_ke,
            rigid_joint_angular_ke,
            rigid_joint_linear_k_start,
            rigid_joint_angular_k_start,
            rigid_joint_linear_kd,
            rigid_joint_angular_kd,
        )

        # Controls whether the next step() refreshes contact state derived from
        # the Contacts buffer or reuses the current rigid/body-particle contact state.
        # Defaults to True and is reset to True when consumed by step().
        self._update_rigid_history = True

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
        self.particle_adjacency = self._compute_particle_force_element_adjacency().to(self.device)

        # Self-contact settings
        self.particle_enable_self_contact = particle_enable_self_contact
        self.particle_self_contact_radius = particle_self_contact_radius
        self.particle_self_contact_margin = particle_self_contact_margin
        self.particle_q_rest = model.particle_q

        # Tile solve settings
        if model.device.is_cpu and particle_enable_tile_solve and wp.config.verbose:
            print("Info: Tiled solve requires model.device='cuda'. Tiled solve is disabled.")

        self.use_particle_tile_solve = particle_enable_tile_solve and model.device.is_cuda

        if particle_enable_self_contact:
            if particle_self_contact_margin < particle_self_contact_radius:
                raise ValueError(
                    "particle_self_contact_margin is smaller than particle_self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make particle_self_contact_margin 1.5-2 times larger than particle_self_contact_radius."
                )

            self.particle_conservative_bound_relaxation = particle_conservative_bound_relaxation
            self.particle_conservative_bounds = wp.zeros((model.particle_count,), dtype=float, device=self.device)

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=particle_vertex_contact_buffer_size,
                edge_collision_buffer_pre_alloc=particle_edge_contact_buffer_size,
                edge_edge_parallel_epsilon=particle_edge_parallel_epsilon,
            )

            self._compute_particle_contact_filtering_list(
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
            self.particle_self_contact_evaluation_kernel_launch_size = None

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
        rigid_avbd_alpha: float,
        rigid_avbd_linear_beta: float,
        rigid_avbd_angular_beta: float,
        rigid_avbd_gamma: float,
        rigid_avbd_joint_alpha: float | None,
        rigid_avbd_contact_alpha: float | None,
        rigid_contact_hard: bool,
        rigid_contact_history: bool,
        rigid_contact_k_start: float,
        rigid_body_contact_buffer_size: int,
        rigid_body_particle_contact_buffer_size: int,
        rigid_joint_linear_ke: float,
        rigid_joint_angular_ke: float,
        rigid_joint_linear_k_start: float,
        rigid_joint_angular_k_start: float,
        rigid_joint_linear_kd: float,
        rigid_joint_angular_kd: float,
    ) -> None:
        """Initialize rigid body-specific AVBD data structures and settings.

        This includes:
          - Rigid-only AVBD state (joints, body-body contacts, Dahl friction)
          - Shared interaction state for body-particle (rigid-particle) soft contacts
        """
        # AVBD penalty parameters
        if not (0.0 <= rigid_avbd_alpha <= 1.0):
            raise ValueError(f"rigid_avbd_alpha must be in [0, 1], got {rigid_avbd_alpha}")
        if rigid_avbd_joint_alpha is not None and not (0.0 <= rigid_avbd_joint_alpha <= 1.0):
            raise ValueError(f"rigid_avbd_joint_alpha must be in [0, 1], got {rigid_avbd_joint_alpha}")
        if rigid_avbd_contact_alpha is not None and not (0.0 <= rigid_avbd_contact_alpha <= 1.0):
            raise ValueError(f"rigid_avbd_contact_alpha must be in [0, 1], got {rigid_avbd_contact_alpha}")
        if rigid_avbd_linear_beta < 0:
            raise ValueError(f"rigid_avbd_linear_beta must be >= 0, got {rigid_avbd_linear_beta}")
        if rigid_avbd_angular_beta < 0:
            raise ValueError(f"rigid_avbd_angular_beta must be >= 0, got {rigid_avbd_angular_beta}")
        if not (0.0 <= rigid_avbd_gamma <= 1.0):
            raise ValueError(f"rigid_avbd_gamma must be in [0, 1], got {rigid_avbd_gamma}")
        if rigid_contact_k_start < 0:
            raise ValueError(f"rigid_contact_k_start must be >= 0, got {rigid_contact_k_start}")
        if rigid_joint_linear_k_start < 0:
            raise ValueError(f"rigid_joint_linear_k_start must be >= 0, got {rigid_joint_linear_k_start}")
        if rigid_joint_angular_k_start < 0:
            raise ValueError(f"rigid_joint_angular_k_start must be >= 0, got {rigid_joint_angular_k_start}")
        if rigid_joint_linear_ke < 0:
            raise ValueError(f"rigid_joint_linear_ke must be >= 0, got {rigid_joint_linear_ke}")
        if rigid_joint_angular_ke < 0:
            raise ValueError(f"rigid_joint_angular_ke must be >= 0, got {rigid_joint_angular_ke}")
        self.rigid_avbd_gamma = rigid_avbd_gamma
        self.rigid_contact_k_start_value = -1.0 if rigid_avbd_linear_beta == 0.0 else float(rigid_contact_k_start)
        self.rigid_joint_linear_k_start = rigid_joint_linear_k_start if rigid_avbd_linear_beta > 0.0 else None
        self.rigid_joint_angular_k_start = rigid_joint_angular_k_start if rigid_avbd_angular_beta > 0.0 else None
        # Resolve internal alpha (joint/contact) and beta (linear/angular)
        self.rigid_joint_alpha = rigid_avbd_joint_alpha if rigid_avbd_joint_alpha is not None else rigid_avbd_alpha
        self.rigid_linear_beta = rigid_avbd_linear_beta
        self.rigid_angular_beta = rigid_avbd_angular_beta
        self.rigid_contact_hard = int(rigid_contact_hard)
        self.rigid_contact_history = rigid_contact_history
        if rigid_avbd_contact_alpha is not None:
            self.rigid_contact_alpha = rigid_avbd_contact_alpha
        else:
            # Zero alpha works better without contact history: no warm duals to stabilize.
            self.rigid_contact_alpha = rigid_avbd_alpha if (rigid_contact_history and rigid_contact_hard) else 0.0

        # DEADZONE body-snap thresholds; suppressed by _STICK_FLAG_ANCHOR.
        self.rigid_contact_stick_freeze_translation_eps = 1.0e-4
        self.rigid_contact_stick_freeze_angular_eps = 1.0e-4

        # Joint constraint stiffness and damping for non-cable structural joints
        self.rigid_joint_linear_ke = rigid_joint_linear_ke
        self.rigid_joint_angular_ke = rigid_joint_angular_ke
        self.rigid_joint_linear_kd = max(0.0, rigid_joint_linear_kd)
        self.rigid_joint_angular_kd = max(0.0, rigid_joint_angular_kd)

        # -------------------------------------------------------------
        # Rigid-only AVBD state (used when SolverVBD integrates bodies)
        # -------------------------------------------------------------
        if not self.integrate_with_external_rigid_solver and model.body_count > 0:
            # Previous-step body transforms, advanced by update_body_velocity() each step.
            # Provides contact friction velocity and joint C0 feedforward for kinematic tracking.
            # Kinematic bodies: set body_q.
            # Dynamic teleportation: also set body_q_prev and body_qd.
            self.body_q_prev = wp.clone(model.body_q, device=self.device)
            self.body_inertia_q = wp.zeros_like(model.body_q, device=self.device)  # inertial target poses for AVBD

            # Adjacency and dimensions
            self.rigid_adjacency = self._compute_rigid_force_element_adjacency(model).to(self.device)

            # Force accumulation arrays
            self.body_torques = wp.zeros(model.body_count, dtype=wp.vec3, device=self.device)
            self.body_forces = wp.zeros(model.body_count, dtype=wp.vec3, device=self.device)

            # Persistent scratch for joint_f accumulation
            self._body_f_for_integration = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self.device)

            # Hessian blocks (6x6 block structure: angular-angular, angular-linear, linear-linear)
            self.body_hessian_aa = wp.zeros(model.body_count, dtype=wp.mat33, device=self.device)
            self.body_hessian_al = wp.zeros(model.body_count, dtype=wp.mat33, device=self.device)
            self.body_hessian_ll = wp.zeros(model.body_count, dtype=wp.mat33, device=self.device)

            # Per-body contact lists (CSR-like: per-body counts + flat index array).
            # Tight: pre_alloc = 0 when the contact source is absent (no shapes / no particles).
            bb_pre_alloc = rigid_body_contact_buffer_size if model.shape_count > 0 else 0
            self.body_body_contact_buffer_pre_alloc = bb_pre_alloc
            self.body_body_contact_counts = wp.zeros(model.body_count, dtype=wp.int32, device=self.device)
            self.body_body_contact_indices = wp.zeros(
                model.body_count * bb_pre_alloc, dtype=wp.int32, device=self.device
            )
            self.body_body_contact_overflow_max = wp.zeros(1, dtype=wp.int32, device=self.device)

            bp_pre_alloc = (
                rigid_body_particle_contact_buffer_size if model.shape_count > 0 and model.particle_count > 0 else 0
            )
            self.body_particle_contact_buffer_pre_alloc = bp_pre_alloc
            self.body_particle_contact_counts = wp.zeros(model.body_count, dtype=wp.int32, device=self.device)
            self.body_particle_contact_indices = wp.zeros(
                model.body_count * bp_pre_alloc, dtype=wp.int32, device=self.device
            )
            self.body_particle_contact_overflow_max = wp.zeros(1, dtype=wp.int32, device=self.device)

            # Joint constraint layout + penalty stiffness (mutable k, frozen bounds)
            self._init_joint_constraint_layout()
            self.joint_penalty_k, self.joint_penalty_k_min, self.joint_penalty_k_max = self._init_joint_penalty_k()
            self.joint_rest_angle = self._init_joint_rest_angle()

            # Body-body contact state (pre-allocated in __init__ when possible, resized on first step otherwise).
            self.body_body_contact_penalty_k = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_ke = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_kd = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_mu = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_lambda = wp.zeros(0, dtype=wp.vec3, device=self.device)
            self.body_body_contact_C0 = wp.zeros(0, dtype=wp.vec3, device=self.device)
            self.body_body_contact_stick_flag = wp.zeros(0, dtype=wp.int32, device=self.device)

            # Rigid contact warm-start buffers.
            self._prev_contact_lambda = None
            self._prev_contact_stick_flag = None
            self._prev_contact_penalty_k = None
            self._prev_contact_point0 = None
            self._prev_contact_point1 = None
            self._prev_contact_normal = None

            # Joint augmented-Lagrangian state (vec3, per-joint, bilateral)
            self.joint_lambda_lin = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_lambda_ang = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C0_lin = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C0_ang = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Dahl friction state (cable bending plasticity, persistent across timesteps)
            self.joint_sigma_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_kappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_dkappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Pre-computed Dahl parameters (frozen during iterations, updated per timestep)
            self.joint_sigma_start = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C_fric = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Dahl friction: auto-detected from model.vbd custom attributes.
            # Enabled when model has dahl_eps_max and dahl_tau (set via register_custom_attributes).
            vbd_attrs: Any = getattr(model, "vbd", None)
            has_dahl = (
                model.joint_count > 0
                and vbd_attrs is not None
                and hasattr(vbd_attrs, "dahl_eps_max")
                and hasattr(vbd_attrs, "dahl_tau")
            )
            self.enable_dahl_friction = has_dahl
            if has_dahl:
                self.joint_dahl_eps_max = vbd_attrs.dahl_eps_max
                self.joint_dahl_tau = vbd_attrs.dahl_tau
            else:
                self.joint_dahl_eps_max = wp.zeros(model.joint_count, dtype=float, device=self.device)
                self.joint_dahl_tau = wp.zeros(model.joint_count, dtype=float, device=self.device)

        # -------------------------------------------------------------
        # Body-particle interaction shared state.
        # -------------------------------------------------------------
        self.body_particle_contact_penalty_k = wp.zeros(0, dtype=float, device=self.device)
        self.body_particle_contact_material_ke = wp.zeros(0, dtype=float, device=self.device)
        self.body_particle_contact_material_kd = wp.zeros(0, dtype=float, device=self.device)
        self.body_particle_contact_material_mu = wp.zeros(0, dtype=float, device=self.device)
        # Zero-length body poses for static-shape contact kernels when State.body_q is absent.
        self._empty_body_q = wp.empty(0, dtype=wp.transform, device=self.device)
        if model.particle_count > 0 and model.shape_count > 0:
            self._init_body_particle_contact_state(model.shape_count * model.particle_count)

        # Kinematic body support: create effective inv_mass / inv_inertia arrays
        # with kinematic bodies zeroed out.
        self._init_kinematic_state()

        # Pre-allocate body-body contact buffers when the contact capacity is
        # already known; otherwise lazy allocation handles the first step.
        rcm = getattr(model, "rigid_contact_max", 0) or 0
        if rcm > 0 and model.body_count > 0 and not self.integrate_with_external_rigid_solver:
            self._init_body_body_contact_state(rcm)
            if self.rigid_contact_history:
                self._init_rigid_contact_warmstart(rcm)

        # Persistent contact-query outputs; per-contact arrays grow on demand.
        self._rigid_contact_body0 = wp.full(0, -1, dtype=wp.int32, device=self.device)
        self._rigid_contact_body1 = wp.full(0, -1, dtype=wp.int32, device=self.device)
        self._rigid_contact_point0_world = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self._rigid_contact_point1_world = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self._rigid_contact_zero_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._rigid_contact_zero_force = wp.zeros(0, dtype=wp.vec3, device=self.device)

        # Validation
        has_bodies = model.body_count > 0
        has_body_coloring = len(model.body_color_groups) > 0

        if has_bodies and not has_body_coloring and not self.integrate_with_external_rigid_solver:
            raise ValueError(
                "model.body_color_groups is empty but rigid bodies are present! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()

    # =====================================================
    # Initialization Helper Methods
    # =====================================================

    def _init_body_body_contact_state(self, rigid_contact_max: int) -> None:
        """Allocate body-body contact state arrays sized to the given contact buffer capacity."""
        self.body_body_contact_penalty_k = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_ke = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_kd = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_mu = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_lambda = wp.zeros(rigid_contact_max, dtype=wp.vec3, device=self.device)
        self.body_body_contact_C0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, device=self.device)
        self.body_body_contact_stick_flag = wp.zeros(rigid_contact_max, dtype=wp.int32, device=self.device)

    def _init_body_particle_contact_state(self, soft_contact_max: int) -> None:
        """Allocate body-particle material arrays sized to the given soft contact capacity."""
        self.body_particle_contact_penalty_k = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_ke = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_kd = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_mu = wp.zeros(soft_contact_max, dtype=float, device=self.device)

    def _init_rigid_contact_warmstart(self, rigid_contact_max: int) -> None:
        """Allocate rigid contact warm-start buffers."""
        cap = max(1, rigid_contact_max)
        self._prev_contact_lambda = wp.zeros(cap, dtype=wp.vec3, device=self.device)
        self._prev_contact_stick_flag = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._prev_contact_penalty_k = wp.zeros(cap, dtype=float, device=self.device)
        self._prev_contact_point0 = wp.zeros(cap, dtype=wp.vec3, device=self.device)
        self._prev_contact_point1 = wp.zeros(cap, dtype=wp.vec3, device=self.device)
        self._prev_contact_normal = wp.zeros(cap, dtype=wp.vec3, device=self.device)

    def _raise_if_capturing_resize(self, name: str, current: int, required: int) -> None:
        if self.device.is_capturing and not wp.is_mempool_enabled(self.device):
            raise RuntimeError(
                f"SolverVBD {name} buffer needs to grow from {current} to {required} "
                "during CUDA graph capture, but Warp mempool is disabled. "
                "Pre-size before capture by constructing CollisionPipeline before SolverVBD, "
                "passing explicit rigid_contact_max/soft_contact_max to CollisionPipeline, or running one "
                "uncaptured step/force-collection pass."
            )

    @staticmethod
    def _to_numpy(arr, dtype=None):
        """Transfer a Warp array to CPU and return as numpy, optionally casting dtype."""
        cpu = arr.to("cpu")
        result = cpu.numpy() if hasattr(cpu, "numpy") else np.asarray(cpu)
        return result if dtype is None else result.astype(dtype, copy=False)

    def _init_joint_constraint_layout(self) -> None:
        """Initialize VBD-owned joint constraint indexing.

        VBD stores and adapts penalty stiffness values for scalar constraint components:
          - CABLE: 2 scalars (stretch/linear, bend/angular)
          - BALL:  1 scalar (isotropic linear anchor-coincidence)
          - FIXED: 2 scalars (isotropic linear + isotropic angular)
          - REVOLUTE:  3 scalars (isotropic linear + 2-DOF perpendicular angular + angular drive/limit)
          - PRISMATIC: 3 scalars (2-DOF perpendicular linear + isotropic angular + linear drive/limit)
          - D6:   2 + lin_count + ang_count scalars (projected linear + projected angular + per-DOF drive/limit)
          - FREE:  0 scalars (not a constraint)

        Drive and limit for each free DOF share one AVBD slot (mutually exclusive at runtime).

        Any other joint type will raise NotImplementedError.
        """
        n_j = self.model.joint_count
        with wp.ScopedDevice("cpu"):
            jt = self._to_numpy(self.model.joint_type, dtype=int)
            jdof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)

            dim_np = np.zeros((n_j,), dtype=np.int32)
            for j in range(n_j):
                if jt[j] == JointType.CABLE:
                    dim_np[j] = 2
                elif jt[j] == JointType.BALL:
                    dim_np[j] = 1
                elif jt[j] == JointType.FIXED:
                    dim_np[j] = 2
                elif jt[j] == JointType.REVOLUTE:
                    dim_np[j] = 3
                elif jt[j] == JointType.PRISMATIC:
                    dim_np[j] = 3
                elif jt[j] == JointType.D6:
                    dim_np[j] = 2 + int(jdof_dim[j, 0]) + int(jdof_dim[j, 1])
                else:
                    if jt[j] != JointType.FREE:
                        raise NotImplementedError(
                            f"SolverVBD rigid joints: JointType.{JointType(jt[j]).name} is not implemented yet "
                            "(only CABLE, BALL, FIXED, REVOLUTE, PRISMATIC, and D6 are supported)."
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

    def _init_joint_penalty_k(self):
        """Build initial joint penalty state on CPU and upload to solver device.

        Returns:
            (joint_penalty_k, joint_penalty_k_min, joint_penalty_k_max) tuple:
              - joint_penalty_k:     mutable current penalty per constraint scalar
              - joint_penalty_k_min: frozen floor (= k_start when beta > 0, slot ceiling when beta = 0)
              - joint_penalty_k_max: frozen ceiling (= slot-specific ke)

        Side effects (stored on self):
              - joint_penalty_kd:    damping coefficient per constraint scalar
              - joint_is_hard:       hard/soft flag per constraint scalar (1 = hard, 0 = soft)
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
        lin_k_start = self.rigid_joint_linear_k_start
        ang_k_start = self.rigid_joint_angular_k_start

        with wp.ScopedDevice("cpu"):
            joint_k_max_np = np.zeros((constraint_count,), dtype=float)
            joint_k_init_np = np.zeros((constraint_count,), dtype=float)
            joint_kd_np = np.zeros((constraint_count,), dtype=float)
            is_hard_np = np.zeros((constraint_count,), dtype=np.int32)

            jt = self._to_numpy(self.model.joint_type, dtype=int)
            jdofs = self._to_numpy(self.model.joint_qd_start, dtype=int)
            jtarget_ke = self._to_numpy(self.model.joint_target_ke, dtype=float)
            jtarget_kd = self._to_numpy(self.model.joint_target_kd, dtype=float)
            jlimit_ke = self._to_numpy(self.model.joint_limit_ke, dtype=float)
            jdof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)
            jc_start = self._to_numpy(self.joint_constraint_start, dtype=np.int32)

            # Per-joint hard/soft mode from model attribute (default=1, hard).
            vbd_attrs: Any = getattr(self.model, "vbd", None)
            if vbd_attrs is not None and hasattr(vbd_attrs, "joint_is_hard"):
                j_is_hard = self._to_numpy(vbd_attrs.joint_is_hard, dtype=np.int32)
                if not np.all((j_is_hard == 0) | (j_is_hard == 1)):
                    raise ValueError("model.vbd.joint_is_hard values must be 0 (soft) or 1 (hard).")
            else:
                j_is_hard = np.ones(self.model.joint_count, dtype=np.int32)

            structural_linear_ke = self.rigid_joint_linear_ke
            structural_angular_ke = self.rigid_joint_angular_ke

            n_j = self.model.joint_count
            for j in range(n_j):
                if jt[j] == JointType.CABLE:
                    c0 = int(jc_start[j])
                    dof0 = int(jdofs[j])
                    if dof0 < 0 or (dof0 + 1) >= len(jtarget_ke) or (dof0 + 1) >= len(jtarget_kd):
                        raise RuntimeError(
                            "SolverVBD _init_joint_penalty_k: JointType.CABLE requires 2 DOF entries in "
                            "model.joint_target_ke/kd starting at joint_qd_start[j]. "
                            f"Got joint_index={j}, joint_qd_start={dof0}, "
                            f"len(joint_target_ke)={len(jtarget_ke)}, len(joint_target_kd)={len(jtarget_kd)}."
                        )
                    ke_stretch = jtarget_ke[dof0]
                    ke_bend = jtarget_ke[dof0 + 1]
                    joint_k_max_np[c0] = ke_stretch
                    joint_k_max_np[c0 + 1] = ke_bend
                    joint_k_init_np[c0] = ke_stretch if lin_k_start is None else min(lin_k_start, ke_stretch)
                    joint_k_init_np[c0 + 1] = ke_bend if ang_k_start is None else min(ang_k_start, ke_bend)
                    joint_kd_np[c0] = jtarget_kd[dof0]
                    joint_kd_np[c0 + 1] = jtarget_kd[dof0 + 1]
                elif jt[j] == JointType.BALL:
                    c0 = int(jc_start[j])
                    joint_k_max_np[c0] = structural_linear_ke
                    joint_k_init_np[c0] = (
                        structural_linear_ke if lin_k_start is None else min(lin_k_start, structural_linear_ke)
                    )
                    joint_kd_np[c0] = self.rigid_joint_linear_kd
                    is_hard_np[c0] = int(j_is_hard[j])
                elif jt[j] == JointType.FIXED:
                    c0 = int(jc_start[j])
                    joint_k_max_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = (
                        structural_linear_ke if lin_k_start is None else min(lin_k_start, structural_linear_ke)
                    )
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = int(j_is_hard[j])
                    joint_k_max_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = (
                        structural_angular_ke if ang_k_start is None else min(ang_k_start, structural_angular_ke)
                    )
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = int(j_is_hard[j])
                elif jt[j] == JointType.REVOLUTE:
                    c0 = int(jc_start[j])
                    joint_k_max_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = (
                        structural_linear_ke if lin_k_start is None else min(lin_k_start, structural_linear_ke)
                    )
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = int(j_is_hard[j])
                    joint_k_max_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = (
                        structural_angular_ke if ang_k_start is None else min(ang_k_start, structural_angular_ke)
                    )
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = int(j_is_hard[j])
                    dof0 = int(jdofs[j])
                    dl_k_max = max(float(jtarget_ke[dof0]), float(jlimit_ke[dof0]))
                    dl_seed = dl_k_max if ang_k_start is None else min(ang_k_start, dl_k_max)
                    joint_k_max_np[c0 + 2] = dl_k_max
                    joint_k_init_np[c0 + 2] = dl_seed
                    joint_kd_np[c0 + 2] = 0.0
                elif jt[j] == JointType.PRISMATIC:
                    c0 = int(jc_start[j])
                    joint_k_max_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = (
                        structural_linear_ke if lin_k_start is None else min(lin_k_start, structural_linear_ke)
                    )
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = int(j_is_hard[j])
                    joint_k_max_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = (
                        structural_angular_ke if ang_k_start is None else min(ang_k_start, structural_angular_ke)
                    )
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = int(j_is_hard[j])
                    dof0 = int(jdofs[j])
                    dl_k_max = max(float(jtarget_ke[dof0]), float(jlimit_ke[dof0]))
                    dl_seed = dl_k_max if lin_k_start is None else min(lin_k_start, dl_k_max)
                    joint_k_max_np[c0 + 2] = dl_k_max
                    joint_k_init_np[c0 + 2] = dl_seed
                    joint_kd_np[c0 + 2] = 0.0
                elif jt[j] == JointType.D6:
                    c0 = int(jc_start[j])
                    dof0 = int(jdofs[j])
                    lc = int(jdof_dim[j, 0])
                    ac = int(jdof_dim[j, 1])
                    joint_k_max_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = (
                        structural_linear_ke if lin_k_start is None else min(lin_k_start, structural_linear_ke)
                    )
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = int(j_is_hard[j])
                    joint_k_max_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = (
                        structural_angular_ke if ang_k_start is None else min(ang_k_start, structural_angular_ke)
                    )
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = int(j_is_hard[j])
                    for li in range(lc):
                        dof_idx = dof0 + li
                        slot = c0 + 2 + li
                        dl_k_max = max(float(jtarget_ke[dof_idx]), float(jlimit_ke[dof_idx]))
                        dl_seed = dl_k_max if lin_k_start is None else min(lin_k_start, dl_k_max)
                        joint_k_max_np[slot] = dl_k_max
                        joint_k_init_np[slot] = dl_seed
                        joint_kd_np[slot] = 0.0
                    for ai in range(ac):
                        dof_idx = dof0 + lc + ai
                        slot = c0 + 2 + lc + ai
                        dl_k_max = max(float(jtarget_ke[dof_idx]), float(jlimit_ke[dof_idx]))
                        dl_seed = dl_k_max if ang_k_start is None else min(ang_k_start, dl_k_max)
                        joint_k_max_np[slot] = dl_k_max
                        joint_k_init_np[slot] = dl_seed
                        joint_kd_np[slot] = 0.0
                else:
                    pass

            self.joint_penalty_kd = wp.array(joint_kd_np, dtype=float, device=self.device)
            self.joint_is_hard = wp.array(is_hard_np, dtype=wp.int32, device=self.device)
            k = wp.array(joint_k_init_np, dtype=float, device=self.device)
            k_min = wp.array(joint_k_init_np.copy(), dtype=float, device=self.device)
            k_max = wp.array(joint_k_max_np, dtype=float, device=self.device)
            return k, k_min, k_max

    def _init_joint_rest_angle(self):
        """Compute per-DOF rest-pose joint angles from ``model.joint_q``.

        VBD computes angular joint angles via ``kappa`` (rotation vector relative to
        the rest pose stored in ``model.body_q``). After ``eval_fk(model, ..., model)``,
        the rest pose encodes the initial joint configuration, so ``kappa = 0`` at the
        initial angles. Drive targets and limits, however, are specified in absolute
        joint coordinates. This array stores the rest-pose angle offset per DOF so that
        ``theta_abs = theta + joint_rest_angle[dof_idx]`` converts rest-relative
        ``theta`` back to absolute coordinates for drive/limit comparison.

        Only angular DOFs of REVOLUTE and D6 joints need nonzero entries. Linear DOFs
        (PRISMATIC, D6 linear) use absolute geometric measurements (``d_along``) and
        are unaffected - their entries are left at 0.
        """
        dof_count = self.model.joint_dof_count
        rest_angle_np = np.zeros(dof_count, dtype=float)

        with wp.ScopedDevice("cpu"):
            jt = self._to_numpy(self.model.joint_type, dtype=int)
            jq = self._to_numpy(self.model.joint_q, dtype=float)
            jq_start = self._to_numpy(self.model.joint_q_start, dtype=int)
            jqd_start = self._to_numpy(self.model.joint_qd_start, dtype=int)
            jdof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)

            for j in range(self.model.joint_count):
                if jt[j] == JointType.REVOLUTE:
                    q_start = int(jq_start[j])
                    qd_start = int(jqd_start[j])
                    rest_angle_np[qd_start] = float(jq[q_start])
                elif jt[j] == JointType.D6:
                    q_start = int(jq_start[j])
                    qd_start = int(jqd_start[j])
                    lin_count = int(jdof_dim[j, 0])
                    ang_count = int(jdof_dim[j, 1])
                    for ai in range(ang_count):
                        rest_angle_np[qd_start + lin_count + ai] = float(jq[q_start + lin_count + ai])

        return wp.array(rest_angle_np, dtype=float, device=self.device)

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register solver-specific custom Model attributes for SolverVBD.

        Currently used for:
          - Cable bending plasticity/hysteresis (Dahl friction model)
          - Per-joint structural constraint mode (hard/soft)

        Attributes are declared in the ``vbd`` namespace so they can be authored in scenes
        and in USD as ``newton:vbd:<attr>``.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_eps_max",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.5,
                namespace="vbd",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_tau",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="vbd",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_is_hard",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=1,
                namespace="vbd",
            )
        )

    # =====================================================
    # Adjacency Building Methods
    # =====================================================

    def _compute_particle_force_element_adjacency(self):
        particle_adjacency = ParticleForceElementAdjacencyInfo()

        with wp.ScopedDevice("cpu"):
            if self.model.edge_indices:
                edges_array = self.model.edge_indices.to("cpu")
                # build vertex-edge particle_adjacency data
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=_count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                    device="cpu",
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
                    kernel=_fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        particle_adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        particle_adjacency.v_adj_edges,
                    ],
                    dim=1,
                    device="cpu",
                )
            else:
                particle_adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.tri_indices:
                face_indices = self.model.tri_indices.to("cpu")
                # compute adjacent triangles
                # count number of adjacent faces for each vertex
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device="cpu")
                wp.launch(
                    kernel=_count_num_adjacent_faces,
                    inputs=[face_indices, num_vertex_adjacent_faces],
                    dim=1,
                    device="cpu",
                )

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
                    kernel=_fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        particle_adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        particle_adjacency.v_adj_faces,
                    ],
                    dim=1,
                    device="cpu",
                )
            else:
                particle_adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.tet_indices:
                tet_indices = self.model.tet_indices.to("cpu")
                num_vertex_adjacent_tets = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=_count_num_adjacent_tets,
                    inputs=[tet_indices, num_vertex_adjacent_tets],
                    dim=1,
                    device="cpu",
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
                    kernel=_fill_adjacent_tets,
                    inputs=[
                        tet_indices,
                        particle_adjacency.v_adj_tets_offsets,
                        vertex_adjacent_tets_fill_count,
                        particle_adjacency.v_adj_tets,
                    ],
                    dim=1,
                    device="cpu",
                )
            else:
                particle_adjacency.v_adj_tets_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_tets = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.spring_indices:
                spring_array = self.model.spring_indices.to("cpu")
                # build vertex-springs particle_adjacency data
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=_count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                    device="cpu",
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
                    kernel=_fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        particle_adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        particle_adjacency.v_adj_springs,
                    ],
                    dim=1,
                    device="cpu",
                )

            else:
                particle_adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

        return particle_adjacency

    def _compute_particle_contact_filtering_list(
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
                ) = set_to_csr(v_tri_filter_sets)
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
                ) = set_to_csr(edge_edge_filter_sets)
                self.particle_edge_edge_contact_filtering_list = wp.array(
                    self.particle_edge_edge_contact_filtering_list, dtype=int, device=self.device
                )
                self.particle_edge_edge_contact_filtering_list_offsets = wp.array(
                    self.particle_edge_edge_contact_filtering_list_offsets, dtype=int, device=self.device
                )

    def _compute_rigid_force_element_adjacency(self, model):
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
                    kernel=_count_num_adjacent_joints,
                    inputs=[joint_parent_cpu, joint_child_cpu, num_body_adjacent_joints],
                    dim=1,
                    device="cpu",
                )

                num_body_adjacent_joints = num_body_adjacent_joints.numpy()
                body_adjacent_joints_offsets = np.empty(shape=(model.body_count + 1,), dtype=wp.int32)
                body_adjacent_joints_offsets[1:] = np.cumsum(num_body_adjacent_joints)[:]
                body_adjacent_joints_offsets[0] = 0
                adjacency.body_adj_joints_offsets = wp.array(body_adjacent_joints_offsets, dtype=wp.int32)

                body_adjacent_joints_fill_count = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(num_body_adjacent_joints.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=_fill_adjacent_joints,
                    inputs=[
                        joint_parent_cpu,
                        joint_child_cpu,
                        adjacency.body_adj_joints_offsets,
                        body_adjacent_joints_fill_count,
                        adjacency.body_adj_joints,
                    ],
                    dim=1,
                    device="cpu",
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
        """Set whether the next step() should update rigid solver history.

        When True (default), the step refreshes rigid contact state from the
        provided ``Contacts`` buffer: rebuilds per-body contact lists, initializes
        penalty_k/lambda/C0, and restores warm-start state from
        ``Contacts.rigid_contact_match_index`` when contact history is enabled.
        When False, the step reuses the current rigid contact lists and contact
        state. In that mode, the caller must pass the same contact result/buffers
        used by the previous refresh; do not run collision into the contacts
        buffer between refreshes. Passing newly collided contacts while update is
        disabled can mismatch stale per-body contact lists with current contact
        rows.

        Joint AVBD maintenance (C0 snapshot, lambda decay)
        runs every step regardless of this flag via step_joint_C0_lambda().
        Rigid contact history snapshotting also runs every step when enabled.

        This setting applies only to the next call to :meth:`step` and is then
        reset to True.  Useful for substepping where collision detection frequency
        differs from the simulation step frequency.

        Args:
            update: If True, update rigid solver state. If False, reuse previous.
        """
        self._update_rigid_history = update

    def set_joint_constraint_mode(self, joint_index: int, hard: bool, slot: int | None = None):
        """Set hard or soft constraint mode for a joint's structural slots at runtime.

        Hard mode (augmented Lagrangian): uses persistent lambda + C0 stabilization
        to drive constraint violation toward zero across iterations.
        Soft mode (penalty-only): uses penalty stiffness only (no lambda or C0 state).

        Structural slots are LINEAR (slot 0) and ANGULAR (slot 1). Drive/limit slots
        (slot 2+) are always soft and cannot be set to hard.

        By default, cable stretch and bend slots are soft, while non-cable
        structural slots are hard.

        For bulk initialization of non-cable joints at build time (avoids
        per-joint roundtrips), use the ``joint_is_hard`` model custom attribute::

            SolverVBD.register_custom_attributes(builder)  # before adding joints
            ...
            model = builder.finalize()
            model.vbd.joint_is_hard.numpy()[j] = 0  # set joint j to soft
            solver = SolverVBD(model, ...)

        Args:
            joint_index: Index of the joint to modify.
            hard: True for hard mode (AL), False for soft mode (penalty-only).
            slot: Specific slot index to set. If None, sets all structural slots.
                  Use JointSlot.LINEAR / JointSlot.ANGULAR (equivalently
                  JointSlot.STRETCH / JointSlot.BEND for cables).

        Raises:
            ValueError: If the joint index is out of range, or the slot is a
                drive/limit slot (>= 2), or the slot exceeds the joint's
                constraint dimension.
        """
        n_j = self.model.joint_count
        if joint_index < 0 or joint_index >= n_j:
            raise ValueError(f"joint_index={joint_index} out of range [0, {n_j}).")

        with wp.ScopedDevice("cpu"):
            c_start_np = self._to_numpy(self.joint_constraint_start, dtype=np.int32)
            c_dim_np = self._to_numpy(self.joint_constraint_dim, dtype=np.int32)
            is_hard_np = self._to_numpy(self.joint_is_hard, dtype=np.int32)

            c0 = int(c_start_np[joint_index])
            cdim = int(c_dim_np[joint_index])
            val = 1 if hard else 0

            if slot is not None:
                if slot < 0 or slot >= 2:
                    raise ValueError(
                        f"Cannot set hard mode on slot={slot}. "
                        "Only structural slots (LINEAR=0, ANGULAR=1) support hard mode."
                    )
                if slot >= cdim:
                    raise ValueError(
                        f"slot={slot} exceeds joint constraint dimension ({cdim}) for joint_index={joint_index}."
                    )
                is_hard_np[c0 + slot] = val
            else:
                structural_count = min(cdim, 2)
                for s in range(structural_count):
                    is_hard_np[c0 + s] = val

            self.joint_is_hard = wp.array(is_hard_np, dtype=wp.int32, device=self.device)

            if not hard:
                lam_lin_np = self._to_numpy(self.joint_lambda_lin)
                lam_ang_np = self._to_numpy(self.joint_lambda_ang)
                C0_lin_np = self._to_numpy(self.joint_C0_lin)
                C0_ang_np = self._to_numpy(self.joint_C0_ang)
                if slot is None or slot == 0:
                    lam_lin_np[joint_index] = [0.0, 0.0, 0.0]
                    C0_lin_np[joint_index] = [0.0, 0.0, 0.0]
                if (slot is None or slot == 1) and cdim > 1:
                    lam_ang_np[joint_index] = [0.0, 0.0, 0.0]
                    C0_ang_np[joint_index] = [0.0, 0.0, 0.0]
                self.joint_lambda_lin = wp.array(lam_lin_np, dtype=wp.vec3, device=self.device)
                self.joint_lambda_ang = wp.array(lam_ang_np, dtype=wp.vec3, device=self.device)
                self.joint_C0_lin = wp.array(C0_lin_np, dtype=wp.vec3, device=self.device)
                self.joint_C0_ang = wp.array(C0_ang_np, dtype=wp.vec3, device=self.device)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Execute one simulation timestep using VBD (particles) and AVBD (rigid bodies).

        The solver follows a 3-phase structure:
        1. Initialize: Forward integrate particles and rigid bodies, detect collisions, initialize contact state
        2. Iterate: Interleave particle VBD iterations and rigid body AVBD iterations
        3. Finalize: Update velocities and persistent state (Dahl friction)

        To control rigid body substepping behavior, call set_rigid_history_update().
        When True (default), the step rebuilds rigid contact lists, re-initializes
        rigid contact state (penalty_k, lambda, C0), and restores from history if enabled.
        When False, reuses previous rigid contact state. The flag is reset to True when consumed.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Control inputs.
            contacts: Contact data produced by :meth:`~newton.Model.collide` (rigid-rigid and rigid-particle contacts).
                If None, rigid contact handling is skipped. Note that particle self-contact (if enabled) does not
                depend on this argument.
            dt: Time step size.
        """
        update_rigid = self._update_rigid_history
        self._update_rigid_history = True

        if control is None:
            control = self.model.control(clone_variables=False)

        self._initialize_rigid_bodies(state_in, control, contacts, dt, update_rigid)
        self._initialize_particles(state_in, state_out, dt)

        for iter_num in range(self.iterations):
            self._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)
            self._solve_particle_iteration(state_in, state_out, contacts, dt, iter_num)

        # Snapshot solved rigid contact state for next-frame warm-start.
        self._snapshot_rigid_contact_history(contacts)
        self._finalize_rigid_bodies(
            state_in, state_out, dt, apply_stick_deadzone=contacts is not None and self.rigid_contact_hard
        )
        self._finalize_particles(state_out, dt)

    def _snapshot_rigid_contact_history(self, contacts: Contacts | None):
        """Write solved contact state for next frame's match-index warm-start."""
        if not self.rigid_contact_history or contacts is None:
            return

        if self.model.body_count == 0 or self.integrate_with_external_rigid_solver:
            return

        contact_launch_dim = contacts.rigid_contact_max
        if contact_launch_dim == 0:
            return

        if self._prev_contact_lambda is None or self._prev_contact_lambda.shape[0] < contact_launch_dim:
            history_cap = 0 if self._prev_contact_lambda is None else self._prev_contact_lambda.shape[0]
            self._raise_if_capturing_resize("rigid contact history", history_cap, contact_launch_dim)
            self._init_rigid_contact_warmstart(contact_launch_dim)

        # Snapshot solved contact rows for the next step's warm-start.
        wp.launch(
            kernel=snapshot_body_body_contact_history,
            dim=contact_launch_dim,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                self.body_body_contact_lambda,
                self.body_body_contact_stick_flag,
                self.body_body_contact_penalty_k,
            ],
            outputs=[
                self._prev_contact_lambda,
                self._prev_contact_stick_flag,
                self._prev_contact_penalty_k,
                self._prev_contact_point0,
                self._prev_contact_point1,
                self._prev_contact_normal,
            ],
            device=self.device,
        )

    def _penetration_free_truncation(self, particle_q_out=None):
        """
        Modify displacements_in in-place, also modify particle_q if its not None

        """
        if not self.particle_enable_self_contact:
            self.truncation_ts.fill_(1.0)
            wp.launch(
                kernel=apply_truncation_ts,
                dim=self.model.particle_count,
                inputs=[
                    self.pos_prev_collision_detection,  # pos: wp.array[wp.vec3],
                    self.particle_displacements,  # displacement_in: wp.array[wp.vec3],
                    self.truncation_ts,  # truncation_ts: wp.array[float],
                    wp.inf,  # max_displacement: float (input threshold)
                ],
                outputs=[
                    self.particle_displacements,  # displacement_out: wp.array[wp.vec3],
                    particle_q_out,  # pos_out: wp.array[wp.vec3],
                ],
                device=self.device,
            )

        else:
            ##  parallel by collision and atomic operation
            self.truncation_ts.fill_(1.0)
            wp.launch(
                kernel=apply_planar_truncation_parallel_by_collision,
                inputs=[
                    self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array[wp.vec3],
                    self.particle_displacements,  # particle_displacements: wp.array[wp.vec3],
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.trimesh_collision_info,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    self.particle_conservative_bound_relaxation,
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
                    self.pos_prev_collision_detection,
                    self.particle_displacements,
                    self.truncation_ts,
                    self.particle_self_contact_margin
                    * self.particle_conservative_bound_relaxation
                    * 0.5,  # max_displacement: degenerate to isotropic truncation
                ],
                outputs=[
                    self.particle_displacements,
                    particle_q_out,
                ],
                device=self.device,
            )

    def _initialize_particles(self, state_in: State, state_out: State, dt: float):
        """Initialize particle positions for the VBD iteration."""
        model = self.model

        # Early exit if no particles
        if model.particle_count == 0:
            return

        # Collision detection before initialization to compute conservative bounds
        if self.particle_enable_self_contact:
            self._collision_detection_penetration_free(state_in)
        else:
            self.pos_prev_collision_detection.assign(state_in.particle_q)
            self.particle_displacements.zero_()

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

        self._penetration_free_truncation(state_in.particle_q)

    def _initialize_rigid_bodies(
        self,
        state_in: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
        refresh: bool,
    ) -> None:
        """Initialize rigid body states for AVBD solver (pre-iteration phase).

        Performs forward integration and initializes contact-related AVBD state when contacts are provided.

        If contacts is None, rigid contact-related work is skipped:
        no per-body contact lists are built, and no contact state is initialized or restored.

        If control provides joint_f, per-DOF joint forces are mapped to body spatial
        wrenches and included in the forward integration (shifting the inertial target).

        The ``refresh`` input controls whether rigid contact lists and contact
        state are rebuilt. It may be promoted locally when contact state needs
        first-time allocation or resizing.
        """
        model = self.model

        # ---------------------------
        # Rigid-only initialization
        # ---------------------------
        if model.body_count > 0 and not self.integrate_with_external_rigid_solver:
            # Force refresh when contact state is not yet allocated or undersized.
            if (
                not refresh
                and contacts is not None
                and contacts.rigid_contact_max > 0
                and self.body_body_contact_penalty_k.shape[0] < contacts.rigid_contact_max
            ):
                refresh = True

            # Contact C0 + history restore BEFORE integration: body_q is the collide frame
            # for all bodies (dynamic and kinematic) at this point.
            if refresh:
                if contacts is None:
                    self.body_body_contact_counts.zero_()
                else:
                    contact_launch_dim = contacts.rigid_contact_max

                    if self.body_body_contact_penalty_k.shape[0] < contact_launch_dim:
                        self._raise_if_capturing_resize(
                            "body-body contact state",
                            self.body_body_contact_penalty_k.shape[0],
                            contact_launch_dim,
                        )
                        self._init_body_body_contact_state(contact_launch_dim)

                    # Build body-body contact lists
                    self.body_body_contact_counts.zero_()
                    self.body_body_contact_overflow_max.zero_()
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
                            self.body_body_contact_overflow_max,
                        ],
                        device=self.device,
                    )
                    wp.launch(
                        kernel=check_contact_overflow,
                        dim=1,
                        inputs=[self.body_body_contact_overflow_max, self.body_body_contact_buffer_pre_alloc, 0],
                        device=self.device,
                    )

                    # Restore AVBD body-body contact state from history and pre-compute material properties
                    if self.rigid_contact_history:
                        if contacts.rigid_contact_match_index is None:
                            raise RuntimeError(
                                "SolverVBD(rigid_contact_history=True) requires Contacts with "
                                "rigid_contact_match_index populated. Create contacts through "
                                'CollisionPipeline(contact_matching="latest") for VBD warm-starting, '
                                "or set rigid_contact_history=False."
                            )

                        history_required = max(1, contact_launch_dim)
                        if self._prev_contact_lambda is None or self._prev_contact_lambda.shape[0] < history_required:
                            history_cap = 0 if self._prev_contact_lambda is None else self._prev_contact_lambda.shape[0]
                            self._raise_if_capturing_resize("rigid contact history", history_cap, history_required)
                            self._init_rigid_contact_warmstart(history_required)

                        history = RigidContactHistory()
                        history.lambda_ = self._prev_contact_lambda
                        history.stick_flag = self._prev_contact_stick_flag
                        history.penalty_k = self._prev_contact_penalty_k
                        history.point0 = self._prev_contact_point0
                        history.point1 = self._prev_contact_point1
                        history.normal = self._prev_contact_normal

                        wp.launch(
                            kernel=init_body_body_contacts_avbd,
                            dim=contact_launch_dim,
                            inputs=[
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                contacts.rigid_contact_normal,
                                model.shape_material_ke,
                                model.shape_material_kd,
                                model.shape_material_mu,
                                self.rigid_contact_hard,
                                contacts.rigid_contact_match_index,
                                history,
                                self.rigid_contact_k_start_value,
                            ],
                            outputs=[
                                contacts.rigid_contact_point0,
                                contacts.rigid_contact_point1,
                                self.body_body_contact_penalty_k,
                                self.body_body_contact_lambda,
                                self.body_body_contact_material_kd,
                                self.body_body_contact_material_mu,
                                self.body_body_contact_material_ke,
                            ],
                            device=self.device,
                        )
                    else:
                        wp.launch(
                            kernel=init_body_body_contact_materials,
                            inputs=[
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                model.shape_material_ke,
                                model.shape_material_kd,
                                model.shape_material_mu,
                                self.rigid_contact_k_start_value,
                            ],
                            outputs=[
                                self.body_body_contact_penalty_k,
                                self.body_body_contact_material_kd,
                                self.body_body_contact_material_mu,
                                self.body_body_contact_material_ke,
                            ],
                            dim=contact_launch_dim,
                            device=self.device,
                        )
                        self.body_body_contact_lambda.zero_()

            # Per-step k decay + lambda decay + C0 (body_q is still collide frame here).
            if contacts is not None and contacts.rigid_contact_max > 0:
                contact_launch_dim = contacts.rigid_contact_max
                gamma_lambda = (
                    self.rigid_avbd_gamma if (self.rigid_contact_history and self.rigid_contact_hard) else 0.0
                )
                wp.launch(
                    kernel=step_body_body_contact_C0_lambda,
                    dim=contact_launch_dim,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        model.shape_body,
                        state_in.body_q,
                        self.rigid_contact_hard,
                        gamma_lambda,
                        self.rigid_avbd_gamma,
                        self.body_body_contact_material_ke,
                        self.rigid_contact_k_start_value,
                    ],
                    outputs=[
                        self.body_body_contact_penalty_k,
                        self.body_body_contact_C0,
                        self.body_body_contact_lambda,
                    ],
                    device=self.device,
                )
                self.body_body_contact_stick_flag.zero_()

            # Accumulate joint_f into body wrenches (scratch buffer avoids mutating user state).
            body_f_for_integration = state_in.body_f
            if model.joint_count > 0 and control is not None and control.joint_f is not None:
                wp.copy(self._body_f_for_integration, state_in.body_f)
                body_f_for_integration = self._body_f_for_integration
                wp.launch(
                    kernel=apply_joint_forces,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
                        model.body_com,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_axis,
                        control.joint_f,
                    ],
                    outputs=[
                        body_f_for_integration,
                    ],
                    device=self.device,
                )

            # Forward integrate rigid bodies (body_q modified in-place for dynamic bodies only).
            wp.launch(
                kernel=forward_step_rigid_bodies,
                inputs=[
                    dt,
                    model.gravity,
                    model.body_world,
                    body_f_for_integration,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    state_in.body_q,  # input/output
                    state_in.body_qd,  # input/output
                ],
                outputs=[
                    self.body_inertia_q,
                ],
                dim=model.body_count,
                device=self.device,
            )

            if model.joint_count > 0:
                wp.launch(
                    kernel=step_joint_C0_lambda,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.body_q_prev,
                        model.body_q,
                        self.joint_constraint_start,
                        self.joint_constraint_dim,
                        self.joint_is_hard,
                        self.rigid_avbd_gamma,
                        self.joint_penalty_k_min,
                        self.joint_penalty_k_max,
                    ],
                    outputs=[
                        self.joint_penalty_k,
                        self.joint_C0_lin,
                        self.joint_C0_ang,
                        self.joint_lambda_lin,
                        self.joint_lambda_ang,
                    ],
                    device=self.device,
                )

            # Compute Dahl hysteresis parameters for cable bending (once per timestep, frozen during iterations)
            if self.enable_dahl_friction and model.joint_count > 0:
                wp.launch(
                    kernel=compute_cable_dahl_parameters,
                    inputs=[
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.joint_constraint_start,
                        self.joint_penalty_k_max,
                        self.body_q_prev,
                        model.body_q,
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
        particle_refresh = refresh
        if (
            not particle_refresh
            and model.particle_count > 0
            and contacts is not None
            and contacts.soft_contact_max > 0
            and self.body_particle_contact_penalty_k.shape[0] < contacts.soft_contact_max
        ):
            particle_refresh = True

        if model.particle_count > 0 and particle_refresh and contacts is not None:
            # Build body-particle contact lists (only when SolverVBD integrates bodies).
            if not self.integrate_with_external_rigid_solver and model.body_count > 0:
                self.body_particle_contact_counts.zero_()
                self.body_particle_contact_overflow_max.zero_()
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
                        self.body_particle_contact_overflow_max,
                    ],
                    device=self.device,
                )
                wp.launch(
                    kernel=check_contact_overflow,
                    dim=1,
                    inputs=[self.body_particle_contact_overflow_max, self.body_particle_contact_buffer_pre_alloc, 1],
                    device=self.device,
                )

            # Init body-particle material properties (needed for both internal and external rigid solver).
            soft_contact_launch_dim = contacts.soft_contact_max
            if self.body_particle_contact_penalty_k.shape[0] < soft_contact_launch_dim:
                self._raise_if_capturing_resize(
                    "body-particle contact state",
                    self.body_particle_contact_penalty_k.shape[0],
                    soft_contact_launch_dim,
                )
                self._init_body_particle_contact_state(soft_contact_launch_dim)
            wp.launch(
                kernel=init_body_particle_contacts,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    model.soft_contact_ke,
                    model.soft_contact_kd,
                    model.soft_contact_mu,
                    model.shape_material_ke,
                    model.shape_material_kd,
                    model.shape_material_mu,
                    self.rigid_contact_k_start_value,
                ],
                outputs=[
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                    self.body_particle_contact_material_ke,
                ],
                dim=soft_contact_launch_dim,
                device=self.device,
            )

    def _solve_particle_iteration(
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
                self._collision_detection_penetration_free(state_in)

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
                        state_in.particle_q,
                        model.particle_colors,
                        # body-particle contact
                        self.friction_epsilon,
                        model.particle_radius,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_count,
                        contacts.soft_contact_max,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        model.shape_material_mu,
                        model.shape_body,
                        body_q_for_particles,
                        body_q_prev_for_particles,
                        body_qd_for_particles,
                        model.body_com,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                    ],
                    outputs=[
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    device=self.device,
                )

            if model.spring_count:
                wp.launch(
                    kernel=accumulate_spring_force_and_hessian,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_in.particle_q,
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
                        state_in.particle_q,
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
                        state_in.particle_q,
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
                        state_in.particle_q,
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
            self._penetration_free_truncation(state_in.particle_q)

        wp.copy(state_out.particle_q, state_in.particle_q)

    def _solve_rigid_body_iteration(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, dt: float
    ):
        """Solve one AVBD iteration for rigid bodies (per-iteration phase).

        Accumulates contact and joint forces/hessians, solves 6x6 rigid body systems per color,
        and updates AVBD penalty parameters (dual update).
        """
        model = self.model

        # Body-particle soft contacts still need penalty updates when VBD skips rigid solves:
        # external rigid mode uses state_out.body_q, while static-shape contacts use _empty_body_q.
        skip_rigid_solve = self.integrate_with_external_rigid_solver or model.body_count == 0
        if skip_rigid_solve:
            if model.particle_count > 0 and contacts is not None:
                body_q = state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q
                if body_q is None:
                    body_q = self._empty_body_q

                wp.launch(
                    kernel=update_duals_body_particle_contacts,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        body_q,
                        self.body_particle_contact_material_ke,
                        self.rigid_linear_beta,
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

            # Accumulate body-particle contact forces/hessians for bodies in this color
            if model.particle_count > 0 and contacts is not None:
                wp.launch(
                    kernel=accumulate_body_particle_contacts_per_body,
                    dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        state_in.particle_q,
                        self.particle_q_prev,
                        model.particle_radius,
                        self.body_q_prev,
                        state_in.body_q,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.friction_epsilon,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
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
                        self.body_inv_mass_effective,
                        self.friction_epsilon,
                        self.body_body_contact_penalty_k,
                        self.body_body_contact_material_kd,
                        self.body_body_contact_material_mu,
                        self.body_body_contact_lambda,
                        self.body_body_contact_C0,
                        self.rigid_contact_alpha,
                        self.rigid_contact_hard,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
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
                    self.body_inv_mass_effective,
                    model.body_inertia,
                    self.body_inertia_q,
                    model.body_com,
                    self.rigid_adjacency,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_qd_start,
                    self.joint_constraint_start,
                    self.joint_penalty_k,
                    self.joint_penalty_kd,
                    self.joint_sigma_start,
                    self.joint_C_fric,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_pos,
                    control.joint_target_vel,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    self.joint_lambda_lin,
                    self.joint_lambda_ang,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    model.joint_dof_dim,
                    self.joint_rest_angle,
                    self.body_forces,
                    self.body_torques,
                    self.body_hessian_ll,
                    self.body_hessian_al,
                    self.body_hessian_aa,
                ],
                outputs=[
                    state_in.body_q,
                ],
                dim=color_group.size,
                device=self.device,
            )

        if contacts is not None:
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
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    model.shape_body,
                    state_in.body_q,
                    self.body_q_prev,
                    self.body_body_contact_material_mu,
                    self.body_body_contact_C0,
                    self.rigid_contact_alpha,
                    self.rigid_contact_hard,
                    self.body_inv_mass_effective,
                    self.body_body_contact_material_ke,
                    self.rigid_linear_beta,
                    self.body_body_contact_penalty_k,  # input/output
                    self.body_body_contact_lambda,  # input/output
                ],
                outputs=[
                    self.body_body_contact_stick_flag,
                ],
                device=self.device,
            )

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
                        state_in.body_q,
                        self.body_particle_contact_material_ke,
                        self.rigid_linear_beta,
                        self.body_particle_contact_penalty_k,  # input/output
                    ],
                    device=self.device,
                )

        if model.joint_count > 0:
            wp.launch(
                kernel=update_duals_joint,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_qd_start,
                    self.joint_constraint_start,
                    state_in.body_q,
                    model.body_q,
                    model.joint_dof_dim,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    self.joint_penalty_k_max,
                    self.rigid_linear_beta,
                    self.rigid_angular_beta,
                    model.joint_target_ke,
                    control.joint_target_pos,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    self.joint_rest_angle,
                    self.joint_penalty_k,  # input/output
                    self.joint_lambda_lin,  # input/output
                    self.joint_lambda_ang,  # input/output
                ],
                device=self.device,
            )

    def collect_rigid_contact_forces(
        self, body_q: wp.array, body_q_prev: wp.array, contacts: Contacts | None, dt: float
    ) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array, wp.array]:
        """Collect per-contact rigid contact forces and world-space application points.

        Args:
            body_q (wp.array[wp.transform]): Current body transforms (world frame),
                typically ``state_out.body_q`` after a ``step()`` call.
            body_q_prev (wp.array[wp.transform]): Previous body transforms (world frame)
                corresponding to the start of the step. The caller must snapshot
                ``solver.body_q_prev`` **before** calling ``step()``, because
                ``step()`` advances the solver's internal ``body_q_prev`` to the
                end-of-step pose.
            contacts (Optional[Contacts]): Contact data buffers containing rigid
                contact geometry/material references. If None, the function
                returns default zero/sentinel outputs.
            dt (float): Time step size [s].

        Note:
            Call after collision generation and ``step()`` with the same
            ``Contacts`` buffer. If rigid contact state is absent or undersized,
            this returns sentinel/zero outputs without growing output buffers.
            Output buffers persist and grow on demand; they do not shrink, so
            iterate up to the returned ``rigid_contact_count`` rather than the
            array length.

        Returns:
            tuple[
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.vec3],
                wp.array[wp.vec3],
                wp.array[wp.vec3],
                wp.array[wp.int32],
            ]: Tuple of per-contact outputs:
                - body0: Body index for shape0, int32.
                - body1: Body index for shape1, int32.
                - point0_world: World-space contact point on body0, wp.vec3 [m].
                - point1_world: World-space contact point on body1, wp.vec3 [m].
                - force_on_body1: Contact force applied to body1 in world frame, wp.vec3 [N].
                - rigid_contact_count: Length-1 active rigid-contact count, int32.
        """
        max_contacts = contacts.rigid_contact_max if contacts is not None else 0

        missing_rigid_state = any(
            arr is None or arr.shape[0] < max_contacts
            for arr in (
                getattr(self, "body_body_contact_penalty_k", None),
                getattr(self, "body_body_contact_material_kd", None),
                getattr(self, "body_body_contact_material_mu", None),
                getattr(self, "body_body_contact_lambda", None),
                getattr(self, "body_body_contact_C0", None),
            )
        )
        no_contact_capacity = contacts is None or max_contacts == 0

        if no_contact_capacity or missing_rigid_state:
            if contacts is not None and contacts.rigid_contact_force is not None:
                contacts.rigid_contact_force.zero_()
            if self._rigid_contact_body0.shape[0] > 0:
                self._rigid_contact_body0.fill_(-1)
                self._rigid_contact_body1.fill_(-1)
                self._rigid_contact_point0_world.zero_()
                self._rigid_contact_point1_world.zero_()
            return (
                self._rigid_contact_body0,
                self._rigid_contact_body1,
                self._rigid_contact_point0_world,
                self._rigid_contact_point1_world,
                contacts.rigid_contact_force if contacts is not None else self._rigid_contact_zero_force,
                self._rigid_contact_zero_count,
            )

        # Type narrowing: remaining path requires a valid Contacts instance.
        assert contacts is not None

        output_capacity = self._rigid_contact_body0.shape[0]
        if output_capacity < max_contacts:
            self._raise_if_capturing_resize(
                "rigid contact output",
                output_capacity,
                max_contacts,
            )
            self._rigid_contact_body0 = wp.full(max_contacts, -1, dtype=wp.int32, device=self.device)
            self._rigid_contact_body1 = wp.full(max_contacts, -1, dtype=wp.int32, device=self.device)
            self._rigid_contact_point0_world = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)
            self._rigid_contact_point1_world = wp.zeros(max_contacts, dtype=wp.vec3, device=self.device)

        wp.launch(
            kernel=compute_rigid_contact_forces,
            dim=max_contacts,
            inputs=[
                float(dt),
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                body_q,
                body_q_prev,
                self.model.body_com,
                self.body_body_contact_penalty_k,
                self.body_body_contact_material_kd,
                self.body_body_contact_material_mu,
                self.body_body_contact_lambda,
                self.body_body_contact_C0,
                self.rigid_contact_alpha,
                self.rigid_contact_hard,
                float(self.friction_epsilon),
            ],
            outputs=[
                self._rigid_contact_body0,
                self._rigid_contact_body1,
                self._rigid_contact_point0_world,
                self._rigid_contact_point1_world,
                contacts.rigid_contact_force,
            ],
            device=self.device,
        )

        return (
            self._rigid_contact_body0,
            self._rigid_contact_body1,
            self._rigid_contact_point0_world,
            self._rigid_contact_point1_world,
            contacts.rigid_contact_force,
            contacts.rigid_contact_count,
        )

    def _finalize_particles(self, state_out: State, dt: float):
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

    def _finalize_rigid_bodies(self, state_in: State, state_out: State, dt: float, apply_stick_deadzone: bool):
        """Finalize rigid body velocities and Dahl friction state after AVBD iterations (post-iteration phase).

        Updates rigid body velocities using BDF1 and updates Dahl hysteresis state for cable bending.
        Also transfers the final body poses from state_in to state_out. When requested,
        the fused finalize kernel first applies the body-level stick-contact deadzone
        before computing velocity from the accepted pose.
        """
        model = self.model

        # Early exit if no rigid bodies or rigid bodies are driven by an external solver
        if model.body_count == 0 or self.integrate_with_external_rigid_solver:
            return

        wp.launch(
            kernel=update_body_velocity,
            inputs=[
                dt,
                state_in.body_q,
                model.body_com,
                self.body_body_contact_buffer_pre_alloc,
                self.body_body_contact_counts,
                self.body_body_contact_indices,
                self.body_body_contact_stick_flag,
                int(apply_stick_deadzone),
                self.rigid_contact_stick_freeze_translation_eps,
                self.rigid_contact_stick_freeze_angular_eps,
            ],
            outputs=[self.body_q_prev, state_out.body_qd, state_in.body_qd, state_out.body_q],
            dim=model.body_count,
            device=self.device,
        )

        if self.enable_dahl_friction and model.joint_count > 0:
            wp.launch(
                kernel=update_cable_dahl_state,
                inputs=[
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    self.joint_constraint_start,
                    self.joint_penalty_k_max,
                    self.joint_is_hard,
                    state_out.body_q,
                    model.body_q,
                    self.joint_dahl_eps_max,
                    self.joint_dahl_tau,
                    self.joint_sigma_prev,
                    self.joint_kappa_prev,
                    self.joint_dkappa_prev,
                ],
                dim=model.joint_count,
                device=self.device,
            )

    def _collision_detection_penetration_free(self, current_state: State):
        # particle_displacements is based on pos_prev_collision_detection
        # so reset them every time we do collision detection
        self.pos_prev_collision_detection.assign(current_state.particle_q)
        self.particle_displacements.zero_()

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
            state (newton.State):  The state whose particle positions (:attr:`~newton.State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.particle_enable_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)
