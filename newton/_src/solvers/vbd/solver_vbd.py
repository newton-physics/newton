# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
from ...sim import (
    BodyFlags,
    Contacts,
    Control,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
)
from ...sim.collide import _count_soft_particle_rigid_contact_pairs
from ...utils import is_graph_capture_allocation_enabled
from ...utils.deprecation import deprecate_nonkeyword_arguments
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from ..xpbd import kernels as xpbd_kernels
from ..xpbd.kernels import apply_joint_forces
from . import particle_vbd_kernels, rigid_vbd_kernels, vbd_coupling_kernels
from .particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
    # Topological filtering helper functions
    accumulate_particle_body_contact_force_and_hessian,
    accumulate_self_contact_force_and_hessian,
    accumulate_spring_force_and_hessian,
    # Planar DAT (Divide and Truncate) kernels
    apply_planar_truncation_parallel_by_collision,
    apply_truncation_ts,
    # Solver kernels (particle VBD)
    forward_step,
    reset_particle_state,
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
    compute_rigid_contact_forces,
    compute_rod_dahl_parameters,
    forward_step_rigid_bodies,
    init_body_body_contact_materials,
    init_body_body_contacts_alm,
    init_body_particle_contacts,
    init_rod_rest_bend_twist,
    refresh_body_structural_k,
    refresh_joint_material_params,
    reset_rigid_state,
    snapshot_body_body_contact_history,
    solve_rigid_body,
    step_body_body_contact_C0_lambda,
    step_joint_C0_lambda_rho,
    update_body_velocity,
    update_duals_body_body_contacts,
    update_duals_body_particle_contacts,
    update_duals_joint,
    update_rod_dahl_state,
)
from .tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)
from .vbd_coupling_kernels import (
    _harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel,
    _harvest_vbd_proxy_particle_body_contact_forces_kernel,
    _harvest_vbd_proxy_particle_self_contact_forces_kernel,
    _harvest_vbd_proxy_wrenches_kernel,
    _update_vbd_body_input_state_kernel,
)

__all__ = ["SolverVBD"]


def _validate_compliant_alm_material_coefficient(
    values: Any,
    name: str,
    *,
    allow_negative: bool = False,
) -> None:
    """Require material coefficients representable by solver float storage."""
    values = np.asarray(values)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if not allow_negative and np.any(values < 0.0):
        raise ValueError(f"{name} must contain nonnegative values")
    if np.any(values > np.finfo(np.float32).max):
        raise ValueError(f"{name} must not exceed the float32 maximum")


def _rigid_lambda_retention(alpha: float, gamma: float, use_compliant_alm: bool) -> float:
    """Resolve cross-step multiplier retention without coupling ALM state to stabilization.

    Compliant ALM uses ``gamma`` because alpha only defines the stabilized
    residual. The legacy path keeps its historical ``alpha * gamma`` policy.
    """
    return gamma if use_compliant_alm else alpha * gamma


class SolverVBD(SolverBase, CouplingInterface):
    """An implicit solver using Vertex Block Descent (VBD) for particles and Augmented VBD (AVBD) for rigid bodies.

    .. experimental::
        SolverVBD's public API and behavior may change without prior notice.

    This unified solver supports:
        - Particle simulation (cloth, soft bodies) using the VBD algorithm
        - Rigid body simulation (joints, contacts) using the AVBD algorithm
        - Coupled particle-rigid body systems

    For rigid bodies, two paths are supported:

    - **Compliant ALM** (``rigid_compliant_alm=True``, recommended): one
      finite-material formulation for structural joints, drives, limits, and
      body-body contacts. Authored finite stiffness controls physical compliance,
      while ``SolverVBD`` selects an internal ALM metric ``rho`` for numerical
      conditioning. For a material row, they combine as
      ``k_eff = k*rho/(k+rho)``, and a multiplier ``lambda`` carries accumulated
      reaction. The per-slot joint **hard/soft** API (authored via
      ``model.vbd.joint_is_hard`` or :meth:`set_joint_constraint_mode`) is
      deprecated. ``rigid_contact_hard`` is a separate legacy-contact control and
      has no formulation effect under compliant ALM. Both distinctions will be
      removed with the legacy path.
    - **Legacy AVBD** (``rigid_compliant_alm=False``, deprecated): penalty stiffness
      that is fixed by default (``rigid_avbd_beta=0``) or ramped per iteration from
      ``k_start`` seeds, where non-rod joint slots default to hard mode (augmented
      Lagrangian with persistent lambda and C0 stabilization) and rod stretch,
      shear, bend, and twist default to soft (penalty-based). Deprecated as of
      Newton 1.6 and will be removed in a future release; omitting
      ``rigid_compliant_alm`` is deprecated because the default will change to
      ``True``.

    Joint limitations:
        - Supported joint types: BALL, FIXED, FREE, REVOLUTE, PRISMATIC, D6, ROD.
          DISTANCE joints are not supported.
        - :attr:`~newton.Model.joint_enabled` is supported for all joint types and
          is read live. After changing enable flags, call
          :meth:`notify_model_changed` with
          :attr:`~newton.ModelFlags.JOINT_PROPERTIES` to refresh derived contact
          conditioning. Structural-slot material (``rigid_joint_linear_ke``/
          ``rigid_joint_angular_ke``), constraint layout, and rest-angle offsets are
          captured at construction; rebuild ``SolverVBD`` after changing them.
        - :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd` are supported
          for REVOLUTE, PRISMATIC, D6 (as drives), and ROD (as stretch, shear,
          bend, and twist stiffness and damping).
          VBD interprets ``kd`` as absolute damping in physical units. ROD values are cached in
          solver-owned buffers at construction; after editing them call
          :meth:`notify_model_changed` with
          :attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES` to refresh that cache. REVOLUTE,
          PRISMATIC, and D6 drive/limit coefficients are gathered live from the model on every
          solve and need no notification -- except on the deprecated legacy AVBD path, whose
          cached penalty state the same flag refreshes.
        - :attr:`~newton.Model.joint_limit_lower`/:attr:`~newton.Model.joint_limit_upper` and
          :attr:`~newton.Model.joint_limit_ke`/:attr:`~newton.Model.joint_limit_kd` are supported
          for REVOLUTE, PRISMATIC, and D6 joints.
        - :attr:`~newton.Control.joint_f` (feedforward forces) is supported.
        - Not supported: :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          :attr:`~newton.Model.joint_target_mode`, equality constraints, mimic constraints.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Buffer sizing:
        Body-body contact state is pre-allocated from ``model.rigid_contact_max`` when a
        :class:`~newton.CollisionPipeline` has already published it and this solver owns the
        rigid system. Body-particle contact state is pre-sized from a world-aware
        particle-shape pair count, which excludes the
        ``enable_rigid_soft_full_surface_contact`` edge/face headroom. Both grow from
        ``Contacts`` on the first :meth:`step`, and the rigid contact force outputs grow in
        :meth:`collect_rigid_contact_forces`. During graph capture,
        ordinary lazy resizing is supported on CPU and on CUDA with Warp's
        stream-ordered memory pool enabled; otherwise the solver raises with
        guidance to pre-size before capture. Rigid contact history is
        cross-replay-persistent state, so it must always be allocated before
        capture regardless of the device's allocation-during-capture support --
        allocating it inside a graph records a `wp.zeros` fill that wipes the
        warm-start buffers on every replay. With ``rigid_contact_history=True``,
        construct :class:`~newton.CollisionPipeline` before ``SolverVBD``, or run
        one uncaptured solver step before capture.

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

        For CUDA graph capture, the recommended construction order is
        ``CollisionPipeline`` -> ``Contacts`` -> ``SolverVBD``, all before capture.

    Example
    -------

    .. code-block:: python

        # Automatically color both particles and rigid bodies
        builder.color()

        model = builder.finalize()

        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        solver = newton.solvers.SolverVBD(
            model,
            rigid_compliant_alm=True,
        )

        # Initialize states and control
        state_in = model.state()
        state_out = model.state()
        control = model.control()

        # Simulation loop
        for i in range(100):
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    class JointSlot:
        """Named constraint slot indices for :meth:`set_joint_constraint_mode`.

        Structural constraint slots by joint type:
          - ROD: STRETCH=0, SHEAR=1, BEND=2, TWIST=3
          - BALL: LINEAR=0 only
          - FIXED/REVOLUTE/PRISMATIC/D6: LINEAR=0, ANGULAR=1

        STRETCH/SHEAR/BEND/TWIST name the four-slot layout emitted by the builder
        rod APIs and apply only to :attr:`~newton.JointType.ROD`. Only structural
        slots are named here; per-DOF drive/limit slots (slot 2+ on non-rod
        joints) are not.
        """

        # Non-rod structural slots.
        LINEAR = 0
        ANGULAR = 1
        # Rod structural slots (all four are linear/angular rod constraints; they
        # are not the non-rod LINEAR/ANGULAR despite STRETCH sharing index 0).
        STRETCH = 0
        SHEAR = 1
        BEND = 2
        TWIST = 3

    @deprecate_nonkeyword_arguments
    def __init__(
        self,
        model: Model,
        *,
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
        # Rigid body - constraint formulation and stabilization
        rigid_compliant_alm: bool | None = None,  # None retains legacy and emits the scoped migration warning
        rigid_avbd_alpha: float | None = None,  # Shared alpha override; None uses mode defaults
        rigid_avbd_joint_alpha: float | None = None,  # Joint alpha override
        rigid_avbd_contact_alpha: float | None = None,  # Body-body contact alpha override
        rigid_avbd_beta: float = 0.0,  # Legacy AVBD penalty ramp rate per iteration
        rigid_avbd_linear_beta: float | None = None,  # Legacy linear beta override
        rigid_avbd_angular_beta: float | None = None,  # Legacy angular beta override
        rigid_avbd_gamma: float = 0.999,  # Per-step decay for persisted lambda (and legacy penalty k)
        # Rigid body - contacts
        rigid_contact_hard: bool = True,  # Legacy body-body contact hard/soft mode
        rigid_contact_history: bool = False,  # Body-body contact numeric warm-start
        rigid_contact_stick_motion_eps: float | None = None,  # Deprecated and ignored
        rigid_contact_stick_freeze_translation_eps: float | None = None,  # Deprecated and ignored
        rigid_contact_stick_freeze_angular_eps: float | None = None,  # Deprecated and ignored
        rigid_contact_k_start: float = 1.0e2,  # Legacy AVBD contact penalty ramp seed
        rigid_body_contact_buffer_size: int = 64,  # Per-body body-body contact list capacity
        rigid_body_particle_contact_buffer_size: int = 256,  # Per-body soft-contact list capacity (particle + edge/face)
        # Rigid body - joints
        rigid_joint_linear_ke: float = 1.0e5,  # Structural linear joint stiffness
        rigid_joint_angular_ke: float = 1.0e5,  # Structural angular joint stiffness
        rigid_joint_linear_k_start: float = 1.0e2,  # Legacy AVBD linear joint penalty ramp seed
        rigid_joint_angular_k_start: float = 1.0e1,  # Legacy AVBD angular joint penalty ramp seed
        rigid_joint_linear_kd: float = 0.0,  # Absolute damping for non-rod linear joint constraints
        rigid_joint_angular_kd: float = 0.0,  # Absolute damping for non-rod angular joint constraints
        deterministic: wp.DeterministicMode | None = None,
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

            rigid_compliant_alm: Unified compliant-ALM mode for body-body contacts,
                structural joints, drives, and limits. This is the recommended path.
                Defaults to ``None``, which currently selects the legacy path. When
                ``SolverVBD`` integrates rigid bodies, omitting this argument emits a
                ``DeprecationWarning`` because the default will change to ``True``
                (deprecated as of Newton 1.6; the legacy path will be removed in a
                future release). Pass ``True`` to adopt compliant ALM now, or ``False``
                to keep the legacy path during the migration window. Finite authored
                coefficients define the material response, while ``SolverVBD`` selects
                ``rho`` internally for numerical conditioning. Values used with legacy
                hard constraints may require retuning for the desired deformation.
                Values must be finite and representable in float32; infinity is unsupported.
            rigid_avbd_alpha: C0 stabilization strength (``C_stab = C - alpha * C0``). Range: [0, 1].
                Controls both joints and body-body contacts when neither class-specific
                override (``rigid_avbd_joint_alpha`` / ``rigid_avbd_contact_alpha``) is set.
                ``None`` leaves mode defaults: compliant ALM uses ``0.0`` for both
                constraint classes (the raw material residual), while the legacy path
                uses ``0.95``.
            rigid_avbd_joint_alpha: Joint-specific alpha override. ``None`` falls back to
                ``rigid_avbd_alpha`` when set, otherwise to ``0.0`` under compliant ALM or
                ``0.95`` on the legacy path.
            rigid_avbd_contact_alpha: Body-body contact alpha override. ``None`` falls back
                to ``rigid_avbd_alpha`` when set, otherwise ``0.0`` under compliant ALM (authored
                contact stiffness applies to the raw residual) or ``0.95`` on the legacy path.
                Under compliant ALM, alpha is stabilization only; retention is set separately by
                ``rigid_avbd_gamma``.
            rigid_avbd_beta: Legacy AVBD penalty ramp rate per iteration. ``0`` (default)
                disables ramping (fixed-k). Set to e.g. ``1e5`` for ramping. Used for both
                linear and angular constraints unless overridden. Does not tune the
                internal compliant-ALM ``rho`` for converted rigid rows. Note: linear
                (meters) and angular (radians) constraints have different units, so the
                overrides should be used for production tuning.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated for all uses. Body-particle contacts
                    continue to honor this control during migration; keep the effective beta
                    at ``0`` (the default behavior) and author fixed material stiffness instead.
            rigid_avbd_linear_beta: Legacy linear beta override for linear constraints
                (meters). ``None`` (default) uses ``rigid_avbd_beta``. Does not tune
                compliant-ALM ``rho``.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated for all uses. Body-particle contacts
                    continue to honor this control during migration; keep the effective beta
                    at ``0`` (the default behavior) and author fixed material stiffness instead.
            rigid_avbd_angular_beta: Legacy angular beta override for angular constraints
                (radians). ``None`` (default) uses ``rigid_avbd_beta``. Does not tune
                compliant-ALM ``rho``.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated. Keep the effective beta at ``0`` (the
                    default behavior) and author fixed material stiffness instead.
            rigid_avbd_gamma: Per-step decay factor for penalty k and persisted lambda.
                Compliant ALM joints and validated contacts retain lambda by ``gamma``;
                the legacy path retains lambda by ``alpha * gamma``. Lower values
                discard history faster.
            rigid_contact_hard: Legacy body-body contact hard/soft mode. With
                ``rigid_compliant_alm=True``, contacts use the ALM path. With
                ``rigid_compliant_alm=False``, ``True`` selects legacy hard AVBD contact
                and ``False`` selects legacy penalty-only contact.

                .. deprecated:: 1.6
                    Use ``rigid_compliant_alm=True`` and author finite contact stiffness.
            rigid_contact_history: Whether to persist body-body numeric contact state
                across steps using ``Contacts.rigid_contact_match_index``. Compliant ALM
                restores the normal multiplier for matched rows. With latest
                matching it also restores projected tangential multipliers as a
                numerical warm start; with sticky matching, tangential memory is
                represented by the collision pipeline's replayed material anchor.
                Legacy hard contacts restore the full multiplier; legacy soft contacts
                restore penalty k only. Contact geometry remains owned by the
                collision pipeline. Requires ``CollisionPipeline(contact_matching="latest")`` or ``"sticky"``.
                Ignored when ``integrate_with_external_rigid_solver=True`` or
                ``model.body_count == 0``. During graph capture, construct the
                collision pipeline before ``SolverVBD`` so history is pre-allocated,
                or run one uncaptured solver step before capture.
            rigid_contact_stick_motion_eps: Deprecated and ignored. SolverVBD no longer
                classifies contacts as sticking. Use
                ``CollisionPipeline(contact_matching="sticky",
                contact_matching_pos_threshold=...)`` for persistent contact geometry.

                .. deprecated:: 1.5
            rigid_contact_stick_freeze_translation_eps: Deprecated and ignored. The
                SolverVBD body-level contact deadzone was removed.

                .. deprecated:: 1.5
            rigid_contact_stick_freeze_angular_eps: Deprecated and ignored. The
                SolverVBD body-level contact deadzone was removed.

                .. deprecated:: 1.5
            rigid_contact_k_start: Body-body and body-particle contact penalty seed for
                legacy AVBD ramping [N/m]. Used when ``rigid_avbd_linear_beta`` (or
                ``rigid_avbd_beta`` fallback) is greater than zero. When the linear beta
                is 0, k is fixed at the contact stiffness regardless of this value.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated for all uses. Body-particle contacts
                    continue to honor this control during migration; keep the effective beta
                    at ``0`` (the default behavior) and author fixed contact stiffness instead.
            rigid_body_contact_buffer_size: Max body-body contacts per rigid body for per-body contact lists.
            rigid_body_particle_contact_buffer_size: Max body-particle soft contacts tracked per rigid
                body, covering both particle-vs-surface and full-surface edge/face contacts.
            rigid_joint_linear_ke: Material stiffness for non-rod structural linear joint slots [N/m].
            rigid_joint_angular_ke: Material stiffness for non-rod structural angular joint slots [N·m/rad].
            rigid_joint_linear_k_start: Linear penalty seed for legacy AVBD ramping [N/m]. Used when
                ``rigid_avbd_linear_beta`` (or ``rigid_avbd_beta`` fallback) is greater than zero.
                When the linear beta is 0, k is fixed at the joint stiffness regardless of this value.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated. Keep the effective beta at ``0`` (the
                    default behavior) and author fixed joint stiffness instead.
            rigid_joint_angular_k_start: Angular penalty seed for legacy AVBD ramping [N·m/rad]. Used when
                ``rigid_avbd_angular_beta`` (or ``rigid_avbd_beta`` fallback) is greater than zero.
                When the angular beta is 0, k is fixed at the joint stiffness regardless of this value.

                .. deprecated:: 1.6
                    Penalty ramping is deprecated. Keep the effective beta at ``0`` (the
                    default behavior) and author fixed joint stiffness instead.
            rigid_joint_linear_kd: Damping coefficient for non-rod linear joint constraints [N·s/m].
                Negative values are clamped to 0.
            rigid_joint_angular_kd: Damping coefficient for non-rod angular joint constraints [N·m·s/rad].
                Negative values are clamped to 0.
            deterministic: Opt-in determinism for this solver's atomic-emitting
                kernel modules. Pass a :class:`warp.DeterministicMode`, or
                ``None`` (default) to inherit the current
                ``wp.config.deterministic`` mode.

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
            - Dahl hysteresis friction for rod angular response is controlled by custom model attributes
              ``model.vbd.dahl_eps_max`` and ``model.vbd.dahl_tau``. Register them with
              ``SolverVBD.register_custom_attributes`` before building the model. Dahl friction is
              enabled only when positive Dahl parameters are authored.

        """
        integrates_rigid_bodies = model.body_count > 0 and not integrate_with_external_rigid_solver

        # TODO: Complete the Newton 1.6 deprecation by defaulting omitted
        # rigid_compliant_alm to True and removing this warning after the migration window.
        if rigid_compliant_alm is None:
            if integrates_rigid_bodies:
                warnings.warn(
                    "Omitting rigid_compliant_alm is deprecated as of Newton 1.6 because the default will "
                    "change from the legacy penalty/AVBD path (False) to unified compliant ALM (True), which "
                    "is becoming the standard for rigid VBD. The legacy path is deprecated and will be removed "
                    "in a future release. Pass rigid_compliant_alm=True to adopt compliant ALM now, or "
                    "rigid_compliant_alm=False to keep the legacy path during the migration window.",
                    DeprecationWarning,
                    # __init__ is wrapped by @deprecate_nonkeyword_arguments.
                    stacklevel=3,
                )
            rigid_compliant_alm = False

        if rigid_avbd_beta < 0:
            raise ValueError(f"rigid_avbd_beta must be >= 0, got {rigid_avbd_beta}")
        rigid_avbd_linear_beta = rigid_avbd_linear_beta if rigid_avbd_linear_beta is not None else rigid_avbd_beta
        rigid_avbd_angular_beta = rigid_avbd_angular_beta if rigid_avbd_angular_beta is not None else rigid_avbd_beta
        if (
            rigid_contact_stick_motion_eps is not None
            or rigid_contact_stick_freeze_translation_eps is not None
            or rigid_contact_stick_freeze_angular_eps is not None
        ):
            warnings.warn(
                "SolverVBD rigid_contact_stick_motion_eps, "
                "rigid_contact_stick_freeze_translation_eps, and "
                "rigid_contact_stick_freeze_angular_eps are deprecated and ignored, "
                "and will be removed in a future release. "
                "Use CollisionPipeline(contact_matching='sticky', "
                "contact_matching_pos_threshold=...) for persistent contact geometry. "
                "The SolverVBD body-level contact deadzone was removed.",
                DeprecationWarning,
                # __init__ is wrapped by @deprecate_nonkeyword_arguments.
                stacklevel=3,
            )

        super().__init__(model)

        effective_deterministic = deterministic if deterministic is not None else wp.config.deterministic
        particle_deterministic_max_records = 0
        coupling_deterministic_max_records = 0
        if particle_enable_self_contact and effective_deterministic != wp.DeterministicMode.NOT_GUARANTEED:
            edge_iterations = (
                particle_edge_contact_buffer_size + NUM_THREADS_PER_COLLISION_PRIMITIVE - 1
            ) // NUM_THREADS_PER_COLLISION_PRIMITIVE
            vertex_iterations = (
                particle_vertex_contact_buffer_size + NUM_THREADS_PER_COLLISION_PRIMITIVE - 1
            ) // NUM_THREADS_PER_COLLISION_PRIMITIVE
            truncation_records = 4 * (edge_iterations + vertex_iterations)
            force_records = 2 * edge_iterations + 4 * vertex_iterations
            if model.shape_count > 0:
                force_records += 1
            particle_deterministic_max_records = max(truncation_records, force_records)
            coupling_deterministic_max_records = 2 * edge_iterations + 3 * vertex_iterations
        if model.particle_count > 0:
            self._set_module_options(
                {
                    "deterministic": effective_deterministic,
                    "deterministic_max_records": particle_deterministic_max_records,
                },
                module=particle_vbd_kernels,
            )
        self._set_module_options(
            {
                "deterministic": effective_deterministic,
                "deterministic_max_records": coupling_deterministic_max_records,
            },
            module=vbd_coupling_kernels,
        )

        options = {"deterministic": effective_deterministic, "deterministic_max_records": 0}
        if integrates_rigid_bodies:
            self._set_module_options(options, module=rigid_vbd_kernels)
        if model.joint_count > 0:
            self._set_module_options(
                {"deterministic": effective_deterministic, "deterministic_max_records": 0},
                module=xpbd_kernels,
            )

        # Common parameters
        self.iterations = iterations
        self.friction_epsilon = friction_epsilon
        self._joint_mode_deprecation_warned = False

        # Rigid integration mode: when True, rigid bodies are integrated by an external
        # solver (one-way coupling). SolverVBD will not move rigid bodies, but can still
        # participate in particle-rigid interaction on the particle side.
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver
        self._integrates_rigid_bodies = integrates_rigid_bodies

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
            rigid_compliant_alm,
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

        # Particle adjacency info: reuse the shared device copy built once at finalize (the VBD
        # solver and the collision pipeline both use it, so it is uploaded only once).
        if self.model.soft_mesh_adjacency_device is None:
            raise ValueError("model.soft_mesh_adjacency_device is missing; finalize the model with ModelBuilder.")
        self.particle_adjacency = self.model.soft_mesh_adjacency_device

        # Self-contact settings
        self.particle_enable_self_contact = particle_enable_self_contact
        self.particle_self_contact_radius = particle_self_contact_radius
        self.particle_self_contact_margin = particle_self_contact_margin
        self.particle_q_rest = model.particle_q

        # Tile solve settings
        if model.device.is_cpu and particle_enable_tile_solve and wp.config.log_level <= wp.LOG_DEBUG:
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
                topological_contact_filter_threshold=particle_topological_contact_filter_threshold,
                external_vertex_triangle_filtering_map=particle_external_vertex_contact_filtering_map,
                external_edge_edge_filtering_map=particle_external_edge_contact_filtering_map,
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
        rigid_compliant_alm: bool,
        rigid_avbd_alpha: float | None,
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
        """Initialize rigid-body VBD data structures and settings.

        This includes:
          - Rigid-only joint/contact state (legacy AVBD and compliant ALM)
          - Shared interaction state for body-particle (rigid-particle) soft contacts
        """
        # Validate rigid constraint parameters
        if rigid_avbd_alpha is not None and not (0.0 <= rigid_avbd_alpha <= 1.0):
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
        if rigid_compliant_alm:
            _validate_compliant_alm_material_coefficient(rigid_joint_linear_ke, "rigid_joint_linear_ke")
            _validate_compliant_alm_material_coefficient(rigid_joint_angular_ke, "rigid_joint_angular_ke")
            _validate_compliant_alm_material_coefficient(
                rigid_joint_linear_kd, "rigid_joint_linear_kd", allow_negative=True
            )
            _validate_compliant_alm_material_coefficient(
                rigid_joint_angular_kd, "rigid_joint_angular_kd", allow_negative=True
            )
        self.rigid_avbd_gamma = rigid_avbd_gamma
        self.rigid_contact_k_start_value = -1.0 if rigid_avbd_linear_beta == 0.0 else float(rigid_contact_k_start)
        self.rigid_compliant_alm = bool(rigid_compliant_alm)
        if self.rigid_compliant_alm:
            self._validate_compliant_contact_materials()
            self._validate_compliant_joint_dof_materials()

        self.rigid_joint_linear_k_start = rigid_joint_linear_k_start if rigid_avbd_linear_beta > 0.0 else None
        self.rigid_joint_angular_k_start = rigid_joint_angular_k_start if rigid_avbd_angular_beta > 0.0 else None
        # Joint alpha precedence: joint override > shared override > mode default.
        # Compliant ALM defaults to the raw material residual (alpha=0).
        if rigid_avbd_joint_alpha is not None:
            self.rigid_joint_alpha = rigid_avbd_joint_alpha
        elif rigid_avbd_alpha is not None:
            self.rigid_joint_alpha = rigid_avbd_alpha
        elif rigid_compliant_alm:
            self.rigid_joint_alpha = 0.0
        else:
            self.rigid_joint_alpha = 0.95
        self.rigid_linear_beta = rigid_avbd_linear_beta
        self.rigid_angular_beta = rigid_avbd_angular_beta
        self.rigid_contact_hard = int(rigid_contact_hard)
        self.rigid_contact_history = rigid_contact_history
        # Contact alpha precedence: contact override > shared override > mode default.
        # Under compliant ALM it controls only residual stabilization.
        if rigid_avbd_contact_alpha is not None:
            self.rigid_contact_alpha = rigid_avbd_contact_alpha
        elif rigid_avbd_alpha is not None:
            self.rigid_contact_alpha = rigid_avbd_alpha
        else:
            self.rigid_contact_alpha = 0.0 if rigid_compliant_alm else 0.95

        # Joint constraint stiffness and damping for non-rod structural joints
        self.rigid_joint_linear_ke = rigid_joint_linear_ke
        self.rigid_joint_angular_ke = rigid_joint_angular_ke
        self.rigid_joint_linear_kd = max(0.0, rigid_joint_linear_kd)
        self.rigid_joint_angular_kd = max(0.0, rigid_joint_angular_kd)

        # -------------------------------------------------------------
        # Rigid-only solver state (used when SolverVBD integrates bodies)
        # -------------------------------------------------------------
        if self._integrates_rigid_bodies:
            # The first step's State establishes pose history; reset marks selected
            # worlds for a new baseline. Final slot: entities without a world.
            history_mask_size = model.world_count + 1
            self._rigid_pose_rebaseline_mask = wp.ones(history_mask_size, dtype=wp.bool, device=self.device)
            # Contact-reset state is consumed only by the warm-start refresh, so
            # allocate it (and let the reset kernel write it) only when enabled.
            if self.rigid_contact_history:
                self._contact_history_reset_mask = wp.zeros(history_mask_size, dtype=wp.bool, device=self.device)
                self._contact_history_reset_pending = wp.zeros(1, dtype=wp.int32, device=self.device)
            else:
                self._contact_history_reset_mask = None
                self._contact_history_reset_pending = None

            # Deterministic fallbacks for inspection before the first step overwrites them.
            self.body_q_prev = wp.clone(model.body_q, device=self.device)
            self._coupling_body_q_prev_snapshot = wp.clone(model.body_q, device=self.device)
            self.body_inertia_q = wp.zeros_like(model.body_q, device=self.device)  # inertial target poses

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

            # Joint constraint layout, legacy penalty state, and material data.
            self._init_joint_constraint_layout()
            (
                self.joint_penalty_k,
                self.joint_penalty_k_min,
                self.joint_material_k,
                self.joint_rho,
                self.joint_penalty_kd,
                self.joint_is_hard,
            ) = self._init_joint_penalty_k()
            self._init_structural_k()
            self.joint_rest_angle = self._init_joint_rest_angle()

            # Body-body contact state (pre-allocated in __init__ when possible, resized on first step otherwise).
            self.body_body_contact_penalty_k = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_normal_rho = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_ke = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_kd = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_material_mu = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_tangent_rho = wp.zeros(0, dtype=float, device=self.device)
            self.body_body_contact_lambda = wp.zeros(0, dtype=wp.vec3, device=self.device)
            self.body_body_contact_C0 = wp.zeros(0, dtype=wp.vec3, device=self.device)

            # Rigid contact warm-start buffers.
            self._prev_contact_lambda = None
            self._prev_contact_penalty_k = None
            self._prev_contact_normal = None

            # Joint augmented-Lagrangian state (vec3, per-joint, bilateral)
            self.joint_lambda_lin = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_lambda_ang = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C0_lin = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C0_ang = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            # Shared directional support; drive and limit derive separate rho policies.
            self.joint_drive_limit_support = wp.zeros(model.joint_dof_count, dtype=float, device=self.device)
            # Bilateral drive dual, cleared whenever the drive row stops existing.
            self.joint_drive_lambda = wp.zeros(model.joint_dof_count, dtype=float, device=self.device)
            # Unilateral limit dual, held separately from the drive so neither can
            # destroy the other's reaction. Sign encodes the bound: >0 upper, <0 lower.
            self.joint_limit_lambda = wp.zeros(model.joint_dof_count, dtype=float, device=self.device)

            # Dahl friction state (rod angular hysteresis, persistent across timesteps)
            self.joint_sigma_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_kappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_dkappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Pre-computed Dahl parameters (frozen during iterations, updated per timestep)
            self.joint_sigma_start = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C_fric = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Dahl friction: registered custom attributes are inert until enabled by positive values.
            vbd_attrs: Any = getattr(model, "vbd", None)
            has_dahl = (
                model.joint_count > 0
                and vbd_attrs is not None
                and hasattr(vbd_attrs, "dahl_eps_max")
                and hasattr(vbd_attrs, "dahl_tau")
            )
            if has_dahl:
                self.joint_dahl_eps_max = vbd_attrs.dahl_eps_max
                self.joint_dahl_tau = vbd_attrs.dahl_tau
                dahl_eps_max = self._to_numpy(self.joint_dahl_eps_max, dtype=float)
                dahl_tau = self._to_numpy(self.joint_dahl_tau, dtype=float)
                self.enable_dahl_friction = bool(np.any((dahl_eps_max > 0.0) & (dahl_tau > 0.0)))
            else:
                self.joint_dahl_eps_max = wp.zeros(model.joint_count, dtype=float, device=self.device)
                self.joint_dahl_tau = wp.zeros(model.joint_count, dtype=float, device=self.device)
                self.enable_dahl_friction = False

            # Per-joint DER rest invariants, refreshed at init and on model change
            # (see _refresh_rod_rest_bend_twist_cache): the parent-local rest
            # curvature binormal (bend) and the rest transported-material twist.
            # Rod joints use local +Z as the material tangent (a SolverVBD convention).
            self.joint_rod_rest_kb_local = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_rod_rest_twist = wp.zeros(model.joint_count, dtype=float, device=self.device)
            self._refresh_rod_rest_bend_twist_cache()

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
            # Not shape_count * particle_count: that counts cross-world pairs, so it is quadratic in
            # world count and can exceed Warp's int32 array shape limit. A hint only -- the first step
            # grows this to contacts.soft_contact_max, raising if capture cannot allocate.
            self._init_body_particle_contact_state(_count_soft_particle_rigid_contact_pairs(model))

        # Kinematic body support: create effective inv_mass / inv_inertia arrays
        # with kinematic bodies zeroed out.
        self._init_kinematic_state()

        # Pre-allocate body-body contact buffers when the contact capacity is
        # already known; otherwise lazy allocation handles the first step.
        rcm = getattr(model, "rigid_contact_max", 0) or 0
        if rcm > 0 and self._integrates_rigid_bodies:
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
        has_body_coloring = len(model.body_color_groups) > 0

        if self._integrates_rigid_bodies and not has_body_coloring:
            raise ValueError(
                "model.body_color_groups is empty but rigid bodies are present! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self._apply_module_options()
        refresh_structural_k = (
            bool(flags & (ModelFlags.JOINT_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES))
            and self._integrates_rigid_bodies
            and self.model.joint_count > 0
        )
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()
        if flags & ModelFlags.JOINT_DOF_PROPERTIES and self._integrates_rigid_bodies and self.model.joint_count > 0:
            if self.rigid_compliant_alm:
                self._validate_compliant_joint_dof_materials()
            # Must run before _refresh_structural_k() below: that summary reads
            # joint_material_k as an input, so a stale material_k here would
            # produce a stale body_structural_k that is harder to spot.
            self._refresh_joint_material_params()
        if refresh_structural_k:
            self._refresh_structural_k()
        if flags & (ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_PROPERTIES):
            self._refresh_rod_rest_bend_twist_cache()

    @override
    def coupling_supports_inertial_property_refresh(self) -> bool:
        return True

    @override
    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        return True

    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """Convert input body pose updates into VBD-compatible history updates."""
        self._apply_module_options()
        flags = int(flags)

        if (
            not (flags & StateFlags.BODY_Q)
            or state.body_q is None
            or state.body_qd is None
            or not self._integrates_rigid_bodies
        ):
            return

        if dt <= 0.0:
            # A reset distributes state before its world mask selects histories.
            if not iteration_restart:
                wp.copy(dest=self.body_q_prev, src=state.body_q)
            return

        if iteration_restart:
            # Restore the beginning-of-iteration history after a previous solve advanced it.
            wp.copy(dest=self.body_q_prev, src=self._coupling_body_q_prev_snapshot)

        wp.launch(
            _update_vbd_body_input_state_kernel,
            dim=self.model.body_count,
            inputs=[
                float(dt),
                self.model.body_flags,
                int(BodyFlags.KINEMATIC),
                self.model.body_world,
                self._rigid_pose_rebaseline_mask,
                state.body_q,
                self.body_q_prev,
                state.body_qd,
            ],
            device=self.device,
        )

        if not iteration_restart:
            # Snapshot pass-0 history so restarted iterations restore the same baseline.
            wp.copy(dest=self._coupling_body_q_prev_snapshot, src=self.body_q_prev)

    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Update rigid history cadence for proxy contacts."""
        # Do not call super(); we can keep proxy-proxy collisions as we
        # are using a custom force harvesting hook
        self.set_rigid_history_update(bool(contacts_freshly_detected))
        return contacts

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Harvest contact-only proxy-body wrenches.

        VBD deliberately does not rely on the default momentum harvest here.
        The generic proxy path filters proxy-vs-proxy and proxy-vs-static rigid
        contacts so harvested momentum only reflects coupling-relevant
        interactions. VBD relaxes that restriction because allowing some proxy
        interaction inside the destination solve can strengthen the coupled
        solve. Those extra interactions still must not feed back through the
        coupling interface, so VBD harvests explicit contact forces instead of
        inferring feedback from total proxy momentum change.
        """
        self._apply_module_options()
        if not self._integrates_rigid_bodies:
            super().coupling_harvest_proxy_wrenches(
                body_local_to_proxy_global,
                out_body_f,
                body_qd_before=body_qd_before,
                state=state,
                state_out=state_out,
                contacts=contacts,
                dt=dt,
            )
            return

        out_body_f.zero_()
        if contacts is None:
            return

        body_q_prev = self._coupling_body_q_prev_snapshot

        if contacts.rigid_contact_max > 0:
            body0, body1, point0, point1, force_on_body1, rigid_contact_count = self.collect_rigid_contact_forces(
                state_out.body_q,
                body_q_prev,
                contacts,
                dt,
            )
            wp.launch(
                _harvest_vbd_proxy_wrenches_kernel,
                dim=contacts.rigid_contact_max,
                inputs=[
                    rigid_contact_count,
                    body0,
                    body1,
                    point0,
                    point1,
                    force_on_body1,
                    self.model.body_inv_mass,
                    self.model.body_flags,
                    body_local_to_proxy_global,
                    int(BodyFlags.PROXY),
                    self.model.body_com,
                    state_out.body_q,
                    out_body_f,
                ],
                device=self.device,
            )

        if contacts.soft_contact_max > 0 and self.body_particle_contact_penalty_k.shape[0] >= contacts.soft_contact_max:
            wp.launch(
                _harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel,
                dim=contacts.soft_contact_max,
                inputs=[
                    float(dt),
                    body_local_to_proxy_global,
                    state_out.particle_q,
                    self.particle_q_prev,
                    self.model.particle_radius,
                    state_out.body_q,
                    body_q_prev,
                    state_out.body_qd,
                    self.model.body_com,
                    float(self.friction_epsilon),
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.model.shape_margin,
                    self.model.shape_body,
                    out_body_f,
                ],
                device=self.device,
            )

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Harvest contact-only proxy-particle forces.

        As for proxy-body harvest, this stays contact-based because VBD allows
        some proxy interaction inside the destination solve for stronger
        coupling, but those proxy-only interactions should not appear as
        feedback forces on the source side.
        """
        self._apply_module_options()
        del particle_qd_before
        out_particle_f.zero_()
        if self.model.particle_count == 0 or particle_local_to_proxy_global.shape[0] == 0:
            return

        if contacts is not None:
            contacts._assert_particle_only_soft_contacts("SolverVBD proxy-particle coupling")

        if (
            contacts is not None
            and contacts.soft_contact_max > 0
            and contacts.soft_contact_count is not None
            and contacts.soft_contact_particle is not None
            and contacts.soft_contact_shape is not None
            and self.body_particle_contact_penalty_k.shape[0] >= contacts.soft_contact_max
        ):
            if self.integrate_with_external_rigid_solver:
                body_q_for_particles = state_out.body_q
                body_q_prev_for_particles = state.body_q
                body_qd_for_particles = state_out.body_qd
            else:
                body_q_for_particles = state.body_q
                body_q_prev_for_particles = self._coupling_body_q_prev_snapshot if self.model.body_count > 0 else None
                body_qd_for_particles = state.body_qd

            wp.launch(
                _harvest_vbd_proxy_particle_body_contact_forces_kernel,
                dim=contacts.soft_contact_max,
                inputs=[
                    float(dt),
                    particle_local_to_proxy_global,
                    state.particle_q,
                    self.particle_q_prev,
                    self.model.particle_flags,
                    self.model.particle_inv_mass,
                    int(ParticleFlags.ACTIVE),
                    int(ParticleFlags.PROXY),
                    self.friction_epsilon,
                    self.model.particle_radius,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                    self.model.shape_body,
                    self.model.body_flags,
                    self.model.body_inv_mass,
                    int(BodyFlags.PROXY),
                    body_q_for_particles,
                    body_q_prev_for_particles,
                    body_qd_for_particles,
                    self.model.body_com,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.model.shape_margin,
                    out_particle_f,
                ],
                device=self.device,
            )

        if self.particle_enable_self_contact:
            wp.launch(
                _harvest_vbd_proxy_particle_self_contact_forces_kernel,
                dim=self.particle_self_contact_evaluation_kernel_launch_size,
                inputs=[
                    float(dt),
                    particle_local_to_proxy_global,
                    self.particle_q_prev,
                    state.particle_q,
                    self.model.particle_flags,
                    self.model.particle_inv_mass,
                    int(ParticleFlags.ACTIVE),
                    int(ParticleFlags.PROXY),
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.trimesh_collision_info,
                    self.particle_self_contact_radius,
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_mu,
                    self.friction_epsilon,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    out_particle_f,
                ],
                device=self.device,
                max_blocks=self.model.device.sm_count,
            )

    # =====================================================
    # Initialization Helper Methods
    # =====================================================

    def _init_body_body_contact_state(self, rigid_contact_max: int) -> None:
        """Allocate body-body contact state arrays sized to the given contact buffer capacity."""
        self.body_body_contact_penalty_k = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_normal_rho = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_ke = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_kd = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_material_mu = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_tangent_rho = wp.zeros(rigid_contact_max, dtype=float, device=self.device)
        self.body_body_contact_lambda = wp.zeros(rigid_contact_max, dtype=wp.vec3, device=self.device)
        self.body_body_contact_C0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, device=self.device)

    def _validate_compliant_contact_materials(self) -> None:
        """Validate physical contact coefficients consumed by compliant ALM."""
        if self.model.shape_count == 0:
            return
        for attribute in ("shape_material_ke", "shape_material_kd", "shape_material_mu"):
            values = self._to_numpy(getattr(self.model, attribute), dtype=float)
            _validate_compliant_alm_material_coefficient(values, f"model.{attribute}")

    def _validate_compliant_joint_dof_materials(self) -> None:
        """Validate drive/limit coefficients consumed live by compliant ALM.

        Skip when an external rigid solver owns the bodies: VBD does not evaluate
        joint drive/limit rows in that coupling mode.
        """
        if not self._integrates_rigid_bodies or self.model.joint_count == 0:
            return
        for attribute in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd"):
            values = self._to_numpy(getattr(self.model, attribute), dtype=float)
            _validate_compliant_alm_material_coefficient(values, f"model.{attribute}")

    def _init_body_particle_contact_state(self, soft_contact_max: int) -> None:
        """Allocate body-particle material arrays sized to the given soft contact capacity."""
        self.body_particle_contact_penalty_k = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_ke = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_kd = wp.zeros(soft_contact_max, dtype=float, device=self.device)
        self.body_particle_contact_material_mu = wp.zeros(soft_contact_max, dtype=float, device=self.device)

    def _init_rigid_contact_warmstart(self, rigid_contact_max: int) -> None:
        """Allocate fresh contact-history buffers."""
        cap = rigid_contact_max
        self._prev_contact_lambda = wp.zeros(cap, dtype=wp.vec3, device=self.device)
        self._prev_contact_penalty_k = wp.zeros(cap, dtype=float, device=self.device)
        self._prev_contact_normal = wp.zeros(cap, dtype=wp.vec3, device=self.device)

    def _raise_if_capturing_resize(self, name: str, current: int, required: int) -> None:
        if self.device.is_capturing and not is_graph_capture_allocation_enabled(self.device):
            raise RuntimeError(
                f"SolverVBD {name} buffer needs to grow from {current} to {required} "
                "during graph capture, but allocation during capture is not enabled on this device. "
                "Run one uncaptured step (or force-collection pass) before capture so the contact "
                "buffers are sized for the scene, or enable Warp's stream-ordered memory pool on this device. "
                "Rigid buffers can also be pre-sized by constructing CollisionPipeline before SolverVBD, "
                "which publishes model.rigid_contact_max; there is no equivalent for body-particle contacts."
            )

    def _refresh_rod_rest_bend_twist_cache(self) -> None:
        """(Re)compute rod rest bend/twist invariants from the current rest pose.

        Called once at init and again from ``notify_model_changed`` whenever joint
        frames or the rest pose change.
        """
        # The cache is only allocated when SolverVBD integrates the rigid system
        # (see _init_rigid_system); skip when bodies are handled externally.
        if not self._integrates_rigid_bodies or self.model.joint_count == 0:
            return

        joint_type_np = self._to_numpy(self.model.joint_type, dtype=np.int32)
        if not np.any(joint_type_np == int(JointType.ROD)):
            return

        wp.launch(
            kernel=init_rod_rest_bend_twist,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.body_q,
            ],
            outputs=[
                self.joint_rod_rest_kb_local,
                self.joint_rod_rest_twist,
            ],
            device=self.device,
        )

    @staticmethod
    def _to_numpy(arr, dtype=None):
        """Transfer a Warp array to CPU and return as numpy, optionally casting dtype."""
        cpu = arr.to("cpu")
        result = cpu.numpy() if hasattr(cpu, "numpy") else np.asarray(cpu)
        return result if dtype is None else result.astype(dtype, copy=False)

    def _init_joint_constraint_layout(self) -> None:
        """Initialize VBD-owned joint constraint indexing.

        VBD indexes scalar constraint components for structural joint penalties,
        compliant-ALM rho, and drive/limit penalty slots:
          - ROD:   4 scalars (stretch, shear, bend, twist)
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
                if jt[j] == JointType.ROD:
                    lin_count = int(jdof_dim[j, 0])
                    ang_count = int(jdof_dim[j, 1])
                    if lin_count != 2 or ang_count != 2:
                        raise RuntimeError(
                            "SolverVBD rigid joints: JointType.ROD requires the split "
                            "stretch/shear/bend/twist layout emitted by the rod builder APIs "
                            f"(got linear={lin_count}, angular={ang_count}) "
                            f"for joint {j}."
                        )
                    dim_np[j] = 4
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
                            "(only ROD, BALL, FIXED, REVOLUTE, PRISMATIC, and D6 are supported)."
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
            (joint_penalty_k, joint_penalty_k_min, joint_material_k, joint_rho,
            joint_penalty_kd, joint_is_hard) tuple:
              - joint_penalty_k:       mutable legacy solver penalty per constraint scalar.
              - joint_penalty_k_min:   floor for the mutable legacy solver penalty; refreshed
                                       by ``notify_model_changed(JOINT_DOF_PROPERTIES)``.
              - joint_material_k:      material stiffness (= slot-specific ke); refreshed
                                       by ``notify_model_changed(JOINT_DOF_PROPERTIES)``.
              - joint_rho:             zeroed solver-owned storage; compliant ALM fills
                                       structural slots automatically each step.
              - joint_penalty_kd:      damping coefficient per constraint scalar.
              - joint_is_hard:         hard/soft flag per constraint scalar (1 = hard, 0 = soft).
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
        legacy_lin_k_start = self.rigid_joint_linear_k_start
        legacy_ang_k_start = self.rigid_joint_angular_k_start

        with wp.ScopedDevice("cpu"):
            joint_material_k_np = np.zeros((constraint_count,), dtype=float)
            joint_k_init_np = np.zeros((constraint_count,), dtype=float)
            joint_rho_np = np.zeros((constraint_count,), dtype=float)
            joint_kd_np = np.zeros((constraint_count,), dtype=float)
            is_hard_np = np.zeros((constraint_count,), dtype=np.int32)

            jt = self._to_numpy(self.model.joint_type, dtype=int)
            jdofs = self._to_numpy(self.model.joint_qd_start, dtype=int)
            jtarget_ke = self._to_numpy(self.model.joint_target_ke, dtype=float)
            jtarget_kd = self._to_numpy(self.model.joint_target_kd, dtype=float)
            jlimit_ke = self._to_numpy(self.model.joint_limit_ke, dtype=float)
            jdof_dim = self._to_numpy(self.model.joint_dof_dim, dtype=int)
            jc_start = self._to_numpy(self.joint_constraint_start, dtype=np.int32)

            # TODO: Complete the Newton 1.6 deprecation by removing per-slot
            # hard/soft attribute compatibility with the legacy non-ALM path.
            # Per-joint hard/soft mode from model attribute (default=1, hard).
            vbd_attrs: Any = getattr(self.model, "vbd", None)
            if vbd_attrs is not None and hasattr(vbd_attrs, "joint_is_hard"):
                j_is_hard = self._to_numpy(vbd_attrs.joint_is_hard, dtype=np.int32)
                if not np.all((j_is_hard == 0) | (j_is_hard == 1)):
                    raise ValueError("model.vbd.joint_is_hard values must be 0 (soft) or 1 (hard).")
                # Authoring a soft slot is deliberate use of the deprecated hard/soft
                # distinction (the default is 1=hard, so all-hard is indistinguishable
                # from the untouched default and stays silent).
                if np.any(j_is_hard == 0):
                    warnings.warn(
                        "model.vbd.joint_is_hard (per-slot joint hard/soft mode) is deprecated as of "
                        "Newton 1.6 and will be removed with the legacy path. Under compliant ALM it has "
                        "no solver-mode effect; legacy AVBD still honors it during the migration window.",
                        DeprecationWarning,
                        # Reaches the constructor call site through _init_rigid_system.
                        stacklevel=5,
                    )
            else:
                j_is_hard = np.ones(self.model.joint_count, dtype=np.int32)

            structural_linear_ke = self.rigid_joint_linear_ke
            structural_angular_ke = self.rigid_joint_angular_ke

            def _penalty_k_init(material_k: float, seed: float | None) -> float:
                # Initial penalty k for any slot (structural or drive/limit): full material_k,
                # or a ramp-capped seed when AVBD ramping is enabled (seed is not None).
                return float(material_k) if seed is None else min(float(seed), float(material_k))

            n_j = self.model.joint_count
            for j in range(n_j):
                if jt[j] == JointType.ROD:
                    c0 = int(jc_start[j])
                    dof0 = int(jdofs[j])
                    if dof0 < 0 or (dof0 + 3) >= len(jtarget_ke) or (dof0 + 3) >= len(jtarget_kd):
                        raise RuntimeError(
                            "SolverVBD _init_joint_penalty_k: JointType.ROD requires "
                            "four material-slot entries in "
                            "model.joint_target_ke/kd starting at joint_qd_start[j]. "
                            f"Got joint_index={j}, joint_qd_start={dof0}, "
                            f"len(joint_target_ke)={len(jtarget_ke)}, len(joint_target_kd)={len(jtarget_kd)}."
                        )
                    stretch_slot = c0
                    shear_slot = c0 + 1
                    bend_slot = c0 + 2
                    twist_slot = c0 + 3

                    stretch_dof = dof0
                    shear_dof = dof0 + 1
                    bend_dof = dof0 + 2
                    twist_dof = dof0 + 3

                    ke_stretch = jtarget_ke[stretch_dof]
                    ke_shear = jtarget_ke[shear_dof]
                    ke_bend = jtarget_ke[bend_dof]
                    ke_twist = jtarget_ke[twist_dof]

                    joint_material_k_np[stretch_slot] = ke_stretch
                    joint_material_k_np[shear_slot] = ke_shear
                    joint_material_k_np[bend_slot] = ke_bend
                    joint_material_k_np[twist_slot] = ke_twist

                    joint_k_init_np[stretch_slot] = _penalty_k_init(ke_stretch, legacy_lin_k_start)
                    joint_k_init_np[shear_slot] = _penalty_k_init(ke_shear, legacy_lin_k_start)
                    joint_k_init_np[bend_slot] = _penalty_k_init(ke_bend, legacy_ang_k_start)
                    joint_k_init_np[twist_slot] = _penalty_k_init(ke_twist, legacy_ang_k_start)

                    joint_kd_np[stretch_slot] = jtarget_kd[stretch_dof]
                    joint_kd_np[shear_slot] = jtarget_kd[shear_dof]
                    joint_kd_np[bend_slot] = jtarget_kd[bend_dof]
                    joint_kd_np[twist_slot] = jtarget_kd[twist_dof]
                elif jt[j] == JointType.BALL:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0] = structural_linear_ke
                    joint_k_init_np[c0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0] = self.rigid_joint_linear_kd
                    is_hard_np[c0] = hard
                elif jt[j] == JointType.FIXED:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = hard
                    joint_material_k_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = hard
                elif jt[j] == JointType.REVOLUTE:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = hard
                    joint_material_k_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = hard
                    dof0 = int(jdofs[j])
                    dl_material_k = max(float(jtarget_ke[dof0]), float(jlimit_ke[dof0]))
                    dl_seed = _penalty_k_init(dl_material_k, legacy_ang_k_start)
                    joint_material_k_np[c0 + 2] = dl_material_k
                    joint_k_init_np[c0 + 2] = dl_seed
                    joint_kd_np[c0 + 2] = 0.0
                elif jt[j] == JointType.PRISMATIC:
                    c0 = int(jc_start[j])
                    hard = int(j_is_hard[j])
                    joint_material_k_np[c0 + 0] = structural_linear_ke
                    joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                    joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                    is_hard_np[c0 + 0] = hard
                    joint_material_k_np[c0 + 1] = structural_angular_ke
                    joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                    joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                    is_hard_np[c0 + 1] = hard
                    dof0 = int(jdofs[j])
                    dl_material_k = max(float(jtarget_ke[dof0]), float(jlimit_ke[dof0]))
                    dl_seed = _penalty_k_init(dl_material_k, legacy_lin_k_start)
                    joint_material_k_np[c0 + 2] = dl_material_k
                    joint_k_init_np[c0 + 2] = dl_seed
                    joint_kd_np[c0 + 2] = 0.0
                elif jt[j] == JointType.D6:
                    c0 = int(jc_start[j])
                    dof0 = int(jdofs[j])
                    lc = int(jdof_dim[j, 0])
                    ac = int(jdof_dim[j, 1])
                    hard = int(j_is_hard[j])
                    if lc < 3:
                        joint_material_k_np[c0 + 0] = structural_linear_ke
                        joint_k_init_np[c0 + 0] = _penalty_k_init(structural_linear_ke, legacy_lin_k_start)
                        joint_kd_np[c0 + 0] = self.rigid_joint_linear_kd
                        is_hard_np[c0 + 0] = hard
                    if ac < 3:
                        joint_material_k_np[c0 + 1] = structural_angular_ke
                        joint_k_init_np[c0 + 1] = _penalty_k_init(structural_angular_ke, legacy_ang_k_start)
                        joint_kd_np[c0 + 1] = self.rigid_joint_angular_kd
                        is_hard_np[c0 + 1] = hard
                    for li in range(lc):
                        dof_idx = dof0 + li
                        slot = c0 + 2 + li
                        dl_material_k = max(float(jtarget_ke[dof_idx]), float(jlimit_ke[dof_idx]))
                        dl_seed = _penalty_k_init(dl_material_k, legacy_lin_k_start)
                        joint_material_k_np[slot] = dl_material_k
                        joint_k_init_np[slot] = dl_seed
                        joint_kd_np[slot] = 0.0
                    for ai in range(ac):
                        dof_idx = dof0 + lc + ai
                        slot = c0 + 2 + lc + ai
                        dl_material_k = max(float(jtarget_ke[dof_idx]), float(jlimit_ke[dof_idx]))
                        dl_seed = _penalty_k_init(dl_material_k, legacy_ang_k_start)
                        joint_material_k_np[slot] = dl_material_k
                        joint_k_init_np[slot] = dl_seed
                        joint_kd_np[slot] = 0.0
                else:
                    pass

            k = wp.array(joint_k_init_np, dtype=float, device=self.device)
            k_min = wp.array(joint_k_init_np.copy(), dtype=float, device=self.device)
            material_k = wp.array(joint_material_k_np, dtype=float, device=self.device)
            rho = wp.array(joint_rho_np, dtype=float, device=self.device)
            kd = wp.array(joint_kd_np, dtype=float, device=self.device)
            is_hard = wp.array(is_hard_np, dtype=wp.int32, device=self.device)
            return k, k_min, material_k, rho, kd, is_hard

    def _init_structural_k(self) -> None:
        """Initialize the per-body structural stiffness summary from joint state.

        ``body_structural_k[b]`` is the max enabled linear-joint stiffness
        anchored on body ``b`` (rod joints use ``max(stretch, shear)``). Contact
        conditioning augments each dynamic endpoint's inertial scale with its
        own summary before combining endpoints.
        Direction- and chain-blind by design: it bounds neighborhood stiffness to
        condition rho and never enters a force law.
        Topology and the non-rod structural constants are construction-time state. The summary
        is refreshed in place after a notified joint-enable change, and after a
        ``JOINT_DOF_PROPERTIES`` change to rod stretch/shear stiffness, which feeds it.
        """
        self.body_structural_k = wp.empty(self.model.body_count, dtype=float, device=self.device)
        self._refresh_structural_k()

    def _refresh_structural_k(self) -> None:
        """Refresh the structural summary in place, without reallocating it.

        Depends on joint-enable flags and on rod stretch/shear ``joint_material_k``, so it is
        rerun for both ``JOINT_PROPERTIES`` and ``JOINT_DOF_PROPERTIES``.
        """
        self.body_structural_k.zero_()
        if self.model.joint_count == 0:
            return
        wp.launch(
            kernel=refresh_body_structural_k,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_type,
                self.model.joint_enabled,
                self.model.joint_parent,
                self.model.joint_child,
                self.joint_constraint_start,
                self.joint_constraint_dim,
                self.joint_material_k,
            ],
            outputs=[self.body_structural_k],
            device=self.device,
        )

    def _refresh_joint_material_params(self) -> None:
        """Refresh ``joint_target_ke``/``joint_target_kd``-derived material stiffness and damping.

        Stiffness covers ROD and the drive/limit slot(s) of REVOLUTE, PRISMATIC, and D6; its
        penalty and floor are reseeded alongside. Only legacy AVBD reads those non-ROD slots --
        compliant ALM gathers the same coefficients live from the model each solve. Not covered:
        BALL, FIXED, and REVOLUTE/PRISMATIC/D6's structural slots, which come from the
        solver-wide ``rigid_joint_linear_ke``/``rigid_joint_angular_ke`` constants.

        Damping is ROD-only -- other joint types read drive damping live from
        ``joint_target_kd`` at step time rather than caching it here.

        Stiffness slots are reseeded only where the effective stiffness changed, so untouched
        slots keep any legacy AVBD ramp they have accumulated.
        """
        if self.model.joint_count == 0:
            return
        wp.launch(
            kernel=refresh_joint_material_params,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_type,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.joint_constraint_start,
                self.model.joint_target_ke,
                self.model.joint_target_kd,
                self.model.joint_limit_ke,
                self.rigid_joint_linear_k_start if self.rigid_joint_linear_k_start is not None else -1.0,
                self.rigid_joint_angular_k_start if self.rigid_joint_angular_k_start is not None else -1.0,
            ],
            outputs=[
                self.joint_material_k,
                self.joint_penalty_k,
                self.joint_penalty_k_min,
                self.joint_penalty_kd,
            ],
            device=self.device,
        )

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
        """Register SolverVBD custom Model attributes.

        Currently registers:
          - ``vbd:joint_is_hard`` for per-joint hard/soft constraint mode (non-rod joints)
          - ``vbd:dahl_eps_max`` and ``vbd:dahl_tau`` for optional rod angular Dahl friction

        Attributes are declared in the ``vbd`` namespace so they can be authored
        in scenes and in USD as ``newton:vbd:<attr>``.

        Dahl rod friction is enabled per joint only where both
        ``model.vbd.dahl_eps_max`` and ``model.vbd.dahl_tau`` are authored
        positive; the attributes default to zero.

        Args:
            builder: Model builder to register attributes on.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_eps_max",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="vbd",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dahl_tau",
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
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
        if self.model.soft_mesh_adjacency is None:
            raise ValueError("model.soft_mesh_adjacency is missing; finalize the model with ModelBuilder.")
        return self.model.soft_mesh_adjacency.init_vertex_adjacency(self.model.particle_count)

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
        rows. For the same reason, do not change a body's solvability (mass or
        kinematic flag) while update is disabled: the per-body lists depend on
        effective inverse mass and are not rebuilt until the next refresh.

        Joint constraint maintenance (C0 snapshot, lambda retention/decay, and
        automatic rho refresh) runs every step regardless of this flag via
        step_joint_C0_lambda_rho().
        Rigid contact history snapshotting also runs every step when enabled.

        This setting applies only to the next call to :meth:`step` and is then
        reset to True.  Useful for substepping where collision detection frequency
        differs from the simulation step frequency.

        Args:
            update: If True, update rigid solver state. If False, reuse previous.
        """
        self._update_rigid_history = update

    # TODO: Complete the Newton 1.6 deprecation by removing this compatibility
    # method with the legacy non-ALM path.
    def set_joint_constraint_mode(self, joint_index: int, hard: bool, slot: int | None = None):
        """Set legacy hard/soft mode for a joint's structural slots at runtime.

        .. deprecated:: 1.6
            Per-slot joint hard/soft mode is deprecated. Under compliant ALM (the
            future default) all structural slots use the unified scheme, so this
            has no solver-mode effect; it will be removed with the legacy path.

        Non-rod structural slots are LINEAR (slot 0) and ANGULAR (slot 1).
        Builder-created rod joints expose STRETCH (slot 0), SHEAR
        (slot 1), BEND (slot 2), and TWIST (slot 3). Other drive/limit slots
        are always soft and cannot be set to hard.

        By default, rod stretch, shear, bend, and twist slots are soft, while
        non-rod structural slots are hard.

        For non-rod joints, hard/soft mode can also be authored per joint at
        build time via the ``vbd:joint_is_hard`` custom attribute, avoiding a
        runtime :meth:`set_joint_constraint_mode` call::

            SolverVBD.register_custom_attributes(builder)  # before adding joints
            builder.add_joint_fixed(..., custom_attributes={"vbd:joint_is_hard": 0})
            model = builder.finalize()
            solver = SolverVBD(model, rigid_compliant_alm=False)

        Args:
            joint_index: Index of the joint to modify.
            hard: In legacy mode, True selects hard AL mode and False selects
                soft penalty mode. Has no solver-mode effect under compliant ALM.
            slot: Specific slot index to set. If None, sets all structural slots.
                Use JointSlot.LINEAR / JointSlot.ANGULAR for non-rod joints,
                or JointSlot.STRETCH / JointSlot.SHEAR / JointSlot.BEND /
                JointSlot.TWIST for rod joints.

        Raises:
            ValueError: If the joint index is out of range or the slot is not a
                structural slot for this joint.
        """
        if not self._joint_mode_deprecation_warned:
            warnings.warn(
                "SolverVBD.set_joint_constraint_mode (per-slot joint hard/soft mode) is deprecated as of "
                "Newton 1.6 and will be removed with the legacy path. Under compliant ALM it has no "
                "solver-mode effect; legacy AVBD still honors it during the migration window.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._joint_mode_deprecation_warned = True
        n_j = self.model.joint_count
        if joint_index < 0 or joint_index >= n_j:
            raise ValueError(f"joint_index={joint_index} out of range [0, {n_j}).")

        with wp.ScopedDevice("cpu"):
            c_start_np = self._to_numpy(self.joint_constraint_start, dtype=np.int32)
            c_dim_np = self._to_numpy(self.joint_constraint_dim, dtype=np.int32)
            is_hard_np = self._to_numpy(self.joint_is_hard, dtype=np.int32)
            joint_type_np = self._to_numpy(self.model.joint_type, dtype=np.int32)

            c0 = int(c_start_np[joint_index])
            cdim = int(c_dim_np[joint_index])
            joint_type = int(joint_type_np[joint_index])
            structural_count = cdim if joint_type == int(JointType.ROD) else min(cdim, 2)
            val = 1 if hard else 0

            if slot is not None:
                if slot < 0 or slot >= structural_count:
                    if structural_count == 0:
                        names = "no structural slots"
                    elif joint_type == int(JointType.ROD):
                        names = "STRETCH=0, SHEAR=1, BEND=2, TWIST=3"
                    elif structural_count == 1:
                        names = "LINEAR=0"
                    else:
                        names = "LINEAR=0, ANGULAR=1"
                    raise ValueError(
                        f"Cannot set hard mode on slot={slot}: this joint has "
                        f"{structural_count} structural slot(s) ({names})."
                    )
                is_hard_np[c0 + slot] = val
            else:
                for s in range(structural_count):
                    is_hard_np[c0 + s] = val

            # Mutate in place: a rebuilt wp.array would orphan pointers captured
            # in existing CUDA graphs, silently ignoring the mode change on replay.
            self.joint_is_hard.assign(is_hard_np)

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
        2. Iterate: Interleave particle and rigid-body VBD iterations
        3. Finalize: Update velocities and persistent state (Dahl friction)

        To control rigid body substepping behavior, call set_rigid_history_update().
        When True (default), the step rebuilds rigid contact lists, re-initializes
        rigid contact state (penalty_k, lambda, C0), and restores from history if enabled.
        When False, reuses previous rigid contact state. The flag is reset to True when consumed.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Control inputs.
            contacts: Contact data produced by :meth:`~newton.CollisionPipeline.collide` (rigid-rigid and
                rigid-particle contacts), allocated with :meth:`~newton.CollisionPipeline.contacts`.
                If None, rigid contact handling is skipped. Note that particle self-contact (if enabled) does not
                depend on this argument.
            dt: Time step size.

        Raises:
            RuntimeError: If required rigid contact-matching data is unavailable, or contact-history storage would
                need to be allocated or grown during graph capture.
        """
        self._apply_module_options()
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
        self._finalize_rigid_bodies(state_in, state_out, dt)
        self._finalize_particles(state_out, dt)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset rigid solver history and optional body and particle state for selected worlds.

        Body fields selected by *flags* are copied from the model defaults.
        Joint penalty is restored to its minimum; joint C0 and AVBD dual history
        is zeroed immediately. Pose and enabled-rod friction history (curvature,
        stress, and increment) are rebaselined together from the next :meth:`step`
        input pose, after any intervening state edits or forward kinematics.
        Selected-world contact warm-start is cold-started when fresh rigid contacts
        are next processed. Internal rigid history is reset regardless of *flags*.
        When an external solver integrates the bodies, reset performs no rigid
        mutation; ``state`` and ``world_mask`` validation and particle reset still
        apply, but body State arrays are not accessed or validated.

        ``BODY_Q`` / ``BODY_QD`` copy ``model.body_q`` / ``model.body_qd`` into
        *state*; they do not restore a previously supplied state. A requested field
        is skipped if its *state* array is ``None``. If your initial pose differs
        from the model defaults, pass ``flags=0`` and author the pose any time
        before the next step; reset then preserves it and only clears VBD history.
        ``JOINT_Q`` / ``JOINT_QD`` are ignored (VBD uses maximal ``body_q`` /
        ``body_qd``); to reset from joint coordinates, run :func:`~newton.eval_fk`
        after reset so the resulting ``body_q`` supersedes reset's model copy.

        ``PARTICLE_Q`` / ``PARTICLE_QD`` likewise copy ``model.particle_q`` /
        ``model.particle_qd`` into *state* for particles in the selected worlds,
        using the same masking as the body fields (``world_mask=None`` also
        restores global ``world == -1`` particles; an explicit mask restores
        globals only through its final entry). One path covers both cloth and
        volumetric (tet) soft bodies, and it runs even when an external solver
        integrates the bodies or the model has none. A requested particle field is
        skipped if its *state* array is ``None``. Particle and body-particle solver
        history is intentionally left untouched: ``particle_q_prev`` is rebaselined
        from the incoming state at the start of the next :meth:`step`, self-contact
        and body-particle contacts rebuild per step, and tet/cloth elasticity is
        stateless, so no particle history cold-start is required. Reset does not
        refresh the particle self-contact BVH; the next :meth:`step` refits it from
        the incoming positions. After a large reset displacement, call
        :meth:`rebuild_bvh` to restore acceleration-structure quality. Both reset
        and :meth:`rebuild_bvh` are graph-capturable, so either may run inside a
        captured episode-reset graph.

        Reset does not run collision detection, and :meth:`step` consumes the
        supplied contacts rather than rerunning
        :meth:`~newton.CollisionPipeline.collide`. After moving bodies or
        particles, regenerate contacts so stale soft contacts are not reused, and
        let the next :meth:`step` refresh rigid contact state. The next
        rigid :meth:`step` consumes the pose and rod rebaseline even when
        ``contacts=None``, so author the final pose (or run :func:`~newton.eval_fk`)
        before stepping; contact invalidation instead waits for a fresh refresh.
        VBD cold-starts its numeric contact state for reset-selected worlds.
        Frame-to-frame correspondence and sticky contact geometry remain owned by
        :class:`~newton.CollisionPipeline`; after a discontinuous episode reset,
        call :meth:`~newton.CollisionPipeline.reset_contact_matching` with the
        same world mask to discard that history. Reset does not change
        ``set_rigid_history_update()``; leave rigid history refresh enabled for the
        next contact-bearing step. Reusing contacts
        (``set_rigid_history_update(False)``) is unsupported only while contact
        invalidation is still pending.

        Args:
            state: The simulation state to reset (modified in place).
            world_mask: One-dimensional Warp boolean mask on the solver device.
                Shape ``(world_count + 1,)``, with the final entry selecting
                entities in global world ``-1``. ``None`` selects all local and
                global entities.

                .. deprecated:: 1.5
                    Passing a mask with shape ``(world_count,)`` is deprecated.
                    Use shape ``(world_count + 1,)`` with a final ``False`` entry
                    to select local worlds only.
            flags: :class:`~newton.StateFlags` (or ``int``) selecting which body
                and particle fields to copy from the model defaults. VBD honors
                :attr:`~newton.StateFlags.BODY_Q`,
                :attr:`~newton.StateFlags.BODY_QD`,
                :attr:`~newton.StateFlags.PARTICLE_Q`, and
                :attr:`~newton.StateFlags.PARTICLE_QD`; ``None`` requests all flags.
        """
        if state is None:
            raise ValueError("'state' argument is required.")
        model = self.model
        world_mask = self._normalize_reset_world_mask(world_mask)

        flags_value = int(StateFlags.ALL if flags is None else flags)

        # Only requested BODY flags reach the launch as actionable arrays; everything
        # else stays None so an unrequested (possibly wrong-device) State array never
        # binds, and a supplied array is itself the kernel's reset signal.
        body_q = None
        body_qd = None
        if self._integrates_rigid_bodies:
            if flags_value & int(StateFlags.BODY_Q) and state.body_q is not None:
                if state.body_q.device != self.device:
                    raise ValueError(
                        f"state.body_q is on device {state.body_q.device}, expected solver device {self.device}."
                    )
                body_q = state.body_q
            if flags_value & int(StateFlags.BODY_QD) and state.body_qd is not None:
                if state.body_qd.device != self.device:
                    raise ValueError(
                        f"state.body_qd is on device {state.body_qd.device}, expected solver device {self.device}."
                    )
                body_qd = state.body_qd

        # Particle state reset mirrors the rigid path: only a requested, on-device
        # State array binds, and a bound array is itself the kernel's per-field
        # reset signal. This runs before the rigid early-return so particle-only
        # models and external-rigid coupling still restore deformables.
        if model.particle_count > 0:
            particle_q = None
            particle_qd = None
            if flags_value & int(StateFlags.PARTICLE_Q) and state.particle_q is not None:
                if state.particle_q.device != self.device:
                    raise ValueError(
                        f"state.particle_q is on device {state.particle_q.device}, "
                        f"expected solver device {self.device}."
                    )
                particle_q = state.particle_q
            if flags_value & int(StateFlags.PARTICLE_QD) and state.particle_qd is not None:
                if state.particle_qd.device != self.device:
                    raise ValueError(
                        f"state.particle_qd is on device {state.particle_qd.device}, "
                        f"expected solver device {self.device}."
                    )
                particle_qd = state.particle_qd
            if particle_q is not None or particle_qd is not None:
                wp.launch(
                    kernel=reset_particle_state,
                    dim=model.particle_count,
                    inputs=[
                        world_mask,
                        world_mask is None,
                        model.world_count,
                        model.particle_world,
                        model.particle_q,
                        model.particle_qd,
                    ],
                    outputs=[particle_q, particle_qd],
                    device=self.device,
                )

        if not self._integrates_rigid_bodies:
            return

        # Entity-parallel launch over the widest lane: world slots (+1 global),
        # joints, and bodies (bodies only when copying model-default body state).
        reset_dim = max(
            model.world_count + 1,
            model.joint_count,
            model.body_count if (body_q is not None or body_qd is not None) else 0,
        )
        wp.launch(
            kernel=reset_rigid_state,
            dim=reset_dim,
            inputs=[
                world_mask,
                world_mask is None,
                model.world_count,
                model.body_world,
                model.joint_world,
                self.joint_constraint_start,
                self.joint_constraint_dim,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.body_q,
                model.body_qd,
                self.joint_penalty_k_min,
            ],
            outputs=[
                body_q,
                body_qd,
                self.joint_penalty_k,
                self.joint_C0_lin,
                self.joint_C0_ang,
                self.joint_lambda_lin,
                self.joint_lambda_ang,
                self.joint_drive_lambda,
                self.joint_limit_lambda,
                self._rigid_pose_rebaseline_mask,
                self._contact_history_reset_mask,
                self._contact_history_reset_pending,
            ],
            device=self.device,
        )

    def _snapshot_rigid_contact_history(self, contacts: Contacts | None):
        """Write solved contact state for next frame's match-index warm-start."""
        if not self.rigid_contact_history or contacts is None:
            return

        if not self._integrates_rigid_bodies:
            return

        contact_launch_dim = contacts.rigid_contact_max
        if contact_launch_dim == 0:
            return

        if self._prev_contact_lambda is None or self._prev_contact_lambda.shape[0] < contact_launch_dim:
            self._init_rigid_contact_warmstart(contact_launch_dim)

        # Snapshot solved contact rows for the next step's warm-start.
        wp.launch(
            kernel=snapshot_body_body_contact_history,
            dim=contact_launch_dim,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_normal,
                self.body_body_contact_lambda,
                self.body_body_contact_penalty_k,
            ],
            outputs=[
                self._prev_contact_lambda,
                self._prev_contact_penalty_k,
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
                model.particle_world,
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
        rigid_capacity = contacts.rigid_contact_max if contacts is not None else 0
        soft_capacity = contacts.soft_contact_max if contacts is not None else 0
        body_undersized = self._integrates_rigid_bodies and self.body_body_contact_penalty_k.shape[0] < rigid_capacity
        particle_undersized = model.particle_count > 0 and self.body_particle_contact_penalty_k.shape[0] < soft_capacity

        # Rigid contact history is cross-replay-persistent state: allocating it
        # during capture records a `wp.zeros` fill into the graph, which then
        # re-zeros the warm-start buffers on every replay -- silently
        # equivalent to `rigid_contact_history=False`. So this guard fires
        # unconditionally when capturing, regardless of the device's
        # allocation-during-capture support.
        if self.device.is_capturing and self._integrates_rigid_bodies and self.rigid_contact_history:
            history_capacity = 0 if self._prev_contact_lambda is None else self._prev_contact_lambda.shape[0]
            if history_capacity < rigid_capacity:
                raise RuntimeError(
                    "SolverVBD contact history must be allocated before graph capture. "
                    "Construct CollisionPipeline before SolverVBD, or run one uncaptured solver step before capture."
                )

        refresh = refresh or body_undersized

        # ---------------------------
        # Rigid-only initialization
        # ---------------------------
        if self._integrates_rigid_bodies:
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
                            self.body_inv_mass_effective,
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
                    if self.rigid_contact_history and contact_launch_dim > 0:
                        if contacts.rigid_contact_match_index is None or contacts.contact_matching_mode not in (
                            "latest",
                            "sticky",
                        ):
                            raise RuntimeError(
                                "SolverVBD(rigid_contact_history=True) requires Contacts with "
                                "valid contact-matching provenance. Use "
                                'CollisionPipeline(contact_matching="latest") or '
                                'CollisionPipeline(contact_matching="sticky"), or set rigid_contact_history=False. '
                                f"Got contact_matching_mode={contacts.contact_matching_mode!r}."
                            )

                        history_required = contact_launch_dim
                        if self._prev_contact_lambda is None or self._prev_contact_lambda.shape[0] < history_required:
                            history_cap = 0 if self._prev_contact_lambda is None else self._prev_contact_lambda.shape[0]
                            self._raise_if_capturing_resize("rigid contact history", history_cap, history_required)
                            self._init_rigid_contact_warmstart(history_required)

                        history = RigidContactHistory()
                        history.lambda_ = self._prev_contact_lambda
                        history.penalty_k = self._prev_contact_penalty_k
                        history.normal = self._prev_contact_normal
                        restore_compliant_tangent_warmstart = int(
                            self.rigid_compliant_alm and contacts.contact_matching_mode == "latest"
                        )

                        wp.launch(
                            kernel=init_body_body_contacts_alm,
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
                                self.rigid_compliant_alm,
                                restore_compliant_tangent_warmstart,
                                contacts.rigid_contact_match_index,
                                history,
                                self._contact_history_reset_pending,
                                self._contact_history_reset_mask,
                                model.shape_world,
                                model.shape_body,
                                model.body_world,
                                self.rigid_contact_k_start_value,
                            ],
                            outputs=[
                                self.body_body_contact_penalty_k,
                                self.body_body_contact_lambda,
                                self.body_body_contact_material_kd,
                                self.body_body_contact_material_mu,
                                self.body_body_contact_material_ke,
                            ],
                            device=self.device,
                        )
                    elif contact_launch_dim > 0:
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

                    # A fresh refresh supersedes the prior contact rows, so consume the
                    # pending reset (contact-reset state exists only with history on).
                    if self.rigid_contact_history and contact_launch_dim > 0:
                        self._contact_history_reset_mask.zero_()
                        self._contact_history_reset_pending.zero_()

            # Per-step penalty decay, lambda retention, C0, and ALM auto-rho
            # (body_q is still collide frame here).
            if contacts is not None and contacts.rigid_contact_max > 0:
                contact_launch_dim = contacts.rigid_contact_max
                contact_lambda_retention = _rigid_lambda_retention(
                    self.rigid_contact_alpha, self.rigid_avbd_gamma, self.rigid_compliant_alm
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
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        model.shape_body,
                        model.body_flags,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        model.body_com,
                        self.body_structural_k,
                        int(BodyFlags.PROXY),
                        state_in.body_q,
                        self.rigid_contact_hard,
                        self.rigid_compliant_alm,
                        contact_lambda_retention,
                        1.0 / (dt * dt),
                        self.rigid_avbd_gamma,
                        self.body_body_contact_material_ke,
                        self.rigid_contact_k_start_value,
                    ],
                    outputs=[
                        self.body_body_contact_normal_rho,
                        self.body_body_contact_penalty_k,
                        self.body_body_contact_C0,
                        self.body_body_contact_lambda,
                        self.body_body_contact_tangent_rho,
                    ],
                    device=self.device,
                )

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
                        dt,
                    ],
                    outputs=[
                        body_f_for_integration,
                        None,  # joint_impulse: VBD does not populate body_parent_f
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
                    self._rigid_pose_rebaseline_mask,
                    body_f_for_integration,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    state_in.body_q,  # input/output
                    state_in.body_qd,  # input/output
                ],
                outputs=[
                    self.body_q_prev,  # rebaselined for flagged worlds (first step / reset)
                    self.body_inertia_q,
                ],
                dim=model.body_count,
                device=self.device,
            )

            if model.joint_count > 0:
                # Per-step joint setup: penalty-k decay, C0 snapshot, lambda retention,
                # and automatic per-row rho refresh for compliant ALM.
                joint_lambda_retention = _rigid_lambda_retention(
                    self.rigid_joint_alpha, self.rigid_avbd_gamma, self.rigid_compliant_alm
                )
                wp.launch(
                    kernel=step_joint_C0_lambda_rho,
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
                        model.joint_dof_dim,
                        self.joint_rod_rest_kb_local,
                        self.joint_rod_rest_twist,
                        self.body_q_prev,
                        model.body_q,
                        self.joint_constraint_start,
                        self.joint_constraint_dim,
                        self.joint_is_hard,
                        self.rigid_compliant_alm,
                        joint_lambda_retention,
                        self.rigid_avbd_gamma,
                        self.joint_penalty_k_min,
                        self.joint_material_k,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_limit_ke,
                        1.0 / (dt * dt),
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                    ],
                    outputs=[
                        self.joint_penalty_k,
                        self.joint_C0_lin,
                        self.joint_C0_ang,
                        self.joint_lambda_lin,
                        self.joint_lambda_ang,
                        self.joint_rho,
                        self.joint_drive_limit_support,
                        self.joint_drive_lambda,
                        self.joint_limit_lambda,
                    ],
                    device=self.device,
                )

            # Compute rod bend/twist Dahl hysteresis parameters once per timestep.
            if self.enable_dahl_friction and model.joint_count > 0:
                wp.launch(
                    kernel=compute_rod_dahl_parameters,
                    inputs=[
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_world,
                        self._rigid_pose_rebaseline_mask,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.joint_constraint_start,
                        self.joint_penalty_k,
                        self.joint_material_k,
                        self.joint_is_hard,
                        self.rigid_compliant_alm,
                        self.joint_rod_rest_kb_local,
                        self.joint_rod_rest_twist,
                        self.body_q_prev,
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

            # The forward step and any enabled rod update have consumed the mask.
            self._rigid_pose_rebaseline_mask.zero_()

        # ---------------------------
        # Body-particle interaction
        # ---------------------------
        particle_refresh = refresh or particle_undersized

        if model.particle_count > 0 and particle_refresh and contacts is not None:
            # Build body-particle contact lists (only when SolverVBD integrates bodies).
            if self._integrates_rigid_bodies:
                self.body_particle_contact_counts.zero_()
                self.body_particle_contact_overflow_max.zero_()
                wp.launch(
                    kernel=build_body_particle_contact_lists,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_shape,
                        model.shape_body,
                        self.body_inv_mass_effective,
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
                        contacts.soft_contact_indices,
                        contacts.soft_contact_count,
                        contacts.soft_contact_max,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_ke,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        model.shape_body,
                        body_q_for_particles,
                        body_q_prev_for_particles,
                        body_qd_for_particles,
                        model.body_com,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        model.shape_margin,
                        contacts.soft_contact_barycentric,
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
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ):
        """Solve one rigid-body VBD iteration (per-iteration phase).

        Accumulates contact and joint forces/hessians, solves 6x6 rigid body systems per color,
        and updates AVBD penalty parameters (dual update).
        """
        model = self.model
        # Body-particle soft contacts still need penalty updates when VBD skips rigid solves:
        # external rigid mode uses state_out.body_q, while static-shape contacts use _empty_body_q.
        skip_rigid_solve = not self._integrates_rigid_bodies
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
                        contacts.soft_contact_indices,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_barycentric,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        model.shape_margin,
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
                        state_in.body_qd,
                        model.body_com,
                        self.body_inv_mass_effective,
                        model.shape_body,
                        self.friction_epsilon,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_ke,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        contacts.soft_contact_count,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_barycentric,
                        model.shape_margin,
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
                        self.body_body_contact_normal_rho,
                        self.body_body_contact_material_ke,
                        self.body_body_contact_material_kd,
                        self.body_body_contact_material_mu,
                        self.body_body_contact_tangent_rho,
                        self.body_body_contact_lambda,
                        self.body_body_contact_C0,
                        self.rigid_contact_alpha,
                        self.rigid_contact_hard,
                        self.rigid_compliant_alm,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
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
                    self.joint_rod_rest_kb_local,
                    self.joint_rod_rest_twist,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    self.joint_constraint_start,
                    self.joint_penalty_k,
                    self.joint_rho,
                    self.joint_material_k,
                    self.joint_penalty_kd,
                    self.joint_sigma_start,
                    self.joint_C_fric,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    self.joint_drive_limit_support,
                    self.joint_drive_lambda,
                    self.joint_limit_lambda,
                    self.joint_lambda_lin,
                    self.joint_lambda_ang,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    self.rigid_compliant_alm,
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

        if contacts is not None and contacts.rigid_contact_max > 0:
            wp.launch(
                kernel=update_duals_body_body_contacts,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
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
                    self.rigid_compliant_alm,
                    self.body_body_contact_material_ke,
                    self.body_body_contact_tangent_rho,
                    self.body_body_contact_normal_rho,
                    self.rigid_linear_beta,
                    self.body_body_contact_penalty_k,  # input/output
                    self.body_body_contact_lambda,  # input/output
                ],
                device=self.device,
            )
        if contacts is not None and model.particle_count > 0:
            wp.launch(
                kernel=update_duals_body_particle_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_barycentric,
                    state_in.particle_q,
                    model.particle_radius,
                    model.shape_body,
                    model.shape_margin,
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
                    self.joint_rod_rest_kb_local,
                    self.joint_rod_rest_twist,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    self.joint_constraint_start,
                    state_in.body_q,
                    self.body_q_prev,
                    model.body_q,
                    model.joint_dof_dim,
                    self.joint_C0_lin,
                    self.joint_C0_ang,
                    self.joint_is_hard,
                    self.rigid_joint_alpha,
                    self.joint_material_k,
                    self.joint_rho,
                    self.rigid_compliant_alm,
                    self.rigid_linear_beta,
                    self.rigid_angular_beta,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    self.joint_rest_angle,
                    self.joint_drive_limit_support,
                    dt,
                    self.joint_penalty_k,  # input/output
                    self.joint_lambda_lin,  # input/output
                    self.joint_lambda_ang,  # input/output
                    self.joint_drive_lambda,  # input/output
                    self.joint_limit_lambda,  # input/output
                ],
                device=self.device,
            )

    def collect_rigid_contact_forces(
        self,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        contacts: Contacts | None,
        dt: float,
    ) -> tuple[
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.vec3],
        wp.array[wp.vec3],
        wp.array[wp.vec3],
        wp.array[wp.int32],
    ]:
        """Collect per-contact rigid contact forces and world-space application points.

        Args:
            body_q: Current body transforms (world frame),
                typically ``state_out.body_q`` after a ``step()`` call.
            body_q_prev: Effective previous-pose history used by the step (world frame).
                Snapshot ``solver.body_q_prev`` before :meth:`step` (it is advanced
                after the step). On a first or reset step, overwrite each rebaselined
                row with that step's input ``body_q`` so its reported force matches the
                solve. For externally integrated bodies, pass the external solver's
                previous transforms.
            contacts: Contact data buffers containing rigid
                contact geometry/material references. If None, the function
                returns default zero/sentinel outputs.
            dt: Time step size [s].

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
                getattr(self, "body_body_contact_normal_rho", None),
                getattr(self, "body_body_contact_material_ke", None),
                getattr(self, "body_body_contact_material_kd", None),
                getattr(self, "body_body_contact_material_mu", None),
                getattr(self, "body_body_contact_tangent_rho", None),
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
            self._raise_if_capturing_resize("rigid contact output", output_capacity, max_contacts)
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
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                body_q,
                body_q_prev,
                self.model.body_com,
                self.body_body_contact_penalty_k,
                self.body_body_contact_normal_rho,
                self.body_body_contact_material_ke,
                self.body_body_contact_material_kd,
                self.body_body_contact_material_mu,
                self.body_body_contact_tangent_rho,
                self.body_body_contact_lambda,
                self.body_body_contact_C0,
                self.rigid_contact_alpha,
                self.rigid_contact_hard,
                self.rigid_compliant_alm,
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

    def _finalize_rigid_bodies(self, state_in: State, state_out: State, dt: float):
        """Finalize rigid body velocities and Dahl friction state after VBD iterations (post-iteration phase).

        Updates rigid body velocities using BDF1 and updates Dahl hysteresis state for rod bend/twist.
        Also transfers the final body poses from state_in to state_out.
        """
        model = self.model

        # Early exit if no rigid bodies or rigid bodies are driven by an external solver
        if not self._integrates_rigid_bodies:
            return

        wp.launch(
            kernel=update_body_velocity,
            inputs=[
                dt,
                state_in.body_q,
                model.body_com,
            ],
            outputs=[self.body_q_prev, state_out.body_qd, state_in.body_qd, state_out.body_q],
            dim=model.body_count,
            device=self.device,
        )

        if self.enable_dahl_friction and model.joint_count > 0:
            wp.launch(
                kernel=update_rod_dahl_state,
                inputs=[
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    self.joint_constraint_start,
                    self.joint_penalty_k,
                    self.joint_material_k,
                    self.joint_is_hard,
                    self.rigid_compliant_alm,
                    self.joint_rod_rest_kb_local,
                    self.joint_rod_rest_twist,
                    state_out.body_q,
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
            state:  The state whose particle positions (:attr:`~newton.State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.particle_enable_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)
