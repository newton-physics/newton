# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import re
import warnings
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import numpy as np
import warp as wp

from ...core.reset import reset_world_selected
from ...core.types import override
from ...geometry.flags import ShapeFlags
from ...sim import BodyFlags, Contacts, Control, JointType, Model, ModelBuilder, ModelFlags, State, StateFlags
from ...sim.articulation import eval_fk
from ..solver import SolverBase
from .friction import FRICTION_PAIR_CUDA
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    _compute_body_net_wrench,
    accumulate_group_diag_worlds,
    allocate_joint_velocity_limit_slots,
    allocate_rigid_velocity_limit_slots,
    allocate_world_contact_slots,
    apply_augmented_mass_diagonal_grouped,
    apply_free_root_transport_to_predictor,
    apply_free_root_velocity_corrections,
    build_mass_update_mask,
    build_mf_contact_rows,
    cholesky_loop,
    compute_com_transforms,
    compute_composite_inertia,
    compute_contact_linear_force_from_impulses,
    compute_mf_body_Hinv,
    compute_mf_effective_mass_and_rhs,
    compute_mf_world_dof_offsets,
    compute_spatial_inertia,
    compute_velocity_predictor,
    compute_world_contact_bias,
    crba_fill_par_dof,
    diag_from_JY_par_art,
    diag_from_JY_world,
    eval_rigid_fk_id,
    eval_rigid_tau,
    eval_rigid_tau_and_augmented_drives,
    finalize_mf_constraint_counts,
    finalize_world_constraint_counts,
    finalize_world_diag_cfm,
    gather_JY_to_world,
    gather_tau_to_groups,
    hinv_jt_par_row,
    integrate_generalized_joints,
    pack_contact_linear_force_as_spatial,
    populate_joint_velocity_limit_J_for_size,
    populate_rigid_velocity_limit_rows,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_world_contact_rows,
    prepare_world_impulses,
    prescale_joint_velocity_limits,
    refresh_masked_body_inertia,
    remove_free_root_transport_from_qdd,
    scatter_qdd_from_groups,
    trisolve_loop,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
)

_SMALL_DOF_THRESHOLD_DEFAULT = 12
_MFGS_RESIDENT_METADATA_MAX_BYTES = 4096
_MFGS_TILE_SHARED_STORAGE_BYTES = 128
# Threads per block of the tiled Cholesky, triangular-solve and H^-1 J^T kernels.
_TILE_THREADS = 64
# Block size of the serial one-thread-per-articulation kernels.
_SERIAL_KERNEL_BLOCK_DIM = 256
# PhysX's default rigid-body angular damping [1/s].
_DEFAULT_ANGULAR_DAMPING = 0.05
# Per-world preset of the first rejected row slot: no reservation was rejected.
_ROW_SLOT_UNBOUNDED = 2**31 - 1
_CONTACT_BUILD_THREAD_CAP = 65536
_CONTACT_JACOBIAN_WORKER_CAP = 4096
_CONTACT_JACOBIAN_MAX_DOF = 10
_JOINT_LIMIT_WARPS_PER_BLOCK = 4
_SUPPORTED_JOINT_TYPES = (
    int(JointType.PRISMATIC),
    int(JointType.REVOLUTE),
    int(JointType.BALL),
    int(JointType.FIXED),
    int(JointType.FREE),
    int(JointType.DISTANCE),
    int(JointType.D6),
)


def _validate_supported_model(model: Model) -> None:
    """Reject model features this solver does not simulate instead of ignoring them."""
    if model.particle_count:
        raise NotImplementedError("SolverFeatherPGS does not simulate particles.")
    if model.joint_count:
        joint_type = model.joint_type.numpy()
        unsupported = sorted({int(t) for t in joint_type if int(t) not in _SUPPORTED_JOINT_TYPES})
        if unsupported:
            names = ", ".join(JointType(t).name for t in unsupported)
            raise NotImplementedError(f"SolverFeatherPGS does not support {names} joints.")
        if model.joint_enabled is not None and not np.all(model.joint_enabled.numpy()):
            raise NotImplementedError("SolverFeatherPGS does not support disabled joints (Model.joint_enabled).")
        if model.joint_mimic_joint is not None and np.any(model.joint_mimic_joint.numpy() >= 0):
            raise NotImplementedError("SolverFeatherPGS does not support mimic joints yet.")
        if model.joint_articulation is not None and model.body_count:
            joint_articulation = model.joint_articulation.numpy()
            joint_child = model.joint_child.numpy()
            owned = np.zeros(model.body_count, dtype=bool)
            tree = joint_articulation >= 0
            owned[joint_child[tree & (joint_child >= 0)]] = True
            closure = (~tree) & (joint_child >= 0)
            if np.any(owned[joint_child[closure]]):
                raise NotImplementedError("SolverFeatherPGS does not support loop-closing joints yet.")
    if int(getattr(model, "constraint_mimic_count", 0)):
        raise NotImplementedError("SolverFeatherPGS does not support mimic constraints yet.")


@wp.kernel
def _clear_dense_row_state(slot_counter: wp.array[int], dropped: wp.array2d[int]):
    """Clear allocation state and row-loss counters together."""
    world = wp.tid()
    slot_counter[world] = 0
    for family in range(2):
        dropped[family, world] = 0


@wp.func
def _world_rows_lost(
    world: int,
    dense_count: wp.array[int],
    mf_count: wp.array[int],
    dropped: wp.array2d[int],
    dense_capacity: int,
    mf_capacity: int,
):
    return (
        dense_count[world] > dense_capacity
        or mf_count[world] > mf_capacity
        or dropped[0, world] > 0
        or dropped[1, world] > 0
    )


@wp.kernel
def _finalize_constraint_status(
    dense_count: wp.array[int],
    mf_count: wp.array[int],
    dropped: wp.array2d[int],
    cross_world_contacts: wp.array[int],
    contact_count: wp.array[int],
    reduction_overflow: wp.array[int],
    dense_capacity: int,
    mf_capacity: int,
    contact_capacity: int,
    has_responding_global: int,
    overflow: wp.array[wp.bool],
):
    """Latch invalid-world status until reset; counts come from the per-family finalize kernels.

    ``overflow`` has one entry per world plus a final entry for global (world ``-1``)
    articulations. Global articulations are solved in world 0's row storage, so a row
    loss in world 0 also flags the global entry while a responding global articulation
    exists.
    """
    slot = wp.tid()
    global_slot = overflow.shape[0] - 1
    lost = contact_count[0] > contact_capacity or reduction_overflow[0] != 0 or cross_world_contacts[slot] > 0
    if slot < global_slot:
        lost = lost or _world_rows_lost(slot, dense_count, mf_count, dropped, dense_capacity, mf_capacity)
    elif has_responding_global != 0:
        lost = lost or _world_rows_lost(0, dense_count, mf_count, dropped, dense_capacity, mf_capacity)
    if lost:
        overflow[slot] = True


@wp.kernel
def _reset_solver_status(
    world_mask: wp.array[wp.bool],
    world_count: int,
    articulation_world: wp.array[int],
    mass_update_requested: wp.array[int],
    overflow: wp.array[wp.bool],
):
    """Clear the status of selected worlds and request a mass refresh of their articulations.

    ``overflow`` and ``world_mask`` share the ``(world_count + 1,)`` layout. ``articulation_world``
    holds the model's articulation worlds, so global articulations (world ``-1``) are selected
    by the final mask entry.
    """
    tid = wp.tid()
    if tid < overflow.shape[0]:
        if not world_mask or world_mask[tid]:
            overflow[tid] = False
    if tid < mass_update_requested.shape[0]:
        if not world_mask or reset_world_selected(articulation_world[tid], world_mask, world_count):
            mass_update_requested[tid] = 1


@wp.kernel
def _warn_constraint_row_overflow(
    dense_raw_counts: wp.array[wp.int32],
    dense_dropped_contact_rows: wp.array[wp.int32],
    dense_capacity: int,
    mf_raw_counts: wp.array[wp.int32],
    mf_dropped_contact_rows: wp.array[wp.int32],
    mf_capacity: int,
    mf_active: int,
    cross_world_contacts: wp.array[wp.int32],
    warning_emitted: wp.array[wp.int32],
):
    """Emit one device-side warning per overflowing FeatherPGS row family."""
    world = wp.tid()
    if cross_world_contacts[world] > 0 and wp.atomic_exch(warning_emitted, 2, 1) == 0:
        wp.printf(
            "Warning: FeatherPGS dropped %d contacts of status entry %d (the last entry is global bodies) "
            "that couple a global body with another world. Dynamic global bodies only interact with world 0.\n",
            cross_world_contacts[world],
            world,
        )
    if world >= dense_raw_counts.shape[0]:
        return

    dense_dropped = dense_dropped_contact_rows[world]
    dense_requested = dense_raw_counts[world]
    if dense_requested > dense_capacity and wp.atomic_exch(warning_emitted, 0, 1) == 0:
        wp.printf(
            "Warning: FeatherPGS dense constraint-row overflow in world %d: requested %d rows, limit %d; "
            "dropped %d contact/friction rows. Increase dense_max_constraints.\n",
            world,
            dense_requested,
            dense_capacity,
            dense_dropped,
        )

    if mf_active != 0:
        mf_dropped = mf_dropped_contact_rows[world]
        mf_requested = mf_raw_counts[world]
        if mf_requested > mf_capacity and wp.atomic_exch(warning_emitted, 1, 1) == 0:
            wp.printf(
                "Warning: FeatherPGS matrix-free constraint-row overflow in world %d: requested %d rows, "
                "limit %d; dropped %d contact/friction rows. Increase mf_max_constraints.\n",
                world,
                mf_requested,
                mf_capacity,
                mf_dropped,
            )


@wp.kernel
def compute_body_parent_f(
    body_to_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_fb_s: wp.array[wp.spatial_vector],
    body_ft_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_parent_f: wp.array[wp.spatial_vector],
):
    """Publish the inverse-dynamics wrench transmitted through each body's inbound joint.

    The backward pass leaves the net wrench of a body's subtree, referenced to the
    articulation origin, in ``body_fb_s + body_ft_s - f_ext``. Translate it to the body's
    center of mass in world frame (linear force first, torque second).
    """
    body = wp.tid()
    articulation = body_to_articulation[body]
    if articulation < 0:
        body_parent_f[body] = wp.spatial_vector()
        return
    origin = articulation_origin[articulation]
    f_s = _compute_body_net_wrench(body, body_ft_s[body], origin, body_fb_s, body_f_ext, body_flags, body_q, body_com)
    force = wp.spatial_top(f_s)
    com_rel = wp.transform_point(body_q[body], body_com[body]) - origin
    body_parent_f[body] = wp.spatial_vector(force, wp.spatial_bottom(f_s) - wp.cross(com_rel, force))


@dataclass(frozen=True)
class _FeatherPGSModelPlan:
    """Immutable articulation and generalized-response plan."""

    articulation_dof_count: np.ndarray
    articulation_dof_start: np.ndarray
    articulation_world: np.ndarray
    is_free_rigid: np.ndarray
    response_free_rigid_body_indices: np.ndarray
    prescribed_articulation: np.ndarray
    response_dof_count: np.ndarray
    world_count: int

    @classmethod
    def build(cls, model: Model, kinematic_dof_mask: np.ndarray) -> "_FeatherPGSModelPlan":
        """Build the immutable response layout from physical model topology.

        Global articulations (world ``-1``) are solved in world 0.
        """
        articulation_count = model.articulation_count
        articulation_dof_count = np.zeros(articulation_count, dtype=np.int32)
        articulation_dof_start = np.zeros(articulation_count, dtype=np.int32)
        articulation_world = np.zeros(articulation_count, dtype=np.int32)
        is_free_rigid = np.zeros(articulation_count, dtype=np.int32)
        free_rigid_body_indices: list[int] = []

        model_articulation_world = np.zeros(articulation_count, dtype=np.int32)
        if articulation_count:
            model_articulation_world = model.articulation_world.numpy().astype(np.int32, copy=True)
            articulation_world = model_articulation_world.copy()
            articulation_world[articulation_world < 0] = 0

        if articulation_count and model.joint_count:
            articulation_start = model.articulation_start.numpy()
            joint_parent = model.joint_parent.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            joint_type = model.joint_type.numpy()
            joint_child = model.joint_child.numpy()
            for art in range(articulation_count):
                first_joint = int(articulation_start[art])
                last_joint = int(articulation_start[art + 1])
                first_dof = int(joint_qd_start[first_joint])
                last_dof = int(joint_qd_start[last_joint])
                articulation_dof_start[art] = first_dof
                articulation_dof_count[art] = last_dof - first_dof
                if (
                    last_joint - first_joint == 1
                    and int(joint_type[first_joint]) == int(JointType.FREE)
                    and int(joint_parent[first_joint]) == -1
                ):
                    is_free_rigid[art] = 1
                    free_rigid_body_indices.append(int(joint_child[first_joint]))

        # A fully kinematic free body has no response of its own. Elide it unless it is
        # the only responding articulation of its world, which keeps the exact
        # large-armature kinematic fallback.
        prescribed = np.zeros(articulation_count, dtype=np.int32)
        if articulation_count and model.joint_count:
            candidates = np.zeros(articulation_count, dtype=np.int32)
            for art in range(articulation_count):
                if is_free_rigid[art] == 0 or articulation_dof_count[art] != 6:
                    continue
                dof_start = int(articulation_dof_start[art])
                dof_end = dof_start + int(articulation_dof_count[art])
                if np.all(kinematic_dof_mask[dof_start:dof_end] != 0):
                    candidates[art] = 1
            retained = (articulation_dof_count > 0) & (candidates == 0)
            for world in range(int(np.max(articulation_world)) + 1):
                in_world = articulation_world == world
                if np.any(in_world & retained):
                    prescribed[in_world & (candidates != 0)] = 1
            # A global kinematic body touches the bodies of every world, so it is elided
            # whenever any world keeps a response.
            if np.any(retained):
                prescribed[(model_articulation_world < 0) & (candidates != 0)] = 1

        response_dof_count = articulation_dof_count.copy()
        response_dof_count[prescribed != 0] = 0
        free_articulations = np.nonzero(is_free_rigid != 0)[0]
        free_bodies = np.asarray(free_rigid_body_indices, dtype=np.int32)
        response_free_rigid_body_indices = free_bodies[prescribed[free_articulations] == 0]
        world_count = max(int(model.world_count), 1)

        arrays = (
            articulation_dof_count,
            articulation_dof_start,
            articulation_world,
            is_free_rigid,
            response_free_rigid_body_indices,
            prescribed,
            response_dof_count,
        )
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays, world_count)


_DENSE_META_ROW_TYPE_BITS = 3
_DENSE_META_ROW_TYPE_MASK = (1 << _DENSE_META_ROW_TYPE_BITS) - 1
_DENSE_META_MAX_PARENT = ((2**31 - 1) >> _DENSE_META_ROW_TYPE_BITS) - 1


def _use_resident_mfgs_metadata(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    max_shared_memory: int,
) -> bool:
    """Select resident dense-row metadata for compact solver shapes.

    Resident row metadata removes repeated global loads across PGS iterations,
    but larger shapes lose more throughput from the resulting occupancy drop.
    Keep at most one 4-KiB metadata working set resident and stream larger sets.
    """
    metadata_bytes = 4 * max_constraints * 4
    base_shared_bytes = 4 * (max_world_dofs + max_constraints + mf_max_constraints)
    total_shared_bytes = _MFGS_TILE_SHARED_STORAGE_BYTES + base_shared_bytes + metadata_bytes
    return metadata_bytes <= _MFGS_RESIDENT_METADATA_MAX_BYTES and total_shared_bytes <= max_shared_memory


_HINV_JT_MAX_CHUNK_SIZE = 32


def _align_shared_memory(size: int) -> int:
    """Align a shared-memory allocation to Warp's 16-byte tile boundary."""
    return (size + 15) // 16 * 16


def _estimate_cholesky_shared_memory(n_dofs: int) -> int:
    """Estimate the complete tiled Cholesky shared-memory footprint [B]."""
    matrix_bytes = _align_shared_memory(4 * n_dofs * n_dofs)
    vector_bytes = _align_shared_memory(4 * n_dofs)
    return 3 * matrix_bytes + vector_bytes


def _estimate_hinv_jt_shared_memory(n_dofs: int, constraint_count: int, *, tile_threads: int) -> int:
    """Estimate the complete tiled H-inverse shared-memory footprint [B]."""
    footprint = _align_shared_memory(4 * n_dofs * n_dofs)
    footprint += 3 * _align_shared_memory(4 * n_dofs * constraint_count)
    return footprint + 4 * tile_threads


def _select_hinv_jt_chunk_size(
    n_dofs: int, max_constraints: int, max_shared_memory: int, tile_threads: int
) -> int | None:
    """Select a validated H-inverse chunk from articulation and device resources.

    Keep response tiles at 32 rows or fewer. Larger tiles waste triangular-solve
    work on the partially occupied tail common to per-world constraint systems;
    inactive follow-on chunks return before loading the factor.
    """
    if n_dofs <= 0 or max_constraints <= 0:
        return None
    candidates = (_HINV_JT_MAX_CHUNK_SIZE, 16, 8, 4, 2, 1)
    tried: set[int] = set()
    for candidate in candidates:
        chunk_size = min(candidate, max_constraints)
        if chunk_size in tried:
            continue
        tried.add(chunk_size)
        if _estimate_hinv_jt_shared_memory(n_dofs, chunk_size, tile_threads=tile_threads) <= max_shared_memory:
            return chunk_size
    return None


@dataclass(frozen=True)
class _FeatherPGSExecutionPlan:
    """Immutable device-resource choices for one solver shape."""

    cholesky_tiled_sizes: frozenset[int]
    hinv_jt_tiled_sizes: frozenset[int]
    hinv_jt_chunk_sizes: tuple[tuple[int, int], ...]

    @classmethod
    def build(
        cls,
        size_groups: list[int],
        *,
        max_constraints: int,
        max_shared_memory: int,
        cholesky_kernel: str,
        hinv_jt_kernel: str,
        small_dof_threshold: int,
        tile_threads: int,
    ) -> "_FeatherPGSExecutionPlan":
        """Resolve the factorization and H^-1 J^T implementations from solver shape and device limits."""
        cholesky_tiled_sizes: set[int] = set()
        tiled_sizes: set[int] = set()
        chunk_sizes: list[tuple[int, int]] = []
        for size in size_groups:
            cholesky_requested = cholesky_kernel == "tiled" or (
                cholesky_kernel == "auto" and size > small_dof_threshold
            )
            if cholesky_requested:
                required = _estimate_cholesky_shared_memory(size)
                if required <= max_shared_memory:
                    cholesky_tiled_sizes.add(size)
                elif cholesky_kernel == "tiled":
                    raise ValueError(
                        f"cholesky_kernel='tiled' requires {required} bytes of shared memory for DOF size {size}, "
                        f"but the device exposes {max_shared_memory} bytes; use cholesky_kernel='auto' or 'loop'."
                    )

            if max_constraints <= 0:
                continue
            requested = hinv_jt_kernel == "tiled" or (hinv_jt_kernel == "auto" and size > small_dof_threshold)
            if not requested:
                continue
            chunk_size = _select_hinv_jt_chunk_size(size, max_constraints, max_shared_memory, tile_threads)
            if chunk_size is None:
                if hinv_jt_kernel == "tiled":
                    required = _estimate_hinv_jt_shared_memory(size, 1, tile_threads=tile_threads)
                    raise ValueError(
                        f"hinv_jt_kernel='tiled' requires at least {required} "
                        f"bytes of shared memory for DOF size {size}, but the device exposes "
                        f"{max_shared_memory} bytes; use hinv_jt_kernel='auto' or 'par_row'."
                    )
                continue
            tiled_sizes.add(size)
            chunk_sizes.append((size, chunk_size))

        return cls(frozenset(cholesky_tiled_sizes), frozenset(tiled_sizes), tuple(chunk_sizes))

    def use_tiled_cholesky(self, size: int) -> bool:
        """Return whether an articulation group uses tiled Cholesky."""
        return size in self.cholesky_tiled_sizes

    def hinv_jt_chunk_size(self, size: int) -> int | None:
        """Return the tiled H-inverse chunk size for a response group."""
        for planned_size, chunk_size in self.hinv_jt_chunk_sizes:
            if planned_size == size:
                return chunk_size
        return None

    def use_tiled_hinv_jt(self, size: int) -> bool:
        """Return whether a response group uses tiled H-inverse application."""
        return size in self.hinv_jt_tiled_sizes


class SolverFeatherPGS(SolverBase):
    """Reduced-coordinate articulated dynamics with a projected Gauss-Seidel constraint solve.

    .. experimental::

        :class:`SolverFeatherPGS`, its constructor options and its solver-specific
        attributes may change without the normal deprecation period.

    Each step evaluates forward kinematics and inverse dynamics on the articulation
    tree, builds the joint-space mass matrix with the composite rigid body algorithm
    (CRBA) and factors it, and integrates an unconstrained velocity prediction
    (Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2014). Contacts, joint
    position limits and joint velocity limits then become constraint rows solved by
    projected Gauss-Seidel (PGS) in impulse space. The solve is *matrix-free*: for each
    row it keeps the response ``Y = H^-1 J^T`` and its diagonal ``J Y`` and recomputes
    ``J v`` every iteration instead of assembling a Delassus matrix. Rows of articulated
    bodies apply their impulse to the articulation immediately (one world-local velocity
    vector per world); contacts between free rigid bodies (a single body with a
    world-rooted free joint) and the ground or other free bodies use per-body inverse
    inertia. Positions are integrated from the solved velocities with symplectic Euler.

    Like :class:`~newton.solvers.SolverFeatherstone`, the solver uses
    :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` as its state and
    does not simulate bodies that are not connected through joints. Floating bases and
    free bodies need an explicit free joint, see :meth:`~newton.ModelBuilder.add_joint_free`
    (:meth:`~newton.ModelBuilder.add_body` adds one). Free-joint ``joint_qd`` is
    ``(v_com, omega)`` in world frame, the convention of :func:`newton.eval_fk`.

    Supported features:

    - Joints: PRISMATIC, REVOLUTE, BALL, FIXED, D6 and root FREE / DISTANCE joints.
    - Joint drives: :attr:`~newton.Model.joint_target_ke` and
      :attr:`~newton.Model.joint_target_kd` on PRISMATIC, REVOLUTE and D6 DOFs are
      integrated implicitly by folding ``dt * kd + dt^2 * ke`` into the mass matrix;
      the drive force is clamped to :attr:`~newton.Model.joint_effort_limit`.
      :attr:`~newton.Model.joint_armature` and :attr:`~newton.Model.joint_damping` are
      applied.
    - Joint limits: every finite :attr:`~newton.Model.joint_limit_lower` /
      :attr:`~newton.Model.joint_limit_upper` of a PRISMATIC, REVOLUTE or D6 DOF is a
      unilateral row. Joint velocity limits (:attr:`~newton.Model.joint_velocity_limit`)
      are enforced as rows when ``enable_joint_velocity_limits`` is set.
    - Contacts: rigid contacts from :class:`~newton.CollisionPipeline` with Coulomb
      point friction (one normal and two coupled tangent rows per contact, friction
      coefficient from the two shapes' ``mu``). Contact restitution, compliance and
      torsional friction are not applied.
    - Kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC`) and heterogeneous worlds.
    - CUDA graph capture of :meth:`step` and :meth:`reset`.

    Limitations:

    - CUDA only; constructing the solver on a CPU device raises :class:`NotImplementedError`.
    - Mimic joints, equality constraints and loop-closing joints raise
      :class:`NotImplementedError`. Particles are not simulated.
    - Gradients are not supported.

    Constraint rows are stored per world with fixed capacities (``dense_max_constraints``
    for rows of articulated bodies, ``mf_max_constraints`` for free-body contacts). Rows
    that do not fit are dropped and the world is flagged in :attr:`constraint_overflow`,
    a device boolean array with one entry per world plus a final entry for global
    (world ``-1``) articulations. It also records contact-buffer overflow and contacts
    dropped by global contact reduction. The flags persist until :meth:`reset`. Read them
    in a device kernel (for example an RL termination term) or call
    :meth:`check_constraint_capacity` outside graph capture.

    Global articulations are solved together with world 0. A kinematic global free body
    (for example a moving platform) contacts the bodies of every world.

    Known limitation: a contact between an articulated body of another world and a dynamic
    global body, or a kinematic global articulation with joints, cannot be solved, because
    it would couple two worlds' solves. Such a contact is dropped, counted, reported once by
    a device-side warning (unless ``warn_constraint_overflow`` is off) and flagged in
    :attr:`constraint_overflow`. The constructor warns once if the model's shape flags and
    collision groups allow such contacts. Use world geometry or a kinematic free body for
    scenery shared by all worlds, or add the body to every world.

    Extended state attributes:
        :attr:`~newton.State.body_parent_f` is populated when requested via
        :meth:`~newton.ModelBuilder.request_state_attributes`. As in
        :class:`~newton.solvers.SolverFeatherstone`, it is the per-body net spatial wrench
        of the inverse-dynamics backward pass at the start of the step, translated to the
        body's center of mass (linear force [N] first, torque [N·m] second, world frame).
        It does not include constraint or contact impulses of the step.

    Example:

        .. code-block:: python

            solver = newton.solvers.SolverFeatherPGS(model)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()

            for i in range(100):
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, dt)
                state_in, state_out = state_out, state_in
    """

    joint_qd_public_convention: bool = True
    """Whether free-joint ``joint_qd`` uses Newton's public twist convention.

    FeatherPGS stores free-joint ``joint_qd`` as ``(v_com_world, omega_world)`` [m/s, rad/s],
    matching :func:`newton.eval_fk`. The vanilla Featherstone solver instead uses an internal
    world-origin-referenced spatial twist, which requires
    ``eval_fk_with_velocity_conversion``.

    Integration layers that refresh maximal body state from joint state (for example after
    reset writes) must dispatch on this attribute. Applying the Featherstone-internal helper
    to public-convention ``joint_qd`` re-references the free-joint twist to the world origin,
    adding ``omega x x_com_world`` of phantom linear velocity to :attr:`newton.State.body_qd`
    that grows with the body's distance from the world origin.
    """

    # Test hook: pin a kernel implementation regardless of the size heuristic
    # (keys: cholesky_kernel, trisolve_kernel, hinv_jt_kernel).
    _kernel_overrides: ClassVar[dict[str, str]] = {}

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register the per-body attributes consumed by FeatherPGS.

        Each attribute maps to the matching PhysX rigid-body USD attribute:

        - ``rigid_body_angular_damping`` [1/s] damps the angular velocity of free bodies
          and floating bases (``physxRigidBody:angularDamping``, default 0.05).
        - ``rigid_body_max_linear_velocity`` [m/s] and ``rigid_body_max_angular_velocity``
          [rad/s] bound the velocity of free bodies (``physxRigidBody:maxLinearVelocity`` and
          ``physxRigidBody:maxAngularVelocity``, default unbounded).
        - ``rigid_body_max_depenetration_velocity`` [m/s] bounds the contact correction
          velocity of free bodies (``physxRigidBody:maxDepenetrationVelocity``, default
          unbounded).

        Models built without these attributes use the defaults.

        Args:
            builder: Model builder to register the attributes with.
        """

        def _angular_velocity_deg_to_rad(value, _context):
            if value is None:
                return float("inf")
            value = float(value)
            if not np.isfinite(value):
                return float("inf")
            return value * np.pi / 180.0

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rigid_body_angular_damping",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=_DEFAULT_ANGULAR_DAMPING,
                usd_attribute_name="physxRigidBody:angularDamping",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rigid_body_max_linear_velocity",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=float("inf"),
                usd_attribute_name="physxRigidBody:maxLinearVelocity",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rigid_body_max_angular_velocity",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=float("inf"),
                usd_attribute_name="physxRigidBody:maxAngularVelocity",
                usd_value_transformer=_angular_velocity_deg_to_rad,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rigid_body_max_depenetration_velocity",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=float("inf"),
                usd_attribute_name="physxRigidBody:maxDepenetrationVelocity",
            )
        )

    def __init__(
        self,
        model: Model,
        *,
        pgs_iterations: int = 12,
        pgs_beta: float = 0.2,
        pgs_cfm: float = 1.0e-6,
        pgs_omega: float = 1.0,
        update_mass_matrix_interval: int = 1,
        joint_limit_activation_gap: float = float("inf"),
        enable_joint_velocity_limits: bool = False,
        velocity_limit_activation_fraction: float = 0.0,
        dense_max_constraints: int = 32,
        mf_max_constraints: int = 512,
        warn_constraint_overflow: bool = True,
    ):
        """Create a FeatherPGS solver for a finalized CUDA model.

        Args:
            model: Model to simulate. It must be finalized on a CUDA device.
            pgs_iterations: Number of projected Gauss-Seidel iterations per step.
            pgs_beta: Baumgarte position-correction factor of contact and joint-limit rows,
                as a fraction of the position error removed per step.
            pgs_cfm: Constraint force mixing added to every row's effective-mass diagonal,
                which regularizes redundant rows.
            pgs_omega: Successive over-relaxation factor of the PGS sweep.
            update_mass_matrix_interval: Rebuild and refactor the mass matrix every this many
                calls to :meth:`step`; ``1`` rebuilds it every step. Larger values reuse a
                stale factorization between rebuilds. A change of ``dt``, :meth:`reset` and
                :meth:`notify_model_changed` still request a refresh of the affected
                articulations. Under CUDA graph capture the
                host-side cadence is baked into the graph, so the interval should divide the
                number of steps captured per graph.
            joint_limit_activation_gap: Distance from a finite position limit [m or rad] at
                which its row is created. ``inf`` creates the rows of every finite limit each
                step; a smaller gap creates a row only when ``q <= lower + gap`` or
                ``q >= upper - gap``.
            enable_joint_velocity_limits: Enforce :attr:`~newton.Model.joint_velocity_limit`
                of PRISMATIC, REVOLUTE and D6 DOFs with one row pair per limited DOF, after
                scaling articulation velocities that already exceed a limit. The free-body
                limits of :meth:`register_custom_attributes` are enforced whenever those
                attributes are registered, independently of this option.
            velocity_limit_activation_fraction: Create the velocity-limit rows of a DOF only
                when ``|qd| >= velocity_limit_activation_fraction * limit``. ``0`` creates
                them for every limited DOF each step; ``inf`` never creates them. The gate
                samples the velocity before the solve, so a DOF that crosses the threshold
                during a step is clamped one step later. Must be in ``[0, 1]`` or ``inf``.
            dense_max_constraints: Capacity of rows involving articulated bodies (contacts,
                joint limits and joint velocity limits) per world. Rows beyond it are
                dropped and reported, see :attr:`constraint_overflow`.
            mf_max_constraints: Capacity of free-body contact rows per world. Rows beyond it
                are dropped and reported, see :attr:`constraint_overflow`.
            warn_constraint_overflow: Print a device-side warning the first time a world
                exceeds a row capacity. The warning does not synchronize the host and is
                compatible with CUDA graph capture.
        """
        super().__init__(model)
        if not model.device.is_cuda:
            raise NotImplementedError("SolverFeatherPGS requires a CUDA device; this solver does not support CPU yet.")
        if model.requires_grad:
            raise NotImplementedError("SolverFeatherPGS does not support gradients (model.requires_grad=True).")
        _validate_supported_model(model)

        self.update_mass_matrix_interval = int(update_mass_matrix_interval)
        if self.update_mass_matrix_interval < 1:
            raise ValueError("update_mass_matrix_interval must be >= 1")
        self.pgs_iterations = int(pgs_iterations)
        if self.pgs_iterations < 0:
            raise ValueError("pgs_iterations must be non-negative")
        self.pgs_beta = float(pgs_beta)
        self.pgs_cfm = float(pgs_cfm)
        self.pgs_omega = float(pgs_omega)
        self.joint_limit_activation_gap = float(joint_limit_activation_gap)
        if np.isnan(self.joint_limit_activation_gap) or self.joint_limit_activation_gap < 0.0:
            raise ValueError("joint_limit_activation_gap must be non-negative or inf")
        self.enable_joint_velocity_limits = bool(enable_joint_velocity_limits)
        self.velocity_limit_activation_fraction = float(velocity_limit_activation_fraction)
        frac = self.velocity_limit_activation_fraction
        if np.isnan(frac) or not (0.0 <= frac <= 1.0 or np.isinf(frac)):
            # Finite values above 1 would leave speeds in (limit, fraction * limit)
            # permanently unclamped: the row only allocates at |qd| >= fraction * limit.
            raise ValueError("velocity_limit_activation_fraction must be in [0, 1] or inf")
        self._requested_dense_max_constraints = int(dense_max_constraints)
        if self._requested_dense_max_constraints < 1:
            raise ValueError("dense_max_constraints must be >= 1")
        if self._requested_dense_max_constraints - 1 > _DENSE_META_MAX_PARENT:
            raise ValueError(f"dense_max_constraints must be at most {_DENSE_META_MAX_PARENT + 1}")
        self.mf_max_constraints = int(mf_max_constraints)
        if self.mf_max_constraints < 1:
            raise ValueError("mf_max_constraints must be >= 1")
        self.warn_constraint_overflow = bool(warn_constraint_overflow)

        self.rigid_body_angular_damping = getattr(model, "rigid_body_angular_damping", None)
        if self.rigid_body_angular_damping is None:
            self.rigid_body_angular_damping = wp.full(
                max(model.body_count, 1), _DEFAULT_ANGULAR_DAMPING, dtype=wp.float32, device=model.device
            )
        self.rigid_body_max_linear_velocity = getattr(model, "rigid_body_max_linear_velocity", None)
        self.rigid_body_max_angular_velocity = getattr(model, "rigid_body_max_angular_velocity", None)
        self.rigid_body_max_depenetration_velocity = getattr(model, "rigid_body_max_depenetration_velocity", None)
        if self.rigid_body_max_depenetration_velocity is None:
            self.rigid_body_max_depenetration_velocity = wp.full(
                max(model.body_count, 1), float("inf"), dtype=wp.float32, device=model.device
            )
        self._has_rigid_body_velocity_limits = (
            self.rigid_body_max_linear_velocity is not None and self.rigid_body_max_angular_velocity is not None
        )

        # Kernel selection is automatic; _kernel_overrides is a test hook that pins
        # one implementation regardless of the articulation-size heuristic.
        self.cholesky_kernel = self._kernel_overrides.get("cholesky_kernel", "auto")
        self.trisolve_kernel = self._kernel_overrides.get("trisolve_kernel", "auto")
        self.hinv_jt_kernel = self._kernel_overrides.get("hinv_jt_kernel", "auto")
        if self.cholesky_kernel not in ("tiled", "loop", "auto"):
            raise ValueError("cholesky_kernel must be one of ['auto', 'loop', 'tiled']")
        if self.trisolve_kernel not in ("tiled", "loop", "auto"):
            raise ValueError("trisolve_kernel must be one of ['auto', 'loop', 'tiled']")
        if self.hinv_jt_kernel not in ("tiled", "par_row", "auto"):
            raise ValueError("hinv_jt_kernel must be one of ['auto', 'par_row', 'tiled']")
        self.small_dof_threshold = _SMALL_DOF_THRESHOLD_DEFAULT

        self._step = 0
        self._force_mass_update = False
        self._mass_update_requested = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        self._last_step_dt = None

        self._model_plan: _FeatherPGSModelPlan | None = None
        self._kinematic_joint_mask = wp.zeros(model.joint_count, dtype=wp.int32, device=model.device)
        self._kinematic_dof_mask = wp.zeros(model.joint_dof_count, dtype=wp.int32, device=model.device)
        self._joint_armature_device = wp.zeros(max(model.joint_dof_count, 1), dtype=wp.float32, device=model.device)
        self._update_kinematic_state()
        # Fully kinematic free bodies are removed from the response mapping; their
        # prescribed motion enters the contact targets of the rows that touch them.
        self._model_plan = _FeatherPGSModelPlan.build(model, self._kinematic_dof_mask_host)
        self._prescribed_articulation = wp.array(
            self._model_plan.prescribed_articulation, dtype=wp.int32, device=model.device
        )
        self._has_prescribed_response = bool(np.any(self._model_plan.prescribed_articulation != 0))
        self._compute_articulation_metadata(model)
        self._warn_unsolvable_global_contacts(model)
        self._setup_passive_joint_forces(model)
        self._compute_world_response_dof_mapping(model)
        self.dense_max_constraints = self._requested_dense_max_constraints
        self._execution_plan = _FeatherPGSExecutionPlan.build(
            self.size_groups,
            max_constraints=self.dense_max_constraints,
            max_shared_memory=int(getattr(model.device, "max_shared_memory_per_block", 0)),
            cholesky_kernel=self.cholesky_kernel,
            hinv_jt_kernel=self.hinv_jt_kernel,
            small_dof_threshold=self.small_dof_threshold,
            tile_threads=_TILE_THREADS,
        )
        self._jy_world_aliased = self._detect_jy_world_identity()
        # Tiled H^-1 J^T writes the world-gathered response directly unless the group and
        # world layouts already alias, and also computes the row diagonal.
        self._hinv_jt_writes_world = not self._jy_world_aliased
        self._hinv_jt_tiled_writes_group = not self._hinv_jt_writes_world
        self._hinv_jt_diag_sizes = frozenset(
            size for size in self.size_groups if self._execution_plan.use_tiled_hinv_jt(size)
        )
        if not self._hinv_jt_writes_world:
            self._hinv_jt_diag_sizes = frozenset()

        self._allocate_common_buffers(model)
        self._allocate_buffers(model)
        self._allocate_world_buffers(model)
        self._allocate_mf_buffers(model)
        self.mf_target_velocity = (
            wp.zeros_like(self.mf_rhs)
            if self._has_prescribed_response
            else wp.zeros((1, 1), dtype=wp.float32, device=model.device)
        )
        self._scatter_armature_to_groups()
        self._init_tiled_kernels(model)
        self._dummy_is_free_rigid = wp.zeros((1,), dtype=wp.int32, device=model.device)
        self._dummy_mf_slot_counter = wp.zeros((self.world_count,), dtype=wp.int32, device=model.device)
        self._dummy_contact_count = wp.zeros((1,), dtype=wp.int32, device=model.device)

        # Capacity status is allocated before any CUDA graph capture.
        self._row_overflow_warning_emitted = (
            wp.zeros(3, dtype=wp.int32, device=model.device) if self.warn_constraint_overflow else None
        )
        self.constraint_overflow = wp.zeros(int(model.world_count) + 1, dtype=wp.bool, device=model.device)
        """Capacity failure flags, shape ``[model.world_count + 1]``, dtype ``bool``.

        One entry per world plus a final entry for global (world ``-1``) articulations:
        the layout of the :meth:`reset` mask, so entry ``i`` is cleared exactly when mask
        entry ``i`` is selected. An entry is set when a step drops constraint
        rows of that world (``dense_max_constraints`` or ``mf_max_constraints`` exceeded)
        or drops a contact that couples a dynamic global body with another world. Global
        articulations share world 0's row storage, so a row loss in world 0 also sets the
        global entry while a dynamic global articulation exists. Receiving more contacts
        than the contact buffer holds, or contacts from which global contact reduction
        dropped candidates, sets every entry. Flags persist until :meth:`reset` clears
        them."""
        self._cross_world_contacts = wp.zeros(self.world_count + 1, dtype=wp.int32, device=model.device)
        if self._model_plan is not None and model.articulation_count:
            self._has_responding_global = bool(
                np.any((model.articulation_world.numpy() < 0) & (self._model_plan.response_dof_count > 0))
            )
        else:
            self._has_responding_global = False
        self._row_dropped_all = wp.zeros((2, max(self.world_count, 1)), dtype=wp.int32, device=model.device)
        self._row_dropped_dense = self._row_dropped_all[0]
        self._row_dropped_mf = self._row_dropped_all[1]

        if model.shape_material_mu is not None:
            self.shape_material_mu = model.shape_material_mu
        else:
            self.shape_material_mu = wp.zeros((1,), dtype=wp.float32, device=model.device)

    def _update_kinematic_state(self) -> None:
        """Refresh cached kinematic flags and effective joint armature."""
        model = self.model
        armature = model.joint_armature.numpy().copy()
        joint_mask = np.zeros(model.joint_count, dtype=np.int32)
        dof_mask = np.zeros(model.joint_dof_count, dtype=np.int32)

        if model.body_count and model.joint_count:
            kinematic_bodies = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
            joint_child = model.joint_child.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            for joint in range(model.joint_count):
                if not kinematic_bodies[joint_child[joint]]:
                    continue
                joint_mask[joint] = 1
                dof_start = joint_qd_start[joint]
                dof_end = joint_qd_start[joint + 1]
                if dof_end <= dof_start:
                    continue
                dof_mask[dof_start:dof_end] = 1
                armature[dof_start:dof_end] = 1.0e10

        if self._model_plan is not None:
            selected = np.nonzero(self._model_plan.prescribed_articulation != 0)[0]
            for articulation in selected:
                dof_start = int(self._model_plan.articulation_dof_start[articulation])
                dof_count = int(self._model_plan.articulation_dof_count[articulation])
                if not np.all(dof_mask[dof_start : dof_start + dof_count] != 0):
                    raise RuntimeError(
                        "A prescribed-response articulation changed kinematic membership; "
                        "reconstruct the solver and recapture its CUDA graph."
                    )

        self._joint_armature_effective = armature
        if armature.size:
            self._joint_armature_device.assign(armature.astype(np.float32))
        self._kinematic_dof_mask_host = dof_mask.copy()
        self._kinematic_joint_mask.assign(joint_mask)
        self._kinematic_dof_mask.assign(dof_mask)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh cached solver data after model changes.

        Joint (frames), joint DOF (armature, drive gains), body (kinematic flags) and
        inertial changes request a mass-matrix refresh of every articulation on the next
        step, independent of ``update_mass_matrix_interval``. Body flags (kinematic
        membership) and joint DOF properties (armature) are re-read. Other model data,
        such as gravity, limits and shape properties, is read every step. A kinematic free
        body that was removed from the response at construction cannot become dynamic
        again; reconstruct the solver in that case. Capacity status in
        :attr:`constraint_overflow` is not cleared, see :meth:`reset`.

        Args:
            flags: Bit-mask of :class:`~newton.ModelFlags` indicating which model properties changed.
        """
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES):
            self._update_kinematic_state()
            self._scatter_armature_to_groups()
            self._mass_update_requested.fill_(1)
        if flags & ModelFlags.JOINT_PROPERTIES:
            # Joint frames move the bodies the mass matrix is built from.
            self._mass_update_requested.fill_(1)
        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and self.model.body_count:
            # Re-derive the buffers baked from body_com/body_mass/body_inertia in
            # _allocate_common_buffers so runtime mass randomization reaches FK and CRBA.
            wp.launch(
                compute_spatial_inertia,
                self.model.body_count,
                inputs=[self.model.body_inertia, self.model.body_mass],
                outputs=[self.body_I_m],
                device=self.model.device,
            )
            wp.launch(
                compute_com_transforms,
                self.model.body_count,
                inputs=[self.model.body_com],
                outputs=[self.body_X_com],
                device=self.model.device,
            )
            self._mass_update_requested.fill_(1)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Clear solver-owned state of the selected worlds.

        The simulation state is not modified. For the selected worlds this clears the
        capacity status in :attr:`constraint_overflow` and requests a mass-matrix refresh
        on the next step, so a teleported articulation does not reuse a stale factorization;
        other worlds keep their refresh cadence. The solver keeps no impulse history between
        steps. The reset is launched on the device without host synchronization and can be
        captured in a CUDA graph.

        Args:
            state: Simulation state; left unchanged.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``. The first
                ``world_count`` entries select worlds; the final entry selects global
                articulations (world ``-1``) and the final entry of
                :attr:`constraint_overflow`. ``None`` resets everything.
            flags: Unused; the solver keeps no per-attribute state.
        """
        del flags
        world_mask = self._normalize_reset_world_mask(world_mask)
        if self.world_count == 0:
            return

        # One launch covers both the status entries (world_count + 1, including the
        # global entry) and the articulations whose mass factors are refreshed.
        wp.launch(
            _reset_solver_status,
            dim=max(self.constraint_overflow.shape[0], self.model.articulation_count),
            inputs=[
                world_mask,
                int(self.model.world_count),
                self._articulation_model_world,
                self._mass_update_requested,
                self.constraint_overflow,
            ],
            device=self.model.device,
        )

    def _compute_articulation_metadata(self, model):
        self._compute_articulation_indices(model)
        self._compute_root_free_metadata(model)
        self._setup_size_grouping(model)
        self._setup_world_mapping(model)
        self._build_body_maps(model)
        self._classify_free_rigid_bodies(model)
        self._compact_contact_jacobian = bool(
            self.body_response_dof_mask is not None
            and self.size_groups
            and max(self.size_groups) <= _CONTACT_JACOBIAN_MAX_DOF
        )
        self._setup_world_size_grouping(model)

    def _unsolvable_global_articulations(self, model) -> np.ndarray:
        """Return global articulations whose contacts with other worlds cannot be solved.

        A global (world ``-1``) articulation that keeps response DOFs (a dynamic one, or a
        kinematic one with joints; a kinematic free body is prescribed instead) is solved in
        world 0. Its contacts with an articulation that has response DOFs in another world
        would couple two worlds. The check is conservative: it only asks whether the shapes'
        flags and collision groups allow such a contact, not whether the bodies ever meet.
        """
        plan = self._model_plan
        none = np.zeros(0, dtype=np.int32)
        if plan is None or not model.articulation_count or not model.shape_count or model.world_count < 2:
            return none
        response = np.asarray(plan.response_dof_count)
        model_world = model.articulation_world.numpy()
        coupled = (model_world < 0) & (response > 0)
        other_world = (np.asarray(plan.articulation_world) != 0) & (model_world >= 0) & (response > 0)
        if not coupled.any() or not other_world.any():
            return none

        body_articulation = np.full(model.body_count, -1, dtype=np.int64)
        joint_child = model.joint_child.numpy()
        articulation_start = model.articulation_start.numpy()
        for articulation in range(model.articulation_count):
            children = joint_child[articulation_start[articulation] : articulation_start[articulation + 1]]
            body_articulation[children[children >= 0]] = articulation

        shape_body = model.shape_body.numpy()
        shape_articulation = np.where(shape_body >= 0, body_articulation[np.maximum(shape_body, 0)], -1)
        colliding = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        has_articulation = shape_articulation >= 0
        global_shapes = colliding & has_articulation & coupled[np.maximum(shape_articulation, 0)]
        world_shapes = colliding & has_articulation & other_world[np.maximum(shape_articulation, 0)]
        if not global_shapes.any() or not world_shapes.any():
            return none

        groups = model.shape_collision_group.numpy()
        world_groups = np.unique(groups[world_shapes])

        def groups_collide(a: int, b: int) -> bool:
            if a == 0 or b == 0:
                return False
            if a > 0:
                return a == b or b < 0
            return a != b

        flagged = set()
        for group in np.unique(groups[global_shapes]):
            if any(groups_collide(int(group), int(other)) for other in world_groups):
                flagged.update(int(a) for a in np.unique(shape_articulation[global_shapes & (groups == group)]))
        return np.array(sorted(flagged), dtype=np.int32)

    def _warn_unsolvable_global_contacts(self, model) -> None:
        """Warn once at construction if the model allows contacts that cannot be solved per world."""
        articulations = self._unsolvable_global_articulations(model)
        if articulations.size:
            warnings.warn(
                f"SolverFeatherPGS: global (world -1) articulations {articulations[:16].tolist()} are dynamic "
                "or kinematic with joints, and their shapes can collide with articulated bodies of worlds "
                "other than 0. Global articulations are solved in world 0, so such contacts cannot be solved: "
                "they are dropped, counted and flagged in constraint_overflow. Use world geometry or a "
                "kinematic free body for shared scenery, or add the body to every world.",
                UserWarning,
                stacklevel=3,
            )

    def _setup_passive_joint_forces(self, model) -> None:
        """Select the passive joint damping applied in the inverse-dynamics pass."""
        zeros = wp.zeros(max(int(model.joint_dof_count), 1), dtype=wp.float32, device=model.device)
        # Passive springs are not part of this solver; the inverse-dynamics kernel takes zeros.
        self._passive_spring_stiffness = zeros
        self._passive_spring_ref = zeros
        damping = model.joint_damping
        self._passive_joint_damping = damping if damping is not None else zeros

    def _compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            self.J_size = 0
            self.M_size = wp.int64(0)
            self.H_size = 0

            articulation_J_start = []
            articulation_M_start = []
            articulation_H_start = []

            articulation_M_rows = []
            articulation_H_rows = []
            articulation_J_rows = []
            articulation_J_cols = []

            articulation_dof_start = []
            articulation_coord_start = []

            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            if self._model_plan is None:
                raise RuntimeError("FeatherPGS model plan must be built before articulation metadata")

            for i in range(model.articulation_count):
                first_joint = articulation_start[i]
                last_joint = articulation_start[i + 1]

                first_coord = joint_q_start[first_joint]

                first_dof = int(self._model_plan.articulation_dof_start[i])
                joint_count = last_joint - first_joint
                dof_count = int(self._model_plan.articulation_dof_count[i])

                articulation_J_start.append(self.J_size)
                articulation_M_start.append(int(self.M_size))
                articulation_H_start.append(self.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)

                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)

                self.J_size += 6 * joint_count * dof_count
                self.M_size = wp.int64(self.M_size + wp.int64(joint_count * 36))
                self.H_size += dof_count * dof_count
            self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)

            self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)

            self.articulation_max_dofs = int(max(articulation_H_rows)) if articulation_H_rows else 0
            self.M_size = int(self.M_size)
        else:
            self.M_size = 0
            self.articulation_max_dofs = 0

    def _compute_root_free_metadata(self, model):
        if model.joint_count:
            joint_type = model.joint_type.numpy()
            joint_parent = model.joint_parent.numpy()
            root_mask = ((joint_type == JointType.FREE) | (joint_type == JointType.DISTANCE)) & (joint_parent < 0)
            free_root_joint_indices = np.flatnonzero(root_mask).astype(np.int32, copy=False)
            self._free_root_joint_count = int(free_root_joint_indices.size)
            self._free_root_joint_indices = (
                wp.array(free_root_joint_indices, dtype=wp.int32, device=model.device)
                if self._free_root_joint_count
                else None
            )
        else:
            self._free_root_joint_count = 0
            self._free_root_joint_indices = None

        if not model.articulation_count or not model.joint_count:
            self.articulation_root_dof_start = None
            return

        articulation_start = model.articulation_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        root_is_free = np.zeros(model.articulation_count, dtype=np.int32)
        root_dof_start = np.zeros(model.articulation_count, dtype=np.int32)

        for art in range(model.articulation_count):
            root_joint = articulation_start[art]
            root_dof_start[art] = int(joint_qd_start[root_joint])
            jt = int(joint_type[root_joint])
            jp = int(joint_parent[root_joint])
            if jp == -1 and (jt == int(JointType.FREE) or jt == int(JointType.DISTANCE)):
                root_is_free[art] = 1
        self.articulation_root_dof_start = wp.array(root_dof_start, dtype=wp.int32, device=model.device)

    def _setup_size_grouping(self, model):
        """Build response-size groups without modifying physical topology."""
        if not model.articulation_count or not model.joint_count:
            self.size_groups = []
            self.n_arts_by_size = {}
            self._free_body_inertia_sizes = frozenset()
            self.articulation_response_dof_count = wp.zeros(
                model.articulation_count, dtype=wp.int32, device=model.device
            )
            self._joint_limit_sizes = frozenset()
            self._joint_limit_q_index = None
            self._joint_limit_warp_kernels = {}
            return

        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before size grouping")
        response_dof_count = self._model_plan.response_dof_count
        solve_sizes = sorted({int(size) for size in response_dof_count if size > 0}, reverse=True)
        self.size_groups = solve_sizes
        self.n_arts_by_size = {size: int(np.sum(response_dof_count == size)) for size in solve_sizes}

        art_group_idx = np.full(model.articulation_count, -1, dtype=np.int32)
        group_to_art = {size: np.zeros(self.n_arts_by_size[size], dtype=np.int32) for size in solve_sizes}
        size_counters = dict.fromkeys(solve_sizes, 0)
        for art, size in enumerate(response_dof_count):
            if size == 0:
                continue
            group_index = size_counters[int(size)]
            art_group_idx[art] = group_index
            group_to_art[int(size)][group_index] = art
            size_counters[int(size)] += 1

        self.articulation_response_dof_count = wp.array(response_dof_count, dtype=wp.int32, device=model.device)
        self.art_group_idx = wp.array(art_group_idx, dtype=wp.int32, device=model.device)
        self.group_to_art = {
            size: wp.array(group_to_art[size], dtype=wp.int32, device=model.device) for size in solve_sizes
        }
        # A one-body free articulation has I_composite == I_body. Select the direct
        # source only for whole size groups so the CRBA kernel remains branch-free.
        self._free_body_inertia_sizes = frozenset(
            size for size in solve_sizes if np.all(self._model_plan.is_free_rigid[group_to_art[size]] != 0)
        )

        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        limit_joint = np.isin(
            joint_type,
            (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6)),
        )
        self._joint_limit_sizes = frozenset(
            int(response_dof_count[art])
            for art in range(model.articulation_count)
            if response_dof_count[art] > 0
            and np.any(limit_joint[articulation_start[art] : articulation_start[art + 1]])
        )
        joint_q_start = model.joint_q_start.numpy().astype(np.int32, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_dof_dim = model.joint_dof_dim.numpy().astype(np.int32, copy=False)
        limit_q_index = np.full(model.joint_dof_count, -1, dtype=np.int32)
        for joint in np.flatnonzero(limit_joint):
            axis_count = int(joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1])
            dof = int(joint_qd_start[joint])
            coord = int(joint_q_start[joint])
            limit_q_index[dof : dof + axis_count] = np.arange(coord, coord + axis_count, dtype=np.int32)
        self._joint_limit_q_index = wp.array(limit_q_index, dtype=wp.int32, device=model.device)
        self._joint_limit_warp_kernels = (
            {
                size: _get_joint_limit_warp_kernel(
                    size,
                    str(getattr(model.device, "arch", "")),
                    warps_per_block=_JOINT_LIMIT_WARPS_PER_BLOCK,
                )
                for size in self._joint_limit_sizes
            }
            if model.device.is_cuda and not model.requires_grad
            else {}
        )

    def _setup_world_mapping(self, model):
        """Materialize articulation-to-world mappings from the model plan."""
        if not model.articulation_count:
            self.world_count = 0
            self.art_to_world = None
            return
        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before world mapping")

        articulation_world = self._model_plan.articulation_world
        self.world_count = self._model_plan.world_count
        self.art_to_world = wp.array(articulation_world, dtype=wp.int32, device=model.device)
        # Articulations own a contiguous joint range [articulation_start, articulation_end).
        self.articulation_joint_end = wp.array(
            model.articulation_start.numpy()[1:], dtype=wp.int32, device=model.device
        )
        # Model worlds (global articulations keep world -1) select reset-mask entries.
        self._articulation_model_world = wp.array(
            model.articulation_world.numpy().astype(np.int32), dtype=wp.int32, device=model.device
        )

    def _setup_world_size_grouping(self, model):
        self.world_group_art_start = {}
        self.world_group_to_art = {}
        self.world_response_group_art_start = {}
        self.world_response_group_to_art = {}
        if (
            not model.articulation_count
            or self.art_to_world is None
            or self._model_plan is None
            or self.is_free_rigid is None
        ):
            return

        device = model.device
        art_to_world_np = self.art_to_world.numpy().astype(np.int32, copy=False)
        art_size_np = self._model_plan.response_dof_count.astype(np.int32, copy=False)
        is_free_rigid_np = self.is_free_rigid.numpy().astype(np.int32, copy=False)
        world_count = max(int(self.world_count), 0)
        for size in self.size_groups:
            # The diagonal accumulator must retain every solved articulation.
            per_world: list[list[int]] = [[] for _ in range(world_count)]
            response_per_world: list[list[int]] = [[] for _ in range(world_count)]
            for art in np.where(art_size_np == int(size))[0]:
                world = int(art_to_world_np[int(art)])
                if 0 <= world < world_count:
                    response_per_world[world].append(int(art))
                    if is_free_rigid_np[art] == 0:
                        per_world[world].append(int(art))

            starts = np.zeros(world_count + 1, dtype=np.int32)
            flat: list[int] = []
            response_starts = np.zeros(world_count + 1, dtype=np.int32)
            response_flat: list[int] = []
            for world, arts in enumerate(per_world):
                starts[world] = len(flat)
                flat.extend(arts)
                response_starts[world] = len(response_flat)
                response_flat.extend(response_per_world[world])
            starts[world_count] = len(flat)
            response_starts[world_count] = len(response_flat)
            flat_np = np.asarray(flat, dtype=np.int32)
            response_flat_np = np.asarray(response_flat, dtype=np.int32)
            self.world_group_art_start[int(size)] = wp.array(starts, dtype=wp.int32, device=device)
            self.world_group_to_art[int(size)] = wp.array(flat_np, dtype=wp.int32, device=device)
            self.world_response_group_art_start[int(size)] = wp.array(response_starts, dtype=wp.int32, device=device)
            self.world_response_group_to_art[int(size)] = wp.array(response_flat_np, dtype=wp.int32, device=device)

    def _build_body_maps(self, model):
        if not model.body_count or not model.articulation_count:
            self.body_to_joint = None
            self.body_to_articulation = None
            self.body_has_response_dofs = None
            self.body_response_dof_mask = None
            return

        joint_child = model.joint_child.numpy()
        joint_ancestor = model.joint_ancestor.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        articulation_start = model.articulation_start.numpy()
        articulation_dof_start = self._model_plan.articulation_dof_start
        response_dof_count = self._model_plan.response_dof_count

        body_to_joint = [-1] * model.body_count
        body_to_articulation = [-1] * model.body_count

        for articulation in range(model.articulation_count):
            joint_start = articulation_start[articulation]
            joint_end = articulation_start[articulation + 1]

            for joint_index in range(joint_start, joint_end):
                child = joint_child[joint_index]
                if child < 0:
                    continue

                body_to_joint[child] = joint_index
                body_to_articulation[child] = articulation

        body_has_response_dofs = np.zeros(model.body_count, dtype=np.int32)
        body_response_dof_mask = np.zeros(model.body_count, dtype=np.uint32)
        body_single_response_dof = np.full(model.body_count, -1, dtype=np.int32)
        for body, joint in enumerate(body_to_joint):
            articulation = body_to_articulation[body]
            if joint < 0 or articulation < 0:
                continue
            response_start = int(articulation_dof_start[articulation])
            response_end = response_start + int(response_dof_count[articulation])
            ancestor_joint = joint
            visited_joints: set[int] = set()
            response_dofs = []
            while ancestor_joint >= 0:
                if ancestor_joint in visited_joints:
                    raise ValueError(
                        "SolverFeatherPGS: cyclic joint ancestry while building body response maps "
                        f"(body {body}, articulation {articulation}, joint {ancestor_joint})."
                    )
                visited_joints.add(ancestor_joint)
                joint_dof_start = int(joint_qd_start[ancestor_joint])
                joint_dof_end = int(joint_qd_start[ancestor_joint + 1])
                overlap_start = max(joint_dof_start, response_start)
                overlap_end = min(joint_dof_end, response_end)
                if overlap_start < overlap_end:
                    body_has_response_dofs[body] = 1
                    for global_dof in range(overlap_start, overlap_end):
                        response_dofs.append(global_dof)
                        local_dof = global_dof - response_start
                        if local_dof < 32:
                            body_response_dof_mask[body] |= np.uint32(1 << local_dof)
                ancestor_joint = int(joint_ancestor[ancestor_joint])
            if len(response_dofs) == 1:
                body_single_response_dof[body] = response_dofs[0]

        device = model.device
        self.body_to_joint = wp.array(body_to_joint, dtype=wp.int32, device=device)
        self.body_to_articulation = wp.array(body_to_articulation, dtype=wp.int32, device=device)
        self.body_has_response_dofs = wp.array(body_has_response_dofs, dtype=wp.int32, device=device)
        self.body_response_dof_mask = wp.array(body_response_dof_mask, dtype=wp.uint32, device=device)

    def _classify_free_rigid_bodies(self, model):
        """Materialize free-rigid execution metadata from the model plan."""
        if not model.articulation_count or not model.joint_count:
            self._has_free_rigid_bodies = False
            self.is_free_rigid = None
            self.free_rigid_body_indices = None
            self._free_rigid_body_count = 0
            return
        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before free-body classification")

        is_free_rigid = self._model_plan.is_free_rigid
        self._free_rigid_body_count = len(self._model_plan.response_free_rigid_body_indices)
        self._has_free_rigid_bodies = self._free_rigid_body_count > 0

        self.is_free_rigid = wp.array(is_free_rigid, dtype=wp.int32, device=model.device)
        self.free_rigid_body_indices = wp.array(
            self._model_plan.response_free_rigid_body_indices, dtype=wp.int32, device=model.device
        )

    def _compute_world_response_dof_mapping(self, model):
        """Build compact per-world response offsets and global-DOF indices."""
        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before response mapping")

        articulation_world = self._model_plan.articulation_world
        response_dof_count = self._model_plan.response_dof_count
        articulation_dof_start = self._model_plan.articulation_dof_start
        articulation_world_dof_offset = np.full(model.articulation_count, -1, dtype=np.int32)
        world_indices: list[list[int]] = [[] for _ in range(self.world_count)]

        for world in range(self.world_count):
            articulations = np.nonzero((articulation_world == world) & (response_dof_count > 0))[0]
            articulations = sorted(articulations, key=lambda art: int(articulation_dof_start[art]))
            for art in articulations:
                articulation_world_dof_offset[art] = len(world_indices[world])
                start = int(articulation_dof_start[art])
                count = int(response_dof_count[art])
                world_indices[world].extend(range(start, start + count))

        world_dof_count = np.asarray([len(indices) for indices in world_indices], dtype=np.int32)
        self.max_world_dofs = int(np.max(world_dof_count)) if len(world_dof_count) else 0
        padded_indices = np.full((self.world_count, self.max_world_dofs), -1, dtype=np.int32)
        for world, indices in enumerate(world_indices):
            padded_indices[world, : len(indices)] = indices

        self.articulation_world_dof_offset = wp.array(
            articulation_world_dof_offset, dtype=wp.int32, device=model.device
        )
        self.world_dof_count = wp.array(world_dof_count, dtype=wp.int32, device=model.device)
        self.world_dof_indices = wp.array(padded_indices, dtype=wp.int32, device=model.device)

    def _detect_jy_world_identity(self) -> bool:
        """Return whether host-side group and compact world J/Y layouts are identical."""
        if len(self.size_groups) != 1 or self.art_to_world is None:
            return False
        size = self.size_groups[0]
        if self.n_arts_by_size[size] != self.world_count:
            return False
        articulation_world = self._model_plan.articulation_world
        group_to_art = np.flatnonzero(self._model_plan.response_dof_count == size)
        expected_world = np.arange(len(group_to_art), dtype=articulation_world.dtype)
        return np.array_equal(articulation_world[group_to_art], expected_world)

    def _allocate_common_buffers(self, model):
        if model.joint_count:
            self.mass_update_mask = wp.zeros((model.articulation_count,), dtype=wp.int32, device=model.device)
            self.v_hat = wp.zeros_like(model.joint_qd)
            self.v_out = wp.zeros_like(model.joint_qd)
            self.qd_work = wp.zeros_like(model.joint_qd)
        else:
            self.mass_update_mask = None
            self.v_hat = None
            self.v_out = None
            self.qd_work = None

        if model.body_count:
            self.body_I_m = wp.empty((model.body_count,), dtype=wp.spatial_matrix, device=model.device)
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )
            self.body_X_com = wp.empty((model.body_count,), dtype=wp.transform, device=model.device)
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )
            self.body_I_c = wp.empty((model.body_count,), dtype=wp.spatial_matrix, device=model.device)
            inertia_term_count = 1
            self._body_inertia_terms = wp.empty((inertia_term_count, 12), dtype=wp.float32, device=model.device)
        else:
            self.body_I_m = None
            self.body_X_com = None
            self.body_I_c = None
            self._body_inertia_terms = None

        if not model.articulation_count or not model.joint_count:
            self.articulation_origin = None
            return

        self.articulation_origin = wp.zeros((model.articulation_count,), dtype=wp.vec3, device=model.device)

        max_dofs = self.articulation_max_dofs
        if max_dofs == 0:
            return

        device = model.device
        articulation_count = model.articulation_count
        total_rows = articulation_count * max_dofs

        self.aug_row_counts = wp.zeros((articulation_count,), dtype=wp.int32, device=device)
        self.aug_row_dof_index = wp.zeros((total_rows,), dtype=wp.int32, device=device)
        self.aug_row_K = wp.zeros((total_rows,), dtype=wp.float32, device=device)

    def _allocate_buffers(self, model):
        """Allocate per-group mass-matrix and response buffers and per-contact row bookkeeping."""
        device = model.device
        max_constraints = self.dense_max_constraints
        self.H_by_size = {}
        self.L_by_size = {}
        self.J_by_size = {}
        self.Y_by_size = {}
        self.diag_by_size = {}
        self.R_by_size = {}
        self.tau_by_size = {}
        self.qdd_by_size = {}
        self._dummy_hinv_diag = wp.zeros((1, 1), dtype=wp.float32, device=device)
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            self.H_by_size[size] = wp.zeros((n_arts, size, size), dtype=wp.float32, device=device)
            self.L_by_size[size] = wp.zeros((n_arts, size, size), dtype=wp.float32, device=device)
            self.J_by_size[size] = wp.zeros((n_arts, max_constraints, size), dtype=wp.float32, device=device)
            self.Y_by_size[size] = wp.zeros((n_arts, max_constraints, size), dtype=wp.float32, device=device)
            self.diag_by_size[size] = (
                wp.zeros((n_arts, max_constraints), dtype=wp.float32, device=device)
                if size in self._hinv_jt_diag_sizes
                else self._dummy_hinv_diag
            )
            # Armature, added to the mass-matrix diagonal by the Cholesky kernels.
            self.R_by_size[size] = wp.zeros((n_arts, size), dtype=wp.float32, device=device)
            self.tau_by_size[size] = wp.zeros((n_arts, size, 1), dtype=wp.float32, device=device)
            self.qdd_by_size[size] = wp.zeros((n_arts, size, 1), dtype=wp.float32, device=device)

        max_contacts = int(model.rigid_contact_max)
        if max_contacts <= 0:
            # The collision pipeline may manage its own capacity and leave
            # model.rigid_contact_max unset; use the same estimator.
            from ...sim.collide import _estimate_rigid_contact_max  # noqa: PLC0415

            max_contacts = int(_estimate_rigid_contact_max(model))
        max_contacts = max(max_contacts, 1)
        self._max_contacts_alloc = max_contacts
        self.contact_world = wp.zeros((max_contacts,), dtype=wp.int32, device=device)
        self.contact_slot = wp.zeros((max_contacts,), dtype=wp.int32, device=device)
        self.contact_art_a = wp.zeros((max_contacts,), dtype=wp.int32, device=device)
        self.contact_art_b = wp.zeros((max_contacts,), dtype=wp.int32, device=device)
        self.contact_path = wp.zeros((max_contacts,), dtype=wp.int32, device=device)
        self.slot_counter = wp.zeros((self.world_count,), dtype=wp.int32, device=device)
        self._dense_first_rejected_slot = wp.full(
            (self.world_count,), _ROW_SLOT_UNBOUNDED, dtype=wp.int32, device=device
        )
        # First free-body slot past the contact/friction rows (start of the velocity-limit tail).
        self.mf_contact_rows_end = wp.zeros((self.world_count,), dtype=wp.int32, device=device)

        if self.enable_joint_velocity_limits and model.joint_dof_count > 0:
            # Two unilateral rows per limited DOF: lower (+e_i) and upper (-e_i).
            self.velocity_limit_slot = wp.full((2 * model.joint_dof_count,), -1, dtype=wp.int32, device=device)
            self.velocity_limit_sign = wp.zeros((2 * model.joint_dof_count,), dtype=wp.float32, device=device)
        else:
            self.velocity_limit_slot = None
            self.velocity_limit_sign = None

    def _allocate_world_buffers(self, model):
        """Allocate the per-world dense row system (response, metadata and impulses)."""
        device = model.device
        shape = (self.world_count, self.dense_max_constraints)
        if self._jy_world_aliased:
            # One articulation per world in a single size group: the world-indexed
            # views alias the group buffers, so no gather or duplicate storage is needed.
            size = self.size_groups[0]
            self.J_world = self.J_by_size[size]
            self.Y_world = self.Y_by_size[size]
        else:
            self.J_world = wp.zeros((*shape, self.max_world_dofs), dtype=wp.float32, device=device)
            self.Y_world = wp.zeros((*shape, self.max_world_dofs), dtype=wp.float32, device=device)
        self.rhs = wp.zeros(shape, dtype=wp.float32, device=device)
        self.impulses = wp.zeros(shape, dtype=wp.float32, device=device)
        self.diag = wp.zeros(shape, dtype=wp.float32, device=device)
        self.row_type = wp.zeros(shape, dtype=wp.int32, device=device)
        self.row_parent = wp.full(shape, -1, dtype=wp.int32, device=device)
        self.row_mu = wp.zeros(shape, dtype=wp.float32, device=device)
        self.phi = wp.zeros(shape, dtype=wp.float32, device=device)
        self.target_velocity = wp.zeros(shape, dtype=wp.float32, device=device)
        self.constraint_count = wp.zeros((self.world_count,), dtype=wp.int32, device=device)

    def _allocate_mf_buffers(self, model):
        """Allocate the per-world free-body row system.

        Models without free bodies keep one-slot arrays: ``mf_constraint_count`` stays
        zero, so no kernel reads or writes a free-body row.
        """
        device = model.device
        worlds = self.world_count
        rows = self.mf_max_constraints if self._has_free_rigid_bodies else 1
        self.mf_constraint_count = wp.zeros((worlds,), dtype=wp.int32, device=device)
        self.mf_slot_counter = wp.zeros((worlds,), dtype=wp.int32, device=device)
        self._mf_first_rejected_slot = wp.full((worlds,), _ROW_SLOT_UNBOUNDED, dtype=wp.int32, device=device)
        self.mf_body_a = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.mf_body_b = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.mf_dof_a = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.mf_dof_b = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.mf_J_a = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.mf_J_b = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.mf_MiJt_a = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.mf_MiJt_b = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.mf_rhs = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.mf_impulses = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.mf_eff_mass_inv = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.mf_row_type = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.mf_row_parent = wp.full((worlds, rows), -1, dtype=wp.int32, device=device)
        self.mf_row_mu = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.mf_phi = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        # Packed metadata of the fused solve (int4 per row):
        #   .x = (dof_a << 16) | (dof_b & 0xFFFF), .y = eff_mass_inv bits,
        #   .z = rhs bits, .w = row_type | (row_parent << 16)
        self.mf_meta_packed = wp.zeros((worlds, rows * 4), dtype=wp.int32, device=device)
        if not self._has_free_rigid_bodies:
            self.rigid_velocity_limit_slot = None
            self.rigid_velocity_limit_sign = None
            return
        self.mf_body_Hinv = wp.zeros((model.body_count,), dtype=wp.spatial_matrix, device=device)
        if self._has_rigid_body_velocity_limits:
            # Lower/upper rows for the three linear and three angular velocity components.
            count = 12 * self._free_rigid_body_count
            self.rigid_velocity_limit_slot = wp.full((count,), -1, dtype=wp.int32, device=device)
            self.rigid_velocity_limit_sign = wp.zeros((count,), dtype=wp.float32, device=device)
        else:
            self.rigid_velocity_limit_slot = None
            self.rigid_velocity_limit_sign = None

    def _scatter_armature_to_groups(self):
        """Copy armature from model (DOF-ordered) to size-grouped storage."""
        if not self.size_groups:
            return

        armature_np = self._joint_armature_effective
        art_dof_start_np = self._model_plan.articulation_dof_start
        art_H_rows_np = self._model_plan.articulation_dof_count

        # R_by_size is sized to actual DOF count (matches H_by_size allocation)
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            R_np = np.zeros((n_arts, size), dtype=np.float32)

            group_to_art_np = np.flatnonzero(self._model_plan.response_dof_count == size)
            for group_idx in range(n_arts):
                art_idx = group_to_art_np[group_idx]
                dof_start = art_dof_start_np[art_idx]
                dof_count = art_H_rows_np[art_idx]
                R_np[group_idx, :dof_count] = armature_np[dof_start : dof_start + dof_count]

            self.R_by_size[size].assign(R_np)

    def _init_tiled_kernels(self, model):
        """Resolve size-specialized Warp kernels once for this solver shape."""
        device_arch = model.device.arch
        self._cholesky_kernels_by_size = {}
        self._triangular_solve_kernels_by_size = {}
        self._hinv_jt_kernels_by_size = {}
        self._hinv_jt_chunk_count_by_size = {}

        for size in self.size_groups:
            self._cholesky_kernels_by_size[size] = (
                _get_cholesky_kernel(size, device_arch, _TILE_THREADS)
                if self._execution_plan.use_tiled_cholesky(size)
                else None
            )
            self._triangular_solve_kernels_by_size[size] = _get_triangular_solve_kernel(
                size, device_arch, _TILE_THREADS
            )
            hinv_jt_chunk_size = self._execution_plan.hinv_jt_chunk_size(size)
            if hinv_jt_chunk_size is None:
                self._hinv_jt_kernels_by_size[size] = None
                self._hinv_jt_chunk_count_by_size[size] = 0
                continue
            self._hinv_jt_chunk_count_by_size[size] = (
                self.dense_max_constraints + hinv_jt_chunk_size - 1
            ) // hinv_jt_chunk_size
            if size in self._hinv_jt_diag_sizes:
                self._hinv_jt_kernels_by_size[size] = _get_hinv_jt_kernel(
                    size,
                    self.dense_max_constraints,
                    device_arch,
                    _TILE_THREADS,
                    constraint_chunk_size=hinv_jt_chunk_size,
                    write_world=self._hinv_jt_writes_world,
                    write_group=self._hinv_jt_tiled_writes_group,
                    compute_diag=True,
                )
            else:
                self._hinv_jt_kernels_by_size[size] = _get_hinv_jt_plain_kernel(
                    size,
                    self.dense_max_constraints,
                    device_arch,
                    _TILE_THREADS,
                    constraint_chunk_size=hinv_jt_chunk_size,
                    write_world=self._hinv_jt_writes_world,
                    write_group=self._hinv_jt_tiled_writes_group,
                )

        self._pack_mf_meta_kernel = _get_pack_mf_meta_kernel(self.mf_meta_packed.shape[1] // 4, device_arch)
        self._pgs_solve_mf_gs_kernel = None
        if self.world_count > 0 and self.max_world_dofs > 0:
            mf_rows = self.mf_meta_packed.shape[1] // 4
            shared_metadata = _use_resident_mfgs_metadata(
                self.dense_max_constraints,
                mf_rows,
                self.max_world_dofs,
                int(getattr(model.device, "max_shared_memory_per_block", 0)),
            )
            self._pgs_solve_mf_gs_kernel = _get_pgs_solve_mf_gs_kernel(
                self.dense_max_constraints,
                mf_rows,
                self.max_world_dofs,
                device_arch,
                has_dense_velocity_limit_rows=self.enable_joint_velocity_limits,
                shared_metadata=shared_metadata,
            )

    def _pack_mf_meta(self) -> None:
        wp.launch_tiled(
            self._pack_mf_meta_kernel,
            dim=[self.world_count],
            inputs=[
                self.mf_constraint_count,
                self.mf_dof_a,
                self.mf_dof_b,
                self.mf_eff_mass_inv,
                self.mf_rhs,
                self.mf_row_type,
                self.mf_row_parent,
            ],
            outputs=[self.mf_meta_packed],
            block_dim=32,
            device=self.model.device,
        )

    def _launch_pgs_solve(self) -> None:
        """Run the fused matrix-free Gauss-Seidel sweep over the dense and free-body rows."""
        if self.pgs_iterations <= 0 or self._pgs_solve_mf_gs_kernel is None:
            return
        wp.launch_tiled(
            self._pgs_solve_mf_gs_kernel,
            dim=[self.world_count],
            inputs=[
                self.constraint_count,
                self.world_dof_indices,
                self.rhs,
                self.diag,
                self.impulses,
                self.J_world,
                self.Y_world,
                self.row_type,
                self.row_parent,
                self.row_mu,
                self.mf_constraint_count,
                self.mf_contact_rows_end,
                self.mf_meta_packed,
                self.mf_impulses,
                self.mf_J_a,
                self.mf_J_b,
                self.mf_MiJt_a,
                self.mf_MiJt_b,
                self.mf_row_mu,
                self.pgs_iterations,
                self.pgs_omega,
            ],
            outputs=[self.v_out],
            block_dim=32,
            device=self.model.device,
        )

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> State:
        """Advance the simulation by one time step.

        Args:
            state_in: Input state. Its maximal coordinates (``body_q``) are refreshed from
                ``joint_q`` by forward kinematics during the step.
            state_out: Output state receiving ``joint_q``, ``joint_qd``, ``body_q``,
                ``body_qd`` and, when requested, ``body_parent_f``.
            control: Control input; ``None`` uses the model's default control.
            contacts: Rigid contacts from :class:`~newton.CollisionPipeline`, or ``None``.
                The buffer must not hold more contacts than ``model.rigid_contact_max``
                allowed when the solver was constructed.
            dt: Time step [s].

        Returns:
            ``state_out``.
        """
        if contacts is not None and contacts.rigid_contact_max > self._max_contacts_alloc:
            raise ValueError(
                "FeatherPGS contact capacity mismatch: received "
                f"{contacts.rigid_contact_max} slots, but solver scratch was allocated for "
                f"{self._max_contacts_alloc}. Set model.rigid_contact_max before constructing the solver."
            )
        if self._last_step_dt is not None and abs(self._last_step_dt - dt) > 1.0e-8:
            # The augmented mass matrix depends on dt through the implicit drive terms.
            self._force_mass_update = True
        self._last_step_dt = dt

        model = self.model
        if control is None:
            control = model.control(clone_variables=False)
        if not model.joint_count:
            self._step += 1
            return state_out
        state_aug = self._prepare_augmented_state(state_in)

        # Stage 1: forward kinematics, inverse dynamics with implicit drives, and CRBA.
        stage3_qd = self._stage1_fk_id(state_in, state_aug, state_out)
        self._stage1_joint_tau(state_in, state_aug, state_out, control, dt)
        self._stage1_crba(state_aug)

        # Stage 2: factor the augmented mass matrix of every articulation group.
        for size in self.size_groups:
            if self._execution_plan.use_tiled_cholesky(size):
                self._stage2_cholesky_tiled(size)
            else:
                self._stage2_cholesky_loop(size)

        # Stage 3: unconstrained acceleration and velocity prediction.
        state_aug.joint_qdd.zero_()
        for size in self.size_groups:
            use_tiled = self.trisolve_kernel == "tiled" or (
                self.trisolve_kernel == "auto" and size > self.small_dof_threshold
            )
            if use_tiled:
                self._stage3_trisolve_tiled(size, state_aug)
            else:
                self._stage3_trisolve_loop(size, state_aug)
        self._stage3_compute_v_hat(state_in, state_aug, dt, stage3_qd)

        # Stage 4: constraint rows, responses Y = H^-1 J^T, diagonals and right-hand sides.
        self._stage4_build_rows(state_in, state_aug, contacts)
        for size in self.size_groups:
            if self._execution_plan.use_tiled_hinv_jt(size):
                self._stage4_hinv_jt_tiled(size)
            else:
                self._stage4_hinv_jt_par_row(size)
        self._stage4_compute_matrix_free_diag()
        wp.launch(
            finalize_world_diag_cfm,
            dim=self.world_count,
            inputs=[self.constraint_count, self.pgs_cfm],
            outputs=[self.diag],
            device=model.device,
        )
        # The right-hand side holds only the bias; J v is recomputed every iteration.
        wp.launch(
            compute_world_contact_bias,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.phi,
                self.row_type,
                self.target_velocity,
                self.pgs_beta,
                dt,
            ],
            outputs=[self.rhs],
            device=model.device,
        )
        if self._has_free_rigid_bodies:
            self._mf_pgs_setup(state_aug, dt)
            wp.launch(
                compute_mf_world_dof_offsets,
                dim=self.world_count * self.mf_max_constraints,
                inputs=[
                    self.mf_constraint_count,
                    self.mf_body_a,
                    self.mf_body_b,
                    self.body_to_articulation,
                    self.articulation_world_dof_offset,
                    self.mf_max_constraints,
                ],
                outputs=[self.mf_dof_a, self.mf_dof_b],
                device=model.device,
            )

        # Stages 5 and 6: projected Gauss-Seidel in impulse space, starting from v_hat.
        wp.launch(
            prepare_world_impulses,
            dim=self.world_count,
            inputs=[self.constraint_count],
            outputs=[self.impulses],
            device=model.device,
        )
        if not self._jy_world_aliased and not self._hinv_jt_writes_world:
            for size in self.size_groups:
                n_arts = self.n_arts_by_size[size]
                wp.launch(
                    gather_JY_to_world,
                    dim=int(n_arts * self.dense_max_constraints * size),
                    inputs=[
                        self.group_to_art[size],
                        self.art_to_world,
                        self.articulation_world_dof_offset,
                        self.constraint_count,
                        self.J_by_size[size],
                        self.Y_by_size[size],
                        size,
                        self.dense_max_constraints,
                        n_arts,
                    ],
                    outputs=[self.J_world, self.Y_world],
                    device=model.device,
                )
        wp.copy(self.v_out, self.v_hat)
        self._pack_mf_meta()
        self._launch_pgs_solve()

        # Stage 7: convert the solved velocity to accelerations, integrate and publish.
        wp.launch(
            update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, self._kinematic_dof_mask, 1.0 / dt],
            outputs=[self.v_out, state_aug.joint_qdd],
            device=model.device,
        )
        if self._free_root_joint_count:
            # Invert the free-root transport of the integrator so it realizes the solved velocity.
            wp.launch(
                remove_free_root_transport_from_qdd,
                dim=self._free_root_joint_count,
                inputs=[
                    self._free_root_joint_indices,
                    model.joint_qd_start,
                    self._kinematic_joint_mask,
                    state_in.joint_qd,
                ],
                outputs=[state_aug.joint_qdd],
                device=model.device,
            )
        wp.launch(
            kernel=integrate_generalized_joints,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                self._kinematic_joint_mask,
                model.joint_dof_dim,
                model.body_com,
                model.joint_X_c,
                state_in.joint_q,
                state_in.joint_qd,
                state_aug.joint_qdd,
                dt,
                self.rigid_body_angular_damping,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd],
            device=model.device,
        )
        self._stage7_update_kinematics(state_out)
        self._step += 1
        return state_out

    def check_constraint_capacity(self) -> None:
        """Raise if any world, or the global entry, lost contacts or constraint rows since its last reset.

        Reads :attr:`constraint_overflow` on the host, so call it outside CUDA graph capture,
        for example at an observation boundary. Kernels can read :attr:`constraint_overflow`
        directly without host synchronization. Recovering from a capacity loss requires larger
        capacities, a new solver (and newly captured graphs) and a :meth:`reset` of the affected
        worlds. Contacts that couple a dynamic global body, or a kinematic global articulation
        with joints, with another world are flagged too; no capacity can solve them.

        Raises:
            RuntimeError: If called during graph capture, or if any entry is flagged.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("check_constraint_capacity() must run outside CUDA graph capture")
        flags = self.constraint_overflow.numpy()
        worlds = np.flatnonzero(flags[:-1])
        global_flagged = bool(flags[-1])
        if worlds.size or global_flagged:
            where = f"worlds {worlds[:16].tolist()} ({worlds.size} invalid worlds)"
            if global_flagged:
                where += " and global (world -1) articulations"
            raise RuntimeError(
                f"FeatherPGS dropped contacts or constraint rows in {where} (capacity exceeded, or "
                "contacts that cannot be solved per world). Rows beyond "
                "dense_max_constraints or mf_max_constraints, and contacts beyond the contact buffer or "
                "dropped by contact reduction, need larger capacities. Contacts that couple a dynamic "
                "global body, or a kinematic global articulation with joints, with another world cannot "
                "be solved per world at any capacity and need a different scene layout. Reset the "
                "affected worlds before accepting transitions."
            )

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Write the linear contact forces of the last step to ``contacts``.

        Fills :attr:`~newton.Contacts.rigid_contact_force` [N] and, when allocated, the
        linear part of :attr:`~newton.Contacts.force` from the normal and friction impulses
        of each contact divided by the time step. The torque part of
        :attr:`~newton.Contacts.force` is left zero; this solver does not report a contact
        wrench. Contacts whose rows were dropped for capacity report zero force.

        Args:
            contacts: The contacts passed to the last :meth:`step`.
            state: Unused.
        """
        del state
        if contacts is None or contacts.rigid_contact_count is None:
            return

        dt = self._last_step_dt
        inv_dt = 0.0 if dt is None or dt <= 0.0 else 1.0 / dt
        wp.launch(
            compute_contact_linear_force_from_impulses,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_normal,
                self.contact_world,
                self.contact_slot,
                self.contact_path,
                self.impulses,
                self.mf_impulses,
                self.constraint_count,
                self.mf_constraint_count,
                inv_dt,
            ],
            outputs=[contacts.rigid_contact_force],
            device=self.model.device,
        )
        if contacts.force is not None:
            wp.launch(
                pack_contact_linear_force_as_spatial,
                dim=contacts.rigid_contact_max,
                inputs=[contacts.rigid_contact_count, contacts.rigid_contact_force],
                outputs=[contacts.force],
                device=self.model.device,
            )

    def _prepare_augmented_state(self, state_in: State) -> "SolverFeatherPGS":
        """Allocate the solver-owned per-step dynamics buffers on first use."""
        if not getattr(self, "_featherstone_augmented", False):
            self._allocate_state_aux_vars(self.model, self)
        return self

    def _allocate_state_aux_vars(self, model, target):
        if not model.body_count:
            return
        target.joint_qdd = wp.zeros_like(model.joint_qd)
        target.joint_tau = wp.empty_like(model.joint_qd)
        target.joint_S_s = wp.empty((model.joint_dof_count,), dtype=wp.spatial_vector, device=model.device)
        target.body_q_com = wp.empty_like(model.body_q)
        target.body_I_s = wp.empty((model.body_count,), dtype=wp.spatial_matrix, device=model.device)
        target.body_v_s = wp.empty((model.body_count,), dtype=wp.spatial_vector, device=model.device)
        target.body_a_s = wp.empty((model.body_count,), dtype=wp.spatial_vector, device=model.device)
        target.body_f_s = wp.zeros((model.body_count,), dtype=wp.spatial_vector, device=model.device)
        target.body_ft_s = wp.zeros((model.body_count,), dtype=wp.spatial_vector, device=model.device)
        target._featherstone_augmented = True

    def _stage1_fk_id(self, state_in: State, state_aug: State, state_out: State) -> wp.array:
        """Evaluate forward kinematics and the inverse-dynamics bias; return the predictor's input velocity."""
        model = self.model
        state_aug.body_f_s.zero_()
        if self.enable_joint_velocity_limits:
            # Scale velocities that already exceed their limit before predicting.
            wp.copy(self.qd_work, state_in.joint_qd)
            wp.launch(
                prescale_joint_velocity_limits,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_start,
                    model.joint_type,
                    model.joint_child,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_velocity_limit,
                    model.body_flags,
                ],
                outputs=[self.qd_work],
                device=model.device,
            )
            stage3_qd = self.qd_work
        else:
            stage3_qd = state_in.joint_qd

        refresh_composite = (self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update
        wp.launch(
            eval_rigid_fk_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.articulation_joint_end,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q,
                stage3_qd,
                model.joint_X_p,
                model.joint_X_c,
                self.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                self.is_free_rigid,
                int(refresh_composite),
                0,
                model.body_world,
                model.gravity,
            ],
            outputs=[
                state_in.body_q,
                state_aug.body_q_com,
                self.articulation_origin,
                state_aug.joint_S_s,
                state_aug.body_I_s,
                self._body_inertia_terms,
                state_aug.body_v_s,
                state_aug.body_f_s,
                state_aug.body_a_s,
            ],
            block_dim=16,
            device=model.device,
        )
        if model.body_count:
            wp.launch(
                update_body_qd_from_featherstone,
                dim=model.body_count,
                inputs=[
                    state_aug.body_v_s,
                    state_in.body_q,
                    model.body_com,
                    self.body_to_articulation,
                    self.articulation_origin,
                ],
                outputs=[state_out.body_qd],
                device=model.device,
            )
        return stage3_qd

    def _stage1_joint_tau(self, state_in: State, state_aug: State, state_out: State, control: Control, dt: float):
        """Accumulate ``joint_tau`` and the implicit-drive mass terms.

        After this call ``state_aug.joint_tau`` holds the rigid-body bias (Coriolis, gravity,
        external body forces), :attr:`~newton.Control.joint_f`, passive joint damping and the
        explicit part ``u0`` of each drive, clamped to :attr:`~newton.Model.joint_effort_limit`
        (an actuator-only clamp, as MuJoCo's ``actuatorfrcrange`` and PhysX's drive
        ``maxForce``). The implicit part ``K = dt * kd + dt^2 * ke`` is stored in
        ``aug_row_K`` and added to the mass-matrix diagonal by :meth:`_stage1_crba`.
        """
        model = self.model
        body_f = state_in.body_f if state_in.body_count else None
        state_aug.body_ft_s.zero_()
        common_inputs = [
            model.articulation_start,
            self.articulation_joint_end,
        ]
        tau_inputs = [
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_articulation,
            model.joint_qd_start,
            model.joint_q_start,
            model.joint_dof_dim,
            control.joint_f,
            state_in.joint_q,
            state_in.joint_qd,
            self._passive_spring_stiffness,
            self._passive_spring_ref,
            self._passive_joint_damping,
            state_aug.joint_S_s,
            state_aug.body_f_s,
            body_f,
            model.body_flags,
            state_in.body_q,
            model.body_com,
            self.articulation_origin,
        ]
        if self.articulation_max_dofs > 0:
            wp.launch(
                eval_rigid_tau_and_augmented_drives,
                dim=model.articulation_count,
                inputs=[
                    *common_inputs,
                    self.articulation_H_rows,
                    *tau_inputs,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_q,
                    model.joint_target_q_start,
                    control.joint_target_qd,
                    model.joint_effort_limit,
                    self.articulation_max_dofs,
                    dt,
                ],
                outputs=[
                    state_aug.body_ft_s,
                    self.aug_row_counts,
                    self.aug_row_dof_index,
                    self.aug_row_K,
                    state_aug.joint_tau,
                ],
                block_dim=_SERIAL_KERNEL_BLOCK_DIM,
                device=model.device,
            )
        else:
            wp.launch(
                eval_rigid_tau,
                dim=model.articulation_count,
                inputs=[*common_inputs, *tau_inputs],
                outputs=[state_aug.body_ft_s, state_aug.joint_tau],
                block_dim=_SERIAL_KERNEL_BLOCK_DIM,
                device=model.device,
            )
        if state_out.body_parent_f is not None:
            wp.launch(
                compute_body_parent_f,
                dim=model.body_count,
                inputs=[
                    self.body_to_articulation,
                    self.articulation_origin,
                    state_aug.body_f_s,
                    state_aug.body_ft_s,
                    body_f,
                    model.body_flags,
                    state_in.body_q,
                    model.body_com,
                ],
                outputs=[state_out.body_parent_f],
                device=model.device,
            )

    def _stage1_crba(self, state_aug: State):
        """Build the joint-space mass matrix of articulations due for a refresh and add drive terms."""
        model = self.model
        global_flag = 1 if ((self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0
        wp.launch(
            build_mass_update_mask,
            dim=model.articulation_count,
            inputs=[global_flag, self._mass_update_requested],
            outputs=[self.mass_update_mask],
            device=model.device,
        )
        if not global_flag:
            # Articulations refreshed between global rebuilds (reset or notifications)
            # need fresh body inertias.
            wp.launch(
                refresh_masked_body_inertia,
                dim=model.joint_count,
                inputs=[
                    self.articulation_joint_end,
                    model.joint_articulation,
                    model.joint_child,
                    self.mass_update_mask,
                    state_aug.body_q_com,
                    self.articulation_origin,
                    self.body_I_m,
                    model.body_mass,
                    model.body_inertia,
                    0,
                ],
                outputs=[state_aug.body_I_s, self._body_inertia_terms],
                device=model.device,
            )
        wp.launch(
            compute_composite_inertia,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.articulation_joint_end,
                self.mass_update_mask,
                model.joint_ancestor,
                model.joint_child,
                state_aug.body_I_s,
            ],
            outputs=[self.body_I_c],
            device=model.device,
            block_dim=128,
        )
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            if global_flag:
                self.H_by_size[size].zero_()
            wp.launch(
                crba_fill_par_dof,
                dim=int(n_arts * size),
                inputs=[
                    model.articulation_start,
                    self.articulation_dof_start,
                    self.mass_update_mask,
                    model.joint_ancestor,
                    model.joint_child,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    state_aug.joint_S_s,
                    state_aug.body_I_s if size in self._free_body_inertia_sizes else self.body_I_c,
                    self.group_to_art[size],
                    size,
                ],
                outputs=[self.H_by_size[size]],
                device=model.device,
                block_dim=128,
            )
            wp.launch(
                apply_augmented_mass_diagonal_grouped,
                dim=n_arts,
                inputs=[
                    self.group_to_art[size],
                    self.articulation_dof_start,
                    size,
                    self.articulation_max_dofs,
                    self.mass_update_mask,
                    self.aug_row_counts,
                    self.aug_row_dof_index,
                    self.aug_row_K,
                ],
                outputs=[self.H_by_size[size]],
                device=model.device,
            )
        self._mass_update_requested.zero_()
        self._force_mass_update = False

    def _stage2_cholesky_tiled(self, size: int):
        wp.launch_tiled(
            self._cholesky_kernels_by_size[size],
            dim=[self.n_arts_by_size[size]],
            inputs=[self.H_by_size[size], self.R_by_size[size], self.group_to_art[size], self.mass_update_mask],
            outputs=[self.L_by_size[size]],
            block_dim=_TILE_THREADS,
            device=self.model.device,
        )

    def _stage2_cholesky_loop(self, size: int):
        wp.launch(
            cholesky_loop,
            dim=self.n_arts_by_size[size],
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
                size,
            ],
            outputs=[self.L_by_size[size]],
            device=self.model.device,
        )

    def _stage3_trisolve_tiled(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            gather_tau_to_groups,
            dim=n_arts,
            inputs=[
                state_aug.joint_tau,
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[self.tau_by_size[size]],
            device=model.device,
        )
        wp.launch_tiled(
            self._triangular_solve_kernels_by_size[size],
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.tau_by_size[size],
                self.group_to_art[size],
            ],
            outputs=[self.qdd_by_size[size]],
            block_dim=_TILE_THREADS,
            device=model.device,
        )
        wp.launch(
            scatter_qdd_from_groups,
            dim=n_arts,
            inputs=[
                self.qdd_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_trisolve_loop(self, size: int, state_aug: State):
        wp.launch(
            trisolve_loop,
            dim=self.n_arts_by_size[size],
            inputs=[
                self.L_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
                state_aug.joint_tau,
            ],
            outputs=[state_aug.joint_qdd],
            device=self.model.device,
        )

    def _stage3_compute_v_hat(self, state_in: State, state_aug: State, dt: float, stage3_qd: wp.array):
        """Predict the unconstrained velocity ``v_hat = qd + dt * qdd``."""
        model = self.model
        wp.launch(
            compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[stage3_qd, self._kinematic_dof_mask, dt],
            outputs=[state_aug.joint_qdd, self.v_hat],
            device=model.device,
        )
        # Keep v_hat on the integrator's free-root convention. The compact launch excludes
        # every non-root joint; a live device mask handles runtime kinematic changes without
        # changing graph topology.
        if not self._free_root_joint_count:
            return
        if self._free_rigid_body_count:
            wp.launch(
                apply_free_root_velocity_corrections,
                dim=self._free_root_joint_count,
                inputs=[
                    self._free_root_joint_indices,
                    model.joint_qd_start,
                    model.joint_child,
                    self.body_to_articulation,
                    self.is_free_rigid,
                    self.art_group_idx,
                    self._kinematic_joint_mask,
                    state_in.body_q,
                    model.body_inertia,
                    self.L_by_size[6],
                    stage3_qd,
                    dt,
                ],
                outputs=[self.v_hat],
                device=model.device,
            )
        else:
            wp.launch(
                apply_free_root_transport_to_predictor,
                dim=self._free_root_joint_count,
                inputs=[
                    self._free_root_joint_indices,
                    model.joint_qd_start,
                    self._kinematic_joint_mask,
                    stage3_qd,
                    dt,
                ],
                outputs=[self.v_hat],
                device=model.device,
            )

    def _stage4_build_rows(self, state_in: State, state_aug: State, contacts: Contacts | None):
        """Allocate and fill the joint-limit, velocity-limit, contact and friction rows of every world.

        Dense rows (rows touching an articulated body) are laid out per world as
        ``[joint limits][joint velocity limits][contacts and friction]``; free-body rows
        hold ``[contacts and friction][free-body velocity limits]``. Rows past a capacity
        are dropped, counted and latched into :attr:`constraint_overflow`.
        """
        model = self.model
        max_constraints = self.dense_max_constraints
        mf_active = self._has_free_rigid_bodies
        if self._has_prescribed_response:
            self.mf_target_velocity.zero_()

        wp.launch(
            _clear_dense_row_state,
            dim=self.world_count,
            inputs=[self.slot_counter, self._row_dropped_all],
            device=model.device,
        )
        self._dense_first_rejected_slot.fill_(_ROW_SLOT_UNBOUNDED)
        self._cross_world_contacts.zero_()
        if mf_active:
            self.mf_slot_counter.zero_()
            self._mf_first_rejected_slot.fill_(_ROW_SLOT_UNBOUNDED)
            # The finalize kernels only run when contacts exist.
            self.mf_constraint_count.zero_()
            self.mf_impulses.zero_()

        is_free_rigid = self.is_free_rigid if self.is_free_rigid is not None else self._dummy_is_free_rigid
        mf_slot_counter = self.mf_slot_counter if mf_active else self._dummy_mf_slot_counter
        mf_first_rejected_slot = self._mf_first_rejected_slot if mf_active else self._dummy_mf_slot_counter

        # Rows are rebuilt every step; clear the grouped Jacobians once before any family writes.
        for size in self.size_groups:
            self.J_by_size[size].zero_()

        for size in self._joint_limit_sizes:
            n_arts = self.n_arts_by_size[size]
            wp.launch_tiled(
                self._joint_limit_warp_kernels[size],
                dim=[(n_arts + _JOINT_LIMIT_WARPS_PER_BLOCK - 1) // _JOINT_LIMIT_WARPS_PER_BLOCK],
                inputs=[
                    n_arts,
                    self.articulation_dof_start,
                    self.art_to_world,
                    self.group_to_art[size],
                    self._joint_limit_q_index,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    state_in.joint_q,
                    self.joint_limit_activation_gap,
                    max_constraints,
                ],
                outputs=[
                    self.slot_counter,
                    self.J_by_size[size],
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self.phi,
                    self.target_velocity,
                ],
                block_dim=32 * _JOINT_LIMIT_WARPS_PER_BLOCK,
                device=model.device,
            )

        if self.velocity_limit_slot is not None:
            wp.launch(
                allocate_joint_velocity_limit_slots,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_start,
                    self.articulation_dof_start,
                    self.articulation_H_rows,
                    model.joint_type,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_velocity_limit,
                    self.v_hat,
                    self.velocity_limit_activation_fraction,
                    self.art_to_world,
                    max_constraints,
                ],
                outputs=[self.velocity_limit_slot, self.velocity_limit_sign, self.slot_counter],
                device=model.device,
            )
            for size in self.size_groups:
                wp.launch(
                    populate_joint_velocity_limit_J_for_size,
                    dim=self.n_arts_by_size[size],
                    inputs=[
                        model.articulation_start,
                        self.articulation_dof_start,
                        model.joint_type,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_velocity_limit,
                        self.art_to_world,
                        self.velocity_limit_slot,
                        self.velocity_limit_sign,
                        self.group_to_art[size],
                    ],
                    outputs=[
                        self.J_by_size[size],
                        self.row_type,
                        self.row_parent,
                        self.row_mu,
                        self.phi,
                        self.target_velocity,
                    ],
                    device=model.device,
                )

        if contacts is not None and contacts.rigid_contact_max > 0:
            contact_build_threads = min(contacts.rigid_contact_max, _CONTACT_BUILD_THREAD_CAP)
            contact_geometry = [
                contacts.rigid_contact_count,
                contact_build_threads,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
            ]
            wp.launch(
                allocate_world_contact_slots,
                dim=contact_build_threads,
                inputs=[
                    contacts.rigid_contact_count,
                    contact_build_threads,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    self.body_to_articulation,
                    self.art_to_world,
                    self.articulation_response_dof_count,
                    model.body_flags,
                    self.body_has_response_dofs,
                    is_free_rigid,
                    int(mf_active),
                    max_constraints,
                    self.mf_max_constraints,
                    self._articulation_model_world,
                ],
                outputs=[
                    self.contact_world,
                    self.contact_slot,
                    self.contact_art_a,
                    self.contact_art_b,
                    self.slot_counter,
                    self.contact_path,
                    mf_slot_counter,
                    self._row_dropped_dense,
                    self._row_dropped_mf,
                    self._dense_first_rejected_slot,
                    mf_first_rejected_slot,
                    self._cross_world_contacts,
                ],
                device=model.device,
            )
            wp.launch(
                prepare_world_contact_rows,
                dim=contact_build_threads,
                inputs=[
                    *contact_geometry,
                    self.contact_world,
                    self.contact_slot,
                    self.contact_art_a,
                    self.contact_art_b,
                    self.contact_path,
                    model.shape_body,
                    state_in.body_q,
                    state_aug.body_v_s,
                    self._prescribed_articulation,
                    self.articulation_origin,
                    self.shape_material_mu,
                ],
                outputs=[
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self.phi,
                    self.target_velocity,
                ],
                device=model.device,
            )
            if self._compact_contact_jacobian:
                # Small articulations: one lane per (row, DOF) of each contact.
                contact_jacobian_workers = min(contacts.rigid_contact_max, _CONTACT_JACOBIAN_WORKER_CAP)
                for size in self.size_groups:
                    wp.launch(
                        populate_world_J_for_compact_size,
                        dim=(contact_jacobian_workers, 32),
                        inputs=[
                            contacts.rigid_contact_count,
                            contact_jacobian_workers,
                            *contact_geometry[2:],
                            self.contact_slot,
                            self.contact_art_a,
                            self.contact_art_b,
                            self.contact_path,
                            size,
                            self.articulation_response_dof_count,
                            self.art_group_idx,
                            self.articulation_dof_start,
                            self.articulation_origin,
                            self.body_response_dof_mask,
                            state_aug.joint_S_s,
                            model.shape_body,
                            state_in.body_q,
                        ],
                        outputs=[self.J_by_size[size]],
                        device=model.device,
                    )
            else:
                for size in self.size_groups:
                    wp.launch(
                        populate_world_J_for_size,
                        dim=contact_build_threads,
                        inputs=[
                            *contact_geometry,
                            self.contact_slot,
                            self.contact_art_a,
                            self.contact_art_b,
                            self.contact_path,
                            size,
                            self.articulation_response_dof_count,
                            self.art_group_idx,
                            self.articulation_dof_start,
                            self.articulation_origin,
                            self.body_to_joint,
                            model.joint_ancestor,
                            model.joint_qd_start,
                            state_aug.joint_S_s,
                            model.shape_body,
                            state_in.body_q,
                        ],
                        outputs=[self.J_by_size[size]],
                        device=model.device,
                    )

            if mf_active:
                wp.launch(
                    build_mf_contact_rows,
                    dim=contact_build_threads,
                    inputs=[
                        *contact_geometry,
                        self.contact_world,
                        self.contact_slot,
                        self.contact_path,
                        self.contact_art_a,
                        self.contact_art_b,
                        self.articulation_response_dof_count,
                        self.articulation_origin,
                        model.shape_body,
                        state_in.body_q,
                        state_aug.body_v_s,
                        self._prescribed_articulation,
                        int(self._has_prescribed_response),
                        self.shape_material_mu,
                    ],
                    outputs=[
                        self.mf_body_a,
                        self.mf_body_b,
                        self.mf_J_a,
                        self.mf_J_b,
                        self.mf_row_type,
                        self.mf_row_parent,
                        self.mf_row_mu,
                        self.mf_phi,
                        self.mf_target_velocity,
                    ],
                    device=model.device,
                )
                wp.launch(
                    finalize_mf_constraint_counts,
                    dim=self.world_count,
                    inputs=[self.mf_slot_counter, self.mf_max_constraints, 3, self._mf_first_rejected_slot],
                    outputs=[self.mf_constraint_count],
                    device=model.device,
                )

        if mf_active:
            # Contact rows precede the free-body velocity-limit rows in each world.
            wp.copy(self.mf_contact_rows_end, self.mf_slot_counter)
        if mf_active and self.rigid_velocity_limit_slot is not None:
            wp.launch(
                allocate_rigid_velocity_limit_slots,
                dim=self._free_rigid_body_count,
                inputs=[
                    self.free_rigid_body_indices,
                    self.body_to_articulation,
                    self.art_to_world,
                    is_free_rigid,
                    model.body_flags,
                    self.rigid_body_max_linear_velocity,
                    self.rigid_body_max_angular_velocity,
                    self.articulation_root_dof_start,
                    self.v_hat,
                    self.velocity_limit_activation_fraction,
                    self.mf_max_constraints,
                ],
                outputs=[self.rigid_velocity_limit_slot, self.rigid_velocity_limit_sign, self.mf_slot_counter],
                device=model.device,
            )
            wp.launch(
                populate_rigid_velocity_limit_rows,
                dim=self._free_rigid_body_count,
                inputs=[
                    self.free_rigid_body_indices,
                    self.body_to_articulation,
                    self.art_to_world,
                    is_free_rigid,
                    self.rigid_body_max_linear_velocity,
                    self.rigid_body_max_angular_velocity,
                    self.rigid_velocity_limit_slot,
                    self.rigid_velocity_limit_sign,
                ],
                outputs=[
                    self.mf_body_a,
                    self.mf_body_b,
                    self.mf_J_a,
                    self.mf_J_b,
                    self.mf_row_type,
                    self.mf_row_parent,
                    self.mf_row_mu,
                    self.mf_phi,
                ],
                device=model.device,
            )
            wp.launch(
                finalize_mf_constraint_counts,
                dim=self.world_count,
                inputs=[self.mf_slot_counter, self.mf_max_constraints, 1, self._mf_first_rejected_slot],
                outputs=[self.mf_constraint_count],
                device=model.device,
            )

        wp.launch(
            finalize_world_constraint_counts,
            dim=self.world_count,
            inputs=[self.slot_counter, max_constraints, 3, self._dense_first_rejected_slot],
            outputs=[self.constraint_count],
            device=model.device,
        )
        wp.launch(
            _finalize_constraint_status,
            dim=self.world_count + 1,
            inputs=[
                self.slot_counter,
                mf_slot_counter,
                self._row_dropped_all,
                self._cross_world_contacts,
                contacts.rigid_contact_count if contacts is not None else self._dummy_contact_count,
                contacts._reduction_overflow if contacts is not None else self._dummy_contact_count,
                self.dense_max_constraints,
                self.mf_max_constraints,
                contacts.rigid_contact_max if contacts is not None else 0,
                int(self._has_responding_global),
                self.constraint_overflow,
            ],
            device=model.device,
        )
        if self.warn_constraint_overflow:
            wp.launch(
                _warn_constraint_row_overflow,
                dim=self.world_count + 1,
                inputs=[
                    self.slot_counter,
                    self._row_dropped_dense,
                    self.dense_max_constraints,
                    mf_slot_counter,
                    self._row_dropped_mf,
                    self.mf_max_constraints,
                    int(mf_active),
                    self._cross_world_contacts,
                    self._row_overflow_warning_emitted,
                ],
                device=model.device,
            )

    def _stage4_hinv_jt_tiled(self, size: int):
        n_arts = self.n_arts_by_size[size]
        world_dof_offset = self.articulation_world_dof_offset if self._hinv_jt_writes_world else self.group_to_art[size]
        outputs = [self.Y_by_size[size], self.J_world, self.Y_world]
        if size in self._hinv_jt_diag_sizes:
            outputs.append(self.diag_by_size[size])
        wp.launch_tiled(
            self._hinv_jt_kernels_by_size[size],
            dim=[n_arts, self._hinv_jt_chunk_count_by_size[size]],
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                world_dof_offset,
                self.constraint_count,
            ],
            outputs=outputs,
            block_dim=_TILE_THREADS,
            device=self.model.device,
        )

    def _stage4_hinv_jt_par_row(self, size: int):
        n_arts = self.n_arts_by_size[size]
        world_dof_offset = self.articulation_world_dof_offset if self._hinv_jt_writes_world else self.group_to_art[size]
        wp.launch(
            hinv_jt_par_row,
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                world_dof_offset,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
                int(self._hinv_jt_writes_world),
            ],
            outputs=[self.Y_by_size[size], self.J_world, self.Y_world],
            device=self.model.device,
        )

    def _stage4_compute_matrix_free_diag(self):
        """Compute each dense row's effective-mass diagonal ``J Y``."""
        self.diag.zero_()
        if self._hinv_jt_diag_sizes:
            for size in self.size_groups:
                if size in self._hinv_jt_diag_sizes:
                    wp.launch(
                        accumulate_group_diag_worlds,
                        dim=self.world_count * self.dense_max_constraints,
                        inputs=[
                            self.diag_by_size[size],
                            self.world_response_group_art_start[size],
                            self.world_response_group_to_art[size],
                            self.art_group_idx,
                            self.constraint_count,
                            self.dense_max_constraints,
                        ],
                        outputs=[self.diag],
                        device=self.model.device,
                    )
                else:
                    self._stage4_diag_from_JY(size)
        elif self._hinv_jt_writes_world:
            wp.launch(
                diag_from_JY_world,
                dim=self.world_count * self.dense_max_constraints,
                inputs=[
                    self.constraint_count,
                    self.world_dof_count,
                    self.J_world,
                    self.Y_world,
                    self.dense_max_constraints,
                ],
                outputs=[self.diag],
                device=self.model.device,
            )
        else:
            for size in self.size_groups:
                self._stage4_diag_from_JY(size)

    def _stage4_diag_from_JY(self, size: int):
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            diag_from_JY_par_art,
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
            ],
            outputs=[self.diag],
            device=self.model.device,
        )

    def _mf_pgs_setup(self, state_aug: State, dt: float):
        """Compute free-body inverse inertias, row responses, effective masses and right-hand sides."""
        model = self.model
        wp.launch(
            compute_mf_body_Hinv,
            dim=self._free_rigid_body_count,
            inputs=[
                self.free_rigid_body_indices,
                state_aug.body_I_s,
                self.is_free_rigid,
                self.body_to_articulation,
                model.body_flags,
                self.articulation_dof_start,
                self._joint_armature_device,
            ],
            outputs=[self.mf_body_Hinv],
            device=model.device,
        )
        self.mf_rhs.zero_()
        self.mf_eff_mass_inv.zero_()
        wp.launch(
            compute_mf_effective_mass_and_rhs,
            dim=self.world_count * self.mf_max_constraints,
            inputs=[
                self.mf_constraint_count,
                self.mf_body_a,
                self.mf_body_b,
                self.mf_J_a,
                self.mf_J_b,
                self.mf_body_Hinv,
                self.mf_phi,
                self.mf_row_type,
                self.mf_target_velocity,
                int(self._has_prescribed_response),
                self.rigid_body_max_depenetration_velocity,
                self.pgs_cfm,
                self.pgs_beta,
                dt,
                self.mf_max_constraints,
            ],
            outputs=[self.mf_eff_mass_inv, self.mf_MiJt_a, self.mf_MiJt_b, self.mf_rhs],
            device=model.device,
        )

    def _stage7_update_kinematics(self, state_out: State) -> None:
        """Publish the maximal-coordinate body state of the integrated joint state."""
        eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)


@cache
def _get_joint_limit_warp_kernel(size: int, device_arch: str, warps_per_block: int) -> "wp.Kernel":
    """Build a deterministic one-warp-per-articulation joint-limit row builder."""
    _ = device_arch
    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int group_idx = block * {warps_per_block} + (threadIdx.x >> 5);
    if (group_idx >= articulation_count) return;

    const int articulation = group_to_art.data[group_idx];
    const int world = art_to_world.data[articulation];
    const int dof_start = articulation_dof_start.data[articulation];
    for (int base = 0; base < {2 * size}; base += 32) {{
        const int candidate = base + lane;
        const int local_dof = candidate >> 1;
        const int side = candidate & 1;
        const int dof = dof_start + local_dof;
        const int q_index = local_dof < {size} ? limit_q_index.data[dof] : -1;
        float phi_value = 0.0f;
        int active = 0;
        if (q_index >= 0) {{
            const float q = joint_q.data[q_index];
            const float bound = side == 0 ? joint_limit_lower.data[dof] : joint_limit_upper.data[dof];
            phi_value = side == 0 ? q - bound : bound - q;
            active = isfinite(bound) && (side == 0 ? q <= bound + activation_gap : q >= bound - activation_gap);
        }}

        const unsigned active_mask = __ballot_sync(MASK, active != 0);
        const int active_count = __popc(active_mask);
        int first_slot = 0;
        if (lane == 0 && active_count != 0)
            first_slot = atomicAdd(&world_slot_counter.data[world], active_count);
        first_slot = __shfl_sync(MASK, first_slot, 0);
        if (active != 0) {{
            const unsigned lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
            const int slot = first_slot + __popc(active_mask & lower_lanes);
            if (slot < max_constraints) {{
                J_group.data[(group_idx * max_constraints + slot) * {size} + local_dof] = side == 0 ? 1.0f : -1.0f;
                const int row = world * max_constraints + slot;
                world_row_type.data[row] = {PGS_CONSTRAINT_TYPE_JOINT_LIMIT};
                world_row_parent.data[row] = -1;
                world_row_mu.data[row] = 0.0f;
                world_phi.data[row] = phi_value;
                world_target_velocity.data[row] = 0.0f;
            }}
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def joint_limit_warp_native(
        block: int,
        articulation_count: int,
        articulation_dof_start: wp.array[int],
        art_to_world: wp.array[int],
        group_to_art: wp.array[int],
        limit_q_index: wp.array[int],
        joint_limit_lower: wp.array[float],
        joint_limit_upper: wp.array[float],
        joint_q: wp.array[float],
        activation_gap: float,
        max_constraints: int,
        world_slot_counter: wp.array[int],
        J_group: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_phi: wp.array2d[float],
        world_target_velocity: wp.array2d[float],
    ): ...

    def joint_limit_warp_template(
        articulation_count: int,
        articulation_dof_start: wp.array[int],
        art_to_world: wp.array[int],
        group_to_art: wp.array[int],
        limit_q_index: wp.array[int],
        joint_limit_lower: wp.array[float],
        joint_limit_upper: wp.array[float],
        joint_q: wp.array[float],
        activation_gap: float,
        max_constraints: int,
        world_slot_counter: wp.array[int],
        J_group: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_phi: wp.array2d[float],
        world_target_velocity: wp.array2d[float],
    ):
        block, _lane = wp.tid()
        joint_limit_warp_native(
            block,
            articulation_count,
            articulation_dof_start,
            art_to_world,
            group_to_art,
            limit_q_index,
            joint_limit_lower,
            joint_limit_upper,
            joint_q,
            activation_gap,
            max_constraints,
            world_slot_counter,
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_phi,
            world_target_velocity,
        )

    name = f"build_joint_limit_rows_warp_{size}_{warps_per_block}"
    joint_limit_warp_template.__name__ = name
    joint_limit_warp_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(joint_limit_warp_template)


@cache
def _get_hinv_jt_plain_kernel(
    n_dofs: int,
    max_constraints: int,
    device_arch: str,
    tile_threads: int = 64,
    constraint_chunk_size: int | None = None,
    write_world: bool = False,
    write_group: bool = True,
) -> "wp.Kernel":
    """Build a specialized H^-1*J^T kernel without a diagonal output.

    Keep this signature structurally separate from :func:`_get_hinv_jt_kernel`.
    Warp retains unused formal outputs and tiled reductions even behind a false
    compile-time flag, which increases work for layouts that do not consume them.
    """
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))
    chunk_size = max_constraints if constraint_chunk_size is None else int(constraint_chunk_size)
    if chunk_size <= 0 or chunk_size > max_constraints:
        raise ValueError("constraint_chunk_size must be in [1, max_constraints]")
    TILE_CONSTRAINTS_LOCAL = wp.constant(chunk_size)
    BOUNDS_CHECK = max_constraints % chunk_size != 0
    WRITE_WORLD = wp.constant(1 if write_world else 0)
    WRITE_GROUP = wp.constant(1 if write_group else 0)

    def hinv_jt_tiled_template(
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
    ):
        idx, chunk = wp.tid()
        art = group_to_art[idx]
        world = art_to_world[art]
        n_constraints = world_constraint_count[world]
        row_start = chunk * TILE_CONSTRAINTS_LOCAL

        if row_start >= n_constraints:
            return

        L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        J_tile = wp.tile_load(
            J_group[idx],
            shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL),
            offset=(row_start, 0),
            bounds_check=BOUNDS_CHECK,
        )
        Jt_tile = wp.tile_transpose(J_tile)
        Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)
        Lt_tile = wp.tile_transpose(L_tile)
        X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)
        Y_out_tile = wp.tile_transpose(X_tile)

        if WRITE_GROUP != 0:
            wp.tile_store(Y_group[idx], Y_out_tile, offset=(row_start, 0), bounds_check=BOUNDS_CHECK)

        if WRITE_WORLD != 0:
            dof_offset = articulation_world_dof_offset[art]
            wp.tile_store(J_world[world], J_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)
            wp.tile_store(Y_world[world], Y_out_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)

    suffix = "_world" if write_world else ""
    suffix += "_nogroup" if not write_group else ""
    hinv_jt_tiled_template.__name__ = f"hinv_jt_tiled_{n_dofs}_{max_constraints}_c{chunk_size}_bd{tile_threads}{suffix}"
    hinv_jt_tiled_template.__qualname__ = hinv_jt_tiled_template.__name__
    return wp.kernel(enable_backward=False, module="unique")(hinv_jt_tiled_template)


@cache
def _get_hinv_jt_kernel(
    n_dofs: int,
    max_constraints: int,
    device_arch: str,
    tile_threads: int = 64,
    constraint_chunk_size: int | None = None,
    write_world: bool = False,
    write_group: bool = True,
    compute_diag: bool = False,
) -> "wp.Kernel":
    """Build specialized H^-1*J^T kernel for given dimensions.

    Solves Y = H^-1 * J^T using tiled Cholesky solve:
      L * L^T * Y = J^T
      => L * Z = J^T (forward solve)
      => L^T * Y = Z (backward solve)
    """
    # Create compile-time constants via closure
    # Convert to Python int to ensure wp.constant() accepts them
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))
    chunk_size = max_constraints if constraint_chunk_size is None else int(constraint_chunk_size)
    if chunk_size <= 0 or chunk_size > max_constraints:
        raise ValueError("constraint_chunk_size must be in [1, max_constraints]")
    TILE_CONSTRAINTS_LOCAL = wp.constant(chunk_size)
    BOUNDS_CHECK = max_constraints % chunk_size != 0
    WRITE_WORLD = wp.constant(1 if write_world else 0)
    WRITE_GROUP = wp.constant(1 if write_group else 0)
    COMPUTE_DIAG = wp.constant(1 if compute_diag else 0)

    def hinv_jt_tiled_template(
        L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        J_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        # output
        Y_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        diag_group: wp.array2d[float],
    ):
        idx, chunk = wp.tid()
        art = group_to_art[idx]
        world = art_to_world[art]
        n_constraints = world_constraint_count[world]
        row_start = chunk * TILE_CONSTRAINTS_LOCAL

        if row_start >= n_constraints:
            return

        # Load L (Cholesky factor) and J (Jacobian rows)
        L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        J_tile = wp.tile_load(
            J_group[idx],
            shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL),
            offset=(row_start, 0),
            bounds_check=BOUNDS_CHECK,
        )

        # Solve L * Z = J^T (forward substitution)
        # J_tile is (max_c x n_dofs), J^T is (n_dofs x max_c)
        Jt_tile = wp.tile_transpose(J_tile)
        Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)

        # Solve L^T * Y = Z (backward substitution)
        Lt_tile = wp.tile_transpose(L_tile)
        X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)

        # Store Y = H^-1 * J^T (transpose back to row layout)
        Y_out_tile = wp.tile_transpose(X_tile)
        if WRITE_GROUP != 0:
            wp.tile_store(Y_group[idx], Y_out_tile, offset=(row_start, 0), bounds_check=BOUNDS_CHECK)

        if COMPUTE_DIAG != 0:
            diag_tile = wp.tile_sum(wp.tile_map(wp.mul, J_tile, Y_out_tile), axis=1)
            wp.tile_store(diag_group[idx], diag_tile, offset=row_start, bounds_check=BOUNDS_CHECK)

        if WRITE_WORLD != 0:
            dof_offset = articulation_world_dof_offset[art]
            wp.tile_store(J_world[world], J_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)
            wp.tile_store(Y_world[world], Y_out_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)

    suffix = "_world" if write_world else ""
    suffix += "_nogroup" if not write_group else ""
    suffix += "_diag" if compute_diag else ""
    hinv_jt_tiled_template.__name__ = f"hinv_jt_tiled_{n_dofs}_{max_constraints}_c{chunk_size}_bd{tile_threads}{suffix}"
    hinv_jt_tiled_template.__qualname__ = hinv_jt_tiled_template.__name__
    return wp.kernel(enable_backward=False, module="unique")(hinv_jt_tiled_template)


@cache
def _get_cholesky_kernel(n_dofs: int, device_arch: str, tile_threads: int = 64) -> "wp.Kernel":
    """Build specialized Cholesky kernel for given DOF count.

    Computes L such that H + diag(armature) = L * L^T.
    """
    # Convert to Python int to ensure wp.constant() accepts them
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))

    def cholesky_tiled_template(
        H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        R_group: wp.array2d[float],  # [n_arts, n_dofs] armature
        group_to_art: wp.array[int],
        mass_update_mask: wp.array[int],
        # output
        L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
    ):
        idx = wp.tid()
        art = group_to_art[idx]

        if mass_update_mask[art] == 0:
            return

        # Load H and armature
        H_tile = wp.tile_load(H_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        armature = wp.tile_load(R_group[idx], shape=(TILE_DOF_LOCAL,), bounds_check=False)

        # Add armature to diagonal
        H_tile = wp.tile_diag_add(H_tile, armature)

        # Compute Cholesky factorization
        L_tile = wp.tile_cholesky(H_tile)

        # Store result
        wp.tile_store(L_group[idx], L_tile)

    cholesky_tiled_template.__name__ = f"cholesky_tiled_{n_dofs}_bd{tile_threads}"
    cholesky_tiled_template.__qualname__ = f"cholesky_tiled_{n_dofs}_bd{tile_threads}"
    return wp.kernel(enable_backward=False, module="unique")(cholesky_tiled_template)


@cache
def _get_triangular_solve_kernel(n_dofs: int, device_arch: str, tile_threads: int = 64) -> "wp.Kernel":
    """Build specialized triangular solve kernel for given DOF count.

    Solves L * L^T * x = b for x using tiled forward and backward substitution.
    """
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))

    def trisolve_tiled_template(
        L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
        group_to_art: wp.array[int],
        qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    ):
        idx = wp.tid()
        L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        tau_tile = wp.tile_load(tau_group[idx], shape=(TILE_DOF_LOCAL, 1), bounds_check=False)

        # Forward substitution: L * z = tau
        z_tile = wp.tile_lower_solve(L_tile, tau_tile)

        # Backward substitution: L^T * qdd = z
        Lt_tile = wp.tile_transpose(L_tile)
        qdd_tile = wp.tile_upper_solve(Lt_tile, z_tile)

        wp.tile_store(qdd_group[idx], qdd_tile)

    trisolve_tiled_template.__name__ = f"trisolve_tiled_{n_dofs}_bd{tile_threads}"
    trisolve_tiled_template.__qualname__ = f"trisolve_tiled_{n_dofs}_bd{tile_threads}"
    return wp.kernel(enable_backward=False, module="unique")(trisolve_tiled_template)


@cache
def _get_pack_mf_meta_kernel(mf_max_constraints: int, device_arch: str) -> "wp.Kernel":
    """Build a kernel to pack MF constraint metadata into int4 structs.

    Packs dof_a, dof_b, eff_mass_inv, rhs, row_type, row_parent into
    4 contiguous int32s per constraint for 128-bit coalesced loads.
    """
    M_MF = mf_max_constraints

    snippet = f"""
#if defined(__CUDA_ARCH__)
    int lane = threadIdx.x;
    int m_mf = mf_constraint_count.data[world];
    int off_mf = world * {M_MF};
    int off_meta = off_mf * 4;

    for (int i = lane; i < m_mf; i += 32) {{
        int da = mf_dof_a.data[off_mf + i];
        int db = mf_dof_b.data[off_mf + i];
        float diag = mf_eff_mass_inv.data[off_mf + i];
        float rhs_val = mf_rhs.data[off_mf + i];
        int rt = mf_row_type.data[off_mf + i];
        int par = mf_row_parent.data[off_mf + i];

        int4 packed;
        packed.x = (da << 16) | (db & 0xFFFF);
        packed.y = __float_as_int(diag);
        packed.z = __float_as_int(rhs_val);
        packed.w = rt | (par << 16);
        *reinterpret_cast<int4*>(&mf_meta.data[off_meta + i * 4]) = packed;
    }}
#endif
"""

    @wp.func_native(snippet)
    def pack_mf_meta_native(
        world: int,
        mf_constraint_count: wp.array[int],
        mf_dof_a: wp.array2d[int],
        mf_dof_b: wp.array2d[int],
        mf_eff_mass_inv: wp.array2d[float],
        mf_rhs: wp.array2d[float],
        mf_row_type: wp.array2d[int],
        mf_row_parent: wp.array2d[int],
        mf_meta: wp.array2d[int],
    ): ...

    def pack_mf_meta_template(
        mf_constraint_count: wp.array[int],
        mf_dof_a: wp.array2d[int],
        mf_dof_b: wp.array2d[int],
        mf_eff_mass_inv: wp.array2d[float],
        mf_rhs: wp.array2d[float],
        mf_row_type: wp.array2d[int],
        mf_row_parent: wp.array2d[int],
        mf_meta: wp.array2d[int],
    ):
        world, _lane = wp.tid()
        pack_mf_meta_native(
            world,
            mf_constraint_count,
            mf_dof_a,
            mf_dof_b,
            mf_eff_mass_inv,
            mf_rhs,
            mf_row_type,
            mf_row_parent,
            mf_meta,
        )

    name = f"pack_mf_meta_{mf_max_constraints}"
    pack_mf_meta_template.__name__ = name
    pack_mf_meta_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pack_mf_meta_template)


@cache
def _get_pgs_solve_mf_gs_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    has_dense_velocity_limit_rows: bool,
    shared_metadata: bool,
) -> "wp.Kernel":
    """Build the fused matrix-free projected Gauss-Seidel kernel for one solver shape.

    One warp (32 threads) solves one world. Every iteration sweeps, in order:

    1. the dense rows (joint limits, contacts and friction of articulated bodies): a
       warp-parallel ``J v`` over the world's ``D`` response DOFs and a ``Y`` update of
       the world velocity, software-pipelined one row ahead;
    2. the free-body contact and friction rows: lanes 0-5 handle body A and lanes 6-11
       body B;
    3. the dense joint velocity-limit rows and the free-body velocity-limit rows, so
       velocity limits have the last word in each iteration.

    Friction rows follow their normal row and solve the two tangent impulses together on
    the Coulomb disk of the current normal impulse (``FRICTION_PAIR_CUDA``). A
    world stops early after an exactly stationary sweep: neither its impulses nor its
    velocity changed, so every later sweep would repeat the same operations; the
    iteration count stays an upper bound and no convergence tolerance is introduced.

    The world velocity and impulses stay in shared memory. With ``shared_metadata`` the
    per-row metadata does as well; larger shapes stream it from global memory to keep
    occupancy.

    Args:
        max_constraints: Dense row capacity ``M_D`` per world.
        mf_max_constraints: Free-body row capacity ``M_MF`` per world.
        max_world_dofs: Response DOFs ``D`` per world.
        device_arch: CUDA architecture; part of the cache key.
        has_dense_velocity_limit_rows: Emit the dense velocity-limit pass.
        shared_metadata: Keep the dense row metadata in shared memory.
    """
    M_D = max_constraints
    M_MF = mf_max_constraints
    D = max_world_dofs
    ELEMS_PER_LANE = (D + 31) // 32
    type_mask = _DENSE_META_ROW_TYPE_MASK
    type_bits = _DENSE_META_ROW_TYPE_BITS

    def lanes(template: str) -> str:
        """Repeat a per-lane snippet for each strided DOF element of the lane."""
        return "\n".join(
            template.replace("{k}", str(k)).replace("{d}", f"lane + {k * 32}" if k else "lane")
            for k in range(ELEMS_PER_LANE)
        )

    dense_pipe_decl = lanes("        float pre_dJ_{k} = 0.0f, pre_dY_{k} = 0.0f;")
    dense_prefetch_init = lanes(
        "            if ({d} < " + str(D) + ") { pre_dJ_{k} = J_world.data[jy_world_base + {d}]; "
        "pre_dY_{k} = Y_world.data[jy_world_base + {d}]; }"
    )
    dense_consume = lanes("            float cur_dJ_{k} = pre_dJ_{k}, cur_dY_{k} = pre_dY_{k};")
    dense_prefetch_next = lanes(
        "                if ({d} < " + str(D) + ") { pre_dJ_{k} = J_world.data[next_jy_base + {d}]; "
        "pre_dY_{k} = Y_world.data[next_jy_base + {d}]; }"
    )
    dense_dot = lanes("            if ({d} < " + str(D) + ") my_sum += cur_dJ_{k} * s_v[{d}];")
    dense_v_update = lanes("                if ({d} < " + str(D) + ") s_v[{d}] += cur_dY_{k} * delta_impulse;")

    dense_velocity_limit_pass = (
        f"""
        for (int i = 0; i < m_dense; i++) {{
            if ((s_meta_dense[i] & {type_mask}) != {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) continue;
            float denom = s_diag_dense[i];
            if (denom <= 0.0f) continue;
            int row_base = jy_world_base + i * {D};
            float my_sum = 0.0f;
            for (int d = lane; d < {D}; d += 32) my_sum += J_world.data[row_base + d] * s_v[d];
            float jv = warp_sum(my_sum);
            // Stateless projection onto the velocity box: each visit applies only the
            // impulse needed for the current overshoot.
            float residual = jv + s_rhs_dense[i];
            float delta_impulse = residual < 0.0f ? -residual / denom : 0.0f;
            if (lane == 0) s_lam_dense[i] = delta_impulse;
            if (delta_impulse != 0.0f) {{
                iteration_changed = 1;
                for (int d = lane; d < {D}; d += 32) s_v[d] += Y_world.data[row_base + d] * delta_impulse;
            }}
            __syncwarp();
        }}"""
        if has_dense_velocity_limit_rows
        else ""
    )

    snippet = f"""
#if defined(__CUDA_ARCH__)
    const unsigned MASK = 0xFFFFFFFF;
    int lane = threadIdx.x;

    int m_dense = world_constraint_count.data[world];
    int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
    // Free-body rows are laid out as [contacts and friction][velocity limits].
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;

    int dof_map_base = world * {D};
    int off_dense = world * {M_D};
    int off_mf = world * {M_MF};
    int off_meta = off_mf * 4;
    int jy_world_base = world * {M_D} * {D};
    int mf6_base = world * {M_MF} * 6;

    __shared__ float s_v[{D}];
    __shared__ float s_lam_dense[{M_D}];
    __shared__ float s_rhs_dense[{M_D}];
    __shared__ float s_diag_dense[{M_D}];
    // Low bits: row type. Remaining bits: parent + 1 (-1 maps to 0).
    __shared__ int   s_meta_dense[{M_D}];
    __shared__ float s_mu_dense[{M_D}];
    __shared__ float s_lam_mf[{M_MF}];

    for (int i = lane; i < m_dense; i += 32) {{
        s_lam_dense[i] = world_impulses.data[off_dense + i];
        s_rhs_dense[i] = rhs_bias.data[off_dense + i];
        s_diag_dense[i] = world_diag.data[off_dense + i];
        int row_type = world_row_type.data[off_dense + i];
        int row_parent = world_row_parent.data[off_dense + i];
        s_meta_dense[i] = (row_type & {type_mask}) | ((row_parent + 1) << {type_bits});
        s_mu_dense[i] = world_row_mu.data[off_dense + i];
    }}
    for (int i = lane; i < m_mf; i += 32) s_lam_mf[i] = mf_impulses.data[off_mf + i];
    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}
    __syncwarp();

{dense_pipe_decl}

    for (int iter = 0; iter < iterations; iter++) {{
        int iteration_changed = 0;

        // Dense rows, prefetching the response of the next row.
        if (m_dense > 0) {{
{dense_prefetch_init}
        }}
        for (int i = 0; i < m_dense; i++) {{
{dense_consume}
            if (i + 1 < m_dense) {{
                int next_jy_base = jy_world_base + (i + 1) * {D};
{dense_prefetch_next}
            }}

            int row_type = s_meta_dense[i] & {type_mask};
            if (row_type == {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) continue;
            float denom = s_diag_dense[i];
            if (denom <= 0.0f && row_type != {int(PGS_CONSTRAINT_TYPE_FRICTION)}) continue;

            float my_sum = 0.0f;
{dense_dot}
            float jv = warp_sum(my_sum);

            float old_impulse = s_lam_dense[i];
            float residual = jv + s_rhs_dense[i];
            float delta = denom > 0.0f ? -residual / denom : 0.0f;
            float new_impulse = old_impulse + omega * delta;

            if (row_type == {int(PGS_CONSTRAINT_TYPE_FRICTION)}) {{
                // The first tangent row solves both tangents of its contact; the second
                // row's impulse was written by the first.
                int parent_idx = (s_meta_dense[i] >> {type_bits}) - 1;
                if (i != parent_idx + 1) {{
                    new_impulse = old_impulse;
                }} else {{
                    int sib = parent_idx + 2;
                    int sib_row_base = jy_world_base + sib * {D};
                    float radius = fmaxf(s_mu_dense[i] * s_lam_dense[parent_idx], 0.0f);
                    float sibling_residual = 0.0f;
                    float cross = 0.0f;
                    for (int d = lane; d < {D}; d += 32) {{
                        sibling_residual += J_world.data[sib_row_base + d] * s_v[d];
                        cross += J_world.data[jy_world_base + i * {D} + d] * Y_world.data[sib_row_base + d];
                    }}
                    sibling_residual = warp_sum(sibling_residual) + s_rhs_dense[sib];
                    cross = warp_sum(cross);
                    float2 pair = friction_pair_candidate(denom, cross, s_diag_dense[sib],
                        residual, sibling_residual, old_impulse, s_lam_dense[sib], radius, omega);
                    float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                    float scale = mag > radius ? radius / mag : 1.0f;
                    new_impulse = pair.x * scale;
                    float sib_delta = pair.y * scale - s_lam_dense[sib];
                    s_lam_dense[sib] = pair.y * scale;
                    if (sib_delta != 0.0f) {{
                        iteration_changed = 1;
                        for (int d = lane; d < {D}; d += 32) s_v[d] += Y_world.data[sib_row_base + d] * sib_delta;
                    }}
                }}
            }} else if (new_impulse < 0.0f) {{
                // Contact and joint-limit rows are unilateral.
                new_impulse = 0.0f;
            }}
            float delta_impulse = new_impulse - old_impulse;
            s_lam_dense[i] = new_impulse;
            if (delta_impulse != 0.0f) {{
                iteration_changed = 1;
{dense_v_update}
            }}
            __syncwarp();
        }}

        // Free-body contact and friction rows, prefetching the next row.
        int4 pre_meta;
        float pre_Ja = 0.0f, pre_Jb = 0.0f;
        float pre_MiJta = 0.0f, pre_MiJtb = 0.0f;
        if (mf_contact_end > 0) {{
            pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta]);
            if (lane < 6) {{
                pre_Ja = mf_J_a.data[mf6_base + lane];
                pre_MiJta = mf_MiJt_a.data[mf6_base + lane];
            }}
            if (lane >= 6 && lane < 12) {{
                pre_Jb = mf_J_b.data[mf6_base + lane - 6];
                pre_MiJtb = mf_MiJt_b.data[mf6_base + lane - 6];
            }}
        }}
        for (int i = 0; i < mf_contact_end; i++) {{
            int4 meta = pre_meta;
            float cur_Ja = pre_Ja;
            float cur_Jb = pre_Jb;
            float cur_MiJta = pre_MiJta;
            float cur_MiJtb = pre_MiJtb;
            if (i + 1 < mf_contact_end) {{
                int next_mf6 = mf6_base + (i + 1) * 6;
                pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + (i + 1) * 4]);
                if (lane < 6) {{
                    pre_Ja = mf_J_a.data[next_mf6 + lane];
                    pre_MiJta = mf_MiJt_a.data[next_mf6 + lane];
                }}
                if (lane >= 6 && lane < 12) {{
                    pre_Jb = mf_J_b.data[next_mf6 + lane - 6];
                    pre_MiJtb = mf_MiJt_b.data[next_mf6 + lane - 6];
                }}
            }}

            int dof_a = meta.x >> 16;
            int dof_b = (meta.x << 16) >> 16;
            float mf_diag = __int_as_float(meta.y);
            int packed_tp = meta.w;
            int mf_rt = packed_tp & 0xFFFF;
            int mf_par = packed_tp >> 16;
            if (mf_rt == {int(PGS_CONSTRAINT_TYPE_FRICTION)} && i != mf_par + 1) continue;
            if (mf_diag <= 0.0f && mf_rt != {int(PGS_CONSTRAINT_TYPE_FRICTION)}) continue;
            float radius = 0.0f;
            if (mf_rt == {int(PGS_CONSTRAINT_TYPE_FRICTION)}) {{
                radius = fmaxf(mf_row_mu.data[off_mf + i] * s_lam_mf[mf_par], 0.0f);
                // A zero disk with no carried impulse cannot change the velocity.
                if (radius == 0.0f && s_lam_mf[i] == 0.0f && s_lam_mf[i + 1] == 0.0f) continue;
            }}

            float my_sum = 0.0f;
            if (lane < 6 && dof_a >= 0) my_sum = cur_Ja * s_v[dof_a + lane];
            if (lane >= 6 && lane < 12 && dof_b >= 0) my_sum = cur_Jb * s_v[dof_b + lane - 6];
            float jv = warp_sum(my_sum);

            float residual = jv + __int_as_float(meta.z);
            float old_impulse = s_lam_mf[i];
            float delta = -residual * mf_diag;
            float new_impulse = old_impulse + omega * delta;
            if (mf_rt == {int(PGS_CONSTRAINT_TYPE_CONTACT)}) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
            }} else if (mf_rt == {int(PGS_CONSTRAINT_TYPE_FRICTION)}) {{
                int sib = mf_par + 2;
                int sib_mf6 = mf6_base + sib * 6;
                float2 pair = make_float2(0.0f, 0.0f);
                // Zero load gives a zero disk; avoid the unused tangent reductions.
                if (radius > 0.0f) {{
                    float sibling_residual = 0.0f;
                    float cross = 0.0f;
                    if (lane < 6 && dof_a >= 0) {{
                        sibling_residual = mf_J_a.data[sib_mf6 + lane] * s_v[dof_a + lane];
                        cross = cur_Ja * mf_MiJt_a.data[sib_mf6 + lane];
                    }}
                    if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                        sibling_residual = mf_J_b.data[sib_mf6 + lane - 6] * s_v[dof_b + lane - 6];
                        cross = cur_Jb * mf_MiJt_b.data[sib_mf6 + lane - 6];
                    }}
                    sibling_residual = warp_sum(sibling_residual) + __int_as_float(mf_meta.data[off_meta + sib * 4 + 2]);
                    cross = warp_sum(cross);
                    float inv_sib = __int_as_float(mf_meta.data[off_meta + sib * 4 + 1]);
                    pair = friction_pair_candidate(mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f, cross,
                        inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f, residual, sibling_residual, old_impulse,
                        s_lam_mf[sib], radius, omega);
                }}
                float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                float scale = mag > radius ? radius / mag : 1.0f;
                new_impulse = pair.x * scale;
                float sib_delta = pair.y * scale - s_lam_mf[sib];
                s_lam_mf[sib] = pair.y * scale;
                if (sib_delta != 0.0f) {{
                    iteration_changed = 1;
                    if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                    if (lane >= 6 && lane < 12 && dof_b >= 0)
                        s_v[dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                }}
            }}
            float delta_impulse = new_impulse - old_impulse;
            s_lam_mf[i] = new_impulse;
            if (delta_impulse != 0.0f) {{
                iteration_changed = 1;
                if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += cur_MiJta * delta_impulse;
                if (lane >= 6 && lane < 12 && dof_b >= 0) s_v[dof_b + lane - 6] += cur_MiJtb * delta_impulse;
            }}
            __syncwarp();
        }}

        // Velocity limits last: dense joint velocity limits, then free-body velocity limits.
{dense_velocity_limit_pass}
        for (int i = mf_contact_end; i < m_mf; i++) {{
            int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + i * 4]);
            if ((meta.w & 0xFFFF) != {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) continue;
            int dof_a = meta.x >> 16;
            int dof_b = (meta.x << 16) >> 16;
            float mf_diag = __int_as_float(meta.y);
            if (mf_diag <= 0.0f) continue;
            int row_mf6 = mf6_base + i * 6;
            float my_sum = 0.0f;
            if (lane < 6 && dof_a >= 0) my_sum = mf_J_a.data[row_mf6 + lane] * s_v[dof_a + lane];
            if (lane >= 6 && lane < 12 && dof_b >= 0) my_sum = mf_J_b.data[row_mf6 + lane - 6] * s_v[dof_b + lane - 6];
            float residual = warp_sum(my_sum) + __int_as_float(meta.z);
            float delta_impulse = residual < 0.0f ? -residual * mf_diag : 0.0f;
            if (lane == 0) s_lam_mf[i] = delta_impulse;
            if (delta_impulse != 0.0f) {{
                iteration_changed = 1;
                if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += mf_MiJt_a.data[row_mf6 + lane] * delta_impulse;
                if (lane >= 6 && lane < 12 && dof_b >= 0)
                    s_v[dof_b + lane - 6] += mf_MiJt_b.data[row_mf6 + lane - 6] * delta_impulse;
            }}
            __syncwarp();
        }}

        // An exactly stationary sweep is a fixed point; later sweeps are redundant.
        if (__ballot_sync(MASK, iteration_changed != 0) == 0u) break;
    }}

    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) v_out.data[global_dof] = s_v[d];
    }}
    for (int i = lane; i < m_dense; i += 32) world_impulses.data[off_dense + i] = s_lam_dense[i];
    for (int i = lane; i < m_mf; i += 32) mf_impulses.data[off_mf + i] = s_lam_mf[i];
#endif
"""

    if not shared_metadata:
        # Larger shapes stream the read-only row metadata from global memory instead of
        # holding it in shared memory, which would otherwise limit occupancy. Values are
        # identical; only the storage class changes.
        for sname, gname in (
            ("s_rhs_dense", "rhs_bias"),
            ("s_diag_dense", "world_diag"),
            ("s_mu_dense", "world_row_mu"),
        ):
            snippet = re.sub(rf"    __shared__ float\s+{sname}\[{M_D}\];\n", "", snippet)
            snippet = re.sub(rf"\s*{sname}\[i\] = {gname}\.data\[off_dense \+ i\];", "", snippet, count=1)
            snippet = re.sub(rf"{sname}\[([^\]]*)\]", rf"{gname}.data[off_dense + (\1)]", snippet)
        snippet = re.sub(r"\s*// Low bits[^\n]*\n\s*__shared__ int   s_meta_dense\[\d+\];\n", "\n", snippet)
        snippet = re.sub(
            r"\s*int row_type = world_row_type\.data\[off_dense \+ i\];"
            r"\s*int row_parent = world_row_parent\.data\[off_dense \+ i\];"
            r"\s*s_meta_dense\[i\] = [^\n]*\n",
            "\n",
            snippet,
        )
        snippet = re.sub(
            r"\(s_meta_dense\[i\] >> \d+\) - 1",
            "world_row_parent.data[off_dense + i]",
            snippet,
        )
        snippet = re.sub(r"s_meta_dense\[i\] & \d+", "world_row_type.data[off_dense + i]", snippet)
        if "s_meta_dense" in snippet:
            raise RuntimeError("streaming the dense row metadata left a reference to the shared array")

    helpers = (
        FRICTION_PAIR_CUDA
        + """
    auto warp_sum = [&](float value) {
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xFFFFFFFF, value, offset);
        return __shfl_sync(0xFFFFFFFF, value, 0);
    };
"""
    )
    snippet = snippet.replace("#if defined(__CUDA_ARCH__)", "#if defined(__CUDA_ARCH__)\n" + helpers, 1)

    @wp.func_native(snippet)
    def pgs_solve_mf_gs_native(
        world: int,
        world_constraint_count: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        v_out: wp.array[float],
    ): ...

    def pgs_solve_mf_gs(
        world_constraint_count: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        v_out: wp.array[float],
    ):
        world, _lane = wp.tid()
        pgs_solve_mf_gs_native(
            world,
            world_constraint_count,
            world_dof_indices,
            rhs_bias,
            world_diag,
            world_impulses,
            J_world,
            Y_world,
            world_row_type,
            world_row_parent,
            world_row_mu,
            mf_constraint_count,
            mf_contact_rows_end,
            mf_meta,
            mf_impulses,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_mu,
            iterations,
            omega,
            v_out,
        )

    name = (
        f"pgs_solve_mf_gs_{max_constraints}_{mf_max_constraints}_{max_world_dofs}"
        f"_vlim{int(has_dense_velocity_limit_rows)}{'' if shared_metadata else '_gmeta'}"
    )
    pgs_solve_mf_gs.__name__ = name
    pgs_solve_mf_gs.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_gs)
