# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import re
import warnings
from dataclasses import dataclass
from functools import cache
from typing import ClassVar, Literal

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
    build_propagation_body_map,
    build_propagation_contact_rows,
    cholesky_loop,
    compute_com_transforms,
    compute_composite_inertia,
    compute_contact_linear_force_from_impulses,
    compute_mf_body_Hinv,
    compute_mf_effective_mass_and_rhs,
    compute_mf_world_dof_offsets,
    compute_propagation_body_com_rel,
    compute_propagation_effective_mass_and_rhs,
    compute_propagation_tree_body_response_for_size,
    compute_spatial_inertia,
    compute_velocity_predictor,
    compute_world_contact_bias,
    copy_free_rigid_propagation_body_response,
    crba_fill_par_dof,
    diag_from_JY_par_art,
    diag_from_JY_world,
    eval_rigid_fk_id,
    eval_rigid_tau,
    eval_rigid_tau_and_augmented_drives,
    factor_propagation_tree_for_size,
    finalize_mf_constraint_counts,
    finalize_world_constraint_counts,
    finalize_world_diag_cfm,
    flatten_propagation_joint_S,
    flush_propagation_free_body_qd_to_vout,
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
    propagate_tree_impulses_for_size,
    refine_same_articulation_propagation_rows,
    refresh_masked_body_inertia,
    refresh_propagation_free_body_qd_from_vout,
    refresh_propagation_tree_body_qd_for_size,
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
# One-warp worlds packed per block of the propagation row sweep, which uses no shared memory.
_PROPAGATION_WORLDS_PER_BLOCK = 2
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
    # Judge MuJoCo equality rows first: a row whose link names the wrong entity is reported as
    # an unenforced equality rather than through whatever entity it happens to name.
    _validate_equality_constraints(model)
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


def _unprojected_equality_constraints(model: Model) -> np.ndarray:
    """Return the enabled MuJoCo equality rows that no Newton loop joint or mimic enforces.

    The MJCF and USD importers convert MuJoCo equalities to Newton loop joints or mimic
    constraints by default and keep the original row in ``model.mujoco.equality_constraint_*``
    with a ``target_kind`` / ``target`` link to the projected entity. Such a row is enforced
    (or rejected) through that entity. A row without a valid link is MuJoCo-only physics.

    A link is valid only when the target is structurally the row's projection, as the importers
    build it: a CONNECT or WELD row links a ball or fixed joint outside any articulation that
    joins the row's two bodies (or its body and the world), and a JOINT row links a mimic
    constraint between the row's two joints. Anchors and coefficients are not compared.
    """
    from ..mujoco.enums import EqType  # noqa: PLC0415
    from ..mujoco.equality import MjcEqualityTargetKind  # noqa: PLC0415

    mujoco = getattr(model, "mujoco", None)
    count = int(getattr(mujoco, "equality_constraint_count", 0) or 0) if mujoco is not None else 0
    enabled = getattr(mujoco, "equality_constraint_enabled", None) if count else None
    if enabled is None:
        return np.zeros(0, dtype=np.int32)
    enabled = enabled.numpy().astype(bool)
    target_kind = getattr(mujoco, "equality_constraint_target_kind", None)
    target = getattr(mujoco, "equality_constraint_target", None)
    projected = np.zeros(count, dtype=bool)
    if target_kind is not None and target is not None:
        target_kind = target_kind.numpy()
        target = target.numpy()

        def field(name: str) -> np.ndarray:
            values = getattr(mujoco, f"equality_constraint_{name}", None)
            return values.numpy() if values is not None else np.full(count, -2, dtype=np.int32)

        eq_type = field("type")
        body1, body2 = field("body1"), field("body2")
        joint1, joint2 = field("joint1"), field("joint2")
        joint_count = int(model.joint_count)
        joint_type = model.joint_type.numpy() if joint_count else None
        joint_parent = model.joint_parent.numpy() if joint_count else None
        joint_child = model.joint_child.numpy() if joint_count else None
        joint_articulation = (
            model.joint_articulation.numpy() if joint_count and model.joint_articulation is not None else None
        )
        mimic_count = int(getattr(model, "constraint_mimic_count", 0) or 0)
        mimic_joint0 = model.constraint_mimic_joint0.numpy() if mimic_count else None
        mimic_joint1 = model.constraint_mimic_joint1.numpy() if mimic_count else None
        loop_joint_type = {int(EqType.CONNECT): int(JointType.BALL), int(EqType.WELD): int(JointType.FIXED)}
        for row in range(count):
            kind, index = int(target_kind[row]), int(target[row])
            if kind == int(MjcEqualityTargetKind.JOINT) and 0 <= index < joint_count:
                expected_type = loop_joint_type.get(int(eq_type[row]))
                in_tree = joint_articulation is not None and int(joint_articulation[index]) >= 0
                # The importers make body1 the parent and body2 the child, or the world the
                # parent of body1; compare the endpoints as a set so either order is accepted.
                rows_bodies = {int(body1[row]), int(body2[row])}
                joint_bodies = {int(joint_parent[index]), int(joint_child[index])}
                projected[row] = (
                    expected_type is not None
                    and int(joint_type[index]) == expected_type
                    and not in_tree
                    and int(joint_child[index]) >= 0
                    and joint_bodies == rows_bodies
                )
            elif kind == int(MjcEqualityTargetKind.MIMIC) and 0 <= index < mimic_count:
                projected[row] = (
                    int(eq_type[row]) == int(EqType.JOINT)
                    and int(mimic_joint0[index]) == int(joint1[row])
                    and int(mimic_joint1[index]) == int(joint2[row])
                )
    return np.flatnonzero(enabled & ~projected).astype(np.int32)


def _validate_equality_constraints(model: Model) -> None:
    """Reject enabled MuJoCo equality constraints that this solver would silently ignore."""
    rows = _unprojected_equality_constraints(model)
    if rows.size:
        raise NotImplementedError(
            f"SolverFeatherPGS does not enforce MuJoCo equality constraints: model.mujoco.equality_constraint "
            f"rows {rows[:16].tolist()} are enabled and not converted to a Newton loop joint or mimic "
            "constraint. Import with convert_mjc_equality_constraints=True, model the constraint with Newton "
            "joints, or disable the rows (equality_constraint_enabled)."
        )


@wp.kernel
def _clear_dense_row_state(
    slot_counter: wp.array[int], dense_contact_world_flag: wp.array[int], dropped: wp.array2d[int]
):
    """Clear allocation state and row-loss counters together."""
    world = wp.tid()
    slot_counter[world] = 0
    dense_contact_world_flag[world] = 0
    for family in range(dropped.shape[0]):
        dropped[family, world] = 0


@wp.func
def _world_rows_lost(
    world: int,
    dense_count: wp.array[int],
    mf_count: wp.array[int],
    propagation_count: wp.array[int],
    dropped: wp.array2d[int],
    dense_capacity: int,
    mf_capacity: int,
    propagation_capacity: int,
):
    return (
        dense_count[world] > dense_capacity
        or mf_count[world] > mf_capacity
        or propagation_count[world] > propagation_capacity
        or dropped[0, world] > 0
        or dropped[1, world] > 0
        or dropped[2, world] > 0
    )


@wp.kernel
def _finalize_constraint_status(
    dense_count: wp.array[int],
    mf_count: wp.array[int],
    propagation_count: wp.array[int],
    dropped: wp.array2d[int],
    cross_world_contacts: wp.array[int],
    contact_count: wp.array[int],
    reduction_overflow: wp.array[int],
    dense_capacity: int,
    mf_capacity: int,
    propagation_capacity: int,
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
    world = slot
    if slot == global_slot:
        world = -1
        if has_responding_global != 0:
            world = 0
    if world >= 0:
        lost = lost or _world_rows_lost(
            world,
            dense_count,
            mf_count,
            propagation_count,
            dropped,
            dense_capacity,
            mf_capacity,
            propagation_capacity,
        )
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
    propagation_raw_counts: wp.array[wp.int32],
    propagation_dropped_contact_rows: wp.array[wp.int32],
    propagation_capacity: int,
    propagation_active: int,
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

    if propagation_active != 0:
        propagation_dropped = propagation_dropped_contact_rows[world]
        propagation_requested = propagation_raw_counts[world]
        if propagation_requested > propagation_capacity and wp.atomic_exch(warning_emitted, 3, 1) == 0:
            wp.printf(
                "Warning: FeatherPGS propagation constraint-row overflow in world %d: requested %d rows, "
                "limit %d; dropped %d contact/friction rows. Increase mf_max_constraints or "
                "dense_max_constraints.\n",
                world,
                propagation_requested,
                propagation_capacity,
                propagation_dropped,
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

    ``articulated_contact_response`` selects how contacts of articulated bodies are
    solved. The default ``"immediate"`` response uses the generalized rows above. The
    ``"propagation"`` responses instead solve them as fixed-size rows of the touched
    bodies with each body's response from the articulated-body factorization of its
    tree, and propagate the impulses of each iteration through the tree to the joint
    velocities; ``"propagation-fused"`` runs the whole solve in one kernel launch.

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
    - Joint limits: with ``enable_joint_limits=True``, every finite
      :attr:`~newton.Model.joint_limit_lower` / :attr:`~newton.Model.joint_limit_upper` of a
      PRISMATIC, REVOLUTE or D6 DOF is a unilateral row. Joint limits are not enforced by
      default. Joint velocity limits (:attr:`~newton.Model.joint_velocity_limit`) are
      enforced as rows when ``enable_joint_velocity_limits`` is set.
    - Contacts: rigid contacts from :class:`~newton.CollisionPipeline` with Coulomb
      point friction (one normal and two coupled tangent rows per contact, friction
      coefficient from the two shapes' ``mu``). Contact restitution, compliance and
      torsional friction are not applied.
    - Kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC`) and heterogeneous worlds.
    - CUDA graph capture of :meth:`step` and :meth:`reset`.

    Limitations:

    - CUDA only; constructing the solver on a CPU device raises :class:`NotImplementedError`.
    - Mimic joints and constraints, loop-closing joints and disabled joints raise
      :class:`NotImplementedError`. So do enabled MuJoCo equality constraints
      (``model.mujoco.equality_constraint_*``) that the importer did not convert to a
      Newton loop joint or mimic constraint; enabling such a row later raises from
      :meth:`notify_model_changed` with :attr:`~newton.ModelFlags.CONSTRAINT_PROPERTIES`.
      A row counts as converted only when its ``target_kind`` / ``target`` link names a
      loop joint or mimic constraint between the row's own bodies or joints.
      Particles are not simulated.
    - Gradients are not supported.

    Constraint rows are stored per world with fixed capacities (``dense_max_constraints``
    for rows of articulated bodies, ``mf_max_constraints`` for free-body contacts, and a
    propagation-row capacity derived from both, see ``articulated_contact_response``). Rows
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
    # (keys: cholesky_kernel, trisolve_kernel, hinv_jt_kernel, and propagation_tree_kernel,
    # where "generic" replaces the one-warp propagation tree kernels by the per-articulation ones).
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
        enable_joint_limits: bool = False,
        joint_limit_activation_gap: float = float("inf"),
        enable_joint_velocity_limits: bool = False,
        velocity_limit_activation_fraction: float = 0.0,
        dense_max_constraints: int = 32,
        mf_max_constraints: int = 512,
        warn_constraint_overflow: bool = True,
        articulated_contact_response: Literal["immediate", "propagation", "propagation-fused"] = "immediate",
        propagation_same_articulation_rows: bool = False,
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
            enable_joint_limits: Enforce :attr:`~newton.Model.joint_limit_lower` and
                :attr:`~newton.Model.joint_limit_upper` of PRISMATIC, REVOLUTE and D6 DOFs as
                unilateral rows. When ``False`` (the default), no joint-limit rows are created,
                joint position limits are not enforced, and limits use no
                ``dense_max_constraints`` capacity. The default matches the FeatherPGS
                reference implementation this solver is ported from.
            joint_limit_activation_gap: Distance from a finite position limit [m or rad] at
                which its row is created, when ``enable_joint_limits`` is set. ``inf`` creates
                the rows of every finite limit each step; a smaller gap creates a row only when
                ``q <= lower + gap`` or ``q >= upper - gap``. Every finite bound is enforced,
                however large: the builder's default limits (``+/-1e10``) are finite, so with
                ``inf`` each such DOF uses two rows of ``dense_max_constraints`` in every step.
                Use a finite gap or unbounded (``+/-inf``) limits to avoid these rows; the
                constructor warns when the finite limits alone exceed the capacity. The value
                is validated even when joint limits are disabled.
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
                enabled joint limits and joint velocity limits) per world. Rows beyond it are
                dropped and reported, see :attr:`constraint_overflow`.
            mf_max_constraints: Capacity of free-body contact rows per world. Rows beyond it
                are dropped and reported, see :attr:`constraint_overflow`.
            warn_constraint_overflow: Print a device-side warning the first time a world
                exceeds a row capacity. The warning does not synchronize the host and is
                compatible with CUDA graph capture.
            articulated_contact_response: How contacts touching an articulation that is not a
                free body are solved.

                - ``"immediate"``: dense rows of the articulation's generalized coordinates,
                  ``Y = H^-1 J^T`` per row.
                - ``"propagation"``: fixed-size rows of the touched bodies at their centers of
                  mass, with each body's 6x6 response from the articulated-body factorization
                  of its tree. Each iteration solves these rows and then propagates the
                  accumulated body impulses through the trees to the joint velocities. All
                  contacts, including those of free bodies, use these rows, whose capacity per
                  world is ``mf_max_constraints + dense_max_constraints``.
                - ``"propagation-fused"``: the same response, with every iteration of the whole
                  solve in one kernel launch. Free-body contacts keep their free-body rows; the
                  propagation row capacity per world is ``dense_max_constraints``. Requires
                  every non-free articulation that responds to have the same number of DOFs.

                The propagation responses avoid the per-row ``H^-1 J^T`` of contact rows at
                the cost of tree passes over every articulation in each iteration; which
                response is faster depends on the model. They keep joint-limit and
                velocity-limit rows dense, and contacts between two links of the same
                articulation dense unless ``propagation_same_articulation_rows`` is set. The
                tree factorization is rebuilt every step, independently of
                ``update_mass_matrix_interval``.
            propagation_same_articulation_rows: Solve contacts between two links of the same
                articulation as propagation rows, with the exact cross response of the two
                links, instead of dense rows. Requires
                ``articulated_contact_response="propagation"``.
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
        self.enable_joint_limits = bool(enable_joint_limits)
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
        if articulated_contact_response not in ("immediate", "propagation", "propagation-fused"):
            raise ValueError(
                "articulated_contact_response must be 'immediate', 'propagation' or 'propagation-fused', "
                f"got {articulated_contact_response!r}"
            )
        self.articulated_contact_response = articulated_contact_response
        self.propagation_same_articulation_rows = bool(propagation_same_articulation_rows)
        if self.propagation_same_articulation_rows and articulated_contact_response != "propagation":
            raise ValueError(
                "propagation_same_articulation_rows requires articulated_contact_response='propagation', "
                f"got {articulated_contact_response!r}"
            )
        # The propagation response routes every contact row to the propagation family, so
        # its capacity absorbs both the free-body and the dense contact budgets.
        if articulated_contact_response == "propagation":
            self.propagation_max_constraints = self.mf_max_constraints + self._requested_dense_max_constraints
        else:
            self.propagation_max_constraints = self._requested_dense_max_constraints

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
        self._warn_persistent_row_capacity(model)
        self._setup_passive_joint_forces(model)
        self._compute_world_response_dof_mapping(model)
        self._setup_propagation(model)
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
        self._allocate_propagation_buffers(model)
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
            wp.zeros(4, dtype=wp.int32, device=model.device) if self.warn_constraint_overflow else None
        )
        self.constraint_overflow = wp.zeros(int(model.world_count) + 1, dtype=wp.bool, device=model.device)
        """Capacity failure flags, shape ``[model.world_count + 1]``, dtype ``bool``.

        One entry per world plus a final entry for global (world ``-1``) articulations:
        the layout of the :meth:`reset` mask, so entry ``i`` is cleared exactly when mask
        entry ``i`` is selected. An entry is set when a step drops constraint
        rows of that world (``dense_max_constraints``, ``mf_max_constraints`` or the
        propagation-row capacity exceeded)
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
        self._row_dropped_all = wp.zeros((3, max(self.world_count, 1)), dtype=wp.int32, device=model.device)
        self._row_dropped_dense = self._row_dropped_all[0]
        self._row_dropped_mf = self._row_dropped_all[1]
        self._row_dropped_propagation = self._row_dropped_all[2]

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
        again; reconstruct the solver in that case. Constraint changes re-check the MuJoCo
        equality rows: enabling one that is not converted to a Newton loop joint or mimic
        constraint raises :class:`NotImplementedError`. Capacity status in
        :attr:`constraint_overflow` is not cleared, see :meth:`reset`.

        Args:
            flags: Bit-mask of :class:`~newton.ModelFlags` indicating which model properties changed.
        """
        if flags & ModelFlags.CONSTRAINT_PROPERTIES:
            _validate_equality_constraints(self.model)
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

    def _persistent_dense_rows_per_world(self, model) -> np.ndarray:
        """Count the dense rows every step allocates regardless of the state, per world.

        With ``enable_joint_limits`` and ``joint_limit_activation_gap=inf`` each finite
        position limit of a responding PRISMATIC, REVOLUTE or D6 DOF is a row in every step,
        mirroring the joint-limit row builder. Large finite bounds, such as the builder's
        default ``+/-1e10``, count like any other finite bound. Finite gaps make the count
        state dependent, and disabled joint limits create no rows, so it is zero then.
        """
        rows = np.zeros(max(self.world_count, 1), dtype=np.int64)
        if not self.enable_joint_limits or not np.isinf(self.joint_limit_activation_gap) or not self._joint_limit_sizes:
            return rows
        limit_q_index = self._joint_limit_q_index.numpy()
        lower = model.joint_limit_lower.numpy()
        upper = model.joint_limit_upper.numpy()
        finite_sides = np.where(limit_q_index >= 0, np.isfinite(lower).astype(np.int64) + np.isfinite(upper), 0)
        dof_start = self.articulation_dof_start.numpy()
        art_to_world = self.art_to_world.numpy()
        for size in self._joint_limit_sizes:
            for art in self.group_to_art[size].numpy():
                start = int(dof_start[art])
                rows[int(art_to_world[art])] += int(np.sum(finite_sides[start : start + size]))
        return rows

    def _warn_persistent_row_capacity(self, model) -> None:
        """Warn once at construction if every step is certain to exceed ``dense_max_constraints``."""
        rows = self._persistent_dense_rows_per_world(model)
        capacity = self._requested_dense_max_constraints
        worlds = np.flatnonzero(rows > capacity)
        if worlds.size:
            warnings.warn(
                f"SolverFeatherPGS: worlds {worlds[:16].tolist()} need at least {int(rows.max())} dense rows in "
                f"every step for the finite joint position limits alone (two per DOF limited on both sides; "
                f"the builder's default +/-1e10 limits are finite), more than dense_max_constraints="
                f"{capacity}. Rows beyond the capacity are dropped and flagged in constraint_overflow. Raise "
                "dense_max_constraints, set a finite joint_limit_activation_gap so that only limits near the "
                "joint position create rows, or set unbounded limits to +/-inf.",
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
        self._init_propagation_kernels(model)
        if self.world_count > 0 and self.max_world_dofs > 0 and not self._propagation_fused:
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
                row_phases=self._propagation_active,
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

    def _mf_gs_inputs(self) -> list:
        """Return the row inputs of the matrix-free Gauss-Seidel kernel."""
        return [
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
        ]

    def _launch_pgs_solve(self) -> None:
        """Run the fused matrix-free Gauss-Seidel sweep over the dense and free-body rows."""
        if self.pgs_iterations <= 0 or self._pgs_solve_mf_gs_kernel is None:
            return
        wp.launch_tiled(
            self._pgs_solve_mf_gs_kernel,
            dim=[self.world_count],
            inputs=[*self._mf_gs_inputs(), self.pgs_iterations, self.pgs_omega, 0],
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
        wp.copy(self.v_out, self.v_hat)
        if self._propagation_active:
            self._propagation_setup(state_in, state_aug, dt)

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
        self._pack_mf_meta()
        if self._propagation_active:
            self._launch_propagation_solve()
        else:
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
                self.propagation_impulses,
                self.constraint_count,
                self.mf_constraint_count,
                self.propagation_constraint_count,
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
            inputs=[self.slot_counter, self.dense_contact_world_flag, self._row_dropped_all],
            device=model.device,
        )
        self._propagation_clear_rows()
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

        # Disabled joint limits create no rows (and use no capacity), as in the reference solver.
        limit_sizes = self._joint_limit_sizes if self.enable_joint_limits else frozenset()
        for size in limit_sizes:
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
                    int(self._propagation_active),
                    int(self.propagation_same_articulation_rows),
                    int(self._propagation_active and self.articulated_contact_response == "propagation"),
                    max_constraints,
                    self.mf_max_constraints,
                    self.propagation_max_constraints,
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
                    self.propagation_slot_counter,
                    self.dense_contact_world_flag,
                    self._row_dropped_dense,
                    self._row_dropped_mf,
                    self._row_dropped_propagation,
                    self._dense_first_rejected_slot,
                    mf_first_rejected_slot,
                    self._propagation_first_rejected_slot,
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

            if self._propagation_active:
                self._propagation_build_rows(state_in, state_aug, contacts, contact_build_threads)

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
                self.propagation_slot_counter,
                self._row_dropped_all,
                self._cross_world_contacts,
                contacts.rigid_contact_count if contacts is not None else self._dummy_contact_count,
                contacts._reduction_overflow if contacts is not None else self._dummy_contact_count,
                self.dense_max_constraints,
                self.mf_max_constraints,
                self.propagation_max_constraints,
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
                    self.propagation_slot_counter,
                    self._row_dropped_propagation,
                    self.propagation_max_constraints,
                    int(self._propagation_active),
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

    # ------------------------------------------------------------------
    # Propagation contact response
    # ------------------------------------------------------------------

    def _setup_propagation(self, model) -> None:
        """Resolve the propagation response plan: tree groups, kernel choice and fused size."""
        self._propagation_active = False
        self._propagation_fused = False
        self._propagation_fused_size = None
        self._propagation_tree_sizes: list[int] = []
        self._propagation_tree_art_count: dict[int, int] = {}
        self._propagation_native_tree: dict[int, bool] = {}
        self._propagation_free_root_tree: dict[int, bool] = {}
        self._propagation_max_joints: dict[int, int] = {}
        self._propagation_needs_body_map = False
        self.max_propagation_bodies = 1
        if self.articulated_contact_response == "immediate" or not self.size_groups or self._model_plan is None:
            return

        plan = self._model_plan
        responding_non_free = (plan.response_dof_count > 0) & (plan.is_free_rigid == 0)
        route_free_free = self.articulated_contact_response == "propagation"
        self._propagation_active = bool(np.any(responding_non_free)) or (
            route_free_free and self._has_free_rigid_bodies
        )
        if not self._propagation_active:
            return

        tree_kernel = self._kernel_overrides.get("propagation_tree_kernel", "auto")
        if tree_kernel not in ("auto", "generic"):
            raise ValueError("propagation_tree_kernel must be one of ['auto', 'generic']")
        articulation_start = model.articulation_start.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        for size in self.size_groups:
            arts = np.flatnonzero(responding_non_free & (plan.response_dof_count == size))
            if arts.size == 0:
                continue
            self._propagation_tree_sizes.append(int(size))
            self._propagation_tree_art_count[int(size)] = int(arts.size)
            single_dof = True
            free_root = True
            max_joints = 0
            for art in arts:
                joint_start = int(articulation_start[art])
                joint_end = int(articulation_start[art + 1])
                max_joints = max(max_joints, joint_end - joint_start)
                dof_counts = np.diff(joint_qd_start[joint_start : joint_end + 1])
                parents = joint_parent[joint_start:joint_end]
                if np.any(dof_counts > 1):
                    single_dof = False
                # Free-root shape: the first joint is the only world-rooted joint (up to 6 DOFs),
                # every other joint has at most one DOF and a parent earlier in joint order.
                if (
                    joint_end <= joint_start
                    or int(parents[0]) != -1
                    or int(dof_counts[0]) > 6
                    or np.any(parents[1:] < 0)
                    or np.any(dof_counts[1:] > 1)
                ):
                    free_root = False
                else:
                    child_to_slot = {int(joint_child[j]): j - joint_start for j in range(joint_start, joint_end)}
                    for j in range(joint_start + 1, joint_end):
                        parent_slot = child_to_slot.get(int(joint_parent[j]), -1)
                        if parent_slot < 0 or parent_slot >= j - joint_start:
                            free_root = False
                            break
            self._propagation_max_joints[int(size)] = max_joints
            native = tree_kernel == "auto" and (single_dof or free_root) and _propagation_tree_kernel_fits(max_joints)
            self._propagation_native_tree[int(size)] = native
            self._propagation_free_root_tree[int(size)] = native and not single_dof
            if not native:
                self._propagation_needs_body_map = True

        if self.articulated_contact_response == "propagation-fused":
            if len(self._propagation_tree_sizes) != 1:
                raise NotImplementedError(
                    "articulated_contact_response='propagation-fused' requires every responding non-free "
                    f"articulation to have the same number of DOFs; found DOF counts {self._propagation_tree_sizes}. "
                    "Use articulated_contact_response='propagation' instead."
                )
            self._propagation_fused = True
            self._propagation_fused_size = self._propagation_tree_sizes[0]
            self._propagation_needs_body_map = True

        # Bodies of responding articulations, per solve world: the capacity of the body list.
        world_body_count = np.zeros(max(self.world_count, 1), dtype=np.int64)
        for art in np.flatnonzero(plan.response_dof_count > 0):
            joint_start = int(articulation_start[art])
            joint_end = int(articulation_start[art + 1])
            world_body_count[int(plan.articulation_world[art])] += joint_end - joint_start
        self.max_propagation_bodies = max(int(world_body_count.max()), 1)

        parent_slot = np.full(max(int(model.joint_count), 1), -1, dtype=np.int32)
        for art in range(model.articulation_count):
            joint_start = int(articulation_start[art])
            joint_end = int(articulation_start[art + 1])
            child_to_slot = {int(joint_child[j]): j - joint_start for j in range(joint_start, joint_end)}
            for j in range(joint_start, joint_end):
                if int(joint_parent[j]) >= 0:
                    parent_slot[j] = child_to_slot.get(int(joint_parent[j]), -1)
        self._propagation_joint_parent_slot = wp.array(parent_slot, dtype=wp.int32, device=model.device)

        # Dense joint-limit and velocity-limit rows change the tree velocities between the
        # propagation passes, so the per-iteration twist refresh cannot be skipped.
        self._propagation_has_limit_rows = self.enable_joint_limits and bool(self._joint_limit_sizes)
        has_velocity_limits = self.enable_joint_velocity_limits or (
            self._has_rigid_body_velocity_limits and self._has_free_rigid_bodies
        )
        self._propagation_has_velocity_limit_rows = bool(
            has_velocity_limits and not np.isinf(self.velocity_limit_activation_fraction)
        )

    def _allocate_propagation_buffers(self, model) -> None:
        """Allocate the per-world propagation rows and the per-body and per-joint tree buffers.

        Without an active propagation response only the per-world counters exist (zero),
        so the status and force kernels can read them unconditionally.
        """
        device = model.device
        worlds = max(self.world_count, 1)
        self.dense_contact_world_flag = wp.zeros((worlds,), dtype=wp.int32, device=device)
        self.propagation_constraint_count = wp.zeros((worlds,), dtype=wp.int32, device=device)
        self.propagation_slot_counter = wp.zeros((worlds,), dtype=wp.int32, device=device)
        self._propagation_first_rejected_slot = wp.full((worlds,), _ROW_SLOT_UNBOUNDED, dtype=wp.int32, device=device)
        rows = self.propagation_max_constraints if self._propagation_active else 1
        self.propagation_impulses = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        if not self._propagation_active:
            return

        bodies = max(model.body_count, 1)
        joints = max(model.joint_count, 1)
        dofs = max(model.joint_dof_count, 1)
        self.propagation_body_count = wp.zeros((worlds,), dtype=wp.int32, device=device)
        self.propagation_body_list = wp.full((worlds, self.max_propagation_bodies), -1, dtype=wp.int32, device=device)
        self.propagation_body_a = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.propagation_body_b = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.propagation_J_a = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.propagation_J_b = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.propagation_MiJt_a = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.propagation_MiJt_b = wp.zeros((worlds, rows, 6), dtype=wp.float32, device=device)
        self.propagation_rhs = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.propagation_eff_mass_inv = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.propagation_row_type = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        self.propagation_row_parent = wp.full((worlds, rows), -1, dtype=wp.int32, device=device)
        self.propagation_row_mu = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.propagation_phi = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.propagation_target_velocity = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        # Per body: 6x6 center-of-mass response, live twist and the impulse accumulated in a sweep.
        self.propagation_body_response = wp.zeros((bodies, 6, 6), dtype=wp.float32, device=device)
        self.propagation_body_qd = wp.zeros((bodies, 6), dtype=wp.float32, device=device)
        self.propagation_body_impulses = wp.zeros((bodies, 6), dtype=wp.float32, device=device)
        self.propagation_body_com_rel = wp.zeros((bodies, 3), dtype=wp.float32, device=device)
        self.propagation_body_seen = wp.zeros((bodies,), dtype=wp.int32, device=device)
        self.propagation_body_local_slot = wp.zeros((bodies,), dtype=wp.int32, device=device)
        # Articulated-body factorization and scratch of the tree passes.
        self.propagation_joint_S_flat = wp.zeros((dofs, 6), dtype=wp.float32, device=device)
        self.propagation_tree_Ia = wp.zeros((bodies, 6, 6), dtype=wp.float32, device=device)
        self.propagation_tree_U = wp.zeros((dofs, 6), dtype=wp.float32, device=device)
        self.propagation_tree_D_chol = wp.zeros((joints, 6, 6), dtype=wp.float32, device=device)
        self.propagation_tree_D_inv = wp.zeros((joints, 6, 6), dtype=wp.float32, device=device)
        self.propagation_tree_pA = wp.zeros((bodies, 6), dtype=wp.float32, device=device)
        self.propagation_tree_u = wp.zeros((dofs,), dtype=wp.float32, device=device)
        self.propagation_tree_qdd = wp.zeros((dofs,), dtype=wp.float32, device=device)
        self.propagation_tree_body_delta = wp.zeros((bodies, 6), dtype=wp.float32, device=device)
        # Responding non-free articulations of each tree group, in solve-world order.
        self._propagation_tree_arts = {size: self.world_group_to_art[size] for size in self._propagation_tree_sizes}

    def _init_propagation_kernels(self, model) -> None:
        """Resolve the size-specialized propagation kernels."""
        self._propagation_factor_kernels: dict[int, wp.Kernel] = {}
        self._propagation_response_kernels: dict[int, wp.Kernel] = {}
        self._propagation_propagate_kernels: dict[int, wp.Kernel] = {}
        self._propagation_refresh_kernels: dict[int, wp.Kernel] = {}
        self._pgs_solve_propagation_kernel = None
        self._pgs_solve_propagation_fused_kernel = None
        if not self._propagation_active:
            return
        device_arch = model.device.arch
        for size in self._propagation_tree_sizes:
            if not self._propagation_native_tree[size]:
                continue
            free_root = self._propagation_free_root_tree[size]
            max_joints = self._propagation_max_joints[size]
            self._propagation_factor_kernels[size] = _get_factor_propagation_tree_revolute_kernel(
                size, device_arch, has_free_root=free_root
            )
            self._propagation_response_kernels[size] = _get_propagation_tree_body_response_revolute_kernel(
                size, device_arch, has_free_root=free_root
            )
            self._propagation_propagate_kernels[size] = _get_propagate_tree_impulses_revolute_kernel(
                size, max_joints, device_arch, has_free_root=free_root
            )
            self._propagation_refresh_kernels[size] = _get_refresh_propagation_tree_body_qd_warp_kernel(
                size, max_joints, device_arch
            )
        if self._propagation_fused:
            self._pgs_solve_propagation_fused_kernel = _get_pgs_solve_propagation_full_iteration_kernel(
                self.dense_max_constraints,
                self.mf_meta_packed.shape[1] // 4,
                self.max_world_dofs,
                self.propagation_max_constraints,
                self.max_propagation_bodies,
                int(self._propagation_fused_size),
                device_arch,
            )
        else:
            self._pgs_solve_propagation_kernel = _get_pgs_solve_propagation_contact_kernel(
                self.propagation_max_constraints, device_arch, worlds_per_block=_PROPAGATION_WORLDS_PER_BLOCK
            )

    def _propagation_clear_rows(self) -> None:
        """Reset the per-step propagation row and body state before the rows are allocated."""
        self.propagation_slot_counter.zero_()
        self._propagation_first_rejected_slot.fill_(_ROW_SLOT_UNBOUNDED)
        self.propagation_constraint_count.zero_()
        if not self._propagation_active:
            return
        self.propagation_impulses.zero_()
        self.propagation_J_a.zero_()
        self.propagation_J_b.zero_()
        self.propagation_MiJt_a.zero_()
        self.propagation_MiJt_b.zero_()
        self.propagation_body_impulses.zero_()
        self.propagation_body_count.zero_()
        self.propagation_body_seen.zero_()

    def _propagation_build_rows(
        self, state_in: State, state_aug: State, contacts: Contacts, contact_build_threads: int
    ):
        """Fill the propagation rows of the allocated contacts and count them per world."""
        model = self.model
        wp.launch(
            build_propagation_contact_rows,
            dim=contact_build_threads,
            inputs=[
                contacts.rigid_contact_count,
                contact_build_threads,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.contact_world,
                self.contact_slot,
                self.contact_path,
                self.contact_art_a,
                self.contact_art_b,
                self.articulation_response_dof_count,
                model.shape_body,
                state_in.body_q,
                model.body_com,
                state_aug.body_v_s,
                self._prescribed_articulation,
                self.articulation_origin,
                self.shape_material_mu,
            ],
            outputs=[
                self.propagation_body_a,
                self.propagation_body_b,
                self.propagation_J_a,
                self.propagation_J_b,
                self.propagation_row_type,
                self.propagation_row_parent,
                self.propagation_row_mu,
                self.propagation_phi,
                self.propagation_target_velocity,
            ],
            device=model.device,
        )
        wp.launch(
            finalize_mf_constraint_counts,
            dim=self.world_count,
            inputs=[
                self.propagation_slot_counter,
                self.propagation_max_constraints,
                3,
                self._propagation_first_rejected_slot,
            ],
            outputs=[self.propagation_constraint_count],
            device=model.device,
        )
        if self._propagation_needs_body_map:
            wp.launch(
                build_propagation_body_map,
                dim=self.world_count * self.propagation_max_constraints,
                inputs=[
                    self.propagation_constraint_count,
                    self.propagation_body_a,
                    self.propagation_body_b,
                    self.propagation_max_constraints,
                    self.max_propagation_bodies,
                    self.propagation_body_seen,
                ],
                outputs=[
                    self.propagation_body_list,
                    self.propagation_body_count,
                    self.propagation_body_local_slot,
                ],
                device=model.device,
            )

    def _propagation_setup(self, state_in: State, state_aug: State, dt: float) -> None:
        """Factor the trees, compute the body responses and twists, and the row responses and biases.

        Runs after ``v_out`` holds the unconstrained velocity of the step.
        """
        model = self.model
        wp.launch(
            compute_propagation_body_com_rel,
            dim=model.body_count,
            inputs=[self.body_to_articulation, state_in.body_q, model.body_com, self.articulation_origin],
            outputs=[self.propagation_body_com_rel],
            device=model.device,
        )
        wp.launch(
            flatten_propagation_joint_S,
            dim=model.joint_count,
            inputs=[model.joint_child, model.joint_qd_start, state_aug.joint_S_s, self.propagation_body_com_rel],
            outputs=[self.propagation_joint_S_flat],
            device=model.device,
        )
        for size in self._propagation_tree_sizes:
            arts = self._propagation_tree_arts[size]
            n_arts = self._propagation_tree_art_count[size]
            factor_inputs = [
                model.articulation_start,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
            ]
            factor_terms = [
                self.propagation_joint_S_flat,
                self._joint_armature_device,
                self.articulation_max_dofs,
                self.aug_row_counts,
                self.aug_row_dof_index,
                self.aug_row_K,
                self.body_I_m,
                state_aug.body_q_com,
                self.propagation_body_com_rel,
            ]
            factor_outputs = [
                self.propagation_tree_Ia,
                self.propagation_tree_U,
                self.propagation_tree_D_chol,
                self.propagation_tree_D_inv,
            ]
            if self._propagation_native_tree[size]:
                wp.launch_tiled(
                    self._propagation_factor_kernels[size],
                    dim=[n_arts],
                    inputs=[arts, *factor_inputs, *factor_terms],
                    outputs=factor_outputs,
                    block_dim=32,
                    device=model.device,
                )
                wp.launch_tiled(
                    self._propagation_response_kernels[size],
                    dim=[n_arts],
                    inputs=[
                        arts,
                        *factor_inputs,
                        self.propagation_joint_S_flat,
                        self.propagation_body_com_rel,
                        self.propagation_tree_U,
                        self.propagation_tree_D_inv,
                    ],
                    outputs=[
                        self.propagation_tree_Ia,
                        self.propagation_tree_body_delta,
                        self.propagation_body_response,
                    ],
                    block_dim=32,
                    device=model.device,
                )
            else:
                wp.launch(
                    factor_propagation_tree_for_size,
                    dim=n_arts,
                    inputs=[arts, *factor_inputs, model.joint_dof_dim, *factor_terms],
                    outputs=factor_outputs,
                    device=model.device,
                )
                wp.launch(
                    compute_propagation_tree_body_response_for_size,
                    dim=n_arts,
                    inputs=[
                        self.propagation_body_count,
                        self.propagation_body_list,
                        self.body_to_articulation,
                        self.body_to_joint,
                        arts,
                        self.art_to_world,
                        *factor_inputs,
                        model.joint_dof_dim,
                        self.propagation_joint_S_flat,
                        self.max_propagation_bodies,
                        self.propagation_body_com_rel,
                        self.propagation_tree_U,
                        self.propagation_tree_D_inv,
                    ],
                    outputs=[
                        self.propagation_tree_pA,
                        self.propagation_tree_u,
                        self.propagation_tree_qdd,
                        self.propagation_tree_body_delta,
                        self.propagation_body_response,
                    ],
                    device=model.device,
                )
        if self._has_free_rigid_bodies:
            wp.launch(
                copy_free_rigid_propagation_body_response,
                dim=self._free_rigid_body_count,
                inputs=[self.free_rigid_body_indices, self.mf_body_Hinv],
                outputs=[self.propagation_body_response],
                device=model.device,
            )
        self._propagation_refresh_twists(force=True)
        wp.launch(
            compute_propagation_effective_mass_and_rhs,
            dim=self.world_count * self.propagation_max_constraints,
            inputs=[
                self.propagation_constraint_count,
                self.propagation_body_a,
                self.propagation_body_b,
                self.propagation_J_a,
                self.propagation_J_b,
                self.propagation_body_response,
                self.propagation_phi,
                self.propagation_row_type,
                self.propagation_target_velocity,
                self.rigid_body_max_depenetration_velocity,
                self.pgs_cfm,
                self.pgs_beta,
                dt,
                self.propagation_max_constraints,
            ],
            outputs=[
                self.propagation_eff_mass_inv,
                self.propagation_MiJt_a,
                self.propagation_MiJt_b,
                self.propagation_rhs,
            ],
            device=model.device,
        )
        if self.propagation_same_articulation_rows:
            wp.launch(
                refine_same_articulation_propagation_rows,
                dim=model.articulation_count,
                inputs=[
                    self.is_free_rigid,
                    self.art_to_world,
                    self.body_to_articulation,
                    model.articulation_start,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    self.propagation_joint_S_flat,
                    self.propagation_body_com_rel,
                    self.propagation_tree_U,
                    self.propagation_tree_D_inv,
                    self.propagation_constraint_count,
                    self.propagation_body_a,
                    self.propagation_body_b,
                    self.propagation_J_a,
                    self.propagation_J_b,
                    self.pgs_cfm,
                    self.propagation_max_constraints,
                    self.propagation_tree_pA,
                    self.propagation_tree_u,
                    self.propagation_tree_qdd,
                    self.propagation_tree_body_delta,
                ],
                outputs=[
                    self.propagation_eff_mass_inv,
                    self.propagation_MiJt_a,
                    self.propagation_MiJt_b,
                ],
                device=model.device,
            )

    def _propagation_refresh_twists(self, *, force: bool) -> None:
        """Recompute the propagation body twists from ``v_out``.

        Without ``force`` the tree pass skips worlds without dense contact rows; free-body
        twists are always copied.
        """
        model = self.model
        for size in self._propagation_tree_sizes:
            arts = self._propagation_tree_arts[size]
            n_arts = self._propagation_tree_art_count[size]
            if self._propagation_native_tree[size]:
                wp.launch_tiled(
                    self._propagation_refresh_kernels[size],
                    dim=[n_arts],
                    inputs=[
                        arts,
                        self.art_to_world,
                        self.dense_contact_world_flag,
                        int(force),
                        model.articulation_start,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        self._propagation_joint_parent_slot,
                        self.propagation_joint_S_flat,
                        self.propagation_body_com_rel,
                        self.v_out,
                    ],
                    outputs=[self.propagation_body_qd],
                    block_dim=32,
                    device=model.device,
                )
            else:
                wp.launch(
                    refresh_propagation_tree_body_qd_for_size,
                    dim=n_arts,
                    inputs=[
                        arts,
                        self.art_to_world,
                        self.dense_contact_world_flag,
                        int(force),
                        model.articulation_start,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        self.propagation_joint_S_flat,
                        self.propagation_body_com_rel,
                        self.v_out,
                    ],
                    outputs=[self.propagation_body_qd],
                    device=model.device,
                )
        if self._has_free_rigid_bodies:
            wp.launch(
                refresh_propagation_free_body_qd_from_vout,
                dim=self._free_rigid_body_count,
                inputs=[
                    self.free_rigid_body_indices,
                    self.body_to_articulation,
                    self.articulation_dof_start,
                    self.v_out,
                ],
                outputs=[self.propagation_body_qd],
                device=model.device,
            )

    def _propagation_propagate_impulses(self) -> None:
        """Apply the impulses accumulated in a propagation sweep to the joint velocities."""
        model = self.model
        for size in self._propagation_tree_sizes:
            arts = self._propagation_tree_arts[size]
            n_arts = self._propagation_tree_art_count[size]
            tree_inputs = [
                model.articulation_start,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
            ]
            if self._propagation_native_tree[size]:
                wp.launch_tiled(
                    self._propagation_propagate_kernels[size],
                    dim=[n_arts],
                    inputs=[
                        arts,
                        *tree_inputs,
                        self._propagation_joint_parent_slot,
                        self.propagation_joint_S_flat,
                        self.propagation_body_com_rel,
                        self.propagation_tree_U,
                        self.propagation_tree_D_inv,
                        self.propagation_body_impulses,
                    ],
                    outputs=[self.propagation_body_qd, self.v_out],
                    block_dim=32,
                    device=model.device,
                )
            else:
                wp.launch(
                    propagate_tree_impulses_for_size,
                    dim=n_arts,
                    inputs=[
                        arts,
                        *tree_inputs,
                        model.joint_dof_dim,
                        self.propagation_joint_S_flat,
                        self.propagation_body_com_rel,
                        self.propagation_tree_U,
                        self.propagation_tree_D_inv,
                    ],
                    outputs=[
                        self.propagation_tree_pA,
                        self.propagation_tree_u,
                        self.propagation_tree_qdd,
                        self.propagation_tree_body_delta,
                        self.propagation_body_impulses,
                        self.propagation_body_qd,
                        self.v_out,
                    ],
                    device=model.device,
                )
        if self._has_free_rigid_bodies:
            wp.launch(
                flush_propagation_free_body_qd_to_vout,
                dim=self._free_rigid_body_count,
                inputs=[self.free_rigid_body_indices, self.body_to_articulation, self.articulation_dof_start],
                outputs=[self.propagation_body_qd, self.propagation_body_impulses, self.v_out],
                device=model.device,
            )

    def _launch_mf_gs_phase(self, row_phase: int) -> None:
        """Run one matrix-free sweep restricted to one dense/free-body row family."""
        wp.launch_tiled(
            self._pgs_solve_mf_gs_kernel,
            dim=[self.world_count],
            inputs=[*self._mf_gs_inputs(), 1, self.pgs_omega, row_phase],
            outputs=[self.v_out],
            block_dim=32,
            device=self.model.device,
        )

    def _launch_propagation_solve(self) -> None:
        """Run the projected Gauss-Seidel iterations of the propagation response.

        Each iteration solves the dense joint-limit rows, the dense and free-body contact
        rows, the propagation rows followed by the tree propagation of their impulses, and
        last the velocity-limit rows.
        """
        if self.pgs_iterations <= 0:
            return
        model = self.model
        if self._propagation_fused:
            size = int(self._propagation_fused_size)
            wp.launch_tiled(
                self._pgs_solve_propagation_fused_kernel,
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
                    self.propagation_constraint_count,
                    self.propagation_body_count,
                    self.propagation_body_list,
                    self.max_propagation_bodies,
                    self.propagation_body_a,
                    self.propagation_body_b,
                    self.propagation_MiJt_a,
                    self.propagation_MiJt_b,
                    self.propagation_J_a,
                    self.propagation_J_b,
                    self.propagation_eff_mass_inv,
                    self.propagation_rhs,
                    self.propagation_row_type,
                    self.propagation_row_parent,
                    self.propagation_row_mu,
                    self.body_to_articulation,
                    self.is_free_rigid,
                    self.articulation_dof_start,
                    self.articulation_world_dof_offset,
                    self.world_group_art_start[size],
                    self.world_group_to_art[size],
                    model.articulation_start,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_qd_start,
                    self.propagation_joint_S_flat,
                    self.propagation_body_com_rel,
                    self.propagation_tree_U,
                    self.propagation_tree_D_inv,
                    self.pgs_iterations,
                    self.pgs_omega,
                ],
                outputs=[
                    self.propagation_impulses,
                    self.propagation_tree_pA,
                    self.propagation_tree_u,
                    self.propagation_tree_qdd,
                    self.propagation_tree_body_delta,
                    self.propagation_body_qd,
                    self.propagation_body_impulses,
                    self.v_out,
                ],
                block_dim=32,
                device=model.device,
            )
            return

        refresh_forced = self._propagation_has_limit_rows or self._propagation_has_velocity_limit_rows
        wpb = _PROPAGATION_WORLDS_PER_BLOCK
        for _ in range(self.pgs_iterations):
            if self._propagation_has_limit_rows:
                self._launch_mf_gs_phase(3)
            self._launch_mf_gs_phase(4)
            self._propagation_refresh_twists(force=refresh_forced)
            wp.launch_tiled(
                self._pgs_solve_propagation_kernel,
                dim=[(self.world_count + wpb - 1) // wpb],
                inputs=[
                    self.world_count,
                    self.propagation_constraint_count,
                    self.propagation_body_a,
                    self.propagation_body_b,
                    self.propagation_MiJt_a,
                    self.propagation_MiJt_b,
                    self.propagation_J_a,
                    self.propagation_J_b,
                    self.propagation_eff_mass_inv,
                    self.propagation_rhs,
                    self.propagation_row_type,
                    self.propagation_row_parent,
                    self.propagation_row_mu,
                    self.pgs_omega,
                ],
                outputs=[
                    self.propagation_impulses,
                    self.propagation_body_qd,
                    self.propagation_body_impulses,
                ],
                block_dim=32 * wpb,
                device=model.device,
            )
            self._propagation_propagate_impulses()
            if self._propagation_has_velocity_limit_rows:
                self._launch_mf_gs_phase(5)


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
    row_phases: bool = False,
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

    With ``row_phases`` the launch argument ``row_phase`` restricts a sweep to one row
    family, so a propagation solve can interleave its own rows: ``3`` the dense
    joint-limit rows, ``4`` the dense and free-body contact rows, ``5`` the velocity-limit
    rows. ``0`` (the only value without ``row_phases``) sweeps every family.

    Args:
        max_constraints: Dense row capacity ``M_D`` per world.
        mf_max_constraints: Free-body row capacity ``M_MF`` per world.
        max_world_dofs: Response DOFs ``D`` per world.
        device_arch: CUDA architecture; part of the cache key.
        has_dense_velocity_limit_rows: Emit the dense velocity-limit pass.
        shared_metadata: Keep the dense row metadata in shared memory.
        row_phases: Honor the ``row_phase`` launch argument.
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

    if row_phases:
        limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        phase_bounds = """    if (row_phase == 3) {
        mf_main_end = 0;
        velocity_limit_pass = 0;
    } else if (row_phase == 4) {
        velocity_limit_pass = 0;
    } else if (row_phase == 5) {
        dense_main_end = 0;
        mf_main_end = 0;
    }"""
        dense_phase_filter = (
            f"            if (row_phase == 3 ? row_type != {limit_type} : (row_phase == 4 && row_type == {limit_type})) "
            "continue;"
        )
    else:
        phase_bounds = ""
        dense_phase_filter = ""

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
    // Row ranges of the main dense and free-body contact passes and of the velocity-limit pass.
    int dense_main_end = m_dense;
    int mf_main_end = mf_contact_end;
    int velocity_limit_pass = 1;
{phase_bounds}

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
        if (dense_main_end > 0) {{
{dense_prefetch_init}
        }}
        for (int i = 0; i < dense_main_end; i++) {{
{dense_consume}
            if (i + 1 < dense_main_end) {{
                int next_jy_base = jy_world_base + (i + 1) * {D};
{dense_prefetch_next}
            }}

            int row_type = s_meta_dense[i] & {type_mask};
            if (row_type == {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) continue;
{dense_phase_filter}
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
        if (mf_main_end > 0) {{
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
        for (int i = 0; i < mf_main_end; i++) {{
            int4 meta = pre_meta;
            float cur_Ja = pre_Ja;
            float cur_Jb = pre_Jb;
            float cur_MiJta = pre_MiJta;
            float cur_MiJtb = pre_MiJtb;
            if (i + 1 < mf_main_end) {{
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
        if (velocity_limit_pass) {{
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
        row_phase: int,
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
        row_phase: int,
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
            row_phase,
            v_out,
        )

    name = (
        f"pgs_solve_mf_gs_{max_constraints}_{mf_max_constraints}_{max_world_dofs}"
        f"_vlim{int(has_dense_velocity_limit_rows)}{'' if shared_metadata else '_gmeta'}"
        f"{'_phased' if row_phases else ''}"
    )
    pgs_solve_mf_gs.__name__ = name
    pgs_solve_mf_gs.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_gs)


# ---------------------------------------------------------------------------
# Propagation contact response kernels
# ---------------------------------------------------------------------------

# Static shared memory one warp may use for the per-joint staging of the native
# propagation tree kernels [B].
_PROPAGATION_TREE_SHARED_BYTES = 40 * 1024
# 32-bit words of per-joint staging in the impulse propagation kernel (the larger of
# the two staged kernels).
_PROPAGATION_TREE_WORDS_PER_JOINT = 31


def _propagation_tree_kernel_fits(max_joints: int) -> bool:
    """Return whether an articulation's joints fit the native tree kernels' shared staging."""
    return 4 * _PROPAGATION_TREE_WORDS_PER_JOINT * max(int(max_joints), 1) <= _PROPAGATION_TREE_SHARED_BYTES


@cache
def _get_pgs_solve_propagation_contact_kernel(
    propagation_max_constraints: int, device_arch: str, worlds_per_block: int = 1
) -> "wp.Kernel":
    """Build the one-warp-per-world Gauss-Seidel sweep over the propagation rows.

    One launch performs one sweep. Lanes 0-5 handle body A and lanes 6-11 body B of
    each row; an impulse updates the touched bodies' twists with the row's ``M^-1 J^T``
    and accumulates ``J^T`` impulses on them, which the tree propagation applies to the
    joint velocities afterwards. Friction rows follow their normal row and solve both
    tangent impulses together on the Coulomb disk (``FRICTION_PAIR_CUDA``).

    The kernel uses no shared memory, so ``worlds_per_block`` packs several one-warp
    worlds into one block to reach full warp occupancy. Every per-row operand except the
    body twists is prefetched one row ahead.
    """
    _ = device_arch
    M = propagation_max_constraints
    W = max(int(worlds_per_block), 1)
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)

    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const int world = tile * {W} + (threadIdx.x >> 5);
    if (world >= world_count) return;
    int m = propagation_constraint_count.data[world];
    if (m > {M}) m = {M};
    const int world_base = world * {M};

    int pf_type = 0;
    float pf_eff = 0.0f;
    float pf_rhs = 0.0f;
    int pf_ba = -1;
    int pf_bb = -1;
    float pf_J = 0.0f;
    float pf_MiJt = 0.0f;
    if (m > 0) {{
        pf_type = propagation_row_type.data[world_base];
        pf_eff = propagation_eff_mass_inv.data[world_base];
        pf_rhs = propagation_rhs.data[world_base];
        pf_ba = propagation_body_a.data[world_base];
        pf_bb = propagation_body_b.data[world_base];
        if (lane < 6) {{
            pf_J = propagation_J_a.data[world_base * 6 + lane];
            pf_MiJt = propagation_MiJt_a.data[world_base * 6 + lane];
        }} else if (lane < 12) {{
            pf_J = propagation_J_b.data[world_base * 6 + lane - 6];
            pf_MiJt = propagation_MiJt_b.data[world_base * 6 + lane - 6];
        }}
    }}
    for (int i = 0; i < m; ++i) {{
        const int off = world_base + i;
        const int row_type = pf_type;
        const float eff_inv = pf_eff;
        const float row_rhs = pf_rhs;
        const int ba = pf_ba;
        const int bb = pf_bb;
        const float my_J = pf_J;
        const float my_MiJt = pf_MiJt;
        if (i + 1 < m) {{
            const int off_n = off + 1;
            pf_type = propagation_row_type.data[off_n];
            pf_eff = propagation_eff_mass_inv.data[off_n];
            pf_rhs = propagation_rhs.data[off_n];
            pf_ba = propagation_body_a.data[off_n];
            pf_bb = propagation_body_b.data[off_n];
            if (lane < 6) {{
                pf_J = propagation_J_a.data[off_n * 6 + lane];
                pf_MiJt = propagation_MiJt_a.data[off_n * 6 + lane];
            }} else if (lane < 12) {{
                pf_J = propagation_J_b.data[off_n * 6 + lane - 6];
                pf_MiJt = propagation_MiJt_b.data[off_n * 6 + lane - 6];
            }}
        }}
        if (eff_inv <= 0.0f && row_type != {friction_type}) {{
            __syncwarp();
            continue;
        }}

        int active_body = -1;
        int active_k = -1;
        float partial = 0.0f;
        if (lane < 6 && ba >= 0) {{
            active_body = ba;
            active_k = lane;
            partial = my_J * propagation_body_qd.data[ba * 6 + lane];
        }} else if (lane >= 6 && lane < 12 && bb >= 0) {{
            active_body = bb;
            active_k = lane - 6;
            partial = my_J * propagation_body_qd.data[bb * 6 + lane - 6];
        }}
        for (int offset = 16; offset > 0; offset >>= 1) {{
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }}

        float delta_impulse = 0.0f;
        int sib = -1;
        float sib_delta = 0.0f;
        if (lane == 0) {{
            const float residual = partial + row_rhs;
            const float old_impulse = propagation_impulses.data[off];
            float new_impulse = old_impulse + omega * (-residual * eff_inv);
            if (row_type == {contact_type}) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
            }} else if (row_type == {friction_type}) {{
                const int parent_idx = propagation_row_parent.data[off];
                const float radius = fmaxf(propagation_row_mu.data[off] * propagation_impulses.data[world_base + parent_idx], 0.0f);
                if (i != parent_idx + 1) {{
                    // The first tangent row solves both tangents of its contact.
                    new_impulse = old_impulse;
                }} else {{
                    sib = parent_idx + 2;
                    const int sib_off = world_base + sib;
                    const float other = propagation_impulses.data[sib_off];
                    float sibling_residual = propagation_rhs.data[sib_off];
                    float cross = 0.0f;
                    for (int k = 0; k < 6; ++k) {{
                        if (ba >= 0) {{
                            sibling_residual += propagation_J_a.data[sib_off * 6 + k] * propagation_body_qd.data[ba * 6 + k];
                            cross += propagation_J_a.data[off * 6 + k] * propagation_MiJt_a.data[sib_off * 6 + k];
                        }}
                        if (bb >= 0) {{
                            sibling_residual += propagation_J_b.data[sib_off * 6 + k] * propagation_body_qd.data[bb * 6 + k];
                            cross += propagation_J_b.data[off * 6 + k] * propagation_MiJt_b.data[sib_off * 6 + k];
                        }}
                    }}
                    const float inv_sib = propagation_eff_mass_inv.data[sib_off];
                    float2 pair = friction_pair_candidate(eff_inv > 0.0f ? 1.0f / eff_inv : 0.0f, cross,
                        inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f, residual, sibling_residual, old_impulse, other,
                        radius, omega);
                    const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                    const float scale = mag > radius ? radius / mag : 1.0f;
                    new_impulse = pair.x * scale;
                    const float sib_new = pair.y * scale;
                    sib_delta = sib_new - other;
                    propagation_impulses.data[sib_off] = sib_new;
                }}
            }}
            delta_impulse = new_impulse - old_impulse;
            propagation_impulses.data[off] = new_impulse;
        }}

        sib = __shfl_sync(0xffffffff, sib, 0);
        sib_delta = __shfl_sync(0xffffffff, sib_delta, 0);
        if (sib_delta != 0.0f) {{
            const int sib_off = world_base + sib;
            const int sib_ba = propagation_body_a.data[sib_off];
            const int sib_bb = propagation_body_b.data[sib_off];
            if (lane < 6 && sib_ba >= 0) {{
                propagation_body_qd.data[sib_ba * 6 + lane] += propagation_MiJt_a.data[sib_off * 6 + lane] * sib_delta;
                propagation_body_impulses.data[sib_ba * 6 + lane] += propagation_J_a.data[sib_off * 6 + lane] * sib_delta;
            }} else if (lane >= 6 && lane < 12 && sib_bb >= 0) {{
                const int k = lane - 6;
                propagation_body_qd.data[sib_bb * 6 + k] += propagation_MiJt_b.data[sib_off * 6 + k] * sib_delta;
                propagation_body_impulses.data[sib_bb * 6 + k] += propagation_J_b.data[sib_off * 6 + k] * sib_delta;
            }}
        }}
        delta_impulse = __shfl_sync(0xffffffff, delta_impulse, 0);
        if (delta_impulse != 0.0f && active_body >= 0) {{
            propagation_body_qd.data[active_body * 6 + active_k] += my_MiJt * delta_impulse;
            propagation_body_impulses.data[active_body * 6 + active_k] += my_J * delta_impulse;
        }}
        __syncwarp();
    }}
#endif
"""
    snippet = snippet.replace("#if defined(__CUDA_ARCH__)", "#if defined(__CUDA_ARCH__)\n" + FRICTION_PAIR_CUDA, 1)

    @wp.func_native(snippet)
    def pgs_solve_propagation_native(
        tile: int,
        world_count: int,
        propagation_constraint_count: wp.array[int],
        propagation_body_a: wp.array2d[int],
        propagation_body_b: wp.array2d[int],
        propagation_MiJt_a: wp.array3d[float],
        propagation_MiJt_b: wp.array3d[float],
        propagation_J_a: wp.array3d[float],
        propagation_J_b: wp.array3d[float],
        propagation_eff_mass_inv: wp.array2d[float],
        propagation_rhs: wp.array2d[float],
        propagation_row_type: wp.array2d[int],
        propagation_row_parent: wp.array2d[int],
        propagation_row_mu: wp.array2d[float],
        omega: float,
        propagation_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        propagation_body_impulses: wp.array2d[float],
    ): ...

    def pgs_solve_propagation(
        world_count: int,
        propagation_constraint_count: wp.array[int],
        propagation_body_a: wp.array2d[int],
        propagation_body_b: wp.array2d[int],
        propagation_MiJt_a: wp.array3d[float],
        propagation_MiJt_b: wp.array3d[float],
        propagation_J_a: wp.array3d[float],
        propagation_J_b: wp.array3d[float],
        propagation_eff_mass_inv: wp.array2d[float],
        propagation_rhs: wp.array2d[float],
        propagation_row_type: wp.array2d[int],
        propagation_row_parent: wp.array2d[int],
        propagation_row_mu: wp.array2d[float],
        omega: float,
        propagation_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        propagation_body_impulses: wp.array2d[float],
    ):
        tile, _lane = wp.tid()
        pgs_solve_propagation_native(
            tile,
            world_count,
            propagation_constraint_count,
            propagation_body_a,
            propagation_body_b,
            propagation_MiJt_a,
            propagation_MiJt_b,
            propagation_J_a,
            propagation_J_b,
            propagation_eff_mass_inv,
            propagation_rhs,
            propagation_row_type,
            propagation_row_parent,
            propagation_row_mu,
            omega,
            propagation_impulses,
            propagation_body_qd,
            propagation_body_impulses,
        )

    name = f"pgs_solve_propagation_contact_{propagation_max_constraints}_w{W}"
    pgs_solve_propagation.__name__ = name
    pgs_solve_propagation.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_propagation)


@cache
def _get_pgs_solve_propagation_full_iteration_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    propagation_max_constraints: int,
    max_propagation_bodies: int,
    target_size: int,
    device_arch: str,
) -> "wp.Kernel":
    """Build the one-warp-per-world kernel that runs every propagation Gauss-Seidel iteration.

    Each iteration keeps the order of the separately launched schedule: the dense
    joint-limit rows, the dense and free-body contact rows, the propagation rows with
    their tree propagation, then the dense and free-body velocity-limit rows. The world
    velocity stays in shared memory for all iterations. The tree passes cover the
    articulations of ``target_size`` response DOFs, the only articulated size group the
    fused response supports.
    """
    _ = device_arch
    M_D = max_constraints
    M_MF = mf_max_constraints
    D = max_world_dofs
    M_PROP = propagation_max_constraints
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    vlimit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)

    snippet = f"""
#if defined(__CUDA_ARCH__)
    const unsigned MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;

    int m_dense = world_constraint_count.data[world];
    int m_mf = mf_constraint_count.data[world];
    int m_prop = propagation_constraint_count.data[world];
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
    if (m_prop > {M_PROP}) m_prop = {M_PROP};
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;

    const int dof_map_base = world * {D};
    const int off_dense = world * {M_D};
    const int off_mf = world * {M_MF};
    const int off_meta = off_mf * 4;
    const int jy_world_base = world * {M_D} * {D};
    const int mf6_base = world * {M_MF} * 6;
    const int prop_world_base = world * {M_PROP};
    const int world_art_begin = world_group_art_start.data[world];
    const int world_art_end = world_group_art_start.data[world + 1];
    int n_bodies = propagation_body_count.data[world];
    if (n_bodies > max_propagation_bodies) n_bodies = max_propagation_bodies;
    const int body_base = world * max_propagation_bodies;

    __shared__ float s_v[{D}];
    __shared__ float s_lam_dense[{M_D}];
    __shared__ float s_rhs_dense[{M_D}];
    __shared__ float s_diag_dense[{M_D}];
    __shared__ int s_rtype_dense[{M_D}];
    __shared__ int s_parent_dense[{M_D}];
    __shared__ float s_mu_dense[{M_D}];
    __shared__ float s_lam_mf[{M_MF}];

    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}
    for (int i = lane; i < m_dense; i += 32) {{
        const int off = off_dense + i;
        s_lam_dense[i] = world_impulses.data[off];
        s_rhs_dense[i] = rhs_bias.data[off];
        s_diag_dense[i] = world_diag.data[off];
        s_rtype_dense[i] = world_row_type.data[off];
        s_parent_dense[i] = world_row_parent.data[off];
        s_mu_dense[i] = world_row_mu.data[off];
    }}
    for (int i = lane; i < m_mf; i += 32) {{
        s_lam_mf[i] = mf_impulses.data[off_mf + i];
    }}
    __syncwarp(MASK);

    for (int iter = 0; iter < iterations; ++iter) {{
        for (int stage = 0; stage < 3; ++stage) {{
            // Stage 0: joint-limit rows; stage 1: contact rows; stage 2: velocity limits.
            for (int i = 0; i < m_dense; ++i) {{
                const int row_type = s_rtype_dense[i];
                if (stage == 0 && row_type != {limit_type}) continue;
                if (stage == 1 && row_type != {contact_type} && row_type != {friction_type}) continue;
                if (stage == 2 && row_type != {vlimit_type}) continue;
                const float denom = s_diag_dense[i];
                if (denom <= 0.0f && row_type != {friction_type}) continue;

                const int row_base = jy_world_base + i * {D};
                float my_sum = 0.0f;
                for (int d = lane; d < {D}; d += 32) my_sum += J_world.data[row_base + d] * s_v[d];
                for (int shfl = 16; shfl > 0; shfl >>= 1) my_sum += __shfl_down_sync(MASK, my_sum, shfl);
                const float jv = __shfl_sync(MASK, my_sum, 0);

                const float old_impulse = s_lam_dense[i];
                const float residual = jv + s_rhs_dense[i];
                float new_impulse = old_impulse + omega * (denom > 0.0f ? -residual / denom : 0.0f);
                float delta_impulse = 0.0f;
                if (row_type == {vlimit_type}) {{
                    // Stateless projection onto the velocity box.
                    delta_impulse = residual < 0.0f ? -residual / denom : 0.0f;
                    new_impulse = delta_impulse;
                }} else if (row_type == {friction_type}) {{
                    const int parent_idx = s_parent_dense[i];
                    if (i != parent_idx + 1) {{
                        new_impulse = old_impulse;
                    }} else {{
                        const int sib = parent_idx + 2;
                        const int sib_row_base = jy_world_base + sib * {D};
                        const float radius = fmaxf(s_mu_dense[i] * s_lam_dense[parent_idx], 0.0f);
                        float sibling_residual = 0.0f;
                        float cross = 0.0f;
                        for (int d = lane; d < {D}; d += 32) {{
                            sibling_residual += J_world.data[sib_row_base + d] * s_v[d];
                            cross += J_world.data[row_base + d] * Y_world.data[sib_row_base + d];
                        }}
                        for (int shfl = 16; shfl > 0; shfl >>= 1) {{
                            sibling_residual += __shfl_down_sync(MASK, sibling_residual, shfl);
                            cross += __shfl_down_sync(MASK, cross, shfl);
                        }}
                        sibling_residual = __shfl_sync(MASK, sibling_residual, 0) + s_rhs_dense[sib];
                        cross = __shfl_sync(MASK, cross, 0);
                        float2 pair = friction_pair_candidate(denom, cross, s_diag_dense[sib],
                            residual, sibling_residual, old_impulse, s_lam_dense[sib], radius, omega);
                        const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                        const float scale = mag > radius ? radius / mag : 1.0f;
                        new_impulse = pair.x * scale;
                        const float sib_delta = pair.y * scale - s_lam_dense[sib];
                        __syncwarp(MASK);
                        if (lane == 0) s_lam_dense[sib] = pair.y * scale;
                        if (sib_delta != 0.0f) {{
                            for (int d = lane; d < {D}; d += 32) s_v[d] += Y_world.data[sib_row_base + d] * sib_delta;
                        }}
                    }}
                    delta_impulse = new_impulse - old_impulse;
                }} else {{
                    // Contact and joint-limit rows are unilateral.
                    if (new_impulse < 0.0f) new_impulse = 0.0f;
                    delta_impulse = new_impulse - old_impulse;
                }}
                __syncwarp(MASK);
                if (lane == 0) s_lam_dense[i] = new_impulse;
                if (delta_impulse != 0.0f) {{
                    for (int d = lane; d < {D}; d += 32) s_v[d] += Y_world.data[row_base + d] * delta_impulse;
                }}
                __syncwarp(MASK);
            }}

            if (stage != 0) {{
                // Free-body rows: [contacts and friction] in stage 1, [velocity limits] in stage 2.
                const int mf_lo = (stage == 2) ? mf_contact_end : 0;
                const int mf_hi = (stage == 1) ? mf_contact_end : m_mf;
                for (int i = mf_lo; i < mf_hi; ++i) {{
                    const int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + i * 4]);
                    const int dof_a = meta.x >> 16;
                    const int dof_b = (meta.x << 16) >> 16;
                    const float mf_diag = __int_as_float(meta.y);
                    const int packed_tp = meta.w;
                    const int mf_rt = packed_tp & 0xFFFF;
                    const int mf_par = packed_tp >> 16;
                    if (stage == 2 && mf_rt != {vlimit_type}) continue;
                    if (mf_rt == {friction_type} && i != mf_par + 1) continue;
                    if (mf_diag <= 0.0f && mf_rt != {friction_type}) continue;
                    float radius = 0.0f;
                    if (mf_rt == {friction_type}) {{
                        radius = fmaxf(mf_row_mu.data[off_mf + i] * s_lam_mf[mf_par], 0.0f);
                        if (radius == 0.0f && s_lam_mf[i] == 0.0f && s_lam_mf[i + 1] == 0.0f) continue;
                    }}
                    const int row_mf6 = mf6_base + i * 6;
                    float my_sum = 0.0f;
                    if (lane < 6 && dof_a >= 0) my_sum = mf_J_a.data[row_mf6 + lane] * s_v[dof_a + lane];
                    if (lane >= 6 && lane < 12 && dof_b >= 0) my_sum = mf_J_b.data[row_mf6 + lane - 6] * s_v[dof_b + lane - 6];
                    for (int shfl = 16; shfl > 0; shfl >>= 1) my_sum += __shfl_down_sync(MASK, my_sum, shfl);
                    const float residual = __shfl_sync(MASK, my_sum, 0) + __int_as_float(meta.z);
                    const float old_impulse = s_lam_mf[i];
                    float new_impulse = old_impulse + omega * (-residual * mf_diag);
                    if (mf_rt == {contact_type}) {{
                        if (new_impulse < 0.0f) new_impulse = 0.0f;
                    }} else if (mf_rt == {vlimit_type}) {{
                        new_impulse = residual < 0.0f ? -residual * mf_diag : 0.0f;
                    }} else if (mf_rt == {friction_type}) {{
                        const int sib = mf_par + 2;
                        const int sib_mf6 = mf6_base + sib * 6;
                        float2 pair = make_float2(0.0f, 0.0f);
                        if (radius > 0.0f) {{
                            float sibling_residual = 0.0f;
                            float cross = 0.0f;
                            if (lane < 6 && dof_a >= 0) {{
                                sibling_residual = mf_J_a.data[sib_mf6 + lane] * s_v[dof_a + lane];
                                cross = mf_J_a.data[row_mf6 + lane] * mf_MiJt_a.data[sib_mf6 + lane];
                            }}
                            if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                                sibling_residual = mf_J_b.data[sib_mf6 + lane - 6] * s_v[dof_b + lane - 6];
                                cross = mf_J_b.data[row_mf6 + lane - 6] * mf_MiJt_b.data[sib_mf6 + lane - 6];
                            }}
                            for (int shfl = 16; shfl > 0; shfl >>= 1) {{
                                sibling_residual += __shfl_down_sync(MASK, sibling_residual, shfl);
                                cross += __shfl_down_sync(MASK, cross, shfl);
                            }}
                            sibling_residual = __shfl_sync(MASK, sibling_residual, 0)
                                + __int_as_float(mf_meta.data[off_meta + sib * 4 + 2]);
                            cross = __shfl_sync(MASK, cross, 0);
                            const float inv_sib = __int_as_float(mf_meta.data[off_meta + sib * 4 + 1]);
                            pair = friction_pair_candidate(mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f, cross,
                                inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f, residual, sibling_residual, old_impulse,
                                s_lam_mf[sib], radius, omega);
                        }}
                        const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                        const float scale = mag > radius ? radius / mag : 1.0f;
                        new_impulse = pair.x * scale;
                        const float sib_delta = pair.y * scale - s_lam_mf[sib];
                        __syncwarp(MASK);
                        if (lane == 0) s_lam_mf[sib] = pair.y * scale;
                        if (sib_delta != 0.0f) {{
                            if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                            if (lane >= 6 && lane < 12 && dof_b >= 0)
                                s_v[dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                        }}
                    }}
                    const float delta_impulse = mf_rt == {vlimit_type} ? new_impulse : new_impulse - old_impulse;
                    __syncwarp(MASK);
                    if (lane == 0) s_lam_mf[i] = new_impulse;
                    if (delta_impulse != 0.0f) {{
                        if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += mf_MiJt_a.data[row_mf6 + lane] * delta_impulse;
                        if (lane >= 6 && lane < 12 && dof_b >= 0)
                            s_v[dof_b + lane - 6] += mf_MiJt_b.data[row_mf6 + lane - 6] * delta_impulse;
                    }}
                    __syncwarp(MASK);
                }}
            }}

            if (stage != 1) continue;

            // Link twists of the world's articulations from the current world velocity.
            for (int group_idx = world_art_begin; group_idx < world_art_end; ++group_idx) {{
                const int art = world_group_to_art.data[group_idx];
                const int local_base = articulation_world_dof_offset.data[art] - articulation_dof_start.data[art];
                for (int joint = articulation_start.data[art]; joint < articulation_start.data[art + 1]; ++joint) {{
                    const int child = joint_child.data[joint];
                    const int parent = joint_parent.data[joint];
                    if (lane < 6) {{
                        float value = 0.0f;
                        if (parent >= 0) {{
                            value = propagation_body_qd.data[parent * 6 + lane];
                            const float ex = propagation_body_com_rel.data[child * 3 + 0] - propagation_body_com_rel.data[parent * 3 + 0];
                            const float ey = propagation_body_com_rel.data[child * 3 + 1] - propagation_body_com_rel.data[parent * 3 + 1];
                            const float ez = propagation_body_com_rel.data[child * 3 + 2] - propagation_body_com_rel.data[parent * 3 + 2];
                            const float wx = propagation_body_qd.data[parent * 6 + 3];
                            const float wy = propagation_body_qd.data[parent * 6 + 4];
                            const float wz = propagation_body_qd.data[parent * 6 + 5];
                            if (lane == 0) value += wy * ez - wz * ey;
                            else if (lane == 1) value += wz * ex - wx * ez;
                            else if (lane == 2) value += wx * ey - wy * ex;
                        }}
                        for (int gdof = joint_qd_start.data[joint]; gdof < joint_qd_start.data[joint + 1]; ++gdof) {{
                            const int local_dof = gdof + local_base;
                            if (local_dof >= 0 && local_dof < {D})
                                value += propagation_joint_S_flat.data[gdof * 6 + lane] * s_v[local_dof];
                        }}
                        propagation_body_qd.data[child * 6 + lane] = value;
                    }}
                    __syncwarp(MASK);
                }}
            }}
            // Free-body twists are their generalized velocities (both at the center of mass).
            for (int local_body = 0; local_body < n_bodies; ++local_body) {{
                const int body = propagation_body_list.data[body_base + local_body];
                const int art = body >= 0 ? body_to_articulation.data[body] : -1;
                if (art >= 0 && is_free_rigid.data[art] != 0 && lane < 6) {{
                    const int local_dof = articulation_world_dof_offset.data[art];
                    if (local_dof >= 0 && local_dof + 5 < {D})
                        propagation_body_qd.data[body * 6 + lane] = s_v[local_dof + lane];
                }}
                __syncwarp(MASK);
            }}

            // Propagation rows.
            for (int i = 0; i < m_prop; ++i) {{
                const int off = prop_world_base + i;
                const int row_type = propagation_row_type.data[off];
                const float eff_inv = propagation_eff_mass_inv.data[off];
                if (eff_inv <= 0.0f && row_type != {friction_type}) continue;
                const int ba = propagation_body_a.data[off];
                const int bb = propagation_body_b.data[off];
                int active_body = -1;
                int active_k = -1;
                float active_J = 0.0f;
                float active_MiJt = 0.0f;
                float partial = 0.0f;
                if (lane < 6 && ba >= 0) {{
                    active_body = ba;
                    active_k = lane;
                    active_J = propagation_J_a.data[off * 6 + lane];
                    active_MiJt = propagation_MiJt_a.data[off * 6 + lane];
                    partial = active_J * propagation_body_qd.data[ba * 6 + lane];
                }} else if (lane >= 6 && lane < 12 && bb >= 0) {{
                    active_body = bb;
                    active_k = lane - 6;
                    active_J = propagation_J_b.data[off * 6 + lane - 6];
                    active_MiJt = propagation_MiJt_b.data[off * 6 + lane - 6];
                    partial = active_J * propagation_body_qd.data[bb * 6 + lane - 6];
                }}
                for (int shfl = 16; shfl > 0; shfl >>= 1) partial += __shfl_down_sync(MASK, partial, shfl);

                float delta_impulse = 0.0f;
                int sib = -1;
                float sib_delta = 0.0f;
                if (lane == 0) {{
                    const float residual = partial + propagation_rhs.data[off];
                    const float old_impulse = propagation_impulses.data[off];
                    float new_impulse = old_impulse + omega * (-residual * eff_inv);
                    if (row_type == {contact_type}) {{
                        if (new_impulse < 0.0f) new_impulse = 0.0f;
                    }} else if (row_type == {friction_type}) {{
                        const int parent_idx = propagation_row_parent.data[off];
                        const float radius = fmaxf(
                            propagation_row_mu.data[off] * propagation_impulses.data[prop_world_base + parent_idx], 0.0f);
                        if (i != parent_idx + 1) {{
                            new_impulse = old_impulse;
                        }} else {{
                            sib = parent_idx + 2;
                            const int sib_off = prop_world_base + sib;
                            const float other = propagation_impulses.data[sib_off];
                            float sibling_residual = propagation_rhs.data[sib_off];
                            float cross = 0.0f;
                            for (int k = 0; k < 6; ++k) {{
                                if (ba >= 0) {{
                                    sibling_residual += propagation_J_a.data[sib_off * 6 + k] * propagation_body_qd.data[ba * 6 + k];
                                    cross += propagation_J_a.data[off * 6 + k] * propagation_MiJt_a.data[sib_off * 6 + k];
                                }}
                                if (bb >= 0) {{
                                    sibling_residual += propagation_J_b.data[sib_off * 6 + k] * propagation_body_qd.data[bb * 6 + k];
                                    cross += propagation_J_b.data[off * 6 + k] * propagation_MiJt_b.data[sib_off * 6 + k];
                                }}
                            }}
                            const float inv_sib = propagation_eff_mass_inv.data[sib_off];
                            float2 pair = friction_pair_candidate(eff_inv > 0.0f ? 1.0f / eff_inv : 0.0f, cross,
                                inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f, residual, sibling_residual, old_impulse, other,
                                radius, omega);
                            const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                            const float scale = mag > radius ? radius / mag : 1.0f;
                            new_impulse = pair.x * scale;
                            const float sib_new = pair.y * scale;
                            sib_delta = sib_new - other;
                            propagation_impulses.data[sib_off] = sib_new;
                        }}
                    }}
                    delta_impulse = new_impulse - old_impulse;
                    propagation_impulses.data[off] = new_impulse;
                }}
                sib = __shfl_sync(MASK, sib, 0);
                sib_delta = __shfl_sync(MASK, sib_delta, 0);
                if (sib_delta != 0.0f) {{
                    const int sib_off = prop_world_base + sib;
                    const int sib_ba = propagation_body_a.data[sib_off];
                    const int sib_bb = propagation_body_b.data[sib_off];
                    if (lane < 6 && sib_ba >= 0) {{
                        propagation_body_qd.data[sib_ba * 6 + lane] += propagation_MiJt_a.data[sib_off * 6 + lane] * sib_delta;
                        propagation_body_impulses.data[sib_ba * 6 + lane] += propagation_J_a.data[sib_off * 6 + lane] * sib_delta;
                    }} else if (lane >= 6 && lane < 12 && sib_bb >= 0) {{
                        const int k = lane - 6;
                        propagation_body_qd.data[sib_bb * 6 + k] += propagation_MiJt_b.data[sib_off * 6 + k] * sib_delta;
                        propagation_body_impulses.data[sib_bb * 6 + k] += propagation_J_b.data[sib_off * 6 + k] * sib_delta;
                    }}
                }}
                delta_impulse = __shfl_sync(MASK, delta_impulse, 0);
                if (delta_impulse != 0.0f && active_body >= 0) {{
                    propagation_body_qd.data[active_body * 6 + active_k] += active_MiJt * delta_impulse;
                    propagation_body_impulses.data[active_body * 6 + active_k] += active_J * delta_impulse;
                }}
                __syncwarp(MASK);
            }}

            // Articulated-body propagation of the accumulated link impulses to the joint velocities.
            for (int group_idx = world_art_begin; group_idx < world_art_end; ++group_idx) {{
                const int art = world_group_to_art.data[group_idx];
                const int local_base = articulation_world_dof_offset.data[art] - articulation_dof_start.data[art];
                const int joint_start = articulation_start.data[art];
                const int joint_end = articulation_start.data[art + 1];
                int has_impulse = 0;
                for (int joint = joint_start; joint < joint_end; ++joint) {{
                    const int body = joint_child.data[joint];
                    if (lane < 6) {{
                        const float impulse = propagation_body_impulses.data[body * 6 + lane];
                        if (impulse != 0.0f) has_impulse = 1;
                        propagation_tree_pA.data[body * 6 + lane] = -impulse;
                        propagation_tree_body_delta.data[body * 6 + lane] = 0.0f;
                        propagation_body_impulses.data[body * 6 + lane] = 0.0f;
                    }}
                    if (lane == 0) {{
                        for (int dof = joint_qd_start.data[joint]; dof < joint_qd_start.data[joint + 1]; ++dof) {{
                            propagation_tree_u.data[dof] = 0.0f;
                            propagation_tree_qdd.data[dof] = 0.0f;
                        }}
                    }}
                    __syncwarp(MASK);
                }}
                if (__any_sync(MASK, has_impulse != 0) == 0) continue;

                for (int offset = 0; offset < joint_end - joint_start; ++offset) {{
                    const int joint = joint_end - 1 - offset;
                    const int child = joint_child.data[joint];
                    const int parent = joint_parent.data[joint];
                    const int dof_start = joint_qd_start.data[joint];
                    const int dof_count = joint_qd_start.data[joint + 1] - dof_start;
                    for (int a = 0; a < dof_count; ++a) {{
                        const int gdof = dof_start + a;
                        float u = 0.0f;
                        if (lane < 6) u = -propagation_joint_S_flat.data[gdof * 6 + lane] * propagation_tree_pA.data[child * 6 + lane];
                        for (int shfl = 16; shfl > 0; shfl >>= 1) u += __shfl_down_sync(MASK, u, shfl);
                        u = __shfl_sync(MASK, u, 0);
                        if (lane == 0) propagation_tree_u.data[gdof] = u;
                    }}
                    __syncwarp(MASK);
                    if (parent >= 0 && lane < 6) {{
                        // pA(child) + U D^-1 u, translated from the child's to the parent's center of mass.
                        float p[6];
                        for (int r = 0; r < 6; ++r) p[r] = propagation_tree_pA.data[child * 6 + r];
                        for (int a = 0; a < dof_count; ++a) {{
                            float coeff = 0.0f;
                            for (int b = 0; b < dof_count; ++b)
                                coeff += propagation_tree_D_inv.data[joint * 36 + a * 6 + b] * propagation_tree_u.data[dof_start + b];
                            for (int r = 0; r < 6; ++r) p[r] += propagation_tree_U.data[(dof_start + a) * 6 + r] * coeff;
                        }}
                        const float ex = propagation_body_com_rel.data[child * 3 + 0] - propagation_body_com_rel.data[parent * 3 + 0];
                        const float ey = propagation_body_com_rel.data[child * 3 + 1] - propagation_body_com_rel.data[parent * 3 + 1];
                        const float ez = propagation_body_com_rel.data[child * 3 + 2] - propagation_body_com_rel.data[parent * 3 + 2];
                        float propagated = p[lane];
                        if (lane == 3) propagated += ey * p[2] - ez * p[1];
                        else if (lane == 4) propagated += ez * p[0] - ex * p[2];
                        else if (lane == 5) propagated += ex * p[1] - ey * p[0];
                        propagation_tree_pA.data[parent * 6 + lane] += propagated;
                    }}
                    __syncwarp(MASK);
                }}

                for (int joint = joint_start; joint < joint_end; ++joint) {{
                    const int child = joint_child.data[joint];
                    const int parent = joint_parent.data[joint];
                    const int dof_start = joint_qd_start.data[joint];
                    const int dof_count = joint_qd_start.data[joint + 1] - dof_start;
                    float parent_delta = 0.0f;
                    if (parent >= 0 && lane < 6) {{
                        parent_delta = propagation_tree_body_delta.data[parent * 6 + lane];
                        const float ex = propagation_body_com_rel.data[child * 3 + 0] - propagation_body_com_rel.data[parent * 3 + 0];
                        const float ey = propagation_body_com_rel.data[child * 3 + 1] - propagation_body_com_rel.data[parent * 3 + 1];
                        const float ez = propagation_body_com_rel.data[child * 3 + 2] - propagation_body_com_rel.data[parent * 3 + 2];
                        const float wx = propagation_tree_body_delta.data[parent * 6 + 3];
                        const float wy = propagation_tree_body_delta.data[parent * 6 + 4];
                        const float wz = propagation_tree_body_delta.data[parent * 6 + 5];
                        if (lane == 0) parent_delta += wy * ez - wz * ey;
                        else if (lane == 1) parent_delta += wz * ex - wx * ez;
                        else if (lane == 2) parent_delta += wx * ey - wy * ex;
                    }}
                    for (int a = 0; a < dof_count; ++a) {{
                        float qdd = 0.0f;
                        for (int b = 0; b < dof_count; ++b) {{
                            float parent_term = 0.0f;
                            if (parent >= 0 && lane < 6) parent_term = propagation_tree_U.data[(dof_start + b) * 6 + lane] * parent_delta;
                            for (int shfl = 16; shfl > 0; shfl >>= 1) parent_term += __shfl_down_sync(MASK, parent_term, shfl);
                            parent_term = __shfl_sync(MASK, parent_term, 0);
                            qdd += propagation_tree_D_inv.data[joint * 36 + a * 6 + b] * (propagation_tree_u.data[dof_start + b] - parent_term);
                        }}
                        if (lane == 0) {{
                            propagation_tree_qdd.data[dof_start + a] = qdd;
                            const int local_dof = dof_start + a + local_base;
                            if (local_dof >= 0 && local_dof < {D}) s_v[local_dof] += qdd;
                        }}
                    }}
                    __syncwarp(MASK);
                    if (lane < 6) {{
                        float value = parent_delta;
                        for (int a = 0; a < dof_count; ++a)
                            value += propagation_joint_S_flat.data[(dof_start + a) * 6 + lane] * propagation_tree_qdd.data[dof_start + a];
                        propagation_tree_body_delta.data[child * 6 + lane] = value;
                    }}
                    __syncwarp(MASK);
                }}
            }}
            // Free-body twists are written back to the world velocity.
            for (int local_body = 0; local_body < n_bodies; ++local_body) {{
                const int body = propagation_body_list.data[body_base + local_body];
                const int art = body >= 0 ? body_to_articulation.data[body] : -1;
                if (art >= 0 && is_free_rigid.data[art] != 0 && lane < 6) {{
                    const int local_dof = articulation_world_dof_offset.data[art];
                    if (local_dof >= 0 && local_dof + 5 < {D})
                        s_v[local_dof + lane] = propagation_body_qd.data[body * 6 + lane];
                    propagation_body_impulses.data[body * 6 + lane] = 0.0f;
                }}
                __syncwarp(MASK);
            }}
        }}
    }}

    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) v_out.data[global_dof] = s_v[d];
    }}
    for (int i = lane; i < m_dense; i += 32) world_impulses.data[off_dense + i] = s_lam_dense[i];
    for (int i = lane; i < m_mf; i += 32) mf_impulses.data[off_mf + i] = s_lam_mf[i];
#endif
"""
    snippet = snippet.replace("#if defined(__CUDA_ARCH__)", "#if defined(__CUDA_ARCH__)\n" + FRICTION_PAIR_CUDA, 1)

    @wp.func_native(snippet)
    def pgs_solve_propagation_full_iteration_native(
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
        propagation_constraint_count: wp.array[int],
        propagation_body_count: wp.array[int],
        propagation_body_list: wp.array2d[int],
        max_propagation_bodies: int,
        propagation_body_a: wp.array2d[int],
        propagation_body_b: wp.array2d[int],
        propagation_MiJt_a: wp.array3d[float],
        propagation_MiJt_b: wp.array3d[float],
        propagation_J_a: wp.array3d[float],
        propagation_J_b: wp.array3d[float],
        propagation_eff_mass_inv: wp.array2d[float],
        propagation_rhs: wp.array2d[float],
        propagation_row_type: wp.array2d[int],
        propagation_row_parent: wp.array2d[int],
        propagation_row_mu: wp.array2d[float],
        body_to_articulation: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_group_art_start: wp.array[int],
        world_group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        iterations: int,
        omega: float,
        propagation_impulses: wp.array2d[float],
        propagation_tree_pA: wp.array2d[float],
        propagation_tree_u: wp.array[float],
        propagation_tree_qdd: wp.array[float],
        propagation_tree_body_delta: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        propagation_body_impulses: wp.array2d[float],
        v_out: wp.array[float],
    ): ...

    def pgs_solve_propagation_full_iteration(
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
        propagation_constraint_count: wp.array[int],
        propagation_body_count: wp.array[int],
        propagation_body_list: wp.array2d[int],
        max_propagation_bodies: int,
        propagation_body_a: wp.array2d[int],
        propagation_body_b: wp.array2d[int],
        propagation_MiJt_a: wp.array3d[float],
        propagation_MiJt_b: wp.array3d[float],
        propagation_J_a: wp.array3d[float],
        propagation_J_b: wp.array3d[float],
        propagation_eff_mass_inv: wp.array2d[float],
        propagation_rhs: wp.array2d[float],
        propagation_row_type: wp.array2d[int],
        propagation_row_parent: wp.array2d[int],
        propagation_row_mu: wp.array2d[float],
        body_to_articulation: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_group_art_start: wp.array[int],
        world_group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        iterations: int,
        omega: float,
        propagation_impulses: wp.array2d[float],
        propagation_tree_pA: wp.array2d[float],
        propagation_tree_u: wp.array[float],
        propagation_tree_qdd: wp.array[float],
        propagation_tree_body_delta: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        propagation_body_impulses: wp.array2d[float],
        v_out: wp.array[float],
    ):
        world, _lane = wp.tid()
        pgs_solve_propagation_full_iteration_native(
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
            propagation_constraint_count,
            propagation_body_count,
            propagation_body_list,
            max_propagation_bodies,
            propagation_body_a,
            propagation_body_b,
            propagation_MiJt_a,
            propagation_MiJt_b,
            propagation_J_a,
            propagation_J_b,
            propagation_eff_mass_inv,
            propagation_rhs,
            propagation_row_type,
            propagation_row_parent,
            propagation_row_mu,
            body_to_articulation,
            is_free_rigid,
            articulation_dof_start,
            articulation_world_dof_offset,
            world_group_art_start,
            world_group_to_art,
            articulation_start,
            joint_parent,
            joint_child,
            joint_qd_start,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            propagation_tree_U,
            propagation_tree_D_inv,
            iterations,
            omega,
            propagation_impulses,
            propagation_tree_pA,
            propagation_tree_u,
            propagation_tree_qdd,
            propagation_tree_body_delta,
            propagation_body_qd,
            propagation_body_impulses,
            v_out,
        )

    name = f"pgs_solve_propagation_full_iteration_{M_D}_{M_MF}_{D}_{M_PROP}_{max_propagation_bodies}_{target_size}"
    pgs_solve_propagation_full_iteration.__name__ = name
    pgs_solve_propagation_full_iteration.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_propagation_full_iteration)


@cache
def _get_factor_propagation_tree_revolute_kernel(
    size: int, device_arch: str, *, has_free_root: bool = False
) -> "wp.Kernel":
    """Build a one-warp propagation tree factor kernel for 0/1-DOF joint trees.

    ``has_free_root`` compiles a special case for articulations whose FIRST
    joint is a multi-DOF world-rooted joint (dof_count <= 6, e.g. a floating
    base) while every remaining joint stays 0/1-DOF. The root joint's U rows,
    joint-space block D (Cholesky-factored and inverted serially on lane 0 —
    once per articulation per factor pass), and the U D^-1 U^T reduction of
    the child's articulated inertia replicate the generic per-DOF math of the
    serial ``factor_propagation_tree_for_size`` kernel. The root has no
    parent, so it never reduces into a parent inertia. Eligibility (root
    joint first, topological joint order) is verified host-side at solver
    build time.
    """
    snippet = """
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;
//FREE_ROOT_DECL
    const int art = group_to_art.data[group_idx];
    const int joint_start = articulation_start.data[art];
    const int joint_end = articulation_start.data[art + 1];

    for (int joint = joint_start; joint < joint_end; ++joint) {
        const int body = joint_child.data[joint];
        const wp::quat_t<wp::float32> q = wp::quat_inverse(body_q_com.data[body].q);
        const wp::mat_t<3, 3, wp::float32> R = wp::quat_to_matrix(q);
        const wp::mat_t<6, 6, wp::float32> I_local = body_I_m.data[body];

        for (int elem = lane; elem < 36; elem += 32) {
            const int row = elem / 6;
            const int col = elem - row * 6;
            const int row_block = (row >= 3) ? 3 : 0;
            const int col_block = (col >= 3) ? 3 : 0;
            const int rr = row - row_block;
            const int cc = col - col_block;
            float value = 0.0f;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    value += R.data[a][rr] * I_local.data[row_block + a][col_block + b] * R.data[b][cc];
                }
            }
            propagation_tree_Ia.data[body * 36 + elem] = value;
        }
        __syncwarp(mask);
    }

    for (int offset = 0; offset < joint_end - joint_start; ++offset) {
        const int joint = joint_end - 1 - offset;
        const int child = joint_child.data[joint];
        const int parent = joint_parent.data[joint];
        const int dof_start = joint_qd_start.data[joint];
        const int dof_end = joint_qd_start.data[joint + 1];
        const int has_dof = (dof_start < dof_end);
        const int gdof = dof_start;
//FREE_ROOT_FACTOR
        float d_part = 0.0f;
        if (has_dof && lane < 6) {
            float u = 0.0f;
            for (int c = 0; c < 6; ++c) {
                u += propagation_tree_Ia.data[child * 36 + lane * 6 + c]
                    * propagation_joint_S_flat.data[gdof * 6 + c];
            }
            propagation_tree_U.data[gdof * 6 + lane] = u;
            d_part = propagation_joint_S_flat.data[gdof * 6 + lane] * u;
        }
        for (int shfl = 16; shfl > 0; shfl >>= 1) {
            d_part += __shfl_down_sync(mask, d_part, shfl);
        }
        if (has_dof && lane == 0) {
            float d_value = d_part + joint_armature.data[gdof];
            const int aug_count = aug_row_counts.data[art];
            for (int aug_i = 0; aug_i < aug_count; ++aug_i) {
                const int row_index = art * max_dofs + aug_i;
                if (aug_row_dof_index.data[row_index] == gdof) {
                    const float K = aug_row_K.data[row_index];
                    if (K > 0.0f) {
                        d_value += K;
                    }
                }
            }
            if (d_value <= 1.0e-12f) {
                d_value = 1.0e-12f;
            }
            propagation_tree_D_chol.data[joint * 36] = sqrtf(d_value);
            propagation_tree_D_inv.data[joint * 36] = 1.0f / d_value;
        }
        __syncwarp(mask);

        const float inv_d = has_dof ? propagation_tree_D_inv.data[joint * 36] : 0.0f;
        for (int elem = lane; elem < 36; elem += 32) {
            const int row = elem / 6;
            const int col = elem - row * 6;
            float reduced = propagation_tree_Ia.data[child * 36 + elem];
            if (has_dof) {
                reduced -= propagation_tree_U.data[gdof * 6 + row] * inv_d
                    * propagation_tree_U.data[gdof * 6 + col];
            }
            propagation_tree_Ia.data[child * 36 + elem] = reduced;
        }
        __syncwarp(mask);

        if (parent >= 0) {
            const float ex = propagation_body_com_rel.data[child * 3 + 0]
                - propagation_body_com_rel.data[parent * 3 + 0];
            const float ey = propagation_body_com_rel.data[child * 3 + 1]
                - propagation_body_com_rel.data[parent * 3 + 1];
            const float ez = propagation_body_com_rel.data[child * 3 + 2]
                - propagation_body_com_rel.data[parent * 3 + 2];

            for (int elem = lane; elem < 36; elem += 32) {
                const int row = elem / 6;
                const int col = elem - row * 6;

                float v0 = 0.0f;
                float v1 = 0.0f;
                float v2 = 0.0f;
                float v3 = 0.0f;
                float v4 = 0.0f;
                float v5 = 0.0f;
                if (col == 0) {
                    v0 = 1.0f;
                } else if (col == 1) {
                    v1 = 1.0f;
                } else if (col == 2) {
                    v2 = 1.0f;
                } else if (col == 3) {
                    v1 = -ez;
                    v2 = ey;
                    v3 = 1.0f;
                } else if (col == 4) {
                    v0 = ez;
                    v2 = -ex;
                    v4 = 1.0f;
                } else {
                    v0 = -ey;
                    v1 = ex;
                    v5 = 1.0f;
                }

                const float w0 = propagation_tree_Ia.data[child * 36 + 0 * 6 + 0] * v0
                    + propagation_tree_Ia.data[child * 36 + 0 * 6 + 1] * v1
                    + propagation_tree_Ia.data[child * 36 + 0 * 6 + 2] * v2
                    + propagation_tree_Ia.data[child * 36 + 0 * 6 + 3] * v3
                    + propagation_tree_Ia.data[child * 36 + 0 * 6 + 4] * v4
                    + propagation_tree_Ia.data[child * 36 + 0 * 6 + 5] * v5;
                const float w1 = propagation_tree_Ia.data[child * 36 + 1 * 6 + 0] * v0
                    + propagation_tree_Ia.data[child * 36 + 1 * 6 + 1] * v1
                    + propagation_tree_Ia.data[child * 36 + 1 * 6 + 2] * v2
                    + propagation_tree_Ia.data[child * 36 + 1 * 6 + 3] * v3
                    + propagation_tree_Ia.data[child * 36 + 1 * 6 + 4] * v4
                    + propagation_tree_Ia.data[child * 36 + 1 * 6 + 5] * v5;
                const float w2 = propagation_tree_Ia.data[child * 36 + 2 * 6 + 0] * v0
                    + propagation_tree_Ia.data[child * 36 + 2 * 6 + 1] * v1
                    + propagation_tree_Ia.data[child * 36 + 2 * 6 + 2] * v2
                    + propagation_tree_Ia.data[child * 36 + 2 * 6 + 3] * v3
                    + propagation_tree_Ia.data[child * 36 + 2 * 6 + 4] * v4
                    + propagation_tree_Ia.data[child * 36 + 2 * 6 + 5] * v5;
                const float w3 = propagation_tree_Ia.data[child * 36 + 3 * 6 + 0] * v0
                    + propagation_tree_Ia.data[child * 36 + 3 * 6 + 1] * v1
                    + propagation_tree_Ia.data[child * 36 + 3 * 6 + 2] * v2
                    + propagation_tree_Ia.data[child * 36 + 3 * 6 + 3] * v3
                    + propagation_tree_Ia.data[child * 36 + 3 * 6 + 4] * v4
                    + propagation_tree_Ia.data[child * 36 + 3 * 6 + 5] * v5;
                const float w4 = propagation_tree_Ia.data[child * 36 + 4 * 6 + 0] * v0
                    + propagation_tree_Ia.data[child * 36 + 4 * 6 + 1] * v1
                    + propagation_tree_Ia.data[child * 36 + 4 * 6 + 2] * v2
                    + propagation_tree_Ia.data[child * 36 + 4 * 6 + 3] * v3
                    + propagation_tree_Ia.data[child * 36 + 4 * 6 + 4] * v4
                    + propagation_tree_Ia.data[child * 36 + 4 * 6 + 5] * v5;
                const float w5 = propagation_tree_Ia.data[child * 36 + 5 * 6 + 0] * v0
                    + propagation_tree_Ia.data[child * 36 + 5 * 6 + 1] * v1
                    + propagation_tree_Ia.data[child * 36 + 5 * 6 + 2] * v2
                    + propagation_tree_Ia.data[child * 36 + 5 * 6 + 3] * v3
                    + propagation_tree_Ia.data[child * 36 + 5 * 6 + 4] * v4
                    + propagation_tree_Ia.data[child * 36 + 5 * 6 + 5] * v5;

                float value = 0.0f;
                if (row == 0) {
                    value = w0;
                } else if (row == 1) {
                    value = w1;
                } else if (row == 2) {
                    value = w2;
                } else if (row == 3) {
                    value = w3 + ey * w2 - ez * w1;
                } else if (row == 4) {
                    value = w4 + ez * w0 - ex * w2;
                } else {
                    value = w5 + ex * w1 - ey * w0;
                }
                propagation_tree_Ia.data[parent * 36 + elem] += value;
            }
        }
        __syncwarp(mask);
    }
#endif
"""

    if has_free_root:
        root_decl = """
    __shared__ float s_D_root[36];
    __shared__ float s_Dinv_root[36];
"""
        root_factor = """
        if (joint == joint_start) {
            const int dc = dof_end - dof_start;
            // U rows for all root DOFs: U_a = Ia(child) S_a.
            for (int elem = lane; elem < dc * 6; elem += 32) {
                const int a = elem / 6;
                const int r = elem - a * 6;
                float u_val = 0.0f;
                for (int c = 0; c < 6; ++c) {
                    u_val += propagation_tree_Ia.data[child * 36 + r * 6 + c]
                        * propagation_joint_S_flat.data[(dof_start + a) * 6 + c];
                }
                propagation_tree_U.data[(dof_start + a) * 6 + r] = u_val;
            }
            __syncwarp(mask);
            // D = S^T U (+ armature and aug-row stiffness on the diagonal).
            for (int elem = lane; elem < dc * dc; elem += 32) {
                const int a = elem / dc;
                const int b = elem - a * dc;
                float value = 0.0f;
                for (int r = 0; r < 6; ++r) {
                    value += propagation_joint_S_flat.data[(dof_start + a) * 6 + r]
                        * propagation_tree_U.data[(dof_start + b) * 6 + r];
                }
                if (a == b) {
                    value += joint_armature.data[dof_start + a];
                    const int aug_count = aug_row_counts.data[art];
                    for (int aug_i = 0; aug_i < aug_count; ++aug_i) {
                        const int row_index = art * max_dofs + aug_i;
                        if (aug_row_dof_index.data[row_index] == dof_start + a) {
                            const float K = aug_row_K.data[row_index];
                            if (K > 0.0f) {
                                value += K;
                            }
                        }
                    }
                }
                s_D_root[a * 6 + b] = value;
            }
            __syncwarp(mask);
            if (lane == 0) {
                // In-place Cholesky mirroring the serial kernel: the lower
                // triangle holds the factor, the upper keeps raw D entries.
                for (int jj = 0; jj < dc; ++jj) {
                    float s = s_D_root[jj * 6 + jj];
                    for (int k = 0; k < jj; ++k) {
                        const float cjk = s_D_root[jj * 6 + k];
                        s -= cjk * cjk;
                    }
                    if (s <= 1.0e-12f) {
                        s = 1.0e-12f;
                    }
                    s = sqrtf(s);
                    s_D_root[jj * 6 + jj] = s;
                    const float inv_s = 1.0f / s;
                    for (int i = jj + 1; i < dc; ++i) {
                        float v = s_D_root[i * 6 + jj];
                        for (int k = 0; k < jj; ++k) {
                            v -= s_D_root[i * 6 + k] * s_D_root[jj * 6 + k];
                        }
                        s_D_root[i * 6 + jj] = v * inv_s;
                    }
                }
                // Invert D column by column via forward/backward solves.
                for (int col = 0; col < dc; ++col) {
                    for (int i = 0; i < dc; ++i) {
                        float v = (i == col) ? 1.0f : 0.0f;
                        for (int k = 0; k < i; ++k) {
                            v -= s_D_root[i * 6 + k] * s_Dinv_root[k * 6 + col];
                        }
                        s_Dinv_root[i * 6 + col] = v / s_D_root[i * 6 + i];
                    }
                    for (int i = dc - 1; i >= 0; --i) {
                        float v = s_Dinv_root[i * 6 + col];
                        for (int k = i + 1; k < dc; ++k) {
                            v -= s_D_root[k * 6 + i] * s_Dinv_root[k * 6 + col];
                        }
                        s_Dinv_root[i * 6 + col] = v / s_D_root[i * 6 + i];
                    }
                }
            }
            __syncwarp(mask);
            for (int elem = lane; elem < dc * dc; elem += 32) {
                const int a = elem / dc;
                const int b = elem - a * dc;
                propagation_tree_D_chol.data[joint * 36 + a * 6 + b] = s_D_root[a * 6 + b];
                propagation_tree_D_inv.data[joint * 36 + a * 6 + b] = s_Dinv_root[a * 6 + b];
            }
            __syncwarp(mask);
            // Ia(child) -= U D^-1 U^T; the root has no parent to reduce into.
            for (int elem = lane; elem < 36; elem += 32) {
                const int row = elem / 6;
                const int col = elem - row * 6;
                float reduced = propagation_tree_Ia.data[child * 36 + elem];
                for (int a = 0; a < dc; ++a) {
                    const float U_ar = propagation_tree_U.data[(dof_start + a) * 6 + row];
                    for (int b = 0; b < dc; ++b) {
                        reduced -= U_ar * s_Dinv_root[a * 6 + b]
                            * propagation_tree_U.data[(dof_start + b) * 6 + col];
                    }
                }
                propagation_tree_Ia.data[child * 36 + elem] = reduced;
            }
            __syncwarp(mask);
            continue;
        }
"""
        snippet = snippet.replace("//FREE_ROOT_DECL", root_decl).replace("//FREE_ROOT_FACTOR", root_factor)
    else:
        snippet = snippet.replace("//FREE_ROOT_DECL", "").replace("//FREE_ROOT_FACTOR", "")

    @wp.func_native(snippet)
    def factor_propagation_tree_revolute_native(
        group_idx: int,
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        joint_armature: wp.array[float],
        max_dofs: int,
        aug_row_counts: wp.array[int],
        aug_row_dof_index: wp.array[int],
        aug_row_K: wp.array[float],
        body_I_m: wp.array[wp.spatial_matrix],
        body_q_com: wp.array[wp.transform],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_Ia: wp.array3d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_chol: wp.array3d[float],
        propagation_tree_D_inv: wp.array3d[float],
    ): ...

    def factor_propagation_tree_revolute_template(
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        joint_armature: wp.array[float],
        max_dofs: int,
        aug_row_counts: wp.array[int],
        aug_row_dof_index: wp.array[int],
        aug_row_K: wp.array[float],
        body_I_m: wp.array[wp.spatial_matrix],
        body_q_com: wp.array[wp.transform],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_Ia: wp.array3d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_chol: wp.array3d[float],
        propagation_tree_D_inv: wp.array3d[float],
    ):
        group_idx, _lane = wp.tid()
        factor_propagation_tree_revolute_native(
            group_idx,
            group_to_art,
            articulation_start,
            joint_parent,
            joint_child,
            joint_qd_start,
            propagation_joint_S_flat,
            joint_armature,
            max_dofs,
            aug_row_counts,
            aug_row_dof_index,
            aug_row_K,
            body_I_m,
            body_q_com,
            propagation_body_com_rel,
            propagation_tree_Ia,
            propagation_tree_U,
            propagation_tree_D_chol,
            propagation_tree_D_inv,
        )

    name = f"factor_propagation_tree_revolute_{size}"
    if has_free_root:
        name += "_fr"
    factor_propagation_tree_revolute_template.__name__ = name
    factor_propagation_tree_revolute_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(factor_propagation_tree_revolute_template)


@cache
def _get_propagation_tree_body_response_revolute_kernel(
    size: int, device_arch: str, *, has_free_root: bool = False
) -> "wp.Kernel":
    """Build a one-warp body response kernel for 0/1-DOF joint trees.

    Writes the per-link 6x6 COM response for EVERY link of the articulation
    (overwriting ``propagation_tree_Ia`` in joint order so each link can reuse
    its parent's already-computed response).

    ``has_free_root`` compiles a special case for articulations whose FIRST
    joint is a multi-DOF world-rooted joint (dof_count <= 6): the root link's
    response to a unit test wrench is S D^-1 S^T (no parent recursion), the
    multi-DOF generalization of the existing 1-DOF root branch. Descendant
    0/1-DOF links keep the scalar recursion unchanged. The serial fallback
    for free-root size groups is the path-restricted generic response kernel
    (``compute_propagation_tree_body_response_for_size``), which fills only
    contact-active bodies — a subset of what this kernel writes, with equal
    values.
    """
    snippet = """
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;
    const int art = group_to_art.data[group_idx];
    const int joint_start = articulation_start.data[art];
    const int joint_end = articulation_start.data[art + 1];

    for (int joint = joint_start; joint < joint_end; ++joint) {
//FREE_ROOT_RESPONSE
        const int child = joint_child.data[joint];
        const int parent = joint_parent.data[joint];
        const int dof_start = joint_qd_start.data[joint];
        const int dof_end = joint_qd_start.data[joint + 1];
        const int has_dof = (dof_start < dof_end);
        const int gdof = dof_start;
        const float inv_d = has_dof ? propagation_tree_D_inv.data[joint * 36] : 0.0f;

        float ex = 0.0f;
        float ey = 0.0f;
        float ez = 0.0f;
        if (parent >= 0) {
            ex = propagation_body_com_rel.data[child * 3 + 0] - propagation_body_com_rel.data[parent * 3 + 0];
            ey = propagation_body_com_rel.data[child * 3 + 1] - propagation_body_com_rel.data[parent * 3 + 1];
            ez = propagation_body_com_rel.data[child * 3 + 2] - propagation_body_com_rel.data[parent * 3 + 2];
        }

        for (int elem = lane; elem < 36; elem += 32) {
            const int row = elem / 6;
            const int col = elem - row * 6;

            float b0 = 0.0f;
            float b1 = 0.0f;
            float b2 = 0.0f;
            float b3 = 0.0f;
            float b4 = 0.0f;
            float b5 = 0.0f;
            if (col == 0) {
                b0 = 1.0f;
            } else if (col == 1) {
                b1 = 1.0f;
            } else if (col == 2) {
                b2 = 1.0f;
            } else if (col == 3) {
                b3 = 1.0f;
            } else if (col == 4) {
                b4 = 1.0f;
            } else {
                b5 = 1.0f;
            }

            const float s_dot_f = has_dof ? propagation_joint_S_flat.data[gdof * 6 + col] : 0.0f;
            float p0 = b0;
            float p1 = b1;
            float p2 = b2;
            float p3 = b3;
            float p4 = b4;
            float p5 = b5;
            if (has_dof) {
                const float scale = inv_d * s_dot_f;
                p0 -= propagation_tree_U.data[gdof * 6 + 0] * scale;
                p1 -= propagation_tree_U.data[gdof * 6 + 1] * scale;
                p2 -= propagation_tree_U.data[gdof * 6 + 2] * scale;
                p3 -= propagation_tree_U.data[gdof * 6 + 3] * scale;
                p4 -= propagation_tree_U.data[gdof * 6 + 4] * scale;
                p5 -= propagation_tree_U.data[gdof * 6 + 5] * scale;
            }

            float parent_delta0 = 0.0f;
            float parent_delta1 = 0.0f;
            float parent_delta2 = 0.0f;
            float parent_delta3 = 0.0f;
            float parent_delta4 = 0.0f;
            float parent_delta5 = 0.0f;
            if (parent >= 0) {
                const float pp0 = p0;
                const float pp1 = p1;
                const float pp2 = p2;
                const float pp3 = p3 + ey * p2 - ez * p1;
                const float pp4 = p4 + ez * p0 - ex * p2;
                const float pp5 = p5 + ex * p1 - ey * p0;

                parent_delta0 = propagation_tree_Ia.data[parent * 36 + 0 * 6 + 0] * pp0
                    + propagation_tree_Ia.data[parent * 36 + 0 * 6 + 1] * pp1
                    + propagation_tree_Ia.data[parent * 36 + 0 * 6 + 2] * pp2
                    + propagation_tree_Ia.data[parent * 36 + 0 * 6 + 3] * pp3
                    + propagation_tree_Ia.data[parent * 36 + 0 * 6 + 4] * pp4
                    + propagation_tree_Ia.data[parent * 36 + 0 * 6 + 5] * pp5;
                parent_delta1 = propagation_tree_Ia.data[parent * 36 + 1 * 6 + 0] * pp0
                    + propagation_tree_Ia.data[parent * 36 + 1 * 6 + 1] * pp1
                    + propagation_tree_Ia.data[parent * 36 + 1 * 6 + 2] * pp2
                    + propagation_tree_Ia.data[parent * 36 + 1 * 6 + 3] * pp3
                    + propagation_tree_Ia.data[parent * 36 + 1 * 6 + 4] * pp4
                    + propagation_tree_Ia.data[parent * 36 + 1 * 6 + 5] * pp5;
                parent_delta2 = propagation_tree_Ia.data[parent * 36 + 2 * 6 + 0] * pp0
                    + propagation_tree_Ia.data[parent * 36 + 2 * 6 + 1] * pp1
                    + propagation_tree_Ia.data[parent * 36 + 2 * 6 + 2] * pp2
                    + propagation_tree_Ia.data[parent * 36 + 2 * 6 + 3] * pp3
                    + propagation_tree_Ia.data[parent * 36 + 2 * 6 + 4] * pp4
                    + propagation_tree_Ia.data[parent * 36 + 2 * 6 + 5] * pp5;
                parent_delta3 = propagation_tree_Ia.data[parent * 36 + 3 * 6 + 0] * pp0
                    + propagation_tree_Ia.data[parent * 36 + 3 * 6 + 1] * pp1
                    + propagation_tree_Ia.data[parent * 36 + 3 * 6 + 2] * pp2
                    + propagation_tree_Ia.data[parent * 36 + 3 * 6 + 3] * pp3
                    + propagation_tree_Ia.data[parent * 36 + 3 * 6 + 4] * pp4
                    + propagation_tree_Ia.data[parent * 36 + 3 * 6 + 5] * pp5;
                parent_delta4 = propagation_tree_Ia.data[parent * 36 + 4 * 6 + 0] * pp0
                    + propagation_tree_Ia.data[parent * 36 + 4 * 6 + 1] * pp1
                    + propagation_tree_Ia.data[parent * 36 + 4 * 6 + 2] * pp2
                    + propagation_tree_Ia.data[parent * 36 + 4 * 6 + 3] * pp3
                    + propagation_tree_Ia.data[parent * 36 + 4 * 6 + 4] * pp4
                    + propagation_tree_Ia.data[parent * 36 + 4 * 6 + 5] * pp5;
                parent_delta5 = propagation_tree_Ia.data[parent * 36 + 5 * 6 + 0] * pp0
                    + propagation_tree_Ia.data[parent * 36 + 5 * 6 + 1] * pp1
                    + propagation_tree_Ia.data[parent * 36 + 5 * 6 + 2] * pp2
                    + propagation_tree_Ia.data[parent * 36 + 5 * 6 + 3] * pp3
                    + propagation_tree_Ia.data[parent * 36 + 5 * 6 + 4] * pp4
                    + propagation_tree_Ia.data[parent * 36 + 5 * 6 + 5] * pp5;

                const float wx = parent_delta3;
                const float wy = parent_delta4;
                const float wz = parent_delta5;
                parent_delta0 += wy * ez - wz * ey;
                parent_delta1 += wz * ex - wx * ez;
                parent_delta2 += wx * ey - wy * ex;
            }

            float parent_dot = 0.0f;
            float qdd = 0.0f;
            if (has_dof) {
                parent_dot = propagation_tree_U.data[gdof * 6 + 0] * parent_delta0
                    + propagation_tree_U.data[gdof * 6 + 1] * parent_delta1
                    + propagation_tree_U.data[gdof * 6 + 2] * parent_delta2
                    + propagation_tree_U.data[gdof * 6 + 3] * parent_delta3
                    + propagation_tree_U.data[gdof * 6 + 4] * parent_delta4
                    + propagation_tree_U.data[gdof * 6 + 5] * parent_delta5;
                qdd = inv_d * (s_dot_f - parent_dot);
            }

            float value = 0.0f;
            if (row == 0) {
                value = parent_delta0;
            } else if (row == 1) {
                value = parent_delta1;
            } else if (row == 2) {
                value = parent_delta2;
            } else if (row == 3) {
                value = parent_delta3;
            } else if (row == 4) {
                value = parent_delta4;
            } else {
                value = parent_delta5;
            }
            if (has_dof) {
                value += propagation_joint_S_flat.data[gdof * 6 + row] * qdd;
            }

            propagation_tree_Ia.data[child * 36 + elem] = value;
            propagation_body_response.data[child * 36 + elem] = value;
        }
        __syncwarp(mask);
    }
#endif
"""

    if has_free_root:
        root_response = """
        if (joint == joint_start) {
            // Multi-DOF root link: response = S D^-1 S^T (no parent).
            const int root_dof_start = joint_qd_start.data[joint];
            const int root_dc = joint_qd_start.data[joint + 1] - root_dof_start;
            const int root_child = joint_child.data[joint];
            for (int elem = lane; elem < 36; elem += 32) {
                const int row = elem / 6;
                const int col = elem - row * 6;
                float value = 0.0f;
                for (int a = 0; a < root_dc; ++a) {
                    float qdd = 0.0f;
                    for (int b = 0; b < root_dc; ++b) {
                        qdd += propagation_tree_D_inv.data[joint * 36 + a * 6 + b]
                            * propagation_joint_S_flat.data[(root_dof_start + b) * 6 + col];
                    }
                    value += propagation_joint_S_flat.data[(root_dof_start + a) * 6 + row] * qdd;
                }
                propagation_tree_Ia.data[root_child * 36 + elem] = value;
                propagation_body_response.data[root_child * 36 + elem] = value;
            }
            __syncwarp(mask);
            continue;
        }
"""
        snippet = snippet.replace("//FREE_ROOT_RESPONSE", root_response)
    else:
        snippet = snippet.replace("//FREE_ROOT_RESPONSE", "")

    @wp.func_native(snippet)
    def propagation_tree_body_response_revolute_native(
        group_idx: int,
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_tree_Ia: wp.array3d[float],
        propagation_tree_body_delta: wp.array2d[float],
        propagation_body_response: wp.array3d[float],
    ): ...

    def propagation_tree_body_response_revolute_template(
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_tree_Ia: wp.array3d[float],
        propagation_tree_body_delta: wp.array2d[float],
        propagation_body_response: wp.array3d[float],
    ):
        group_idx, _lane = wp.tid()
        propagation_tree_body_response_revolute_native(
            group_idx,
            group_to_art,
            articulation_start,
            joint_parent,
            joint_child,
            joint_qd_start,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            propagation_tree_U,
            propagation_tree_D_inv,
            propagation_tree_Ia,
            propagation_tree_body_delta,
            propagation_body_response,
        )

    name = f"propagation_tree_body_response_revolute_{size}"
    if has_free_root:
        name += "_fr"
    propagation_tree_body_response_revolute_template.__name__ = name
    propagation_tree_body_response_revolute_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(propagation_tree_body_response_revolute_template)


@cache
def _get_propagate_tree_impulses_revolute_kernel(
    size: int, max_joints: int, device_arch: str, *, has_free_root: bool = False
) -> "wp.Kernel":
    """Build a one-warp propagation tree propagation kernel for 0/1-DOF joint trees.

    All per-joint metadata (indices, motion subspace, U, D_inv, COM offsets)
    is preloaded into shared memory with lane-parallel strided loads, so the
    three serial tree passes are pure on-chip compute plus warp shuffles. Only
    v_out, propagation_body_qd, and the deferred impulse clear touch global
    memory after the preload. This matters because the passes are dependent
    serial chains whose cost is dominated by memory latency per joint.

    ``has_free_root`` compiles a special case for articulations whose FIRST
    joint is a multi-DOF world-rooted joint (e.g. a floating base, up to
    6 DOFs) while every remaining joint stays 0/1-DOF. The root joint gets
    dedicated shared blocks (S_root/U_root/D_inv_root 6x6, u_root/qdd_root 6)
    and replicates the generic per-DOF math of the serial
    ``propagate_tree_impulses_for_size`` kernel; the root has no parent so it
    never participates in the upward/downward parent propagation. Eligibility
    (root joint first, topological joint order) is verified host-side at
    solver build time.
    """
    joints_cap = max(int(max_joints), 1)
    if has_free_root:
        root_decl = """
    __shared__ float s_S_root[36];
    __shared__ float s_U_root[36];
    __shared__ float s_Dinv_root[36];
    __shared__ float s_u_root[6];
    __shared__ float s_qdd_root[6];
    const int root_gdof = joint_qd_start.data[joint_start];
    const int root_dc = joint_qd_start.data[joint_start + 1] - root_gdof;
"""
        root_gdof_guard = " && j != 0"
        root_preload = """
    for (int idx = lane; idx < root_dc * 6; idx += 32) {
        const int a = idx / 6;
        const int comp = idx - a * 6;
        s_S_root[a * 6 + comp] = propagation_joint_S_flat.data[(root_gdof + a) * 6 + comp];
        s_U_root[a * 6 + comp] = propagation_tree_U.data[(root_gdof + a) * 6 + comp];
    }
    for (int idx = lane; idx < root_dc * root_dc; idx += 32) {
        const int a = idx / root_dc;
        const int b = idx - a * root_dc;
        s_Dinv_root[a * 6 + b] = propagation_tree_D_inv.data[joint_start * 36 + a * 6 + b];
    }
"""
        root_backward = """
            if (j == 0) {
                // Multi-DOF root: u_a = -S_a . pA(root); no parent to propagate to.
                if (lane < root_dc) {
                    float u_root = 0.0f;
                    for (int r = 0; r < 6; ++r) {
                        u_root -= s_S_root[lane * 6 + r] * s_pA[r];
                    }
                    s_u_root[lane] = u_root;
                }
                __syncwarp(mask);
                continue;
            }
"""
        root_forward = """
            if (j == 0) {
                // Multi-DOF root: qdd = D^-1 u (no parent term), delta = S qdd.
                if (lane < root_dc) {
                    float qdd_root = 0.0f;
                    for (int b = 0; b < root_dc; ++b) {
                        qdd_root += s_Dinv_root[lane * 6 + b] * s_u_root[b];
                    }
                    s_qdd_root[lane] = qdd_root;
                    v_out.data[root_gdof + lane] += qdd_root;
                }
                __syncwarp(mask);
                if (lane < 6) {
                    float value = 0.0f;
                    for (int a = 0; a < root_dc; ++a) {
                        value += s_S_root[a * 6 + lane] * s_qdd_root[a];
                    }
                    s_bd[lane] = value;
                }
                __syncwarp(mask);
                continue;
            }
"""
        root_recompute = """
            if (j == 0) {
                if (lane < 6) {
                    float value = 0.0f;
                    for (int a = 0; a < root_dc; ++a) {
                        value += s_S_root[a * 6 + lane] * v_out.data[root_gdof + a];
                    }
                    s_pA[lane] = value;
                    propagation_body_qd.data[s_child[0] * 6 + lane] = value;
                    propagation_body_impulses.data[s_child[0] * 6 + lane] = 0.0f;
                }
                __syncwarp(mask);
                continue;
            }
"""
    else:
        root_decl = ""
        root_gdof_guard = ""
        root_preload = ""
        root_backward = ""
        root_forward = ""
        root_recompute = ""
    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;
    const int art = group_to_art.data[group_idx];
    const int joint_start = articulation_start.data[art];
    const int joint_end = articulation_start.data[art + 1];
    const int n_joints = joint_end - joint_start;

    __shared__ float s_pA[{joints_cap} * 6];
    __shared__ float s_bd[{joints_cap} * 6];
    __shared__ float s_u[{joints_cap}];
    __shared__ float s_U[{joints_cap} * 6];
    __shared__ float s_S[{joints_cap} * 6];
    __shared__ float s_e[{joints_cap} * 3];
    __shared__ float s_dinv[{joints_cap}];
    __shared__ int s_child[{joints_cap}];
    __shared__ int s_pslot[{joints_cap}];
    __shared__ int s_gdof[{joints_cap}];
{root_decl}
    for (int j = lane; j < n_joints; j += 32) {{
        const int joint = joint_start + j;
        const int child = joint_child.data[joint];
        const int dof_start = joint_qd_start.data[joint];
        const int has_dof = (dof_start < joint_qd_start.data[joint + 1]);
        s_child[j] = child;
        s_pslot[j] = joint_parent_slot.data[joint];
        s_gdof[j] = (has_dof{root_gdof_guard}) ? dof_start : -1;
        s_dinv[j] = has_dof ? propagation_tree_D_inv.data[joint * 36] : 0.0f;
        s_u[j] = 0.0f;
    }}
    __syncwarp(mask);

    int any_impulse = 0;
    for (int idx = lane; idx < n_joints * 6; idx += 32) {{
        const int j = idx / 6;
        const int comp = idx - j * 6;
        const int gdof = s_gdof[j];
        s_U[idx] = (gdof >= 0) ? propagation_tree_U.data[gdof * 6 + comp] : 0.0f;
        s_S[idx] = (gdof >= 0) ? propagation_joint_S_flat.data[gdof * 6 + comp] : 0.0f;
        const float imp = propagation_body_impulses.data[s_child[j] * 6 + comp];
        s_pA[idx] = -imp;
        if (imp != 0.0f) any_impulse = 1;
    }}
    any_impulse = __any_sync(mask, any_impulse != 0);
    for (int idx = lane; idx < n_joints * 3; idx += 32) {{
        const int j = idx / 3;
        const int comp = idx - j * 3;
        float e = 0.0f;
        if (s_pslot[j] >= 0) {{
            const int parent = joint_parent.data[joint_start + j];
            e = propagation_body_com_rel.data[s_child[j] * 3 + comp]
                - propagation_body_com_rel.data[parent * 3 + comp];
        }}
        s_e[idx] = e;
    }}
{root_preload}
    __syncwarp(mask);

    if (any_impulse != 0) {{
        for (int offset = 0; offset < n_joints; ++offset) {{
            const int j = n_joints - 1 - offset;
{root_backward}
            const int parent_slot = s_pslot[j];
            const int has_dof = (s_gdof[j] >= 0);

            float u = 0.0f;
            if (has_dof && lane < 6) {{
                u = -s_S[j * 6 + lane] * s_pA[j * 6 + lane];
            }}
            for (int shfl = 16; shfl > 0; shfl >>= 1) {{
                u += __shfl_down_sync(mask, u, shfl);
            }}
            u = __shfl_sync(mask, u, 0);
            if (has_dof && lane == 0) {{
                s_u[j] = u;
            }}

            if (parent_slot >= 0 && lane < 6) {{
                const float inv_du = has_dof ? s_dinv[j] * u : 0.0f;
                float propagated = s_pA[j * 6 + lane] + s_U[j * 6 + lane] * inv_du;
                if (lane >= 3) {{
                    const float ex = s_e[j * 3 + 0];
                    const float ey = s_e[j * 3 + 1];
                    const float ez = s_e[j * 3 + 2];
                    const float px = s_pA[j * 6 + 0] + s_U[j * 6 + 0] * inv_du;
                    const float py = s_pA[j * 6 + 1] + s_U[j * 6 + 1] * inv_du;
                    const float pz = s_pA[j * 6 + 2] + s_U[j * 6 + 2] * inv_du;
                    if (lane == 3) propagated += ey * pz - ez * py;
                    else if (lane == 4) propagated += ez * px - ex * pz;
                    else propagated += ex * py - ey * px;
                }}
                s_pA[parent_slot * 6 + lane] += propagated;
            }}
            __syncwarp(mask);
        }}

        for (int j = 0; j < n_joints; ++j) {{
{root_forward}
            const int parent_slot = s_pslot[j];
            const int gdof = s_gdof[j];
            const int has_dof = (gdof >= 0);

            float parent_delta = 0.0f;
            if (parent_slot >= 0 && lane < 6) {{
                parent_delta = s_bd[parent_slot * 6 + lane];
                const float ex = s_e[j * 3 + 0];
                const float ey = s_e[j * 3 + 1];
                const float ez = s_e[j * 3 + 2];
                const float wx = s_bd[parent_slot * 6 + 3];
                const float wy = s_bd[parent_slot * 6 + 4];
                const float wz = s_bd[parent_slot * 6 + 5];
                if (lane == 0) parent_delta += wy * ez - wz * ey;
                else if (lane == 1) parent_delta += wz * ex - wx * ez;
                else if (lane == 2) parent_delta += wx * ey - wy * ex;
            }}

            float qdd = 0.0f;
            if (has_dof) {{
                float parent_term = 0.0f;
                if (parent_slot >= 0 && lane < 6) {{
                    parent_term = s_U[j * 6 + lane] * parent_delta;
                }}
                for (int shfl = 16; shfl > 0; shfl >>= 1) {{
                    parent_term += __shfl_down_sync(mask, parent_term, shfl);
                }}
                parent_term = __shfl_sync(mask, parent_term, 0);
                qdd = s_dinv[j] * (s_u[j] - parent_term);
                if (lane == 0) {{
                    v_out.data[gdof] += qdd;
                }}
                qdd = __shfl_sync(mask, qdd, 0);
            }}

            if (lane < 6) {{
                s_bd[j * 6 + lane] = parent_delta + (has_dof ? s_S[j * 6 + lane] * qdd : 0.0f);
            }}
            __syncwarp(mask);
        }}

        // Recompute full live body velocities from the updated generalized
        // velocities (replacing the row solve's diagonal-response estimates)
        // and clear deferred impulses for the next GS iteration. s_pA is dead
        // after the reverse pass and is reused to stage the full twists. With
        // no deferred impulses v_out and propagation_body_qd are untouched
        // since the previous consistent recompute, so the pass is skipped.
        for (int j = 0; j < n_joints; ++j) {{
{root_recompute}
            const int parent_slot = s_pslot[j];
            const int gdof = s_gdof[j];
            const int child = s_child[j];

            if (lane < 6) {{
                float value = 0.0f;
                if (parent_slot >= 0) {{
                    value = s_pA[parent_slot * 6 + lane];
                    const float ex = s_e[j * 3 + 0];
                    const float ey = s_e[j * 3 + 1];
                    const float ez = s_e[j * 3 + 2];
                    const float wx = s_pA[parent_slot * 6 + 3];
                    const float wy = s_pA[parent_slot * 6 + 4];
                    const float wz = s_pA[parent_slot * 6 + 5];
                    if (lane == 0) value += wy * ez - wz * ey;
                    else if (lane == 1) value += wz * ex - wx * ez;
                    else if (lane == 2) value += wx * ey - wy * ex;
                }}
                if (gdof >= 0) {{
                    value += s_S[j * 6 + lane] * v_out.data[gdof];
                }}
                s_pA[j * 6 + lane] = value;
                propagation_body_qd.data[child * 6 + lane] = value;
                propagation_body_impulses.data[child * 6 + lane] = 0.0f;
            }}
            __syncwarp(mask);
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def propagate_tree_impulses_revolute_native(
        group_idx: int,
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        joint_parent_slot: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_body_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        v_out: wp.array[float],
    ): ...

    def propagate_tree_impulses_revolute_template(
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        joint_parent_slot: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_body_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        v_out: wp.array[float],
    ):
        group_idx, _lane = wp.tid()
        propagate_tree_impulses_revolute_native(
            group_idx,
            group_to_art,
            articulation_start,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_parent_slot,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            propagation_tree_U,
            propagation_tree_D_inv,
            propagation_body_impulses,
            propagation_body_qd,
            v_out,
        )

    name = f"propagate_tree_impulses_revolute_{size}_j{joints_cap}"
    if has_free_root:
        name += "_fr"
    propagate_tree_impulses_revolute_template.__name__ = name
    propagate_tree_impulses_revolute_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(propagate_tree_impulses_revolute_template)


@cache
def _get_refresh_propagation_tree_body_qd_warp_kernel(size: int, max_joints: int, device_arch: str) -> "wp.Kernel":
    """Build a one-warp variant of ``refresh_propagation_tree_body_qd_for_size``.

    The refresh is a single root-to-leaf pass: each link's live COM twist is
    its parent's twist translated to the link COM plus the inbound joint's
    S * v contribution. Per-joint S * v terms (generic over dof_count, so a
    multi-DOF free root needs no special casing), COM edges, and topology are
    preloaded into shared memory with lane-parallel strided loads; the serial
    parent chain then runs on-chip with six lanes per link. The parent twist
    lives in shared memory only — the serial kernel's
    ``propagation_tree_body_delta`` global scratch is not written (no other
    kernel consumes it without re-initializing it first).

    Registered only for size groups whose articulations have the single-DOF
    or free-root shape: those checks also guarantee the topological joint
    order (parents before children) and in-articulation parents that
    ``joint_parent_slot`` indexing relies on.
    """
    joints_cap = max(int(max_joints), 1)
    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;
    const int art = group_to_art.data[group_idx];
    if (force_refresh == 0) {{
        const int world = art_to_world.data[art];
        if (world >= 0 && dense_contact_world_flag.data[world] == 0) {{
            return;
        }}
    }}
    const int joint_start = articulation_start.data[art];
    const int joint_end = articulation_start.data[art + 1];
    const int n_joints = joint_end - joint_start;

    __shared__ float s_bd[{joints_cap} * 6];
    __shared__ float s_Sv[{joints_cap} * 6];
    __shared__ float s_e[{joints_cap} * 3];
    __shared__ int s_child[{joints_cap}];
    __shared__ int s_pslot[{joints_cap}];

    for (int j = lane; j < n_joints; j += 32) {{
        s_child[j] = joint_child.data[joint_start + j];
        s_pslot[j] = joint_parent_slot.data[joint_start + j];
    }}
    __syncwarp(mask);
    for (int idx = lane; idx < n_joints * 6; idx += 32) {{
        const int j = idx / 6;
        const int comp = idx - j * 6;
        const int joint = joint_start + j;
        const int dof_start = joint_qd_start.data[joint];
        const int dof_end = joint_qd_start.data[joint + 1];
        float value = 0.0f;
        for (int dof = dof_start; dof < dof_end; ++dof) {{
            value += propagation_joint_S_flat.data[dof * 6 + comp] * v_out.data[dof];
        }}
        s_Sv[idx] = value;
    }}
    for (int idx = lane; idx < n_joints * 3; idx += 32) {{
        const int j = idx / 3;
        const int comp = idx - j * 3;
        float e = 0.0f;
        if (s_pslot[j] >= 0) {{
            const int parent = joint_parent.data[joint_start + j];
            e = propagation_body_com_rel.data[s_child[j] * 3 + comp]
                - propagation_body_com_rel.data[parent * 3 + comp];
        }}
        s_e[idx] = e;
    }}
    __syncwarp(mask);

    for (int j = 0; j < n_joints; ++j) {{
        if (lane < 6) {{
            float value = s_Sv[j * 6 + lane];
            const int pslot = s_pslot[j];
            if (pslot >= 0) {{
                value += s_bd[pslot * 6 + lane];
                if (lane < 3) {{
                    const float ex = s_e[j * 3 + 0];
                    const float ey = s_e[j * 3 + 1];
                    const float ez = s_e[j * 3 + 2];
                    const float wx = s_bd[pslot * 6 + 3];
                    const float wy = s_bd[pslot * 6 + 4];
                    const float wz = s_bd[pslot * 6 + 5];
                    if (lane == 0) value += wy * ez - wz * ey;
                    else if (lane == 1) value += wz * ex - wx * ez;
                    else value += wx * ey - wy * ex;
                }}
            }}
            s_bd[j * 6 + lane] = value;
            propagation_body_qd.data[s_child[j] * 6 + lane] = value;
        }}
        __syncwarp(mask);
    }}
#endif
"""

    @wp.func_native(snippet)
    def refresh_propagation_tree_body_qd_warp_native(
        group_idx: int,
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        dense_contact_world_flag: wp.array[int],
        force_refresh: int,
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        joint_parent_slot: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        v_out: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ): ...

    def refresh_propagation_tree_body_qd_warp_template(
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        dense_contact_world_flag: wp.array[int],
        force_refresh: int,
        articulation_start: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        joint_parent_slot: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        v_out: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ):
        group_idx, _lane = wp.tid()
        refresh_propagation_tree_body_qd_warp_native(
            group_idx,
            group_to_art,
            art_to_world,
            dense_contact_world_flag,
            force_refresh,
            articulation_start,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_parent_slot,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            v_out,
            propagation_body_qd,
        )

    name = f"refresh_propagation_tree_body_qd_warp_{size}_j{joints_cap}"
    refresh_propagation_tree_body_qd_warp_template.__name__ = name
    refresh_propagation_tree_body_qd_warp_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(refresh_propagation_tree_body_qd_warp_template)
