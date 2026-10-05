# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import re
import warnings
import weakref
from dataclasses import dataclass
from functools import cache
from typing import ClassVar, Literal

import numpy as np
import warp as wp

from ...core.reset import reset_world_selected
from ...core.types import Vec3, override
from ...geometry.flags import ShapeFlags
from ...sim import BodyFlags, Contacts, Control, JointType, Model, ModelBuilder, ModelFlags, State, StateFlags
from ...sim.articulation import eval_fk
from ..solver import SolverBase
from . import contact_compliance as _contact_compliance
from .contact_torsion import (
    configure_contact_torsion,
    prepare_torsion_rows,
    prepare_torsion_velocity_pass,
    torque_sweep_source,
    validate_torsion_step,
)
from .friction import FRICTION_PAIR_CUDA
from .friction_patches import _FrictionPatchState, finish_patch_impulses, link_patch_rows, seed_patch_impulses
from .kernels import (
    CONTACT_GENERATION_NONE,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
    PREELIM_MAX_ROWS,
    ROW_WATERMARK_CONTACT_SLOT,
    ROW_WATERMARK_FAMILY_STRIDE,
    _compute_body_net_wrench,
    _get_tree_fk_kernel,
    _get_tree_tau_kernel,
    accumulate_contact_watermark,
    accumulate_group_diag_worlds,
    accumulate_row_watermarks,
    allocate_connect_slots,
    allocate_joint_velocity_limit_slots,
    allocate_mimic_slots,
    allocate_physx_drive_slots,
    allocate_rigid_velocity_limit_slots,
    allocate_world_contact_slots,
    apply_augmented_mass_diagonal_grouped,
    apply_free_root_angular_damping,
    apply_free_root_transport_to_predictor,
    apply_free_root_velocity_corrections,
    apply_impulses_world_par_dof,
    apply_mf_warmstart_impulses,
    apply_world_contact_restitution,
    apply_world_impulses_to_velocity,
    build_joint_limit_rows,
    build_mass_update_mask,
    build_mf_body_map,
    build_mf_contact_rows,
    build_propagation_body_map,
    build_propagation_contact_rows,
    cholesky_loop,
    compute_com_transforms,
    compute_composite_inertia,
    compute_contact_linear_force_from_impulses,
    compute_delta_and_accumulate,
    compute_mf_body_Hinv,
    compute_mf_effective_mass_and_rhs,
    compute_mf_velocity_rhs,
    compute_mf_world_dof_offsets,
    compute_physx_pgs_drive_desc,
    compute_propagation_body_com_rel,
    compute_propagation_effective_mass_and_rhs,
    compute_propagation_tree_body_response_for_size,
    compute_spatial_inertia,
    compute_velocity_predictor,
    compute_world_contact_bias,
    compute_world_contact_velocity_bias,
    copy_free_rigid_propagation_body_response,
    crba_fill_par_dof,
    delassus_par_row_col,
    diag_from_JY_par_art,
    diag_from_JY_world,
    eval_augmented_drives,
    eval_rigid_fk_id,
    eval_rigid_tau,
    eval_rigid_tau_and_augmented_drives,
    factor_diagonal_mass,
    factor_propagation_tree_for_size,
    finalize_mf_constraint_counts,
    finalize_world_constraint_counts,
    finalize_world_diag_cfm,
    flatten_propagation_joint_S,
    flush_propagation_free_body_qd_to_vout,
    gather_contact_warmstart,
    gather_JY_to_world,
    gather_tau_to_groups,
    hinv_jt_diagonal,
    hinv_jt_par_row,
    integrate_generalized_joints,
    invert_lower_factor_grouped,
    pack_contact_linear_force_as_spatial,
    pgs_solve_loop,
    pgs_solve_mf_loop,
    populate_connect_J_for_size,
    populate_joint_velocity_limit_J_for_size,
    populate_mimic_J_for_size,
    populate_physx_drive_J_for_size,
    populate_rigid_velocity_limit_rows,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    preelim_correct_Y_for_size,
    preelim_project_velocity_for_size,
    preelim_setup_for_size,
    prepare_world_contact_rows,
    prepare_world_impulses,
    prescale_joint_velocity_limits,
    propagate_tree_impulses_for_size,
    refine_same_articulation_propagation_rows,
    refresh_masked_body_inertia,
    refresh_propagation_free_body_qd_from_vout,
    refresh_propagation_tree_body_qd_for_size,
    remove_free_root_transport_from_qdd,
    reset_row_warmstart,
    rhs_accum_world_par_art,
    scatter_augmented_drive_dof_K,
    scatter_qdd_from_groups,
    snapshot_contact_warmstart,
    snapshot_row_warmstart,
    snapshot_step_warmstart,
    solve_diagonal_mass,
    trisolve_loop,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
    vector_add_inplace,
)
from .sleeping import _SleepState
from .sparse_contact import (
    _get_sparse_contact_response_kernel,
    apply_sparse_factor_velocity,
    apply_sparse_free_velocity,
    build_sparse_joint_limit_rows,
)
from .sparse_mass_matrix import _get_crba_sparse_factor_kernel, _SparseMassMatrixPlan, solve_sparse_mass_matrix
from .sparse_pgs import _get_pgs_solve_sparse_kernel

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
# Warps (articulations) per block of the sparse mass-factor kernel, bounded by shared memory.
_SPARSE_FACTOR_WARPS_PER_BLOCK = 4
# Warps (articulations) per block of the warp-parallel composite-inertia reduction.
_COMPOSITE_INERTIA_WARPS_PER_BLOCK = 4
# Largest contact regularization; larger values do not change the float32 weight usefully.
_MAX_CONTACT_REGULARIZATION = 1.0e6
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


def _finite_non_negative(name: str, value) -> float:
    """Return ``value`` as a float, raising :class:`ValueError` unless it is finite and non-negative."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_supported_model(model: Model) -> None:
    """Reject model features this solver does not simulate instead of ignoring them.

    Joint-owned and legacy mimic relationships and loop-closing joints are validated
    when the solver builds their constraint rows.
    """
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
        if model.joint_enabled is not None:
            disabled = ~model.joint_enabled.numpy().astype(bool)
            if model.joint_articulation is not None:
                # A disabled loop-closing joint is a released closure, see set_loop_joint_enabled().
                disabled &= model.joint_articulation.numpy() >= 0
            if np.any(disabled):
                raise NotImplementedError(
                    "SolverFeatherPGS does not support disabled joints in an articulation tree (Model.joint_enabled)."
                )


def _model_has_bilateral_constraints(model: Model) -> bool:
    """Return whether the model has mimic relationships or loop-closing joints.

    These become dense bilateral rows wherever the solver supports them. Disabled joints
    count too, because a released closure can be re-enabled without rebuilding the solver.
    """
    if int(getattr(model, "constraint_mimic_count", 0)):
        return True
    if not model.joint_count:
        return False
    if model.joint_mimic_joint is not None and np.any(model.joint_mimic_joint.numpy() >= 0):
        return True
    if model.joint_articulation is None or not model.body_count:
        return False
    joint_articulation = model.joint_articulation.numpy()
    joint_child = model.joint_child.numpy()
    owned = np.zeros(model.body_count, dtype=bool)
    tree = joint_articulation >= 0
    owned[joint_child[tree & (joint_child >= 0)]] = True
    closure = (~tree) & (joint_child >= 0)
    return bool(np.any(owned[joint_child[closure]]))


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
            "dropped %d contact, friction, mimic or connect rows. Increase dense_max_constraints.\n",
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
    articulation_joint_end: np.ndarray
    loop_joint_articulation: np.ndarray
    world_count: int

    @classmethod
    def build(cls, model: Model, kinematic_dof_mask: np.ndarray) -> "_FeatherPGSModelPlan":
        """Build the immutable response layout from physical model topology.

        Global articulations (world ``-1``) are solved in world 0. An articulation's tree
        is the contiguous prefix of its joint range owned through
        :attr:`~newton.Model.joint_articulation`. A joint without an owner whose child body
        belongs to an articulation is a loop-closing joint of that articulation: it stays
        out of the tree and its mass matrix and is enforced by constraint rows. Unowned
        joints of bodies without an articulation are not simulated.
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

        articulation_joint_end = np.zeros(articulation_count, dtype=np.int32)
        loop_joint_articulation = np.full(model.joint_count, -1, dtype=np.int32)
        if articulation_count and model.joint_count:
            articulation_start = model.articulation_start.numpy()
            joint_parent = model.joint_parent.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            joint_type = model.joint_type.numpy()
            joint_child = model.joint_child.numpy()
            joint_articulation = (
                model.joint_articulation.numpy()
                if model.joint_articulation is not None
                else np.full(model.joint_count, -1, dtype=np.int32)
            )
            body_articulation = np.full(model.body_count, -1, dtype=np.int32)
            owned = (joint_articulation >= 0) & (joint_child >= 0)
            body_articulation[joint_child[owned]] = joint_articulation[owned]
            closure = (joint_articulation < 0) & (joint_child >= 0)
            loop_joint_articulation[closure] = body_articulation[joint_child[closure]]
            for art in range(articulation_count):
                first_joint = int(articulation_start[art])
                last_joint = int(articulation_start[art + 1])
                # A loop joint appended after a later articulation's joints falls into that
                # articulation's range, so the tree is found by ownership, not by position.
                tree_end = first_joint
                for joint in range(first_joint, last_joint):
                    if int(joint_articulation[joint]) != art:
                        continue
                    if joint != tree_end:
                        raise ValueError(
                            f"SolverFeatherPGS: the tree joints of articulation {art} are not contiguous "
                            f"(joint {joint} follows the foreign joint {tree_end})."
                        )
                    tree_end = joint + 1
                articulation_joint_end[art] = tree_end
                first_dof = int(joint_qd_start[first_joint])
                last_dof = int(joint_qd_start[tree_end])
                articulation_dof_start[art] = first_dof
                articulation_dof_count[art] = last_dof - first_dof
                if (
                    tree_end - first_joint == 1
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
            articulation_joint_end,
            loop_joint_articulation,
        )
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays, world_count)


@dataclass(frozen=True)
class _FeatherPGSTreeGroup:
    """Articulations sharing one traversal width."""

    lanes: int
    """Threads per articulation, a power of two."""
    articulation_count: int
    """Number of articulations in this group."""
    max_levels: int
    """Level count of the deepest tree in the group; shorter trees have empty levels."""
    articulations: wp.array[int]
    """Global articulation indices."""
    level_offsets: wp.array2d[int]
    """Segment offsets per articulation and level, shape [articulation_count, max_levels + 1]."""
    segment_offsets: wp.array[int]
    """Offsets into :attr:`segment_joints` of each segment (a unary chain or a single joint)."""
    segment_joints: wp.array[int]
    """Global joint indices, ordered from each segment's parent to its child."""
    child_offsets: wp.array[int]
    """Offsets into :attr:`child_segments` of each segment."""
    child_segments: wp.array[int]
    """Child segments in descending joint order, the summation order of the serial backward pass."""


@dataclass(frozen=True)
class _FeatherPGSTreePlan:
    """Schedule the independent branches of each articulation tree in parallel.

    Unary chains are compressed into single-lane segments when that does not lengthen the
    schedule. Only topology is stored; frames, velocities and inertias stay live inputs.
    """

    groups: tuple[_FeatherPGSTreeGroup, ...]
    """Groups with independent traversal widths and level tables."""

    @classmethod
    def build(cls, model: Model, articulation_joint_end: np.ndarray) -> "_FeatherPGSTreePlan | None":
        """Return a plan, or ``None`` when no articulation branches or the topology is unsupported."""
        if not model.articulation_count or not model.joint_count:
            return None
        starts = model.articulation_start.numpy()
        ends = np.asarray(articulation_joint_end)
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        if not np.array_equal(ends, starts[1:]):
            return None
        if np.any(children < 0):
            return None
        owners = np.bincount(children, minlength=model.body_count)
        if np.any(owners > 1):
            return None
        body_owner = np.full(model.body_count, -1, dtype=np.int32)
        body_owner[children] = np.arange(model.joint_count, dtype=np.int32)
        joint_children = [[] for _ in range(model.joint_count)]
        joint_depth = np.zeros(model.joint_count, dtype=np.int32)
        for art in range(model.articulation_count):
            start, end = int(starts[art]), int(ends[art])
            for joint in range(start, end):
                parent = int(parents[joint])
                if parent >= 0:
                    parent_joint = int(body_owner[parent])
                    # Parents must precede their children within the articulation.
                    if not start <= parent_joint < joint:
                        return None
                    joint_children[parent_joint].append(joint)
                    joint_depth[joint] = joint_depth[parent_joint] + 1

        def schedule_cost(levels):
            width = max(map(len, levels), default=1)
            lanes = min(32, 1 << (width - 1).bit_length())
            rounds = sum(
                max(map(len, level[begin : begin + lanes])) for level in levels for begin in range(0, len(level), lanes)
            )
            return lanes, rounds

        segment_depth = np.zeros(model.joint_count, dtype=np.int32)
        grouped = {}
        has_branches = False
        for art in range(model.articulation_count):
            start, end = int(starts[art]), int(ends[art])
            levels = []
            joint_levels = []
            for joint in range(start, end):
                joint_level = int(joint_depth[joint])
                if joint_level == len(joint_levels):
                    joint_levels.append([])
                joint_levels[joint_level].append([joint])
                parent = int(parents[joint])
                level = 0
                if parent >= 0:
                    parent_joint = int(body_owner[parent])
                    if len(joint_children[parent_joint]) == 1:
                        continue
                    level = int(segment_depth[parent_joint]) + 1
                segment = [joint]
                while len(joint_children[segment[-1]]) == 1:
                    segment.append(joint_children[segment[-1]][0])
                segment_depth[segment] = level
                if level == len(levels):
                    levels.append([])
                levels[level].append(segment)
            lanes, rounds = schedule_cost(levels)
            joint_lanes, joint_rounds = schedule_cost(joint_levels)
            # A long chain can hold up the other branches of its level. Keep the schedule with
            # fewer sequential joint rounds; this is a topology proxy, not a runtime guarantee.
            if rounds > joint_rounds:
                levels, lanes = joint_levels, joint_lanes
            has_branches |= lanes > 1
            grouped.setdefault(lanes, []).append((art, levels))
        if not has_branches:
            return None
        groups = []
        for lanes, trees in sorted(grouped.items()):
            max_levels = max(len(levels) for _, levels in trees)
            offsets = np.empty((len(trees), max_levels + 1), dtype=np.int32)
            segments = []
            for row, (_, levels) in enumerate(trees):
                offsets[row, 0] = len(segments)
                for level in range(max_levels):
                    if level < len(levels):
                        segments.extend(levels[level])
                    offsets[row, level + 1] = len(segments)
            segment_ids = {segment[0]: index for index, segment in enumerate(segments)}
            segment_offsets, joints = [0], []
            child_offsets, child_segments = [0], []
            for segment in segments:
                joints.extend(segment)
                segment_offsets.append(len(joints))
                child_segments.extend(segment_ids[child] for child in reversed(joint_children[segment[-1]]))
                child_offsets.append(len(child_segments))
            groups.append(
                _FeatherPGSTreeGroup(
                    lanes=lanes,
                    articulation_count=len(trees),
                    max_levels=max_levels,
                    articulations=wp.array([art for art, _ in trees], dtype=wp.int32, device=model.device),
                    level_offsets=wp.array(offsets, dtype=wp.int32, device=model.device),
                    segment_offsets=wp.array(segment_offsets, dtype=wp.int32, device=model.device),
                    segment_joints=wp.array(joints, dtype=wp.int32, device=model.device),
                    child_offsets=wp.array(child_offsets, dtype=wp.int32, device=model.device),
                    child_segments=wp.array(child_segments, dtype=wp.int32, device=model.device),
                )
            )
        return cls(tuple(groups))


_DENSE_META_ROW_TYPE_BITS = 3
_DENSE_META_ROW_TYPE_MASK = (1 << _DENSE_META_ROW_TYPE_BITS) - 1
_DENSE_META_MAX_PARENT = ((2**31 - 1) >> _DENSE_META_ROW_TYPE_BITS) - 1


def _use_resident_mfgs_metadata(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    max_shared_memory: int,
    *,
    has_drive_rows: bool = False,
    fuse_vel_limits: bool = False,
) -> bool:
    """Select resident dense-row metadata for compact solver shapes.

    Resident row metadata removes repeated global loads across PGS iterations,
    but larger shapes lose more throughput from the resulting occupancy drop.
    Keep at most one 4-KiB metadata working set resident and stream larger sets.
    Drive rows add four per-row arrays and the fused velocity-limit clamp one more.
    """
    metadata_arrays = 4 + 4 * int(has_drive_rows) + int(fuse_vel_limits)
    metadata_bytes = 4 * max_constraints * metadata_arrays
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


def _estimate_hinv_jt_shared_memory(
    n_dofs: int, constraint_count: int, *, tile_threads: int, fused: bool = False
) -> int:
    """Estimate the complete tiled H-inverse shared-memory footprint [B].

    ``fused`` adds the ``constraint_count x constraint_count`` Delassus tile of the fused
    split-mode kernel.
    """
    footprint = _align_shared_memory(4 * n_dofs * n_dofs)
    footprint += 3 * _align_shared_memory(4 * n_dofs * constraint_count)
    if fused:
        footprint += _align_shared_memory(4 * constraint_count * constraint_count)
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
    hinv_jt_fused_sizes: frozenset[int]
    diagonal_mass_sizes: frozenset[int] = frozenset()

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
        diagonal_mass_sizes: frozenset[int] = frozenset(),
    ) -> "_FeatherPGSExecutionPlan":
        """Resolve the factorization and H^-1 J^T implementations from solver shape and device limits.

        Groups in ``diagonal_mass_sizes`` have a structurally diagonal mass matrix and use
        the per-DOF diagonal kernels instead of a factorization, solve or response kernel.
        """
        cholesky_tiled_sizes: set[int] = set()
        tiled_sizes: set[int] = set()
        chunk_sizes: list[tuple[int, int]] = []
        fused_sizes: set[int] = set()
        diagonal_sizes = frozenset(size for size in size_groups if size in diagonal_mass_sizes)
        for size in size_groups:
            if size in diagonal_sizes:
                continue
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
            # The fused split-mode kernel holds the whole row set and its Delassus tile.
            if (
                _estimate_hinv_jt_shared_memory(size, max_constraints, tile_threads=tile_threads, fused=True)
                <= max_shared_memory
            ):
                fused_sizes.add(size)

        return cls(
            frozenset(cholesky_tiled_sizes),
            frozenset(tiled_sizes),
            tuple(chunk_sizes),
            frozenset(fused_sizes),
            diagonal_sizes,
        )

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

    def use_fused_hinv_jt(self, size: int) -> bool:
        """Return whether a response group may fuse H-inverse application and Delassus assembly."""
        return size in self.hinv_jt_fused_sizes

    def use_diagonal_mass(self, size: int) -> bool:
        """Return whether an articulation group uses the diagonal mass-matrix kernels."""
        return size in self.diagonal_mass_sizes


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
    projected Gauss-Seidel (PGS) in impulse space. Rows of articulated bodies use the
    response ``Y = H^-1 J^T``; contacts between free rigid bodies (a single body with a
    world-rooted free joint) and the ground or other free bodies use per-body inverse
    inertia. Positions are integrated from the solved velocities with symplectic Euler.

    ``pgs_mode`` selects the solve:

    - ``"matrix_free"`` (default, CUDA only): for each row the solver keeps ``Y`` and the
      diagonal ``J Y`` and recomputes ``J v`` every iteration instead of assembling a
      Delassus matrix. Every row applies its impulse to one world-local velocity vector
      immediately, and one fused kernel sweeps all rows of a world. Each feature below is
      available in it; some combinations are rejected, as the options document.
    - ``"split"`` (CPU and CUDA; required on CPU): the rows of articulated bodies of each
      world are assembled into a dense Delassus matrix ``C = J H^-1 J^T``
      (``dense_max_constraints`` squared per world) and solved in impulse space; the
      free-body rows are then solved against the resulting velocity. Worlds with both kinds
      of rows alternate one sweep of each per iteration. It implements the base feature set
      and raises :class:`NotImplementedError` when any of these is active: joint velocity
      limits, PGS drive rows, mimic and loop-closing joints, friction patches, contact
      regularization, restitution, warm start, velocity-only iterations, contact torsion,
      contact compliance, sleeping and the propagation contact responses. Contact torsion is CUDA-only, so on a CPU device a
      positive ``contact_torsion_radius`` raises :class:`ValueError` first.

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
      :attr:`~newton.Model.joint_target_kd` on PRISMATIC, REVOLUTE and D6 DOFs. By
      default (``drive_mode="augmented"``) they are integrated implicitly by folding
      ``dt * kd + dt^2 * ke`` into the mass matrix; only the explicit drive force is
      clamped to :attr:`~newton.Model.joint_effort_limit`, and the implicit stiffness and
      damping response is unbounded, so under a large external load the drive reaction
      can exceed the limit. ``drive_mode="physx_pgs"`` (``pgs_mode="matrix_free"`` only)
      instead solves one PGS row per driven DOF with the PhysX articulation force-drive
      update and clamps its impulse, which bounds the complete reaction.
      :attr:`~newton.Model.joint_armature` and :attr:`~newton.Model.joint_damping` are
      applied.
    - Joint limits: with ``enable_joint_limits=True``, every finite
      :attr:`~newton.Model.joint_limit_lower` / :attr:`~newton.Model.joint_limit_upper` of a
      PRISMATIC, REVOLUTE or D6 DOF is a unilateral row. Joint limits are not enforced by
      default. Joint velocity limits (:attr:`~newton.Model.joint_velocity_limit`) are
      enforced as rows when ``enable_joint_velocity_limits`` is set.
    - Mimic joints (:meth:`~newton.ModelBuilder.set_joint_mimic`, and the deprecated
      mimic constraints) within one articulation: one bilateral row per follower
      coordinate enforces ``q_follower = coeffs[0] + coeffs[1] * q_leader``. The
      coefficients are read every step.
    - Loop-closing BALL joints (joints outside the articulation tree whose child belongs
      to an articulation): three bilateral rows pin the joint's parent and child anchors
      together. The parent is in the child's articulation, a kinematic body, or the
      world. Closures can be released and re-anchored at runtime, see
      :meth:`set_loop_joint_enabled` and :meth:`set_loop_joint_anchors`. With
      ``enable_bilateral_preelimination`` the mimic and connect rows are eliminated
      before the sweep through a regularized Schur complement; this is not exact
      elimination, see the option.
    - Contacts: rigid contacts from :class:`~newton.CollisionPipeline`, one normal row
      per contact, with Coulomb friction from the mean of the two shapes' ``mu``. By
      default friction acts through persistent friction patches: compatible contacts of
      a body pair form a region whose friction is applied at up to two anchors that
      share the region's normal load and carry their tangential displacement across
      steps, so static friction holds without creep (``friction_anchor_beta``). Setting
      ``friction_anchor_beta=0`` selects point friction, two coupled tangent rows per
      contact. Restitution follows the shapes'
      :attr:`~newton.ModelBuilder.ShapeConfig.restitution`. Optional contact
      regularization, warm start from the previous step's impulses, velocity-only
      iterations and gap gates for speculative contacts are configured on the
      constructor. Experimental, default-off contact torsion (``contact_torsion_radius``)
      adds a load-bounded spin-friction row per contact group, and experimental implicit
      compliance of hydroelastic contacts is opt-in (``contact_compliance``). Friction
      patches, restitution, regularization, warm start, velocity-only iterations, contact
      torsion and contact compliance need ``pgs_mode="matrix_free"``; the split solve uses
      point friction.
    - Kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC`) and heterogeneous worlds.
    - CUDA graph capture of :meth:`step` and :meth:`reset`.
    - Experimental passive-island sleeping (``enable_sleeping``, ``pgs_mode="matrix_free"``
      only): settled, supported islands of articulations freeze their published state and
      skip their rows and dynamics until a wake event.

    Limitations:

    - ``pgs_mode="matrix_free"``, the default, requires a CUDA device; constructing it on a
      CPU device raises :class:`NotImplementedError` that names ``pgs_mode="split"``.
    - Mimic relationships across articulations or on joints whose position and velocity
      coordinates differ (BALL, FREE, DISTANCE), loop-closing joints of other types, and
      loop closures between two dynamic articulations raise
      :class:`NotImplementedError`. With ``pgs_mode="split"``, every mimic relationship and
      loop-closing joint raises :class:`NotImplementedError`; they need
      ``pgs_mode="matrix_free"``. :attr:`~newton.Model.joint_enabled` is supported only
      for loop-closing joints. Enabled MuJoCo equality constraints
      (``model.mujoco.equality_constraint_*``) are enforced through the loop joint or mimic
      constraint the importer converted them to; an enabled row without such a conversion
      raises :class:`NotImplementedError`, also when it is enabled later and
      :meth:`notify_model_changed` is called with
      :attr:`~newton.ModelFlags.CONSTRAINT_PROPERTIES`. A row counts as converted only
      when its ``target_kind`` / ``target`` link names a loop joint or mimic constraint
      between the row's own bodies or joints. Particles are not simulated.
    - Gradients are not supported.

    With ``pgs_mode="matrix_free"``, branched articulations use sparse mass factors when
    every articulated (non-free-body) response group shares one joint topology with at most
    64 DOFs, joint velocity-limit rows are disabled, the model has no mimic or loop-closing
    joints, drives are augmented and contacts use hard point friction
    (``friction_anchor_beta=0``, no warm start, regularization, velocity-only iterations,
    friction gap threshold, restitution, contact torsion or compliance): the mass matrix is assembled and factored in
    the fill-free pattern of the kinematic tree, and constraint rows keep only the DOFs
    that support them.
    Otherwise, and for free bodies, dense factors are used. Both give the same dynamics up
    to floating-point rounding. The selection follows the model's structure only; it is not
    a performance prediction. Sparse factors take longer to set up and, depending on the
    articulation and the number of worlds, can run faster or markedly slower than dense
    factors. Articulations whose DOFs are not coupled at all, for example independent
    single-DOF branches of a fixed base, have a diagonal mass matrix; they skip the
    factorization and solve each DOF by division, bitwise identical to the dense ``loop``
    kernels and equal to the ``tiled`` kernels up to float32 rounding, and take precedence
    over sparse factors. ``parallel_tree=True`` additionally traverses the independent
    branches of each tree in parallel.

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
    # (keys: cholesky_kernel, trisolve_kernel, hinv_jt_kernel; split mode also
    # delassus_kernel and pgs_kernel, which selects the native or scalar Gauss-Seidel
    # kernels of both row families; sparse_mass_matrix=False keeps dense mass factors;
    # diagonal_mass=False keeps diagonal mass matrices on the factor paths;
    # propagation_tree_kernel="generic" replaces the one-warp propagation tree kernels
    # by the per-articulation ones).
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
        pgs_mode: Literal["matrix_free", "split"] = "matrix_free",
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
        row_watermark: bool = False,
        drive_mode: Literal["augmented", "physx_pgs"] = "augmented",
        fuse_joint_velocity_limits: bool = True,
        enable_bilateral_preelimination: bool = False,
        bilateral_preelimination_include_mimics: bool = True,
        parallel_tree: bool = False,
        use_parallel_streams: bool = False,
        double_buffer: bool = False,
        friction_anchor_beta: float | None = None,
        pgs_contact_regularization: float = 0.0,
        pgs_velocity_iterations: int = 0,
        pgs_velocity_drive_mode: Literal["freeze", "active"] = "freeze",
        pgs_schedule: Literal["interleaved", "contact_then_internal", "physx_grasp"] = "interleaved",
        friction_mode: Literal["current", "bisection", "bisection_desaxce", "coulomb_newton"] = "current",
        pgs_warmstart: bool = False,
        pgs_warmstart_decay: float = 1.0,
        restitution_velocity_threshold: float = 0.5,
        contact_speculative_scale: float = 1.0,
        contact_gap_gate: float = 0.0,
        same_articulation_contact_gap_gate: float = 0.0,
        articulation_pair_contact_gap_gate: float = 0.0,
        contact_friction_gap_threshold: float = float("inf"),
        contact_friction_articulation_pairs_only: bool = False,
        contact_shared_anchor: bool = False,
        contact_friction_shared_anchor: bool = False,
        contact_torsion_radius: float = 0.0,
        contact_torsion_shape_indices: tuple[int, ...] | None = None,
        contact_torsion_shape_patterns: tuple[str, ...] | None = None,
        contact_torsion_device: bool = False,
        contact_compliance: bool = False,
        enable_sleeping: bool = False,
        sleep_linear_threshold: float = 0.05,
        sleep_angular_threshold: float = 0.15,
        sleep_quiet_time: float = 0.5,
        sleep_skip_constraints: bool = True,
        articulated_contact_response: Literal["immediate", "propagation", "propagation-fused"] = "immediate",
        propagation_same_articulation_rows: bool = False,
    ):
        """Create a FeatherPGS solver for a finalized model.

        The options below configure this solver instance at runtime; they are not model or USD
        attributes. Per-body physical parameters are model attributes, see
        :meth:`register_custom_attributes`.

        Args:
            model: Model to simulate.
            pgs_mode: Constraint solve. ``"matrix_free"`` (default, CUDA only) sweeps every
                row in one fused kernel and recomputes ``J v`` from the current velocity.
                ``"split"`` (CPU and CUDA; pass it explicitly on CPU) assembles the rows of
                articulated bodies into a dense Delassus matrix ``J H^-1 J^T`` per world,
                solves it in impulse space and then solves the free-body rows against the
                resulting velocity; worlds that contain both kinds of rows alternate one sweep
                of each per iteration. The split solve raises for the matrix-free-only
                options listed in the class description.
            pgs_iterations: Number of projected Gauss-Seidel iterations per step.
            pgs_beta: Baumgarte position-correction factor of contact and joint-limit rows,
                as a fraction of the position error removed per step.
            pgs_cfm: Constraint force mixing added to every row's effective-mass diagonal,
                which regularizes redundant rows. Drive rows of ``drive_mode="physx_pgs"`` use
                their exact unit response instead.
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
                scaling articulation velocities that already exceed a limit. Requires
                ``pgs_mode="matrix_free"``. The free-body limits of
                :meth:`register_custom_attributes` are enforced whenever those attributes are
                registered, independently of this option and of ``pgs_mode``.
            velocity_limit_activation_fraction: Create the velocity-limit rows of a DOF only
                when ``|qd| >= velocity_limit_activation_fraction * limit``. ``0`` creates
                them for every limited DOF each step; ``inf`` never creates them. The gate
                samples the velocity before the solve, so a DOF that crosses the threshold
                during a step is clamped one step later. Must be in ``[0, 1]`` or ``inf``.
            dense_max_constraints: Capacity of rows involving articulated bodies (mimic and
                connect rows, contacts, enabled joint limits, joint velocity limits and, with
                ``drive_mode="physx_pgs"``, joint drives) per world. Rows beyond it are
                dropped and reported, see :attr:`constraint_overflow`. The split solve also
                stores a ``dense_max_constraints x dense_max_constraints`` Delassus matrix per
                world.
            mf_max_constraints: Capacity of free-body contact rows per world. Rows beyond it
                are dropped and reported, see :attr:`constraint_overflow`.
            warn_constraint_overflow: Print a device-side warning the first time a world
                exceeds a row capacity. The warning does not synchronize the host and is
                compatible with CUDA graph capture.
            row_watermark: Accumulate the high-water marks of the per-world row counts of
                every row family and of the contact count on the device in every step,
                including captured ones; read them with :meth:`constraint_row_watermarks`.
            drive_mode: Joint drive formulation. ``"augmented"`` integrates drives
                implicitly in the mass matrix. ``"physx_pgs"`` (``pgs_mode="matrix_free"``
                only) adds one dense PGS row per driven DOF (positive ``joint_target_ke`` or
                ``joint_target_kd``) whose impulse
                follows the PhysX articulation force-drive update in every iteration, so
                drives are solved together with contacts and limits instead of before them.
                Without contacts or limits and below the effort limit, the converged row
                reproduces the implicit drive of ``"augmented"``. Under effort saturation
                the two differ: ``"augmented"`` clamps the explicit drive force before the
                implicit solve, while the row clamps its accumulated impulse at
                ``joint_effort_limit * dt``. Drive rows are allocated first in each world, count against
                ``dense_max_constraints``, and a drive whose row does not fit is not applied
                (the world is flagged in :attr:`constraint_overflow`).
            fuse_joint_velocity_limits: With ``drive_mode="physx_pgs"`` and
                ``enable_joint_velocity_limits``, enforce the velocity limit of each driven DOF
                with a clamp of its velocity at the end of every iteration (after the
                contact rows, like the velocity-limit rows it replaces) instead of two
                velocity-limit rows, and exclude it from the pre-solve velocity scaling.
                Limited DOFs without a drive row keep their rows. In every other
                configuration, including ``velocity_limit_activation_fraction=inf``, the
                option has no effect; :attr:`fuse_joint_velocity_limits` reports whether
                the clamp is active.
            enable_bilateral_preelimination: Eliminate the mimic and connect rows of each
                articulation before the iterative sweep with a Schur complement of the
                bilateral block ``S = J_B H^-1 J_B^T``: the predicted velocity is projected
                once and the responses of the other rows are corrected, so closures and mimic
                couplings depend much less on ``pgs_iterations``. The block is factored with a
                diagonal regularization ``R`` of ``1e-3`` times each row's diagonal plus a
                floor of ``max(pgs_cfm, 1e-7)``, which keeps nearly dependent closure axes
                positive definite. The elimination is therefore not exact: a bilateral
                velocity residual of ``R (S + R)^-1 (J_B v + b_B)`` remains after the
                projection, and the corrected responses of other rows still couple into the
                bilateral rows by the same factor. The rows stay allocated and in the sweep,
                which reduces this residual further. If any articulation owns more than
                eight bilateral rows, any loop closure has a kinematic or world parent, or
                ``pgs_velocity_iterations`` is positive, elimination is disabled for the
                whole solver (every articulation keeps iterative rows) with a warning at
                construction.
            bilateral_preelimination_include_mimics: With pre-elimination enabled, also
                eliminate mimic rows. ``False`` eliminates only the connect rows and keeps
                mimic rows iterative, which avoids a singular block when a mimic row is
                nearly dependent on a closure.
            parallel_tree: Traverse the independent branches of each articulation tree in
                parallel (forward kinematics and dynamics, the inverse-dynamics backward pass
                and state publication), with up to 32 threads per articulation instead of
                one. Results match the serial traversal up to floating-point summation order.
                Unbranched articulations, models whose joints are not ordered parent before
                child, and CPU devices keep the serial traversal. Broad trees can benefit; narrow trees can be
                slower, so measure the full step before enabling it.
            use_parallel_streams: Run the factorization, unconstrained solve and response
                kernels of each articulation-size group on a CUDA stream of its own, so groups
                of different sizes overlap. Results are unchanged; CPU devices and models with
                one size group ignore it.
            double_buffer: Keep two sets of the grouped mass-matrix and Jacobian buffers and
                clear the set a step used on a separate CUDA stream while the next step runs
                on the other set. Results are unchanged; CPU devices and the sparse mass
                factors ignore it. To capture :meth:`step` in a CUDA graph, call
                :meth:`seed_double_buffer_events` inside the capture before the first step.
            friction_anchor_beta: Position-correction gain of persistent friction patches;
                ``0`` selects point friction. ``None`` selects ``0.2`` with
                ``pgs_mode="matrix_free"``, and point friction with ``pgs_mode="split"``,
                which has no friction patches, or when ``contact_compliance`` is enabled.
                With a positive gain, the contacts of a body
                pair whose normals, contact planes and friction coefficients agree and whose
                shapes touch form a region. Each region's friction acts at up to two anchors
                along its principal extent, which share the region's total normal impulse;
                every contact keeps its own normal row. An anchor carries its tangential
                displacement across steps, following the pose increments of both bodies,
                and the tangent rows add ``friction_anchor_beta * displacement / dt``, so a
                held object does not creep under a static load. Anchors are released when
                their region is unloaded or slides at the Coulomb limit; history is matched
                by body pair and geometry, independently of collision contact matching. A
                one-anchor region has no torsional friction of its own. Call :meth:`reset`
                after teleporting a body to discard its history.
            pgs_contact_regularization: Dimensionless regularization ``g`` of penetrating
                contact rows (``pgs_mode="matrix_free"`` only). Each iteration moves a row's impulse toward the rigid solution
                with weight ``1 / (1 + g)`` and toward zero with weight ``g / (1 + g)``, the
                update of an implicitly integrated contact spring. It makes statically
                indeterminate normal-force splits unique and damps the sweep, at the cost of
                a resting penetration: a row at rest settles at ``pgs_beta * phi / dt = -g * d * lambda``
                (``d`` its inverse effective mass, ``lambda`` its impulse), about
                ``g * a * dt^2 / pgs_beta`` for a body under acceleration ``a`` resting on one row.
                Speculative (positive-gap) rows, rows whose rebound fires and the
                velocity-only iterations stay rigid. ``0`` is the rigid law; at most ``1e6``.
            pgs_velocity_iterations: Number of velocity-only iterations after the position
                solve (``pgs_mode="matrix_free"`` only). Positions are integrated from the position solve; these iterations
                then refine the velocity without the position bias, so stabilization does
                not add velocity to the bodies. A positive-gap contact whose linearized gap
                at the end of the step stays open keeps its speculative allowance, and an
                impacting contact keeps its rebound target.
            pgs_velocity_drive_mode: Treatment of PGS joint-drive rows during the
                velocity-only iterations: ``"freeze"`` keeps the drive impulses of the
                position solve, ``"active"`` keeps solving the drive rows. Implicit
                (mass-matrix) drives have no rows, so the option has no effect for them.
            friction_mode: Experimental Coulomb friction update of the matrix-free solve
                (``pgs_mode="matrix_free"`` only). ``"current"`` updates both tangent impulses
                together on the friction disk of the current normal impulse: sticking uses the
                inverse of their 2x2 block, sliding a bounded scalar solve that keeps isotropic
                maximum dissipation. ``"bisection"`` bisects the normal impulse of each contact
                and re-solves the 2x2 tangent problem at every probe, so normal and friction
                impulses are updated together. ``"bisection_desaxce"`` adds the de Saxce
                maximum-dissipation bias ``mu * |c_T|`` to the normal target velocity of
                free-body contacts (Le Lidec and Carpentier, 2024). ``"coulomb_newton"`` solves
                each free-body contact's Coulomb cone exactly with a bracketed scalar Newton
                iteration on the tangential-force ratio; articulated contacts keep the
                ``"current"`` update. The other modes use point friction: an omitted
                ``friction_anchor_beta`` selects it with a warning, and friction patches,
                contact torsion, contact compliance and the propagation responses require
                ``"current"``. This option may change without the normal deprecation period.
            pgs_schedule: Row order of the matrix-free sweeps (``pgs_mode="matrix_free"`` only).
                ``"interleaved"`` sweeps every row family in each iteration: the dense rows,
                the free-body contact rows, then the velocity limits. ``"contact_then_internal"``
                runs every iteration of the contact rows first, then every iteration of the
                internal rows (drive, mimic, connect and joint-limit rows) and the velocity
                limits. ``"physx_grasp"`` approximates the order PhysX uses for grasping: each
                iteration solves the internal rows, then the contact rows, then the velocity
                limits, each in a launch of its own; the propagation responses use this order
                with ``"interleaved"`` as well. Contact torsion and contact compliance require
                ``"interleaved"``, ``"propagation-fused"`` rejects ``"contact_then_internal"``,
                and the sparse mass factors are selected only with ``"interleaved"``.
            pgs_warmstart: Start each step (``pgs_mode="matrix_free"`` only) from the previous
                step's contact impulses, matched
                by contact identity through :attr:`~newton.Contacts.rigid_contact_match_index`,
                so the :class:`~newton.CollisionPipeline` must be created with contact matching
                enabled. Substeps that reuse a contact set without a collision pass seed each
                contact from its own previous substep. History carries across one collision
                pass into the same :class:`~newton.Contacts` buffer when that pass matched
                against the solved contact set
                (:attr:`~newton.Contacts.rigid_contact_match_generation`); a different
                buffer, skipped passes, or a pass after the pipeline wrote another buffer
                start cold. Carried impulses are scaled by the
                ratio of the step to the previous one, including under graph replay; tangent
                impulses are rotated into the current tangent frame and clamped to the
                current friction cone; other rows start cold.
            pgs_warmstart_decay: Non-negative scale applied to the carried impulses.
            restitution_velocity_threshold: Minimum incident normal speed [m/s] for a
                rebound. Contacts bounce with the arithmetic mean of the two shapes'
                restitution coefficients (:attr:`~newton.ModelBuilder.ShapeConfig.restitution`,
                clamped to ``[0, 1]``; zero gives no rebound): an impacting contact replaces
                its position bias by the rebound target ``-e * u``, where ``u`` is its
                incident normal velocity in the unconstrained prediction. The contact must
                also touch, or be predicted to reach the surface during the step. Slower
                contacts keep the ordinary contact law, so resting contacts do not bounce
                under small accelerations. The largest finite float32 value turns
                restitution off for every contact without editing materials. Restitution
                needs ``pgs_mode="matrix_free"``; the split solve rejects shapes with a
                positive restitution unless this value turns restitution off.
            contact_speculative_scale: Fraction of a positive contact gap that the contact
                may close during the step (``rhs = scale * phi / dt``). ``0`` removes the
                speculative allowance; penetration correction is unchanged.
            contact_gap_gate: When positive, contacts whose gap [m] exceeds this distance
                get no rows. ``0`` keeps every contact of the collision pipeline.
            same_articulation_contact_gap_gate: When positive, the gap gate [m] of contacts
                between two links of the same articulation (free bodies excluded).
            articulation_pair_contact_gap_gate: When positive, the gap gate [m] of contacts
                between two articulated (non-free) bodies.
            contact_friction_gap_threshold: Contacts with a gap [m] above this threshold get
                a normal row only. ``inf`` gives every contact friction.
            contact_friction_articulation_pairs_only: Apply
                ``contact_friction_gap_threshold`` only to contacts between two articulated
                (non-free) bodies.
            contact_shared_anchor: Apply every contact row at the midpoint of the two witness
                points instead of at each body's own witness point, as PhysX applies a contact at
                one point. ``phi`` still comes from the witness points. Friction patch rows keep
                their patch anchors, so with patches this moves only the normal rows.
            contact_friction_shared_anchor: Apply the friction rows at the witness midpoint,
                which removes the tangential force couple of witness points separated along the
                normal. It has no effect on friction patch rows, which act at their anchors.
            contact_torsion_radius: Experimental effective spin radius [m] of contact
                torsion; ``0`` disables it. A positive radius adds one angular row per
                contact group of an articulated body, about the group's normal, bounded by
                ``radius * (mu * N - sliding)``: sliding and spin share one Coulomb budget,
                and the normal load ``N`` is the final solved load of the group's normal
                rows. With point friction a group is a coplanar cluster of touching contacts
                of one shape pair; with friction patches it is one patch region, budgeted by
                its pooled load and undivided friction coefficient. The radius is an explicit
                footprint assumption, not derived from the geometry: for uniform pressure on
                a disk of radius ``R`` it is ``2 * R / 3``. Rows are rebuilt every step and
                carry no spin history. Velocity-only iterations keep the position solve's
                spin impulse and retire the rows of groups that no longer touch. Free-body
                contacts get no torsion rows. Requires ``pgs_warmstart=False``; contacts
                with hydroelastic stiffness are rejected, and running out of
                ``dense_max_constraints`` raises instead of dropping rows. Configure at
                construction.
            contact_torsion_shape_indices: Global shape indices selecting the contacts that
                get torsion (contacts with at least one selected shape). ``None`` selects
                every shape; an empty tuple selects none. Supported geometry is sphere, box,
                capsule, cylinder, cone, ellipsoid, plane and convex mesh: selecting another
                type raises, and without a selection contacts involving other types are
                skipped. Mutually exclusive with ``contact_torsion_shape_patterns``.
            contact_torsion_shape_patterns: Regular expressions full-matched against
                :attr:`~newton.Model.shape_label` at construction, as an alternative to
                ``contact_torsion_shape_indices``. A pattern that matches no shape raises.
            contact_torsion_device: Prepare the torsion rows on the device instead of on
                the host. The host preparation is the reference implementation; it reads
                the contacts back every step and cannot be captured in a CUDA graph. The
                device preparation builds the same groups and rows with preallocated
                buffers. Eager steps check its errors synchronously. To capture
                :meth:`step` in a CUDA graph, call :meth:`prepare_contact_torsion_capture`
                outside capture, then :meth:`validate_contact_torsion` after every replay
                batch before using the results. Grouping runs serially per world with
                worst-case quadratic cost in the contacts of a world. No effect without a
                positive ``contact_torsion_radius``.
            contact_compliance: Experimental: solve contacts that carry a positive
                :attr:`~newton.Contacts.rigid_contact_stiffness` [N/m] with an implicit
                unilateral spring-damper law instead of the rigid normal law. The material
                comes from the contacts (per-shape hydroelastic stiffness, the contact damping
                [N s/m], zero staying zero, and the friction weight, applied once to the pair
                friction); contacts with zero stiffness stay rigid. A compliant row's force is
                ``max(0, -k phi_next - c u_next)``, solved by updating its residual
                ``u + k phi / (dt k + c) + lambda / (dt (dt k + c))`` after every sweep; the
                damping term is dropped while the gap is open. Requires point friction
                (``None`` or ``0`` for ``friction_anchor_beta``; ``None`` warns), positive
                ``pgs_iterations``, no warm start, no velocity-only iterations, no contact
                regularization, no shared anchors and zero shape restitution. The contacts must carry the
                hydroelastic material arrays. Each step synchronizes the row metadata with the
                host, so the option rejects CUDA graph capture and is not a performance path.
                Contacts the allocator excludes on purpose are counted in
                ``compliance_skipped_contact_count`` and compliant contacts in
                ``compliance_contact_count``; a compliant contact lost to a row capacity
                raises, even with ``warn_constraint_overflow`` off. This option may change
                without the normal deprecation period.
            enable_sleeping: Experimental passive-island sleeping. Articulations in contact
                are joined into islands every step. Static geometry and articulations with no
                responding degrees of freedom (such as a body welded to the world) support an
                island without joining it. A fixed-base articulation with responding joints,
                such as a robot arm, joins the island of everything it touches, so it can
                connect them into one island; its fixed root supports that island. A
                supported island whose bodies stay below ``sleep_linear_threshold`` and
                ``sleep_angular_threshold`` for ``sleep_quiet_time`` falls asleep, and its
                published state (``joint_q``, ``joint_qd``, ``body_q``, ``body_qd`` and, when
                requested, ``body_parent_f``) is frozen until a wake event: a nonzero or nonfinite
                external force (:attr:`~newton.State.body_f`, :attr:`~newton.Control.joint_f`),
                a changed state or gravity, a contact with an awake or moving body, lost
                support or row capacity, a :meth:`reset` of its world, or
                :meth:`notify_model_changed` for properties of its joints, DOFs, bodies or
                shapes (other notifications and changes to static or kinematic owners wake
                every island). Driven articulations (nonzero drive gains)
                stay awake. Sleeping is configured at construction; rebuild captured graphs
                to change it. Construction raises :class:`ValueError` with ``pgs_warmstart``
                or ``pgs_velocity_iterations``, or for a model without articulations;
                :meth:`step` raises :class:`ValueError` for a nonpositive or nonfinite
                ``dt`` and when ``state_in`` and ``state_out`` share arrays. The sleeping
                options are runtime solver settings with no USD schema.
            sleep_linear_threshold: Body center-of-mass speed below which a body is quiet
                [m/s], finite and positive.
            sleep_angular_threshold: Body angular speed below which a body is quiet [rad/s],
                finite and positive.
            sleep_quiet_time: Quiet, supported interval after which an island sleeps [s],
                finite and positive.
            sleep_skip_constraints: Skip the work of sleeping islands: their contact and
                joint-limit rows (collision still runs, so wake events are complete), the
                rebuild of their friction-patch history (carried instead), and their
                articulated dynamics (kinematics, inverse dynamics, drives, mass matrix
                and factorization, velocity prediction and integration). ``False`` keeps
                the rows and dynamics of sleeping islands and only freezes their published
                state.
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
        if contact_compliance:
            # Reject unsupported combinations before any allocation.
            _contact_compliance.validate_configuration(
                {
                    "pgs_iterations": int(pgs_iterations),
                    "pgs_velocity_iterations": int(pgs_velocity_iterations),
                    "pgs_warmstart": bool(pgs_warmstart),
                    "pgs_contact_regularization": float(pgs_contact_regularization),
                    "contact_shared_anchor": bool(contact_shared_anchor),
                    "contact_friction_shared_anchor": bool(contact_friction_shared_anchor),
                    "friction_mode": friction_mode,
                }
            )
        if friction_mode not in ("current", "bisection", "bisection_desaxce", "coulomb_newton"):
            raise ValueError(
                "friction_mode must be 'current', 'bisection', 'bisection_desaxce' or 'coulomb_newton', "
                f"got {friction_mode!r}"
            )
        self.friction_mode = friction_mode
        if friction_anchor_beta is None:
            if contact_compliance:
                friction_anchor_beta = 0.0
                warnings.warn(
                    "The contact_compliance material law uses point friction; it does not support "
                    "persistent friction patches. Set friction_anchor_beta=0 to select point friction "
                    "without this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            elif (
                friction_mode != "current" and pgs_mode == "matrix_free" and articulated_contact_response == "immediate"
            ):
                friction_anchor_beta = 0.0
                warnings.warn(
                    f"friction_mode={friction_mode!r} uses point friction; it does not support persistent "
                    "friction patches. Set friction_anchor_beta=0 to select point friction without this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # The split solve and the propagation responses have no friction patches.
                friction_anchor_beta = (
                    0.2 if pgs_mode == "matrix_free" and articulated_contact_response == "immediate" else 0.0
                )
        elif contact_compliance and float(friction_anchor_beta) > 0.0:
            raise ValueError("contact_compliance is not validated with friction_anchor_beta > 0")
        elif friction_mode != "current" and float(friction_anchor_beta) > 0.0:
            raise ValueError(
                f"Friction patches require friction_mode='current', got {friction_mode!r}; "
                "set friction_anchor_beta=0 for point friction"
            )
        self.contact_compliance = bool(contact_compliance)
        self.compliance_contact_count = 0
        """Compliant contacts consumed by the last step (``contact_compliance`` only)."""
        self.compliance_skipped_contact_count = 0
        """Positive-stiffness contacts the row allocator excluded in the last step (``contact_compliance`` only)."""
        self._compliant_contacts = None
        self._compliant_prepared = False
        super().__init__(model)
        if pgs_mode not in ("matrix_free", "split"):
            raise ValueError(f"pgs_mode must be 'matrix_free' or 'split', got {pgs_mode!r}")
        self.pgs_mode = pgs_mode
        if pgs_mode == "matrix_free" and not model.device.is_cuda:
            raise NotImplementedError(
                "SolverFeatherPGS pgs_mode='matrix_free' requires a CUDA device; use pgs_mode='split' on CPU."
            )
        if pgs_mode == "split" and enable_joint_velocity_limits:
            raise NotImplementedError("enable_joint_velocity_limits=True requires pgs_mode='matrix_free'")
        if pgs_mode == "split" and drive_mode == "physx_pgs":
            raise NotImplementedError("drive_mode='physx_pgs' requires pgs_mode='matrix_free'")
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
        self.row_watermark = bool(row_watermark)
        self._row_watermarks = (
            wp.zeros(ROW_WATERMARK_CONTACT_SLOT + 1, dtype=wp.int32, device=model.device)
            if self.row_watermark
            else None
        )
        if drive_mode not in ("augmented", "physx_pgs"):
            raise ValueError(f"drive_mode must be 'augmented' or 'physx_pgs', got {drive_mode!r}")
        self.drive_mode = drive_mode
        self._has_drive_rows = drive_mode == "physx_pgs" and model.joint_dof_count > 0
        # The fused clamp replaces the velocity-limit rows of driven DOFs, so it only
        # engages where both exist; an inf activation fraction disables velocity limits.
        self.fuse_joint_velocity_limits = (
            bool(fuse_joint_velocity_limits)
            and self._has_drive_rows
            and self.enable_joint_velocity_limits
            and not np.isinf(self.velocity_limit_activation_fraction)
        )
        self.enable_bilateral_preelimination = bool(enable_bilateral_preelimination)
        self.bilateral_preelimination_include_mimics = bool(bilateral_preelimination_include_mimics)
        self.friction_anchor_beta = _finite_non_negative("friction_anchor_beta", friction_anchor_beta)
        self._friction_anchors_enabled = self.friction_anchor_beta > 0.0
        self.pgs_contact_regularization = _finite_non_negative("pgs_contact_regularization", pgs_contact_regularization)
        if self.pgs_contact_regularization > _MAX_CONTACT_REGULARIZATION:
            raise ValueError(
                f"pgs_contact_regularization must be at most {_MAX_CONTACT_REGULARIZATION:g}; "
                "larger values are not numerically useful in the float32 solve"
            )
        # Kernels use the float32 weight w = 1 / (1 + g). Values too small to change w
        # are the exact rigid update.
        self._contact_w = float(np.float32(1.0 / (1.0 + self.pgs_contact_regularization)))
        self._regularization_enabled = self._contact_w < 1.0
        self.pgs_velocity_iterations = int(pgs_velocity_iterations)
        if self.pgs_velocity_iterations < 0:
            raise ValueError("pgs_velocity_iterations must be non-negative")
        if pgs_velocity_drive_mode not in ("freeze", "active"):
            raise ValueError(f"pgs_velocity_drive_mode must be 'freeze' or 'active', got {pgs_velocity_drive_mode!r}")
        self.pgs_velocity_drive_mode = pgs_velocity_drive_mode
        if pgs_schedule not in ("interleaved", "contact_then_internal", "physx_grasp"):
            raise ValueError(
                f"pgs_schedule must be 'interleaved', 'contact_then_internal' or 'physx_grasp', got {pgs_schedule!r}"
            )
        self.pgs_schedule = pgs_schedule
        self.pgs_warmstart = bool(pgs_warmstart)
        self.pgs_warmstart_decay = _finite_non_negative("pgs_warmstart_decay", pgs_warmstart_decay)
        self.restitution_velocity_threshold = _finite_non_negative(
            "restitution_velocity_threshold", restitution_velocity_threshold
        )
        self.contact_speculative_scale = _finite_non_negative("contact_speculative_scale", contact_speculative_scale)
        self.contact_gap_gate = _finite_non_negative("contact_gap_gate", contact_gap_gate)
        self.same_articulation_contact_gap_gate = _finite_non_negative(
            "same_articulation_contact_gap_gate", same_articulation_contact_gap_gate
        )
        self.articulation_pair_contact_gap_gate = _finite_non_negative(
            "articulation_pair_contact_gap_gate", articulation_pair_contact_gap_gate
        )
        self.contact_friction_gap_threshold = float(contact_friction_gap_threshold)
        if np.isnan(self.contact_friction_gap_threshold):
            raise ValueError("contact_friction_gap_threshold must not be NaN")
        self.contact_friction_articulation_pairs_only = bool(contact_friction_articulation_pairs_only)
        self.contact_shared_anchor = bool(contact_shared_anchor)
        self.contact_friction_shared_anchor = bool(contact_friction_shared_anchor)
        self._contact_anchor_args = (int(self.contact_shared_anchor), int(self.contact_friction_shared_anchor))
        if self._friction_anchors_enabled and (self.contact_shared_anchor or self.contact_friction_shared_anchor):
            warnings.warn(
                "Patch friction selects its own friction locations and carries tangential displacement. "
                "contact_shared_anchor still applies to normal rows; set friction_anchor_beta=0 "
                "to apply shared-anchor flags to point friction rows as well.",
                UserWarning,
                stacklevel=2,
            )
        configure_contact_torsion(
            self, contact_torsion_radius, contact_torsion_shape_indices, contact_torsion_shape_patterns
        )
        if pgs_mode == "split":
            unsupported = [
                name
                for name, requested in (
                    ("friction_anchor_beta > 0", self._friction_anchors_enabled),
                    ("pgs_contact_regularization > 0", self._regularization_enabled),
                    ("pgs_velocity_iterations > 0", self.pgs_velocity_iterations > 0),
                    ("pgs_warmstart=True", self.pgs_warmstart),
                    ("contact_torsion_radius > 0", self._contact_torsion_enabled),
                    ("contact_compliance=True", self.contact_compliance),
                    ("enable_sleeping=True", bool(enable_sleeping)),
                    (f"pgs_schedule={self.pgs_schedule!r}", self.pgs_schedule != "interleaved"),
                    (f"friction_mode={self.friction_mode!r}", self.friction_mode != "current"),
                )
                if requested
            ]
            if unsupported:
                raise NotImplementedError(f"{', '.join(unsupported)} requires pgs_mode='matrix_free'")
        if self.pgs_schedule != "interleaved":
            incompatible = [
                name
                for name, requested in (
                    ("contact_torsion_radius > 0", self._contact_torsion_enabled),
                    ("contact_compliance=True", self.contact_compliance),
                )
                if requested
            ]
            if incompatible:
                raise NotImplementedError(
                    f"{', '.join(incompatible)} requires pgs_schedule='interleaved', got {self.pgs_schedule!r}"
                )
        if self.pgs_schedule == "contact_then_internal" and articulated_contact_response == "propagation-fused":
            # The fused kernel sweeps internal, contact, propagation and velocity-limit rows in every iteration.
            raise NotImplementedError(
                "articulated_contact_response='propagation-fused' does not support "
                "pgs_schedule='contact_then_internal'; use 'propagation' for that schedule"
            )
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
        if articulated_contact_response != "immediate":
            if pgs_mode == "split":
                raise NotImplementedError(
                    f"articulated_contact_response={articulated_contact_response!r} requires pgs_mode='matrix_free'"
                )
            unsupported = [
                name
                for name, requested in (
                    ("drive_mode='physx_pgs'", self._has_drive_rows),
                    ("friction_anchor_beta > 0", self._friction_anchors_enabled),
                    ("pgs_contact_regularization > 0", self._regularization_enabled),
                    ("pgs_velocity_iterations > 0", self.pgs_velocity_iterations > 0),
                    ("pgs_warmstart=True", self.pgs_warmstart),
                    ("contact_torsion_radius > 0", self._contact_torsion_enabled),
                    ("contact_compliance=True", self.contact_compliance),
                    ("enable_sleeping=True", bool(enable_sleeping)),
                    (f"friction_mode={self.friction_mode!r}", self.friction_mode != "current"),
                )
                if requested
            ]
            if unsupported:
                raise NotImplementedError(
                    f"{', '.join(unsupported)} is not supported with "
                    f"articulated_contact_response={articulated_contact_response!r}"
                )

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
        self.delassus_kernel = self._kernel_overrides.get("delassus_kernel", "auto")
        self.pgs_kernel = self._kernel_overrides.get("pgs_kernel", "auto")
        if self.cholesky_kernel not in ("tiled", "loop", "auto"):
            raise ValueError("cholesky_kernel must be one of ['auto', 'loop', 'tiled']")
        if self.trisolve_kernel not in ("tiled", "loop", "auto"):
            raise ValueError("trisolve_kernel must be one of ['auto', 'loop', 'tiled']")
        if self.hinv_jt_kernel not in ("tiled", "par_row", "auto"):
            raise ValueError("hinv_jt_kernel must be one of ['auto', 'par_row', 'tiled']")
        if self.delassus_kernel not in ("tiled", "par_row_col", "auto"):
            raise ValueError("delassus_kernel must be one of ['auto', 'par_row_col', 'tiled']")
        if self.pgs_kernel not in ("tiled", "loop", "auto", "tiled_contact", "streaming"):
            raise ValueError("pgs_kernel must be one of ['auto', 'loop', 'streaming', 'tiled', 'tiled_contact']")
        self._pgs_chunk_size = int(self._kernel_overrides.get("pgs_chunk_size", 1))
        if self._pgs_chunk_size < 1:
            raise ValueError("pgs_chunk_size must be positive")
        self._tile_threads = int(self._kernel_overrides.get("tile_threads", _TILE_THREADS))
        if self._tile_threads not in (32, 64, 128, 256):
            raise ValueError("tile_threads must be one of (32, 64, 128, 256)")
        self._serial_kernel_block_dim = int(
            self._kernel_overrides.get("serial_kernel_block_dim", _SERIAL_KERNEL_BLOCK_DIM)
        )
        if self._serial_kernel_block_dim <= 0 or self._serial_kernel_block_dim % 32:
            raise ValueError("serial_kernel_block_dim must be a positive multiple of 32")
        if pgs_mode == "matrix_free":
            # The dense Gauss-Seidel kernels belong to the split solve.
            self.pgs_kernel = "loop"
        elif self.pgs_kernel in ("tiled_contact", "streaming") and (
            self.enable_joint_limits or np.isfinite(self.contact_friction_gap_threshold)
        ):
            # These kernels solve whole contacts (one normal and two friction rows) only.
            raise ValueError(
                f"pgs_kernel={self.pgs_kernel!r} solves contact rows only; it rejects joint-limit and normal-only rows"
            )
        if not model.device.is_cuda:
            # The tiled and native kernels are CUDA-only; CPU runs the scalar Warp kernels.
            self.cholesky_kernel = "loop"
            self.trisolve_kernel = "loop"
            self.hinv_jt_kernel = "par_row"
            self.delassus_kernel = "par_row_col"
            self.pgs_kernel = "loop"
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
        self._build_mimic_plan(model)
        self._build_connect_plan(model)
        if self.pgs_mode == "split" and (self._mimic_count or self._connect_count):
            # The split solve has no bilateral rows; it would project them as contacts.
            raise NotImplementedError("Mimic relationships and loop-closing joints require pgs_mode='matrix_free'")
        if self.articulated_contact_response != "immediate" and (self._mimic_count or self._connect_count):
            raise NotImplementedError(
                "Mimic relationships and loop-closing joints are not supported with "
                f"articulated_contact_response={self.articulated_contact_response!r}"
            )
        self._build_preelimination_plan(model)
        self._setup_passive_joint_forces(model)
        self._compute_world_response_dof_mapping(model)
        self.parallel_tree = bool(parallel_tree)
        self.use_parallel_streams = bool(use_parallel_streams)
        self.double_buffer = bool(double_buffer)
        # The cooperative traversal synchronizes warp lanes, so CPU devices keep the serial one.
        self._tree_plan = (
            _FeatherPGSTreePlan.build(model, self.articulation_joint_end.numpy())
            if self.parallel_tree and model.device.is_cuda
            else None
        )
        self._tree_net_wrenches = (
            tuple(
                wp.empty(group.segment_offsets.shape[0] - 1, dtype=wp.spatial_vector, device=model.device)
                for group in self._tree_plan.groups
            )
            if self._tree_plan is not None
            else ()
        )
        self._setup_propagation(model)
        self.dense_max_constraints = self._requested_dense_max_constraints
        self._setup_diagonal_mass(model)
        self._setup_sparse_mass_matrix(model)
        self._execution_plan = _FeatherPGSExecutionPlan.build(
            self.size_groups,
            max_constraints=self.dense_max_constraints,
            max_shared_memory=int(getattr(model.device, "max_shared_memory_per_block", 0)),
            cholesky_kernel=self.cholesky_kernel,
            hinv_jt_kernel=self.hinv_jt_kernel,
            small_dof_threshold=self.small_dof_threshold,
            tile_threads=self._tile_threads,
            diagonal_mass_sizes=self._diagonal_mass_sizes,
        )
        split = self.pgs_mode == "split"
        # Split mode assembles the Delassus matrix from the grouped responses and never
        # reads world-gathered J/Y; sparse rows store factor-coordinate responses.
        self._jy_world_aliased = (
            not split and self._sparse_mass_matrix_size is None and self._detect_jy_world_identity()
        )
        # Matrix-free tiled H^-1 J^T writes the world-gathered response directly unless the
        # group and world layouts already alias, and also computes the row diagonal.
        # Bilateral pre-elimination corrects the grouped response afterwards, so it keeps
        # the group response, the separate world gather and the separate diagonal pass.
        self._hinv_jt_writes_world = not split and not self._jy_world_aliased and not self._preelim_active
        self._hinv_jt_tiled_writes_group = not self._hinv_jt_writes_world
        self._hinv_jt_diag_sizes = frozenset(
            size for size in self.size_groups if self._execution_plan.use_tiled_hinv_jt(size)
        )
        if not self._hinv_jt_writes_world or self._preelim_active or self._sparse_mass_matrix_size is not None:
            self._hinv_jt_diag_sizes = frozenset()

        self._allocate_common_buffers(model)
        self._allocate_buffers(model)
        self._allocate_world_buffers(model)
        self._allocate_mf_buffers(model)
        # Contact rows keep row_parent for their friction/patch load linkage; torsion keeps
        # its own group membership.
        self._contact_torsion_group = wp.full(
            self.row_parent.shape if self._contact_torsion_enabled else (1, 1),
            -1,
            dtype=wp.int32,
            device=model.device,
        )
        self._allocate_propagation_buffers(model)
        self.mf_target_velocity = (
            wp.zeros_like(self.mf_rhs)
            if self._has_prescribed_response
            else wp.zeros((1, 1), dtype=wp.float32, device=model.device)
        )
        self._scatter_armature_to_groups()
        self._init_tiled_kernels(model)
        self._init_streams(model)
        # Free-body groups read their body inertia directly; every other responding
        # articulation reads composite inertias.
        response_dof_count = self._model_plan.response_dof_count
        composite_articulations = np.flatnonzero(
            (response_dof_count > 0) & ~np.isin(response_dof_count, tuple(self._free_body_inertia_sizes))
        ).astype(np.int32)
        self._composite_articulation_count = int(composite_articulations.size)
        self._composite_articulations = wp.array(composite_articulations, dtype=wp.int32, device=model.device)
        # The warp reduction is a native CUDA kernel; CPU devices use the scalar reduction.
        self._composite_inertia_warp_kernel = (
            _get_composite_inertia_warp_kernel(str(model.device.arch), _COMPOSITE_INERTIA_WARPS_PER_BLOCK)
            if self._composite_articulation_count and model.device.is_cuda
            else None
        )
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

        self.shape_material_mu = wp.zeros(max(model.shape_count, 1), dtype=wp.float32, device=model.device)
        self.shape_material_restitution = wp.zeros(max(model.shape_count, 1), dtype=wp.float32, device=model.device)
        self._refresh_shape_materials()
        self._check_restitution_support()

        self.contact_torsion_device = bool(contact_torsion_device)
        self._device_torsion = None
        if self._contact_torsion_enabled and self.contact_torsion_device:
            from .contact_torsion_device import enable_device_torsion  # noqa: PLC0415

            enable_device_torsion(self)
        # Per-entity activity of the articulated dynamics and of the constraint rows; sleeping
        # clears the entries of islands that sleep through a step, otherwise they stay one.
        self._dynamics_art_active = wp.ones(model.articulation_count, dtype=wp.int32, device=model.device)
        self._dynamics_art_mask = wp.ones(model.articulation_count, dtype=wp.bool, device=model.device)
        self._dynamics_joint_active = wp.ones(model.joint_count, dtype=wp.int32, device=model.device)
        self._dynamics_dof_active = wp.ones(model.joint_dof_count, dtype=wp.int32, device=model.device)
        self._dynamics_body_active = wp.ones(model.body_count, dtype=wp.int32, device=model.device)
        self._constraint_art_active = wp.ones(model.articulation_count, dtype=wp.int32, device=model.device)
        self.sleeping = (
            _SleepState(
                self,
                float(sleep_linear_threshold),
                float(sleep_angular_threshold),
                float(sleep_quiet_time),
                bool(sleep_skip_constraints),
            )
            if enable_sleeping
            else None
        )

    def _refresh_shape_materials(self) -> None:
        """Copy the model's current friction and restitution coefficients into the solver's fixed buffers."""
        # Users may replace the model arrays; captured graphs keep these buffers' addresses.
        model = self.model
        for buffer, values in (
            (self.shape_material_mu, model.shape_material_mu),
            (self.shape_material_restitution, model.shape_material_restitution),
        ):
            if values is not None and model.shape_count:
                wp.copy(buffer, values, count=model.shape_count)

    @property
    def contact_torsion_radius(self) -> float:
        """Effective contact torsion radius [m]; ``0`` when torsion is disabled."""
        return self._contact_torsion_radius

    @property
    def contact_torsion_shape_indices(self) -> tuple[int, ...] | None:
        """Shape indices selecting contact torsion, as given at construction."""
        return self._contact_torsion_shape_indices

    @property
    def contact_torsion_shape_patterns(self) -> tuple[str, ...] | None:
        """Shape-label patterns selecting contact torsion, as given at construction."""
        return self._contact_torsion_shape_patterns

    def prepare_contact_torsion_capture(self, state_in: State, state_out: State) -> None:
        """Prepare device contact torsion for CUDA graph capture of :meth:`step`.

        Call outside capture, with the states the graph will use, before capturing a
        step with contact torsion. It requires ``contact_torsion_device=True`` and does not
        advance the simulation. Afterwards torsion errors latch on the device: the failing
        step publishes its input state unchanged and solves no rows. An eager :meth:`step`
        still validates and raises them when it returns; a graph replay cannot, so call
        :meth:`validate_contact_torsion` after replays. Does nothing when contact torsion
        is disabled.

        Args:
            state_in: Input state of the captured step.
            state_out: Output state of the captured step.
        """
        if not self._contact_torsion_enabled:
            return
        preparation = self._device_torsion
        if preparation is None:
            raise RuntimeError("Contact torsion graph capture requires contact_torsion_device=True")
        if wp.get_stream(self.model.device).is_capturing:
            raise RuntimeError("Prepare contact torsion buffers outside CUDA graph capture")
        preparation.validate()
        preparation.deferred_errors = True
        preparation.begin_step(state_in, state_out)

    def validate_contact_torsion(self) -> None:
        """Raise latched device contact torsion errors.

        After :meth:`prepare_contact_torsion_capture`, call this outside capture after
        every batch of graph replays, before using their results. Errors stay latched:
        reconstruct the solver after correcting the input. Does nothing without device
        torsion preparation.
        """
        preparation = self._device_torsion
        if preparation is not None:
            if wp.get_stream(self.model.device).is_capturing:
                raise RuntimeError("Validate contact torsion outside CUDA graph capture")
            preparation.validate()

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

        self._validate_connect_parent_membership()
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
        membership), joint DOF properties (armature, damping) and shape friction and
        restitution coefficients are re-read. Other model data, such as gravity, limits and
        shape transforms, is read every step. A kinematic free
        body that was removed from the response at construction cannot become dynamic
        again; reconstruct the solver in that case. Constraint changes re-check the MuJoCo
        equality rows: enabling one that is not converted to a Newton loop joint or mimic
        constraint raises :class:`NotImplementedError`. Capacity status in
        :attr:`constraint_overflow` is not cleared, see :meth:`reset`.

        Body, inertial, joint DOF and shape changes discard the warm-start impulses. With
        friction patches, shape changes also discard the anchors of the bodies whose
        collision geometry changed; this copies the shape geometry to the host, so issue
        such notifications outside CUDA graph capture.

        Args:
            flags: Bit-mask of :class:`~newton.ModelFlags` indicating which model properties changed.
        """
        if flags & ModelFlags.CONSTRAINT_PROPERTIES:
            _validate_equality_constraints(self.model)
        if self.sleeping is not None:
            self.sleeping.notify(flags)
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES):
            self._update_kinematic_state()
            self._scatter_armature_to_groups()
            self._mass_update_requested.fill_(1)
        if flags & ModelFlags.JOINT_DOF_PROPERTIES:
            self._refresh_passive_joint_damping()
        if flags & ModelFlags.SHAPE_PROPERTIES:
            self._refresh_shape_materials()
            self._check_restitution_support()
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
        if self._friction_anchors_enabled and flags & ModelFlags.SHAPE_PROPERTIES:
            self._friction_patches.update_geometry(self.model)
        if (
            self._sparse_mass_matrix_size is not None
            and flags & ModelFlags.SHAPE_PROPERTIES
            and not self._sparse_supports_contact_options(self.model)
        ):
            raise ValueError(
                "Sparse mass factors do not apply contact restitution; reconstruct the solver after "
                "giving shapes a positive restitution"
            )
        if flags & (
            ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        ):
            self._clear_warmstart_history(None)

    def _clear_warmstart_history(self, world_mask: wp.array[wp.bool] | None) -> None:
        """Discard the carried contact impulses of the selected worlds (all worlds for ``None``)."""
        if not self.pgs_warmstart or self.world_count == 0:
            return
        families = [(self._ws_prev_impulses, self._ws_prev_row_type, self._ws_prev_row_parent)]
        if self._has_free_rigid_bodies:
            families.append((self._ws_prev_mf_impulses, self._ws_prev_mf_row_type, self._ws_prev_mf_row_parent))
        for impulses, row_type, row_parent in families:
            wp.launch(
                reset_row_warmstart,
                dim=impulses.shape,
                inputs=[
                    world_mask,
                    int(self.model.world_count),
                    int(self._has_responding_global),
                    impulses,
                    row_type,
                    row_parent,
                ],
                device=self.model.device,
            )

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Clear solver-owned state of the selected worlds.

        The simulation state is not modified. For the selected worlds this clears the
        capacity status in :attr:`constraint_overflow`, discards the warm-start impulses
        and the friction-patch anchors, and requests a mass-matrix refresh on the next
        step, so a teleported articulation does not reuse a stale factorization or anchor;
        other worlds keep their refresh cadence and history. An anchor between bodies of
        two entries (for example a global body and a world body) is discarded when either
        is selected. Global articulations share world 0's rows, so selecting the global
        entry also discards world 0's warm-start impulses while a dynamic global
        articulation exists. The reset is launched on the device without host
        synchronization and can be captured in a CUDA graph.

        Args:
            state: Simulation state; left unchanged.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``. The first
                ``world_count`` entries select worlds; the final entry selects global
                articulations (world ``-1``) and the final entry of
                :attr:`constraint_overflow`. ``None`` resets everything.
            flags: Unused; the solver's history is cleared regardless of the state flags.
        """
        del flags
        if self.contact_compliance:
            _contact_compliance.clear(self)
        world_mask = self._normalize_reset_world_mask(world_mask)
        if self.sleeping is not None:
            self.sleeping.wake(world_mask)
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
        if self._friction_anchors_enabled:
            self._friction_patches.reset(world_mask, int(self.model.world_count))
        self._clear_warmstart_history(world_mask)

    def _compute_articulation_metadata(self, model):
        self._compute_articulation_indices(model)
        self._compute_root_free_metadata(model)
        self._setup_size_grouping(model)
        self._setup_world_mapping(model)
        self._build_body_maps(model)
        self._classify_free_rigid_bodies(model)
        self._compact_contact_jacobian = bool(
            model.device.is_cuda
            and self.body_response_dof_mask is not None
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
            # Tree joints only: a loop joint in this range may close another articulation.
            children = joint_child[articulation_start[articulation] : plan.articulation_joint_end[articulation]]
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

    def _build_mimic_plan(self, model) -> None:
        """Build the static lookup tables of the mimic rows.

        Rows come from joint-owned mimics (:attr:`~newton.Model.joint_mimic_joint`, one row
        per follower coordinate) and from the deprecated ``Model.constraint_mimic_*``
        entries, which take precedence for their follower joint as in
        :class:`~newton.solvers.SolverMuJoCo`. Both joints of a row must belong to the same
        articulation and have distinct DOFs; legacy entries must couple REVOLUTE or
        PRISMATIC joints. Unsupported relationships raise :class:`NotImplementedError`.
        Coefficients and ``constraint_mimic_enabled`` are read every step. All buffers are
        allocated here, so the per-step launches are compatible with CUDA graph capture.
        """
        self._mimic_count = 0
        self._mimic_art_start_np = None
        self._mimic_sizes = frozenset()
        self.mimic_slot = None
        if not model.articulation_count or not model.joint_count:
            return

        legacy_count = int(getattr(model, "constraint_mimic_count", 0) or 0)
        joint_mimic = (
            model.joint_mimic_joint.numpy().astype(np.int32, copy=False)
            if model.joint_mimic_joint is not None
            else np.full(model.joint_count, -1, dtype=np.int32)
        )
        if legacy_count == 0 and not np.any(joint_mimic >= 0):
            return

        joint_type = model.joint_type.numpy()
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_q_start = model.joint_q_start.numpy().astype(np.int32, copy=False)
        joint_articulation = model.joint_articulation.numpy().astype(np.int32, copy=False)
        response_dof_count = self._model_plan.response_dof_count
        articulation_world = self._model_plan.articulation_world

        # Per row: follower/leader DOF and coordinate, articulation, and the coefficient
        # source (legacy constraint index, or the follower joint for joint-owned rows).
        dof0, dof1, q0, q1, row_art, legacy, owner = [], [], [], [], [], [], []

        def add_row(name: str, follower_dof: int, leader_dof: int, follower_q: int, leader_q: int, joints, source):
            a0, a1 = (int(joint_articulation[j]) for j in joints)
            if a0 < 0 or a0 != a1:
                raise NotImplementedError(
                    f"SolverFeatherPGS: mimic '{name}' couples joints of different articulations "
                    f"({a0} and {a1}); only mimics within one articulation are supported."
                )
            if follower_dof == leader_dof:
                raise ValueError(f"SolverFeatherPGS: mimic '{name}' couples a DOF to itself.")
            if response_dof_count[a0] == 0:
                return  # A fully prescribed articulation has nothing to enforce.
            dof0.append(follower_dof)
            dof1.append(leader_dof)
            q0.append(follower_q)
            q1.append(leader_q)
            row_art.append(a0)
            legacy.append(source[0])
            owner.append(source[1])

        legacy_followers = set()
        if legacy_count:
            j0 = model.constraint_mimic_joint0.numpy().astype(np.int32, copy=False)
            j1 = model.constraint_mimic_joint1.numpy().astype(np.int32, copy=False)
            labels = list(model.constraint_mimic_label or [])
            one_dof = (int(JointType.REVOLUTE), int(JointType.PRISMATIC))
            legacy_followers = {int(j) for j in j0}
            for k in range(legacy_count):
                name = labels[k] if k < len(labels) and labels[k] else f"constraint_mimic_{k}"
                follower, leader = int(j0[k]), int(j1[k])
                if int(joint_type[follower]) not in one_dof or int(joint_type[leader]) not in one_dof:
                    raise NotImplementedError(
                        f"SolverFeatherPGS: mimic constraint '{name}' must couple REVOLUTE or PRISMATIC joints."
                    )
                add_row(
                    name,
                    int(joint_qd_start[follower]),
                    int(joint_qd_start[leader]),
                    int(joint_q_start[follower]),
                    int(joint_q_start[leader]),
                    (follower, leader),
                    (k, -1),
                )

        joint_qd_end = np.append(joint_qd_start[1:], model.joint_dof_count)
        joint_q_end = np.append(joint_q_start[1:], model.joint_coord_count)
        for follower in np.flatnonzero(joint_mimic >= 0).tolist():
            if follower in legacy_followers:
                continue
            leader = int(joint_mimic[follower])
            for joint in (follower, leader):
                if joint_q_end[joint] - joint_q_start[joint] != joint_qd_end[joint] - joint_qd_start[joint]:
                    raise NotImplementedError(
                        f"SolverFeatherPGS: mimic joint {joint} ({JointType(joint_type[joint]).name}) has different "
                        "position and velocity coordinate counts; only REVOLUTE, PRISMATIC, FIXED and D6 "
                        "mimics are supported."
                    )
            for i in range(int(joint_qd_end[follower] - joint_qd_start[follower])):
                add_row(
                    f"joint_mimic_{follower}[{i}]",
                    int(joint_qd_start[follower]) + i,
                    int(joint_qd_start[leader]) + i,
                    int(joint_q_start[follower]) + i,
                    int(joint_q_start[leader]) + i,
                    (follower, leader),
                    (-1, follower),
                )

        n = len(dof0)
        if n == 0:
            return
        row_art_np = np.asarray(row_art, dtype=np.int32)
        order = np.argsort(row_art_np, kind="stable").astype(np.int32)
        counts = np.bincount(row_art_np, minlength=model.articulation_count)
        art_start = np.zeros(model.articulation_count + 1, dtype=np.int32)
        art_start[1:] = np.cumsum(counts)

        device = model.device
        self._mimic_art_start_np = art_start
        self._mimic_sizes = frozenset(int(response_dof_count[a]) for a in np.unique(row_art_np))
        self._mimic_art_start = wp.array(art_start, dtype=wp.int32, device=device)
        self._mimic_art_list = wp.array(order, dtype=wp.int32, device=device)
        self._mimic_world = wp.array(articulation_world[row_art_np], dtype=wp.int32, device=device)
        self._mimic_dof0 = wp.array(dof0, dtype=wp.int32, device=device)
        self._mimic_dof1 = wp.array(dof1, dtype=wp.int32, device=device)
        self._mimic_q0 = wp.array(q0, dtype=wp.int32, device=device)
        self._mimic_q1 = wp.array(q1, dtype=wp.int32, device=device)
        self._mimic_legacy = wp.array(legacy, dtype=wp.int32, device=device)
        self._mimic_owner = wp.array(owner, dtype=wp.int32, device=device)
        if legacy_count:
            self._mimic_enabled = model.constraint_mimic_enabled
            self._mimic_coef0 = model.constraint_mimic_coef0
            self._mimic_coef1 = model.constraint_mimic_coef1
        else:
            self._mimic_enabled = wp.zeros(1, dtype=wp.bool, device=device)
            self._mimic_coef0 = wp.zeros(1, dtype=wp.float32, device=device)
            self._mimic_coef1 = wp.zeros(1, dtype=wp.float32, device=device)
        self._joint_mimic_coeffs = (
            model.joint_mimic_coeffs
            if model.joint_mimic_coeffs is not None
            else wp.zeros(1, dtype=wp.vec2, device=device)
        )
        self.mimic_slot = wp.full((n,), -1, dtype=wp.int32, device=device)
        self._mimic_count = n

    def _build_connect_plan(self, model) -> None:
        """Build the static lookup tables of the connect (loop-closure) rows.

        Loop-closing BALL joints are kept out of the articulation tree by
        :meth:`_FeatherPGSModelPlan.build` and enforced as three rows pinning the joint's
        parent and child anchors together. A closure belongs to its child body's
        articulation. Its parent is either in that articulation or prescribed: a kinematic
        body (:attr:`~newton.BodyFlags.KINEMATIC`) or the world. A prescribed parent
        contributes no DOFs; its anchor velocity enters the row target. Other loop-closing
        joints raise :class:`NotImplementedError`. The anchors are the joint frames' origins
        at construction and can be changed with :meth:`set_loop_joint_anchors`; a closure
        whose joint is disabled in :attr:`~newton.Model.joint_enabled` starts released, see
        :meth:`set_loop_joint_enabled`.
        """
        self._connect_count = 0
        self.connect_slot = None
        self._connect_joint_to_index: dict[int, int] = {}
        loop_joint_articulation = self._model_plan.loop_joint_articulation
        loop_joints = np.flatnonzero(loop_joint_articulation >= 0)
        if loop_joints.size == 0:
            return

        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_X_p = model.joint_X_p.numpy()
        joint_X_c = model.joint_X_c.numpy()
        joint_enabled = model.joint_enabled.numpy() if model.joint_enabled is not None else None
        body_articulation = self.body_to_articulation.numpy()
        kinematic_bodies = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
        response_dof_count = self._model_plan.response_dof_count
        articulation_world = self._model_plan.articulation_world

        art_l, body_p, body_c, anchors_p, anchors_c, enabled, prescribed_l = [], [], [], [], [], [], []
        joint_l, foreign_l = [], []
        for j in loop_joints.tolist():
            art = int(loop_joint_articulation[j])
            if int(joint_type[j]) != int(JointType.BALL):
                raise NotImplementedError(
                    f"SolverFeatherPGS: loop-closing joint {j} is a {JointType(joint_type[j]).name} joint; "
                    "only BALL loop-closing joints are supported."
                )
            parent = int(joint_parent[j])
            prescribed = 0
            foreign = 0
            if parent < 0:
                prescribed = 1
            elif int(body_articulation[parent]) != art:
                foreign = 1
                if not kinematic_bodies[parent]:
                    raise NotImplementedError(
                        f"SolverFeatherPGS: loop-closing joint {j} connects articulations "
                        f"{int(body_articulation[parent])} and {art}; its parent must belong to the child's "
                        "articulation, be kinematic or be the world."
                    )
                prescribed = 1
            if response_dof_count[art] == 0:
                continue  # A fully prescribed articulation has nothing to enforce.
            self._connect_joint_to_index[j] = len(art_l)
            art_l.append(art)
            body_p.append(parent)
            body_c.append(int(joint_child[j]))
            anchors_p.append(joint_X_p[j][:3])
            anchors_c.append(joint_X_c[j][:3])
            enabled.append(1 if joint_enabled is None or joint_enabled[j] else 0)
            prescribed_l.append(prescribed)
            joint_l.append(j)
            foreign_l.append(foreign)

        n = len(art_l)
        if n == 0:
            return
        device = model.device
        art_np = np.asarray(art_l, dtype=np.int32)
        self._connect_art_np = art_np
        self._connect_parent_prescribed_np = np.asarray(prescribed_l, dtype=np.int32)
        # Closures whose parent is a kinematic body of another articulation; body-flag
        # notifications re-check that the parent stays kinematic.
        self._connect_parent_foreign_np = np.asarray(foreign_l, dtype=np.int32)
        self._connect_body_p_np = np.asarray(body_p, dtype=np.int32)
        self._connect_joint_np = np.asarray(joint_l, dtype=np.int32)
        self._connect_enabled_np = np.asarray(enabled, dtype=np.int32)
        self._connect_anchor_p_np = np.asarray(anchors_p, dtype=np.float32).reshape(n, 3)
        self._connect_anchor_c_np = np.asarray(anchors_c, dtype=np.float32).reshape(n, 3)
        self._connect_art = wp.array(art_np, dtype=wp.int32, device=device)
        self._connect_world = wp.array(articulation_world[art_np], dtype=wp.int32, device=device)
        self._connect_body_p = wp.array(body_p, dtype=wp.int32, device=device)
        self._connect_body_c = wp.array(body_c, dtype=wp.int32, device=device)
        self._connect_anchor_p = wp.array(self._connect_anchor_p_np, dtype=wp.vec3, device=device)
        self._connect_anchor_c = wp.array(self._connect_anchor_c_np, dtype=wp.vec3, device=device)
        self._connect_parent_prescribed = wp.array(self._connect_parent_prescribed_np, dtype=wp.int32, device=device)
        self._connect_enabled = wp.array(self._connect_enabled_np, dtype=wp.int32, device=device)
        self._connect_sizes = frozenset(int(response_dof_count[a]) for a in np.unique(art_np))
        self.connect_slot = wp.full((n,), -1, dtype=wp.int32, device=device)
        self._connect_count = n

    def _validate_connect_parent_membership(self) -> None:
        """Reject a body-flag change that would invalidate the prescribed-parent closures.

        A closure whose parent is a kinematic body of another articulation enforces only
        the child side, with the parent's anchor velocity as the row target. If that
        parent became dynamic, the closure would couple two dynamic articulations, which
        :meth:`_build_connect_plan` rejects at construction; raise the same way instead of
        keeping the one-way rows.
        """
        if not getattr(self, "_connect_count", 0):
            return
        kinematic = (self.model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
        for index in np.flatnonzero(self._connect_parent_foreign_np).tolist():
            parent = int(self._connect_body_p_np[index])
            if not kinematic[parent]:
                raise NotImplementedError(
                    f"SolverFeatherPGS: the parent body {parent} of loop-closing joint "
                    f"{int(self._connect_joint_np[index])} belongs to another articulation and is no longer "
                    "kinematic. A closure between two dynamic articulations is not supported; keep the parent "
                    "kinematic, or remove the loop joint from the model and reconstruct the solver."
                )

    def _connect_index(self, joint: int) -> int:
        index = self._connect_joint_to_index.get(int(joint))
        if index is None:
            raise ValueError(f"SolverFeatherPGS: joint {joint} is not an enforced loop-closing BALL joint.")
        return index

    def set_loop_joint_enabled(self, joint: int, enabled: bool) -> None:
        """Enable or release the connect rows of a loop-closing BALL joint.

        Rows are reserved into fixed buffers every step, so toggling takes effect on the
        next :meth:`step` without recapturing a CUDA graph. Call it outside graph capture.

        Args:
            joint: Model joint index of a loop-closing BALL joint.
            enabled: ``True`` to enforce the closure, ``False`` to release it.
        """
        index = self._connect_index(joint)
        self._connect_enabled_np[index] = 1 if enabled else 0
        self._connect_enabled.assign(self._connect_enabled_np)

    def set_loop_joint_anchors(self, joint: int, parent_anchor: Vec3, child_anchor: Vec3) -> None:
        """Move the anchors of a loop-closing BALL joint.

        Use it, for example, to attach a body to a prescribed carrier at the relative pose
        measured when the closure is engaged. Call it outside graph capture.

        Args:
            joint: Model joint index of a loop-closing BALL joint.
            parent_anchor: Anchor in the parent body frame [m], or in the world frame
                for a world parent.
            child_anchor: Anchor in the child body frame [m].
        """
        index = self._connect_index(joint)
        self._connect_anchor_p_np[index] = np.asarray(parent_anchor, dtype=np.float32)
        self._connect_anchor_c_np[index] = np.asarray(child_anchor, dtype=np.float32)
        self._connect_anchor_p.assign(self._connect_anchor_p_np)
        self._connect_anchor_c.assign(self._connect_anchor_c_np)

    def loop_joint_anchors(self, joint: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(parent_anchor, child_anchor)`` of a loop-closing BALL joint [m]."""
        index = self._connect_index(joint)
        return self._connect_anchor_p_np[index].copy(), self._connect_anchor_c_np[index].copy()

    def _build_preelimination_plan(self, model) -> None:
        """Build the static tables of bilateral pre-elimination.

        Per articulation that owns bilateral rows this allocates an index into the packed
        per-articulation buffers and the factor scratch. The row slots and the factor are
        rebuilt on the device every step. Unsupported configurations warn and keep the
        mimic and connect rows in the iterative sweep.
        """
        self._preelim_count = 0
        self._preelim_active = False
        include_mimics = self.bilateral_preelimination_include_mimics
        if not self.enable_bilateral_preelimination:
            return
        if self._connect_count == 0 and (self._mimic_count == 0 or not include_mimics):
            return
        if self.pgs_velocity_iterations > 0:
            # The velocity-only iterations rebuild the velocity without the bilateral projection.
            warnings.warn(
                "SolverFeatherPGS: enable_bilateral_preelimination does not support pgs_velocity_iterations > 0; "
                "pre-elimination is disabled for the whole solver and every articulation keeps iterative mimic "
                "and connect rows.",
                stacklevel=3,
            )
            return
        if self._connect_count and np.any(self._connect_parent_prescribed_np != 0):
            warnings.warn(
                "SolverFeatherPGS: enable_bilateral_preelimination does not support loop closures with a "
                "kinematic or world parent; pre-elimination is disabled for the whole solver and every "
                "articulation keeps iterative mimic and connect rows.",
                stacklevel=3,
            )
            return

        # Per-articulation bilateral row capacity (every row the articulation can own).
        counts: dict[int, int] = {}
        if self._mimic_count and include_mimics:
            for art, count in enumerate(np.diff(self._mimic_art_start_np)):
                if count:
                    counts[art] = int(count)
        for art in self._connect_art_np.tolist() if self._connect_count else ():
            counts[art] = counts.get(art, 0) + 3
        if not counts:
            return
        rows = max(counts.values())
        if rows > PREELIM_MAX_ROWS:
            warnings.warn(
                f"SolverFeatherPGS: an articulation owns {rows} bilateral rows, more than the pre-elimination "
                f"capacity of {PREELIM_MAX_ROWS}; pre-elimination is disabled for the whole solver and every "
                "articulation keeps iterative mimic and connect rows. Set "
                "bilateral_preelimination_include_mimics=False to pre-eliminate only the connect rows.",
                stacklevel=3,
            )
            return

        device = model.device
        arts = sorted(counts)
        art_to_idx = np.full(model.articulation_count, -1, dtype=np.int32)
        art_to_idx[arts] = np.arange(len(arts), dtype=np.int32)
        n_pe = len(arts)
        self._preelim_art_to_idx = wp.array(art_to_idx, dtype=wp.int32, device=device)
        self._preelim_slots = wp.full((n_pe * PREELIM_MAX_ROWS,), -1, dtype=wp.int32, device=device)
        self._preelim_nB = wp.zeros((n_pe,), dtype=wp.int32, device=device)
        self._preelim_S = wp.zeros((n_pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS,), dtype=wp.float32, device=device)
        self._preelim_LS = wp.zeros_like(self._preelim_S)
        self._preelim_reg = wp.zeros((n_pe * PREELIM_MAX_ROWS,), dtype=wp.float32, device=device)
        # Relative diagonal regularization of the block factor, see preelim_setup_for_size.
        self._preelim_reg_rel = 1.0e-3
        self._preelim_reg_floor = max(self.pgs_cfm, 1.0e-7)
        self._preelim_count = n_pe
        self._preelim_active = True

    def _stage4_build_bilateral_rows(self, state_in: State, state_aug: State) -> None:
        """Allocate and fill the mimic and connect rows; they precede the joint-limit rows."""
        model = self.model
        max_constraints = self.dense_max_constraints
        if self._mimic_count:
            wp.launch(
                allocate_mimic_slots,
                dim=self._mimic_count,
                inputs=[self._mimic_legacy, self._mimic_enabled, self._mimic_world, max_constraints],
                outputs=[
                    self.mimic_slot,
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                    self._row_dropped_dense,
                ],
                device=model.device,
            )
            for size in self.size_groups:
                if size not in self._mimic_sizes:
                    continue
                wp.launch(
                    populate_mimic_J_for_size,
                    dim=self.n_arts_by_size[size],
                    inputs=[
                        self.articulation_dof_start,
                        self.art_to_world,
                        self.group_to_art[size],
                        self.mimic_slot,
                        self._mimic_art_start,
                        self._mimic_art_list,
                        self._mimic_dof0,
                        self._mimic_dof1,
                        self._mimic_q0,
                        self._mimic_q1,
                        self._mimic_legacy,
                        self._mimic_owner,
                        self._mimic_coef0,
                        self._mimic_coef1,
                        self._joint_mimic_coeffs,
                        state_in.joint_q,
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
        if self._connect_count:
            wp.launch(
                allocate_connect_slots,
                dim=self._connect_count,
                inputs=[self._connect_enabled, self._connect_world, max_constraints],
                outputs=[
                    self.connect_slot,
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                    self._row_dropped_dense,
                ],
                device=model.device,
            )
            for size in self._connect_sizes:
                wp.launch(
                    populate_connect_J_for_size,
                    dim=self.n_arts_by_size[size],
                    inputs=[
                        self.articulation_dof_start,
                        self.art_to_world,
                        self.group_to_art[size],
                        size,
                        self.connect_slot,
                        self._connect_art,
                        self._connect_body_p,
                        self._connect_body_c,
                        self._connect_anchor_p,
                        self._connect_anchor_c,
                        self._connect_parent_prescribed,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        self.body_to_joint,
                        model.joint_ancestor,
                        model.joint_qd_start,
                        state_aug.joint_S_s,
                        self.articulation_origin,
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

    def _stage4_bilateral_preelim(self) -> None:
        """Form and factor each bilateral block and correct the other rows' responses.

        Runs after ``Y = H^-1 J^T`` and before the diagonal pass, so the corrected
        diagonals follow from the unchanged ``J Y`` computation.
        """
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            wp.launch(
                preelim_setup_for_size,
                dim=n_arts,
                inputs=[
                    self.group_to_art[size],
                    self._preelim_art_to_idx,
                    self.mimic_slot if self._mimic_count else self._dummy_is_free_rigid,
                    self._mimic_art_start if self._mimic_count else self._dummy_is_free_rigid,
                    self._mimic_art_list if self._mimic_count else self._dummy_is_free_rigid,
                    self._mimic_count if self.bilateral_preelimination_include_mimics else 0,
                    self.connect_slot if self._connect_count else self._dummy_is_free_rigid,
                    self._connect_art if self._connect_count else self._dummy_is_free_rigid,
                    self._connect_count,
                    self.J_by_size[size],
                    self.Y_by_size[size],
                    size,
                    self._preelim_reg_rel,
                    self._preelim_reg_floor,
                ],
                outputs=[
                    self._preelim_slots,
                    self._preelim_nB,
                    self._preelim_S,
                    self._preelim_reg,
                    self._preelim_LS,
                ],
                device=self.model.device,
            )
            wp.launch(
                preelim_correct_Y_for_size,
                dim=n_arts * self.dense_max_constraints,
                inputs=[
                    self.group_to_art[size],
                    self.art_to_world,
                    self._preelim_art_to_idx,
                    self.constraint_count,
                    self._preelim_slots,
                    self._preelim_nB,
                    self._preelim_LS,
                    self.J_by_size[size],
                    size,
                    self.dense_max_constraints,
                    n_arts,
                ],
                outputs=[self.Y_by_size[size]],
                device=self.model.device,
            )

    def _stage5_project_bilateral_velocity(self) -> None:
        """Project the predictor velocity once toward ``J_B v = -b_B``, up to the regularization residual."""
        for size in self.size_groups:
            wp.launch(
                preelim_project_velocity_for_size,
                dim=self.n_arts_by_size[size],
                inputs=[
                    self.group_to_art[size],
                    self.art_to_world,
                    self._preelim_art_to_idx,
                    self.articulation_dof_start,
                    self._preelim_slots,
                    self._preelim_nB,
                    self._preelim_LS,
                    self.J_by_size[size],
                    self.Y_by_size[size],
                    self.rhs,
                    size,
                ],
                outputs=[self.v_out],
                device=self.model.device,
            )

    def _setup_passive_joint_forces(self, model) -> None:
        """Select the passive damping of the inverse-dynamics pass; its passive springs are zero.

        Passive joint springs are not applied yet (newton-physics/newton#4516): the spring
        buffers stay zero, and nonzero ``mujoco:dof_passive_stiffness`` warns at construction.
        """
        n_dofs = max(int(model.joint_dof_count), 1)
        self._passive_joint_damping = wp.zeros(n_dofs, dtype=wp.float32, device=model.device)
        self._refresh_passive_joint_damping()
        self._passive_spring_stiffness = wp.zeros(n_dofs, dtype=wp.float32, device=model.device)
        self._passive_spring_ref = wp.zeros(n_dofs, dtype=wp.float32, device=model.device)
        stiffness = getattr(getattr(model, "mujoco", None), "dof_passive_stiffness", None)
        if stiffness is not None and model.joint_dof_count and np.any(stiffness.numpy() != 0.0):
            warnings.warn(
                "SolverFeatherPGS does not support passive joint springs yet and ignores nonzero "
                "mujoco:dof_passive_stiffness (see newton-physics/newton#4516).",
                UserWarning,
                stacklevel=3,
            )

    def _refresh_passive_joint_damping(self) -> None:
        """Copy the model's current joint damping into the solver's fixed damping buffer."""
        # Users may replace model.joint_damping; captured graphs keep this buffer's address.
        damping = self.model.joint_damping
        if damping is not None and self.model.joint_dof_count:
            wp.copy(self._passive_joint_damping, damping, count=self.model.joint_dof_count)

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
                last_joint = int(self._model_plan.articulation_joint_end[i])

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
            and np.any(limit_joint[articulation_start[art] : self._model_plan.articulation_joint_end[art]])
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

    def _setup_diagonal_mass(self, model: Model) -> None:
        """Select the diagonal mass-matrix kernels for articulations without coupled DOFs.

        The selection is automatic and structural. A response group qualifies when all its
        articulations share one local topology in which every joint has at most one DOF and
        no DOF's joint is an ancestor of another DOF's joint, for example independent
        single-DOF branches of a fixed base. Every off-diagonal entry of their mass matrix is
        then zero, so the factor, the unconstrained solve and the row responses reduce to
        per-DOF divisions with the same results as the dense loop kernels. These groups keep
        the dense row and response storage.
        """
        self._diagonal_mass_sizes = frozenset()
        if not self._kernel_overrides.get("diagonal_mass", True):
            return
        # The split solve and the propagation responses build their responses from dense factors.
        if self.pgs_mode != "matrix_free" or self.articulated_contact_response != "immediate":
            return
        if (self.cholesky_kernel, self.trisolve_kernel, self.hinv_jt_kernel) != ("auto", "auto", "auto"):
            return
        if not self.size_groups or not model.joint_count:
            return
        plan = self._model_plan
        articulation_start = model.articulation_start.numpy()
        joint_end = self.articulation_joint_end.numpy()
        joint_ancestor = model.joint_ancestor.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        free_rigid = plan.is_free_rigid != 0
        diagonal_sizes: set[int] = set()
        for size in self.size_groups:
            arts = np.flatnonzero(plan.response_dof_count == size)
            if not arts.size or np.any(free_rigid[arts]):
                continue
            reference = None
            for art in arts:
                start, end = int(articulation_start[art]), int(joint_end[art])
                dof_start = int(plan.articulation_dof_start[art])
                # Local joint of each response DOF, and the local parent of each joint.
                dof_joint = np.full(int(size), -1, dtype=np.int32)
                for joint in range(start, end):
                    first = max(0, int(joint_qd_start[joint]) - dof_start)
                    last = min(int(size), int(joint_qd_start[joint + 1]) - dof_start)
                    dof_joint[first:last] = joint - start
                parents = joint_ancestor[start:end].astype(np.int32, copy=True)
                parents = np.where(parents >= start, parents - start, -1).astype(np.int32)
                if reference is None:
                    reference = (dof_joint, parents)
                elif not (np.array_equal(dof_joint, reference[0]) and np.array_equal(parents, reference[1])):
                    reference = None
                    break
            if reference is None:
                continue
            dof_joint, parents = reference
            # Every DOF has its own joint, and no DOF joint is an ancestor of another.
            if np.any(dof_joint < 0) or np.unique(dof_joint).size != dof_joint.size:
                continue
            owners = {int(joint) for joint in dof_joint}
            coupled = False
            for joint in owners:
                ancestor = int(parents[joint])
                while ancestor >= 0 and not coupled:
                    coupled = ancestor in owners
                    ancestor = int(parents[ancestor])
                if coupled:
                    break
            if not coupled:
                diagonal_sizes.add(int(size))
        self._diagonal_mass_sizes = frozenset(diagonal_sizes)

    def _sparse_supports_contact_options(self, model: Model) -> bool:
        """Whether the sparse factor kernels implement the configured contact law.

        They solve hard point-friction contacts (one normal and two tangent rows per
        contact) without restitution, so friction patches, warm start, regularization,
        velocity-only iterations, normal-only contacts, restitution, contact torsion,
        contact compliance, the propagation responses and the other friction modes keep
        the dense path.
        """
        if (
            self.articulated_contact_response != "immediate"
            or self.friction_mode != "current"
            or self._contact_torsion_enabled
            or self.contact_compliance
            or self._friction_anchors_enabled
            or self.pgs_warmstart
            or self._regularization_enabled
            or self.pgs_velocity_iterations > 0
            or np.isfinite(self.contact_friction_gap_threshold)
        ):
            return False
        restitution = model.shape_material_restitution
        return restitution is None or model.shape_count == 0 or not np.any(restitution.numpy() > 0.0)

    def _check_restitution_support(self) -> None:
        """Reject positive restitution in the split solve and the propagation responses, which have no rebound targets."""
        supports_rebound = self.pgs_mode == "matrix_free" and self.articulated_contact_response == "immediate"
        if supports_rebound or self.restitution_velocity_threshold >= np.finfo(np.float32).max:
            return
        restitution = self.model.shape_material_restitution
        if restitution is not None and self.model.shape_count and np.any(restitution.numpy() > 0.0):
            if self.pgs_mode == "split":
                raise NotImplementedError("Contact restitution requires pgs_mode='matrix_free'")
            raise NotImplementedError(
                f"Contact restitution is not supported with articulated_contact_response={self.articulated_contact_response!r}"
            )

    def _setup_sparse_mass_matrix(self, model: Model) -> None:
        """Select topology-derived sparse mass factors for branched articulations.

        The selection is automatic. It applies when every articulated (non-free-body)
        response group has one size and one joint topology, the factor of that topology has
        fewer nonzeros than a dense lower triangle (the tree branches), the articulation has at
        most 64 DOFs, joint velocity-limit rows are disabled and the contact options are ones
        the sparse kernels implement (:meth:`_sparse_supports_contact_options`). Free bodies keep
        their dense 6 x 6 factors. Otherwise the dense factors are kept.

        Models with mimic relationships or loop-closing joints, including disabled ones that
        can be re-enabled at runtime, keep the dense factors: their bilateral rows are built
        on the dense response storage, which the sparse factors replace by placeholders.
        """
        self._sparse_mass_matrix_size = None
        self._sparse_mass_matrix_plan = None
        if not self._kernel_overrides.get("sparse_mass_matrix", True):
            return
        if _model_has_bilateral_constraints(model):
            return
        # The sparse rows and sweep exist only in the interleaved matrix-free solve, without drive rows.
        if self.pgs_mode != "matrix_free" or self._has_drive_rows or self.pgs_schedule != "interleaved":
            return
        if self.enable_joint_velocity_limits or self.pgs_iterations <= 0 or not self.size_groups:
            return
        if not self._sparse_supports_contact_options(model):
            return
        plan = self._model_plan
        response_dofs = plan.response_dof_count
        free_rigid = plan.is_free_rigid != 0
        articulated = np.flatnonzero((response_dofs > 0) & ~free_rigid)
        sizes = np.unique(response_dofs[articulated])
        if len(sizes) != 1:
            return
        size = int(sizes[0])
        # A six-DOF articulated group would share the free bodies' dense factor group.
        if self._has_free_rigid_bodies and (size == 6 or 6 not in self._free_body_inertia_sizes):
            return
        if size > 64:
            return
        # Diagonal mass matrices need no factor at all.
        if size in self._diagonal_mass_sizes:
            return

        articulation_start = model.articulation_start.numpy()
        joint_end = self.articulation_joint_end.numpy()
        joint_ancestor = model.joint_ancestor.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_child = model.joint_child.numpy()
        # Every articulation of the group must share one local topology.
        reference = None
        for art in articulated:
            start, end = int(articulation_start[art]), int(joint_end[art])
            parents = joint_ancestor[start:end].astype(np.int32, copy=True)
            parents = np.where(parents >= start, parents - start, -1).astype(np.int32)
            counts = np.diff(joint_qd_start[start : end + 1]).astype(np.int32)
            if reference is None:
                reference = (parents, counts)
            elif not (np.array_equal(parents, reference[0]) and np.array_equal(counts, reference[1])):
                return
        try:
            factor_plan = _SparseMassMatrixPlan.build(*reference)
        except ValueError:
            return
        if factor_plan.dof_count != size or factor_plan.nonzero_count >= size * (size + 1) // 2:
            return

        # Endpoint support of each body: the sparse coordinates of its ancestor joints.
        joint_masks = factor_plan.joint_ancestor_mask
        body_masks = np.zeros(model.body_count, dtype=np.uint64)
        for art in articulated:
            first, last = int(articulation_start[art]), int(joint_end[art])
            children = joint_child[first:last]
            valid = children >= 0
            body_masks[children[valid]] = joint_masks[valid]
        shape_body = model.shape_body.numpy() if model.shape_count else np.zeros(0, dtype=np.int32)
        contact_masks = np.unique(body_masks[shape_body[shape_body >= 0]])
        max_mask_bits = max((int(mask).bit_count() for mask in joint_masks), default=0)
        max_support = max_mask_bits
        for first in contact_masks:
            for second in contact_masks:
                max_support = max(max_support, int(first | second).bit_count())
        articulated_counts = np.bincount(plan.articulation_world[articulated], minlength=self.world_count)
        if np.any(articulated_counts > 1):
            # Different articulations of a world have disjoint coordinate blocks.
            max_support = max(max_support, 2 * max_mask_bits)
        if self._has_free_rigid_bodies:
            max_support = max(max_support, max_mask_bits + 6)
        # No row has more distinct coordinates than the largest world's response vector.
        max_support = min(max_support, self.max_world_dofs)
        if not max_support:
            return
        # The kernels use static shared memory.
        shared_limit = min(48 * 1024, int(getattr(model.device, "max_shared_memory_per_block", 0)))
        mf_capacity = self.mf_max_constraints if self._has_free_rigid_bodies else 0
        # Two worlds share a solve block: three float row arrays, row types, velocity deltas.
        solve_shared_bytes = 8 * self.max_world_dofs + 26 * self.dense_max_constraints + 16 * mf_capacity
        if solve_shared_bytes > shared_limit:
            return
        shared_bytes_per_warp = 4 * (2 * factor_plan.nonzero_count + 6 * size)
        factor_warps_per_block = min(_SPARSE_FACTOR_WARPS_PER_BLOCK, shared_limit // shared_bytes_per_warp)
        if factor_warps_per_block < 1:
            return

        device = model.device
        group_count = self.n_arts_by_size[size]
        self._sparse_mass_matrix_size = size
        self._sparse_mass_matrix_plan = factor_plan
        self._sparse_mass_matrix_indices = factor_plan.to_device(device)
        self._sparse_body_dof_mask = wp.array(body_masks, dtype=wp.uint64, device=device)
        self._sparse_Linv = wp.zeros((group_count, factor_plan.nonzero_count), dtype=wp.float32, device=device)
        self._sparse_mass_matrix_status = wp.zeros(group_count, dtype=wp.int32, device=device)
        self._sparse_mass_matrix_scratch = wp.empty((group_count, size), dtype=wp.float32, device=device)
        self._sparse_drive_dof_K = wp.zeros(max(model.joint_dof_count, 1), dtype=wp.float32, device=device)
        shape = (self.world_count, self.dense_max_constraints, max_support)
        self._sparse_row_dof = wp.empty(shape, dtype=wp.int32, device=device)
        self._sparse_row_factor = wp.empty(shape, dtype=wp.float32, device=device)
        # A dense contact has at most one free-body endpoint; its physical response is
        # stored separately from the row's factor coordinates.
        free_shape = (*shape[:2], 6) if self._has_free_rigid_bodies else (1, 1, 1)
        self._sparse_row_free_response = wp.empty(free_shape, dtype=wp.float32, device=device)
        self._sparse_row_incident = wp.empty(shape[:2], dtype=wp.float32, device=device)
        self._sparse_factor_velocity_delta = wp.empty(
            (self.world_count, self.max_world_dofs), dtype=wp.float32, device=device
        )
        free_dof_mask = np.zeros((self.world_count, self.max_world_dofs), dtype=np.int32)
        offsets = self.articulation_world_dof_offset.numpy()
        for art in np.flatnonzero(free_rigid & (response_dofs > 0)):
            world = plan.articulation_world[art]
            free_dof_mask[world, offsets[art] : offsets[art] + response_dofs[art]] = 1
        self._sparse_world_free_dof_mask = wp.array(free_dof_mask, dtype=wp.int32, device=device)
        self._sparse_free_factor_dummy = wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
        self._crba_sparse_factor_kernel = _get_crba_sparse_factor_kernel(
            size, factor_plan.nonzero_count, warps_per_block=factor_warps_per_block
        )
        self._sparse_factor_warps_per_block = factor_warps_per_block
        self._sparse_contact_lanes = 8
        self._sparse_contact_response_kernel = _get_sparse_contact_response_kernel(
            size, lanes_per_contact=self._sparse_contact_lanes
        )
        self._pgs_solve_sparse_kernel = _get_pgs_solve_sparse_kernel(
            self.dense_max_constraints,
            self.max_world_dofs,
            max_support,
            mf_max_constraints=mf_capacity,
            free_row_dofs=6 if self._has_free_rigid_bodies else 0,
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
        # Tree joints of an articulation: [articulation_start, articulation_joint_end).
        self.articulation_joint_end = wp.array(
            self._model_plan.articulation_joint_end, dtype=wp.int32, device=model.device
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
            # Tree joints only: a loop joint must not replace its child's inbound tree joint.
            joint_end = int(self._model_plan.articulation_joint_end[articulation])

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
        self._has_mixed_contacts = False
        self._is_one_solve_art_per_world = False
        self._max_free_bodies_per_world = 0
        if not model.articulation_count or not model.joint_count:
            self._has_free_rigid_bodies = False
            self.is_free_rigid = None
            self.free_rigid_body_indices = None
            self._free_rigid_body_count = 0
            self._max_free_rigid_bodies_per_world = 0
            return
        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before free-body classification")

        is_free_rigid = self._model_plan.is_free_rigid
        self._free_rigid_body_count = len(self._model_plan.response_free_rigid_body_indices)
        self._has_free_rigid_bodies = self._free_rigid_body_count > 0

        # Per-world census of solved articulations for split mode: worlds holding both
        # free bodies and articulated bodies interleave the dense and free-body solves,
        # and the free-body solve keeps every free body of a world in its body table.
        world_count = self._model_plan.world_count
        articulation_world = self._model_plan.articulation_world
        solved = self._model_plan.response_dof_count > 0
        free = is_free_rigid != 0
        free_counts = np.bincount(articulation_world[solved & free], minlength=world_count)
        articulated_counts = np.bincount(articulation_world[solved & ~free], minlength=world_count)
        self._has_mixed_contacts = bool(np.any((free_counts > 0) & (articulated_counts > 0)))
        self._is_one_solve_art_per_world = bool(np.all(free_counts + articulated_counts == 1))
        self._max_free_bodies_per_world = int(np.max(free_counts)) if free_counts.size else 0

        self.is_free_rigid = wp.array(is_free_rigid, dtype=wp.int32, device=model.device)
        self.free_rigid_body_indices = wp.array(
            self._model_plan.response_free_rigid_body_indices, dtype=wp.int32, device=model.device
        )
        free_bodies = np.asarray(self._model_plan.response_free_rigid_body_indices, dtype=np.int64)
        if free_bodies.size:
            body_art = self.body_to_articulation.numpy()
            worlds = self._model_plan.articulation_world[body_art[free_bodies]]
            self._max_free_rigid_bodies_per_world = int(np.max(np.bincount(np.maximum(worlds, 0))))
        else:
            self._max_free_rigid_bodies_per_world = 0

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
            # Velocity of the position solve, kept while the velocity-only iterations refine v_out.
            self.v_out_snap = wp.zeros_like(model.joint_qd) if self.pgs_velocity_iterations else None
        else:
            self.mass_update_mask = None
            self.v_hat = None
            self.v_out = None
            self.qd_work = None
            self.v_out_snap = None

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
        self.Linv_by_size = {}
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            # Armature, added to the mass-matrix diagonal by the factorization kernels.
            self.R_by_size[size] = wp.zeros((n_arts, size), dtype=wp.float32, device=device)
            if size == self._sparse_mass_matrix_size:
                # Sparse factors replace the dense matrices, rows and responses of this group;
                # one-element stand-ins keep shared kernel arguments valid.
                stand_in = wp.zeros((1, 1, 1), dtype=wp.float32, device=device)
                self.H_by_size[size] = self.L_by_size[size] = stand_in
                self.J_by_size[size] = self.Y_by_size[size] = stand_in
                self.diag_by_size[size] = self._dummy_hinv_diag
                continue
            self.H_by_size[size] = wp.zeros((n_arts, size, size), dtype=wp.float32, device=device)
            self.L_by_size[size] = wp.zeros((n_arts, size, size), dtype=wp.float32, device=device)
            self.J_by_size[size] = wp.zeros((n_arts, max_constraints, size), dtype=wp.float32, device=device)
            self.Y_by_size[size] = wp.zeros((n_arts, max_constraints, size), dtype=wp.float32, device=device)
            self.diag_by_size[size] = (
                wp.zeros((n_arts, max_constraints), dtype=wp.float32, device=device)
                if size in self._hinv_jt_diag_sizes
                else self._dummy_hinv_diag
            )
            if self._sparse_mass_matrix_size is not None:
                # The sparse contact response reads the free bodies' inverse factors.
                self.Linv_by_size[size] = wp.zeros((n_arts, size, size), dtype=wp.float32, device=device)
            self.tau_by_size[size] = wp.zeros((n_arts, size, 1), dtype=wp.float32, device=device)
            self.qdd_by_size[size] = wp.zeros((n_arts, size, 1), dtype=wp.float32, device=device)
        # Double buffering ping-pongs H and J; sparse factors keep their own storage.
        self._double_buffered = bool(
            self.double_buffer and device.is_cuda and self.size_groups and self._sparse_mass_matrix_size is None
        )
        self._H_bufs = None
        self._J_bufs = None
        if self._double_buffered:
            self._H_bufs = (self.H_by_size, {size: wp.zeros_like(h) for size, h in self.H_by_size.items()})
            self._J_bufs = (self.J_by_size, {size: wp.zeros_like(j) for size, j in self.J_by_size.items()})

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
        # Rows reserved by each contact: 1 (normal only) or 3 (normal and friction).
        self.contact_slots_needed = wp.zeros((max_contacts,), dtype=wp.int32, device=device)
        # Friction patches own their history, independently of collision contact matching.
        # Row builders read a zero anchor displacement when patches are off.
        self._friction_patches = _FrictionPatchState(
            model,
            max_contacts,
            self._friction_anchors_enabled,
            wp.zeros(max_contacts, dtype=wp.vec2, device=device),
        )
        if self.pgs_warmstart:
            # Previous step's first row of each (sorted) contact per row family, and its normal.
            self._ws_prev_dense_slot = wp.full((max_contacts,), -1, dtype=wp.int32, device=device)
            self._ws_prev_mf_slot = wp.full((max_contacts,), -1, dtype=wp.int32, device=device)
            self._ws_prev_contact_normal = wp.zeros((max_contacts,), dtype=wp.vec3, device=device)
        # Previous solver step, and the contact buffer and generation it solved, on the
        # device so a captured graph replays them instead of baking the capture's values.
        # Generations count collision passes per buffer, so each buffer gets its own
        # nonzero stream id (0: no contacts).
        self._ws_history_dt = wp.zeros(1, dtype=float, device=device)
        self._ws_history_generation = wp.full(1, CONTACT_GENERATION_NONE, dtype=wp.int32, device=device)
        self._ws_history_stream = wp.zeros(1, dtype=wp.int32, device=device)
        self._ws_contact_streams: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._ws_last_stream = 0
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

        # Dense drive row of each DOF (drive_mode="physx_pgs"), or -1. Allocated with the
        # constraint rows; the pre-solve velocity scaling reads the current step's slots.
        if self._has_drive_rows:
            self.drive_slot = wp.full((model.joint_dof_count,), -1, dtype=wp.int32, device=device)
        else:
            self.drive_slot = wp.full((1,), -1, dtype=wp.int32, device=device)

    def _allocate_world_buffers(self, model):
        """Allocate the per-world dense row system (response, metadata and impulses)."""
        device = model.device
        shape = (self.world_count, self.dense_max_constraints)
        if self.pgs_mode == "split":
            # Split mode solves the assembled Delassus system and never reads world J/Y; the
            # one-element stand-ins satisfy the shared H^-1 J^T launches.
            self.C = wp.zeros((*shape, self.dense_max_constraints), dtype=wp.float32, device=device)
            self.J_world = wp.zeros((1, 1, 1), dtype=wp.float32, device=device)
            self.Y_world = wp.zeros((1, 1, 1), dtype=wp.float32, device=device)
        elif self._sparse_mass_matrix_size is not None:
            # Sparse rows store factor-coordinate responses instead of world J and Y.
            self.J_world = self.Y_world = wp.zeros((1, 1, 1), dtype=wp.float32, device=device)
        elif self._jy_world_aliased:
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
        self.row_restitution = wp.zeros(shape, dtype=wp.float32, device=device)
        # Regularization weight per row, read by the solve only when regularization is on.
        self.row_w = wp.ones(shape if self._regularization_enabled else (1, 1), dtype=wp.float32, device=device)
        # Right-hand side of the velocity-only iterations.
        self.rhs_unbiased = wp.zeros(shape if self.pgs_velocity_iterations else (1, 1), dtype=wp.float32, device=device)
        self.constraint_count = wp.zeros((self.world_count,), dtype=wp.int32, device=device)
        # Drive-row parameters and their per-step force-drive coefficients; (1, 1)
        # placeholders keep the kernel signatures fixed when there are no drive rows.
        drive_shape = shape if self._has_drive_rows else (1, 1)
        self.drive_stiffness = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_damping = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_geom_error = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_max_force = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_target_vel_bias = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_vel_multiplier = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_impulse_multiplier = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_max_impulse = wp.zeros(drive_shape, dtype=wp.float32, device=device)
        self.drive_vel_limit = wp.zeros(
            shape if self.fuse_joint_velocity_limits else (1, 1), dtype=wp.float32, device=device
        )
        if self.pgs_warmstart:
            self._ws_prev_impulses = wp.zeros(shape, dtype=wp.float32, device=device)
            self._ws_prev_row_type = wp.full(shape, -1, dtype=wp.int32, device=device)
            self._ws_prev_row_parent = wp.full(shape, -1, dtype=wp.int32, device=device)

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
        self.mf_row_restitution = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
        self.mf_row_w = wp.ones(
            (worlds, rows) if self._regularization_enabled else (1, 1), dtype=wp.float32, device=device
        )
        self.mf_rhs_unbiased = wp.zeros(
            (worlds, rows) if self.pgs_velocity_iterations else (1, 1), dtype=wp.float32, device=device
        )
        if self.pgs_warmstart:
            self._ws_prev_mf_impulses = wp.zeros((worlds, rows), dtype=wp.float32, device=device)
            self._ws_prev_mf_row_type = wp.full((worlds, rows), -1, dtype=wp.int32, device=device)
            self._ws_prev_mf_row_parent = wp.full((worlds, rows), -1, dtype=wp.int32, device=device)
            # Free bodies with rows in each world, for installing the seeded impulses.
            self.max_mf_bodies = max(self._max_free_rigid_bodies_per_world, 1)
            self.mf_body_list = wp.zeros((worlds, self.max_mf_bodies), dtype=wp.int32, device=device)
            self.mf_body_dof_start = wp.zeros((worlds, self.max_mf_bodies), dtype=wp.int32, device=device)
            self.mf_body_count = wp.zeros((worlds,), dtype=wp.int32, device=device)
            self.mf_local_body_a = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
            self.mf_local_body_b = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        if self.pgs_mode == "split":
            # Local body table of the native free-body solve, which keeps the velocities of a
            # world's free bodies in shared memory.
            bodies = max(self._max_free_bodies_per_world, 1)
            self.mf_body_list = wp.zeros((worlds, bodies), dtype=wp.int32, device=device)
            self.mf_body_dof_start = wp.zeros((worlds, bodies), dtype=wp.int32, device=device)
            self.mf_body_count = wp.zeros((worlds,), dtype=wp.int32, device=device)
            self.mf_local_body_a = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
            self.mf_local_body_b = wp.zeros((worlds, rows), dtype=wp.int32, device=device)
        if self.pgs_mode == "split" and self._has_mixed_contacts:
            # Free-body velocity change accumulated across the interleaved iterations, and
            # the per-iteration snapshot it is computed from.
            self.v_mf_accum = wp.zeros_like(model.joint_qd)
            self.v_out_snap = wp.zeros_like(model.joint_qd)
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
            # Sparse and diagonal mass matrices need no dense factorization, solve or response kernels.
            dense = size != self._sparse_mass_matrix_size and not self._execution_plan.use_diagonal_mass(size)
            self._cholesky_kernels_by_size[size] = (
                _get_cholesky_kernel(size, device_arch, self._tile_threads)
                if dense and self._execution_plan.use_tiled_cholesky(size)
                else None
            )
            self._triangular_solve_kernels_by_size[size] = (
                _get_triangular_solve_kernel(size, device_arch, self._tile_threads) if dense else None
            )
            hinv_jt_chunk_size = self._execution_plan.hinv_jt_chunk_size(size) if dense else None
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
                    self._tile_threads,
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
                    self._tile_threads,
                    constraint_chunk_size=hinv_jt_chunk_size,
                    write_world=self._hinv_jt_writes_world,
                    write_group=self._hinv_jt_tiled_writes_group,
                )

        if self.pgs_mode == "split":
            self._pack_mf_meta_kernel = None
            self._pgs_solve_mf_gs_kernel = None
            self._init_split_kernels(model)
            return

        self._pack_mf_meta_kernel = _get_pack_mf_meta_kernel(self.mf_meta_packed.shape[1] // 4, device_arch)
        self._pgs_solve_mf_gs_kernel = None
        self._init_propagation_kernels(model)
        if (
            self.world_count > 0
            and self.max_world_dofs > 0
            and self._sparse_mass_matrix_size is None
            and not self._propagation_fused
        ):
            mf_rows = self.mf_meta_packed.shape[1] // 4
            shared_metadata = _use_resident_mfgs_metadata(
                self.dense_max_constraints,
                mf_rows,
                self.max_world_dofs,
                int(getattr(model.device, "max_shared_memory_per_block", 0)),
                has_drive_rows=self._has_drive_rows,
                fuse_vel_limits=self.fuse_joint_velocity_limits,
            )
            self._pgs_solve_mf_gs_kernel = _get_pgs_solve_mf_gs_kernel(
                self.dense_max_constraints,
                mf_rows,
                self.max_world_dofs,
                device_arch,
                has_dense_velocity_limit_rows=self.enable_joint_velocity_limits,
                shared_metadata=shared_metadata,
                has_drive_rows=self._has_drive_rows,
                fuse_vel_limits=self.fuse_joint_velocity_limits,
                contact_torsion=self._contact_torsion_enabled,
                row_phases=self._propagation_active or self.pgs_schedule != "interleaved",
                friction_mode=self.friction_mode,
            )

    def _init_split_kernels(self, model):
        """Resolve the split-mode Delassus and Gauss-Seidel kernels for this solver shape.

        Every native kernel has a scalar Warp fallback; a kernel whose shared-memory
        working set does not fit the device falls back instead of failing to launch.
        """
        device_arch = model.device.arch
        is_cuda = model.device.is_cuda
        max_constraints = self.dense_max_constraints
        # One fused tile kernel per world computes H^-1 J^T and stores the whole Delassus
        # block; it requires every world to solve exactly one articulation.
        self._split_fused_response = bool(
            is_cuda
            and self.size_groups
            and self._is_one_solve_art_per_world
            and self.hinv_jt_kernel != "par_row"
            and self.delassus_kernel != "par_row_col"
            and all(self._execution_plan.use_fused_hinv_jt(size) for size in self.size_groups)
        )
        self._hinv_jt_fused_kernels_by_size = {
            size: _get_hinv_jt_fused_kernel(size, max_constraints, device_arch, self._tile_threads)
            for size in (self.size_groups if self._split_fused_response else ())
        }
        self._delassus_kernels_by_size = {}
        for size in self.size_groups:
            chunk = _select_delassus_chunk_size(size, max_constraints)
            self._delassus_kernels_by_size[size] = (
                _get_delassus_kernel(size, max_constraints, chunk, device_arch)
                if is_cuda and self.delassus_kernel != "par_row_col" and chunk is not None
                else None
            )
        self._pgs_solve_tiled_row_kernel = (
            _get_pgs_solve_tiled_row_kernel(max_constraints, device_arch)
            if is_cuda
            and self.pgs_kernel in ("auto", "tiled")
            and _estimate_tiled_row_shared_memory(max_constraints) <= _STATIC_SHARED_MEMORY_BYTES
            else None
        )
        # Contact-only kernels: whole contacts as 3x3 blocks, selected only through _kernel_overrides.
        self._pgs_solve_contact_kernel = None
        if is_cuda and self.pgs_kernel in ("tiled_contact", "streaming"):
            required = _estimate_contact_pgs_shared_memory(max_constraints, self.pgs_kernel, self._pgs_chunk_size)
            if required > _STATIC_SHARED_MEMORY_BYTES:
                raise ValueError(
                    f"pgs_kernel={self.pgs_kernel!r} needs {required} bytes of shared memory at "
                    f"dense_max_constraints={max_constraints}"
                )
        if is_cuda and self.pgs_kernel == "tiled_contact":
            self._pgs_solve_contact_kernel = _get_pgs_solve_tiled_contact_kernel(max_constraints, device_arch)
        elif is_cuda and self.pgs_kernel == "streaming":
            self._pgs_solve_contact_kernel = _get_pgs_solve_streaming_kernel(
                max_constraints, device_arch, pgs_chunk_size=self._pgs_chunk_size
            )
        mf_rows = self.mf_body_a.shape[1]
        mf_bodies = self.mf_body_dof_start.shape[1]
        self._pgs_solve_mf_kernel = (
            _get_pgs_solve_mf_kernel(mf_rows, mf_bodies, device_arch)
            if is_cuda
            and self._has_free_rigid_bodies
            and self.pgs_kernel != "loop"
            and _estimate_mf_solve_shared_memory(mf_rows, mf_bodies) <= _STATIC_SHARED_MEMORY_BYTES
            else None
        )

    def _pack_mf_meta(self, mf_rhs: wp.array) -> None:
        wp.launch_tiled(
            self._pack_mf_meta_kernel,
            dim=[self.world_count],
            inputs=[
                self.mf_constraint_count,
                self.mf_dof_a,
                self.mf_dof_b,
                self.mf_eff_mass_inv,
                mf_rhs,
                self.mf_row_type,
                self.mf_row_parent,
            ],
            outputs=[self.mf_meta_packed],
            block_dim=32,
            device=self.model.device,
        )

    def _mf_gs_inputs(self, rhs: wp.array) -> list:
        """Return the row inputs of the matrix-free Gauss-Seidel kernel."""
        return [
            self.constraint_count,
            self.world_dof_indices,
            rhs,
            self.diag,
            self.impulses,
            self.J_world,
            self.Y_world,
            self.row_type,
            self.row_parent,
            self.row_mu,
            self.drive_target_vel_bias,
            self.drive_vel_multiplier,
            self.drive_impulse_multiplier,
            self.drive_max_impulse,
            self.drive_vel_limit,
            self.mf_constraint_count,
            self.mf_contact_rows_end,
            self.mf_meta_packed,
            self.mf_impulses,
            self.mf_J_a,
            self.mf_J_b,
            self.mf_MiJt_a,
            self.mf_MiJt_b,
            self.mf_row_mu,
            self.row_w,
            self.mf_row_w,
            self._contact_torsion_group,
            self._contact_torsion_radius,
        ]

    def _launch_pgs_solve(
        self, rhs: wp.array, iterations: int, *, regularize: bool, freeze_drive_rows: bool = False
    ) -> None:
        """Run the fused matrix-free Gauss-Seidel sweep over the dense and free-body rows.

        Args:
            rhs: Dense right-hand side (position or velocity-only); the free-body one is
                packed into ``mf_meta_packed`` by :meth:`_pack_mf_meta`.
            iterations: Number of sweeps.
            regularize: Apply the per-row regularization weights.
            freeze_drive_rows: Skip PGS joint-drive rows (velocity-only iterations).
        """
        if self._sparse_mass_matrix_size is not None:
            # Sparse selection excludes every option that changes the right-hand side or sweep.
            self._launch_sparse_pgs_solve()
            return
        if iterations <= 0 or self._pgs_solve_mf_gs_kernel is None:
            return
        launch = functools.partial(
            self._launch_mf_gs_phase, rhs=rhs, regularize=regularize, freeze_drive_rows=freeze_drive_rows
        )
        if self.pgs_schedule == "contact_then_internal":
            launch(1, iterations)
            launch(2, iterations)
        elif self.pgs_schedule == "physx_grasp":
            has_internal_rows, has_velocity_limits = self._internal_row_phases()
            for _ in range(iterations):
                if has_internal_rows:
                    launch(3, 1)
                launch(4, 1)
                if has_velocity_limits:
                    launch(5, 1)
        else:
            launch(0, iterations)

    def _internal_row_phases(self) -> tuple[bool, bool]:
        """Return whether internal rows and velocity limits can exist (the phases with work)."""
        has_internal_rows = bool(
            self._has_drive_rows
            or (self.enable_joint_limits and self._joint_limit_sizes)
            or self._mimic_count
            or self._connect_count
        )
        has_velocity_limits = bool(
            (
                self.enable_joint_velocity_limits
                or self.fuse_joint_velocity_limits
                or (self._has_rigid_body_velocity_limits and self._has_free_rigid_bodies)
            )
            and not np.isinf(self.velocity_limit_activation_fraction)
        )
        return has_internal_rows, has_velocity_limits

    def _launch_sparse_pgs_solve(self) -> None:
        """Sweep the rows in factor coordinates, then decode the velocity change of every world."""
        model = self.model
        if self.world_count == 0 or self.max_world_dofs == 0:
            return
        wp.launch_tiled(
            self._pgs_solve_sparse_kernel,
            dim=[(self.world_count + 1) // 2],
            inputs=[
                self.world_count,
                self.constraint_count,
                self.rhs,
                self.diag,
                self.impulses,
                self._sparse_row_dof,
                self._sparse_row_factor,
                self._sparse_row_free_response,
                self._sparse_row_incident,
                self.row_type,
                self.row_parent,
                self.row_mu,
                self._sparse_world_free_dof_mask,
                self.world_dof_indices,
                self.v_hat,
                self.mf_constraint_count,
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
            outputs=[self._sparse_factor_velocity_delta],
            block_dim=64,
            device=model.device,
        )
        size = self._sparse_mass_matrix_size
        indices = self._sparse_mass_matrix_indices
        wp.launch(
            apply_sparse_factor_velocity,
            dim=(self.n_arts_by_size[size], size),
            inputs=[
                self.group_to_art[size],
                self.art_to_world,
                self.articulation_world_dof_offset,
                self.articulation_dof_start,
                indices.permutation,
                indices.lookup,
                self._sparse_Linv,
                self._sparse_factor_velocity_delta,
                self.v_hat,
            ],
            outputs=[self.v_out],
            device=model.device,
        )
        if self._has_free_rigid_bodies:
            wp.launch(
                apply_sparse_free_velocity,
                dim=(self.n_arts_by_size[6], 6),
                inputs=[
                    self.group_to_art[6],
                    self.art_to_world,
                    self.articulation_world_dof_offset,
                    self.articulation_dof_start,
                    self._sparse_factor_velocity_delta,
                    self.v_hat,
                ],
                outputs=[self.v_out],
                device=model.device,
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
        if self.contact_compliance:
            # Reject unsupported state before any stage can launch work.
            _contact_compliance.validate_step(self)
        if contacts is not None and contacts.rigid_contact_max > self._max_contacts_alloc:
            raise ValueError(
                "FeatherPGS contact capacity mismatch: received "
                f"{contacts.rigid_contact_max} slots, but solver scratch was allocated for "
                f"{self._max_contacts_alloc}. Set model.rigid_contact_max before constructing the solver."
            )
        if self.pgs_warmstart and contacts is not None and contacts.rigid_contact_match_index is None:
            raise ValueError(
                "pgs_warmstart=True matches contacts across steps and requires a Contacts buffer created "
                'with contact matching; create the CollisionPipeline with contact_matching="latest".'
            )
        if self._contact_torsion_enabled:
            validate_torsion_step(self)
            if self._device_torsion is not None:
                self._device_torsion.begin_step(state_in, state_out)
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
        if self.sleeping is not None:
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("Experimental sleeping requires a finite positive timestep")
            if any(
                a.size and a.ptr == b.ptr
                for a, b in (
                    (state_in.joint_q, state_out.joint_q),
                    (state_in.joint_qd, state_out.joint_qd),
                    (state_in.body_q, state_out.body_q),
                    (state_in.body_qd, state_out.body_qd),
                )
            ):
                raise ValueError("Experimental sleeping requires distinct input/output states")
            self.sleeping.begin(state_in, control, contacts)
        if self._double_buffered:
            self._select_double_buffer()
        state_aug = self._prepare_augmented_state(state_in)
        self._begin_dense_rows()

        # Stage 1: forward kinematics, inverse dynamics with implicit drives, and CRBA.
        stage3_qd = self._stage1_fk_id(state_in, state_aug, state_out)
        self._stage1_joint_tau(state_in, state_aug, state_out, control, dt)
        self._stage1_crba(state_aug)

        # Stage 2: factor the augmented mass matrix of every articulation group.
        for size in self._for_sizes():
            if size == self._sparse_mass_matrix_size:
                continue
            if self._execution_plan.use_diagonal_mass(size):
                self._stage2_factor_diagonal(size)
            elif self._execution_plan.use_tiled_cholesky(size):
                self._stage2_cholesky_tiled(size)
            else:
                self._stage2_cholesky_loop(size)
            if size in self.Linv_by_size:
                wp.launch(
                    invert_lower_factor_grouped,
                    dim=self.n_arts_by_size[size],
                    inputs=[self.group_to_art[size], self.mass_update_mask, size, self.L_by_size[size]],
                    outputs=[self.Linv_by_size[size]],
                    device=model.device,
                )

        # Stage 3: unconstrained acceleration and velocity prediction.
        state_aug.joint_qdd.zero_()
        for size in self._for_sizes():
            if size == self._sparse_mass_matrix_size:
                wp.launch(
                    solve_sparse_mass_matrix,
                    dim=self.n_arts_by_size[size] * 32,
                    inputs=[
                        self.group_to_art[size],
                        self.articulation_dof_start,
                        self._sparse_mass_matrix_indices,
                        self._sparse_Linv,
                        state_aug.joint_tau,
                        self._dynamics_art_active,
                    ],
                    outputs=[self._sparse_mass_matrix_scratch, state_aug.joint_qdd],
                    block_dim=128,
                    device=model.device,
                )
                continue
            if self._execution_plan.use_diagonal_mass(size):
                self._stage3_solve_diagonal(size, state_aug)
                continue
            use_tiled = self.trisolve_kernel == "tiled" or (
                self.trisolve_kernel == "auto" and size > self.small_dof_threshold
            )
            if use_tiled:
                self._stage3_trisolve_tiled(size, state_aug)
            else:
                self._stage3_trisolve_loop(size, state_aug)
        self._stage3_compute_v_hat(state_in, state_aug, dt, stage3_qd)

        # Stage 4: constraint rows, responses Y = H^-1 J^T, diagonals and right-hand sides.
        self._stage4_build_rows(state_in, state_aug, control, contacts, dt)
        if self._contact_torsion_enabled:
            prepare_torsion_rows(self, state_in, state_aug, contacts)
        has_contacts = contacts is not None and contacts.rigid_contact_max > 0
        if self.pgs_mode == "split":
            self._solve_split(state_aug, dt)
        else:
            self._solve_matrix_free(state_in, state_aug, contacts, dt)

        # Stage 7: convert the solved velocity to accelerations, integrate and publish.
        if self.pgs_velocity_iterations > 0:
            # Positions follow the position solve; the velocity-only iterations then refine
            # the velocity without the position bias.
            wp.copy(self.v_out_snap, self.v_out)
            self._compute_velocity_pass_rhs(dt)
            self._pack_mf_meta(self.mf_rhs_unbiased)
            self._launch_pgs_solve(
                self.rhs_unbiased,
                self.pgs_velocity_iterations,
                regularize=False,
                freeze_drive_rows=self.pgs_velocity_drive_mode == "freeze",
            )
            wp.copy(self.qd_work, self.v_out_snap)
            self._integrate(state_in, state_aug, state_out, dt, self.qd_work)
            wp.launch(
                update_qdd_from_velocity,
                dim=model.joint_dof_count,
                inputs=[state_in.joint_qd, self._kinematic_dof_mask, 1.0 / dt, self._dynamics_dof_active],
                outputs=[self.v_out, state_aug.joint_qdd],
                device=model.device,
            )
            if self._free_root_joint_count:
                # The integrator damped the position-solve velocity; damp the published one alike.
                wp.launch(
                    apply_free_root_angular_damping,
                    dim=self._free_root_joint_count,
                    inputs=[
                        self._free_root_joint_indices,
                        model.joint_qd_start,
                        self._kinematic_joint_mask,
                        model.joint_child,
                        self.rigid_body_angular_damping,
                        dt,
                    ],
                    outputs=[self.v_out],
                    device=model.device,
                )
            wp.copy(state_out.joint_qd, self.v_out)
        else:
            self._integrate(state_in, state_aug, state_out, dt, self.v_out)
        self._stage7_update_kinematics(state_out)
        if self.sleeping is not None:
            self.sleeping.finish(state_in, state_out, state_aug, dt)

        if self._friction_anchors_enabled:
            if has_contacts:
                for route, parents, mu, impulses in self._patch_row_arrays():
                    wp.launch(
                        finish_patch_impulses,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            contacts.rigid_contact_count,
                            self._friction_patches.current,
                            state_in.body_q,
                            state_out.body_q,
                            state_out.body_qd,
                            model.body_com,
                            self.contact_world,
                            self.contact_slot,
                            self.contact_path,
                            self.contact_slots_needed,
                            route,
                            parents,
                            mu,
                            impulses,
                            dt,
                        ],
                        device=model.device,
                    )
                self._friction_patches.store(state_in)
            else:
                self._friction_patches.previous.valid.zero_()
        if self.pgs_warmstart:
            self._snapshot_warmstart(contacts, dt)
        if self._double_buffered:
            self._clear_double_buffer()
        if self.row_watermark:
            self._accumulate_row_watermarks(contacts)
        if self._device_torsion is not None:
            self._device_torsion.end_step(state_out)
            if self._device_torsion.deferred_errors and not wp.get_stream(model.device).is_capturing:
                self.validate_contact_torsion()
        self._step += 1
        return state_out

    def _init_streams(self, model: Model) -> None:
        """Create the per-size-group streams and the double-buffer memset stream."""
        self._size_streams = (
            {size: wp.Stream(model.device) for size in self.size_groups}
            if self.use_parallel_streams and model.device.is_cuda and len(self.size_groups) > 1
            else {}
        )
        self._memset_stream = wp.Stream(model.device) if self._double_buffered else None
        self._buffer_index = 0
        # Per buffer set: the event that ends its clearing and the capture it was recorded in.
        self._memset_done_event = [None, None]
        self._memset_done_capture = [None, None]
        streams = [*self._size_streams.values(), self._memset_stream]
        # Kernels on these streams can still run when the solver is released; wait for them first.
        weakref.finalize(self, _synchronize_streams, [stream for stream in streams if stream is not None])

    def _for_sizes(self):
        """Yield every size group, each on its own stream with ``use_parallel_streams``.

        The groups' streams wait for the work enqueued so far, and the main stream waits for
        every group's stream once the loop ends.
        """
        if not self._size_streams:
            yield from self.size_groups
            return
        main_stream = wp.get_stream(self.model.device)
        start = main_stream.record_event()
        try:
            for size in self.size_groups:
                stream = self._size_streams[size]
                stream.wait_event(start)
                with wp.ScopedStream(stream, sync_enter=False):
                    yield size
        finally:
            for stream in self._size_streams.values():
                main_stream.wait_event(stream.record_event())

    def seed_double_buffer_events(self) -> None:
        """Seed the double-buffer waits of the first two steps of a CUDA graph capture.

        With ``double_buffer``, call this inside the capture before the first :meth:`step`;
        each step waits for the clearing of its buffer set, which the seeded events stand in
        for. Without double buffering it does nothing.
        """
        if self._memset_stream is None:
            return
        main_stream = wp.get_stream(self.model.device)
        self._memset_done_event = [main_stream.record_event(), main_stream.record_event()]
        self._memset_done_capture = [self._current_capture()] * 2

    def _current_capture(self):
        """Return the graph the current stream captures into, or ``None`` outside a capture."""
        stream = wp.get_stream(self.model.device)
        return self.model.device.captures.get(stream) if stream.is_capturing else None

    def _select_double_buffer(self) -> None:
        """Make the current buffer set active and wait until it was cleared."""
        index = self._buffer_index
        self.H_by_size = self._H_bufs[index]
        self.J_by_size = self._J_bufs[index]
        if self._jy_world_aliased:
            self.J_world = self.J_by_size[self.size_groups[0]]
        event = self._memset_done_event[index]
        capture = self._current_capture()
        if event is not None and self._memset_done_capture[index] is not capture:
            if capture is not None:
                raise RuntimeError(
                    "double_buffer requires seed_double_buffer_events() inside the capture before the first step"
                )
            # The capture's clearing ran in its graph launches, ordered before this step.
            event = None
        if event is not None:
            wp.get_stream(self.model.device).wait_event(event)

    def _clear_double_buffer(self) -> None:
        """Clear the buffer set of this step on the memset stream and switch to the other set."""
        index = self._buffer_index
        with wp.ScopedStream(self._memset_stream):
            for size in self.size_groups:
                # A step without a global mass refresh wrote no mass matrices.
                if self._mass_update_global_flag:
                    self._H_bufs[index][size].zero_()
                self._J_bufs[index][size].zero_()
        self._memset_done_event[index] = self._memset_stream.record_event()
        self._memset_done_capture[index] = self._current_capture()
        self._buffer_index = 1 - index

    def _accumulate_row_watermarks(self, contacts: Contacts | None) -> None:
        """Fold this step's row counts into the high-water marks."""
        families = [
            (self.constraint_count, self.slot_counter, self._row_dropped_dense, self.dense_max_constraints),
            (self.mf_constraint_count, self.mf_slot_counter, self._row_dropped_mf, self.mf_max_constraints),
        ]
        if self._propagation_active:
            families.append(
                (
                    self.propagation_constraint_count,
                    self.propagation_slot_counter,
                    self._row_dropped_propagation,
                    self.propagation_max_constraints,
                )
            )
        for family, (count, requested, dropped, capacity) in enumerate(families):
            wp.launch(
                accumulate_row_watermarks,
                dim=self.world_count,
                inputs=[count, requested, dropped, capacity, family * ROW_WATERMARK_FAMILY_STRIDE],
                outputs=[self._row_watermarks],
                device=self.model.device,
            )
        if contacts is not None:
            wp.launch(
                accumulate_contact_watermark,
                dim=1,
                inputs=[contacts.rigid_contact_count],
                outputs=[self._row_watermarks],
                device=self.model.device,
            )

    def _integrate(self, state_in: State, state_aug: State, state_out: State, dt: float, velocity: wp.array) -> None:
        """Integrate the joint state from a solved generalized velocity (which is updated in place)."""
        model = self.model
        wp.launch(
            update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, self._kinematic_dof_mask, 1.0 / dt, self._dynamics_dof_active],
            outputs=[velocity, state_aug.joint_qdd],
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
                    self._dynamics_joint_active,
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
                self._dynamics_joint_active,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd],
            device=model.device,
        )

    def _compute_velocity_pass_rhs(self, dt: float) -> None:
        """Build the right-hand sides of the velocity-only iterations from the position solve."""
        model = self.model
        wp.launch(
            compute_world_contact_velocity_bias,
            dim=self.world_count * self.dense_max_constraints,
            inputs=[
                self.constraint_count,
                self.dense_max_constraints,
                self.world_dof_count,
                self.phi,
                self.row_type,
                self.target_velocity,
                self.row_restitution,
                self.v_out_snap,
                self.v_hat,
                self.world_dof_indices,
                self.J_world,
                dt,
                self.restitution_velocity_threshold,
            ],
            outputs=[self.rhs_unbiased],
            device=model.device,
        )
        if self._contact_torsion_enabled:
            prepare_torsion_velocity_pass(self, dt)
        if self._has_free_rigid_bodies:
            wp.launch(
                compute_mf_velocity_rhs,
                dim=self.world_count * self.mf_max_constraints,
                inputs=[
                    self.mf_constraint_count,
                    self.mf_dof_a,
                    self.mf_dof_b,
                    self.mf_J_a,
                    self.mf_J_b,
                    self.world_dof_indices,
                    self.mf_phi,
                    self.mf_row_type,
                    self.mf_target_velocity,
                    self.mf_row_restitution,
                    int(self._has_prescribed_response),
                    dt,
                    self.v_out_snap,
                    self.v_hat,
                    self.restitution_velocity_threshold,
                    self.mf_max_constraints,
                ],
                outputs=[self.mf_rhs_unbiased],
                device=model.device,
            )

    def _apply_mf_warmstart_velocity(self) -> None:
        """Install the velocity of the seeded free-body impulses into ``v_out``."""
        model = self.model
        wp.launch(
            build_mf_body_map,
            dim=self.world_count,
            inputs=[
                self.mf_constraint_count,
                self.mf_body_a,
                self.mf_body_b,
                self.body_to_articulation,
                self.articulation_dof_start,
                self.max_mf_bodies,
            ],
            outputs=[
                self.mf_body_list,
                self.mf_body_dof_start,
                self.mf_body_count,
                self.mf_local_body_a,
                self.mf_local_body_b,
            ],
            device=model.device,
        )
        wp.launch(
            apply_mf_warmstart_impulses,
            dim=self.world_count * self.max_mf_bodies * 6,
            inputs=[
                self.mf_constraint_count,
                self.mf_body_count,
                self.mf_body_dof_start,
                self.mf_local_body_a,
                self.mf_local_body_b,
                self.mf_MiJt_a,
                self.mf_MiJt_b,
                self.mf_impulses,
                self.max_mf_bodies,
            ],
            outputs=[self.v_out],
            device=model.device,
        )

    def _snapshot_warmstart(self, contacts: Contacts | None, dt: float) -> None:
        """Keep this step's impulses, row layout and contact slots for the next step's gather."""
        model = self.model
        has_contacts = int(contacts is not None)
        families = [
            (
                self.impulses,
                self.row_type,
                self.row_parent,
                self._ws_prev_impulses,
                self._ws_prev_row_type,
                self._ws_prev_row_parent,
            )
        ]
        if self._has_free_rigid_bodies:
            families.append(
                (
                    self.mf_impulses,
                    self.mf_row_type,
                    self.mf_row_parent,
                    self._ws_prev_mf_impulses,
                    self._ws_prev_mf_row_type,
                    self._ws_prev_mf_row_parent,
                )
            )
        for impulses, row_type, row_parent, prev_impulses, prev_type, prev_parent in families:
            wp.launch(
                snapshot_row_warmstart,
                dim=impulses.shape,
                inputs=[impulses, row_type, row_parent, has_contacts],
                outputs=[prev_impulses, prev_type, prev_parent],
                device=model.device,
            )
        if contacts is not None:
            wp.launch(
                snapshot_contact_warmstart,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    self.contact_path,
                    self.contact_slot,
                    contacts.rigid_contact_normal,
                ],
                outputs=[self._ws_prev_dense_slot, self._ws_prev_mf_slot, self._ws_prev_contact_normal],
                device=model.device,
            )
        wp.launch(
            snapshot_step_warmstart,
            dim=1,
            inputs=[
                float(dt),
                contacts.contact_generation if contacts is not None else self._ws_history_generation,
                self._contact_stream(contacts),
            ],
            outputs=[self._ws_history_dt, self._ws_history_generation, self._ws_history_stream],
            device=model.device,
        )

    def _contact_stream(self, contacts: Contacts | None) -> int:
        """Return the warm-start stream id of a contact buffer; 0 for no contacts."""
        if contacts is None:
            return 0
        stream = self._ws_contact_streams.get(contacts)
        if stream is None:
            # Ids are never reused, so a replaced buffer cannot alias saved history.
            self._ws_last_stream += 1
            stream = self._ws_last_stream
            self._ws_contact_streams[contacts] = stream
        return stream

    def _solve_matrix_free(self, state_in: State, state_aug: State, contacts: Contacts | None, dt: float) -> None:
        """Build the matrix-free responses and right-hand sides and run the fused solve into ``v_out``."""
        model = self.model
        if self._sparse_mass_matrix_size is None:
            for size in self._for_sizes():
                if self._execution_plan.use_diagonal_mass(size):
                    self._stage4_hinv_jt_diagonal(size)
                elif self._execution_plan.use_tiled_hinv_jt(size):
                    self._stage4_hinv_jt_tiled(size)
                else:
                    self._stage4_hinv_jt_par_row(size)
            if self._preelim_active:
                self._stage4_bilateral_preelim()
            self._stage4_compute_matrix_free_diag()
        # Sparse row builders write the response diagonals directly.
        wp.launch(
            finalize_world_diag_cfm,
            dim=self.world_count,
            inputs=[self.constraint_count, self.row_type, self.pgs_cfm],
            outputs=[self.diag],
            device=model.device,
        )
        if self._has_drive_rows:
            wp.launch(
                compute_physx_pgs_drive_desc,
                dim=self.world_count,
                inputs=[
                    self.constraint_count,
                    self.row_type,
                    self.diag,
                    self.target_velocity,
                    self.drive_stiffness,
                    self.drive_damping,
                    self.drive_geom_error,
                    self.drive_max_force,
                    dt,
                ],
                outputs=[
                    self.drive_target_vel_bias,
                    self.drive_vel_multiplier,
                    self.drive_impulse_multiplier,
                    self.drive_max_impulse,
                ],
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
                self.friction_anchor_beta,
                self.contact_speculative_scale,
                self._contact_w,
                dt,
            ],
            outputs=[self.rhs, self.row_w],
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
            inputs=[self.constraint_count, self.dense_max_constraints, int(self.pgs_warmstart)],
            outputs=[self.impulses],
            device=model.device,
        )
        has_contacts = contacts is not None and contacts.rigid_contact_max > 0
        if self.pgs_warmstart and has_contacts:
            self._gather_warmstart(contacts, route=0, dt=dt)
        if self._friction_anchors_enabled and has_contacts and self.pgs_warmstart and self.pgs_warmstart_decay > 0.0:
            # Carried anchors replace the identity-matched tangent impulses of their rows.
            for route, parents, mu, impulses in self._patch_row_arrays():
                wp.launch(
                    seed_patch_impulses,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        contacts.rigid_contact_count,
                        self._friction_patches.current,
                        self._friction_patches.previous,
                        state_in.body_q,
                        self.contact_world,
                        self.contact_slot,
                        self.contact_path,
                        self.contact_slots_needed,
                        route,
                        parents,
                        mu,
                        impulses,
                        self.pgs_warmstart_decay,
                        dt,
                        self._ws_history_dt,
                    ],
                    device=model.device,
                )
        if self._sparse_mass_matrix_size is None and not self._jy_world_aliased and not self._hinv_jt_writes_world:
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
        # Impacts replace the position bias by their rebound target, from the unconstrained velocity.
        wp.launch(
            apply_world_contact_restitution,
            dim=self.world_count * self.dense_max_constraints,
            inputs=[
                self.constraint_count,
                self.dense_max_constraints,
                self.world_dof_count,
                self.phi,
                self.row_type,
                self.target_velocity,
                self.row_restitution,
                self.v_hat,
                self.world_dof_indices,
                self.J_world,
                dt,
                self.restitution_velocity_threshold,
                int(self._regularization_enabled),
            ],
            outputs=[self.rhs, self.row_w],
            device=model.device,
        )
        wp.copy(self.v_out, self.v_hat)
        if self.pgs_warmstart:
            # The solve updates the velocity from impulse deltas, so the seeded impulses'
            # velocity must be installed before the first sweep.
            if self.max_world_dofs > 0:
                wp.launch(
                    apply_world_impulses_to_velocity,
                    dim=self.world_count * self.max_world_dofs,
                    inputs=[
                        self.constraint_count,
                        self.world_dof_indices,
                        self.max_world_dofs,
                        self.Y_world,
                        self.impulses,
                    ],
                    outputs=[self.v_out],
                    device=model.device,
                )
            if self._has_free_rigid_bodies:
                self._apply_mf_warmstart_velocity()
        if self._preelim_active:
            # After the warm-start installs, so a carried contact impulse cannot reopen the projection.
            self._stage5_project_bilateral_velocity()
        if self.contact_compliance:
            _contact_compliance.solve(self, iterations=self.pgs_iterations)
        else:
            self._pack_mf_meta(self.mf_rhs)
            if self._propagation_active:
                self._propagation_setup(state_in, state_aug, dt)
                self._launch_propagation_solve()
            else:
                self._launch_pgs_solve(self.rhs, self.pgs_iterations, regularize=self._regularization_enabled)

    def _solve_split(self, state_aug: State, dt: float) -> None:
        """Assemble and solve the dense Delassus systems, then the free-body rows, into ``v_out``.

        Worlds with both articulated and free-body rows alternate one dense sweep and one
        free-body sweep per iteration: the dense right-hand side absorbs ``J dv`` of each
        free-body sweep, and ``v_out`` carries the accumulated free-body velocity change.
        """
        model = self.model
        if self._split_fused_response:
            for size in self.size_groups:
                wp.launch_tiled(
                    self._hinv_jt_fused_kernels_by_size[size],
                    dim=[self.n_arts_by_size[size]],
                    inputs=[
                        self.L_by_size[size],
                        self.J_by_size[size],
                        self.group_to_art[size],
                        self.art_to_world,
                        self.constraint_count,
                        self.pgs_cfm,
                    ],
                    outputs=[self.C, self.diag, self.Y_by_size[size]],
                    block_dim=self._tile_threads,
                    device=model.device,
                )
        else:
            self.C.zero_()
            self.diag.zero_()
            for size in self._for_sizes():
                if self._execution_plan.use_tiled_hinv_jt(size):
                    self._stage4_hinv_jt_tiled(size)
                else:
                    self._stage4_hinv_jt_par_row(size)
            for size in self.size_groups:
                self._stage4_delassus(size)
            wp.launch(
                finalize_world_diag_cfm,
                dim=self.world_count,
                inputs=[self.constraint_count, self.row_type, self.pgs_cfm],
                outputs=[self.diag],
                device=model.device,
            )
        # rhs = bias + J v_hat; the dense solve sees velocity only through C lambda.
        wp.launch(
            compute_world_contact_bias,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.phi,
                self.row_type,
                self.target_velocity,
                self.pgs_beta,
                self.friction_anchor_beta,
                self.contact_speculative_scale,
                self._contact_w,
                dt,
            ],
            outputs=[self.rhs, self.row_w],
            device=model.device,
        )
        for size in self.size_groups:
            self._accumulate_dense_rhs(size, self.v_hat)
        wp.launch(
            prepare_world_impulses,
            dim=self.world_count,
            inputs=[self.constraint_count, self.dense_max_constraints, 0],
            outputs=[self.impulses],
            device=model.device,
        )

        if not self._has_free_rigid_bodies:
            self._dense_pgs_solve(self.pgs_iterations)
            self._apply_dense_impulses()
            return

        self._mf_pgs_setup(state_aug, dt)
        if self._pgs_solve_mf_kernel is not None:
            wp.launch(
                build_mf_body_map,
                dim=self.world_count,
                inputs=[
                    self.mf_constraint_count,
                    self.mf_body_a,
                    self.mf_body_b,
                    self.body_to_articulation,
                    self.articulation_dof_start,
                    self.mf_body_dof_start.shape[1],
                ],
                outputs=[
                    self.mf_body_list,
                    self.mf_body_dof_start,
                    self.mf_body_count,
                    self.mf_local_body_a,
                    self.mf_local_body_b,
                ],
                device=model.device,
            )
        if not self._has_mixed_contacts:
            self._dense_pgs_solve(self.pgs_iterations)
            self._apply_dense_impulses()
            self._mf_pgs_solve(self.pgs_iterations)
            return

        self.v_mf_accum.zero_()
        wp.copy(self.v_out, self.v_hat)
        for _ in range(self.pgs_iterations):
            self._dense_pgs_solve(1)
            # v_out = v_hat + Y lambda + accumulated free-body change.
            self._apply_dense_impulses()
            wp.launch(
                vector_add_inplace,
                dim=self.v_out.shape[0],
                inputs=[self.v_out, self.v_mf_accum],
                device=model.device,
            )
            wp.copy(self.v_out_snap, self.v_out)
            self._mf_pgs_solve(1)
            # v_mf_accum += dv; v_out_snap = dv, so the dense rhs absorbs J dv.
            wp.launch(
                compute_delta_and_accumulate,
                dim=self.v_out.shape[0],
                inputs=[self.v_out, self.v_out_snap, self.v_mf_accum],
                device=model.device,
            )
            for size in self.size_groups:
                self._accumulate_dense_rhs(size, self.v_out_snap)

    def _stage4_delassus(self, size: int) -> None:
        """Accumulate one size group's ``C += J Y^T`` and its diagonal."""
        n_arts = self.n_arts_by_size[size]
        kernel = self._delassus_kernels_by_size[size]
        if kernel is not None:
            wp.launch_tiled(
                kernel,
                dim=[n_arts],
                inputs=[
                    self.J_by_size[size],
                    self.Y_by_size[size],
                    self.group_to_art[size],
                    self.art_to_world,
                    self.constraint_count,
                    n_arts,
                ],
                outputs=[self.C, self.diag],
                block_dim=128,
                device=self.model.device,
            )
            return
        wp.launch(
            delassus_par_row_col,
            dim=n_arts * self.dense_max_constraints * self.dense_max_constraints,
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
            outputs=[self.C, self.diag],
            device=self.model.device,
        )

    def _accumulate_dense_rhs(self, size: int, velocity: wp.array) -> None:
        """Add one size group's ``J velocity`` to the dense right-hand side."""
        wp.launch(
            rhs_accum_world_par_art,
            dim=self.n_arts_by_size[size],
            inputs=[
                self.constraint_count,
                self.art_to_world,
                self.articulation_dof_start,
                velocity,
                self.group_to_art[size],
                self.J_by_size[size],
                size,
            ],
            outputs=[self.rhs],
            device=self.model.device,
        )

    def _dense_pgs_solve(self, iterations: int) -> None:
        """Run ``iterations`` Gauss-Seidel sweeps over each world's dense Delassus system."""
        if iterations <= 0:
            return
        inputs = [
            self.constraint_count,
            self.diag,
            self.C,
            self.rhs,
            iterations,
            self.pgs_omega,
            self.row_type,
            self.row_parent,
            self.row_mu,
        ]
        if self._pgs_solve_contact_kernel is not None:
            wp.launch_tiled(
                self._pgs_solve_contact_kernel,
                dim=[self.world_count],
                inputs=[
                    self.constraint_count,
                    self.C,
                    self.rhs,
                    self.impulses,
                    iterations,
                    self.pgs_omega,
                    self.row_mu,
                ],
                block_dim=32,
                device=self.model.device,
            )
            return
        if self._pgs_solve_tiled_row_kernel is not None:
            wp.launch_tiled(
                self._pgs_solve_tiled_row_kernel,
                dim=[self.world_count],
                inputs=inputs,
                outputs=[self.impulses],
                block_dim=32,
                device=self.model.device,
            )
            return
        wp.launch(
            pgs_solve_loop,
            dim=self.world_count,
            inputs=inputs,
            outputs=[self.impulses],
            device=self.model.device,
        )

    def _apply_dense_impulses(self) -> None:
        """Write ``v_out = v_hat + Y lambda`` from the dense impulses."""
        wp.copy(self.v_out, self.v_hat)
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            wp.launch(
                apply_impulses_world_par_dof,
                dim=n_arts * size,
                inputs=[
                    self.group_to_art[size],
                    self.art_to_world,
                    self.articulation_dof_start,
                    size,
                    n_arts,
                    self.constraint_count,
                    self.Y_by_size[size],
                    self.impulses,
                    self.v_hat,
                ],
                outputs=[self.v_out],
                device=self.model.device,
            )

    def _mf_pgs_solve(self, iterations: int) -> None:
        """Run ``iterations`` Gauss-Seidel sweeps over the free-body rows, updating ``v_out``."""
        if iterations <= 0:
            return
        if self._pgs_solve_mf_kernel is not None:
            wp.launch_tiled(
                self._pgs_solve_mf_kernel,
                dim=[self.world_count],
                inputs=[
                    self.mf_constraint_count,
                    self.mf_body_count,
                    self.mf_body_dof_start,
                    self.mf_local_body_a,
                    self.mf_local_body_b,
                    self.mf_J_a,
                    self.mf_J_b,
                    self.mf_MiJt_a,
                    self.mf_MiJt_b,
                    self.mf_eff_mass_inv,
                    self.mf_rhs,
                    self.mf_row_type,
                    self.mf_row_parent,
                    self.mf_row_mu,
                    iterations,
                    self.pgs_omega,
                ],
                outputs=[self.mf_impulses, self.v_out],
                block_dim=32,
                device=self.model.device,
            )
            return
        wp.launch(
            pgs_solve_mf_loop,
            dim=self.world_count,
            inputs=[
                self.mf_constraint_count,
                self.mf_body_a,
                self.mf_body_b,
                self.mf_MiJt_a,
                self.mf_MiJt_b,
                self.mf_J_a,
                self.mf_J_b,
                self.mf_eff_mass_inv,
                self.mf_rhs,
                self.mf_row_type,
                self.mf_row_parent,
                self.mf_row_mu,
                self.body_to_articulation,
                self.articulation_dof_start,
                iterations,
                self.pgs_omega,
            ],
            outputs=[self.mf_impulses, self.v_out],
            device=self.model.device,
        )

    def constraint_row_watermarks(self) -> dict[str, int]:
        """Return the row high-water marks accumulated since construction (``row_watermark`` only).

        Per family (``dense``, ``mf`` for free-body rows, ``propagation``): the largest retained
        and requested row count of any world and step, the largest number of dropped contact
        rows, the largest excess over the capacity, and the number of overflowing world-steps;
        and ``contact_high_water``, the largest contact count. Reads the device on the host, so
        call it outside CUDA graph capture. Every value is ``0`` when ``row_watermark`` is off.
        """
        values = (
            self._row_watermarks.numpy().tolist()
            if self._row_watermarks is not None
            else [0] * (ROW_WATERMARK_CONTACT_SLOT + 1)
        )
        result = {}
        for family, name in enumerate(("dense", "mf", "propagation")):
            base = family * ROW_WATERMARK_FAMILY_STRIDE
            result[f"{name}_high_water"] = int(values[base])
            result[f"{name}_raw_high_water"] = int(values[base + 1])
            result[f"{name}_dropped_contact_rows_high_water"] = int(values[base + 2])
            result[f"{name}_overflow_excess_high_water"] = int(values[base + 3])
            result[f"{name}_overflow_world_steps"] = int(values[base + 4])
        result["contact_high_water"] = int(values[ROW_WATERMARK_CONTACT_SLOT])
        return result

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
                self.contact_slots_needed,
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
                    self.drive_slot,
                    int(self.fuse_joint_velocity_limits),
                    self._dynamics_art_active,
                ],
                outputs=[self.qd_work],
                device=model.device,
            )
            stage3_qd = self.qd_work
        else:
            stage3_qd = state_in.joint_qd

        refresh_composite = (self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update
        fk_inputs = [
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
        ]
        fk_outputs = [
            state_in.body_q,
            state_aug.body_q_com,
            self.articulation_origin,
            state_aug.joint_S_s,
            state_aug.body_I_s,
            self._body_inertia_terms,
            state_aug.body_v_s,
            state_aug.body_f_s,
            state_aug.body_a_s,
        ]
        if self._tree_plan is not None:
            self._launch_tree_fk("id", [*fk_inputs, self._dynamics_art_active], fk_outputs)
        else:
            wp.launch(
                eval_rigid_fk_id,
                dim=model.articulation_count,
                inputs=[*fk_inputs, self._dynamics_art_active],
                outputs=fk_outputs,
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
                    self._dynamics_body_active,
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
        ``aug_row_K`` and added to the mass-matrix diagonal by :meth:`_stage1_crba`. With
        ``drive_mode="physx_pgs"`` drives contribute nothing here; their rows are solved
        with the constraints.
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
        if self._tree_plan is not None:
            self._launch_tree_tau(tau_inputs, state_aug)
            if self.drive_mode == "augmented" and self.articulation_max_dofs > 0:
                wp.launch(
                    eval_augmented_drives,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        self.articulation_H_rows,
                        model.joint_type,
                        model.joint_qd_start,
                        model.joint_q_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        model.joint_target_q_start,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                        self.articulation_max_dofs,
                        dt,
                        self._dynamics_art_active,
                    ],
                    outputs=[self.aug_row_counts, self.aug_row_dof_index, self.aug_row_K, state_aug.joint_tau],
                    block_dim=self._serial_kernel_block_dim,
                    device=model.device,
                )
        elif self.drive_mode == "augmented" and self.articulation_max_dofs > 0:
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
                    self._dynamics_art_active,
                ],
                outputs=[
                    state_aug.body_ft_s,
                    self.aug_row_counts,
                    self.aug_row_dof_index,
                    self.aug_row_K,
                    state_aug.joint_tau,
                ],
                block_dim=self._serial_kernel_block_dim,
                device=model.device,
            )
        else:
            wp.launch(
                eval_rigid_tau,
                dim=model.articulation_count,
                inputs=[*common_inputs, *tau_inputs, self._dynamics_art_active],
                outputs=[state_aug.body_ft_s, state_aug.joint_tau],
                block_dim=self._serial_kernel_block_dim,
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

    def _launch_tree_fk(self, mode: str, inputs: list, outputs: list) -> None:
        """Run one cooperative tree traversal per traversal width."""
        for group in self._tree_plan.groups:
            per_warp = 32 // group.lanes
            wp.launch(
                _get_tree_fk_kernel(group.lanes, mode),
                dim=((group.articulation_count + per_warp - 1) // per_warp, 32),
                inputs=[
                    group.articulation_count,
                    group.max_levels,
                    group.articulations,
                    group.level_offsets,
                    group.segment_offsets,
                    group.segment_joints,
                    *inputs,
                ],
                outputs=outputs,
                block_dim=32,
                device=self.model.device,
            )

    def _launch_tree_tau(self, tau_inputs: list, state_aug: State) -> None:
        """Run the inverse-dynamics backward pass over independent branches."""
        # tau_inputs follows the eval_rigid_tau signature after the joint range arrays:
        # joint_type, joint_parent, joint_child, joint_articulation, ...
        joint_type, _joint_parent, *rest = tau_inputs
        for group, net_wrench in zip(self._tree_plan.groups, self._tree_net_wrenches, strict=True):
            per_warp = 32 // group.lanes
            wp.launch(
                _get_tree_tau_kernel(group.lanes),
                dim=((group.articulation_count + per_warp - 1) // per_warp, 32),
                inputs=[
                    group.articulation_count,
                    group.max_levels,
                    group.articulations,
                    group.level_offsets,
                    group.segment_offsets,
                    group.segment_joints,
                    group.child_offsets,
                    group.child_segments,
                    joint_type,
                    *rest,
                    self._dynamics_art_active,
                ],
                outputs=[state_aug.body_ft_s, state_aug.joint_tau, net_wrench],
                block_dim=32,
                device=self.model.device,
            )

    def _stage1_crba(self, state_aug: State):
        """Build the joint-space mass matrix of articulations due for a refresh and add drive terms."""
        model = self.model
        global_flag = 1 if ((self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0
        self._mass_update_global_flag = global_flag
        wp.launch(
            build_mass_update_mask,
            dim=model.articulation_count,
            inputs=[global_flag, self._mass_update_requested, self._dynamics_art_active],
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
        if global_flag and self._composite_inertia_warp_kernel is not None:
            # A global refresh reduces every articulation that reads composite inertias,
            # one warp per articulation.
            wp.launch_tiled(
                self._composite_inertia_warp_kernel,
                dim=[
                    (self._composite_articulation_count + _COMPOSITE_INERTIA_WARPS_PER_BLOCK - 1)
                    // _COMPOSITE_INERTIA_WARPS_PER_BLOCK
                ],
                inputs=[
                    self._composite_articulation_count,
                    self._composite_articulations,
                    model.articulation_start,
                    self.articulation_joint_end,
                    model.joint_ancestor,
                    model.joint_child,
                    state_aug.body_I_s,
                    self._dynamics_art_active,
                ],
                outputs=[self.body_I_c],
                block_dim=32 * _COMPOSITE_INERTIA_WARPS_PER_BLOCK,
                device=model.device,
            )
        elif not global_flag or self._composite_articulation_count:
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
            if size == self._sparse_mass_matrix_size:
                self._stage1_sparse_factor(state_aug, size)
                continue
            if global_flag and not self._double_buffered:
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
            if self.drive_mode != "augmented":
                continue
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

    def _stage1_sparse_factor(self, state_aug: State, size: int) -> None:
        """Assemble and factor the sparse mass matrices of articulations due for a refresh."""
        model = self.model
        n_arts = self.n_arts_by_size[size]
        if self.articulation_max_dofs > 0:
            wp.launch(
                scatter_augmented_drive_dof_K,
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
                outputs=[self._sparse_drive_dof_K],
                device=model.device,
            )
        wp.launch(
            self._crba_sparse_factor_kernel,
            dim=n_arts * 32,
            inputs=[
                self.group_to_art[size],
                self.mass_update_mask,
                model.articulation_start,
                self.articulation_dof_start,
                model.joint_child,
                state_aug.joint_S_s,
                self.body_I_c,
                self.R_by_size[size],
                self._sparse_drive_dof_K,
                self._sparse_mass_matrix_indices,
            ],
            outputs=[self._sparse_Linv, self._sparse_mass_matrix_status],
            block_dim=32 * self._sparse_factor_warps_per_block,
            device=model.device,
        )

    def _stage2_cholesky_tiled(self, size: int):
        wp.launch_tiled(
            self._cholesky_kernels_by_size[size],
            dim=[self.n_arts_by_size[size]],
            inputs=[self.H_by_size[size], self.R_by_size[size], self.group_to_art[size], self.mass_update_mask],
            outputs=[self.L_by_size[size]],
            block_dim=self._tile_threads,
            device=self.model.device,
        )

    def _stage2_factor_diagonal(self, size: int) -> None:
        """Factor a group whose mass matrix is structurally diagonal."""
        wp.launch(
            factor_diagonal_mass,
            dim=self.n_arts_by_size[size] * size,
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
                self._dynamics_art_active,
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
                self._dynamics_art_active,
            ],
            outputs=[self.qdd_by_size[size]],
            block_dim=self._tile_threads,
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
                self._dynamics_art_active,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_solve_diagonal(self, size: int, state_aug: State) -> None:
        """Solve the unconstrained accelerations of a diagonal-mass group."""
        wp.launch(
            solve_diagonal_mass,
            dim=self.n_arts_by_size[size] * size,
            inputs=[
                self.L_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
                state_aug.joint_tau,
                self._dynamics_art_active,
            ],
            outputs=[state_aug.joint_qdd],
            device=self.model.device,
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
                self._dynamics_art_active,
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
            inputs=[stage3_qd, self._kinematic_dof_mask, dt, self._dynamics_dof_active],
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
                    self._dynamics_joint_active,
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
                    self._dynamics_joint_active,
                ],
                outputs=[self.v_hat],
                device=model.device,
            )

    def _patch_row_arrays(self):
        """Yield ``(route, row_parent, row_mu, impulses)`` of each row family that can hold contact rows."""
        yield 0, self.row_parent, self.row_mu, self.impulses
        if self._has_free_rigid_bodies:
            yield 1, self.mf_row_parent, self.mf_row_mu, self.mf_impulses

    def _gather_warmstart(self, contacts: Contacts, route: int, dt: float) -> None:
        """Seed one row family's contact impulses from the previous step."""
        if route == 0:
            arrays = (
                self._ws_prev_dense_slot,
                self._ws_prev_impulses,
                self._ws_prev_row_type,
                self._ws_prev_row_parent,
                self.constraint_count,
                self.row_type,
                self.row_parent,
                self.row_mu,
                self.dense_max_constraints,
                self.impulses,
            )
        else:
            arrays = (
                self._ws_prev_mf_slot,
                self._ws_prev_mf_impulses,
                self._ws_prev_mf_row_type,
                self._ws_prev_mf_row_parent,
                self.mf_constraint_count,
                self.mf_row_type,
                self.mf_row_parent,
                self.mf_row_mu,
                self.mf_max_constraints,
                self.mf_impulses,
            )
        prev_slot, prev_impulses, prev_type, prev_parent, count, row_type, row_parent, row_mu, capacity, impulses = (
            arrays
        )
        wp.launch(
            gather_contact_warmstart,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                route,
                self.contact_path,
                self.contact_slot,
                self.contact_world,
                contacts.rigid_contact_match_index,
                contacts.rigid_contact_match_generation,
                prev_slot,
                prev_impulses,
                prev_type,
                prev_parent,
                count,
                row_type,
                row_parent,
                contacts.rigid_contact_normal,
                self._ws_prev_contact_normal,
                row_mu,
                self.pgs_warmstart_decay,
                dt,
                self._ws_history_dt,
                contacts.contact_generation,
                self._contact_stream(contacts),
                self._ws_history_generation,
                self._ws_history_stream,
                capacity,
            ],
            outputs=[impulses],
            device=self.model.device,
        )

    def _begin_dense_rows(self) -> None:
        """Start the step's dense row allocation and reserve the drive rows first.

        Drive slots are reserved before the pre-solve velocity scaling, which skips the
        DOFs whose velocity limit the fused clamp enforces.
        """
        model = self.model
        wp.launch(
            _clear_dense_row_state,
            dim=self.world_count,
            inputs=[self.slot_counter, self.dense_contact_world_flag, self._row_dropped_all],
            device=model.device,
        )
        self._propagation_clear_rows()
        self._dense_first_rejected_slot.fill_(_ROW_SLOT_UNBOUNDED)
        if self._has_drive_rows:
            wp.launch(
                allocate_physx_drive_slots,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_start,
                    self.articulation_dof_start,
                    self.articulation_H_rows,
                    model.joint_type,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    self.art_to_world,
                    self.dense_max_constraints,
                ],
                outputs=[self.drive_slot, self.slot_counter],
                device=model.device,
            )

    def _stage4_build_rows(
        self, state_in: State, state_aug: State, control: Control, contacts: Contacts | None, dt: float
    ):
        """Allocate and fill the drive, bilateral, joint-limit, velocity-limit, contact and friction rows.

        Dense rows (rows touching an articulated body) are laid out per world as
        ``[joint drives][mimic][connect][joint limits][joint velocity limits][contacts and
        friction]``, where drive rows exist only with ``drive_mode="physx_pgs"`` and were
        reserved by :meth:`_begin_dense_rows`; free-body rows hold ``[contacts and
        friction][free-body velocity limits]``. Rows past a capacity are dropped, counted and
        latched into :attr:`constraint_overflow`.
        """
        if self.contact_compliance:
            _contact_compliance.start_step(self, contacts, dt)
        model = self.model
        max_constraints = self.dense_max_constraints
        mf_active = self._has_free_rigid_bodies
        if self._has_prescribed_response:
            self.mf_target_velocity.zero_()

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

        sparse_size = self._sparse_mass_matrix_size
        sleeping = self.sleeping
        # Sleeping islands get no joint-limit or contact rows (their limit coordinates and
        # contact response are masked) when they skip constraints.
        limit_q_index = self._joint_limit_q_index
        contact_response = self.body_has_response_dofs
        frozen_bodies = None
        if sleeping is not None:
            if sleeping.limit_q_index is not None:
                limit_q_index = sleeping.limit_q_index
            contact_response = sleeping.contact_response_mask
            frozen_bodies = sleeping.patch_frozen_bodies
        # Rows are rebuilt every step; clear the grouped Jacobians once before any family writes.
        # Double-buffered Jacobians were cleared on the memset stream after their last use.
        if sparse_size is None and not self._double_buffered:
            for size in self.size_groups:
                self.J_by_size[size].zero_()

        if self._has_drive_rows:
            for size in self.size_groups:
                wp.launch(
                    populate_physx_drive_J_for_size,
                    dim=self.n_arts_by_size[size],
                    inputs=[
                        model.articulation_start,
                        self.articulation_dof_start,
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_effort_limit,
                        state_in.joint_q,
                        control.joint_target_q,
                        model.joint_target_q_start,
                        control.joint_target_qd,
                        model.joint_velocity_limit,
                        int(self.fuse_joint_velocity_limits),
                        self.art_to_world,
                        self.drive_slot,
                        self.group_to_art[size],
                    ],
                    outputs=[
                        self.J_by_size[size],
                        self.row_type,
                        self.row_parent,
                        self.row_mu,
                        self.phi,
                        self.target_velocity,
                        self.drive_stiffness,
                        self.drive_damping,
                        self.drive_geom_error,
                        self.drive_max_force,
                        self.drive_vel_limit,
                    ],
                    device=model.device,
                )

        self._stage4_build_bilateral_rows(state_in, state_aug)

        # Disabled joint limits create no rows (and use no capacity), as in the reference solver.
        limit_sizes = self._joint_limit_sizes if self.enable_joint_limits else frozenset()
        for size in limit_sizes:
            n_arts = self.n_arts_by_size[size]
            if size == sparse_size:
                indices = self._sparse_mass_matrix_indices
                wp.launch(
                    build_sparse_joint_limit_rows,
                    dim=n_arts * 32,
                    inputs=[
                        self.group_to_art[size],
                        self.art_to_world,
                        self.articulation_world_dof_offset,
                        self.articulation_dof_start,
                        limit_q_index,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        state_in.joint_q,
                        self.joint_limit_activation_gap,
                        indices.ancestor_mask,
                        indices.inverse_permutation,
                        indices.lookup,
                        self._sparse_Linv,
                        self.v_hat,
                    ],
                    outputs=[
                        self.slot_counter,
                        self.row_type,
                        self.row_parent,
                        self.row_mu,
                        self.phi,
                        self.target_velocity,
                        self._sparse_row_dof,
                        self._sparse_row_factor,
                        self._sparse_row_incident,
                        self.diag,
                    ],
                    block_dim=128,
                    device=model.device,
                )
                continue
            if size not in self._joint_limit_warp_kernels:
                wp.launch(
                    build_joint_limit_rows,
                    dim=n_arts,
                    inputs=[
                        self.articulation_dof_start,
                        self.art_to_world,
                        self.group_to_art[size],
                        self._joint_limit_q_index,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        state_in.joint_q,
                        self.joint_limit_activation_gap,
                        max_constraints,
                        size,
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
                    device=model.device,
                )
                continue
            wp.launch_tiled(
                self._joint_limit_warp_kernels[size],
                dim=[(n_arts + _JOINT_LIMIT_WARPS_PER_BLOCK - 1) // _JOINT_LIMIT_WARPS_PER_BLOCK],
                inputs=[
                    n_arts,
                    self.articulation_dof_start,
                    self.art_to_world,
                    self.group_to_art[size],
                    limit_q_index,
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
                    self.drive_slot,
                    int(self.fuse_joint_velocity_limits),
                    self.art_to_world,
                    max_constraints,
                    self._constraint_art_active,
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
            if self._friction_anchors_enabled:
                self._friction_patches.build(
                    model,
                    state_in,
                    contacts,
                    # The solver's buffer: notified edits reach eager and captured steps alike.
                    shape_material_mu=self.shape_material_mu,
                    body_to_articulation=self.body_to_articulation,
                    is_free_rigid=is_free_rigid,
                    contact_gap_gate=self.contact_gap_gate,
                    same_articulation_gap_gate=self.same_articulation_contact_gap_gate,
                    articulation_pair_gap_gate=self.articulation_pair_contact_gap_gate,
                    friction_gap=self.contact_friction_gap_threshold,
                    friction_articulation_pairs_only=self.contact_friction_articulation_pairs_only,
                    frozen_bodies=frozen_bodies,
                )
            wp.launch(
                allocate_world_contact_slots,
                dim=contact_build_threads,
                inputs=[
                    contacts.rigid_contact_count,
                    contact_build_threads,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    state_in.body_q,
                    model.shape_body,
                    self.body_to_articulation,
                    self.art_to_world,
                    self.articulation_response_dof_count,
                    model.body_flags,
                    contact_response,
                    is_free_rigid,
                    int(mf_active),
                    int(self._propagation_active),
                    int(self.propagation_same_articulation_rows),
                    int(self._propagation_active and self.articulated_contact_response == "propagation"),
                    max_constraints,
                    self.mf_max_constraints,
                    self.propagation_max_constraints,
                    self._articulation_model_world,
                    self.contact_gap_gate,
                    self.same_articulation_contact_gap_gate,
                    self.articulation_pair_contact_gap_gate,
                    self.contact_friction_gap_threshold,
                    int(self.contact_friction_articulation_pairs_only),
                    self._friction_patches.view,
                ],
                outputs=[
                    self.contact_world,
                    self.contact_slot,
                    self.contact_art_a,
                    self.contact_art_b,
                    self.contact_slots_needed,
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
                    self.contact_slots_needed,
                    model.shape_body,
                    state_in.body_q,
                    state_aug.body_v_s,
                    self._prescribed_articulation,
                    self.articulation_origin,
                    self.shape_material_mu,
                    self.shape_material_restitution,
                    self._friction_patches.view,
                    *self._contact_anchor_args,
                ],
                outputs=[
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self.phi,
                    self.target_velocity,
                    self.row_restitution,
                ],
                device=model.device,
            )
            if sparse_size is not None:
                # Factor-coordinate responses of the branched group and physical responses of
                # free-body endpoints, one lane group per contact; no Jacobian is stored.
                lanes = self._sparse_contact_lanes
                workers = min(contact_build_threads, _CONTACT_JACOBIAN_WORKER_CAP * 32 // lanes)
                wp.launch(
                    self._sparse_contact_response_kernel,
                    dim=workers * lanes,
                    inputs=[
                        contacts.rigid_contact_count,
                        workers,
                        *contact_geometry[2:],
                        self.contact_world,
                        self.contact_slot,
                        self.contact_art_a,
                        self.contact_art_b,
                        self.contact_path,
                        self.art_group_idx,
                        self.articulation_response_dof_count,
                        self.is_free_rigid,
                        self.articulation_world_dof_offset,
                        self.articulation_dof_start,
                        self.articulation_origin,
                        self._sparse_body_dof_mask,
                        state_aug.joint_S_s,
                        model.shape_body,
                        state_in.body_q,
                        *self._contact_anchor_args,
                        self._sparse_mass_matrix_indices.permutation,
                        self._sparse_mass_matrix_indices.row_offsets,
                        self._sparse_Linv,
                        self.Linv_by_size[6] if self._has_free_rigid_bodies else self._sparse_free_factor_dummy,
                        self.v_hat,
                    ],
                    outputs=[
                        self._sparse_row_dof,
                        self._sparse_row_factor,
                        self._sparse_row_free_response,
                        self._sparse_row_incident,
                        self.diag,
                    ],
                    block_dim=128,
                    device=model.device,
                )
            elif self._compact_contact_jacobian:
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
                            self.contact_slots_needed,
                            size,
                            self.articulation_response_dof_count,
                            self.art_group_idx,
                            self.articulation_dof_start,
                            self.articulation_origin,
                            self.body_response_dof_mask,
                            state_aug.joint_S_s,
                            model.shape_body,
                            state_in.body_q,
                            self._friction_patches.view,
                            *self._contact_anchor_args,
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
                            self.contact_slots_needed,
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
                            self._friction_patches.view,
                            *self._contact_anchor_args,
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
                        self.contact_slots_needed,
                        self.shape_material_restitution,
                        self._friction_patches.view,
                        self.friction_anchor_beta,
                        *self._contact_anchor_args,
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
                        self.mf_row_restitution,
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
                if self.pgs_warmstart:
                    self._gather_warmstart(contacts, route=1, dt=dt)

            if self._friction_anchors_enabled:
                # Chain each region's normal rows and share the friction budget among the
                # anchors that received rows.
                for route, parents, mu, _impulses in self._patch_row_arrays():
                    wp.launch(
                        link_patch_rows,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            contacts.rigid_contact_count,
                            self._friction_patches.view,
                            self.contact_world,
                            self.contact_slot,
                            self.contact_path,
                            self.contact_slots_needed,
                            route,
                            parents,
                            mu,
                        ],
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
                    self._constraint_art_active,
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
            block_dim=self._tile_threads,
            device=self.model.device,
        )

    def _stage4_hinv_jt_diagonal(self, size: int) -> None:
        """Compute the row responses of a diagonal-mass group."""
        n_arts = self.n_arts_by_size[size]
        world_dof_offset = self.articulation_world_dof_offset if self._hinv_jt_writes_world else self.group_to_art[size]
        wp.launch(
            hinv_jt_diagonal,
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
                self.mf_row_restitution,
                int(self._has_prescribed_response),
                self.body_to_articulation,
                self.articulation_dof_start,
                self.v_hat,
                self.rigid_body_max_depenetration_velocity,
                self.pgs_cfm,
                self.pgs_beta,
                self._contact_w,
                dt,
                self.contact_speculative_scale,
                self.restitution_velocity_threshold,
                self.mf_max_constraints,
            ],
            outputs=[self.mf_eff_mass_inv, self.mf_MiJt_a, self.mf_MiJt_b, self.mf_rhs, self.mf_row_w],
            device=model.device,
        )

    @property
    def _sleep_skips_dynamics(self) -> bool:
        """Whether sleeping islands skip their articulated dynamics."""
        sleeping = self.sleeping
        return sleeping is not None and sleeping.skip_constraints and sleeping.skip_dynamics

    def _stage7_update_kinematics(self, state_out: State) -> None:
        """Publish the maximal-coordinate body state of the integrated joint state."""
        model = self.model
        if self._tree_plan is None:
            # Sleeping islands that skipped their dynamics keep their frozen bodies.
            mask = self._dynamics_art_mask if self._sleep_skips_dynamics else None
            eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out, mask=mask)
            return
        self._launch_tree_fk(
            "public",
            [
                model.joint_articulation,
                state_out.joint_q,
                state_out.joint_qd,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                model.body_flags,
                int(BodyFlags.ALL),
                self._dynamics_art_active,
            ],
            [state_out.body_q, state_out.body_qd],
        )

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
                *self._contact_anchor_args,
                self._friction_patches.view,
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

    def _launch_mf_gs_phase(
        self,
        row_phase: int,
        iterations: int = 1,
        *,
        rhs: wp.array | None = None,
        regularize: bool = False,
        freeze_drive_rows: bool = False,
    ) -> None:
        """Run matrix-free sweeps restricted to the row families of ``row_phase`` (0 for all)."""
        wp.launch_tiled(
            self._pgs_solve_mf_gs_kernel,
            dim=[self.world_count],
            inputs=[
                *self._mf_gs_inputs(self.rhs if rhs is None else rhs),
                int(iterations),
                self.pgs_omega,
                int(regularize),
                int(freeze_drive_rows),
                int(row_phase),
            ],
            outputs=[self.v_out],
            block_dim=32,
            device=self.model.device,
        )

    def _launch_propagation_solve(self) -> None:
        """Run the projected Gauss-Seidel iterations of the propagation response.

        Each iteration solves the dense joint-limit rows, the dense and free-body contact
        rows, the propagation rows followed by the tree propagation of their impulses, and
        last the velocity-limit rows. With ``pgs_schedule="contact_then_internal"`` the
        iterations solve the contact and propagation rows only, and the joint-limit and
        velocity-limit rows follow in their own iterations.
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

        contact_then_internal = self.pgs_schedule == "contact_then_internal"
        refresh_forced = (
            contact_then_internal or self._propagation_has_limit_rows or self._propagation_has_velocity_limit_rows
        )
        wpb = _PROPAGATION_WORLDS_PER_BLOCK
        for _ in range(self.pgs_iterations):
            if self._propagation_has_limit_rows and not contact_then_internal:
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
            if self._propagation_has_velocity_limit_rows and not contact_then_internal:
                self._launch_mf_gs_phase(5)
        if contact_then_internal and (self._propagation_has_limit_rows or self._propagation_has_velocity_limit_rows):
            self._launch_mf_gs_phase(2, self.pgs_iterations)


@cache
def _get_composite_inertia_warp_kernel(device_arch: str, warps_per_block: int) -> "wp.Kernel":
    """Build a one-warp-per-articulation composite-inertia reduction.

    The lanes of a warp split the 36 elements of each spatial inertia; the joints are
    reduced child to parent in the order of :func:`compute_composite_inertia`, so the
    results are identical.
    """
    _ = device_arch
    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const int candidate = block * {warps_per_block} + (threadIdx.x >> 5);
    if (candidate >= composite_articulation_count) return;
    const int articulation = composite_articulations.data[candidate];
    // One warp per articulation, so skipping a sleeping articulation is warp-uniform.
    if (articulation_active.data[articulation] == 0) return;
    const int start = articulation_start.data[articulation];
    const int end = articulation_joint_end.data[articulation];
    for (int joint = start; joint < end; ++joint) {{
        const int body = joint_child.data[joint];
        const float* src = reinterpret_cast<const float*>(&body_I_s.data[body]);
        float* dst = reinterpret_cast<float*>(&body_I_c.data[body]);
        for (int element = lane; element < 36; element += 32) dst[element] = src[element];
    }}
    __syncwarp();

    for (int joint = end - 1; joint >= start; --joint) {{
        const int parent_joint = joint_ancestor.data[joint];
        if (parent_joint >= start) {{
            const int body = joint_child.data[joint];
            const int parent_body = joint_child.data[parent_joint];
            const float* src = reinterpret_cast<const float*>(&body_I_c.data[body]);
            float* dst = reinterpret_cast<float*>(&body_I_c.data[parent_body]);
            for (int element = lane; element < 36; element += 32) dst[element] += src[element];
        }}
        __syncwarp();
    }}
#endif
"""

    @wp.func_native(snippet)
    def composite_inertia_warp_native(
        block: int,
        composite_articulation_count: int,
        composite_articulations: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_ancestor: wp.array[int],
        joint_child: wp.array[int],
        body_I_s: wp.array[wp.spatial_matrix],
        articulation_active: wp.array[int],
        body_I_c: wp.array[wp.spatial_matrix],
    ): ...

    def composite_inertia_warp_template(
        composite_articulation_count: int,
        composite_articulations: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_ancestor: wp.array[int],
        joint_child: wp.array[int],
        body_I_s: wp.array[wp.spatial_matrix],
        articulation_active: wp.array[int],
        body_I_c: wp.array[wp.spatial_matrix],
    ):
        block, _lane = wp.tid()
        composite_inertia_warp_native(
            block,
            composite_articulation_count,
            composite_articulations,
            articulation_start,
            articulation_joint_end,
            joint_ancestor,
            joint_child,
            body_I_s,
            articulation_active,
            body_I_c,
        )

    name = f"compute_composite_inertia_warp{warps_per_block}"
    composite_inertia_warp_template.__name__ = name
    composite_inertia_warp_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(composite_inertia_warp_template)

    # ------------------------------------------------------------------


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
        articulation_active: wp.array[int],
        qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    ):
        idx = wp.tid()
        # One articulation per block, so the early exit is uniform across the tile.
        if articulation_active[group_to_art[idx]] == 0:
            return
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
    has_drive_rows: bool = False,
    fuse_vel_limits: bool = False,
    contact_torsion: bool = False,
    row_phases: bool = False,
    friction_mode: str = "current",
) -> "wp.Kernel":
    """Build the fused matrix-free projected Gauss-Seidel kernel for one solver shape.

    One warp (32 threads) solves one world. Every iteration sweeps, in order:

    1. the dense rows (joint drives, mimic and connect rows, joint limits, contacts and
       friction of articulated bodies): a warp-parallel ``J v`` over the world's ``D``
       response DOFs and a ``Y`` update of the world velocity, software-pipelined one row
       ahead;
    2. the free-body contact and friction rows: lanes 0-5 handle body A and lanes 6-11
       body B;
    3. with ``contact_torsion``, the contact torsion rows, which the dense pass skips
       (``contact_torsion.torque_sweep_source``);
    4. the dense joint velocity-limit rows, the fused velocity clamp of driven DOFs and
       the free-body velocity-limit rows, so velocity limits have the last word in each
       iteration.

    A drive row's impulse is not projected: each visit replaces it with the force-drive
    update ``lambda * (1 - x) + (J v) * (-x a) + bias`` precomputed per row by
    :func:`compute_physx_pgs_drive_desc`, clamped to the row's maximum impulse.

    Friction rows follow their normal row and solve the two tangent impulses together on
    the Coulomb disk of the current normal impulse (``FRICTION_PAIR_CUDA``). A
    world stops early after an exactly stationary sweep: neither its impulses nor its
    velocity changed, so every later sweep would repeat the same operations; the
    iteration count stays an upper bound and no convergence tolerance is introduced.

    The world velocity and impulses stay in shared memory. With ``shared_metadata`` the
    per-row metadata does as well; larger shapes stream it from global memory to keep
    occupancy.

    With ``row_phases`` the launch argument ``row_phase`` restricts a sweep to some row
    families, for the PGS schedules and so a propagation solve can interleave its own rows:
    ``1`` and ``4`` the contact rows (dense and free-body contact, friction and torsion
    rows), ``2`` the internal rows (drive, mimic, connect and joint-limit rows) and the
    velocity limits, ``3`` the internal rows without the velocity limits, ``5`` the
    velocity limits (velocity-limit rows and the fused clamp). ``0`` (the only value
    without ``row_phases``) sweeps every family.

    Args:
        max_constraints: Dense row capacity ``M_D`` per world.
        mf_max_constraints: Free-body row capacity ``M_MF`` per world.
        max_world_dofs: Response DOFs ``D`` per world.
        device_arch: CUDA architecture; part of the cache key.
        has_dense_velocity_limit_rows: Emit the dense velocity-limit pass.
        shared_metadata: Keep the dense row metadata in shared memory.
        has_drive_rows: Emit the drive-row update (``drive_mode="physx_pgs"``).
        fuse_vel_limits: Emit the end-of-iteration velocity clamp of driven DOFs
            (``fuse_joint_velocity_limits``); requires ``has_drive_rows``.
        contact_torsion: Emit the contact torsion pass.
        row_phases: Honor the ``row_phase`` launch argument.
        friction_mode: Contact update of the friction rows (see ``SolverFeatherPGS``):
            ``"current"`` solves the two tangents on the friction disk of the current
            normal impulse; the other modes solve each contact's normal and tangent rows
            together, once per sweep at its first friction row (dense contacts keep the
            ``"current"`` update under ``"coulomb_newton"``).
    """
    if fuse_vel_limits and not has_drive_rows:
        raise ValueError("fuse_vel_limits requires has_drive_rows")
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

    # Stateless clamp of driven DOFs (PhysX PxClamp on the joint velocity): an overshoot
    # gets the impulse that returns the DOF exactly to its limit. It runs where the
    # velocity-limit rows run, after the contact rows, so the limit keeps the last word
    # in an under-converged sweep; the drive impulse itself is left unchanged.
    fused_velocity_clamp_pass = (
        f"""
        for (int i = 0; i < m_dense; i++) {{
            if ((s_meta_dense[i] & {type_mask}) != {int(PGS_CONSTRAINT_TYPE_JOINT_TARGET)}) continue;
            float denom = s_diag_dense[i];
            if (denom <= 0.0f) continue;
            float qdot_max = s_drive_vel_limit_dense[i];
            int row_base = jy_world_base + i * {D};
            float my_sum = 0.0f;
            for (int d = lane; d < {D}; d += 32) my_sum += J_world.data[row_base + d] * s_v[d];
            float jv = warp_sum(my_sum);
            if (fabsf(jv) > qdot_max) {{
                float delta_impulse = (fminf(fmaxf(jv, -qdot_max), qdot_max) - jv) / denom;
                if (delta_impulse != 0.0f) {{
                    iteration_changed = 1;
                    for (int d = lane; d < {D}; d += 32) s_v[d] += Y_world.data[row_base + d] * delta_impulse;
                }}
            }}
            __syncwarp();
        }}"""
        if fuse_vel_limits
        else ""
    )
    drive_shared_declarations = (
        f"""
    __shared__ float s_drive_target_dense[{M_D}];
    __shared__ float s_drive_vel_mul_dense[{M_D}];
    __shared__ float s_drive_imp_mul_dense[{M_D}];
    __shared__ float s_drive_max_imp_dense[{M_D}];"""
        if has_drive_rows
        else ""
    )
    drive_loads = (
        """
        s_drive_target_dense[i] = world_drive_target_vel_bias.data[off_dense + i];
        s_drive_vel_mul_dense[i] = world_drive_vel_multiplier.data[off_dense + i];
        s_drive_imp_mul_dense[i] = world_drive_impulse_multiplier.data[off_dense + i];
        s_drive_max_imp_dense[i] = world_drive_max_impulse.data[off_dense + i];"""
        if has_drive_rows
        else ""
    )
    if fuse_vel_limits:
        drive_shared_declarations += f"""
    __shared__ float s_drive_vel_limit_dense[{M_D}];"""
        drive_loads += """
        s_drive_vel_limit_dense[i] = world_drive_vel_limit.data[off_dense + i];"""
    drive_update = (
        f"""
            if (row_type == {int(PGS_CONSTRAINT_TYPE_JOINT_TARGET)}) {{
                // PhysX force-drive update; not relaxed by omega.
                float max_impulse = s_drive_max_imp_dense[i];
                new_impulse = old_impulse * s_drive_imp_mul_dense[i] + jv * s_drive_vel_mul_dense[i]
                    + s_drive_target_dense[i];
                new_impulse = fminf(fmaxf(new_impulse, -max_impulse), max_impulse);
            }} else"""
        if has_drive_rows
        else ""
    )
    torsion_skip = f"if (row_type == {int(PGS_CONSTRAINT_TYPE_TORSION)}) continue;" if contact_torsion else ""
    torsion_sweep = torque_sweep_source(D) if contact_torsion else ""

    if row_phases:
        phase_bounds = """    if (row_phase == 1 || row_phase == 4) {
        internal_rows = 0;
        velocity_limit_pass = 0;
    } else if (row_phase == 2 || row_phase == 3) {
        contact_rows = 0;
        mf_main_end = 0;
        if (row_phase == 3) velocity_limit_pass = 0;
    } else if (row_phase == 5) {
        dense_main_end = 0;
        mf_main_end = 0;
        contact_rows = 0;
    }"""
        contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
        friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
        dense_phase_filter = (
            f"            if ((row_type == {contact_type} || row_type == {friction_type}) ? contact_rows == 0 "
            ": internal_rows == 0) continue;"
        )
    else:
        phase_bounds = ""
        dense_phase_filter = ""
    if torsion_sweep:
        torsion_sweep = f"        if (contact_rows) {{\n{torsion_sweep}\n        }}"

    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    dense_friction_update = f"""                // The first tangent row solves both tangents of its contact; the second
                // row's impulse was written by the first.
                int parent_idx = (s_meta_dense[i] >> {type_bits}) - 1;
                if (i != parent_idx + 1) {{
                    new_impulse = old_impulse;
                }} else {{
                    int sib = parent_idx + 2;
                    int sib_row_base = jy_world_base + sib * {D};
                    // A patch's normal rows form a cycle through their parent links; its
                    // anchors share the pooled normal load (a point contact links to -1).
                    float lambda_n = s_lam_dense[parent_idx];
                    for (int patch_row = ((s_meta_dense[parent_idx] >> {type_bits}) - 1);
                         patch_row >= 0 && patch_row != parent_idx;
                         patch_row = ((s_meta_dense[patch_row] >> {type_bits}) - 1))
                        lambda_n += s_lam_dense[patch_row];
                    float radius = fmaxf(s_mu_dense[i] * lambda_n, 0.0f);
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
                }}"""
    if friction_mode == "current":
        mf_friction_update = """                int sib = mf_par + 2;
                int sib_mf6 = mf6_base + sib * 6;
                float2 pair = make_float2(0.0f, 0.0f);
                // Zero load gives a zero disk; avoid the unused tangent reductions.
                if (radius > 0.0f) {
                    float sibling_residual = 0.0f;
                    float cross = 0.0f;
                    if (lane < 6 && dof_a >= 0) {
                        sibling_residual = mf_J_a.data[sib_mf6 + lane] * s_v[dof_a + lane];
                        cross = cur_Ja * mf_MiJt_a.data[sib_mf6 + lane];
                    }
                    if (lane >= 6 && lane < 12 && dof_b >= 0) {
                        sibling_residual = mf_J_b.data[sib_mf6 + lane - 6] * s_v[dof_b + lane - 6];
                        cross = cur_Jb * mf_MiJt_b.data[sib_mf6 + lane - 6];
                    }
                    sibling_residual = warp_sum(sibling_residual) + __int_as_float(mf_meta.data[off_meta + sib * 4 + 2]);
                    cross = warp_sum(cross);
                    float inv_sib = __int_as_float(mf_meta.data[off_meta + sib * 4 + 1]);
                    pair = friction_pair_candidate(mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f, cross,
                        inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f, residual, sibling_residual, old_impulse,
                        s_lam_mf[sib], radius, omega);
                }
                float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                float scale = mag > radius ? radius / mag : 1.0f;
                new_impulse = pair.x * scale;
                float sib_delta = pair.y * scale - s_lam_mf[sib];
                s_lam_mf[sib] = pair.y * scale;
                if (sib_delta != 0.0f) {
                    iteration_changed = 1;
                    if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                    if (lane >= 6 && lane < 12 && dof_b >= 0)
                        s_v[dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                }"""
        mf_friction_radius = f"""            float radius = 0.0f;
            if (mf_rt == {int(PGS_CONSTRAINT_TYPE_FRICTION)}) {{
                float lambda_n = s_lam_mf[mf_par];
                for (int patch_row = (mf_meta.data[off_meta + mf_par * 4 + 3] >> 16);
                     patch_row >= 0 && patch_row != mf_par;
                     patch_row = (mf_meta.data[off_meta + patch_row * 4 + 3] >> 16))
                    lambda_n += s_lam_mf[patch_row];
                radius = fmaxf(mf_row_mu.data[off_mf + i] * lambda_n, 0.0f);
                // A zero disk with no carried impulse cannot change the velocity.
                if (radius == 0.0f && s_lam_mf[i] == 0.0f && s_lam_mf[i + 1] == 0.0f) continue;
            }}"""
        dense_friction_denominator_guard = f" && row_type != {friction_type}"
        mf_friction_denominator_guard = f" && mf_rt != {friction_type}"
    else:
        # The other modes solve each contact's three rows together; their friction rows may
        # have no diagonal of their own.
        if friction_mode in ("bisection", "bisection_desaxce"):
            dense_friction_update = (
                _DENSE_BISECTION_FRICTION_SOURCE.replace("__D__", str(D))
                .replace("__DENSE_META_ROW_TYPE_BITS__", str(type_bits))
                .replace("// __DENSE_DESAXCE_BIAS__", "")
            )
            mf_friction_update = _MF_BISECTION_FRICTION_SOURCE.replace(
                "// __DESAXCE_BIAS__",
                _MF_DESAXCE_BIAS_SOURCE if friction_mode == "bisection_desaxce" else "",
            )
        elif friction_mode == "coulomb_newton":
            mf_friction_update = _MF_COULOMB_NEWTON_FRICTION_SOURCE
        else:
            raise ValueError(f"Unknown friction_mode {friction_mode!r}")
        mf_friction_radius = ""
        dense_friction_denominator_guard = ""
        mf_friction_denominator_guard = ""

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
    int contact_rows = 1;
    int internal_rows = 1;
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
{drive_shared_declarations}
    __shared__ float s_lam_mf[{M_MF}];

    for (int i = lane; i < m_dense; i += 32) {{
        s_lam_dense[i] = world_impulses.data[off_dense + i];
        s_rhs_dense[i] = rhs_bias.data[off_dense + i];
        s_diag_dense[i] = world_diag.data[off_dense + i];
        int row_type = world_row_type.data[off_dense + i];
        int row_parent = world_row_parent.data[off_dense + i];
        s_meta_dense[i] = (row_type & {type_mask}) | ((row_parent + 1) << {type_bits});
        s_mu_dense[i] = world_row_mu.data[off_dense + i];
{drive_loads}
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
            {torsion_skip}
            if (row_type == {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) continue;
            if (freeze_drive_rows != 0 && row_type == {int(PGS_CONSTRAINT_TYPE_JOINT_TARGET)}) continue;
{dense_phase_filter}
            float denom = s_diag_dense[i];
            if (denom <= 0.0f{dense_friction_denominator_guard}) continue;

            float my_sum = 0.0f;
{dense_dot}
            float jv = warp_sum(my_sum);

            float old_impulse = s_lam_dense[i];
            float residual = jv + s_rhs_dense[i];
            // Regularized rows (weight w < 1) also relax the impulse toward zero.
            float w = regularize != 0 ? world_row_w.data[off_dense + i] : 1.0f;
            float delta = denom > 0.0f ? -residual / denom * w - (1.0f - w) * old_impulse : 0.0f;
            float new_impulse = old_impulse + omega * delta;

{drive_update}
            if (row_type == {int(PGS_CONSTRAINT_TYPE_FRICTION)}) {{
{dense_friction_update}
            }} else if (new_impulse < 0.0f && (row_type == {int(PGS_CONSTRAINT_TYPE_CONTACT)} ||
                                               row_type == {int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)})) {{
                // Contact and joint-limit rows are unilateral; mimic and connect rows are bilateral.
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
            if (mf_diag <= 0.0f{mf_friction_denominator_guard}) continue;
{mf_friction_radius}

            float my_sum = 0.0f;
            if (lane < 6 && dof_a >= 0) my_sum = cur_Ja * s_v[dof_a + lane];
            if (lane >= 6 && lane < 12 && dof_b >= 0) my_sum = cur_Jb * s_v[dof_b + lane - 6];
            float jv = warp_sum(my_sum);

            float residual = jv + __int_as_float(meta.z);
            float old_impulse = s_lam_mf[i];
            float delta = -residual * mf_diag;
            if (mf_rt == {int(PGS_CONSTRAINT_TYPE_CONTACT)}) {{
                float w = regularize != 0 ? mf_row_w.data[off_mf + i] : 1.0f;
                delta = -residual * mf_diag * w - (1.0f - w) * old_impulse;
            }}
            float new_impulse = old_impulse + omega * delta;
            if (mf_rt == {int(PGS_CONSTRAINT_TYPE_CONTACT)}) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
            }} else if (mf_rt == {int(PGS_CONSTRAINT_TYPE_FRICTION)}) {{
{mf_friction_update}
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

{torsion_sweep}
        // Velocity limits last: dense joint velocity limits, the fused clamp of driven DOFs,
        // then free-body velocity limits.
        if (velocity_limit_pass) {{
{dense_velocity_limit_pass}
{fused_velocity_clamp_pass}
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
        streamed = [
            ("s_rhs_dense", "rhs_bias"),
            ("s_diag_dense", "world_diag"),
            ("s_mu_dense", "world_row_mu"),
        ]
        if has_drive_rows:
            streamed += [
                ("s_drive_target_dense", "world_drive_target_vel_bias"),
                ("s_drive_vel_mul_dense", "world_drive_vel_multiplier"),
                ("s_drive_imp_mul_dense", "world_drive_impulse_multiplier"),
                ("s_drive_max_imp_dense", "world_drive_max_impulse"),
            ]
        if fuse_vel_limits:
            streamed.append(("s_drive_vel_limit_dense", "world_drive_vel_limit"))
        for sname, gname in streamed:
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
            r"\(\(s_meta_dense\[(\w+)\] >> \d+\) - 1\)",
            r"world_row_parent.data[off_dense + \1]",
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
        world_drive_target_vel_bias: wp.array2d[float],
        world_drive_vel_multiplier: wp.array2d[float],
        world_drive_impulse_multiplier: wp.array2d[float],
        world_drive_max_impulse: wp.array2d[float],
        world_drive_vel_limit: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        world_row_w: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        world_torsion_group: wp.array2d[int],
        contact_torsion_radius: float,
        iterations: int,
        omega: float,
        regularize: int,
        freeze_drive_rows: int,
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
        world_drive_target_vel_bias: wp.array2d[float],
        world_drive_vel_multiplier: wp.array2d[float],
        world_drive_impulse_multiplier: wp.array2d[float],
        world_drive_max_impulse: wp.array2d[float],
        world_drive_vel_limit: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        world_row_w: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        world_torsion_group: wp.array2d[int],
        contact_torsion_radius: float,
        iterations: int,
        omega: float,
        regularize: int,
        freeze_drive_rows: int,
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
            world_drive_target_vel_bias,
            world_drive_vel_multiplier,
            world_drive_impulse_multiplier,
            world_drive_max_impulse,
            world_drive_vel_limit,
            mf_constraint_count,
            mf_contact_rows_end,
            mf_meta,
            mf_impulses,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_mu,
            world_row_w,
            mf_row_w,
            world_torsion_group,
            contact_torsion_radius,
            iterations,
            omega,
            regularize,
            freeze_drive_rows,
            row_phase,
            v_out,
        )

    name = (
        f"pgs_solve_mf_gs_{max_constraints}_{mf_max_constraints}_{max_world_dofs}"
        f"_vlim{int(has_dense_velocity_limit_rows)}_drive{int(has_drive_rows)}_fvl{int(fuse_vel_limits)}"
        f"{'' if shared_metadata else '_gmeta'}"
        f"{'_torsion' if contact_torsion else ''}"
        f"{'_phased' if row_phases else ''}"
        f"{'' if friction_mode == 'current' else '_' + friction_mode}"
    )
    pgs_solve_mf_gs.__name__ = name
    pgs_solve_mf_gs.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_gs)


# Static shared memory available to a CUDA block without opt-in [B].
_STATIC_SHARED_MEMORY_BYTES = 48 * 1024
# Shared-memory budget of the streaming Delassus kernel's J and Y row chunks [B].
_DELASSUS_SHARED_MEMORY_BYTES = 45000
_DELASSUS_MAX_CHUNK_SIZE = 64


def _select_delassus_chunk_size(n_dofs: int, max_constraints: int) -> int | None:
    """Select the rows staged per chunk by the streaming Delassus kernel.

    The whole row set is staged when its ``J`` and ``Y`` rows fit the budget; otherwise
    chunks of at most 64 rows. Returns ``None`` when a single row does not fit. The chunk
    size does not change the result: every Delassus entry is computed once.
    """
    row_bytes = 2 * 4 * n_dofs
    if max_constraints * row_bytes <= _DELASSUS_SHARED_MEMORY_BYTES:
        return max_constraints
    chunk = min(_DELASSUS_MAX_CHUNK_SIZE, _DELASSUS_SHARED_MEMORY_BYTES // row_bytes)
    return chunk if chunk >= 1 else None


def _estimate_tiled_row_shared_memory(max_constraints: int) -> int:
    """Estimate the shared memory of the dense split-mode Gauss-Seidel kernel [B]."""
    return 4 * (max_constraints * (max_constraints + 1) // 2 + 6 * max_constraints)


def _estimate_contact_pgs_shared_memory(max_constraints: int, kernel: str, chunk_size: int) -> int:
    """Estimate the static shared memory of a contact-only Gauss-Seidel kernel [B]."""
    contacts = max_constraints // 3
    floats = 2 * 3 * contacts + contacts + 9 * contacts
    if kernel == "tiled_contact":
        floats += 9 * contacts * (contacts + 1) // 2
    else:
        floats += 9 * chunk_size * contacts
    return 4 * floats


def _estimate_mf_solve_shared_memory(mf_max_constraints: int, max_mf_bodies: int) -> int:
    """Estimate the shared memory of the split-mode free-body Gauss-Seidel kernel [B]."""
    return 4 * (7 * max_mf_bodies + mf_max_constraints)


@cache
def _get_hinv_jt_fused_kernel(
    n_dofs: int, max_constraints: int, device_arch: str, tile_threads: int = 64
) -> "wp.Kernel":
    """Build the fused split-mode ``Y = H^-1 J^T`` and Delassus ``C = J Y^T`` kernel.

    One block per articulation stores the whole ``max_constraints x max_constraints``
    Delassus tile of its world, so it requires one solved articulation per world. The
    diagonal receives the constraint force mixing.
    """
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))
    TILE_CONSTRAINTS_LOCAL = wp.constant(int(max_constraints))

    def hinv_jt_tiled_fused_template(
        L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        J_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        world_constraint_count: wp.array[int],
        pgs_cfm: float,
        # outputs
        world_C: wp.array3d[float],  # [world_count, max_c, max_c]
        world_diag: wp.array2d[float],  # [world_count, max_c]
        Y_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
    ):
        idx, thread = wp.tid()
        art = group_to_art[idx]
        world = art_to_world[art]
        n_constraints = world_constraint_count[world]

        if n_constraints == 0:
            return

        # Load L (Cholesky factor) and J (Jacobian rows)
        L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        J_tile = wp.tile_load(J_group[idx], shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL), bounds_check=False)

        # Solve L * Z = J^T (forward substitution)
        Jt_tile = wp.tile_transpose(J_tile)
        Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)

        # Solve L^T * Y = Z (backward substitution)
        Lt_tile = wp.tile_transpose(L_tile)
        X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)

        # Store Y = H^-1 * J^T (transpose back to row layout)
        Y_out_tile = wp.tile_transpose(X_tile)
        wp.tile_store(Y_group[idx], Y_out_tile)

        # Form C = J * H^-1 * J^T
        C_tile = wp.tile_zeros(shape=(TILE_CONSTRAINTS_LOCAL, TILE_CONSTRAINTS_LOCAL), dtype=wp.float32)
        wp.tile_matmul(J_tile, X_tile, C_tile)
        wp.tile_store(world_C[world], C_tile)

        if thread == 0:
            for i in range(n_constraints):
                world_diag[world, i] = C_tile[i, i] + pgs_cfm

    hinv_jt_tiled_fused_template.__name__ = f"hinv_jt_tiled_fused_{n_dofs}_{max_constraints}_bd{tile_threads}"
    hinv_jt_tiled_fused_template.__qualname__ = f"hinv_jt_tiled_fused_{n_dofs}_{max_constraints}_bd{tile_threads}"
    return wp.kernel(enable_backward=False, module="unique")(hinv_jt_tiled_fused_template)


@cache
def _get_delassus_kernel(n_dofs: int, max_constraints: int, chunk_size: int, device_arch: str) -> "wp.Kernel":
    """Build the streaming Delassus kernel ``C += J Y^T`` of one size group.

    One block per articulation stages ``chunk_size`` rows of ``J`` and ``Y`` at a time in
    shared memory (see :func:`_select_delassus_chunk_size`) and adds each entry of its
    world's block once.
    """
    _ = device_arch
    TILE_D = n_dofs
    TILE_M = max_constraints
    CHUNK = chunk_size

    snippet = f"""
#if defined(__CUDA_ARCH__)
const int TILE_D = {TILE_D};
const int TILE_M = {TILE_M};
const int CHUNK = {CHUNK};

int lane = threadIdx.x;
int art = group_to_art.data[idx];
int world = art_to_world.data[art];
int m = world_constraint_count.data[world];
if (m == 0) return;

__shared__ float s_J[CHUNK * TILE_D];
__shared__ float s_Y[CHUNK * TILE_D];

int num_chunks = (m + CHUNK - 1) / CHUNK;

for (int ci = 0; ci < num_chunks; ci++) {{
    int i0 = ci * CHUNK, i1 = min(i0 + CHUNK, m);

    for (int t = lane; t < (i1 - i0) * TILE_D; t += blockDim.x)
        s_J[t] = J_group.data[idx * TILE_M * TILE_D + i0 * TILE_D + t];
    __syncthreads();

    for (int cj = 0; cj < num_chunks; cj++) {{
        int j0 = cj * CHUNK, j1 = min(j0 + CHUNK, m);

        for (int t = lane; t < (j1 - j0) * TILE_D; t += blockDim.x)
            s_Y[t] = Y_group.data[idx * TILE_M * TILE_D + j0 * TILE_D + t];
        __syncthreads();

        // Each thread computes multiple C elements
        for (int e = lane; e < (i1 - i0) * (j1 - j0); e += blockDim.x) {{
            int il = e / (j1 - j0), jl = e % (j1 - j0);
            float sum = 0.0f;
            for (int k = 0; k < TILE_D; k++)
                sum += s_J[il * TILE_D + k] * s_Y[jl * TILE_D + k];
            if (sum != 0.0f) {{
                int ig = i0 + il, jg = j0 + jl;
                atomicAdd(&world_C.data[world * TILE_M * TILE_M + ig * TILE_M + jg], sum);
                if (ig == jg) atomicAdd(&world_diag.data[world * TILE_M + ig], sum);
            }}
        }}
        __syncthreads();
    }}
}}
#endif
"""

    @wp.func_native(snippet)
    def delassus_native(
        idx: int,
        J_group: wp.array3d[float],
        Y_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        world_constraint_count: wp.array[int],
        world_C: wp.array3d[float],
        world_diag: wp.array2d[float],
    ): ...

    def delassus_template(
        J_group: wp.array3d[float],
        Y_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        world_constraint_count: wp.array[int],
        n_arts: int,
        world_C: wp.array3d[float],
        world_diag: wp.array2d[float],
    ):
        idx, _lane = wp.tid()
        if idx < n_arts:
            delassus_native(
                idx, J_group, Y_group, group_to_art, art_to_world, world_constraint_count, world_C, world_diag
            )

    delassus_template.__name__ = f"delassus_streaming_{n_dofs}_{max_constraints}_chunk{CHUNK}"
    delassus_template.__qualname__ = f"delassus_streaming_{n_dofs}_{max_constraints}_chunk{CHUNK}"
    return wp.kernel(enable_backward=False, module="unique")(delassus_template)


@cache
def _get_pgs_solve_tiled_row_kernel(max_constraints: int, device_arch: str) -> "wp.Kernel":
    """Build the one-warp-per-world Gauss-Seidel kernel of the dense Delassus system.

    The kernel stages only the lower triangle of ``C`` (``M (M + 1) / 2`` floats, see
    :func:`_estimate_tiled_row_shared_memory`) and reads ``C_ij`` by symmetry. Each row's
    ``sum_j C_ij lambda_j`` is a warp reduction; the projection matches
    :func:`~newton._src.solvers.feather_pgs.kernels.pgs_solve_loop`.
    """
    TILE_M = max_constraints
    TILE_M_SQ = TILE_M * TILE_M
    TILE_TRI = TILE_M * (TILE_M + 1) // 2

    ELEMS_PER_THREAD_1D = (TILE_M + 31) // 32

    def gen_load_1d(dst, src):
        lines = []
        for k in range(ELEMS_PER_THREAD_1D):
            offset = k * 32
            guard = f"if (lane + {offset} < TILE_M) " if offset + 32 > TILE_M else ""
            lines.append(f"    {guard}{dst}[lane + {offset}] = {src}.data[off1 + lane + {offset}];")
        return "\n".join(lines)

    # Build a deterministic packed-lower-tri index order: row-major over (i, j<=i)
    # idx = i*(i+1)/2 + j
    tri_pairs = []
    for i in range(TILE_M):
        base = i * (i + 1) // 2
        for j in range(i + 1):
            tri_pairs.append((base + j, i, j))
    assert len(tri_pairs) == TILE_TRI

    load_code = "\n".join(
        [
            gen_load_1d("s_lam", "world_impulses"),
            gen_load_1d("s_rhs", "world_rhs"),
            gen_load_1d("s_diag", "world_diag"),
            gen_load_1d("s_rtype", "world_row_type"),
            gen_load_1d("s_parent", "world_row_parent"),
            gen_load_1d("s_mu", "world_row_mu"),
        ]
    )

    # Precompute lane's column indices (j_k) and their triangular bases (j_k*(j_k+1)/2)
    # so inside the dot we avoid multiply.
    precompute_j = []
    for k in range(ELEMS_PER_THREAD_1D):
        j = k * 32
        if j < TILE_M:
            precompute_j.append(f"    const int j{k} = lane + {j};\n    const int jb{k} = (j{k} * (j{k} + 1)) >> 1;")
    precompute_j_code = "\n".join(precompute_j)

    # Dot code: guarded on j_k < m
    dot_terms = []
    for k in range(ELEMS_PER_THREAD_1D):
        joff = k * 32
        if joff < TILE_M:
            dot_terms.append(
                f"""    if (j{k} < m) {{
        // Use symmetry to fetch C(i, j{k}) from packed-lower shared.
        // base_i = i*(i+1)/2
        float cij = (j{k} <= i) ? s_Ctri[base_i + j{k}] : s_Ctri[jb{k} + i];
        my_sum += cij * s_lam[j{k}];
    }}"""
            )
    dot_code = "\n".join(["float my_sum = 0.0f;", "int base_i = (i * (i + 1)) >> 1;", *dot_terms])

    store_lines = []
    for k in range(ELEMS_PER_THREAD_1D):
        offset = k * 32
        guard = f"if (lane + {offset} < TILE_M) " if offset + 32 > TILE_M else ""
        store_lines.append(f"    {guard}world_impulses.data[off1 + lane + {offset}] = s_lam[lane + {offset}];")
    store_code = "\n".join(store_lines)

    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int TILE_M = {TILE_M};
    const int TILE_M_SQ = {TILE_M_SQ};
    const int TILE_TRI = {TILE_TRI};
    const unsigned MASK = 0xFFFFFFFF;

    int lane = threadIdx.x;

    int m = world_constraint_count.data[world];
    if (m == 0) return;

    // Packed LOWER triangle of C in row-major (i*(i+1)/2 + j), j<=i
    __shared__ float s_Ctri[TILE_TRI];

    __shared__ float s_lam[TILE_M];
    __shared__ float s_rhs[TILE_M];
    __shared__ float s_diag[TILE_M];
    __shared__ int   s_rtype[TILE_M];
    __shared__ int   s_parent[TILE_M];
    __shared__ float s_mu[TILE_M];

    int off1 = world * TILE_M;
    int off2 = world * TILE_M_SQ;

{load_code}

    // Load only lower triangle from global full matrix into packed shared.
    // Work distribution: each lane walks rows; for each row i, lane loads j = lane, lane+32, lane+64...
    for (int i = 0; i < TILE_M; ++i) {{
        int base = (i * (i + 1)) >> 1; // packed base for row i
        for (int j = lane; j <= i; j += 32) {{
            s_Ctri[base + j] = world_C.data[off2 + i * TILE_M + j];
        }}
    }}
    __syncwarp();

{precompute_j_code}

    for (int iter = 0; iter < iterations; iter++) {{
        for (int i = 0; i < m; i++) {{
            {dot_code}

            // Warp reduce my_sum
            my_sum += __shfl_down_sync(MASK, my_sum, 16);
            my_sum += __shfl_down_sync(MASK, my_sum, 8);
            my_sum += __shfl_down_sync(MASK, my_sum, 4);
            my_sum += __shfl_down_sync(MASK, my_sum, 2);
            my_sum += __shfl_down_sync(MASK, my_sum, 1);
            float dot_sum = __shfl_sync(MASK, my_sum, 0);

            float denom = s_diag[i];
            if (denom <= 0.0f && s_rtype[i] != 2) continue;

            float w_val = s_rhs[i] + dot_sum;
            float delta = denom > 0.0f ? -w_val / denom : 0.0f;
            float new_impulse = s_lam[i] + omega * delta;
            int row_type = s_rtype[i];

            if (row_type == 2 && i != s_parent[i] + 1) continue;

            // Contact (0) and joint-limit (3) rows are unilateral.
            if (row_type == 0 || row_type == 3) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
                s_lam[i] = new_impulse;
            }} else if (row_type == 2) {{
                int parent_idx = s_parent[i];
                float radius = fmaxf(s_mu[i] * s_lam[parent_idx], 0.0f);
                int sib = parent_idx + 2;
                float sibling_residual = 0.0f;
                int sib_base = (sib * (sib + 1)) >> 1;
                for (int j = lane; j < m; j += 32) {{
                    float c = j <= sib ? s_Ctri[sib_base + j] : s_Ctri[((j * (j + 1)) >> 1) + sib];
                    sibling_residual += c * s_lam[j];
                }}
                for (int offset = 16; offset > 0; offset >>= 1)
                    sibling_residual += __shfl_down_sync(MASK, sibling_residual, offset);
                sibling_residual = __shfl_sync(MASK, sibling_residual, 0) + s_rhs[sib];
                float2 pair = friction_pair_candidate(denom, s_Ctri[sib_base + i], s_diag[sib],
                    w_val, sibling_residual, s_lam[i], s_lam[sib], radius, omega);
                float a = pair.x;
                float b = pair.y;
                float mag = sqrtf(a * a + b * b);
                float scale = mag > radius ? radius / mag : 1.0f;
                s_lam[i] = a * scale;
                s_lam[sib] = b * scale;
            }} else {{
                s_lam[i] = new_impulse;
            }}
        }}
    }}

{store_code}
#endif
"""

    snippet = snippet.replace("#if defined(__CUDA_ARCH__)", "#if defined(__CUDA_ARCH__)\n" + FRICTION_PAIR_CUDA, 1)

    @wp.func_native(snippet)
    def pgs_solve_native(
        world: int,
        world_constraint_count: wp.array[int],
        world_diag: wp.array2d[float],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_impulses: wp.array2d[float],
    ): ...

    def pgs_solve_tiled_template(
        world_constraint_count: wp.array[int],
        world_diag: wp.array2d[float],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_impulses: wp.array2d[float],
    ):
        world, _lane = wp.tid()
        pgs_solve_native(
            world,
            world_constraint_count,
            world_diag,
            world_C,
            world_rhs,
            iterations,
            omega,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_impulses,
        )

    pgs_solve_tiled_template.__name__ = f"pgs_solve_tiled_row_{max_constraints}"
    pgs_solve_tiled_template.__qualname__ = f"pgs_solve_tiled_row_{max_constraints}"
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_tiled_template)


@cache
def _get_pgs_solve_tiled_contact_kernel(max_constraints: int, device_arch: str) -> "wp.Kernel":
    """Build the contact-only Gauss-Seidel kernel of the dense Delassus system in 3x3 blocks.

    Stores only the LOWER triangle of block Delassus matrix.
    Each contact is a 3-vector (normal, tangent1, tangent2).
    Reduces serial depth from M to M/3.

    TILE_M can be any value (power of 2 recommended for other kernels).
    Runtime m must be divisible by 3.
    """
    TILE_M = max_constraints
    # Max contacts we can handle (rounded down)
    NUM_CONTACTS_MAX = TILE_M // 3
    # Actual max constraints we'll process (may be < TILE_M)
    TILE_M_USABLE = NUM_CONTACTS_MAX * 3

    # Lower triangle of block matrix (sized for max)
    NUM_BLOCKS_TRI = NUM_CONTACTS_MAX * (NUM_CONTACTS_MAX + 1) // 2
    BLOCK_TRI_FLOATS = NUM_BLOCKS_TRI * 9

    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int TILE_M = {TILE_M};
    const int TILE_M_USABLE = {TILE_M_USABLE};
    const int NUM_CONTACTS_MAX = {NUM_CONTACTS_MAX};
    const int BLOCK_TRI_FLOATS = {BLOCK_TRI_FLOATS};
    const unsigned MASK = 0xFFFFFFFF;

    int lane = threadIdx.x;

    int m = world_constraint_count.data[world];
    if (m == 0) return;

    // Clamp m to usable range and ensure divisible by 3
    if (m > TILE_M_USABLE) m = TILE_M_USABLE;
    int num_contacts = m / 3;

    // Shared memory (sized for max)
    __shared__ float s_Dtri[BLOCK_TRI_FLOATS];
    __shared__ float s_Dinv[NUM_CONTACTS_MAX * 9];
    __shared__ float s_lam[TILE_M_USABLE];
    __shared__ float s_rhs[TILE_M_USABLE];
    __shared__ float s_mu[NUM_CONTACTS_MAX];

    int off1 = world * TILE_M;
    int off2 = world * TILE_M * TILE_M;

    // ============ LOAD PHASE ============

    // Load lambda and rhs
    for (int i = lane; i < TILE_M_USABLE; i += 32) {{
        if (i < m) {{
            s_lam[i] = world_impulses.data[off1 + i];
            s_rhs[i] = world_rhs.data[off1 + i];
        }} else {{
            s_lam[i] = 0.0f;
            s_rhs[i] = 0.0f;
        }}
    }}

    // Load mu (one per contact, stored on tangent1 row)
    for (int c = lane; c < NUM_CONTACTS_MAX; c += 32) {{
        if (c < num_contacts) {{
            s_mu[c] = world_row_mu.data[off1 + c * 3 + 1];
        }}
    }}

    // Load lower triangle of block Delassus
    for (int c = 0; c < num_contacts; c++) {{
        int base_block = (c * (c + 1)) >> 1;
        int floats_in_row = (c + 1) * 9;

        for (int f = lane; f < floats_in_row; f += 32) {{
            int j = f / 9;
            int k = f % 9;
            int lr = k / 3;
            int lc = k % 3;
            int gr = c * 3 + lr;
            int gc = j * 3 + lc;
            s_Dtri[(base_block + j) * 9 + k] = world_C.data[off2 + gr * TILE_M + gc];
        }}
    }}
    __syncwarp();

    // Compute diagonal block inverses
    for (int c = lane; c < num_contacts; c += 32) {{
        int diag_block_idx = ((c * (c + 1)) >> 1) + c;
        const float* D = &s_Dtri[diag_block_idx * 9];
        float* Dinv = &s_Dinv[c * 9];

        float det = D[0] * (D[4] * D[8] - D[5] * D[7])
                - D[1] * (D[3] * D[8] - D[5] * D[6])
                + D[2] * (D[3] * D[7] - D[4] * D[6]);

        float inv_det = 1.0f / det;

        Dinv[0] = (D[4] * D[8] - D[5] * D[7]) * inv_det;
        Dinv[1] = (D[2] * D[7] - D[1] * D[8]) * inv_det;
        Dinv[2] = (D[1] * D[5] - D[2] * D[4]) * inv_det;
        Dinv[3] = (D[5] * D[6] - D[3] * D[8]) * inv_det;
        Dinv[4] = (D[0] * D[8] - D[2] * D[6]) * inv_det;
        Dinv[5] = (D[2] * D[3] - D[0] * D[5]) * inv_det;
        Dinv[6] = (D[3] * D[7] - D[4] * D[6]) * inv_det;
        Dinv[7] = (D[1] * D[6] - D[0] * D[7]) * inv_det;
        Dinv[8] = (D[0] * D[4] - D[1] * D[3]) * inv_det;
    }}
    __syncwarp();

    // ============ ITERATION PHASE ============

    for (int iter = 0; iter < iterations; iter++) {{
        for (int c = 0; c < num_contacts; c++) {{
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

            for (int j = lane; j < num_contacts; j += 32) {{
                float l0 = s_lam[j * 3 + 0];
                float l1 = s_lam[j * 3 + 1];
                float l2 = s_lam[j * 3 + 2];

                int block_off;
                bool transpose;
                if (j <= c) {{
                    block_off = (((c * (c + 1)) >> 1) + j) * 9;
                    transpose = false;
                }} else {{
                    block_off = (((j * (j + 1)) >> 1) + c) * 9;
                    transpose = true;
                }}

                const float* B = &s_Dtri[block_off];

                if (!transpose) {{
                    sum0 += B[0] * l0 + B[1] * l1 + B[2] * l2;
                    sum1 += B[3] * l0 + B[4] * l1 + B[5] * l2;
                    sum2 += B[6] * l0 + B[7] * l1 + B[8] * l2;
                }} else {{
                    sum0 += B[0] * l0 + B[3] * l1 + B[6] * l2;
                    sum1 += B[1] * l0 + B[4] * l1 + B[7] * l2;
                    sum2 += B[2] * l0 + B[5] * l1 + B[8] * l2;
                }}
            }}

            // Warp reduce
            sum0 += __shfl_down_sync(MASK, sum0, 16);
            sum1 += __shfl_down_sync(MASK, sum1, 16);
            sum2 += __shfl_down_sync(MASK, sum2, 16);
            sum0 += __shfl_down_sync(MASK, sum0, 8);
            sum1 += __shfl_down_sync(MASK, sum1, 8);
            sum2 += __shfl_down_sync(MASK, sum2, 8);
            sum0 += __shfl_down_sync(MASK, sum0, 4);
            sum1 += __shfl_down_sync(MASK, sum1, 4);
            sum2 += __shfl_down_sync(MASK, sum2, 4);
            sum0 += __shfl_down_sync(MASK, sum0, 2);
            sum1 += __shfl_down_sync(MASK, sum1, 2);
            sum2 += __shfl_down_sync(MASK, sum2, 2);
            sum0 += __shfl_down_sync(MASK, sum0, 1);
            sum1 += __shfl_down_sync(MASK, sum1, 1);
            sum2 += __shfl_down_sync(MASK, sum2, 1);

            if (lane == 0) {{
                // Corrected sign: -(rhs + D*lambda)
                float res0 = -(s_rhs[c * 3 + 0] + sum0);
                float res1 = -(s_rhs[c * 3 + 1] + sum1);
                float res2 = -(s_rhs[c * 3 + 2] + sum2);

                const float* Dinv = &s_Dinv[c * 9];
                float d0 = Dinv[0] * res0 + Dinv[1] * res1 + Dinv[2] * res2;
                float d1 = Dinv[3] * res0 + Dinv[4] * res1 + Dinv[5] * res2;
                float d2 = Dinv[6] * res0 + Dinv[7] * res1 + Dinv[8] * res2;

                float new_n  = s_lam[c * 3 + 0] + omega * d0;
                float new_t1 = s_lam[c * 3 + 1] + omega * d1;
                float new_t2 = s_lam[c * 3 + 2] + omega * d2;

                // Friction cone projection
                new_n = fmaxf(new_n, 0.0f);

                float mu = s_mu[c];
                float radius = mu * new_n;

                if (radius <= 0.0f) {{
                    new_t1 = 0.0f;
                    new_t2 = 0.0f;
                }} else {{
                    float t_mag_sq = new_t1 * new_t1 + new_t2 * new_t2;
                    if (t_mag_sq > radius * radius) {{
                        float scale = radius * rsqrtf(t_mag_sq);
                        new_t1 *= scale;
                        new_t2 *= scale;
                    }}
                }}

                s_lam[c * 3 + 0] = new_n;
                s_lam[c * 3 + 1] = new_t1;
                s_lam[c * 3 + 2] = new_t2;
            }}
            __syncwarp();
        }}
    }}

    // ============ STORE PHASE ============

    for (int i = lane; i < TILE_M_USABLE; i += 32) {{
        if (i < m) {{
            world_impulses.data[off1 + i] = s_lam[i];
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def pgs_solve_contact_native(
        world: int,
        world_constraint_count: wp.array[int],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_mu: wp.array2d[float],
    ): ...

    def pgs_solve_tiled_contact_template(
        world_constraint_count: wp.array[int],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_mu: wp.array2d[float],
    ):
        world, _lane = wp.tid()
        pgs_solve_contact_native(
            world,
            world_constraint_count,
            world_C,
            world_rhs,
            world_impulses,
            iterations,
            omega,
            world_row_mu,
        )

    pgs_solve_tiled_contact_template.__name__ = f"pgs_solve_tiled_contact_{max_constraints}"
    pgs_solve_tiled_contact_template.__qualname__ = f"pgs_solve_tiled_contact_{max_constraints}"
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_tiled_contact_template)


@cache
def _get_pgs_solve_streaming_kernel(max_constraints: int, device_arch: str, pgs_chunk_size: int = 1) -> "wp.Kernel":
    """Streaming contact-wise PGS kernel that streams block-rows from global memory.

    Unlike tiled_contact which loads the entire Delassus matrix into shared memory,
    this kernel keeps only lambda and auxiliaries in shared memory and streams
    block-rows of C on demand. This enables handling much larger constraint counts
    (hundreds of contacts) at the cost of increased global memory bandwidth.

    When pgs_chunk_size > 1, multiple block-rows are preloaded into shared memory
    at once, reducing the number of global memory round-trips per PGS iteration.

    Algorithm:
    - Load lambda, rhs, mu, and compute diagonal block inverses once
    - For each PGS iteration:
        - For each chunk of pgs_chunk_size contacts:
            - Preload pgs_chunk_size block-rows of C into shared memory
            - For each contact c in the chunk:
                - Compute block-row dot product with lambda (warp-parallel)
                - Update lambda[c] with friction cone projection (lane 0)
    - Store final lambda back to global memory
    """
    TILE_M = max_constraints
    NUM_CONTACTS_MAX = TILE_M // 3
    TILE_M_USABLE = NUM_CONTACTS_MAX * 3
    PGS_CHUNK = pgs_chunk_size

    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int TILE_M = {TILE_M};
    const int TILE_M_USABLE = {TILE_M_USABLE};
    const int NUM_CONTACTS_MAX = {NUM_CONTACTS_MAX};
    const int PGS_CHUNK = {PGS_CHUNK};
    const unsigned MASK = 0xFFFFFFFF;

    int lane = threadIdx.x;

    int m = world_constraint_count.data[world];
    if (m == 0) return;

    // Clamp m to usable range and ensure divisible by 3
    if (m > TILE_M_USABLE) m = TILE_M_USABLE;
    int num_contacts = m / 3;

    // ═══════════════════════════════════════════════════════════════
    // SHARED MEMORY: lambda, rhs, mu, diagonal inverses, and
    // block-row buffer for PGS_CHUNK contacts at a time
    // ═══════════════════════════════════════════════════════════════
    __shared__ float s_lam[{TILE_M_USABLE}];
    __shared__ float s_rhs[{TILE_M_USABLE}];
    __shared__ float s_mu[{NUM_CONTACTS_MAX}];
    __shared__ float s_Dinv[{NUM_CONTACTS_MAX} * 9];
    __shared__ float s_block_rows[{PGS_CHUNK} * {NUM_CONTACTS_MAX} * 9];

    int off1 = world * TILE_M;
    int off2 = world * TILE_M * TILE_M;

    // ═══════════════════════════════════════════════════════════════
    // LOAD PHASE: Load persistent data into shared memory
    // ═══════════════════════════════════════════════════════════════

    // Load lambda and rhs (coalesced)
    for (int i = lane; i < TILE_M_USABLE; i += 32) {{
        if (i < m) {{
            s_lam[i] = world_impulses.data[off1 + i];
            s_rhs[i] = world_rhs.data[off1 + i];
        }} else {{
            s_lam[i] = 0.0f;
            s_rhs[i] = 0.0f;
        }}
    }}

    // Load mu (one per contact, stored on tangent1 row)
    for (int c = lane; c < NUM_CONTACTS_MAX; c += 32) {{
        if (c < num_contacts) {{
            s_mu[c] = world_row_mu.data[off1 + c * 3 + 1];
        }}
    }}
    __syncwarp();

    // Compute diagonal block inverses (each thread handles one contact)
    for (int c = lane; c < num_contacts; c += 32) {{
        // Load diagonal block D[c,c] from global memory
        int diag_row = c * 3;
        float D[9];
        for (int k = 0; k < 9; k++) {{
            int lr = k / 3;
            int lc = k % 3;
            D[k] = world_C.data[off2 + (diag_row + lr) * TILE_M + (diag_row + lc)];
        }}

        // Compute 3x3 inverse
        float det = D[0] * (D[4] * D[8] - D[5] * D[7])
                  - D[1] * (D[3] * D[8] - D[5] * D[6])
                  + D[2] * (D[3] * D[7] - D[4] * D[6]);

        float inv_det = 1.0f / det;
        float* Dinv = &s_Dinv[c * 9];

        Dinv[0] = (D[4] * D[8] - D[5] * D[7]) * inv_det;
        Dinv[1] = (D[2] * D[7] - D[1] * D[8]) * inv_det;
        Dinv[2] = (D[1] * D[5] - D[2] * D[4]) * inv_det;
        Dinv[3] = (D[5] * D[6] - D[3] * D[8]) * inv_det;
        Dinv[4] = (D[0] * D[8] - D[2] * D[6]) * inv_det;
        Dinv[5] = (D[2] * D[3] - D[0] * D[5]) * inv_det;
        Dinv[6] = (D[3] * D[7] - D[4] * D[6]) * inv_det;
        Dinv[7] = (D[1] * D[6] - D[0] * D[7]) * inv_det;
        Dinv[8] = (D[0] * D[4] - D[1] * D[3]) * inv_det;
    }}
    __syncwarp();

    // ═══════════════════════════════════════════════════════════════
    // ITERATION PHASE: Stream block-rows in chunks and solve
    // ═══════════════════════════════════════════════════════════════

    for (int iter = 0; iter < iterations; iter++) {{
        for (int chunk_start = 0; chunk_start < num_contacts; chunk_start += PGS_CHUNK) {{
            int chunk_end = min(chunk_start + PGS_CHUNK, num_contacts);
            int chunk_len = chunk_end - chunk_start;

            // ─────────────────────────────────────────────────────────
            // STREAM: Preload chunk_len block-rows of Delassus matrix
            // ─────────────────────────────────────────────────────────
            for (int ci = 0; ci < chunk_len; ci++) {{
                int c = chunk_start + ci;
                int c_row = c * 3;
                float* row_base = &s_block_rows[ci * NUM_CONTACTS_MAX * 9];
                for (int j = lane; j < num_contacts; j += 32) {{
                    int j_col = j * 3;
                    float* dst = &row_base[j * 9];
                    for (int k = 0; k < 9; k++) {{
                        int lr = k / 3;
                        int lc = k % 3;
                        dst[k] = world_C.data[off2 + (c_row + lr) * TILE_M + (j_col + lc)];
                    }}
                }}
            }}
            __syncwarp();

            // ─────────────────────────────────────────────────────────
            // SOLVE: Process each contact in the chunk sequentially
            // ─────────────────────────────────────────────────────────
            for (int ci = 0; ci < chunk_len; ci++) {{
                int c = chunk_start + ci;
                const float* row_base = &s_block_rows[ci * NUM_CONTACTS_MAX * 9];

                // Block-row dot product sum_j C[c,j] * lambda[j]
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

                for (int j = lane; j < num_contacts; j += 32) {{
                    float l0 = s_lam[j * 3 + 0];
                    float l1 = s_lam[j * 3 + 1];
                    float l2 = s_lam[j * 3 + 2];

                    const float* B = &row_base[j * 9];

                    sum0 += B[0] * l0 + B[1] * l1 + B[2] * l2;
                    sum1 += B[3] * l0 + B[4] * l1 + B[5] * l2;
                    sum2 += B[6] * l0 + B[7] * l1 + B[8] * l2;
                }}

                // Warp reduce
                sum0 += __shfl_down_sync(MASK, sum0, 16);
                sum1 += __shfl_down_sync(MASK, sum1, 16);
                sum2 += __shfl_down_sync(MASK, sum2, 16);
                sum0 += __shfl_down_sync(MASK, sum0, 8);
                sum1 += __shfl_down_sync(MASK, sum1, 8);
                sum2 += __shfl_down_sync(MASK, sum2, 8);
                sum0 += __shfl_down_sync(MASK, sum0, 4);
                sum1 += __shfl_down_sync(MASK, sum1, 4);
                sum2 += __shfl_down_sync(MASK, sum2, 4);
                sum0 += __shfl_down_sync(MASK, sum0, 2);
                sum1 += __shfl_down_sync(MASK, sum1, 2);
                sum2 += __shfl_down_sync(MASK, sum2, 2);
                sum0 += __shfl_down_sync(MASK, sum0, 1);
                sum1 += __shfl_down_sync(MASK, sum1, 1);
                sum2 += __shfl_down_sync(MASK, sum2, 1);

                // Update: Solve and project (lane 0 only)
                if (lane == 0) {{
                    float res0 = -(s_rhs[c * 3 + 0] + sum0);
                    float res1 = -(s_rhs[c * 3 + 1] + sum1);
                    float res2 = -(s_rhs[c * 3 + 2] + sum2);

                    const float* Dinv = &s_Dinv[c * 9];
                    float d0 = Dinv[0] * res0 + Dinv[1] * res1 + Dinv[2] * res2;
                    float d1 = Dinv[3] * res0 + Dinv[4] * res1 + Dinv[5] * res2;
                    float d2 = Dinv[6] * res0 + Dinv[7] * res1 + Dinv[8] * res2;

                    float new_n  = s_lam[c * 3 + 0] + omega * d0;
                    float new_t1 = s_lam[c * 3 + 1] + omega * d1;
                    float new_t2 = s_lam[c * 3 + 2] + omega * d2;

                    // Friction cone projection
                    new_n = fmaxf(new_n, 0.0f);

                    float mu = s_mu[c];
                    float radius = mu * new_n;

                    if (radius <= 0.0f) {{
                        new_t1 = 0.0f;
                        new_t2 = 0.0f;
                    }} else {{
                        float t_mag_sq = new_t1 * new_t1 + new_t2 * new_t2;
                        if (t_mag_sq > radius * radius) {{
                            float scale = radius * rsqrtf(t_mag_sq);
                            new_t1 *= scale;
                            new_t2 *= scale;
                        }}
                    }}

                    s_lam[c * 3 + 0] = new_n;
                    s_lam[c * 3 + 1] = new_t1;
                    s_lam[c * 3 + 2] = new_t2;
                }}
                __syncwarp();
            }}
        }}
    }}

    // ═══════════════════════════════════════════════════════════════
    // STORE PHASE: Write final lambda back to global memory
    // ═══════════════════════════════════════════════════════════════
    for (int i = lane; i < TILE_M_USABLE; i += 32) {{
        if (i < m) {{
            world_impulses.data[off1 + i] = s_lam[i];
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def pgs_solve_streaming_native(
        world: int,
        world_constraint_count: wp.array[int],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_mu: wp.array2d[float],
    ): ...

    def pgs_solve_streaming_template(
        world_constraint_count: wp.array[int],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_mu: wp.array2d[float],
    ):
        world, _lane = wp.tid()
        pgs_solve_streaming_native(
            world,
            world_constraint_count,
            world_C,
            world_rhs,
            world_impulses,
            iterations,
            omega,
            world_row_mu,
        )

    pgs_solve_streaming_template.__name__ = f"pgs_solve_streaming_{max_constraints}_chunk{pgs_chunk_size}"
    pgs_solve_streaming_template.__qualname__ = f"pgs_solve_streaming_{max_constraints}_chunk{pgs_chunk_size}"
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_streaming_template)


@cache
def _get_pgs_solve_mf_kernel(mf_max_constraints: int, max_mf_bodies: int, device_arch: str) -> "wp.Kernel":
    """Build the one-warp-per-world Gauss-Seidel kernel of the split-mode free-body rows.

    The velocities of the world's free bodies (through the local body table of
    :func:`~newton._src.solvers.feather_pgs.kernels.build_mf_body_map`) and the row impulses
    stay in shared memory for all iterations, see :func:`_estimate_mf_solve_shared_memory`;
    lane 0 sweeps the rows. The projection matches
    :func:`~newton._src.solvers.feather_pgs.kernels.pgs_solve_mf_loop`.
    """
    MF_MAX_C = mf_max_constraints
    MAX_BODIES = max_mf_bodies

    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int MF_MAX_C = {MF_MAX_C};
    const int MAX_BODIES = {MAX_BODIES};

    int lane = threadIdx.x;

    int m = mf_constraint_count.data[world];
    if (m == 0) return;
    if (m > MF_MAX_C) m = MF_MAX_C;

    int n_bodies = mf_body_count.data[world];
    if (n_bodies > MAX_BODIES) n_bodies = MAX_BODIES;

    // ═══════════════════════════════════════════════════════════════
    // SHARED MEMORY
    // ═══════════════════════════════════════════════════════════════
    __shared__ float s_vel[{MAX_BODIES * 6}];
    __shared__ float s_impulse[{MF_MAX_C}];
    __shared__ int s_dof_start[{MAX_BODIES}];

    int body_off = world * MAX_BODIES;
    int c_off = world * MF_MAX_C;

    // ═══════════════════════════════════════════════════════════════
    // LOAD PHASE
    // ═══════════════════════════════════════════════════════════════

    // Load body DOF starts and velocities
    for (int b = lane; b < n_bodies; b += 32) {{
        int dof = mf_body_dof_start.data[body_off + b];
        s_dof_start[b] = dof;
        for (int k = 0; k < 6; k++) {{
            s_vel[b * 6 + k] = v_out.data[dof + k];
        }}
    }}

    // Load impulses
    for (int i = lane; i < m; i += 32) {{
        s_impulse[i] = mf_impulses.data[c_off + i];
    }}
    __syncwarp();

    // ═══════════════════════════════════════════════════════════════
    // SOLVE PHASE (lane 0)
    // ═══════════════════════════════════════════════════════════════

    if (lane == 0) {{
        for (int iter = 0; iter < iterations; iter++) {{
            for (int i = 0; i < m; i++) {{
                int row_type = mf_row_type.data[c_off + i];

                float eff_inv = mf_eff_mass_inv.data[c_off + i];
                if (eff_inv <= 0.0f && row_type != 2) continue;

                int lba = mf_local_body_a.data[c_off + i];
                int lbb = mf_local_body_b.data[c_off + i];

                // Load J from global memory
                int j_base = (c_off + i) * 6;
                float ja0 = mf_J_a.data[j_base + 0];
                float ja1 = mf_J_a.data[j_base + 1];
                float ja2 = mf_J_a.data[j_base + 2];
                float ja3 = mf_J_a.data[j_base + 3];
                float ja4 = mf_J_a.data[j_base + 4];
                float ja5 = mf_J_a.data[j_base + 5];

                float jb0 = mf_J_b.data[j_base + 0];
                float jb1 = mf_J_b.data[j_base + 1];
                float jb2 = mf_J_b.data[j_base + 2];
                float jb3 = mf_J_b.data[j_base + 3];
                float jb4 = mf_J_b.data[j_base + 4];
                float jb5 = mf_J_b.data[j_base + 5];

                // Compute J * v from shared memory
                float jv = 0.0f;
                if (lba >= 0) {{
                    int va = lba * 6;
                    jv += ja0 * s_vel[va] + ja1 * s_vel[va+1] + ja2 * s_vel[va+2]
                        + ja3 * s_vel[va+3] + ja4 * s_vel[va+4] + ja5 * s_vel[va+5];
                }}
                if (lbb >= 0) {{
                    int vb = lbb * 6;
                    jv += jb0 * s_vel[vb] + jb1 * s_vel[vb+1] + jb2 * s_vel[vb+2]
                        + jb3 * s_vel[vb+3] + jb4 * s_vel[vb+4] + jb5 * s_vel[vb+5];
                }}

                // PGS update
                float rhs_i = mf_rhs.data[c_off + i];
                float old_impulse = s_impulse[i];
                float delta = -(jv + rhs_i) * eff_inv;
                float new_impulse = old_impulse + omega * delta;
                float delta_impulse = 0.0f;

                if (row_type == {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) {{
                    // Stateless velocity limit: apply only the impulse of the current overshoot.
                    new_impulse = jv + rhs_i < 0.0f ? delta : 0.0f;
                    delta_impulse = new_impulse;
                }}
                // Project: contact
                else if (row_type == 0) {{
                    if (new_impulse < 0.0f) new_impulse = 0.0f;
                }}
                // Project: friction
                else if (row_type == 2) {{
                    int parent_idx = mf_row_parent.data[c_off + i];
                    float radius = fmaxf(mf_row_mu.data[c_off + i] * s_impulse[parent_idx], 0.0f);

                    if (i != parent_idx + 1) {{
                        new_impulse = old_impulse;
                    }} else {{
                        int sib = parent_idx + 2;
                        int sib_base = (c_off + sib) * 6;
                        float sibling_residual = mf_rhs.data[c_off + sib];
                        for (int k = 0; k < 6; ++k) {{
                            if (lba >= 0) sibling_residual += mf_J_a.data[sib_base + k] * s_vel[lba * 6 + k];
                            if (lbb >= 0) sibling_residual += mf_J_b.data[sib_base + k] * s_vel[lbb * 6 + k];
                        }}
                        float inv_sib = mf_eff_mass_inv.data[c_off + sib];
                        float cross = 0.0f;
                        for (int k = 0; k < 6; ++k) {{
                            if (lba >= 0) cross += mf_J_a.data[j_base + k] * mf_MiJt_a.data[sib_base + k];
                            if (lbb >= 0) cross += mf_J_b.data[j_base + k] * mf_MiJt_b.data[sib_base + k];
                        }}
                        float2 pair = friction_pair_candidate(eff_inv > 0.0f ? 1.0f / eff_inv : 0.0f, cross, inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f,
                            jv + rhs_i, sibling_residual, old_impulse, s_impulse[sib], radius, omega);
                        float a_val = pair.x;
                        float b_val = pair.y;
                        float mag = sqrtf(a_val * a_val + b_val * b_val);
                        float scale = mag > radius ? radius / mag : 1.0f;
                        new_impulse = a_val * scale;
                        float sib_new = b_val * scale;
                        float sib_delta = sib_new - s_impulse[sib];
                        s_impulse[sib] = sib_new;

                        // Apply sibling correction to body velocities
                        int sib_lba = mf_local_body_a.data[c_off + sib];
                        int sib_lbb = mf_local_body_b.data[c_off + sib];
                        int sib_j_base = (c_off + sib) * 6;
                        if (sib_lba >= 0) {{
                            int sva = sib_lba * 6;
                            s_vel[sva+0] += mf_MiJt_a.data[sib_j_base+0] * sib_delta;
                            s_vel[sva+1] += mf_MiJt_a.data[sib_j_base+1] * sib_delta;
                            s_vel[sva+2] += mf_MiJt_a.data[sib_j_base+2] * sib_delta;
                            s_vel[sva+3] += mf_MiJt_a.data[sib_j_base+3] * sib_delta;
                            s_vel[sva+4] += mf_MiJt_a.data[sib_j_base+4] * sib_delta;
                            s_vel[sva+5] += mf_MiJt_a.data[sib_j_base+5] * sib_delta;
                        }}
                        if (sib_lbb >= 0) {{
                            int svb = sib_lbb * 6;
                            s_vel[svb+0] += mf_MiJt_b.data[sib_j_base+0] * sib_delta;
                            s_vel[svb+1] += mf_MiJt_b.data[sib_j_base+1] * sib_delta;
                            s_vel[svb+2] += mf_MiJt_b.data[sib_j_base+2] * sib_delta;
                            s_vel[svb+3] += mf_MiJt_b.data[sib_j_base+3] * sib_delta;
                            s_vel[svb+4] += mf_MiJt_b.data[sib_j_base+4] * sib_delta;
                            s_vel[svb+5] += mf_MiJt_b.data[sib_j_base+5] * sib_delta;
                        }}
                    }}
                }}

                if (row_type != {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) delta_impulse = new_impulse - old_impulse;
                s_impulse[i] = new_impulse;

                // Apply velocity correction: v += MiJt * delta_impulse
                int mijt_base = (c_off + i) * 6;
                if (lba >= 0) {{
                    int va = lba * 6;
                    s_vel[va+0] += mf_MiJt_a.data[mijt_base+0] * delta_impulse;
                    s_vel[va+1] += mf_MiJt_a.data[mijt_base+1] * delta_impulse;
                    s_vel[va+2] += mf_MiJt_a.data[mijt_base+2] * delta_impulse;
                    s_vel[va+3] += mf_MiJt_a.data[mijt_base+3] * delta_impulse;
                    s_vel[va+4] += mf_MiJt_a.data[mijt_base+4] * delta_impulse;
                    s_vel[va+5] += mf_MiJt_a.data[mijt_base+5] * delta_impulse;
                }}
                if (lbb >= 0) {{
                    int vb = lbb * 6;
                    s_vel[vb+0] += mf_MiJt_b.data[mijt_base+0] * delta_impulse;
                    s_vel[vb+1] += mf_MiJt_b.data[mijt_base+1] * delta_impulse;
                    s_vel[vb+2] += mf_MiJt_b.data[mijt_base+2] * delta_impulse;
                    s_vel[vb+3] += mf_MiJt_b.data[mijt_base+3] * delta_impulse;
                    s_vel[vb+4] += mf_MiJt_b.data[mijt_base+4] * delta_impulse;
                    s_vel[vb+5] += mf_MiJt_b.data[mijt_base+5] * delta_impulse;
                }}
            }}
        }}
    }}
    __syncwarp();

    // ═══════════════════════════════════════════════════════════════
    // STORE PHASE
    // ═══════════════════════════════════════════════════════════════

    // Write body velocities back to v_out
    for (int b = lane; b < n_bodies; b += 32) {{
        int dof = s_dof_start[b];
        for (int k = 0; k < 6; k++) {{
            v_out.data[dof + k] = s_vel[b * 6 + k];
        }}
    }}

    // Write impulses back
    for (int i = lane; i < m; i += 32) {{
        mf_impulses.data[c_off + i] = s_impulse[i];
    }}
#endif
"""

    snippet = snippet.replace("#if defined(__CUDA_ARCH__)", "#if defined(__CUDA_ARCH__)\n" + FRICTION_PAIR_CUDA, 1)

    @wp.func_native(snippet)
    def pgs_solve_mf_native(
        world: int,
        mf_constraint_count: wp.array[int],
        mf_body_count: wp.array[int],
        mf_body_dof_start: wp.array2d[int],
        mf_local_body_a: wp.array2d[int],
        mf_local_body_b: wp.array2d[int],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_eff_mass_inv: wp.array2d[float],
        mf_rhs: wp.array2d[float],
        mf_row_type: wp.array2d[int],
        mf_row_parent: wp.array2d[int],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        mf_impulses: wp.array2d[float],
        v_out: wp.array[float],
    ): ...

    def pgs_solve_mf_template(
        mf_constraint_count: wp.array[int],
        mf_body_count: wp.array[int],
        mf_body_dof_start: wp.array2d[int],
        mf_local_body_a: wp.array2d[int],
        mf_local_body_b: wp.array2d[int],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_eff_mass_inv: wp.array2d[float],
        mf_rhs: wp.array2d[float],
        mf_row_type: wp.array2d[int],
        mf_row_parent: wp.array2d[int],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        mf_impulses: wp.array2d[float],
        v_out: wp.array[float],
    ):
        world, _lane = wp.tid()
        pgs_solve_mf_native(
            world,
            mf_constraint_count,
            mf_body_count,
            mf_body_dof_start,
            mf_local_body_a,
            mf_local_body_b,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_eff_mass_inv,
            mf_rhs,
            mf_row_type,
            mf_row_parent,
            mf_row_mu,
            iterations,
            omega,
            mf_impulses,
            v_out,
        )

    name = f"pgs_solve_mf_{mf_max_constraints}_{max_mf_bodies}"
    pgs_solve_mf_template.__name__ = name
    pgs_solve_mf_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_template)


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


def _synchronize_streams(streams: list) -> None:
    """Wait for the work queued on ``streams``."""
    for stream in streams:
        wp.synchronize_stream(stream)


# Contact updates of the non-default friction modes, spliced into the matrix-free kernel.
_DENSE_BISECTION_FRICTION_SOURCE = """
                int parent_idx = (s_meta_dense[i] >> __DENSE_META_ROW_TYPE_BITS__) - 1;
                int i_t1 = parent_idx + 1;
                int i_t2 = parent_idx + 2;

                if (i != i_t1) {
                    new_impulse = s_lam_dense[i];
                } else {
                    int n_row_base = jy_world_base + parent_idx * __D__;
                    int t1_row_base = jy_world_base + i_t1 * __D__;
                    int t2_row_base = jy_world_base + i_t2 * __D__;

                    float target_vel_n = -s_rhs_dense[parent_idx];
                    float mu = s_mu_dense[i];
                    float old_lambda_n = s_lam_dense[parent_idx];
                    float old_lambda_t1 = s_lam_dense[i_t1];
                    float old_lambda_t2 = s_lam_dense[i_t2];

                    float new_lambda_n = old_lambda_n;
                    float new_lambda_t1 = old_lambda_t1;
                    float new_lambda_t2 = old_lambda_t2;
                    float d_n_total = 0.0f;
                    float d_t2_total = 0.0f;

                    if (lane == 0) {
                        float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                        float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                        float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;

                        for (int k = 0; k < __D__; k++) {
                            float vk = s_v[k];
                            float Jn = J_world.data[n_row_base + k];
                            float Jt1 = J_world.data[t1_row_base + k];
                            float Jt2 = J_world.data[t2_row_base + k];
                            float Yn = Y_world.data[n_row_base + k];
                            float Yt1 = Y_world.data[t1_row_base + k];
                            float Yt2 = Y_world.data[t2_row_base + k];

                            u_n += Jn * vk;
                            u_t1 += Jt1 * vk;
                            u_t2 += Jt2 * vk;

                            G_nn += Jn * Yn;
                            G_nt1 += Jn * Yt1;
                            G_nt2 += Jn * Yt2;
                            G_t1t1 += Jt1 * Yt1;
                            G_t1t2 += Jt1 * Yt2;
                            G_t2t2 += Jt2 * Yt2;
                        }

                        // __DENSE_DESAXCE_BIAS__

                        if (G_nn >= 1.0e-20f) {
                            float u_n_at_zero = u_n + G_nn * (0.0f - old_lambda_n);
                            if (u_n_at_zero >= target_vel_n) {
                                new_lambda_n = 0.0f;
                                new_lambda_t1 = 0.0f;
                                new_lambda_t2 = 0.0f;
                            } else {
                                float lo = 0.0f;
                                float hi = fmaxf(
                                    old_lambda_n * 2.0f,
                                    (target_vel_n - u_n) / G_nn + old_lambda_n);
                                hi = fmaxf(hi, 1.0f);

                                for (int _bi = 0; _bi < 20; _bi++) {
                                    float mid = 0.5f * (lo + hi);
                                    float d_n = mid - old_lambda_n;
                                    float ut1_eff = u_t1 + G_nt1 * d_n;
                                    float ut2_eff = u_t2 + G_nt2 * d_n;

                                    float det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                    float d_t1 = 0.0f, d_t2 = 0.0f;
                                    if (fabsf(det) > 1.0e-20f) {
                                        d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det;
                                        d_t2 = ( ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det;
                                    }

                                    float trial_t1 = old_lambda_t1 + d_t1;
                                    float trial_t2 = old_lambda_t2 + d_t2;
                                    float flimit = mu * mid;
                                    float tmag = sqrtf(trial_t1 * trial_t1 + trial_t2 * trial_t2);
                                    if (tmag > flimit && tmag > 1.0e-20f) {
                                        float sc = flimit / tmag;
                                        trial_t1 *= sc;
                                        trial_t2 *= sc;
                                    }

                                    float d_t1_actual = trial_t1 - old_lambda_t1;
                                    float d_t2_actual = trial_t2 - old_lambda_t2;
                                    float u_n_trial = u_n + G_nn * d_n
                                        + G_nt1 * d_t1_actual
                                        + G_nt2 * d_t2_actual;
                                    if (u_n_trial < target_vel_n) lo = mid;
                                    else hi = mid;
                                }

                                new_lambda_n = 0.5f * (lo + hi);
                                float d_n_final = new_lambda_n - old_lambda_n;
                                float ut1_f = u_t1 + G_nt1 * d_n_final;
                                float ut2_f = u_t2 + G_nt2 * d_n_final;
                                float det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                float d_t1_f = 0.0f, d_t2_f = 0.0f;
                                if (fabsf(det_f) > 1.0e-20f) {
                                    d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f;
                                    d_t2_f = ( ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f;
                                }

                                new_lambda_t1 = old_lambda_t1 + d_t1_f;
                                new_lambda_t2 = old_lambda_t2 + d_t2_f;
                                float flimit_f = mu * new_lambda_n;
                                float tmag_f = sqrtf(new_lambda_t1 * new_lambda_t1 + new_lambda_t2 * new_lambda_t2);
                                if (tmag_f > flimit_f && tmag_f > 1.0e-20f) {
                                    float sc_f = flimit_f / tmag_f;
                                    new_lambda_t1 *= sc_f;
                                    new_lambda_t2 *= sc_f;
                                }
                            }

                            d_n_total = new_lambda_n - old_lambda_n;
                            d_t2_total = new_lambda_t2 - old_lambda_t2;
                            s_lam_dense[parent_idx] = new_lambda_n;
                            s_lam_dense[i_t2] = new_lambda_t2;
                        }
                    }

                    __syncwarp();
                    new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                    d_n_total = __shfl_sync(MASK, d_n_total, 0);
                    d_t2_total = __shfl_sync(MASK, d_t2_total, 0);
                    if (d_n_total != 0.0f || d_t2_total != 0.0f) iteration_changed = 1;

                    if (d_n_total != 0.0f) {
                        for (int d = lane; d < __D__; d += 32) {
                            s_v[d] += Y_world.data[n_row_base + d] * d_n_total;
                        }
                    }
                    if (d_t2_total != 0.0f) {
                        for (int d = lane; d < __D__; d += 32) {
                            s_v[d] += Y_world.data[t2_row_base + d] * d_t2_total;
                        }
                    }
                    __syncwarp();
                    new_impulse = new_lambda_t1;
                }
"""
_MF_BISECTION_FRICTION_SOURCE = """
                int i_t1 = mf_par + 1;
                int i_t2 = mf_par + 2;

                if (i != i_t1) {
                    new_impulse = s_lam_mf[i];
                } else {
                    int n_mf6 = mf6_base + mf_par * 6;
                    int t1_mf6 = mf6_base + i_t1 * 6;
                    int t2_mf6 = mf6_base + i_t2 * 6;

                    int parent_packed_dofs = mf_meta.data[off_meta + mf_par * 4];
                    int dof_a_par = parent_packed_dofs >> 16;
                    int dof_b_par = (parent_packed_dofs << 16) >> 16;
                    float target_vel_n = -__int_as_float(
                        mf_meta.data[off_meta + mf_par * 4 + 2]);
                    float mu = mf_row_mu.data[off_mf + i];

                    float old_lambda_n = s_lam_mf[mf_par];
                    float old_lambda_t1 = s_lam_mf[i_t1];
                    float old_lambda_t2 = s_lam_mf[i_t2];

                    float new_lambda_n = old_lambda_n;
                    float new_lambda_t1 = old_lambda_t1;
                    float new_lambda_t2 = old_lambda_t2;
                    float d_n_total = 0.0f;
                    float d_t2_total = 0.0f;

                    if (lane == 0) {
                        float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float va = s_v[dof_a_par + k];
                                u_n  += mf_J_a.data[n_mf6  + k] * va;
                                u_t1 += mf_J_a.data[t1_mf6 + k] * va;
                                u_t2 += mf_J_a.data[t2_mf6 + k] * va;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float vb = s_v[dof_b_par + k];
                                u_n  += mf_J_b.data[n_mf6  + k] * vb;
                                u_t1 += mf_J_b.data[t1_mf6 + k] * vb;
                                u_t2 += mf_J_b.data[t2_mf6 + k] * vb;
                            }
                        }

                        // __DESAXCE_BIAS__

                        float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                        float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jna = mf_J_a.data[n_mf6  + k];
                                float Jt1a = mf_J_a.data[t1_mf6 + k];
                                float Jt2a = mf_J_a.data[t2_mf6 + k];
                                float Mna = mf_MiJt_a.data[n_mf6  + k];
                                float Mt1a = mf_MiJt_a.data[t1_mf6 + k];
                                float Mt2a = mf_MiJt_a.data[t2_mf6 + k];
                                G_nn   += Jna * Mna;
                                G_nt1  += Jna * Mt1a;
                                G_nt2  += Jna * Mt2a;
                                G_t1t1 += Jt1a * Mt1a;
                                G_t1t2 += Jt1a * Mt2a;
                                G_t2t2 += Jt2a * Mt2a;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jnb = mf_J_b.data[n_mf6  + k];
                                float Jt1b = mf_J_b.data[t1_mf6 + k];
                                float Jt2b = mf_J_b.data[t2_mf6 + k];
                                float Mnb = mf_MiJt_b.data[n_mf6  + k];
                                float Mt1b = mf_MiJt_b.data[t1_mf6 + k];
                                float Mt2b = mf_MiJt_b.data[t2_mf6 + k];
                                G_nn   += Jnb * Mnb;
                                G_nt1  += Jnb * Mt1b;
                                G_nt2  += Jnb * Mt2b;
                                G_t1t1 += Jt1b * Mt1b;
                                G_t1t2 += Jt1b * Mt2b;
                                G_t2t2 += Jt2b * Mt2b;
                            }
                        }

                        if (G_nn >= 1.0e-20f) {
                            float u_n_at_zero = u_n + G_nn * (0.0f - old_lambda_n);
                            if (u_n_at_zero >= target_vel_n) {
                                new_lambda_n = 0.0f;
                                new_lambda_t1 = 0.0f;
                                new_lambda_t2 = 0.0f;
                            } else {
                                float lo = 0.0f;
                                float hi = fmaxf(
                                    old_lambda_n * 2.0f,
                                    (target_vel_n - u_n) / G_nn + old_lambda_n);
                                hi = fmaxf(hi, 1.0f);

                                for (int _bi = 0; _bi < 20; _bi++) {
                                    float mid = 0.5f * (lo + hi);
                                    float d_n = mid - old_lambda_n;
                                    float ut1_eff = u_t1 + G_nt1 * d_n;
                                    float ut2_eff = u_t2 + G_nt2 * d_n;

                                    float det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                    float d_t1 = 0.0f, d_t2 = 0.0f;
                                    if (fabsf(det) > 1.0e-20f) {
                                        d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det;
                                        d_t2 = ( ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det;
                                    }
                                    float trial_t1 = old_lambda_t1 + d_t1;
                                    float trial_t2 = old_lambda_t2 + d_t2;

                                    float flimit = mu * mid;
                                    float tmag = sqrtf(
                                        trial_t1 * trial_t1 + trial_t2 * trial_t2);
                                    if (tmag > flimit && tmag > 1.0e-20f) {
                                        float sc = flimit / tmag;
                                        trial_t1 *= sc;
                                        trial_t2 *= sc;
                                    }
                                    float d_t1_actual = trial_t1 - old_lambda_t1;
                                    float d_t2_actual = trial_t2 - old_lambda_t2;
                                    float u_n_trial = u_n + G_nn * d_n
                                        + G_nt1 * d_t1_actual
                                        + G_nt2 * d_t2_actual;
                                    if (u_n_trial < target_vel_n) lo = mid;
                                    else hi = mid;
                                }
                                new_lambda_n = 0.5f * (lo + hi);

                                float d_n_final = new_lambda_n - old_lambda_n;
                                float ut1_f = u_t1 + G_nt1 * d_n_final;
                                float ut2_f = u_t2 + G_nt2 * d_n_final;
                                float det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                float d_t1_f = 0.0f, d_t2_f = 0.0f;
                                if (fabsf(det_f) > 1.0e-20f) {
                                    d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f;
                                    d_t2_f = ( ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f;
                                }
                                new_lambda_t1 = old_lambda_t1 + d_t1_f;
                                new_lambda_t2 = old_lambda_t2 + d_t2_f;

                                float flimit_f = mu * new_lambda_n;
                                float tmag_f = sqrtf(
                                    new_lambda_t1 * new_lambda_t1
                                    + new_lambda_t2 * new_lambda_t2);
                                if (tmag_f > flimit_f && tmag_f > 1.0e-20f) {
                                    float sc_f = flimit_f / tmag_f;
                                    new_lambda_t1 *= sc_f;
                                    new_lambda_t2 *= sc_f;
                                }
                            }
                            d_n_total = new_lambda_n - old_lambda_n;
                            d_t2_total = new_lambda_t2 - old_lambda_t2;
                            s_lam_mf[mf_par] = new_lambda_n;
                            s_lam_mf[i_t2]   = new_lambda_t2;
                        }
                    }

                    __syncwarp();
                    new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                    d_n_total = __shfl_sync(MASK, d_n_total, 0);
                    d_t2_total = __shfl_sync(MASK, d_t2_total, 0);
                    if (d_n_total != 0.0f || d_t2_total != 0.0f) iteration_changed = 1;

                    if (d_n_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[n_mf6 + lane] * d_n_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[n_mf6 + lane - 6] * d_n_total;
                        }
                    }
                    if (d_t2_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[t2_mf6 + lane] * d_t2_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[t2_mf6 + lane - 6] * d_t2_total;
                        }
                    }
                    __syncwarp();
                    new_impulse = new_lambda_t1;
                }
"""
# de Saxce maximum-dissipation bias: mu * |c_T| raises the normal target velocity.
_MF_DESAXCE_BIAS_SOURCE = """{
                                float c_T_mag = sqrtf(u_t1 * u_t1 + u_t2 * u_t2);
                                target_vel_n = target_vel_n + mu * c_T_mag;
                            }"""
_MF_COULOMB_NEWTON_FRICTION_SOURCE = """
                int i_t1 = mf_par + 1;
                int i_t2 = mf_par + 2;

                if (i != i_t1) {
                    new_impulse = s_lam_mf[i];
                } else {
                    int n_mf6 = mf6_base + mf_par * 6;
                    int t1_mf6 = mf6_base + i_t1 * 6;
                    int t2_mf6 = mf6_base + i_t2 * 6;

                    int parent_packed_dofs = mf_meta.data[off_meta + mf_par * 4];
                    int dof_a_par = parent_packed_dofs >> 16;
                    int dof_b_par = (parent_packed_dofs << 16) >> 16;
                    float target_vel_n = -__int_as_float(
                        mf_meta.data[off_meta + mf_par * 4 + 2]);
                    float mu = mf_row_mu.data[off_mf + i];

                    float old_lambda_n = s_lam_mf[mf_par];
                    float old_lambda_t1 = s_lam_mf[i_t1];
                    float old_lambda_t2 = s_lam_mf[i_t2];

                    float new_lambda_n = old_lambda_n;
                    float new_lambda_t1 = old_lambda_t1;
                    float new_lambda_t2 = old_lambda_t2;
                    float d_n_total = 0.0f;
                    float d_t2_total = 0.0f;

                    if (lane == 0) {
                        float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float va = s_v[dof_a_par + k];
                                u_n  += mf_J_a.data[n_mf6  + k] * va;
                                u_t1 += mf_J_a.data[t1_mf6 + k] * va;
                                u_t2 += mf_J_a.data[t2_mf6 + k] * va;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float vb = s_v[dof_b_par + k];
                                u_n  += mf_J_b.data[n_mf6  + k] * vb;
                                u_t1 += mf_J_b.data[t1_mf6 + k] * vb;
                                u_t2 += mf_J_b.data[t2_mf6 + k] * vb;
                            }
                        }

                        float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                        float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jna = mf_J_a.data[n_mf6  + k];
                                float Jt1a = mf_J_a.data[t1_mf6 + k];
                                float Jt2a = mf_J_a.data[t2_mf6 + k];
                                float Mna = mf_MiJt_a.data[n_mf6  + k];
                                float Mt1a = mf_MiJt_a.data[t1_mf6 + k];
                                float Mt2a = mf_MiJt_a.data[t2_mf6 + k];
                                G_nn   += Jna * Mna;
                                G_nt1  += Jna * Mt1a;
                                G_nt2  += Jna * Mt2a;
                                G_t1t1 += Jt1a * Mt1a;
                                G_t1t2 += Jt1a * Mt2a;
                                G_t2t2 += Jt2a * Mt2a;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jnb = mf_J_b.data[n_mf6  + k];
                                float Jt1b = mf_J_b.data[t1_mf6 + k];
                                float Jt2b = mf_J_b.data[t2_mf6 + k];
                                float Mnb = mf_MiJt_b.data[n_mf6  + k];
                                float Mt1b = mf_MiJt_b.data[t1_mf6 + k];
                                float Mt2b = mf_MiJt_b.data[t2_mf6 + k];
                                G_nn   += Jnb * Mnb;
                                G_nt1  += Jnb * Mt1b;
                                G_nt2  += Jnb * Mt2b;
                                G_t1t1 += Jt1b * Mt1b;
                                G_t1t2 += Jt1b * Mt2b;
                                G_t2t2 += Jt2b * Mt2b;
                            }
                        }

                        if (G_nn >= 1.0e-20f) {
                            float u_free_n = u_n - (G_nn * old_lambda_n
                                + G_nt1 * old_lambda_t1
                                + G_nt2 * old_lambda_t2);
                            float u_free_t1 = u_t1 - (G_nt1 * old_lambda_n
                                + G_t1t1 * old_lambda_t1
                                + G_t1t2 * old_lambda_t2);
                            float u_free_t2 = u_t2 - (G_nt2 * old_lambda_n
                                + G_t1t2 * old_lambda_t1
                                + G_t2t2 * old_lambda_t2);

                            if (u_free_n < target_vel_n) {
                                float bN = u_free_n - target_vel_n;
                                float bT0 = u_free_t1;
                                float bT1 = u_free_t2;

                                float WN = G_nn;
                                float wNT0 = G_nt1;
                                float wNT1 = G_nt2;
                                float AT00 = G_t1t1 - (wNT0 * wNT0) / WN;
                                float AT01 = G_t1t2 - (wNT0 * wNT1) / WN;
                                float AT10 = G_t1t2 - (wNT1 * wNT0) / WN;
                                float AT11 = G_t2t2 - (wNT1 * wNT1) / WN;
                                float cT0 = bT0 - (bN / WN) * wNT0;
                                float cT1 = bT1 - (bN / WN) * wNT1;

                                float inv_det0 = 1.0f /
                                    (AT00 * AT11 - AT01 * AT10);
                                float s0_0 = (AT11 * cT0 - AT01 * cT1) * inv_det0;
                                float s0_1 = (AT00 * cT1 - AT10 * cT0) * inv_det0;
                                float norm_s0 = sqrtf(s0_0 * s0_0 + s0_1 * s0_1);
                                float phi0 = norm_s0 - mu *
                                    (wNT0 * s0_0 + wNT1 * s0_1 - bN) / WN;

                                float last_s0 = 0.0f, last_s1 = 0.0f;
                                float alpha = 0.0f;

                                if (phi0 <= 0.0f) {
                                    last_s0 = s0_0;
                                    last_s1 = s0_1;
                                    alpha = 0.0f;
                                } else {
                                    float hi = 1.0f;
                                    for (int _e = 0; _e < 30; _e++) {
                                        float a_hi = AT00 + hi;
                                        float d_hi = AT11 + hi;
                                        float idet_hi = 1.0f /
                                            (a_hi * d_hi - AT01 * AT10);
                                        float sh0 = (d_hi * cT0 - AT01 * cT1) * idet_hi;
                                        float sh1 = (a_hi * cT1 - AT10 * cT0) * idet_hi;
                                        float ns_hi = sqrtf(sh0 * sh0 + sh1 * sh1);
                                        float phi_hi = ns_hi - mu *
                                            (wNT0 * sh0 + wNT1 * sh1 - bN) / WN;
                                        if (phi_hi < 0.0f) break;
                                        hi *= 2.0f;
                                    }

                                    float lo = 0.0f;
                                    float x = 0.5f * (lo + hi);
                                    float tol = 1.0e-6f + 1.0e-6f * phi0;

                                    for (int _it = 0; _it < 50; _it++) {
                                        float ax = AT00 + x;
                                        float dx = AT11 + x;
                                        float idet = 1.0f /
                                            (ax * dx - AT01 * AT10);
                                        float s0 = (dx * cT0 - AT01 * cT1) * idet;
                                        float s1 = (ax * cT1 - AT10 * cT0) * idet;
                                        float t0 = (dx * s0 - AT01 * s1) * idet;
                                        float t1 = (ax * s1 - AT10 * s0) * idet;
                                        float ns = sqrtf(s0 * s0 + s1 * s1);
                                        float fx = ns - mu *
                                            (wNT0 * s0 + wNT1 * s1 - bN) / WN;
                                        float dfx = -(s0 * t0 + s1 * t1) / ns
                                            + mu * (wNT0 * t0 + wNT1 * t1) / WN;

                                        last_s0 = s0;
                                        last_s1 = s1;

                                        if (fabsf(fx) < tol
                                            || fabsf(hi - lo) < 1.0e-6f * (1.0f + hi)) {
                                            alpha = x;
                                            break;
                                        }

                                        if (fx > 0.0f) lo = x;
                                        else hi = x;

                                        float x_new = 0.5f * (lo + hi);
                                        if (dfx != 0.0f) {
                                            float x_newton = x - fx / dfx;
                                            if (x_newton > lo && x_newton < hi) {
                                                x_new = x_newton;
                                            }
                                        }
                                        x = x_new;
                                        alpha = x;
                                    }
                                }

                                float rT0 = -last_s0;
                                float rT1 = -last_s1;
                                float rN = -(wNT0 * rT0 + wNT1 * rT1 + bN) / WN;
                                new_lambda_n = rN;
                                new_lambda_t1 = rT0;
                                new_lambda_t2 = rT1;
                            } else {
                                new_lambda_n = 0.0f;
                                new_lambda_t1 = 0.0f;
                                new_lambda_t2 = 0.0f;
                            }

                            d_n_total = new_lambda_n - old_lambda_n;
                            d_t2_total = new_lambda_t2 - old_lambda_t2;
                            s_lam_mf[mf_par] = new_lambda_n;
                            s_lam_mf[i_t2]   = new_lambda_t2;
                        }
                    }

                    __syncwarp();
                    new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                    d_n_total = __shfl_sync(MASK, d_n_total, 0);
                    d_t2_total = __shfl_sync(MASK, d_t2_total, 0);
                    if (d_n_total != 0.0f || d_t2_total != 0.0f) iteration_changed = 1;

                    if (d_n_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[n_mf6 + lane] * d_n_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[n_mf6 + lane - 6] * d_n_total;
                        }
                    }
                    if (d_t2_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[t2_mf6 + lane] * d_t2_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[t2_mf6 + lane - 6] * d_t2_total;
                        }
                    }
                    __syncwarp();
                    new_impulse = new_lambda_t1;
                }
"""
