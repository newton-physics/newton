# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-level fixed-joint merger utility.

Produces an effective-index layer that routes shapes, joints, and body forces
onto surviving bodies without modifying the :class:`~newton.Model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..sim import BodyFlags, JointType
from ..sim.model import Model


@dataclass
class FixedJointMergeInfo:
    """Solver-level merge metadata for collapsed FIXED joints."""

    has_merges: bool
    survivor_of: list[int]
    relative_xform_of: list[wp.transform]
    survivor_indices_gpu: wp.array[wp.int32]
    relative_xforms_gpu: wp.array[wp.transform]
    merged_body_inv_mass_gpu: wp.array[float]
    merged_body_inv_inertia_gpu: wp.array[wp.mat33]
    merged_body_mass_gpu: wp.array[float]
    merged_body_inertia_gpu: wp.array[wp.mat33]
    merged_body_com_gpu: wp.array[wp.vec3]
    joint_enabled_effective_gpu: wp.array[wp.bool]
    joint_parent_effective_gpu: wp.array[wp.int32]
    joint_child_effective_gpu: wp.array[wp.int32]
    joint_X_p_effective_gpu: wp.array[wp.transform]
    joint_X_c_effective_gpu: wp.array[wp.transform]
    shape_body_effective_gpu: wp.array[wp.int32]


def compute_fixed_joint_merge(
    model: Model,
    joints_to_keep: list[str] | None = None,
) -> FixedJointMergeInfo | None:
    """Compute solver-level fixed-joint merge metadata from a finalized model.

    Args:
        model: The finalized model to analyse. Never modified.
        joints_to_keep: Joint labels to exempt from collapsing.

    Returns:
        Merge metadata, or ``None`` when no FIXED joints are collapsed.
    """
    if joints_to_keep is None:
        joints_to_keep = []

    body_count = model.body_count
    joint_count = model.joint_count

    if body_count == 0 or joint_count == 0:
        return None

    joint_type_np = model.joint_type.numpy()
    joint_parent_np = model.joint_parent.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_enabled_np = model.joint_enabled.numpy()
    joint_X_p_np = model.joint_X_p.numpy()
    joint_X_c_np = model.joint_X_c.numpy()
    body_inv_mass_np = model.body_inv_mass.numpy()
    body_inertia_np = model.body_inertia.numpy()
    body_com_np = model.body_com.numpy()
    body_mass_np = model.body_mass.numpy()
    body_flags_np = model.body_flags.numpy()

    def _is_kinematic(b: int) -> bool:
        return b >= 0 and bool(int(body_flags_np[b]) & int(BodyFlags.KINEMATIC))

    # Bodies referenced by equality constraints — never merge.
    bodies_in_constraints: set[int] = set()
    if model.equality_constraint_count > 0 and model.equality_constraint_body1 is not None:
        b1 = model.equality_constraint_body1.numpy()
        b2 = model.equality_constraint_body2.numpy()
        bodies_in_constraints.update(int(x) for x in b1 if x >= 0)
        bodies_in_constraints.update(int(x) for x in b2 if x >= 0)

    body_children: dict[int, list[int]] = {-1: []}
    joint_of: dict[tuple[int, int], int] = {}
    for i in range(body_count):
        body_children[i] = []
    for j in range(joint_count):
        p = int(joint_parent_np[j])
        c = int(joint_child_np[j])
        body_children[p].append(c)
        joint_of[(p, c)] = j

    for children in body_children.values():
        children.sort()

    survivor_of: list[int] = list(range(body_count))
    relative_xform_of: list[wp.transform] = [wp.transform() for _ in range(body_count)]

    merged_mass: list[float] = [float(body_mass_np[b]) for b in range(body_count)]
    merged_inertia: list[wp.mat33] = [wp.mat33(*body_inertia_np[b].flatten()) for b in range(body_count)]
    merged_inv_mass: list[float] = [float(body_inv_mass_np[b]) for b in range(body_count)]
    merged_inv_inertia: list[wp.mat33] = [
        _safe_inv_mat33(wp.mat33(*body_inertia_np[b].flatten()), float(body_mass_np[b])) for b in range(body_count)
    ]
    merged_com: list[wp.vec3] = [wp.vec3(*body_com_np[b]) for b in range(body_count)]

    # Inherit model.joint_enabled so user-disabled joints stay disabled.
    joint_enabled_eff: list[bool] = [bool(joint_enabled_np[j]) for j in range(joint_count)]

    visited: list[bool] = [False] * body_count

    def dfs(parent_body: int, child_body: int, incoming_xform: wp.transform, last_dynamic_body: int) -> None:
        j = joint_of[(parent_body, child_body)]
        jtype = int(joint_type_np[j])
        joint_is_enabled = bool(joint_enabled_np[j])
        # Equality constraints index by body and aren't remapped; never absorb.
        should_skip_merge = child_body in bodies_in_constraints
        # Kinematic survivors are treated like world: don't absorb dynamic bodies into them.
        survivor_is_kinematic = _is_kinematic(last_dynamic_body)
        joint_in_keep_list = model.joint_label[j] in joints_to_keep

        if (
            jtype == JointType.FIXED
            and joint_is_enabled
            and not should_skip_merge
            and not survivor_is_kinematic
            and not joint_in_keep_list
        ):
            X_p = _np_row_to_transform(joint_X_p_np[j])
            X_c = _np_row_to_transform(joint_X_c_np[j])
            joint_xform = X_p * wp.transform_inverse(X_c)
            incoming_xform = incoming_xform * joint_xform

            joint_enabled_eff[j] = False

            if last_dynamic_body >= 0:
                survivor_of[child_body] = survivor_of[last_dynamic_body]
                relative_xform_of[child_body] = incoming_xform

                sv = survivor_of[last_dynamic_body]
                new_m, new_com, new_I = _accumulate_child_into_survivor(
                    merged_mass[sv],
                    merged_com[sv],
                    merged_inertia[sv],
                    float(body_mass_np[child_body]),
                    wp.vec3(*body_com_np[child_body]),
                    wp.mat33(*body_inertia_np[child_body].flatten()),
                    incoming_xform,
                )
                merged_mass[sv] = new_m
                merged_com[sv] = new_com
                merged_inertia[sv] = new_I
                if new_m > 0.0:
                    merged_inv_mass[sv] = 1.0 / new_m
                    merged_inv_inertia[sv] = _safe_inv_mat33(new_I, new_m)
                merged_inv_mass[child_body] = 0.0
                merged_inv_inertia[child_body] = wp.mat33(0.0)
        else:
            last_dynamic_body = child_body
            incoming_xform = wp.transform()

        visited[child_body] = True
        for next_child in body_children.get(child_body, []):
            if not visited[next_child]:
                dfs(child_body, next_child, incoming_xform, last_dynamic_body)

    for root in body_children[-1]:
        if not visited[root]:
            dfs(-1, root, wp.transform(), -1)

    # Disconnected subtrees that don't reach world.
    children_in_joints = {c for p, cs in body_children.items() if p >= 0 for c in cs}
    for b in range(body_count):
        if visited[b]:
            continue
        if b in children_in_joints:
            continue
        visited[b] = True
        for child in body_children.get(b, []):
            if not visited[child]:
                dfs(b, child, wp.transform(), b)

    has_merges = any(survivor_of[b] != b for b in range(body_count))
    if not has_merges:
        return None

    # Effective-index layer: anchors pre-multiplied so they land in the survivor's frame.
    joint_parent_eff: list[int] = [int(joint_parent_np[j]) for j in range(joint_count)]
    joint_child_eff: list[int] = [int(joint_child_np[j]) for j in range(joint_count)]
    joint_X_p_eff: list[wp.transform] = [_np_row_to_transform(joint_X_p_np[j]) for j in range(joint_count)]
    joint_X_c_eff: list[wp.transform] = [_np_row_to_transform(joint_X_c_np[j]) for j in range(joint_count)]
    for j in range(joint_count):
        p = joint_parent_eff[j]
        c = joint_child_eff[j]
        if p >= 0 and survivor_of[p] != p:
            joint_X_p_eff[j] = relative_xform_of[p] * joint_X_p_eff[j]
            joint_parent_eff[j] = survivor_of[p]
        if c >= 0 and survivor_of[c] != c:
            joint_X_c_eff[j] = relative_xform_of[c] * joint_X_c_eff[j]
            joint_child_eff[j] = survivor_of[c]

    shape_count = model.shape_count
    if shape_count > 0 and model.shape_body is not None:
        shape_body_np = model.shape_body.numpy()
        shape_body_eff: list[int] = [int(b) for b in shape_body_np]
        for s in range(shape_count):
            b = shape_body_eff[s]
            if b >= 0 and survivor_of[b] != b:
                shape_body_eff[s] = survivor_of[b]
    else:
        shape_body_eff = []

    dev = model.device
    return FixedJointMergeInfo(
        has_merges=has_merges,
        survivor_of=survivor_of,
        relative_xform_of=relative_xform_of,
        survivor_indices_gpu=wp.array(survivor_of, dtype=wp.int32, device=dev),
        relative_xforms_gpu=wp.array(relative_xform_of, dtype=wp.transform, device=dev),
        merged_body_inv_mass_gpu=wp.array(merged_inv_mass, dtype=wp.float32, device=dev),
        merged_body_inv_inertia_gpu=wp.array(merged_inv_inertia, dtype=wp.mat33, device=dev),
        merged_body_mass_gpu=wp.array(merged_mass, dtype=wp.float32, device=dev),
        merged_body_inertia_gpu=wp.array(merged_inertia, dtype=wp.mat33, device=dev),
        merged_body_com_gpu=wp.array(merged_com, dtype=wp.vec3, device=dev),
        joint_enabled_effective_gpu=wp.array(joint_enabled_eff, dtype=wp.bool, device=dev),
        joint_parent_effective_gpu=wp.array(joint_parent_eff, dtype=wp.int32, device=dev),
        joint_child_effective_gpu=wp.array(joint_child_eff, dtype=wp.int32, device=dev),
        joint_X_p_effective_gpu=wp.array(joint_X_p_eff, dtype=wp.transform, device=dev),
        joint_X_c_effective_gpu=wp.array(joint_X_c_eff, dtype=wp.transform, device=dev),
        shape_body_effective_gpu=(
            wp.array(shape_body_eff, dtype=wp.int32, device=dev)
            if shape_body_eff
            else wp.empty(0, dtype=wp.int32, device=dev)
        ),
    )


def _np_row_to_transform(row) -> wp.transform:
    """Reconstruct a wp.transform from a 7-element float32 numpy row [p, q]."""
    return wp.transform(
        wp.vec3(float(row[0]), float(row[1]), float(row[2])),
        wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )


def _safe_inv_mat33(m: wp.mat33, mass: float) -> wp.mat33:
    """Return the inverse of m, or a zero matrix when mass is zero."""
    if mass <= 0.0:
        return wp.mat33(0.0)
    return wp.inverse(m)


_I3 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _shift_inertia(inertia: wp.mat33, mass: float, offset: wp.vec3) -> wp.mat33:
    """Shift an inertia tensor by ``offset`` via the parallel-axis theorem."""
    return inertia + mass * (wp.dot(offset, offset) * _I3 - wp.outer(offset, offset))


def _accumulate_child_into_survivor(
    sv_m: float,
    sv_com: wp.vec3,
    sv_I: wp.mat33,
    child_m: float,
    child_com_local: wp.vec3,
    child_I_local: wp.mat33,
    incoming_xform: wp.transform,
):
    """Fuse a child body into the survivor.

    Returns ``(mass, com, inertia)`` in the survivor's body frame, with
    ``inertia`` parallel-axis-shifted to be expressed about the new combined
    COM (Newton's convention for ``body_inertia``).
    """
    com_c_in_sv = wp.transform_point(incoming_xform, child_com_local)
    new_m = sv_m + child_m
    if new_m <= 0.0:
        return new_m, sv_com, sv_I
    new_com = (sv_m * sv_com + child_m * com_c_in_sv) * (1.0 / new_m)

    # Shift both sides to the new combined COM so the sum is consistent.
    sv_I_at_new = _shift_inertia(sv_I, sv_m, sv_com - new_com)
    R = wp.quat_to_matrix(incoming_xform.q)
    child_I_in_sv = R @ child_I_local @ wp.transpose(R)
    child_I_at_new = _shift_inertia(child_I_in_sv, child_m, com_c_in_sv - new_com)

    return new_m, new_com, sv_I_at_new + child_I_at_new


@wp.kernel
def _propagate_merged_body_poses(
    survivor_indices: wp.array[wp.int32],
    relative_xforms: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
):
    """Scatter survivor pose into each merged child's body_q slot."""
    tid = wp.tid()
    s = survivor_indices[tid]
    if s != tid:
        body_q[tid] = body_q[s] * relative_xforms[tid]


@wp.kernel
def _propagate_merged_body_velocities(
    survivor_indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    relative_xforms: wp.array[wp.transform],
    body_com_original: wp.array[wp.vec3],
    merged_body_com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
):
    """Propagate survivor spatial velocity to each merged child."""
    # body_qd is (linear-at-COM, angular); lever arm is the COM-to-COM world offset.
    tid = wp.tid()
    s = survivor_indices[tid]
    if s != tid:
        tw = body_qd[s]
        v = wp.spatial_top(tw)
        w = wp.spatial_bottom(tw)
        com_s_world = wp.transform_point(body_q[s], merged_body_com[s])
        child_q_world = body_q[s] * relative_xforms[tid]
        com_c_world = wp.transform_point(child_q_world, body_com_original[tid])
        r_world = com_c_world - com_s_world
        body_qd[tid] = wp.spatial_vector(v + wp.cross(w, r_world), w)


@wp.kernel
def _scatter_body_forces_to_survivors(
    survivor_indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    relative_xforms: wp.array[wp.transform],
    body_f: wp.array[wp.spatial_vector],
):
    """Move merged-child body_f onto the survivor and zero the child slot."""
    tid = wp.tid()
    s = survivor_indices[tid]
    if s == tid:
        return
    fc = body_f[tid]
    f_lin = wp.spatial_top(fc)
    f_ang = wp.spatial_bottom(fc)
    sv_q = body_q[s]
    com_s_world = wp.transform_point(sv_q, body_com[s])
    # Child body_q may be stale (propagation runs at end-of-step); rebuild via the rigid relation.
    rel = relative_xforms[tid]
    child_q = sv_q * rel
    com_c_world = wp.transform_point(child_q, body_com[tid])
    r_world = com_c_world - com_s_world
    wp.atomic_add(body_f, s, wp.spatial_vector(f_lin, f_ang + wp.cross(r_world, f_lin)))
    body_f[tid] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))


@wp.kernel
def _update_effective_inv_mass_inertia_merged(
    body_flags: wp.array[wp.int32],
    survivor_indices: wp.array[wp.int32],
    merged_inv_mass: wp.array[float],
    merged_inv_inertia: wp.array[wp.mat33],
    eff_inv_mass: wp.array[float],
    eff_inv_inertia: wp.array[wp.mat33],
):
    """Effective inverse mass/inertia, zeroed for kinematic and merged-child bodies."""
    tid = wp.tid()
    zero_mat = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if (body_flags[tid] & BodyFlags.KINEMATIC) != 0 or survivor_indices[tid] != tid:
        eff_inv_mass[tid] = float(0.0)
        eff_inv_inertia[tid] = zero_mat
    else:
        eff_inv_mass[tid] = merged_inv_mass[tid]
        eff_inv_inertia[tid] = merged_inv_inertia[tid]
