# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-level fixed-joint merger utility.

Computes merge metadata from a finalized :class:`~newton.Model` and provides
Warp kernels to propagate merged-body poses/velocities after each solver step.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..geometry.inertia import transform_inertia
from ..sim import BodyFlags, JointType
from ..sim.model import Model


@dataclass
class FixedJointMergeInfo:
    """Describes which bodies are collapsed at the solver level.

    Computed once at solver construction from the finalized :class:`~newton.Model`.
    All GPU arrays are allocated on the same device as the model.

    Attributes:
        has_merges: True iff at least one FIXED joint was collapsed.
        survivor_of: Per-body survivor index, shape [body_count].
            ``survivor_of[b] == b`` for bodies that are not merged children.
        relative_xform_of: Per-body transform from the survivor frame to body
            ``b``'s frame [m, unitless quat], shape [body_count].
            Identity transform for survivors.
        merged_body_mass: Per-body accumulated mass [kg], shape [body_count].
            Survivor bodies hold the sum of their own mass plus all merged
            children; merged children retain their original mass (it is not
            used in dynamics since their effective inv_mass is zeroed).
        merged_body_inertia: Per-body accumulated inertia tensor [kg·m²],
            shape [body_count].
        merged_body_inv_mass: Per-body effective inverse mass [1/kg],
            shape [body_count].  0 for merged children and for zero-mass bodies.
        merged_body_inv_inertia: Per-body effective inverse inertia [1/(kg·m²)],
            shape [body_count].  Zero matrix for merged children.
        merged_body_com: Per-body accumulated center of mass [m],
            shape [body_count].  Mass-weighted average for survivors.
        joint_enabled_effective: Per-joint effective enabled flag,
            shape [joint_count].  False for FIXED joints that were collapsed.
        survivor_indices_gpu: GPU int32 array, shape [body_count].
        relative_xforms_gpu: GPU transform array, shape [body_count].
        merged_body_inv_mass_gpu: GPU float array, shape [body_count].
        merged_body_inv_inertia_gpu: GPU mat33 array, shape [body_count].
        merged_body_mass_gpu: GPU float array, shape [body_count].
        merged_body_inertia_gpu: GPU mat33 array, shape [body_count].
        merged_body_com_gpu: GPU vec3 array, shape [body_count].
    """

    has_merges: bool
    survivor_of: list[int]
    relative_xform_of: list[wp.transform]
    merged_body_mass: list[float]
    merged_body_inertia: list[wp.mat33]
    merged_body_inv_mass: list[float]
    merged_body_inv_inertia: list[wp.mat33]
    merged_body_com: list[wp.vec3]
    joint_enabled_effective: list[bool]
    survivor_indices_gpu: wp.array
    relative_xforms_gpu: wp.array
    merged_body_inv_mass_gpu: wp.array
    merged_body_inv_inertia_gpu: wp.array
    merged_body_mass_gpu: wp.array
    merged_body_inertia_gpu: wp.array
    merged_body_com_gpu: wp.array


def compute_fixed_joint_merge(
    model: Model,
    joints_to_keep: list[str] | None = None,
) -> FixedJointMergeInfo | None:
    """Compute solver-level fixed-joint merge metadata from a finalized model.

    Reproduces the same DFS algorithm used by
    :meth:`~newton.ModelBuilder.collapse_fixed_joints` but reads from the
    finalized :class:`~newton.Model` arrays instead of the builder's Python
    lists.  The model is never modified.

    Args:
        model: The finalized model to analyse.
        joints_to_keep: Optional list of joint labels to exempt from collapsing.

    Returns:
        :class:`FixedJointMergeInfo` when at least one FIXED joint is
        collapsed, or ``None`` when no merging is needed.
    """
    if joints_to_keep is None:
        joints_to_keep = []

    body_count = model.body_count
    joint_count = model.joint_count

    if body_count == 0 or joint_count == 0:
        return None

    # Pull everything to CPU once.
    joint_type_np = model.joint_type.numpy()
    joint_parent_np = model.joint_parent.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_enabled_np = model.joint_enabled.numpy()
    joint_X_p_np = model.joint_X_p.numpy()  # shape [joint_count, 7]
    joint_X_c_np = model.joint_X_c.numpy()  # shape [joint_count, 7]
    body_inv_mass_np = model.body_inv_mass.numpy()
    body_inertia_np = model.body_inertia.numpy()  # shape [body_count, 3, 3]
    body_com_np = model.body_com.numpy()  # shape [body_count, 3]
    body_mass_np = model.body_mass.numpy()
    # Collect constraint bodies to avoid merging into world.
    bodies_in_constraints: set[int] = set()
    if model.equality_constraint_count > 0 and model.equality_constraint_body1 is not None:
        b1 = model.equality_constraint_body1.numpy()
        b2 = model.equality_constraint_body2.numpy()
        bodies_in_constraints.update(int(x) for x in b1 if x >= 0)
        bodies_in_constraints.update(int(x) for x in b2 if x >= 0)

    # Build body → children adjacency (joint index included for lookup).
    body_children: dict[int, list[int]] = {-1: []}
    joint_of: dict[tuple[int, int], int] = {}
    for i in range(body_count):
        body_children[i] = []
    for j in range(joint_count):
        p = int(joint_parent_np[j])
        c = int(joint_child_np[j])
        body_children[p].append(c)
        joint_of[(p, c)] = j

    # Sort children so traversal order matches body index order.
    for children in body_children.values():
        children.sort()

    # Initialise per-body working state.
    survivor_of: list[int] = list(range(body_count))
    relative_xform_of: list[wp.transform] = [wp.transform() for _ in range(body_count)]

    merged_body_mass: list[float] = [float(body_mass_np[b]) for b in range(body_count)]
    merged_body_inertia: list[wp.mat33] = [wp.mat33(*body_inertia_np[b].flatten()) for b in range(body_count)]
    merged_body_inv_mass: list[float] = [float(body_inv_mass_np[b]) for b in range(body_count)]
    merged_body_inv_inertia: list[wp.mat33] = [
        _safe_inv_mat33(wp.mat33(*body_inertia_np[b].flatten()), float(body_mass_np[b])) for b in range(body_count)
    ]
    merged_body_com: list[wp.vec3] = [wp.vec3(*body_com_np[b]) for b in range(body_count)]

    joint_enabled_effective: list[bool] = [bool(joint_enabled_np[j]) for j in range(joint_count)]

    visited: list[bool] = [False] * body_count

    def dfs(parent_body: int, child_body: int, incoming_xform: wp.transform, last_dynamic_body: int) -> None:
        j = joint_of[(parent_body, child_body)]
        jtype = int(joint_type_np[j])
        should_skip_merge = child_body in bodies_in_constraints and last_dynamic_body == -1
        joint_in_keep_list = model.joint_label[j] in joints_to_keep

        if jtype == JointType.FIXED and not should_skip_merge and not joint_in_keep_list:
            # Accumulate this fixed joint's transform.
            X_p = _np_row_to_transform(joint_X_p_np[j])
            X_c = _np_row_to_transform(joint_X_c_np[j])
            joint_xform = X_p * wp.transform_inverse(X_c)
            incoming_xform = incoming_xform * joint_xform

            # Mark joint as collapsed.
            joint_enabled_effective[j] = False

            if last_dynamic_body >= 0:
                # Record which survivor this body belongs to.
                survivor_of[child_body] = survivor_of[last_dynamic_body]
                relative_xform_of[child_body] = incoming_xform

                # Accumulate mass, COM and inertia into the survivor.
                sv = survivor_of[last_dynamic_body]
                child_m = float(body_mass_np[child_body])
                child_com = wp.transform_point(incoming_xform, wp.vec3(*body_com_np[child_body]))
                child_inertia = wp.mat33(*body_inertia_np[child_body].flatten())
                child_inertia_transformed = transform_inertia(
                    child_m, child_inertia, incoming_xform.p, incoming_xform.q
                )

                sv_m = merged_body_mass[sv]
                sv_com = merged_body_com[sv]
                new_m = sv_m + child_m
                merged_body_inertia[sv] = merged_body_inertia[sv] + child_inertia_transformed
                merged_body_mass[sv] = new_m
                if new_m > 0.0:
                    merged_body_com[sv] = (child_m * child_com + sv_m * sv_com) * (1.0 / new_m)
                    merged_body_inv_mass[sv] = 1.0 / new_m
                    merged_body_inv_inertia[sv] = _safe_inv_mat33(merged_body_inertia[sv], new_m)
                # Merged children: zeroed so they don't accumulate contact deltas.
                merged_body_inv_mass[child_body] = 0.0
                merged_body_inv_inertia[child_body] = wp.mat33(0.0)
        else:
            # Non-fixed (or exempted fixed) joint: child is its own survivor.
            last_dynamic_body = child_body
            incoming_xform = wp.transform()

        visited[child_body] = True
        for next_child in body_children.get(child_body, []):
            if not visited[next_child]:
                dfs(child_body, next_child, incoming_xform, last_dynamic_body)

    # Traverse from world root.
    for root in body_children[-1]:
        if not visited[root]:
            dfs(-1, root, wp.transform(), -1)

    # Handle bodies not reachable from world (disconnected subtrees).
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

    # Allocate GPU arrays.
    dev = model.device
    survivor_indices_gpu = wp.array(survivor_of, dtype=wp.int32, device=dev)
    relative_xforms_gpu = wp.array(relative_xform_of, dtype=wp.transform, device=dev)
    merged_body_inv_mass_gpu = wp.array(merged_body_inv_mass, dtype=wp.float32, device=dev)
    merged_body_inv_inertia_gpu = wp.array(merged_body_inv_inertia, dtype=wp.mat33, device=dev)
    merged_body_mass_gpu = wp.array(merged_body_mass, dtype=wp.float32, device=dev)
    merged_body_inertia_gpu = wp.array(merged_body_inertia, dtype=wp.mat33, device=dev)
    merged_body_com_gpu = wp.array(merged_body_com, dtype=wp.vec3, device=dev)

    return FixedJointMergeInfo(
        has_merges=has_merges,
        survivor_of=survivor_of,
        relative_xform_of=relative_xform_of,
        merged_body_mass=merged_body_mass,
        merged_body_inertia=merged_body_inertia,
        merged_body_inv_mass=merged_body_inv_mass,
        merged_body_inv_inertia=merged_body_inv_inertia,
        merged_body_com=merged_body_com,
        joint_enabled_effective=joint_enabled_effective,
        survivor_indices_gpu=survivor_indices_gpu,
        relative_xforms_gpu=relative_xforms_gpu,
        merged_body_inv_mass_gpu=merged_body_inv_mass_gpu,
        merged_body_inv_inertia_gpu=merged_body_inv_inertia_gpu,
        merged_body_mass_gpu=merged_body_mass_gpu,
        merged_body_inertia_gpu=merged_body_inertia_gpu,
        merged_body_com_gpu=merged_body_com_gpu,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Warp kernels (called from SolverBase)
# ---------------------------------------------------------------------------


@wp.kernel
def _propagate_merged_body_poses(
    survivor_indices: wp.array[wp.int32],
    relative_xforms: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
):
    """Scatter survivor pose into each merged child's body_q slot.

    Survivors are left untouched (their thread is a no-op).  Each merged child
    gets ``body_q[child] = body_q[survivor] * relative_xform[child]``.
    There is no write-aliasing: each thread writes exactly its own slot, and
    survivor indices are always distinct from child indices by construction.
    """
    tid = wp.tid()
    s = survivor_indices[tid]
    if s != tid:
        body_q[tid] = body_q[s] * relative_xforms[tid]


@wp.kernel
def _propagate_merged_body_velocities(
    survivor_indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    relative_xforms: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Propagate the survivor's spatial velocity to each merged child.

    Uses the rigid-body formula:
    ``v_child = v_survivor + omega_survivor x r``
    where ``r`` is the offset from the survivor's COM to the child's frame
    origin in world space.
    """
    tid = wp.tid()
    s = survivor_indices[tid]
    if s != tid:
        tw = body_qd[s]
        w = wp.spatial_top(tw)  # angular velocity
        v = wp.spatial_bottom(tw)  # linear velocity at survivor COM
        r_local = relative_xforms[tid].p
        r_world = wp.transform_vector(body_q[s], r_local)
        body_qd[tid] = wp.spatial_vector(w, v + wp.cross(w, r_world))


@wp.kernel
def _update_effective_inv_mass_inertia_merged(
    body_flags: wp.array[wp.int32],
    survivor_indices: wp.array[wp.int32],
    merged_inv_mass: wp.array[float],
    merged_inv_inertia: wp.array[wp.mat33],
    eff_inv_mass: wp.array[float],
    eff_inv_inertia: wp.array[wp.mat33],
):
    """Compute effective inverse mass/inertia with merged-body accumulation.

    Kinematic bodies and merged children both get zero effective mass so they
    do not accumulate contact deltas.  Survivor bodies use their accumulated
    inverse mass/inertia from the merge computation.
    """
    tid = wp.tid()
    zero_mat = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if (body_flags[tid] & BodyFlags.KINEMATIC) != 0 or survivor_indices[tid] != tid:
        eff_inv_mass[tid] = float(0.0)
        eff_inv_inertia[tid] = zero_mat
    else:
        eff_inv_mass[tid] = merged_inv_mass[tid]
        eff_inv_inertia[tid] = merged_inv_inertia[tid]
