# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded, local-only spin resistance with an explicit effective radius.

No material radius is enabled implicitly. This diagnostic adds one angular
row per coherent contact group, not one full torque budget per witness. With
point friction a group is a coplanar cluster of one exact shape pair, and every
member carries its own tangent pair. With persistent friction patches a group is
one patch region: its budget is the region's pooled normal load times the
undivided friction coefficient, and only the anchor rows that exist consume the
shared sliding budget. Normal rows stay independent in both cases. It uses the
same articulated response and bilateral projection as existing dense rows. All
state is rebuilt per step; there is no anchor or elastic orientation memory.
Host grouping precludes graphs and throughput use.
"""

import re
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...utils.selection import match_labels
from .kernels import (
    _FPGS_CONTACT_END_GAP_SLOP,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_TORSION,
)

_TOUCH_TOLERANCE = 1.0e-5
_NORMAL_COSINE = 0.999
_MAX_GROUP_CONTACTS = 4096
# Normal rows store their spin-row index in the membership array. A spin row
# has no parent membership, so its otherwise-unused cell records admission.
# Both host and device preparation reset every cell to -1 each step.
_TORSION_SPIN_RETIRED = -2
_SUPPORTED_TYPES = (
    GeoType.SPHERE,
    GeoType.BOX,
    GeoType.CAPSULE,
    GeoType.CYLINDER,
    GeoType.CONE,
    GeoType.ELLIPSOID,
    GeoType.PLANE,
    GeoType.CONVEX_MESH,
)


def configure_contact_torsion(solver, radius, indices, patterns):
    """Validate the experimental opt-in and resolve stable shape-name selectors."""
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("contact_torsion_radius must be finite and non-negative [m]")
    if radius > 0 and solver.contact_compliance:
        raise ValueError("contact_torsion_radius > 0 is not validated with contact_compliance")
    if indices is not None and patterns is not None:
        raise ValueError("Use either contact_torsion_shape_indices or contact_torsion_shape_patterns")
    selected = None
    if indices is not None:
        indices = tuple(indices)
        if any(
            not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= solver.model.shape_count
            for i in indices
        ):
            raise ValueError("contact_torsion_shape_indices contains invalid global shape indices")
        indices = tuple(int(i) for i in indices)
        selected = frozenset(indices)
    if patterns is not None:
        if isinstance(patterns, str):
            raise ValueError("contact_torsion_shape_patterns must be a sequence, not a string")
        patterns = tuple(patterns)
        selected = set()
        for pattern in patterns:
            if not isinstance(pattern, str):
                raise ValueError("contact_torsion_shape_patterns entries must be strings")
            try:
                regex = re.compile(pattern)
            except re.error as error:
                raise ValueError("Invalid contact_torsion_shape_patterns regex") from error
            matches = set(match_labels(solver.model.shape_label, regex))
            if not matches:
                raise ValueError(f"contact_torsion_shape_patterns matched no shapes: {pattern!r}")
            selected.update(matches)
        selected = frozenset(selected)
    if radius > 0 and selected is not None:
        types = solver.model.shape_type.numpy()
        unsupported = [i for i in selected if types[i] not in _SUPPORTED_TYPES]
        if unsupported:
            raise ValueError(f"Unsupported contact torsion shape indices: {unsupported}")
    solver._contact_torsion_radius = radius
    solver._contact_torsion_enabled = radius > 0
    solver._contact_torsion_shape_indices = indices
    solver._contact_torsion_shape_patterns = patterns
    solver._contact_torsion_shape_set = selected
    solver._torsion_stats = {}
    if radius > 0:
        _validate_torsion_mode(solver)


def _validate_torsion_mode(solver):
    """Reject unsupported construction and runtime mode combinations."""
    if (
        not solver.model.device.is_cuda
        or solver.model.requires_grad
        or solver.pgs_mode != "matrix_free"
        or solver.articulated_contact_response != "immediate"
        or solver.friction_mode != "current"
        or solver.pgs_schedule != "interleaved"
        or solver.pgs_warmstart
        or solver.pgs_debug
        or solver.contact_compliance
        or not solver.enable_contact_friction
    ):
        raise ValueError(
            "Contact torsion requires CUDA non-differentiable matrix_free/immediate/current/"
            "interleaved, without warmstart, compliance, or debug"
        )
    if solver.pgs_velocity_iterations > 0 and solver.enable_bilateral_preelimination:
        raise ValueError("Contact torsion velocity post-passes require enable_bilateral_preelimination=False")


@dataclass
class _Witness:
    """Represent one dense contact normal row admitted to torsion grouping."""

    world: int
    slot: int
    shape_a: int
    shape_b: int
    body_a: int
    body_b: int
    normal: np.ndarray
    point: np.ndarray
    gap: float
    anchor: bool
    """Whether this normal row is followed by its own tangent pair."""


def _dot3(a, b):
    """Round three float32 products and sum in float64, independent of BLAS.

    Admission and grouping thresholds must use the same operation order as
    device preparation. NumPy dot can choose different reductions by platform.
    """
    x = float(np.float32(a[0] * b[0]))
    y = float(np.float32(a[1] * b[1]))
    z = float(np.float32(a[2] * b[2]))
    return np.float32((x + y) + z)


def _transform_point(pose, point):
    """Transform a body-frame witness using scalar-last quaternion storage."""
    q = pose[3:]
    cross = 2.0 * np.cross(q[:3], point)
    return point + q[3] * cross + np.cross(q[:3], cross) + pose[:3]


def _contact_groups(solver, state, contacts):
    """Group current dense witnesses without anchor or impulse history.

    Point friction clusters touching witnesses of one exact shape pair by normal
    and contact plane. Persistent patches already define the friction unit, so
    each patch region becomes one group holding every dense normal of its load
    ring, touching or not, and a region with any inadmissible member is skipped.
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count > contacts.rigid_contact_max:
        raise RuntimeError("Contact input overflow before contact torsion")
    if count == 0:
        return []
    stiffness = contacts.rigid_contact_stiffness
    if stiffness is not None and np.any(stiffness.numpy()[:count] > 0):
        raise ValueError("Contact torsion does not support hydroelastic contacts")
    paths = solver.contact_path.numpy()[:count]
    slots = solver.contact_slot.numpy()[:count]
    worlds = solver.contact_world.numpy()[:count]
    slots_needed = solver.contact_slots_needed.numpy()[:count]
    shape_a = contacts.rigid_contact_shape0.numpy()[:count]
    shape_b = contacts.rigid_contact_shape1.numpy()[:count]
    normals = -contacts.rigid_contact_normal.numpy()[:count]
    points_a = contacts.rigid_contact_point0.numpy()[:count]
    points_b = contacts.rigid_contact_point1.numpy()[:count]
    margins_a = contacts.rigid_contact_margin0.numpy()[:count]
    margins_b = contacts.rigid_contact_margin1.numpy()[:count]
    shape_body = solver.model.shape_body.numpy()
    shape_types = solver.model.shape_type.numpy()
    poses = state.body_q.numpy()
    row_types = solver.row_type.numpy()
    parents = solver.row_parent.numpy()
    row_counts = solver.constraint_count.numpy()
    patches = bool(solver._friction_anchors_enabled)
    owners = solver._friction_patches.current.owner.numpy()[:count] if patches else None
    groups = {}
    rejected = set()
    selected = solver._contact_torsion_shape_set
    admitted = 0
    for c in range(count):
        world, slot = int(worlds[c]), int(slots[c])
        if paths[c] != 0 or slot < 0:
            continue
        region = (world, int(owners[c])) if patches else None
        a, b = int(shape_a[c]), int(shape_b[c])
        if a < 0 or b < 0 or (selected is not None and a not in selected and b not in selected):
            rejected.add(region)
            continue
        if shape_types[a] not in _SUPPORTED_TYPES or shape_types[b] not in _SUPPORTED_TYPES:
            rejected.add(region)
            continue
        # Point friction gives every admitted normal its tangent pair. Patches give
        # tangent rows only to the selected anchors; other normals occupy one slot.
        anchor = int(slots_needed[c]) == 3
        if (
            world < 0
            or world >= len(row_counts)
            or slot >= row_counts[world]
            or slot >= row_types.shape[1]
            or row_types[world, slot] != PGS_CONSTRAINT_TYPE_CONTACT
            or (not anchor and not patches)
            or (
                anchor
                and (
                    slot + 2 >= row_counts[world]
                    or slot + 2 >= row_types.shape[1]
                    or any(
                        row_types[world, slot + k] != PGS_CONSTRAINT_TYPE_FRICTION or parents[world, slot + k] != slot
                        for k in (1, 2)
                    )
                )
            )
        ):
            rejected.add(region)
            continue
        ba, bb = int(shape_body[a]), int(shape_body[b])
        pa, pb = points_a[c].copy(), points_b[c].copy()
        if ba >= 0:
            pa = _transform_point(poses[ba], pa)
        if bb >= 0:
            pb = _transform_point(poses[bb], pb)
        pa -= margins_a[c] * normals[c]
        pb += margins_b[c] * normals[c]
        gap = float(_dot3(normals[c], pa - pb))
        if gap > _TOUCH_TOLERANCE and not patches:
            continue
        admitted += 1
        if admitted > _MAX_GROUP_CONTACTS:
            raise RuntimeError("Contact torsion host grouping capacity exceeded; refusing dropped work")
        witness = _Witness(world, slot, a, b, ba, bb, normals[c], (pa + pb) * 0.5, gap, anchor)
        if patches:
            groups.setdefault(region, []).append(witness)
            continue
        clusters = groups.setdefault((world, a, b, ba, bb), [])
        for cluster in clusters:
            if all(
                _dot3(witness.normal, other.normal) >= _NORMAL_COSINE
                and abs(_dot3(witness.normal, witness.point - other.point)) <= _TOUCH_TOLERANCE
                for other in cluster
            ):
                cluster.append(witness)
                break
        else:
            clusters.append([witness])
    if not patches:
        return [cluster for clusters in groups.values() for cluster in clusters]
    regions = [group for region, group in groups.items() if region not in rejected]
    for group in regions:
        # The pooled load walks the normal-parent ring; the budget is only exact
        # when that ring and this group hold the same rows.
        members = {witness.slot for witness in group}
        if any(int(parents[witness.world, witness.slot]) not in members for witness in group):
            raise RuntimeError("Persistent friction patch load ring does not match its contact torsion group")
    return regions


def _group_budget(solver, group, row_mu):
    """Return the undivided Coulomb coefficient, anchor witnesses and touching witnesses of a group.

    Point friction gives every member the full coefficient on its own tangent
    pair. Persistent patches divide the coefficient on each anchor's tangent pair
    by the number of anchors that received rows, so the region's budget is that
    coefficient multiplied back. Returns None when the group cannot carry spin.
    """
    anchors = [witness for witness in group if witness.anchor]
    touching = [witness for witness in group if witness.gap <= _TOUCH_TOLERANCE]
    if not anchors or not touching:
        return None
    coefficient = float(row_mu[group[0].world, anchors[0].slot + 1])
    if solver._friction_anchors_enabled:
        coefficient *= len(anchors)
    if coefficient <= 0:
        return None
    return coefficient, anchors, touching


def validate_torsion_step(solver):
    """Reject incompatible runtime state before any solver stage consumes it."""
    _validate_torsion_mode(solver)
    if wp.get_stream(solver.model.device).is_capturing:
        device = getattr(solver, "_device_torsion", None)
        if device is None or not device.deferred_errors:
            raise RuntimeError("Contact torsion graph capture requires device preparation with deferred validation")


@wp.kernel
def _prepare_velocity_torsion_rows(
    count: wp.array[int],
    capacity: int,
    row_type: wp.array2d[int],
    group: wp.array2d[int],
    phi: wp.array2d[float],
    target: wp.array2d[float],
    position_velocity: wp.array[float],
    dof_count: wp.array[int],
    dof_indices: wp.array2d[int],
    jacobian: wp.array3d[float],
    dt: float,
    rhs: wp.array2d[float],
):
    """Conclude speculative spin admission without discarding touching support."""
    tid = wp.tid()
    world = tid // capacity
    spin = tid % capacity
    if spin >= count[world] or row_type[world, spin] != PGS_CONSTRAINT_TYPE_TORSION:
        return
    touching = bool(False)
    for row in range(count[world]):
        if row_type[world, row] == PGS_CONSTRAINT_TYPE_CONTACT and group[world, row] == spin:
            speed = float(0.0)
            for dof in range(dof_count[world]):
                global_dof = dof_indices[world, dof]
                if global_dof >= 0:
                    speed += jacobian[world, row, dof] * position_velocity[global_dof]
            end_gap = phi[world, row] + dt * (speed - target[world, row])
            # Retain touching support through rebound, with the existing end-gap
            # slop as the initial touching tolerance. Otherwise require contact
            # by the position end. Final normal load still bounds all traction.
            if phi[world, row] <= _FPGS_CONTACT_END_GAP_SLOP or end_gap <= _FPGS_CONTACT_END_GAP_SLOP:
                touching = True
    # The angular row has no positional spring: retain prescribed angular
    # target motion but not position-solve contact correction or restitution.
    rhs[world, spin] = -target[world, spin]
    # Keep the undivided coefficient: still-loaded normal rows retain sliding
    # friction even when separated support retires the additional spin row.
    group[world, spin] = -1 if touching else _TORSION_SPIN_RETIRED
    # Do not zero lambda here. The coupled sweep must apply Y * (new - old)
    # to refund position-phase spin and tangential impulses coherently.


def prepare_torsion_velocity_pass(solver, dt):
    """Conclude touching eligibility and angular RHS for the unbiased pass.

    Keep accumulated impulses and their velocity response from the position
    solve. Each velocity sweep then recomputes the shared sliding/spin bound
    from its final normal load and applies every impulse delta once. Initially
    touching rows retain admission through rebound, using end-gap slop as the
    initial touching tolerance. Final normal load remains authoritative, even
    within this tolerance. Unreached groups refund carried spin.
    """
    wp.launch(
        _prepare_velocity_torsion_rows,
        dim=solver.world_count * solver.dense_max_constraints,
        inputs=[
            solver.constraint_count,
            solver.dense_max_constraints,
            solver.row_type,
            solver._contact_torsion_group,
            solver.phi,
            solver.target_velocity,
            solver.v_out_snap,
            solver.world_dof_count,
            solver.world_dof_indices,
            solver.J_world,
            dt,
        ],
        outputs=[solver.rhs_unbiased],
        device=solver.model.device,
    )


def prepare_torsion_rows(solver, state, augmented_state, contacts):
    """Append current touching-group angular rows before H-inverse/J response."""
    validate_torsion_step(solver)
    if getattr(solver, "_device_torsion", None) is not None:
        solver._device_torsion.prepare(state, augmented_state, contacts)
        return
    solver._torsion_stats = {"rows": 0, "groups": []}
    if contacts is None:
        return
    # The allocator rolls counters back on failure; count <= capacity cannot
    # prove that every contact was retained. Tracking is mandatory for torsion.
    if np.any(solver._row_dropped_dense.numpy()):
        raise RuntimeError("Dense contact row overflow before contact torsion; refusing dropped rows")
    groups = _contact_groups(solver, state, contacts)
    if not groups:
        return
    count = solver.constraint_count.numpy()
    membership = np.full(solver._contact_torsion_group.shape, -1, dtype=np.int32)
    fields = {
        key: getattr(solver, key).numpy()
        for key in (
            "row_type",
            "row_parent",
            "row_mu",
            "row_beta",
            "row_cfm",
            "phi",
            "target_velocity",
            "row_restitution",
        )
    }
    jacobians = {size: array.numpy() for size, array in solver.J_by_size.items()}
    body_joint = solver.body_to_joint.numpy()
    body_art = solver.body_to_articulation.numpy()
    ancestor = solver.model.joint_ancestor.numpy()
    qd_start = solver.model.joint_qd_start.numpy()
    art_start = solver.articulation_dof_start.numpy()
    art_size = solver.articulation_response_dof_count.numpy()
    art_group = solver.art_group_idx.numpy()
    motions = augmented_state.joint_S_s.numpy()
    prescribed = solver._prescribed_articulation.numpy()
    body_velocities = augmented_state.body_v_s.numpy()
    for group in groups:
        world = group[0].world
        budget = _group_budget(solver, group, fields["row_mu"])
        if budget is None:
            continue
        coefficient, anchors, touching = budget
        lead = touching[0]
        row = int(count[world])
        if row >= solver.dense_max_constraints:
            raise RuntimeError("Experimental torsion row capacity exceeded; refusing dropped work")
        count[world] += 1
        for field in fields.values():
            field[world, row] = 0
        fields["row_type"][world, row] = PGS_CONSTRAINT_TYPE_TORSION
        fields["row_parent"][world, row] = lead.slot
        # The sweep reads the group's coefficient from its own row; anchor tangent
        # rows carry it divided by the anchor count under persistent patches.
        fields["row_mu"][world, row] = coefficient
        fields["row_cfm"][world, row] = solver.pgs_cfm
        for witness in group:
            membership[world, witness.slot] = row
        normal = lead.normal
        for body, sign in ((lead.body_a, 1.0), (lead.body_b, -1.0)):
            if body < 0:
                continue
            art = int(body_art[body])
            if art < 0:
                continue
            if prescribed[art]:
                fields["target_velocity"][world, row] -= sign * _dot3(normal, body_velocities[body, 3:])
                continue
            size, index = int(art_size[art]), int(art_group[art])
            joint = int(body_joint[body])
            while joint >= 0:
                for global_dof in range(int(qd_start[joint]), int(qd_start[joint + 1])):
                    local = global_dof - int(art_start[art])
                    if 0 <= local < size:
                        jacobians[size][index, row, local] += sign * _dot3(normal, motions[global_dof, 3:])
                joint = int(ancestor[joint])
        solver._torsion_stats["groups"].append(
            {
                "world": int(world),
                "row": row,
                "normal_rows": [w.slot for w in group],
                "anchor_rows": [w.slot for w in anchors],
                "shape_pair": [lead.shape_a, lead.shape_b],
                "effective_radius_m": solver.contact_torsion_radius,
                "mu": coefficient,
                "gap_max_m": max(w.gap for w in touching),
            }
        )
    solver._torsion_stats["rows"] = len(solver._torsion_stats["groups"])
    if not solver._torsion_stats["rows"]:
        return
    for name, values in fields.items():
        getattr(solver, name).assign(values)
    for size, values in jacobians.items():
        solver.J_by_size[size].assign(values)
    solver.constraint_count.assign(count)
    solver.slot_counter.assign(count)
    solver._contact_torsion_group.assign(membership)


def torque_sweep_source(dofs):
    """Return the coupled residual-budget angular solve inside each PGS sweep."""
    return f"""
        if (row_phase == 0 || row_phase == 1 || row_phase == 4) {{
            for (int spin = 0; spin < m_dense; ++spin) {{
                if (world_row_type.data[off_dense + spin] != {PGS_CONSTRAINT_TYPE_TORSION}) continue;
                float radius = contact_torsion_radius;
                // The torsion row carries its group's undivided coefficient. Persistent
                // patches divide the anchor tangent rows' coefficient by the anchor count.
                float mu = fmaxf(world_row_mu.data[off_dense + spin], 0.0f);
                float normal_load = 0.0f;
                float sliding_used = 0.0f;
                for (int n = 0; n < m_dense; ++n) {{
                    if (world_row_type.data[off_dense + n] != {PGS_CONSTRAINT_TYPE_CONTACT} ||
                        world_torsion_group.data[off_dense + n] != spin) continue;
                    normal_load += fmaxf(s_lam_dense[n], 0.0f);
                    // Only a normal followed by its own tangent pair consumes sliding budget;
                    // under persistent patches, non-anchor normals occupy a single slot.
                    if (n + 2 < m_dense &&
                        world_row_type.data[off_dense + n + 1] == {PGS_CONSTRAINT_TYPE_FRICTION} &&
                        world_row_parent.data[off_dense + n + 1] == n &&
                        world_row_type.data[off_dense + n + 2] == {PGS_CONSTRAINT_TYPE_FRICTION} &&
                        world_row_parent.data[off_dense + n + 2] == n) {{
                        float a = s_lam_dense[n + 1], b = s_lam_dense[n + 2];
                        sliding_used += sqrtf(a * a + b * b);
                    }}
                }}
                // Normal rows have already applied their accumulated-impulse
                // update (regularized positions, unbiased velocity cleanup).
                // Use the actual final load without a second contact_w factor.
                float normal_budget = mu * normal_load;
                // Later normal-only rows can lower a patch's load after its
                // anchors have been solved. Project against the final load
                // and apply the velocity response, not just the stored impulses.
                if (sliding_used > normal_budget) {{
                    float scale = normal_budget / sliding_used;
                    sliding_used = 0.0f;
                    for (int n = 0; n < m_dense; ++n) {{
                        if (world_row_type.data[off_dense + n] != {PGS_CONSTRAINT_TYPE_CONTACT} ||
                            world_torsion_group.data[off_dense + n] != spin ||
                            n + 2 >= m_dense ||
                            world_row_type.data[off_dense + n + 1] != {PGS_CONSTRAINT_TYPE_FRICTION} ||
                            world_row_parent.data[off_dense + n + 1] != n ||
                            world_row_type.data[off_dense + n + 2] != {PGS_CONSTRAINT_TYPE_FRICTION} ||
                            world_row_parent.data[off_dense + n + 2] != n) continue;
                        float t1 = s_lam_dense[n + 1] * scale;
                        float t2 = s_lam_dense[n + 2] * scale;
                        float delta1 = t1 - s_lam_dense[n + 1];
                        float delta2 = t2 - s_lam_dense[n + 2];
                        if (delta1 != 0.0f || delta2 != 0.0f) {{
                            iteration_changed = 1;
                            if (lane == 0) {{
                                s_lam_dense[n + 1] = t1;
                                s_lam_dense[n + 2] = t2;
                            }}
                            for (int d = lane; d < {dofs}; d += 32) {{
                                s_v[d] += Y_world.data[jy_world_base + (n + 1) * {dofs} + d] * delta1
                                    + Y_world.data[jy_world_base + (n + 2) * {dofs} + d] * delta2;
                            }}
                        }}
                        sliding_used += sqrtf(t1 * t1 + t2 * t2);
                        __syncwarp();
                    }}
                }}
                float bound = radius * fmaxf(normal_budget - sliding_used, 0.0f);
                if (world_torsion_group.data[off_dense + spin] == {_TORSION_SPIN_RETIRED}) bound = 0.0f;
                if (global_iter < friction_start_iteration) bound = 0.0f;
                float dot = 0.0f;
                for (int d = lane; d < {dofs}; d += 32)
                    dot += J_world.data[jy_world_base + spin * {dofs} + d] * s_v[d];
                dot += __shfl_down_sync(MASK, dot, 16);
                dot += __shfl_down_sync(MASK, dot, 8);
                dot += __shfl_down_sync(MASK, dot, 4);
                dot += __shfl_down_sync(MASK, dot, 2);
                dot += __shfl_down_sync(MASK, dot, 1);
                float residual = __shfl_sync(MASK, dot, 0) + rhs_bias.data[off_dense + spin];
                float old = s_lam_dense[spin];
                float diagonal = world_diag.data[off_dense + spin];
                float trial = diagonal > 0.0f ? old - omega * residual / diagonal : 0.0f;
                float next = fminf(fmaxf(trial, -bound), bound);
                float delta = next - old;
                if (delta != 0.0f) {{
                    iteration_changed = 1;
                    if (lane == 0) s_lam_dense[spin] = next;
                    for (int d = lane; d < {dofs}; d += 32)
                        s_v[d] += Y_world.data[jy_world_base + spin * {dofs} + d] * delta;
                }}
                __syncwarp();
            }}
        }}
"""
