# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Device preparation for the experimental contact spin rows.

The host implementation remains the oracle during qualification. This module
changes preparation only; the coupled PGS budget and final-load projection are
shared with the host path. Enable explicitly with ``enable_device_torsion``.
"""

from typing import Any

import numpy as np
import warp as wp

from .contact_torsion import _NORMAL_COSINE, _SUPPORTED_TYPES, _TOUCH_TOLERANCE
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_TORSION

# Match the retained host oracle's rounding at angular cancellation and
# touching/grouping thresholds. This option affects only this preparation module.
wp.set_module_options({"fuse_fp": False})


@wp.func
def _host_dot3(a: wp.vec3, b: wp.vec3):
    # Match contact_torsion._dot3's explicit arithmetic contract, not NumPy's
    # platform-dependent BLAS. Fused or float32-only sums differ at thresholds.
    x = wp.float64(a[0] * b[0])
    y = wp.float64(a[1] * b[1])
    z = wp.float64(a[2] * b[2])
    return float(x + y + z)


@wp.func
def _host_transform_point(pose: wp.transform, point: wp.vec3):
    # Use the oracle's cross-product expression, not the algebraically
    # equivalent quaternion polynomial used by wp.transform_point.
    q = wp.transform_get_rotation(pose)
    v = wp.vec3(q[0], q[1], q[2])
    cross = 2.0 * wp.cross(v, point)
    return point + q[3] * cross + wp.cross(v, cross) + wp.transform_get_translation(pose)


@wp.struct
class _Workspace:
    keys: wp.array[wp.int64]
    indices: wp.array[int]
    world: wp.array[int]
    slot: wp.array[int]
    shape_a: wp.array[int]
    shape_b: wp.array[int]
    body_a: wp.array[int]
    body_b: wp.array[int]
    normal: wp.array[wp.vec3]
    point: wp.array[wp.vec3]
    gap: wp.array[float]
    owner: wp.array[int]
    eligible: wp.array[int]
    anchor: wp.array[int]
    group: wp.array[int]
    next_member: wp.array[int]
    tail: wp.array[int]
    first_pair: wp.array[int]
    lead: wp.array[int]
    coefficient: wp.array[float]
    spin_row: wp.array[int]
    row_contact: wp.array2d[int]
    spin_source: wp.array2d[int]
    planned_count: wp.array[int]
    status: wp.array[int]
    patch_keys: wp.array[wp.int64]
    patch_indices: wp.array[int]
    rejected: wp.array[int]


@wp.kernel(enable_backward=False)
def _gather(
    count: wp.array[int],
    capacity: int,
    input_capacity: int,
    world_count: int,
    paths: wp.array[int],
    slots: wp.array[int],
    worlds: wp.array[int],
    slots_needed: wp.array[int],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    normals: wp.array[wp.vec3],
    point_a: wp.array[wp.vec3],
    point_b: wp.array[wp.vec3],
    margin_a: wp.array[float],
    margin_b: wp.array[float],
    stiffness: wp.array[float],
    has_stiffness: int,
    shape_body: wp.array[int],
    supported: wp.array[int],
    selected: wp.array[int],
    poses: wp.array[wp.transform],
    row_type: wp.array2d[int],
    parents: wp.array2d[int],
    row_count: wp.array[int],
    patch_mode: int,
    owner: wp.array[int],
    work: _Workspace,
):
    c = wp.tid()
    work.keys[c] = wp.int64(9223372036854775807)
    work.indices[c] = c
    work.eligible[c] = 0
    work.group[c] = -1
    work.next_member[c] = -1
    work.tail[c] = c
    work.spin_row[c] = -1
    work.lead[c] = -1
    work.coefficient[c] = 0.0
    if c == 0 and (count[0] > input_capacity or count[0] < 0):
        wp.atomic_or(work.status, 0, 1)
    if c >= wp.min(count[0], input_capacity):
        return
    if has_stiffness != 0 and stiffness[c] > 0.0:
        wp.atomic_or(work.status, 0, 2)
    world = worlds[c]
    slot = slots[c]
    if paths[c] != 0 or slot < 0 or world < 0 or world >= world_count:
        return
    work.keys[c] = wp.int64(world) * wp.int64(capacity + 1) + wp.int64(c)
    work.world[c] = world
    work.slot[c] = slot
    work.owner[c] = -1
    if patch_mode != 0:
        work.owner[c] = owner[c]
    work.eligible[c] = -1
    a = shape_a[c]
    b = shape_b[c]
    if a < 0 or b < 0 or a >= shape_body.shape[0] or b >= shape_body.shape[0]:
        return
    if selected[a] == 0 and selected[b] == 0:
        return
    if supported[a] == 0 or supported[b] == 0:
        return
    anchor = int(slots_needed[c] == 3)
    if slot >= row_count[world] or slot >= row_type.shape[1]:
        return
    if row_type[world, slot] != PGS_CONSTRAINT_TYPE_CONTACT or (anchor == 0 and patch_mode == 0):
        return
    if anchor != 0:
        if slot + 2 >= row_count[world] or slot + 2 >= row_type.shape[1]:
            return
        for k in range(1, 3):
            if row_type[world, slot + k] != PGS_CONSTRAINT_TYPE_FRICTION or parents[world, slot + k] != slot:
                return
    normal = -normals[c]
    ba = shape_body[a]
    bb = shape_body[b]
    pa = point_a[c]
    pb = point_b[c]
    if ba >= 0:
        pa = _host_transform_point(poses[ba], pa)
    if bb >= 0:
        pb = _host_transform_point(poses[bb], pb)
    pa -= margin_a[c] * normal
    pb += margin_b[c] * normal
    gap = _host_dot3(normal, pa - pb)
    if gap > _TOUCH_TOLERANCE and patch_mode == 0:
        work.eligible[c] = 0
        return
    work.eligible[c] = 1
    work.shape_a[c] = a
    work.shape_b[c] = b
    work.body_a[c] = ba
    work.body_b[c] = bb
    work.normal[c] = normal
    work.point[c] = 0.5 * (pa + pb)
    work.gap[c] = gap
    work.anchor[c] = anchor
    # Valid row allocation gives a unique contact per normal slot.
    work.row_contact[world, slot] = c


@wp.kernel(enable_backward=False)
def _patch_keys(work: _Workspace):
    """Key contacts by (world, owner, ineligible) so each patch sorts contiguously in contact order."""
    c = wp.tid()
    work.patch_indices[c] = c
    work.patch_keys[c] = wp.int64(9223372036854775807)
    if work.keys[c] == wp.int64(9223372036854775807):
        return
    eligible = work.eligible[c]
    if eligible != 1:
        # Emission visits representatives only, so the (world, contact) order keeps eligible contacts alone.
        work.keys[c] = wp.int64(9223372036854775807)
    owner = wp.int64(work.owner[c]) + wp.int64(2147483648)
    key = (wp.int64(work.world[c]) << wp.int64(33)) | (owner << wp.int64(1))
    if eligible != 1:
        key = key | wp.int64(1)
    work.patch_keys[c] = key


@wp.kernel(enable_backward=False)
def _group_patches(capacity: int, mu: wp.array2d[float], work: _Workspace):
    """Summarize one patch from its head; eligible members precede rejection witnesses."""
    i = wp.tid()
    key = work.patch_keys[i]
    if key == wp.int64(9223372036854775807):
        return
    patch = key >> wp.int64(1)
    if i > 0 and (work.patch_keys[i - 1] >> wp.int64(1)) == patch:
        return
    rep = work.patch_indices[i]
    if work.eligible[rep] != 1:
        return
    rejected = int(0)
    anchor_count = int(0)
    first_anchor = int(-1)
    touching = int(-1)
    j = i
    while j < capacity and (work.patch_keys[j] >> wp.int64(1)) == patch:
        c = work.patch_indices[j]
        if work.eligible[c] == 1:
            work.group[c] = rep
            work.first_pair[c] = rep
            if work.anchor[c] != 0:
                anchor_count += 1
                if first_anchor < 0:
                    first_anchor = c
            if touching < 0 and work.gap[c] <= _TOUCH_TOLERANCE:
                touching = c
        elif work.eligible[c] == -1:
            rejected = 1
        j += 1
    work.rejected[rep] = rejected
    if rejected != 0 or first_anchor < 0 or touching < 0:
        return
    coefficient = mu[work.world[rep], work.slot[first_anchor] + 1] * float(anchor_count)
    if coefficient <= 0.0:
        return
    work.coefficient[rep] = coefficient
    work.lead[rep] = touching


@wp.kernel(enable_backward=False)
def _check_patch_membership(row_capacity: int, parents: wp.array2d[int], work: _Workspace):
    c = wp.tid()
    rep = work.group[c]
    if rep < 0 or work.rejected[rep] != 0:
        return
    world = work.world[c]
    parent = parents[world, work.slot[c]]
    member = int(-1)
    if parent >= 0 and parent < row_capacity:
        member = work.row_contact[world, parent]
    if member < 0:
        wp.atomic_or(work.status, 0, 16)
    elif work.group[member] != rep:
        wp.atomic_or(work.status, 0, 16)


@wp.kernel(enable_backward=False)
def _emit_patches(
    contact_capacity: int,
    row_capacity: int,
    row_count: wp.array[int],
    row_dropped: wp.array[int],
    work: _Workspace,
):
    world = wp.tid()
    if row_dropped[world] != 0:
        wp.atomic_or(work.status, 0, 4)
    stride = wp.int64(contact_capacity + 1)
    start = _lower_bound(work.keys, wp.int64(world) * stride, contact_capacity)
    end = _lower_bound(work.keys, wp.int64(world + 1) * stride, contact_capacity)
    row = row_count[world]
    for i in range(start, end):
        rep = work.indices[i]
        if work.lead[rep] >= 0:
            row = _append_group(work, rep, world, row, row_capacity)
    work.planned_count[world] = row


@wp.func
def _lower_bound(keys: wp.array[wp.int64], target: wp.int64, size: int):
    lo = int(0)
    hi = size
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.func
def _append_group(work: _Workspace, representative: int, world: int, row: int, capacity: int):
    if row >= capacity:
        wp.atomic_or(work.status, 0, 8)
        return row
    work.spin_row[representative] = row
    work.spin_source[world, row] = representative
    return row + 1


@wp.kernel(enable_backward=False)
def _group_world(
    contact_capacity: int,
    row_capacity: int,
    patch_mode: int,
    row_count: wp.array[int],
    row_dropped: wp.array[int],
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    work: _Workspace,
):
    world = wp.tid()
    if row_dropped[world] != 0:
        wp.atomic_or(work.status, 0, 4)
    work.planned_count[world] = row_count[world]
    stride = wp.int64(contact_capacity + 1)
    start = _lower_bound(work.keys, wp.int64(world) * stride, contact_capacity)
    end = _lower_bound(work.keys, wp.int64(world + 1) * stride, contact_capacity)
    # Sorting by (world, original contact index) reproduces the host's ordered
    # greedy clustering without any scan over another world's witnesses.
    for i in range(start, end):
        c = work.indices[i]
        if work.eligible[c] != 1:
            continue
        first_pair = c
        chosen = int(-1)
        for j in range(start, i):
            other = work.indices[j]
            if work.eligible[other] != 1:
                continue
            same = bool(False)
            if patch_mode != 0:
                same = work.owner[c] == work.owner[other]
            else:
                same = work.shape_a[c] == work.shape_a[other] and work.shape_b[c] == work.shape_b[other]
            if not same:
                continue
            first_pair = wp.min(first_pair, other)
            if work.group[other] != other:
                continue
            compatible = bool(True)
            if patch_mode == 0:
                member = other
                while member >= 0:
                    if _host_dot3(work.normal[c], work.normal[member]) < _NORMAL_COSINE:
                        compatible = False
                    if wp.abs(_host_dot3(work.normal[c], work.point[c] - work.point[member])) > _TOUCH_TOLERANCE:
                        compatible = False
                    member = work.next_member[member]
            if compatible and chosen < 0:
                chosen = other
        if chosen < 0:
            chosen = c
        else:
            work.next_member[work.tail[chosen]] = c
            work.tail[chosen] = c
        work.group[c] = chosen
        work.first_pair[c] = first_pair

    for i in range(start, end):
        rep = work.indices[i]
        if work.group[rep] != rep:
            continue
        rejected = bool(False)
        anchor_count = int(0)
        first_anchor = int(-1)
        touching = int(-1)
        for j in range(start, end):
            c = work.indices[j]
            if patch_mode != 0 and work.eligible[c] == -1 and work.owner[c] == work.owner[rep]:
                rejected = True
            if work.group[c] != rep:
                continue
            if work.anchor[c] != 0:
                anchor_count += 1
                if first_anchor < 0:
                    first_anchor = c
            if touching < 0 and work.gap[c] <= _TOUCH_TOLERANCE:
                touching = c
        if rejected:
            continue
        # Check every non-rejected region, even when it has no touching anchor.
        # This is the host oracle's exact load-ring membership condition.
        if patch_mode != 0:
            for j in range(start, end):
                c = work.indices[j]
                if work.group[c] != rep:
                    continue
                parent = parents[world, work.slot[c]]
                member = int(-1)
                if parent >= 0 and parent < row_capacity:
                    member = work.row_contact[world, parent]
                if member < 0:
                    wp.atomic_or(work.status, 0, 16)
                elif work.group[member] != rep:
                    wp.atomic_or(work.status, 0, 16)
        if first_anchor < 0 or touching < 0:
            continue
        coefficient = mu[world, work.slot[first_anchor] + 1]
        if patch_mode != 0:
            coefficient *= float(anchor_count)
        if coefficient <= 0.0:
            continue
        work.coefficient[rep] = coefficient
        work.lead[rep] = touching

    row = row_count[world]
    for i in range(start, end):
        rep = work.indices[i]
        if patch_mode != 0:
            if work.lead[rep] >= 0:
                row = _append_group(work, rep, world, row, row_capacity)
        elif work.group[rep] == rep and work.first_pair[rep] == rep:
            # Python's dictionary emits all clusters of its first-seen shape
            # pair before the next pair, not simply global cluster-seed order.
            for j in range(i, end):
                cluster = work.indices[j]
                if work.lead[cluster] >= 0 and work.first_pair[cluster] == rep:
                    row = _append_group(work, cluster, world, row, row_capacity)
    work.planned_count[world] = row


@wp.kernel(enable_backward=False)
def _apply_rows(
    work: _Workspace,
    pgs_cfm: float,
    body_art: wp.array[int],
    prescribed: wp.array[int],
    body_velocity: wp.array[wp.spatial_vector],
    row_type: wp.array2d[int],
    parent: wp.array2d[int],
    mu: wp.array2d[float],
    beta: wp.array2d[float],
    cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target: wp.array2d[float],
    restitution: wp.array2d[float],
    membership: wp.array2d[int],
):
    c = wp.tid()
    if work.status[0] != 0:
        return
    rep = work.group[c]
    if rep < 0:
        return
    row = work.spin_row[rep]
    if row < 0:
        return
    world = work.world[c]
    membership[world, work.slot[c]] = row
    if c != rep:
        return
    lead = work.lead[rep]
    row_type[world, row] = PGS_CONSTRAINT_TYPE_TORSION
    parent[world, row] = work.slot[lead]
    mu[world, row] = work.coefficient[rep]
    beta[world, row] = 0.0
    cfm[world, row] = pgs_cfm
    phi[world, row] = 0.0
    restitution[world, row] = 0.0
    velocity = float(0.0)
    for side in range(2):
        body = work.body_a[lead]
        sign = float(1.0)
        if side == 1:
            body = work.body_b[lead]
            sign = -1.0
        if body >= 0:
            art = body_art[body]
            if art >= 0 and prescribed[art] != 0:
                omega = wp.spatial_bottom(body_velocity[body])
                velocity -= sign * _host_dot3(work.normal[lead], omega)
    target[world, row] = velocity


@wp.kernel(enable_backward=False)
def _apply_counts(work: _Workspace, count: wp.array[int], slots: wp.array[int], mf_count: wp.array[int]):
    world = wp.tid()
    if work.status[0] == 0:
        count[world] = work.planned_count[world]
        slots[world] = work.planned_count[world]
    else:
        # Do not solve an incomplete contact system while a captured error is
        # awaiting its mandatory host validation boundary.
        count[world] = 0
        slots[world] = 0
        if world < mf_count.shape[0]:
            mf_count[world] = 0


@wp.kernel(enable_backward=False)
def _rollback_state(status: wp.array[int], saved: wp.array[Any], output: wp.array[Any]):
    i = wp.tid()
    if status[0] != 0:
        output[i] = saved[i]


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def _jacobians(
    work: _Workspace,
    group_art: wp.array[int],
    art_world: wp.array[int],
    art_start: wp.array[int],
    body_art: wp.array[int],
    body_joint: wp.array[int],
    prescribed: wp.array[int],
    ancestor: wp.array[int],
    qd_start: wp.array[int],
    motion: wp.array[wp.spatial_vector],
    jacobian: wp.array3d[float],
):
    group, row, dof = wp.tid()
    if work.status[0] != 0:
        return
    art = group_art[group]
    world = art_world[art]
    rep = work.spin_source[world, row]
    if rep < 0:
        return
    lead = work.lead[rep]
    value = float(0.0)
    global_dof = art_start[art] + dof
    if prescribed[art] == 0:
        for side in range(2):
            body = work.body_a[lead]
            sign = float(1.0)
            if side == 1:
                body = work.body_b[lead]
                sign = -1.0
            if body >= 0 and body_art[body] == art:
                joint = body_joint[body]
                while joint >= 0:
                    if qd_start[joint] <= global_dof and global_dof < qd_start[joint + 1]:
                        value += sign * _host_dot3(work.normal[lead], wp.spatial_bottom(motion[global_dof]))
                    joint = ancestor[joint]
    jacobian[group, row, dof] = value


class DeviceTorsionPreparation:
    """Own fixed-capacity preparation buffers and explicit error readback."""

    def __init__(self, solver, *, deferred_errors=False, patch_segments=True):
        self.solver = solver
        self.deferred_errors = bool(deferred_errors)
        self.patch_segments = bool(patch_segments)
        self.capacity = solver._max_contacts_alloc
        device = solver.model.device
        self.work = _Workspace()
        w = self.work
        w.keys = wp.empty(2 * self.capacity, dtype=wp.int64, device=device)
        w.indices = wp.empty(2 * self.capacity, dtype=int, device=device)
        for name in (
            "world",
            "slot",
            "shape_a",
            "shape_b",
            "body_a",
            "body_b",
            "owner",
            "eligible",
            "anchor",
            "group",
            "next_member",
            "tail",
            "first_pair",
            "lead",
            "spin_row",
        ):
            setattr(w, name, wp.empty(self.capacity, dtype=int, device=device))
        for name in ("gap", "coefficient"):
            setattr(w, name, wp.empty(self.capacity, dtype=float, device=device))
        for name in ("normal", "point"):
            setattr(w, name, wp.empty(self.capacity, dtype=wp.vec3, device=device))
        shape = (solver.world_count, solver.dense_max_constraints)
        w.row_contact = wp.full(shape, -1, dtype=int, device=device)
        w.spin_source = wp.full(shape, -1, dtype=int, device=device)
        w.planned_count = wp.zeros(solver.world_count, dtype=int, device=device)
        w.status = wp.zeros(1, dtype=int, device=device)
        w.patch_keys = wp.empty(2 * self.capacity, dtype=wp.int64, device=device)
        w.patch_indices = wp.empty(2 * self.capacity, dtype=int, device=device)
        w.rejected = wp.empty(self.capacity, dtype=int, device=device)
        w.spin_row.fill_(-1)
        selected = solver._contact_torsion_shape_set
        self.selected = wp.array(
            np.array([selected is None or i in selected for i in range(solver.model.shape_count)], dtype=np.int32),
            dtype=int,
            device=device,
        )
        self.supported = wp.array(
            np.isin(solver.model.shape_type.numpy(), _SUPPORTED_TYPES).astype(np.int32), dtype=int, device=device
        )
        self._dummy_stiffness = wp.zeros(1, dtype=float, device=device)
        self._dummy_owner = wp.full(1, -1, dtype=int, device=device)
        self._dummy_mf_count = wp.zeros(solver.world_count, dtype=int, device=device)
        self._saved_state = None

    def begin_step(self, state_in, state_out):
        """Snapshot dynamic state on-device for fail-stop deferred validation."""
        if not self.deferred_errors:
            return
        names = ("body_q", "body_qd", "body_qdd", "joint_q", "joint_qd", "particle_q", "particle_qd")
        if self._saved_state is None:
            if self.solver.model.device.is_cuda and wp.get_stream(self.solver.model.device).is_capturing:
                raise RuntimeError("Warm up device torsion state buffers before graph capture")
            self._saved_state = {
                name: wp.empty_like(getattr(state_in, name))
                for name in names
                if getattr(state_in, name, None) is not None and getattr(state_out, name, None) is not None
            }
        for name, saved in self._saved_state.items():
            source = getattr(state_in, name, None)
            target = getattr(state_out, name, None)
            if source is None or target is None or source.shape != saved.shape or target.shape != saved.shape:
                raise ValueError("Device torsion state layout changed; recreate preparation and recapture")
            wp.copy(saved, source)

    def end_step(self, state_out):
        """Prevent invalid captured work from advancing published dynamic state."""
        if self.deferred_errors and self._saved_state is not None:
            for name, saved in self._saved_state.items():
                wp.launch(
                    _rollback_state,
                    dim=saved.shape,
                    inputs=[self.work.status, saved, getattr(state_out, name)],
                    device=self.solver.model.device,
                )

    def validate(self):
        """Synchronously raise latched errors; mandatory after every deferred replay batch.

        Errors remain latched: recreate the preparation after fixing invalid
        input rather than clearing evidence and continuing an invalid graph.
        """
        status = int(self.work.status.numpy()[0])
        if status & 1:
            raise RuntimeError("Contact input overflow before contact torsion")
        if status & 2:
            raise ValueError("Contact torsion does not support hydroelastic contacts")
        if status & 4:
            raise RuntimeError("Dense contact row overflow before contact torsion; refusing dropped rows")
        if status & 8:
            raise RuntimeError("Experimental torsion row capacity exceeded; refusing dropped work")
        if status & 16:
            raise RuntimeError("Persistent friction patch load ring does not match its contact torsion group")

    def read_stats(self):
        """Explicit diagnostic readback, never part of normal stepping or capture.

        Unlike the host oracle, device preparation does not refresh Python
        dictionaries each step. Return a fresh snapshot only when requested.
        """
        if self.solver.model.device.is_cuda and wp.get_stream(self.solver.model.device).is_capturing:
            raise RuntimeError("Read torsion statistics outside graph capture")
        self.validate()
        names = ("world", "slot", "group", "spin_row", "lead", "anchor", "shape_a", "shape_b", "gap", "coefficient")
        arrays = {name: getattr(self.work, name).numpy() for name in names}
        groups = []
        for representative in np.flatnonzero(arrays["spin_row"] >= 0):
            lead = arrays["lead"][representative]
            members = np.flatnonzero(arrays["group"] == representative)
            anchors = members[arrays["anchor"][members] != 0]
            touching = members[arrays["gap"][members] <= _TOUCH_TOLERANCE]
            groups.append(
                {
                    "world": int(arrays["world"][representative]),
                    "row": int(arrays["spin_row"][representative]),
                    "normal_rows": arrays["slot"][members].tolist(),
                    "anchor_rows": arrays["slot"][anchors].tolist(),
                    "shape_pair": [int(arrays["shape_a"][lead]), int(arrays["shape_b"][lead])],
                    "effective_radius_m": self.solver.contact_torsion_radius,
                    "mu": float(arrays["coefficient"][representative]),
                    "gap_max_m": float(np.max(arrays["gap"][touching])),
                }
            )
        return {"rows": len(groups), "groups": groups}

    def prepare(self, state, augmented_state, contacts):
        """Build rows entirely on device, followed by optional scalar error readback."""
        s = self.solver
        w = self.work
        device = s.model.device
        w.row_contact.fill_(-1)
        w.spin_source.fill_(-1)
        s._contact_torsion_group.fill_(-1)
        if contacts is None:
            w.spin_row.fill_(-1)
            if not self.deferred_errors:
                self.validate()
            return
        if contacts.rigid_contact_max > self.capacity:
            raise ValueError("Contact torsion capacity mismatch; recreate the solver for the new contact buffer")
        stiffness = contacts.rigid_contact_stiffness
        patches = s._friction_anchors_enabled
        wp.launch(
            _gather,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                self.capacity,
                contacts.rigid_contact_max,
                s.world_count,
                s.contact_path,
                s.contact_slot,
                s.contact_world,
                s.contact_slots_needed,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                stiffness if stiffness is not None else self._dummy_stiffness,
                int(stiffness is not None),
                s.model.shape_body,
                self.supported,
                self.selected,
                state.body_q,
                s.row_type,
                s.row_parent,
                s.constraint_count,
                int(patches),
                s._friction_patches.current.owner if patches else self._dummy_owner,
                w,
            ],
            device=device,
        )
        if patches and self.patch_segments:
            wp.launch(_patch_keys, dim=self.capacity, inputs=[w], device=device)
            wp.utils.radix_sort_pairs(w.keys, w.indices, self.capacity)
            # Stable sort keeps contact order inside each patch segment.
            wp.utils.radix_sort_pairs(w.patch_keys, w.patch_indices, self.capacity)
            wp.launch(_group_patches, dim=self.capacity, inputs=[self.capacity, s.row_mu, w], device=device)
            wp.launch(
                _check_patch_membership,
                dim=self.capacity,
                inputs=[s.dense_max_constraints, s.row_parent, w],
                device=device,
            )
            wp.launch(
                _emit_patches,
                dim=s.world_count,
                inputs=[self.capacity, s.dense_max_constraints, s.constraint_count, s._row_dropped_dense, w],
                device=device,
            )
        else:
            wp.utils.radix_sort_pairs(w.keys, w.indices, self.capacity)
            wp.launch(
                _group_world,
                dim=s.world_count,
                inputs=[
                    self.capacity,
                    s.dense_max_constraints,
                    int(patches),
                    s.constraint_count,
                    s._row_dropped_dense,
                    s.row_parent,
                    s.row_mu,
                    w,
                ],
                device=device,
            )
        if not self.deferred_errors:
            self.validate()
        wp.launch(
            _apply_rows,
            dim=self.capacity,
            inputs=[
                w,
                s.pgs_cfm,
                s.body_to_articulation,
                s._prescribed_articulation,
                augmented_state.body_v_s,
                s.row_type,
                s.row_parent,
                s.row_mu,
                s.row_beta,
                s.row_cfm,
                s.phi,
                s.target_velocity,
                s.row_restitution,
                s._contact_torsion_group,
            ],
            device=device,
        )
        for size, jacobian in s.J_by_size.items():
            wp.launch(
                _jacobians,
                dim=jacobian.shape,
                inputs=[
                    w,
                    s.group_to_art[size],
                    s.art_to_world,
                    s.articulation_dof_start,
                    s.body_to_articulation,
                    s.body_to_joint,
                    s._prescribed_articulation,
                    s.model.joint_ancestor,
                    s.model.joint_qd_start,
                    augmented_state.joint_S_s,
                    jacobian,
                ],
                device=device,
            )
        wp.launch(
            _apply_counts,
            dim=s.world_count,
            inputs=[w, s.constraint_count, s.slot_counter, getattr(s, "mf_constraint_count", self._dummy_mf_count)],
            device=device,
        )


def enable_device_torsion(solver, *, deferred_errors=False):
    """Opt into the local experimental port without changing the host default.

    The caller must invoke ``solver._device_torsion.validate()`` after every
    deferred replay batch before consuming results. Invalid input latches a
    permanent error and rolls back the published dynamic state; recreate the
    solver after fixing the input instead of reusing an invalid captured graph.
    Warm up one eager step before capturing, to allocate rollback buffers.
    """
    if not solver._contact_torsion_enabled:
        raise ValueError("Enable a positive contact_torsion_radius before device preparation")
    if wp.get_stream(solver.model.device).is_capturing:
        raise RuntimeError("Configure device torsion before CUDA graph capture")
    solver._device_torsion = DeviceTorsionPreparation(solver, deferred_errors=deferred_errors)
    solver._torsion_stats = {"device_resident": True, "rows": None, "groups": None}
    return solver._device_torsion
