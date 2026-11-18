# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prove a two-face wall patch and find its smaller directional separation."""

from functools import cache
from typing import Any

import warp as wp

from .contact_data import SHAPE_PAIR_HFIELD_BIT, ContactData
from .flags import MeshProperties
from .hashtable import HASHTABLE_EMPTY_KEY, HashTable, hashtable_find, hashtable_find_or_insert
from .sdf_texture import TextureSDFData, texture_sample_sdf_hw


@wp.struct
class PlanarExit:
    valid: bool
    wall_point: wp.vec3
    probe_point: wp.vec3
    normal: wp.vec3
    depth: float
    thickness: float


@wp.func
def _to_mesh(xform: wp.transform, inv_scale: wp.vec3, point: wp.vec3) -> wp.vec3:
    return wp.cw_mul(wp.transform_point(wp.transform_inverse(xform), point), inv_scale)


@wp.func
def _axis_segment(mesh_id: wp.uint64, local_start: wp.vec3, local_end: wp.vec3):
    delta = local_end - local_start
    length = wp.length(delta)
    return wp.mesh_query_ray(mesh_id, local_start, delta / length, length)


@wp.func_native("""
return wp::mesh_query_inside_parity(mesh_id, point, wp::vec3(1.0f, 1.0f, 1.0f), 1, 0.1f) < 0.0f;
""")
def _mesh_contains(mesh_id: wp.uint64, point: wp.vec3) -> bool:
    """Use Warp's parity test directly when a closest triangle is unnecessary."""
    ...


@wp.struct
class _Support:
    valid: bool
    projection: float
    point: wp.vec3
    wide: bool


@wp.func
def _add_support(support: _Support, point: wp.vec3, scale: wp.vec3, projection: float, tolerance: float) -> _Support:
    offset_sq = wp.length_sq(wp.cw_mul(point - support.point, scale))
    separated = offset_sq > tolerance * tolerance
    if not support.valid or projection > support.projection:
        support.wide = support.valid and projection - support.projection <= tolerance and (support.wide or separated)
        support.point = point
        support.projection = projection
        support.valid = True
    elif support.projection - projection <= tolerance and separated:
        support.wide = True
    return support


@wp.func
def _aligned_support(
    mesh_id: wp.uint64,
    direction: wp.vec3,
    lower: wp.vec3,
    upper: wp.vec3,
    scale: wp.vec3,
    axis: int,
    tolerance: float,
) -> _Support:
    """Find support at one bounding face without scanning the full mesh."""
    direction_abs = wp.abs(direction)
    extent = upper - lower
    off_axis = wp.dot(direction_abs, extent) - direction_abs[axis] * extent[axis]
    width = (tolerance + wp.max(0.0, off_axis)) / direction_abs[axis]
    bounds_min = lower
    bounds_max = upper
    corner = wp.vec3(
        upper[0] if direction[0] >= 0.0 else lower[0],
        upper[1] if direction[1] >= 0.0 else lower[1],
        upper[2] if direction[2] >= 0.0 else lower[2],
    )
    if direction[axis] >= 0.0:
        bounds_min[axis] = upper[axis] - width
    else:
        bounds_max[axis] = lower[axis] + width
    query = wp.mesh_query_aabb(mesh_id, bounds_min, bounds_max)
    result = _Support()
    face = int(0)
    visited = int(0)
    while wp.mesh_query_aabb_next(query, face):
        visited += 1
        if visited > 32:
            return _Support()
        for i in range(3):
            point = wp.mesh_eval_position(mesh_id, face, float(i == 0), float(i == 1))
            result = _add_support(result, point, scale, wp.dot(point, direction), tolerance)
    if result.valid and wp.dot(corner, direction) - result.projection > tolerance:
        return _Support()
    return result


@wp.func
def planar_exit(
    wall_id: wp.uint64,
    probe_id: wp.uint64,
    probe_pose: wp.transform,
    wall_scale: wp.vec3,
    probe_scale: wp.vec3,
    probe_minimum: wp.vec3,
    probe_maximum: wp.vec3,
    wall_sdf: TextureSDFData,
    probe_sdf: TextureSDFData,
    wall_boundary_flags: wp.array[wp.uint8],
    wall_boundary_start: int,
    continuing: bool = False,
) -> PlanarExit:
    result = PlanarExit()
    probe_center = wp.transform_point(probe_pose, 0.5 * (probe_minimum + probe_maximum))
    probe_extent = probe_maximum - probe_minimum
    radius = 0.5 * wp.length(probe_extent)
    if radius <= 0.0:
        return result
    tolerance = wp.max(radius * 1.0e-5, 1.0e-9)
    wall_inverse = wp.cw_div(wp.vec3(1.0), wall_scale)
    probe_inverse = wp.cw_div(wp.vec3(1.0), probe_scale)
    local_center = wp.cw_mul(probe_center, wall_inverse)
    closest = wp.mesh_query_point_no_sign(wall_id, local_center, radius * wp.max(wall_inverse))
    if not closest.result:
        return result

    local_face_normal = wp.mesh_eval_face_normal(wall_id, closest.face)
    axis = wp.normalize(wp.cw_mul(local_face_normal, wall_inverse))
    first_normal = axis
    local_first = wp.mesh_eval_position(wall_id, closest.face, closest.u, closest.v)
    face0 = wp.cw_mul(local_first, wall_scale)
    probe_point = _to_mesh(probe_pose, probe_inverse, face0)
    if not continuing and probe_sdf.coarse_texture.width > 0:
        sample_point = wp.cw_mul(probe_point, probe_scale) if probe_sdf.scale_baked else probe_point
        # A rotated probe's AABB can straddle the wall before the probe
        # does. Reject marginal samples before doing the exact mesh proof.
        if texture_sample_sdf_hw(probe_sdf, sample_point) >= -0.5 * probe_sdf.voxel_radius:
            return result
    start_distance = 8.0 * tolerance
    local_axis = wp.cw_mul(axis, wall_inverse)
    plus = local_first + start_distance * local_axis
    minus = local_first - start_distance * local_axis
    positive_hit = _axis_segment(wall_id, plus, local_first + 0.5 * radius * local_axis)
    negative_hit = _axis_segment(wall_id, minus, local_first - 0.5 * radius * local_axis)
    wall_end = wp.vec3()
    second_face = int(-1)
    found = int(0)
    for side in range(2):
        hit = positive_hit if side == 0 else negative_hit
        if hit.result:
            endpoint = wp.mesh_eval_position(wall_id, hit.face, hit.u, hit.v)
            midpoint = 0.5 * (endpoint + local_first)
            if _mesh_contains(wall_id, midpoint):
                wall_end = endpoint
                second_face = hit.face
                found += 1
    if found != 1:
        return result
    face1 = wp.cw_mul(wall_end, wall_scale)
    if wp.dot(face1 - face0, axis) < 0.0:
        axis = -axis
    thickness = wp.dot(face1 - face0, axis)
    if thickness <= tolerance or thickness > 0.5 * radius:
        return result

    axis_probe = wp.quat_rotate_inv(wp.transform_get_rotation(probe_pose), axis)
    projected_radius = 0.5 * wp.dot(wp.abs(axis_probe), probe_extent)
    projected_center = wp.dot(probe_center - face0, axis)
    # Hand the pair back to ordinary contacts once the remaining overlap is
    # smaller than one bounded correction, even if it remains under load.
    minimum_depth = wp.max(tolerance, thickness / 32.0) if continuing else thickness * 0.35
    if wp.min(projected_center + projected_radius, thickness - projected_center + projected_radius) <= minimum_depth:
        return result

    if wall_boundary_start >= 0:
        far_local = wp.mesh_eval_face_normal(wall_id, second_face)
        far_normal = wp.normalize(wp.cw_mul(far_local, wall_inverse))
        if wp.dot(first_normal, axis) >= -0.99 or wp.dot(far_normal, axis) <= 0.99:
            return result
        midpoint = 0.5 * (wall_end + local_first)
        wall_width = wp.abs(wp.dot(wall_end - local_first, local_face_normal))
        if wall_sdf.scale_baked:
            midpoint = wp.cw_mul(midpoint, wall_scale)
            wall_width = thickness
        if texture_sample_sdf_hw(wall_sdf, midpoint) >= -0.125 * wall_width:
            return result

    # The selected wall face is inside the probe; projection alone cannot
    # detect a gap or a solid with a hole along the chosen direction.
    if not _mesh_contains(probe_id, probe_point):
        return result

    mesh = wp.mesh_get(probe_id)
    if mesh.points.shape[0] > 4096:
        return result
    axis_unscaled = wp.cw_mul(axis_probe, probe_scale)
    plane_offset = wp.dot(wp.transform_get_translation(probe_pose) - face0, axis)
    min_support = _Support()
    max_support = _Support()
    for axis_index in range(3):
        if wp.abs(axis_probe[axis_index]) > 0.9999999:
            low = wp.cw_mul(probe_minimum, probe_inverse)
            high = wp.cw_mul(probe_maximum, probe_inverse)
            min_support = _aligned_support(probe_id, -axis_unscaled, low, high, probe_scale, axis_index, tolerance)
            max_support = _aligned_support(probe_id, axis_unscaled, low, high, probe_scale, axis_index, tolerance)
    if not min_support.valid or not max_support.valid:
        min_support = _Support()
        max_support = _Support()
        for index in range(mesh.points.shape[0]):
            point = mesh.points[index]
            proj = wp.dot(point, axis_unscaled)
            min_support = _add_support(min_support, point, probe_scale, -proj, tolerance)
            max_support = _add_support(max_support, point, probe_scale, proj, tolerance)
    if not min_support.valid or not max_support.valid:
        return result
    tolerance_sq = tolerance * tolerance
    retreat = max_support.projection + plane_offset
    forward = thickness + min_support.projection - plane_offset
    if wp.min(retreat, forward) <= minimum_depth or wp.abs(retreat - forward) <= thickness * 0.125:
        return result
    # A single anchor cannot replace the torque support of a broad contact.
    if (retreat < forward and max_support.wide) or (forward < retreat and min_support.wide):
        return result
    normal = -axis
    extreme = max_support.point
    depth = retreat
    if forward < retreat:
        normal = axis
        extreme = min_support.point
        depth = forward
    # Require a triangle witness because the vertex buffer may contain
    # points that are not referenced by the mesh.
    checked_probe = wp.mesh_query_point_no_sign(probe_id, extreme, tolerance * wp.max(probe_inverse))
    if not checked_probe.result:
        return result
    triangle_point = wp.mesh_eval_position(probe_id, checked_probe.face, checked_probe.u, checked_probe.v)
    if wp.length_sq(wp.cw_mul(triangle_point - extreme, probe_scale)) > tolerance_sq:
        return result
    extreme = wp.transform_point(probe_pose, wp.cw_mul(extreme, probe_scale))

    # Prove that no other wall face intersects the swept probe footprint.
    # A watertight wall cannot end or have a hole inside the footprint
    # without exposing a triangle off these two planes.
    lower = wp.vec3(1.0e30)
    upper = wp.vec3(-1.0e30)
    for corner in range(8):
        x = probe_minimum[0] if corner & 1 == 0 else probe_maximum[0]
        y = probe_minimum[1] if corner & 2 == 0 else probe_maximum[1]
        z = probe_minimum[2] if corner & 4 == 0 else probe_maximum[2]
        p = wp.transform_point(probe_pose, wp.vec3(x, y, z))
        q = wp.cw_mul(p, wall_inverse)
        lower = wp.min(lower, q)
        upper = wp.max(upper, q)
    sweep = wp.cw_mul(normal * depth, wall_inverse)
    # A held contact can move slightly before the solver consumes it.
    # Include the translation allowed by active_recovery_contact and room
    # for roundoff in both body orientations across the swept probe.
    inward = wp.cw_mul(-normal * (thickness * 0.25), wall_inverse)
    padding = (4.0 * tolerance) * wall_inverse
    query = wp.mesh_query_aabb(
        wall_id,
        lower + wp.min(sweep, inward) - padding,
        upper + wp.max(sweep, inward) + padding,
    )
    face = int(0)
    face_count = int(0)
    while wp.mesh_query_aabb_next(query, face):
        face_count += 1
        if face_count > 128:
            return result
        if wall_boundary_start >= 0 and wall_boundary_flags[wall_boundary_start + face] != wp.uint8(0):
            return result
        face0_error = float(0.0)
        face1_error = float(0.0)
        for corner in range(3):
            point = wp.mesh_eval_position(wall_id, face, float(corner == 0), float(corner == 1))
            proj = wp.dot(wp.cw_mul(point - local_first, wall_scale), axis)
            face0_error = wp.max(face0_error, wp.abs(proj))
            face1_error = wp.max(face1_error, wp.abs(proj - thickness))
        if wp.min(face0_error, face1_error) > tolerance:
            return result

    # The support vertex projects onto a real triangle of the intended face.
    wall_contact = extreme + normal * depth
    checked = wp.mesh_query_point_no_sign(
        wall_id,
        wp.cw_mul(wall_contact, wall_inverse),
        4.0 * tolerance * wp.max(wall_inverse),
    )
    if not checked.result:
        return result
    confirmed = wp.cw_mul(wp.mesh_eval_position(wall_id, checked.face, checked.u, checked.v), wall_scale)
    if wp.length_sq(confirmed - wall_contact) > tolerance * tolerance:
        return result

    result.valid = True
    result.wall_point = wall_contact
    result.probe_point = extreme
    result.normal = normal
    result.depth = depth
    result.thickness = thickness
    return result


@wp.struct
class MeshRecoveryContact:
    relative_translation: wp.vec3
    thickness: float
    initial_depth: float
    rotation0: wp.quat
    rotation1: wp.quat


@wp.struct
class MeshRecoveryData:
    keys: wp.array[wp.uint64]
    active_slots: wp.array[int]
    previous_keys: wp.array[wp.uint64]
    previous_slots: wp.array[int]
    contacts: wp.array[MeshRecoveryContact]
    generation: wp.array[int]


class MeshRecoveryContacts:
    """Store solver stabilization data for verified deep contacts."""

    def __init__(self, capacity, device):
        self.table = HashTable(capacity, device=device)
        self.data = MeshRecoveryData()
        self.data.keys = self.table.keys
        self.data.active_slots = self.table.active_slots
        previous = HashTable(capacity, device=device)
        self.data.previous_keys = previous.keys
        self.data.previous_slots = previous.active_slots
        self.data.contacts = wp.empty(self.table.capacity, dtype=MeshRecoveryContact, device=device)
        self.data.generation = wp.zeros(1, dtype=int, device=device)


@wp.func
def reset_mesh_recovery(data: MeshRecoveryData):
    """Clear the sparse list from the previous refresh with one work item."""
    capacity = data.keys.shape[0]
    for i in range(data.previous_slots[capacity]):
        data.previous_keys[data.previous_slots[i]] = HASHTABLE_EMPTY_KEY
    data.previous_slots[capacity] = data.active_slots[capacity]
    for i in range(data.active_slots[capacity]):
        slot = data.active_slots[i]
        data.previous_slots[i] = slot
        data.previous_keys[slot] = data.keys[slot]
        data.keys[slot] = HASHTABLE_EMPTY_KEY
    data.active_slots[capacity] = 0


@wp.func
def data_has_previous_pair(data: MeshRecoveryData, key: wp.uint64) -> bool:
    # Only the pair identity persists. Always verify the current geometry.
    return data.previous_slots[data.keys.shape[0]] > 0 and hashtable_find(key, data.previous_keys) >= 0


@wp.func
def _pair_key(a: int, b: int) -> wp.uint64:
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    return (wp.uint64(lo) << wp.uint64(32)) | wp.uint64(hi)


@wp.func
def has_recovery_contacts(data: MeshRecoveryData, generation: int) -> bool:
    return data.keys.shape[0] > 0 and data.generation[0] == generation and data.active_slots[data.keys.shape[0]] > 0


@wp.func
def find_recovery_contact(data: MeshRecoveryData, generation: int, a: int, b: int) -> int:
    if a < 0 or b < 0 or not has_recovery_contacts(data, generation):
        return -1
    return hashtable_find(_pair_key(a, b), data.keys)


@wp.func
def get_recovery_contact(data: MeshRecoveryData, slot: int, a: int, b: int) -> MeshRecoveryContact:
    contact = data.contacts[slot]
    if a > b:
        first_rotation = contact.rotation0
        contact.rotation0 = contact.rotation1
        contact.rotation1 = first_rotation
        contact.relative_translation = -contact.relative_translation
    return contact


@wp.func
def _orientation_changed(current: wp.quat, original: wp.quat) -> bool:
    return wp.min(wp.length_sq(current - original), wp.length_sq(current + original)) > 1.0e-12


@wp.func
def active_recovery_contact(
    data: MeshRecoveryData,
    generation: int,
    a: int,
    b: int,
    pose_a: wp.transform,
    pose_b: wp.transform,
    normal: wp.vec3,
) -> int:
    """Return the slot, -1 for an ordinary pair, or -2 for a stale recovery contact."""
    slot = find_recovery_contact(data, generation, a, b)
    if slot < 0:
        return slot
    contact = get_recovery_contact(data, slot, a, b)
    rot_a = wp.transform_get_rotation(pose_a)
    rot_b = wp.transform_get_rotation(pose_b)
    # Two anchors alone cannot detect rotation around either witness.
    # Keep the certified direction only while the original orientations hold.
    if _orientation_changed(rot_a, contact.rotation0) or _orientation_changed(rot_b, contact.rotation1):
        return -2
    # Invertible transforms of two contact anchors can round differently at
    # large world offsets. Compare body displacement from the detection pose,
    # which gives zero displacement for a freshly generated contact.
    relative_translation = wp.transform_get_translation(pose_b) - wp.transform_get_translation(pose_a)
    motion = relative_translation - contact.relative_translation
    axial_motion = wp.dot(motion, normal)
    separation = axial_motion - contact.initial_depth
    side_motion = motion - axial_motion * normal
    if separation >= 0.0 or -separation > contact.initial_depth + contact.thickness * 0.25:
        return -2
    # Thickness is at most half the probe radius. This lateral allowance
    # fits inside the roundoff padding used by the swept mesh query.
    if wp.length_sq(side_motion) > (contact.thickness * 1.0e-5) ** 2.0:
        return -2
    return slot


@cache
def get_compute_planar_recovery_contacts(writer_func):
    """Build the prepass that replaces an eligible manifold with one verified exit."""
    module = f"sdf_planar_recovery_{writer_func.__name__}"

    @wp.kernel(enable_backward=False, module=module)
    def compute_planar_recovery_contacts(
        pairs: wp.array[wp.vec2i],
        count: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        mesh_properties: wp.array[int],
        mesh_boundary_flags: wp.array[wp.uint8],
        shape_boundary_start: wp.array[int],
        shape_sdf_index: wp.array[int],
        texture_sdfs: wp.array[TextureSDFData],
        local_lower: wp.array[wp.vec3],
        local_upper: wp.array[wp.vec3],
        world_lower: wp.array[wp.vec3],
        world_upper: wp.array[wp.vec3],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        recovery: MeshRecoveryData,
        writer_data: Any,
        skipped_pairs: wp.array[wp.bool],
        threads: int,
    ):
        pair_limit = wp.min(pairs.shape[0], count[0])
        for row in range(wp.tid(), pair_limit, threads):
            skipped_pairs[row] = False
            pair = pairs[row]
            a = pair[0]
            b = pair[1]
            if (a & SHAPE_PAIR_HFIELD_BIT) != 0:
                continue
            if a > b:
                a = pair[1]
                b = pair[0]
            if ((mesh_properties[a] | mesh_properties[b]) & MeshProperties.SURFACE_VELOCITY) != 0:
                continue
            sdf_a = shape_sdf_index[a]
            sdf_b = shape_sdf_index[b]
            if sdf_a < 0 or sdf_b < 0 or sdf_a >= texture_sdfs.shape[0] or sdf_b >= texture_sdfs.shape[0]:
                continue
            extent_a = local_upper[a] - local_lower[a]
            extent_b = local_upper[b] - local_lower[b]
            key = _pair_key(a, b)
            continuing = data_has_previous_pair(recovery, key)
            minimum_overlap = 0.0 if continuing else 0.04 * wp.min(wp.min(extent_a), wp.min(extent_b))
            global_overlap = wp.min(wp.min(world_upper[a], world_upper[b]) - wp.max(world_lower[a], world_lower[b]))
            if global_overlap <= minimum_overlap:
                continue
            wall = a
            probe = b
            if wp.length_sq(extent_a) < wp.length_sq(extent_b):
                wall = b
                probe = a
            if (mesh_properties[probe] & MeshProperties.WATERTIGHT) == 0:
                continue
            wall_sdf = texture_sdfs[shape_sdf_index[wall]]
            wall_boundary_start = shape_boundary_start[wall]
            if (mesh_properties[wall] & MeshProperties.WATERTIGHT) == 0 and wall_boundary_start < 0:
                continue

            wall_data = shape_data[wall]
            probe_data = shape_data[probe]
            wall_scale = wp.vec3(wall_data[0], wall_data[1], wall_data[2])
            probe_scale = wp.vec3(probe_data[0], probe_data[1], probe_data[2])
            if not (wp.min(wall_scale) > 0.0 and wp.min(probe_scale) > 0.0):
                continue
            wall_id = shape_source[wall]
            probe_id = shape_source[probe]
            if wall_id == wp.uint64(0) or probe_id == wp.uint64(0):
                continue
            X_ww = shape_transform[wall]
            X_wp = shape_transform[probe]
            wall_rotation = wp.transform_get_rotation(X_ww)
            relative_translation = wp.quat_rotate_inv(
                wall_rotation, wp.transform_get_translation(X_wp) - wp.transform_get_translation(X_ww)
            )
            probe_sdf = texture_sdfs[shape_sdf_index[probe]]
            if probe_sdf.mesh_radius > 0.0:
                sphere = wp.vec3(probe_sdf.mesh_radius * wp.max(probe_scale))
                sphere_overlap = wp.min(
                    wp.min(relative_translation + sphere, local_upper[wall])
                    - wp.max(relative_translation - sphere, local_lower[wall])
                )
                if sphere_overlap <= minimum_overlap:
                    continue
            relative_rotation = wp.quat_inverse(wall_rotation) * wp.transform_get_rotation(X_wp)
            probe_lower = local_lower[probe]
            probe_upper = local_upper[probe]
            center = relative_translation + wp.quat_rotate(relative_rotation, 0.5 * (probe_lower + probe_upper))
            half = 0.5 * (probe_upper - probe_lower)
            projection = (
                wp.abs(wp.quat_rotate(relative_rotation, wp.vec3(half[0], 0.0, 0.0)))
                + wp.abs(wp.quat_rotate(relative_rotation, wp.vec3(0.0, half[1], 0.0)))
                + wp.abs(wp.quat_rotate(relative_rotation, wp.vec3(0.0, 0.0, half[2])))
            )
            overlap = wp.min(
                wp.min(center + projection, local_upper[wall]) - wp.max(center - projection, local_lower[wall])
            )
            if overlap <= minimum_overlap:
                continue
            hit = planar_exit(
                wall_id,
                probe_id,
                wp.transform(relative_translation, relative_rotation),
                wall_scale,
                probe_scale,
                probe_lower,
                probe_upper,
                wall_sdf,
                probe_sdf,
                mesh_boundary_flags,
                wall_boundary_start,
                continuing,
            )
            if not hit.valid:
                continue
            slot = hashtable_find_or_insert(key, recovery.keys, recovery.active_slots)
            if slot < 0:
                continue

            point_wall = wp.transform_point(X_ww, hit.wall_point)
            point_probe = wp.transform_point(X_ww, hit.probe_point)
            normal_world = wp.transform_vector(X_ww, hit.normal)
            body_a = shape_body[a]
            body_b = shape_body[b]
            contact = MeshRecoveryContact()
            point_a = point_wall if wall == a else point_probe
            point_b = point_probe if wall == a else point_wall
            contact.rotation0 = wp.quat_identity()
            contact.rotation1 = wp.quat_identity()
            if body_a >= 0:
                contact.relative_translation = -wp.transform_get_translation(body_q[body_a])
                contact.rotation0 = wp.transform_get_rotation(body_q[body_a])
            if body_b >= 0:
                contact.relative_translation += wp.transform_get_translation(body_q[body_b])
                contact.rotation1 = wp.transform_get_rotation(body_q[body_b])
            contact.thickness = hit.thickness
            contact.initial_depth = hit.depth
            recovery.contacts[slot] = contact
            canonical = ContactData()
            canonical.shape_a = a
            canonical.shape_b = b
            canonical.contact_point_center = 0.5 * (point_a + point_b)
            canonical.contact_normal_a_to_b = normal_world if wall == a else -normal_world
            canonical.contact_distance = -hit.depth
            canonical.margin_a = shape_data[a][3]
            canonical.margin_b = shape_data[b][3]
            writer_func(canonical, writer_data, -1)
            skipped_pairs[row] = True

    return compute_planar_recovery_contacts
