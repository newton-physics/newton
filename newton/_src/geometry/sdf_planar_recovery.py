# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prove a two-face wall patch and find its smaller directional separation."""

from functools import cache
from typing import Any

import warp as wp

from .contact_data import SHAPE_PAIR_HFIELD_BIT, ContactData
from .flags import MeshProperties
from .hashtable import HASHTABLE_EMPTY_KEY, HashTable, hashtable_find, hashtable_insert
from .sdf_texture import TextureSDFData, texture_sample_sdf_hw

# A crowded table falls back to ordinary mesh contacts, with bounded lookup work.
_RECOVERY_MAX_PROBES = wp.constant(32)


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


@wp.func
def _add_support(support: _Support, point: wp.vec3, projection: float) -> _Support:
    replace = not support.valid or projection > support.projection
    if projection == support.projection and support.valid:
        # Use the same anchor when multiple vertices attain the extreme.
        for i in range(3):
            if point[i] != support.point[i]:
                replace = point[i] < support.point[i]
                break
    if replace:
        support.point = point
        support.projection = projection
        support.valid = True
    return support


@wp.func
def _bounding_face_support(
    mesh_id: wp.uint64, direction: wp.vec3, low: wp.vec3, high: wp.vec3, scale: wp.vec3, axis: int, tolerance: float
) -> _Support:
    """Certify an isolated support vertex near a bounding face, or request a full search."""
    bound = 0.5 * (wp.dot(direction, low + high) + wp.dot(wp.abs(direction), high - low))
    width = 2.0 * tolerance / wp.abs(direction[axis])
    if direction[axis] > 0.0:
        low[axis] = high[axis] - width
    else:
        high[axis] = low[axis] + width
    faces = wp.mesh_query_aabb(mesh_id, low, high)
    face = int(0)
    anchor = wp.vec3()
    support = _Support()
    visited = int(0)
    while wp.mesh_query_aabb_next(faces, face):
        visited += 1
        if visited > 32:
            return _Support()
        for i in range(3):
            point = wp.mesh_eval_position(mesh_id, face, float(i == 0), float(i == 1))
            projection = wp.dot(point, direction)
            if bound - projection <= 2.0 * tolerance:
                if not support.valid:
                    anchor = point
                elif wp.length_sq(wp.cw_mul(point - anchor, scale)) > 0.25 * tolerance * tolerance:
                    return _Support()
                support = _add_support(support, point, projection)
    # Every vertex within tolerance of this extreme was in the queried slab.
    if not support.valid or bound - support.projection > tolerance:
        return _Support()
    return support


@wp.func
def _on_mesh_surface(mesh_id: wp.uint64, local: wp.vec3, scale: wp.vec3, inverse_scale: wp.vec3, tolerance: float):
    hit = wp.mesh_query_point_no_sign(mesh_id, local, 4.0 * tolerance * wp.max(inverse_scale))
    if not hit.result:
        return False
    surface = wp.mesh_eval_position(mesh_id, hit.face, hit.u, hit.v)
    return wp.length_sq(wp.cw_mul(surface - local, scale)) <= tolerance * tolerance


@wp.func
def planar_exit(
    wall_id: wp.uint64,
    probe_id: wp.uint64,
    probe_pose: wp.transform,
    wall_scale: wp.vec3,
    probe_scale: wp.vec3,
    probe_minimum: wp.vec3,
    probe_maximum: wp.vec3,
    bounds_lower: wp.vec3,
    bounds_upper: wp.vec3,
    wall_sdf: TextureSDFData,
    probe_sdf: TextureSDFData,
    wall_boundary_flags: wp.array[wp.uint8],
    wall_boundary_start: int,
) -> PlanarExit:
    result = PlanarExit()
    mesh = wp.mesh_get(probe_id)
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
    if probe_sdf.coarse_texture.width > 0:
        sample_point = wp.cw_mul(probe_point, probe_scale) if probe_sdf.scale_baked else probe_point
        # A rotated probe's AABB can straddle the wall before the probe
        # does. Reject marginal samples before doing the exact mesh proof.
        if texture_sample_sdf_hw(probe_sdf, sample_point) >= 0.0:
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
    # Ordinary SDF contacts handle shallow overlaps on either wall face.
    minimum_depth = wp.max(tolerance, thickness / 32.0)
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

    axis_unscaled = wp.cw_mul(axis_probe, probe_scale)
    plane_offset = wp.dot(wp.transform_get_translation(probe_pose) - face0, axis)
    min_support = _Support()
    max_support = _Support()
    if mesh.points.shape[0] > 256:
        for i in range(3):
            if wp.abs(axis_probe[i]) > 0.9999999:
                low = wp.cw_mul(probe_minimum, probe_inverse)
                high = wp.cw_mul(probe_maximum, probe_inverse)
                min_support = _bounding_face_support(probe_id, -axis_unscaled, low, high, probe_scale, i, tolerance)
                max_support = _bounding_face_support(probe_id, axis_unscaled, low, high, probe_scale, i, tolerance)
    isolated = min_support.valid and max_support.valid
    if not isolated:
        for index in range(mesh.points.shape[0]):
            point = mesh.points[index]
            projection = wp.dot(point, axis_unscaled)
            min_support = _add_support(min_support, point, -projection)
            max_support = _add_support(max_support, point, projection)
    if not min_support.valid:
        return result
    # Unreferenced vertices are included by the fallback vertex scan. Either
    # extreme can change the exit choice, even when the selected one is real.
    if not isolated and (
        not _on_mesh_surface(probe_id, min_support.point, probe_scale, probe_inverse, tolerance)
        or not _on_mesh_surface(probe_id, max_support.point, probe_scale, probe_inverse, tolerance)
    ):
        return result
    retreat = max_support.projection + plane_offset
    forward = thickness + min_support.projection - plane_offset
    if wp.min(retreat, forward) <= minimum_depth or wp.abs(retreat - forward) <= thickness * 0.125:
        return result
    normal = -axis
    support = max_support
    direction = axis_unscaled
    depth = retreat
    if forward < retreat:
        normal = axis
        support = min_support
        direction = -axis_unscaled
        depth = forward
    # A single anchor cannot replace the torque support of a broad contact.
    if not isolated:
        for index in range(mesh.points.shape[0]):
            point = mesh.points[index]
            if (
                support.projection - wp.dot(point, direction) <= tolerance
                and wp.length_sq(wp.cw_mul(point - support.point, probe_scale)) > tolerance * tolerance
            ):
                return result
    extreme = wp.transform_point(probe_pose, wp.cw_mul(support.point, probe_scale))

    # Prove that no other wall face intersects the swept probe footprint.
    # A watertight wall cannot end or have a hole inside the footprint
    # without exposing a triangle off these two planes.
    sweep = wp.cw_mul(normal * depth, wall_inverse)
    # A held contact can move slightly before the solver consumes it.
    # Include the translation allowed by active_recovery_contact and room
    # for roundoff in both body orientations across the swept probe.
    inward = wp.cw_mul(-normal * (thickness * 0.25), wall_inverse)
    padding = (thickness / 8.0 + 4.0 * tolerance) * wall_inverse
    query = wp.mesh_query_aabb(
        wall_id,
        wp.cw_mul(bounds_lower, wall_inverse) + wp.min(sweep, inward) - padding,
        wp.cw_mul(bounds_upper, wall_inverse) + wp.max(sweep, inward) + padding,
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
    if not _on_mesh_surface(wall_id, wp.cw_mul(wall_contact, wall_inverse), wall_scale, wall_inverse, tolerance):
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
    sweep_radii: wp.vec2


@wp.struct
class MeshRecoveryData:
    keys: wp.array[wp.uint64]
    active_slots: wp.array[int]
    contacts: wp.array[MeshRecoveryContact]
    generation: wp.array[int]
    deterministic: bool


@cache
def empty_mesh_recovery(device_alias: str) -> MeshRecoveryData:
    """Return empty descriptors for ordinary contacts, including during Warp tape replay."""
    data = MeshRecoveryData()
    data.keys = wp.empty(0, dtype=wp.uint64, device=device_alias)
    data.active_slots = data.generation = wp.empty(0, dtype=int, device=device_alias)
    data.contacts = wp.empty(0, dtype=MeshRecoveryContact, device=device_alias)
    return data


def allocate_mesh_recovery(capacity: int, device, *, deterministic: bool = False) -> MeshRecoveryData:
    """Allocate sparse solver metadata keyed by shape pair."""
    table = HashTable(capacity, device=device)
    data = MeshRecoveryData()
    data.keys = table.keys
    data.active_slots = table.active_slots
    data.contacts = wp.empty(table.capacity, dtype=MeshRecoveryContact, device=device)
    data.generation = wp.zeros(1, dtype=int, device=device)
    data.deterministic = deterministic
    return data


@wp.func
def reset_mesh_recovery(data: MeshRecoveryData):
    """Clear the sparse list from the previous refresh with one work item."""
    capacity = data.keys.shape[0]
    for i in range(data.active_slots[capacity]):
        data.keys[data.active_slots[i]] = HASHTABLE_EMPTY_KEY
    data.active_slots[capacity] = 0


@wp.func
def _pair_key(a: int, b: int) -> wp.uint64:
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    return (wp.uint64(lo) << wp.uint64(32)) | wp.uint64(hi)


@wp.func
def pair_was_recovered(data: MeshRecoveryData, pair: wp.vec2i) -> bool:
    if data.keys.shape[0] == 0 or data.active_slots[data.keys.shape[0]] == 0:
        return False
    return _find_published_recovery(data, pair[0], pair[1]) >= 0


@wp.func
def _recovery_probe_limit(data: MeshRecoveryData) -> int:
    return data.keys.shape[0] if data.deterministic else _RECOVERY_MAX_PROBES


@wp.func
def _find_published_recovery(data: MeshRecoveryData, a: int, b: int) -> int:
    slot = hashtable_find(_pair_key(a, b), data.keys, _recovery_probe_limit(data))
    if slot >= 0 and data.contacts[slot].thickness > 0.0:
        return slot
    return -1


@wp.func
def has_recovery_contacts(data: MeshRecoveryData, generation: int) -> bool:
    return data.keys.shape[0] > 0 and data.generation[0] == generation and data.active_slots[data.keys.shape[0]] > 0


@wp.func
def find_recovery_contact(data: MeshRecoveryData, generation: int, a: int, b: int) -> int:
    if a < 0 or b < 0 or not has_recovery_contacts(data, generation):
        return -1
    return _find_published_recovery(data, a, b)


@wp.func
def _rotation_chord(current: wp.quat, original: wp.quat) -> float:
    return wp.sqrt(wp.min(wp.length_sq(current - original), wp.length_sq(current + original)))


@wp.func
def recovery_step_depth(thickness: float, depth: float) -> float:
    """Bound a solver's positional target for a verified thin-wall contact."""
    return wp.min(thickness / 32.0, depth / 8.0)


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
    contact = data.contacts[slot]
    first = pose_a if a < b else pose_b
    second = pose_b if a < b else pose_a
    chord0 = _rotation_chord(first.q, contact.rotation0)
    chord1 = _rotation_chord(second.q, contact.rotation1)
    # A quaternion chord of length c moves any point at radius r by at
    # most 2*c*r. Bound the entire swept probe, including rotations about
    # either witness, against the padding verified by the mesh query.
    angular_motion = 2.0 * (chord0 * contact.sweep_radii[0] + chord1 * contact.sweep_radii[1])
    # Invertible transforms of two contact anchors can round differently at
    # large world offsets. Compare body displacement from the detection pose,
    # which gives zero displacement for a freshly generated contact.
    motion = (second.p - first.p) - contact.relative_translation
    if a > b:
        motion = -motion
    axial_motion = wp.dot(motion, normal)
    separation = axial_motion - contact.initial_depth
    side_motion = motion - axial_motion * normal
    if (
        separation + angular_motion >= 0.0
        or -separation + angular_motion > contact.initial_depth + contact.thickness * 0.25
    ):
        return -2
    # The swept AABB includes this lateral clearance around the entire probe.
    if wp.length(side_motion) + angular_motion > contact.thickness / 8.0:
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
        threads: int,
    ):
        pair_limit = wp.min(pairs.shape[0], count[0])
        for row in range(wp.tid(), pair_limit, threads):
            pair = pairs[row]
            if (pair[0] & SHAPE_PAIR_HFIELD_BIT) != 0:
                continue
            a = wp.min(pair[0], pair[1])
            b = wp.max(pair[0], pair[1])
            if ((mesh_properties[a] | mesh_properties[b]) & MeshProperties.SURFACE_VELOCITY) != 0:
                continue
            sdf_a = shape_sdf_index[a]
            sdf_b = shape_sdf_index[b]
            if sdf_a < 0 or sdf_b < 0 or sdf_a >= texture_sdfs.shape[0] or sdf_b >= texture_sdfs.shape[0]:
                continue
            extent_a = local_upper[a] - local_lower[a]
            extent_b = local_upper[b] - local_lower[b]
            minimum_overlap = 0.04 * wp.min(wp.min(extent_a), wp.min(extent_b))
            global_overlap = wp.min(wp.min(world_upper[a], world_upper[b]) - wp.max(world_lower[a], world_lower[b]))
            if global_overlap <= minimum_overlap:
                continue
            wall = a
            probe = b
            if wp.length_sq(extent_a) < wp.length_sq(extent_b):
                wall = b
                probe = a
            if (mesh_properties[probe] & MeshProperties.RECOVERY_PROBE) == 0:
                continue
            wall_sdf = texture_sdfs[shape_sdf_index[wall]]
            probe_sdf = texture_sdfs[shape_sdf_index[probe]]
            if wall_sdf.coarse_texture.width == 0 or probe_sdf.coarse_texture.width == 0:
                continue
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
            bounds_lower = center - projection
            bounds_upper = center + projection
            overlap = wp.min(wp.min(bounds_upper, local_upper[wall]) - wp.max(bounds_lower, local_lower[wall]))
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
                bounds_lower,
                bounds_upper,
                wall_sdf,
                probe_sdf,
                mesh_boundary_flags,
                wall_boundary_start,
            )
            if not hit.valid:
                continue
            key = _pair_key(a, b)
            slot, inserted = hashtable_insert(
                key, recovery.keys, recovery.active_slots, _recovery_probe_limit(recovery)
            )
            if not inserted:
                continue

            point_wall = wp.transform_point(X_ww, hit.wall_point)
            point_probe = wp.transform_point(X_ww, hit.probe_point)
            normal_world = wp.transform_vector(X_ww, hit.normal)
            body_a = shape_body[a]
            body_b = shape_body[b]
            contact = MeshRecoveryContact()
            pose_a = wp.transform_identity() if body_a < 0 else body_q[body_a]
            pose_b = wp.transform_identity() if body_b < 0 else body_q[body_b]
            contact.relative_translation = pose_b.p - pose_a.p
            contact.rotation0 = pose_a.q
            contact.rotation1 = pose_b.q
            contact.thickness = hit.thickness
            contact.initial_depth = hit.depth
            # Bound the entire swept probe about each body's rotation origin.
            center_offset = wp.quat_rotate(X_wp.q, 0.5 * (probe_lower + probe_upper))
            sweep_radius = wp.length(half) + hit.depth + hit.thickness
            contact.sweep_radii = wp.vec2(
                wp.length((X_wp.p - pose_a.p) + center_offset) + sweep_radius,
                wp.length((X_wp.p - pose_b.p) + center_offset) + sweep_radius,
            )
            canonical = ContactData()
            canonical.shape_a = a
            canonical.shape_b = b
            canonical.contact_point_center = 0.5 * (point_wall + point_probe)
            canonical.contact_normal_a_to_b = normal_world if wall == a else -normal_world
            canonical.contact_distance = -hit.depth
            canonical.margin_a = shape_data[a][3]
            canonical.margin_b = shape_data[b][3]
            if not writer_func(canonical, writer_data, -1):
                # Keep the claim for deduplication, but allow the ordinary manifold.
                contact.thickness = 0.0
            recovery.contacts[slot] = contact

    return compute_planar_recovery_contacts
