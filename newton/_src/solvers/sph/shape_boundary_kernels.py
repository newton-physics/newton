# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for analytic SPH shape-boundary projection."""

import warp as wp

from ...geometry import GeoType, ParticleFlags, ShapeFlags
from ...math.spatial import quat_velocity, velocity_at_point
from .kernels import SPH_ROLE_FLUID, _same_world

wp.set_module_options({"enable_backward": False})

SPH_COLLIDER_VELOCITY_FORWARD = 0
SPH_COLLIDER_VELOCITY_BACKWARD = 1


@wp.func
def _clip_velocity_against_normal(
    velocity: wp.vec3,
    normal: wp.vec3,
    boundary_friction: float,
    shape_friction: float,
    boundary_velocity: wp.vec3,
) -> wp.vec3:
    v = velocity - boundary_velocity
    vn = wp.dot(v, normal)
    if vn < 0.0:
        v -= normal * vn
    vt = v - normal * wp.dot(v, normal)
    friction = wp.max(boundary_friction, shape_friction)
    return boundary_velocity + v - vt * wp.min(friction, 1.0)


@wp.func
def _apply_adhesion_velocity(
    velocity: wp.vec3,
    normal: wp.vec3,
    adhesion: float,
    dt: float,
    enable_adhesion: bool,
) -> wp.vec3:
    if enable_adhesion and adhesion > 0.0 and dt > 0.0:
        return velocity - normal * adhesion * dt
    return velocity


@wp.func
def _project_local_capsule(
    local: wp.vec3,
    radius: float,
    half_height: float,
    min_distance: float,
) -> tuple[bool, wp.vec3, wp.vec3]:
    expanded_radius = radius + min_distance
    axis_z = wp.clamp(local[2], -half_height, half_height)
    axis_point = wp.vec3(0.0, 0.0, axis_z)
    radial = local - axis_point
    distance = wp.length(radial)
    if distance >= expanded_radius:
        return False, local, wp.vec3(0.0)

    normal = wp.vec3(1.0, 0.0, 0.0)
    if distance > 1.0e-7:
        normal = radial / distance
    return True, axis_point + normal * expanded_radius, normal


@wp.func
def _project_local_cylinder(
    local: wp.vec3,
    radius: float,
    half_height: float,
    min_distance: float,
) -> tuple[bool, wp.vec3, wp.vec3]:
    expanded_radius = radius + min_distance
    expanded_half_height = half_height + min_distance
    radial_distance = wp.sqrt(local[0] * local[0] + local[1] * local[1])
    axial_distance = wp.abs(local[2])
    if radial_distance >= expanded_radius or axial_distance >= expanded_half_height:
        return False, local, wp.vec3(0.0)

    radial_penetration = expanded_radius - radial_distance
    axial_penetration = expanded_half_height - axial_distance
    projected = local
    normal = wp.vec3(0.0)
    if radial_penetration <= axial_penetration:
        normal_xy = wp.vec3(1.0, 0.0, 0.0)
        if radial_distance > 1.0e-7:
            normal_xy = wp.vec3(local[0] / radial_distance, local[1] / radial_distance, 0.0)
        projected = wp.vec3(normal_xy[0] * expanded_radius, normal_xy[1] * expanded_radius, local[2])
        normal = normal_xy
    else:
        sign = float(1.0)
        if local[2] < 0.0:
            sign = -1.0
        projected = wp.vec3(local[0], local[1], sign * expanded_half_height)
        normal = wp.vec3(0.0, 0.0, sign)
    return True, projected, normal


@wp.func
def _project_local_ellipsoid(
    local: wp.vec3,
    radii: wp.vec3,
    min_distance: float,
) -> tuple[bool, wp.vec3, wp.vec3]:
    rx = wp.max(wp.abs(radii[0]) + min_distance, 1.0e-7)
    ry = wp.max(wp.abs(radii[1]) + min_distance, 1.0e-7)
    rz = wp.max(wp.abs(radii[2]) + min_distance, 1.0e-7)
    sx = local[0] / rx
    sy = local[1] / ry
    sz = local[2] / rz
    scaled_distance = wp.sqrt(sx * sx + sy * sy + sz * sz)
    if scaled_distance >= 1.0:
        return False, local, wp.vec3(0.0)

    projected = wp.vec3(0.0)
    normal = wp.vec3(0.0)
    if scaled_distance <= 1.0e-7:
        axis = int(0)
        if ry <= rx and ry <= rz:
            axis = 1
        elif rz <= rx and rz <= ry:
            axis = 2
        if axis == 0:
            projected = wp.vec3(rx, 0.0, 0.0)
            normal = wp.vec3(1.0, 0.0, 0.0)
        elif axis == 1:
            projected = wp.vec3(0.0, ry, 0.0)
            normal = wp.vec3(0.0, 1.0, 0.0)
        else:
            projected = wp.vec3(0.0, 0.0, rz)
            normal = wp.vec3(0.0, 0.0, 1.0)
    else:
        projected = local / scaled_distance
        gradient = wp.vec3(projected[0] / (rx * rx), projected[1] / (ry * ry), projected[2] / (rz * rz))
        gradient_length = wp.length(gradient)
        normal = wp.vec3(0.0, 0.0, 1.0)
        if gradient_length > 1.0e-7:
            normal = gradient / gradient_length
    return True, projected, normal


@wp.func
def _project_local_cone(
    local: wp.vec3,
    radius: float,
    half_height: float,
    min_distance: float,
) -> tuple[bool, wp.vec3, wp.vec3]:
    expanded_radius = radius + min_distance
    expanded_half_height = half_height + min_distance
    if expanded_radius <= 0.0 or expanded_half_height <= 0.0:
        return False, local, wp.vec3(0.0)

    z = local[2]
    radial_distance = wp.sqrt(local[0] * local[0] + local[1] * local[1])
    if z <= -expanded_half_height or z >= expanded_half_height:
        return False, local, wp.vec3(0.0)

    allowed_radius = expanded_radius * (expanded_half_height - z) / (2.0 * expanded_half_height)
    if radial_distance >= allowed_radius:
        return False, local, wp.vec3(0.0)

    base_penetration = z + expanded_half_height
    side_penetration = allowed_radius - radial_distance
    projected = local
    normal = wp.vec3(0.0)
    if base_penetration <= side_penetration:
        projected = wp.vec3(local[0], local[1], -expanded_half_height)
        normal = wp.vec3(0.0, 0.0, -1.0)
    else:
        radial_dir = wp.vec3(1.0, 0.0, 0.0)
        if radial_distance > 1.0e-7:
            radial_dir = wp.vec3(local[0] / radial_distance, local[1] / radial_distance, 0.0)
        projected = wp.vec3(radial_dir[0] * allowed_radius, radial_dir[1] * allowed_radius, local[2])
        normal = wp.vec3(radial_dir[0], radial_dir[1], expanded_radius / (2.0 * expanded_half_height))
        normal_length = wp.length(normal)
        if normal_length > 1.0e-7:
            normal = normal / normal_length
        else:
            normal = wp.vec3(1.0, 0.0, 0.0)
    return True, projected, normal


@wp.func
def _body_velocity_at_point(
    body: int,
    x: wp.vec3,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    collider_velocity_mode: int,
    dt: float,
) -> wp.vec3:
    X_wb = body_q[body]
    com_world = wp.transform_point(X_wb, body_com[body])
    if collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD and dt > 0.0:
        X_prev = body_q_prev[body]
        com_prev = wp.transform_point(X_prev, body_com[body])
        linear_velocity = (com_world - com_prev) / dt
        angular_velocity = quat_velocity(wp.transform_get_rotation(X_wb), wp.transform_get_rotation(X_prev), dt)
        return linear_velocity + wp.cross(angular_velocity, x - com_world)
    return velocity_at_point(body_qd[body], x - com_world)


@wp.kernel
def collide_particle_shapes(
    shape_count: int,
    body_count: int,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    sph_role: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_adhesion: wp.array[float],
    shape_projection_threshold: wp.array[float],
    explicit_mesh_count: int,
    explicit_mesh_ids: wp.array[wp.uint64],
    explicit_mesh_margin: wp.array[float],
    explicit_mesh_friction: wp.array[float],
    explicit_mesh_adhesion: wp.array[float],
    explicit_mesh_projection_threshold: wp.array[float],
    explicit_mesh_body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    analytic_body_impulse: wp.array[wp.vec3],
    analytic_body_angular_impulse: wp.array[wp.vec3],
    boundary_margin: float,
    boundary_friction: float,
    collider_velocity_mode: int,
    enable_boundary_adhesion: bool,
    dt: float,
    boundary_impulse: wp.array[wp.vec3],
):
    particle = wp.tid()

    if (particle_flags[particle] & ParticleFlags.ACTIVE) == 0 or sph_role[particle] != SPH_ROLE_FLUID:
        boundary_impulse[particle] = wp.vec3(0.0)
        return

    x = particle_q[particle]
    v = particle_qd[particle]
    v_initial = v
    radius = particle_radius[particle]
    world = particle_world[particle]

    hit = bool(False)

    for shape in range(shape_count):
        if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0 or not _same_world(world, shape_world[shape]):
            continue

        shape_xform = shape_transform[shape]
        body = shape_body[shape]
        boundary_velocity = wp.vec3(0.0)
        if 0 <= body and body < body_count:
            X_wb = body_q[body]
            shape_xform = wp.transform_multiply(X_wb, shape_xform)
            boundary_velocity = _body_velocity_at_point(
                body,
                x,
                body_q,
                body_qd,
                body_q_prev,
                body_com,
                collider_velocity_mode,
                dt,
            )

        point = wp.transform_get_translation(shape_xform)
        quat = wp.transform_get_rotation(shape_xform)
        min_distance = radius + shape_margin[shape] + shape_projection_threshold[shape] + boundary_margin
        v_before_shape = v
        hit_shape = bool(False)
        normal = wp.vec3(0.0)

        if shape_type[shape] == GeoType.PLANE:
            normal = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
            normal_length = wp.length(normal)
            if normal_length <= 1.0e-7:
                continue
            normal = normal / normal_length
            signed_distance = wp.dot(normal, x - point)

            if signed_distance < min_distance:
                x += normal * (min_distance - signed_distance)
                hit_shape = True

        elif shape_type[shape] == GeoType.SPHERE:
            offset = x - point
            distance = wp.length(offset)
            target_radius = wp.abs(shape_scale[shape][0]) + min_distance
            if distance < target_radius:
                if distance > 1.0e-7:
                    normal = offset / distance
                else:
                    normal = wp.vec3(0.0, 1.0, 0.0)
                x = point + normal * target_radius
                hit_shape = True

        elif shape_type[shape] == GeoType.BOX:
            local = wp.quat_rotate_inv(quat, x - point)
            hx = wp.abs(shape_scale[shape][0]) + min_distance
            hy = wp.abs(shape_scale[shape][1]) + min_distance
            hz = wp.abs(shape_scale[shape][2]) + min_distance
            ax = wp.abs(local[0])
            ay = wp.abs(local[1])
            az = wp.abs(local[2])
            if ax < hx and ay < hy and az < hz:
                px = hx - ax
                py = hy - ay
                pz = hz - az
                projected = local
                normal_local = wp.vec3(0.0)
                if px <= py and px <= pz:
                    sign = float(1.0)
                    if local[0] < 0.0:
                        sign = -1.0
                    projected = wp.vec3(sign * hx, local[1], local[2])
                    normal_local = wp.vec3(sign, 0.0, 0.0)
                elif py <= pz:
                    sign = float(1.0)
                    if local[1] < 0.0:
                        sign = -1.0
                    projected = wp.vec3(local[0], sign * hy, local[2])
                    normal_local = wp.vec3(0.0, sign, 0.0)
                else:
                    sign = float(1.0)
                    if local[2] < 0.0:
                        sign = -1.0
                    projected = wp.vec3(local[0], local[1], sign * hz)
                    normal_local = wp.vec3(0.0, 0.0, sign)
                normal = wp.quat_rotate(quat, normal_local)
                normal_length = wp.length(normal)
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = point + wp.quat_rotate(quat, projected)
                hit_shape = True

        elif shape_type[shape] == GeoType.CAPSULE:
            local = wp.quat_rotate_inv(quat, x - point)
            hit_shape, projected_local, normal_local = _project_local_capsule(
                local,
                wp.abs(shape_scale[shape][0]),
                wp.abs(shape_scale[shape][1]),
                min_distance,
            )
            if hit_shape:
                normal = wp.quat_rotate(quat, normal_local)
                normal_length = wp.length(normal)
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = point + wp.quat_rotate(quat, projected_local)

        elif shape_type[shape] == GeoType.CYLINDER:
            local = wp.quat_rotate_inv(quat, x - point)
            hit_shape, projected_local, normal_local = _project_local_cylinder(
                local,
                wp.abs(shape_scale[shape][0]),
                wp.abs(shape_scale[shape][1]),
                min_distance,
            )
            if hit_shape:
                normal = wp.quat_rotate(quat, normal_local)
                normal_length = wp.length(normal)
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = point + wp.quat_rotate(quat, projected_local)

        elif shape_type[shape] == GeoType.ELLIPSOID:
            local = wp.quat_rotate_inv(quat, x - point)
            hit_shape, projected_local, normal_local = _project_local_ellipsoid(
                local,
                shape_scale[shape],
                min_distance,
            )
            if hit_shape:
                normal = wp.quat_rotate(quat, normal_local)
                normal_length = wp.length(normal)
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = point + wp.quat_rotate(quat, projected_local)

        elif shape_type[shape] == GeoType.CONE:
            local = wp.quat_rotate_inv(quat, x - point)
            hit_shape, projected_local, normal_local = _project_local_cone(
                local,
                wp.abs(shape_scale[shape][0]),
                wp.abs(shape_scale[shape][1]),
                min_distance,
            )
            if hit_shape:
                normal = wp.quat_rotate(quat, normal_local)
                normal_length = wp.length(normal)
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = point + wp.quat_rotate(quat, projected_local)

        elif shape_type[shape] == GeoType.MESH or shape_type[shape] == GeoType.CONVEX_MESH:
            mesh = shape_source_ptr[shape]
            if mesh == wp.uint64(0):
                continue
            local = wp.quat_rotate_inv(quat, x - point)
            mesh_scale = wp.vec3(
                wp.max(wp.abs(shape_scale[shape][0]), 1.0e-7),
                wp.max(wp.abs(shape_scale[shape][1]), 1.0e-7),
                wp.max(wp.abs(shape_scale[shape][2]), 1.0e-7),
            )
            min_scale = wp.min(mesh_scale)
            query = wp.mesh_query_point_no_sign(mesh, wp.cw_div(local, mesh_scale), min_distance / min_scale)
            if query.result:
                closest = wp.cw_mul(wp.mesh_eval_position(mesh, query.face, query.u, query.v), mesh_scale)
                normal_local = wp.cw_div(wp.mesh_eval_face_normal(mesh, query.face), mesh_scale)
                normal_length = wp.length(normal_local)
                if normal_length <= 1.0e-7:
                    continue
                normal_local = normal_local / normal_length
                if wp.dot(local - closest, normal_local) < 0.0:
                    normal_local = -normal_local
                normal = wp.quat_rotate(quat, normal_local)
                normal_length = wp.length(normal)
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = point + wp.quat_rotate(quat, closest + normal_local * min_distance)
                hit_shape = True

        if hit_shape:
            v = _clip_velocity_against_normal(
                v,
                normal,
                boundary_friction,
                shape_material_mu[shape],
                boundary_velocity,
            )
            v = _apply_adhesion_velocity(v, normal, shape_adhesion[shape], dt, enable_boundary_adhesion)
            if 0 <= body and body < body_count:
                particle_impulse = particle_mass[particle] * (v - v_before_shape)
                body_impulse = -particle_impulse
                X_wb = body_q[body]
                com_world = wp.transform_point(X_wb, body_com[body])
                wp.atomic_add(analytic_body_impulse, body, body_impulse)
                wp.atomic_add(analytic_body_angular_impulse, body, wp.cross(x - com_world, body_impulse))
            hit = True

    for mesh_index in range(explicit_mesh_count):
        mesh = explicit_mesh_ids[mesh_index]
        if mesh == wp.uint64(0):
            continue
        body = explicit_mesh_body_ids[mesh_index]
        local_x = x
        boundary_velocity = wp.vec3(0.0)
        if 0 <= body and body < body_count:
            X_wb = body_q[body]
            point = wp.transform_get_translation(X_wb)
            quat = wp.transform_get_rotation(X_wb)
            local_x = wp.quat_rotate_inv(quat, x - point)
            boundary_velocity = _body_velocity_at_point(
                body,
                x,
                body_q,
                body_qd,
                body_q_prev,
                body_com,
                collider_velocity_mode,
                dt,
            )

        min_distance = (
            radius + explicit_mesh_margin[mesh_index] + explicit_mesh_projection_threshold[mesh_index] + boundary_margin
        )
        query = wp.mesh_query_point_no_sign(mesh, local_x, min_distance)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            normal = wp.mesh_eval_face_normal(mesh, query.face)
            normal_length = wp.length(normal)
            if normal_length <= 1.0e-7:
                continue
            normal = normal / normal_length
            if wp.dot(local_x - closest, normal) < 0.0:
                normal = -normal
            projected = closest + normal * min_distance
            if 0 <= body and body < body_count:
                X_wb = body_q[body]
                point = wp.transform_get_translation(X_wb)
                quat = wp.transform_get_rotation(X_wb)
                normal = wp.quat_rotate(quat, normal)
                normal_length = wp.length(normal)
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = point + wp.quat_rotate(quat, projected)
            else:
                x = projected
            v_before_mesh = v
            v = _clip_velocity_against_normal(
                v,
                normal,
                boundary_friction,
                explicit_mesh_friction[mesh_index],
                boundary_velocity,
            )
            v = _apply_adhesion_velocity(
                v,
                normal,
                explicit_mesh_adhesion[mesh_index],
                dt,
                enable_boundary_adhesion,
            )
            if 0 <= body and body < body_count:
                particle_impulse = particle_mass[particle] * (v - v_before_mesh)
                body_impulse = -particle_impulse
                X_wb = body_q[body]
                com_world = wp.transform_point(X_wb, body_com[body])
                wp.atomic_add(analytic_body_impulse, body, body_impulse)
                wp.atomic_add(analytic_body_angular_impulse, body, wp.cross(x - com_world, body_impulse))
            hit = True

    if hit:
        particle_q[particle] = x
        particle_qd[particle] = v
        boundary_impulse[particle] = boundary_impulse[particle] + particle_mass[particle] * (v - v_initial)
