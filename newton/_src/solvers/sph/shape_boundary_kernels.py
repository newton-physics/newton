# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for analytic SPH shape-boundary projection."""

import warp as wp

from ...core.types import Axis
from ...geometry import GeoType, ParticleFlags, ShapeFlags
from ...geometry.kernels import (
    closest_point_plane,
    sdf_box,
    sdf_box_grad,
    sdf_capsule,
    sdf_capsule_grad,
    sdf_cone,
    sdf_cone_grad,
    sdf_cylinder,
    sdf_cylinder_grad,
    sdf_ellipsoid,
    sdf_ellipsoid_grad,
    sdf_sphere,
    sdf_sphere_grad,
)
from ...math.spatial import quat_velocity, velocity_at_point
from ...sim.enums import BodyFlags
from .kernels import _same_world

wp.set_module_options({"enable_backward": False})

SPH_COLLIDER_VELOCITY_FORWARD = 0
SPH_COLLIDER_VELOCITY_BACKWARD = 1


@wp.func
def _contact_inverse_mass(
    direction: wp.vec3,
    particle_mass: float,
    body: int,
    position: wp.vec3,
    body_q: wp.array[wp.transform],
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
) -> float:
    inverse_mass = 1.0 / wp.max(particle_mass, 1.0e-12)
    if body < 0 or body >= body_q.shape[0]:
        return inverse_mass
    if body_mass[body] <= 0.0 or (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return inverse_mass

    X_wb = body_q[body]
    rotation = wp.transform_get_rotation(X_wb)
    moment_arm = position - wp.transform_point(X_wb, body_com[body])
    torque = wp.cross(moment_arm, direction)
    angular_response = wp.quat_rotate(
        rotation,
        body_inv_inertia[body] * wp.quat_rotate_inv(rotation, torque),
    )
    return inverse_mass + 1.0 / body_mass[body] + wp.dot(wp.cross(angular_response, moment_arm), direction)


@wp.func
def _solve_coulomb_velocity(
    velocity: wp.vec3,
    normal: wp.vec3,
    boundary_friction: float,
    shape_friction: float,
    boundary_velocity: wp.vec3,
    particle_mass: float,
    body: int,
    position: wp.vec3,
    body_q: wp.array[wp.transform],
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
) -> wp.vec3:
    relative_velocity = velocity - boundary_velocity
    normal_velocity = wp.dot(relative_velocity, normal)
    if normal_velocity >= 0.0:
        return velocity

    particle_inverse_mass = 1.0 / wp.max(particle_mass, 1.0e-12)
    normal_inverse_mass = _contact_inverse_mass(
        normal, particle_mass, body, position, body_q, body_flags, body_com, body_mass, body_inv_inertia
    )
    normal_impulse = -normal_velocity / wp.max(normal_inverse_mass, 1.0e-12)
    particle_impulse = normal * normal_impulse

    tangential_velocity = relative_velocity - normal * normal_velocity
    tangential_speed = wp.length(tangential_velocity)
    friction = wp.max(boundary_friction, shape_friction)
    if tangential_speed > 1.0e-7 and friction > 0.0:
        tangent = tangential_velocity / tangential_speed
        tangent_inverse_mass = _contact_inverse_mass(
            tangent, particle_mass, body, position, body_q, body_flags, body_com, body_mass, body_inv_inertia
        )
        tangent_impulse = wp.min(tangential_speed / wp.max(tangent_inverse_mass, 1.0e-12), friction * normal_impulse)
        particle_impulse -= tangent * tangent_impulse

    return velocity + particle_impulse * particle_inverse_mass


@wp.func
def _project_signed_distance(
    local: wp.vec3,
    distance: float,
    normal: wp.vec3,
    separation: float,
    projection_threshold: float,
) -> tuple[bool, wp.vec3, wp.vec3]:
    normal_length = wp.length(normal)
    if normal_length <= 1.0e-7 or distance + projection_threshold >= separation:
        return False, local, wp.vec3(0.0)
    normal /= normal_length
    return True, local + normal * (separation - distance), normal


@wp.func
def _project_local_plane(
    local: wp.vec3,
    width: float,
    length: float,
    separation: float,
    projection_threshold: float,
) -> tuple[bool, wp.vec3, wp.vec3]:
    normal = wp.vec3(0.0, 0.0, 1.0)
    if width <= 0.0 or length <= 0.0:
        return _project_signed_distance(local, local[2], normal, separation, projection_threshold)

    closest = closest_point_plane(width, length, local)
    inside_bounds = wp.abs(local[0]) <= width and wp.abs(local[1]) <= length
    if inside_bounds:
        return _project_signed_distance(local, local[2], normal, separation, projection_threshold)

    offset = local - closest
    distance = wp.length(offset)
    if distance + projection_threshold >= separation:
        return False, local, wp.vec3(0.0)
    if distance > 1.0e-7:
        normal = offset / distance
    return True, closest + normal * separation, normal


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
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_projection_threshold: wp.array[float],
    explicit_mesh_count: int,
    explicit_mesh_ids: wp.array[wp.uint64],
    explicit_mesh_margin: wp.array[float],
    explicit_mesh_friction: wp.array[float],
    explicit_mesh_projection_threshold: wp.array[float],
    explicit_mesh_body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    analytic_body_impulse: wp.array[wp.vec3],
    analytic_body_angular_impulse: wp.array[wp.vec3],
    boundary_margin: float,
    boundary_friction: float,
    mesh_query_max_distance: float,
    collider_velocity_mode: int,
    dt: float,
    boundary_impulse: wp.array[wp.vec3],
):
    particle = wp.tid()

    if (particle_flags[particle] & ParticleFlags.ACTIVE) == 0 or particle_mass[particle] <= 0.0:
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
        local = wp.quat_rotate_inv(quat, x - point)
        separation = radius + shape_margin[shape] + boundary_margin
        projection_threshold = shape_projection_threshold[shape]
        v_before_shape = v
        hit_shape = bool(False)
        projected_local = local
        normal_local = wp.vec3(0.0)
        normal = wp.vec3(0.0)

        if shape_type[shape] == GeoType.PLANE:
            hit_shape, projected_local, normal_local = _project_local_plane(
                local,
                0.5 * wp.abs(shape_scale[shape][0]),
                0.5 * wp.abs(shape_scale[shape][1]),
                separation,
                projection_threshold,
            )

        elif shape_type[shape] == GeoType.SPHERE:
            distance = sdf_sphere(local, wp.abs(shape_scale[shape][0]))
            normal_local = sdf_sphere_grad(local, wp.abs(shape_scale[shape][0]))
            hit_shape, projected_local, normal_local = _project_signed_distance(
                local, distance, normal_local, separation, projection_threshold
            )

        elif shape_type[shape] == GeoType.BOX:
            hx = wp.abs(shape_scale[shape][0])
            hy = wp.abs(shape_scale[shape][1])
            hz = wp.abs(shape_scale[shape][2])
            distance = sdf_box(local, hx, hy, hz)
            normal_local = sdf_box_grad(local, hx, hy, hz)
            hit_shape, projected_local, normal_local = _project_signed_distance(
                local, distance, normal_local, separation, projection_threshold
            )

        elif shape_type[shape] == GeoType.CAPSULE:
            shape_radius = wp.abs(shape_scale[shape][0])
            half_height = wp.abs(shape_scale[shape][1])
            distance = sdf_capsule(local, shape_radius, half_height, int(Axis.Z))
            normal_local = sdf_capsule_grad(local, shape_radius, half_height, int(Axis.Z))
            hit_shape, projected_local, normal_local = _project_signed_distance(
                local, distance, normal_local, separation, projection_threshold
            )

        elif shape_type[shape] == GeoType.CYLINDER:
            shape_radius = wp.abs(shape_scale[shape][0])
            half_height = wp.abs(shape_scale[shape][1])
            distance = sdf_cylinder(local, shape_radius, half_height, int(Axis.Z))
            normal_local = sdf_cylinder_grad(local, shape_radius, half_height, int(Axis.Z))
            hit_shape, projected_local, normal_local = _project_signed_distance(
                local, distance, normal_local, separation, projection_threshold
            )

        elif shape_type[shape] == GeoType.ELLIPSOID:
            radii = wp.vec3(
                wp.abs(shape_scale[shape][0]),
                wp.abs(shape_scale[shape][1]),
                wp.abs(shape_scale[shape][2]),
            )
            distance = sdf_ellipsoid(local, radii)
            normal_local = sdf_ellipsoid_grad(local, radii)
            hit_shape, projected_local, normal_local = _project_signed_distance(
                local, distance, normal_local, separation, projection_threshold
            )

        elif shape_type[shape] == GeoType.CONE:
            shape_radius = wp.abs(shape_scale[shape][0])
            half_height = wp.abs(shape_scale[shape][1])
            distance = sdf_cone(local, shape_radius, half_height, int(Axis.Z))
            normal_local = sdf_cone_grad(local, shape_radius, half_height, int(Axis.Z))
            hit_shape, projected_local, normal_local = _project_signed_distance(
                local, distance, normal_local, separation, projection_threshold
            )

        elif shape_type[shape] == GeoType.MESH or shape_type[shape] == GeoType.CONVEX_MESH:
            mesh = shape_source_ptr[shape]
            if mesh == wp.uint64(0):
                continue
            mesh_scale = shape_scale[shape]
            min_scale = wp.min(
                wp.min(wp.abs(mesh_scale[0]), wp.abs(mesh_scale[1])),
                wp.abs(mesh_scale[2]),
            )
            if min_scale <= 1.0e-7:
                continue
            query_distance = wp.max(mesh_query_max_distance, separation + projection_threshold)
            query = wp.mesh_query_point_sign_parity(mesh, wp.cw_div(local, mesh_scale), query_distance / min_scale)
            if query.result:
                closest = wp.cw_mul(wp.mesh_eval_position(mesh, query.face, query.u, query.v), mesh_scale)
                offset = local - closest
                distance = wp.length(offset) * query.sign
                normal_local = wp.cw_div(wp.mesh_eval_face_normal(mesh, query.face), mesh_scale)
                if wp.length(offset) > 1.0e-7:
                    normal_local = wp.normalize(offset) * query.sign
                hit_shape, projected_local, normal_local = _project_signed_distance(
                    local, distance, normal_local, separation, projection_threshold
                )

        if hit_shape:
            normal = wp.normalize(wp.quat_rotate(quat, normal_local))
            x = point + wp.quat_rotate(quat, projected_local)
            v = _solve_coulomb_velocity(
                v,
                normal,
                boundary_friction,
                shape_material_mu[shape],
                boundary_velocity,
                particle_mass[particle],
                body,
                x,
                body_q,
                body_flags,
                body_com,
                body_mass,
                body_inv_inertia,
            )
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

        separation = radius + explicit_mesh_margin[mesh_index] + boundary_margin
        projection_threshold = explicit_mesh_projection_threshold[mesh_index]
        query_distance = wp.max(mesh_query_max_distance, separation + projection_threshold)
        query = wp.mesh_query_point_sign_parity(mesh, local_x, query_distance)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            offset = local_x - closest
            distance = wp.length(offset) * query.sign
            normal = wp.mesh_eval_face_normal(mesh, query.face)
            if wp.length(offset) > 1.0e-7:
                normal = wp.normalize(offset) * query.sign
            hit_mesh, projected, normal = _project_signed_distance(
                local_x, distance, normal, separation, projection_threshold
            )
            if not hit_mesh:
                continue
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
            v = _solve_coulomb_velocity(
                v,
                normal,
                boundary_friction,
                explicit_mesh_friction[mesh_index],
                boundary_velocity,
                particle_mass[particle],
                body,
                x,
                body_q,
                body_flags,
                body_com,
                body_mass,
                body_inv_inertia,
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
