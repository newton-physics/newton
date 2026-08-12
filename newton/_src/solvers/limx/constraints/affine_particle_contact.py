# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free surface contact between LIMX particles and affine bodies."""

from __future__ import annotations

import numpy as np
import warp as wp

from ....geometry.kernels import triangle_closest_point_barycentric
from ....sim import Model
from ....utils.mesh import MeshAdjacency
from ..affine_body import AffineBodyModel
from ..affine_types import mat1212, vec12
from .affine_body_contact import (
    _lift_weighted_affine_matrix,
    _closest_triangle_feature_normal,
    _update_affine_edge_bounds,
    _update_affine_triangle_bounds,
    _vertex_pseudo_normal,
)
from .affine_static_plane_contact import _affine_point, _lift_affine_vector
from .self_collision import _edge_edge_mollifier_is_active, _edge_pseudo_normal
from .self_collision import (
    _edge_edge_gauss_newton_diagonal_block,
    _edge_edge_gauss_newton_multiply,
    _edge_edge_mollified_residual_data,
    _edge_edge_mollified_residual_jacobian_transpose_multiply,
)

_MIN_CONTACT_DISTANCE = 1.0e-7
_EE_MOLLIFIER_THRESHOLD_SCALE = 1.0e-3


@wp.kernel
def _detect_cloth_vertex_affine_face_contacts(
    triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    cloth_surface_vertices: wp.array[int],
    affine_positions: wp.array[wp.vec3],
    affine_triangles: wp.array2d[int],
    affine_triangle_edges: wp.array2d[int],
    affine_edge_triangles: wp.array2d[int],
    affine_vertex_triangle_offsets: wp.array[int],
    affine_vertex_triangles: wp.array[int],
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    count: wp.array[int],
    overflow_count: wp.array[int],
):
    vertex = cloth_surface_vertices[wp.tid()]
    vertex_position = cloth_positions[vertex]
    query = wp.bvh_query_aabb(
        triangle_bvh_id,
        vertex_position - wp.vec3(thickness),
        vertex_position + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = affine_triangles[triangle, 0]
        index_1 = affine_triangles[triangle, 1]
        index_2 = affine_triangles[triangle, 2]
        position_0 = affine_positions[index_0]
        position_1 = affine_positions[index_1]
        position_2 = affine_positions[index_2]
        barycentric = triangle_closest_point_barycentric(
            position_0,
            position_1,
            position_2,
            vertex_position,
        )
        closest_point = barycentric[0] * position_0 + barycentric[1] * position_1 + barycentric[2] * position_2
        separation = vertex_position - closest_point
        distance = wp.length(separation)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue

        feature_normal = _closest_triangle_feature_normal(
            triangle,
            barycentric,
            affine_positions,
            affine_triangles,
            affine_triangle_edges,
            affine_edge_triangles,
            affine_vertex_triangle_offsets,
            affine_vertex_triangles,
        )
        if wp.length(feature_normal) <= _MIN_CONTACT_DISTANCE:
            continue
        direction = separation / distance
        signed_distance = distance
        if wp.dot(separation, feature_normal) < 0.0:
            direction = -direction
            signed_distance = -distance

        contact = wp.atomic_add(count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        particle_ids[contact, 0] = vertex
        particle_ids[contact, 1] = -1
        particle_ids[contact, 2] = -1
        particle_weights[contact, 0] = 1.0
        particle_weights[contact, 1] = 0.0
        particle_weights[contact, 2] = 0.0
        affine_ids[contact, 0] = index_0
        affine_ids[contact, 1] = index_1
        affine_ids[contact, 2] = index_2
        affine_weights[contact, 0] = -barycentric[0]
        affine_weights[contact, 1] = -barycentric[1]
        affine_weights[contact, 2] = -barycentric[2]
        directions[contact] = direction
        depths[contact] = thickness - signed_distance


@wp.kernel
def _detect_affine_vertex_cloth_face_contacts(
    triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    cloth_triangles: wp.array2d[int],
    affine_positions: wp.array[wp.vec3],
    affine_triangles: wp.array2d[int],
    affine_vertex_triangle_offsets: wp.array[int],
    affine_vertex_triangles: wp.array[int],
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    count: wp.array[int],
    overflow_count: wp.array[int],
):
    affine_vertex = wp.tid()
    vertex_position = affine_positions[affine_vertex]
    outward_normal = _vertex_pseudo_normal(
        affine_vertex,
        affine_positions,
        affine_triangles,
        affine_vertex_triangle_offsets,
        affine_vertex_triangles,
    )
    if wp.length(outward_normal) <= _MIN_CONTACT_DISTANCE:
        return
    query = wp.bvh_query_aabb(
        triangle_bvh_id,
        vertex_position - wp.vec3(thickness),
        vertex_position + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = cloth_triangles[triangle, 0]
        index_1 = cloth_triangles[triangle, 1]
        index_2 = cloth_triangles[triangle, 2]
        position_0 = cloth_positions[index_0]
        position_1 = cloth_positions[index_1]
        position_2 = cloth_positions[index_2]
        barycentric = triangle_closest_point_barycentric(
            position_0,
            position_1,
            position_2,
            vertex_position,
        )
        closest_point = barycentric[0] * position_0 + barycentric[1] * position_1 + barycentric[2] * position_2
        separation = vertex_position - closest_point
        distance = wp.length(separation)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue
        direction = separation / distance
        signed_distance = distance
        if wp.dot(separation, outward_normal) > 0.0:
            direction = -direction
            signed_distance = -distance

        contact = wp.atomic_add(count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        particle_ids[contact, 0] = index_0
        particle_ids[contact, 1] = index_1
        particle_ids[contact, 2] = index_2
        particle_weights[contact, 0] = -barycentric[0]
        particle_weights[contact, 1] = -barycentric[1]
        particle_weights[contact, 2] = -barycentric[2]
        affine_ids[contact, 0] = affine_vertex
        affine_ids[contact, 1] = -1
        affine_ids[contact, 2] = -1
        affine_weights[contact, 0] = 1.0
        affine_weights[contact, 1] = 0.0
        affine_weights[contact, 2] = 0.0
        directions[contact] = direction
        depths[contact] = thickness - signed_distance


@wp.kernel
def _detect_affine_cloth_edge_edge_contacts(
    cloth_edge_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    cloth_rest_positions: wp.array[wp.vec3],
    cloth_edges: wp.array2d[int],
    affine_positions: wp.array[wp.vec3],
    affine_rest_positions: wp.array[wp.vec3],
    affine_triangles: wp.array2d[int],
    affine_edges: wp.array2d[int],
    affine_edge_triangles: wp.array2d[int],
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    overflow_count: wp.array[int],
):
    affine_edge = wp.tid()
    affine_index_0 = affine_edges[affine_edge, 2]
    affine_index_1 = affine_edges[affine_edge, 3]
    affine_position_0 = affine_positions[affine_index_0]
    affine_position_1 = affine_positions[affine_index_1]
    query = wp.bvh_query_aabb(
        cloth_edge_bvh_id,
        wp.min(affine_position_0, affine_position_1) - wp.vec3(thickness),
        wp.max(affine_position_0, affine_position_1) + wp.vec3(thickness),
    )
    cloth_edge = wp.int32(-1)
    while wp.bvh_query_next(query, cloth_edge):
        particle_index_0 = cloth_edges[cloth_edge, 2]
        particle_index_1 = cloth_edges[cloth_edge, 3]
        particle_position_0 = cloth_positions[particle_index_0]
        particle_position_1 = cloth_positions[particle_index_1]
        parameters = wp.closest_point_edge_edge(
            affine_position_0,
            affine_position_1,
            particle_position_0,
            particle_position_1,
            1.0e-5,
        )
        affine_parameter = parameters[0]
        particle_parameter = parameters[1]
        if affine_parameter <= 0.0 or affine_parameter >= 1.0:
            continue
        if particle_parameter <= 0.0 or particle_parameter >= 1.0:
            continue

        closest_affine = wp.lerp(affine_position_0, affine_position_1, affine_parameter)
        closest_particle = wp.lerp(particle_position_0, particle_position_1, particle_parameter)
        separation = closest_particle - closest_affine
        distance = wp.length(separation)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue
        feature_normal = _edge_pseudo_normal(
            affine_edge,
            affine_positions,
            affine_triangles,
            affine_edge_triangles,
        )
        if wp.length(feature_normal) <= _MIN_CONTACT_DISTANCE:
            continue
        direction = separation / distance
        signed_distance = distance
        if wp.dot(separation, feature_normal) < 0.0:
            direction = -direction
            signed_distance = -distance

        contact = wp.atomic_add(count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        particle_ids[contact, 0] = particle_index_0
        particle_ids[contact, 1] = particle_index_1
        particle_ids[contact, 2] = -1
        particle_weights[contact, 0] = 1.0 - particle_parameter
        particle_weights[contact, 1] = particle_parameter
        particle_weights[contact, 2] = 0.0
        affine_ids[contact, 0] = affine_index_0
        affine_ids[contact, 1] = affine_index_1
        affine_ids[contact, 2] = -1
        affine_weights[contact, 0] = -(1.0 - affine_parameter)
        affine_weights[contact, 1] = -affine_parameter
        affine_weights[contact, 2] = 0.0
        directions[contact] = direction
        depths[contact] = thickness - signed_distance
        rest_affine_edge = affine_rest_positions[affine_index_1] - affine_rest_positions[affine_index_0]
        rest_particle_edge = cloth_rest_positions[particle_index_1] - cloth_rest_positions[particle_index_0]
        mollifier_thresholds[contact] = (
            _EE_MOLLIFIER_THRESHOLD_SCALE
            * wp.dot(rest_affine_edge, rest_affine_edge)
            * wp.dot(rest_particle_edge, rest_particle_edge)
        )
        mollifier_active[contact] = 0


@wp.kernel
def _prepare_mixed_edge_mollifier(
    particle_ids: wp.array2d[int],
    affine_ids: wp.array2d[int],
    mollifier_thresholds: wp.array[float],
    count: wp.array[int],
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    affine_positions: wp.array[wp.vec3],
    mollifier_active: wp.array[int],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    active = _edge_edge_mollifier_is_active(
        affine_positions[affine_ids[contact, 0]],
        affine_positions[affine_ids[contact, 1]],
        cloth_positions[particle_ids[contact, 0]],
        cloth_positions[particle_ids[contact, 1]],
        mollifier_thresholds[contact],
    )
    mollifier_active[contact] = wp.int32(active)


@wp.kernel
def _prepare_mixed_contact_response(
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    use_mollifier: int,
    particle_positions: wp.array[wp.vec3],
    particle_velocities: wp.array[wp.vec3],
    affine_positions: wp.array[wp.vec3],
    affine_rest_positions: wp.array[wp.vec3],
    affine_ownership: wp.array[int],
    affine_velocities: wp.array[vec12],
    stiffness: float,
    normal_damping: float,
    friction: float,
    friction_epsilon: float,
    dt: float,
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    relative_velocity = wp.vec3(0.0)
    for local_index in range(3):
        particle = particle_ids[contact, local_index]
        if particle >= 0:
            relative_velocity += particle_weights[contact, local_index] * particle_velocities[particle]
        affine_vertex = affine_ids[contact, local_index]
        if affine_vertex >= 0:
            body = affine_ownership[affine_vertex]
            point_velocity = _affine_point(affine_velocities[body], affine_rest_positions[affine_vertex])
            relative_velocity += affine_weights[contact, local_index] * point_velocity

    direction = directions[contact]
    depth = depths[contact]
    normal_outer = wp.outer(direction, direction)
    normal_scale = float(1.0)
    friction_load_scale = float(1.0)
    if use_mollifier != 0 and mollifier_active[contact] != 0:
        normal_scale = 0.0
        affine_edge = affine_positions[affine_ids[contact, 1]] - affine_positions[affine_ids[contact, 0]]
        particle_edge = particle_positions[particle_ids[contact, 1]] - particle_positions[particle_ids[contact, 0]]
        cross_product = wp.cross(affine_edge, particle_edge)
        cross_squared = wp.dot(cross_product, cross_product)
        threshold = mollifier_thresholds[contact]
        if threshold > 0.0:
            friction_load_scale = wp.clamp(
                cross_squared * (2.0 * threshold - cross_squared) / (threshold * threshold),
                0.0,
                1.0,
            )
        else:
            friction_load_scale = 0.0
    force = normal_scale * stiffness * depth * direction
    hessian = normal_scale * stiffness * normal_outer

    normal_velocity = wp.dot(direction, relative_velocity)
    if normal_velocity < 0.0 and normal_damping > 0.0:
        force -= normal_damping * normal_velocity * direction
        hessian += normal_damping / dt * normal_outer

    tangent = wp.identity(3, float) - normal_outer
    tangent_displacement = dt * tangent * relative_velocity
    tangent_length = wp.length(tangent_displacement)
    friction_over_length = float(0.0)
    if tangent_length > friction_epsilon:
        friction_over_length = 1.0 / tangent_length
    else:
        friction_over_length = (-tangent_length / friction_epsilon + 2.0) / friction_epsilon
    alpha = friction_load_scale * friction * stiffness * depth * friction_over_length
    force -= alpha * tangent_displacement
    hessian += alpha * tangent
    forces[contact] = force
    hessians[contact] = hessian


@wp.kernel
def _accumulate_mixed_contact_force(
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    forces: wp.array[wp.vec3],
    count: wp.array[int],
    capacity: int,
    affine_rest_positions: wp.array[wp.vec3],
    affine_ownership: wp.array[int],
    particle_output: wp.array[wp.vec3],
    affine_output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    force = forces[contact]
    for local_index in range(3):
        particle = particle_ids[contact, local_index]
        if particle >= 0:
            wp.atomic_add(
                particle_output,
                particle,
                particle_weights[contact, local_index] * force,
            )
        affine_vertex = affine_ids[contact, local_index]
        if affine_vertex >= 0:
            body = affine_ownership[affine_vertex]
            lifted = _lift_affine_vector(
                affine_weights[contact, local_index] * force,
                affine_rest_positions[affine_vertex],
            )
            wp.atomic_add(affine_output, body, lifted)


@wp.kernel
def _mixed_contact_hessian_multiply(
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    hessians: wp.array[wp.mat33],
    count: wp.array[int],
    capacity: int,
    affine_rest_positions: wp.array[wp.vec3],
    affine_ownership: wp.array[int],
    particle_input: wp.array[wp.vec3],
    affine_input: wp.array[vec12],
    particle_output: wp.array[wp.vec3],
    affine_output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    world_vector = wp.vec3(0.0)
    for local_index in range(3):
        particle = particle_ids[contact, local_index]
        if particle >= 0:
            world_vector += particle_weights[contact, local_index] * particle_input[particle]
        affine_vertex = affine_ids[contact, local_index]
        if affine_vertex >= 0:
            body = affine_ownership[affine_vertex]
            point_vector = _affine_point(affine_input[body], affine_rest_positions[affine_vertex])
            world_vector += affine_weights[contact, local_index] * point_vector
    world_product = hessians[contact] * world_vector
    for local_index in range(3):
        particle = particle_ids[contact, local_index]
        if particle >= 0:
            wp.atomic_add(
                particle_output,
                particle,
                particle_weights[contact, local_index] * world_product,
            )
        affine_vertex = affine_ids[contact, local_index]
        if affine_vertex >= 0:
            body = affine_ownership[affine_vertex]
            lifted = _lift_affine_vector(
                affine_weights[contact, local_index] * world_product,
                affine_rest_positions[affine_vertex],
            )
            wp.atomic_add(affine_output, body, lifted)


@wp.kernel
def _accumulate_mixed_contact_diagonal(
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    hessians: wp.array[wp.mat33],
    count: wp.array[int],
    capacity: int,
    affine_rest_positions: wp.array[wp.vec3],
    affine_ownership: wp.array[int],
    particle_output: wp.array[wp.mat33],
    affine_output: wp.array[mat1212],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    hessian = hessians[contact]
    for local_index in range(3):
        particle = particle_ids[contact, local_index]
        if particle >= 0:
            weight = particle_weights[contact, local_index]
            wp.atomic_add(particle_output, particle, weight * weight * hessian)

    first_affine_vertex = affine_ids[contact, 0]
    if first_affine_vertex < 0:
        first_affine_vertex = affine_ids[contact, 1]
    if first_affine_vertex < 0:
        first_affine_vertex = affine_ids[contact, 2]
    body = affine_ownership[first_affine_vertex]
    translation_weight = float(0.0)
    rest_weight = wp.vec3(0.0)
    for local_index in range(3):
        affine_vertex = affine_ids[contact, local_index]
        if affine_vertex >= 0 and affine_ownership[affine_vertex] == body:
            weight = affine_weights[contact, local_index]
            translation_weight += weight
            rest_weight += weight * affine_rest_positions[affine_vertex]
    block = _lift_weighted_affine_matrix(hessian, translation_weight, rest_weight)
    wp.atomic_add(affine_output, body, block)


@wp.kernel
def _accumulate_mollified_mixed_edge_force(
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    particle_positions: wp.array[wp.vec3],
    affine_positions: wp.array[wp.vec3],
    affine_rest_positions: wp.array[wp.vec3],
    affine_ownership: wp.array[int],
    particle_output: wp.array[wp.vec3],
    affine_output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity) or mollifier_active[contact] == 0:
        return
    affine_index_0 = affine_ids[contact, 0]
    affine_index_1 = affine_ids[contact, 1]
    particle_index_0 = particle_ids[contact, 0]
    particle_index_1 = particle_ids[contact, 1]
    edge_0 = affine_positions[affine_index_1] - affine_positions[affine_index_0]
    edge_1 = particle_positions[particle_index_1] - particle_positions[particle_index_0]
    weights = wp.vec4(
        affine_weights[contact, 0],
        affine_weights[contact, 1],
        particle_weights[contact, 0],
        particle_weights[contact, 1],
    )
    direction = directions[contact]
    depth = depths[contact]
    threshold = mollifier_thresholds[contact]
    cross_product, residual_scale, _scale_gradient = _edge_edge_mollified_residual_data(edge_0, edge_1, threshold)
    edge_product_0, edge_product_1, depth_product = _edge_edge_mollified_residual_jacobian_transpose_multiply(
        edge_0,
        edge_1,
        depth,
        threshold,
        depth * residual_scale * cross_product,
    )
    gradient_0 = -edge_product_0 - weights[0] * depth_product * direction
    gradient_1 = edge_product_0 - weights[1] * depth_product * direction
    gradient_2 = -edge_product_1 - weights[2] * depth_product * direction
    gradient_3 = edge_product_1 - weights[3] * depth_product * direction
    body = affine_ownership[affine_index_0]
    wp.atomic_add(
        affine_output,
        body,
        _lift_affine_vector(-stiffness * gradient_0, affine_rest_positions[affine_index_0]),
    )
    wp.atomic_add(
        affine_output,
        body,
        _lift_affine_vector(-stiffness * gradient_1, affine_rest_positions[affine_index_1]),
    )
    wp.atomic_add(particle_output, particle_index_0, -stiffness * gradient_2)
    wp.atomic_add(particle_output, particle_index_1, -stiffness * gradient_3)


@wp.kernel
def _mollified_mixed_edge_hessian_multiply(
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    particle_positions: wp.array[wp.vec3],
    affine_positions: wp.array[wp.vec3],
    affine_rest_positions: wp.array[wp.vec3],
    affine_ownership: wp.array[int],
    particle_input: wp.array[wp.vec3],
    affine_input: wp.array[vec12],
    particle_output: wp.array[wp.vec3],
    affine_output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity) or mollifier_active[contact] == 0:
        return
    affine_index_0 = affine_ids[contact, 0]
    affine_index_1 = affine_ids[contact, 1]
    particle_index_0 = particle_ids[contact, 0]
    particle_index_1 = particle_ids[contact, 1]
    body = affine_ownership[affine_index_0]
    weights = wp.vec4(
        affine_weights[contact, 0],
        affine_weights[contact, 1],
        particle_weights[contact, 0],
        particle_weights[contact, 1],
    )
    product_0, product_1, product_2, product_3 = _edge_edge_gauss_newton_multiply(
        affine_positions[affine_index_1] - affine_positions[affine_index_0],
        particle_positions[particle_index_1] - particle_positions[particle_index_0],
        weights,
        directions[contact],
        depths[contact],
        mollifier_thresholds[contact],
        _affine_point(affine_input[body], affine_rest_positions[affine_index_0]),
        _affine_point(affine_input[body], affine_rest_positions[affine_index_1]),
        particle_input[particle_index_0],
        particle_input[particle_index_1],
    )
    wp.atomic_add(
        affine_output,
        body,
        stiffness
        * (
            _lift_affine_vector(product_0, affine_rest_positions[affine_index_0])
            + _lift_affine_vector(product_1, affine_rest_positions[affine_index_1])
        ),
    )
    wp.atomic_add(particle_output, particle_index_0, stiffness * product_2)
    wp.atomic_add(particle_output, particle_index_1, stiffness * product_3)


@wp.kernel
def _accumulate_mollified_mixed_edge_diagonal(
    particle_ids: wp.array2d[int],
    particle_weights: wp.array2d[float],
    affine_ids: wp.array2d[int],
    affine_weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    particle_positions: wp.array[wp.vec3],
    affine_positions: wp.array[wp.vec3],
    affine_rest_positions: wp.array[wp.vec3],
    affine_ownership: wp.array[int],
    particle_output: wp.array[wp.mat33],
    affine_output: wp.array[mat1212],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity) or mollifier_active[contact] == 0:
        return
    affine_index_0 = affine_ids[contact, 0]
    affine_index_1 = affine_ids[contact, 1]
    particle_index_0 = particle_ids[contact, 0]
    particle_index_1 = particle_ids[contact, 1]
    body = affine_ownership[affine_index_0]
    edge_0 = affine_positions[affine_index_1] - affine_positions[affine_index_0]
    edge_1 = particle_positions[particle_index_1] - particle_positions[particle_index_0]
    weights = wp.vec4(
        affine_weights[contact, 0],
        affine_weights[contact, 1],
        particle_weights[contact, 0],
        particle_weights[contact, 1],
    )
    wp.atomic_add(
        particle_output,
        particle_index_0,
        stiffness
        * _edge_edge_gauss_newton_diagonal_block(
            edge_0,
            edge_1,
            weights[2],
            directions[contact],
            depths[contact],
            mollifier_thresholds[contact],
            2,
        ),
    )
    wp.atomic_add(
        particle_output,
        particle_index_1,
        stiffness
        * _edge_edge_gauss_newton_diagonal_block(
            edge_0,
            edge_1,
            weights[3],
            directions[contact],
            depths[contact],
            mollifier_thresholds[contact],
            3,
        ),
    )

    block = mat1212(0.0)
    for column in range(12):
        vector_0 = wp.vec3(0.0)
        vector_1 = wp.vec3(0.0)
        if column < 3:
            vector_0[column] = 1.0
            vector_1[column] = 1.0
        else:
            axis = (column - 3) // 3
            coordinate = (column - 3) % 3
            vector_0[axis] = affine_rest_positions[affine_index_0][coordinate]
            vector_1[axis] = affine_rest_positions[affine_index_1][coordinate]
        product_0, product_1, _product_2, _product_3 = _edge_edge_gauss_newton_multiply(
            edge_0,
            edge_1,
            weights,
            directions[contact],
            depths[contact],
            mollifier_thresholds[contact],
            vector_0,
            vector_1,
            wp.vec3(0.0),
            wp.vec3(0.0),
        )
        generalized_product = _lift_affine_vector(product_0, affine_rest_positions[affine_index_0])
        generalized_product += _lift_affine_vector(product_1, affine_rest_positions[affine_index_1])
        for row in range(12):
            block[row, column] = stiffness * generalized_product[row]
    wp.atomic_add(affine_output, body, block)


class _MixedContactBuffer:
    def __init__(self, capacity: int, device: wp.context.Device):
        self.capacity = capacity
        self.device = device
        self.particle_ids = wp.empty((capacity, 3), dtype=wp.int32, device=device)
        self.particle_weights = wp.empty((capacity, 3), dtype=wp.float32, device=device)
        self.affine_ids = wp.empty((capacity, 3), dtype=wp.int32, device=device)
        self.affine_weights = wp.empty((capacity, 3), dtype=wp.float32, device=device)
        self.directions = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.depths = wp.empty(capacity, dtype=wp.float32, device=device)
        self.forces = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.hessians = wp.empty(capacity, dtype=wp.mat33, device=device)
        self.mollifier_thresholds = wp.zeros(capacity, dtype=wp.float32, device=device)
        self.mollifier_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.overflow_count = wp.zeros(1, dtype=wp.int32, device=device)

    def clear(self) -> None:
        self.count.zero_()
        self.overflow_count.zero_()


class ConstraintAffineParticleContact:
    """Detect mixed VF and strict-interior EE contact between particles and affine bodies."""

    def __init__(
        self,
        particle_model: Model,
        body_model: AffineBodyModel,
        thickness: float,
        stiffness: float,
        normal_damping: float,
        friction: float,
        friction_epsilon: float,
        max_contacts: int = 262144,
    ):
        """Create particle-affine surface contact.

        Args:
            particle_model: Particle model containing the triangle surface.
            body_model: Affine-body model containing closed outward surfaces.
            thickness: Contact activation distance [m].
            stiffness: Normal penalty stiffness [N/m].
            normal_damping: Approaching normal-velocity damping [N·s/m].
            friction: Coulomb friction coefficient.
            friction_epsilon: Tangential displacement regularization [m].
            max_contacts: Capacity of each mixed contact family.
        """
        if not isinstance(particle_model, Model):
            raise TypeError("particle_model must be a Model")
        if not isinstance(body_model, AffineBodyModel):
            raise TypeError("body_model must be an AffineBodyModel")
        if particle_model.particle_count <= 0 or particle_model.tri_count <= 0 or particle_model.tri_indices is None:
            raise ValueError("particle_model must contain a particle triangle surface")
        if wp.get_device(particle_model.device) != body_model.device:
            raise ValueError("particle_model and body_model must use the same device")
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("thickness must be finite and positive")
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("stiffness must be finite and positive")
        if not np.isfinite(normal_damping) or normal_damping < 0.0:
            raise ValueError("normal_damping must be finite and nonnegative")
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(friction_epsilon) or friction_epsilon <= 0.0:
            raise ValueError("friction_epsilon must be finite and positive")
        if max_contacts <= 0:
            raise ValueError("max_contacts must be positive")

        self.particle_model = particle_model
        self.body_model = body_model
        self.particle_count = particle_model.particle_count
        self.body_count = body_model.body_count
        self.device = body_model.device
        self.thickness = float(thickness)
        self.stiffness = float(stiffness)
        self.normal_damping = float(normal_damping)
        self.friction = float(friction)
        self.friction_epsilon = float(friction_epsilon)
        self.max_contacts = int(max_contacts)

        cloth_triangles = np.asarray(particle_model.tri_indices.numpy(), dtype=np.int32).reshape(-1, 3)
        cloth_adjacency = MeshAdjacency(cloth_triangles)
        cloth_edges = cloth_adjacency.edge_indices
        affine_triangles = np.asarray(body_model.surface_triangle_indices.numpy(), dtype=np.int32).reshape(-1, 3)
        affine_adjacency = MeshAdjacency(affine_triangles)
        affine_edges = affine_adjacency.edge_indices
        if np.any(affine_adjacency.edge_tri_indices < 0):
            raise ValueError("Affine tetrahedral collision surfaces must be closed")

        self.cloth_triangles = wp.array(cloth_triangles, dtype=wp.int32, device=self.device)
        self.cloth_edges = wp.array(cloth_edges, dtype=wp.int32, device=self.device)
        self.cloth_surface_vertices = wp.array(
            np.unique(cloth_triangles.reshape(-1)).astype(np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.affine_triangles = wp.array(affine_triangles, dtype=wp.int32, device=self.device)
        self.affine_edges = wp.array(affine_edges, dtype=wp.int32, device=self.device)
        self.affine_triangle_edges = wp.array(
            affine_adjacency.tri_edge_indices,
            dtype=wp.int32,
            device=self.device,
        )
        self.affine_edge_triangles = wp.array(
            affine_adjacency.edge_tri_indices,
            dtype=wp.int32,
            device=self.device,
        )
        vertex_triangles = [[] for _ in range(body_model.surface_vertex_count)]
        for triangle, indices in enumerate(affine_triangles):
            for vertex in indices:
                vertex_triangles[int(vertex)].append(triangle)
        vertex_triangle_offsets = np.zeros(body_model.surface_vertex_count + 1, dtype=np.int32)
        vertex_triangle_offsets[1:] = np.cumsum([len(indices) for indices in vertex_triangles], dtype=np.int32)
        vertex_triangle_indices = np.fromiter(
            (triangle for triangles in vertex_triangles for triangle in triangles),
            dtype=np.int32,
            count=int(vertex_triangle_offsets[-1]),
        )
        self.affine_vertex_triangle_offsets = wp.array(
            vertex_triangle_offsets,
            dtype=wp.int32,
            device=self.device,
        )
        self.affine_vertex_triangles = wp.array(
            vertex_triangle_indices,
            dtype=wp.int32,
            device=self.device,
        )

        self.cloth_rest_positions = wp.clone(particle_model.particle_q)
        self.affine_positions = wp.empty(body_model.surface_vertex_count, dtype=wp.vec3, device=self.device)
        body_model.update_surface_positions(body_model.q, self.affine_positions)
        self.cloth_triangle_lower = wp.empty(len(cloth_triangles), dtype=wp.vec3, device=self.device)
        self.cloth_triangle_upper = wp.empty_like(self.cloth_triangle_lower)
        self.cloth_edge_lower = wp.empty(len(cloth_edges), dtype=wp.vec3, device=self.device)
        self.cloth_edge_upper = wp.empty_like(self.cloth_edge_lower)
        self.affine_triangle_lower = wp.empty(len(affine_triangles), dtype=wp.vec3, device=self.device)
        self.affine_triangle_upper = wp.empty_like(self.affine_triangle_lower)
        self._update_bounds(particle_model.particle_q)
        self.cloth_triangle_bvh = wp.Bvh(self.cloth_triangle_lower, self.cloth_triangle_upper)
        self.cloth_edge_bvh = wp.Bvh(self.cloth_edge_lower, self.cloth_edge_upper)
        self.affine_triangle_bvh = wp.Bvh(self.affine_triangle_lower, self.affine_triangle_upper)

        self.cloth_vertex_face_contacts = _MixedContactBuffer(self.max_contacts, self.device)
        self.affine_vertex_face_contacts = _MixedContactBuffer(self.max_contacts, self.device)
        self.edge_edge_contacts = _MixedContactBuffer(self.max_contacts, self.device)
        self._particle_positions: wp.array[wp.vec3] | None = None
        self._particle_velocities: wp.array[wp.vec3] | None = None
        self._affine_velocities: wp.array[vec12] | None = None
        self._dt = 0.0
        self._prepared = False

    def begin_step(
        self,
        particle_q: wp.array[wp.vec3],
        particle_qd: wp.array[wp.vec3],
        affine_q: wp.array[vec12],
        affine_qd: wp.array[vec12],
        dt: float,
    ) -> None:
        """Cache step-start velocities for mixed contact response."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self._validate_particle_vector(particle_q, "particle_q")
        self._validate_particle_vector(particle_qd, "particle_qd")
        self._validate_affine_vector(affine_q, "affine_q")
        self._validate_affine_vector(affine_qd, "affine_qd")
        self._particle_velocities = particle_qd
        self._affine_velocities = affine_qd
        self._dt = float(dt)
        self._prepared = False

    def prepare(self, particle_q: wp.array[wp.vec3], affine_q: wp.array[vec12]) -> None:
        """Reconstruct both surfaces and freeze mixed contacts."""
        if self._particle_velocities is None or self._affine_velocities is None:
            raise RuntimeError("begin_step() must be called before prepare()")
        self._validate_particle_vector(particle_q, "particle_q")
        self._validate_affine_vector(affine_q, "affine_q")
        self._particle_positions = particle_q
        self.body_model.update_surface_positions(affine_q, self.affine_positions)
        self._update_bounds(particle_q)
        self.cloth_triangle_bvh.refit()
        self.cloth_edge_bvh.refit()
        self.affine_triangle_bvh.refit()
        for buffer in (
            self.cloth_vertex_face_contacts,
            self.affine_vertex_face_contacts,
            self.edge_edge_contacts,
        ):
            buffer.clear()

        wp.launch(
            _detect_cloth_vertex_affine_face_contacts,
            dim=len(self.cloth_surface_vertices),
            inputs=[
                self.affine_triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                particle_q,
                self.cloth_surface_vertices,
                self.affine_positions,
                self.affine_triangles,
                self.affine_triangle_edges,
                self.affine_edge_triangles,
                self.affine_vertex_triangle_offsets,
                self.affine_vertex_triangles,
            ],
            outputs=[
                self.cloth_vertex_face_contacts.particle_ids,
                self.cloth_vertex_face_contacts.particle_weights,
                self.cloth_vertex_face_contacts.affine_ids,
                self.cloth_vertex_face_contacts.affine_weights,
                self.cloth_vertex_face_contacts.directions,
                self.cloth_vertex_face_contacts.depths,
                self.cloth_vertex_face_contacts.count,
                self.cloth_vertex_face_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            _detect_affine_vertex_cloth_face_contacts,
            dim=self.body_model.surface_vertex_count,
            inputs=[
                self.cloth_triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                particle_q,
                self.cloth_triangles,
                self.affine_positions,
                self.affine_triangles,
                self.affine_vertex_triangle_offsets,
                self.affine_vertex_triangles,
            ],
            outputs=[
                self.affine_vertex_face_contacts.particle_ids,
                self.affine_vertex_face_contacts.particle_weights,
                self.affine_vertex_face_contacts.affine_ids,
                self.affine_vertex_face_contacts.affine_weights,
                self.affine_vertex_face_contacts.directions,
                self.affine_vertex_face_contacts.depths,
                self.affine_vertex_face_contacts.count,
                self.affine_vertex_face_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            _detect_affine_cloth_edge_edge_contacts,
            dim=len(self.affine_edges),
            inputs=[
                self.cloth_edge_bvh.id,
                self.thickness,
                self.max_contacts,
                particle_q,
                self.cloth_rest_positions,
                self.cloth_edges,
                self.affine_positions,
                self.body_model.rest_surface_vertices,
                self.affine_triangles,
                self.affine_edges,
                self.affine_edge_triangles,
            ],
            outputs=[
                self.edge_edge_contacts.particle_ids,
                self.edge_edge_contacts.particle_weights,
                self.edge_edge_contacts.affine_ids,
                self.edge_edge_contacts.affine_weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.mollifier_active,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            _prepare_mixed_edge_mollifier,
            dim=self.edge_edge_contacts.capacity,
            inputs=[
                self.edge_edge_contacts.particle_ids,
                self.edge_edge_contacts.affine_ids,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.capacity,
                particle_q,
                self.affine_positions,
            ],
            outputs=[self.edge_edge_contacts.mollifier_active],
            device=self.device,
        )
        self._prepare_response(self.cloth_vertex_face_contacts, use_mollifier=False)
        self._prepare_response(self.affine_vertex_face_contacts, use_mollifier=False)
        self._prepare_response(self.edge_edge_contacts, use_mollifier=True)
        self._prepared = True

    def accumulate_force(
        self,
        particle_q: wp.array[wp.vec3],
        affine_q: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Accumulate frozen contact forces into both native domains."""
        self._require_prepared()
        self._validate_particle_vector(particle_q, "particle_q")
        self._validate_affine_vector(affine_q, "affine_q")
        self._validate_particle_vector(particle_output, "particle_output")
        self._validate_affine_vector(affine_output, "affine_output")
        for buffer in (
            self.cloth_vertex_face_contacts,
            self.affine_vertex_face_contacts,
            self.edge_edge_contacts,
        ):
            wp.launch(
                _accumulate_mixed_contact_force,
                dim=buffer.capacity,
                inputs=[
                    buffer.particle_ids,
                    buffer.particle_weights,
                    buffer.affine_ids,
                    buffer.affine_weights,
                    buffer.forces,
                    buffer.count,
                    buffer.capacity,
                    self.body_model.rest_surface_vertices,
                    self.body_model.surface_ownership,
                ],
                outputs=[particle_output, affine_output],
                device=self.device,
            )
        buffer = self.edge_edge_contacts
        wp.launch(
            _accumulate_mollified_mixed_edge_force,
            dim=buffer.capacity,
            inputs=[
                buffer.particle_ids,
                buffer.particle_weights,
                buffer.affine_ids,
                buffer.affine_weights,
                buffer.directions,
                buffer.depths,
                buffer.mollifier_thresholds,
                buffer.mollifier_active,
                buffer.count,
                buffer.capacity,
                self.stiffness,
                self._particle_positions,
                self.affine_positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
            ],
            outputs=[particle_output, affine_output],
            device=self.device,
        )

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Accumulate the complete mixed contact Hessian-vector product."""
        self._require_prepared()
        self._validate_particle_vector(particle_input, "particle_input")
        self._validate_affine_vector(affine_input, "affine_input")
        self._validate_particle_vector(particle_output, "particle_output")
        self._validate_affine_vector(affine_output, "affine_output")
        for buffer in (
            self.cloth_vertex_face_contacts,
            self.affine_vertex_face_contacts,
            self.edge_edge_contacts,
        ):
            wp.launch(
                _mixed_contact_hessian_multiply,
                dim=buffer.capacity,
                inputs=[
                    buffer.particle_ids,
                    buffer.particle_weights,
                    buffer.affine_ids,
                    buffer.affine_weights,
                    buffer.hessians,
                    buffer.count,
                    buffer.capacity,
                    self.body_model.rest_surface_vertices,
                    self.body_model.surface_ownership,
                    particle_input,
                    affine_input,
                ],
                outputs=[particle_output, affine_output],
                device=self.device,
            )
        buffer = self.edge_edge_contacts
        wp.launch(
            _mollified_mixed_edge_hessian_multiply,
            dim=buffer.capacity,
            inputs=[
                buffer.particle_ids,
                buffer.particle_weights,
                buffer.affine_ids,
                buffer.affine_weights,
                buffer.directions,
                buffer.depths,
                buffer.mollifier_thresholds,
                buffer.mollifier_active,
                buffer.count,
                buffer.capacity,
                self.stiffness,
                self._particle_positions,
                self.affine_positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                particle_input,
                affine_input,
            ],
            outputs=[particle_output, affine_output],
            device=self.device,
        )

    def accumulate_diagonal(
        self,
        particle_diagonal: wp.array[wp.mat33],
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        """Accumulate exact native block-Jacobi contact diagonals."""
        self._require_prepared()
        if (
            len(particle_diagonal) != self.particle_count
            or particle_diagonal.device != self.device
            or particle_diagonal.dtype != wp.mat33
        ):
            raise ValueError("particle_diagonal must match the particle domain")
        if (
            len(affine_diagonal) != self.body_count
            or affine_diagonal.device != self.device
            or affine_diagonal.dtype != mat1212
        ):
            raise ValueError("affine_diagonal must match the affine domain")
        for buffer in (
            self.cloth_vertex_face_contacts,
            self.affine_vertex_face_contacts,
            self.edge_edge_contacts,
        ):
            wp.launch(
                _accumulate_mixed_contact_diagonal,
                dim=buffer.capacity,
                inputs=[
                    buffer.particle_ids,
                    buffer.particle_weights,
                    buffer.affine_ids,
                    buffer.affine_weights,
                    buffer.hessians,
                    buffer.count,
                    buffer.capacity,
                    self.body_model.rest_surface_vertices,
                    self.body_model.surface_ownership,
                ],
                outputs=[particle_diagonal, affine_diagonal],
                device=self.device,
            )
        buffer = self.edge_edge_contacts
        wp.launch(
            _accumulate_mollified_mixed_edge_diagonal,
            dim=buffer.capacity,
            inputs=[
                buffer.particle_ids,
                buffer.particle_weights,
                buffer.affine_ids,
                buffer.affine_weights,
                buffer.directions,
                buffer.depths,
                buffer.mollifier_thresholds,
                buffer.mollifier_active,
                buffer.count,
                buffer.capacity,
                self.stiffness,
                self._particle_positions,
                self.affine_positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
            ],
            outputs=[particle_diagonal, affine_diagonal],
            device=self.device,
        )

    def _prepare_response(self, buffer: _MixedContactBuffer, *, use_mollifier: bool) -> None:
        wp.launch(
            _prepare_mixed_contact_response,
            dim=buffer.capacity,
            inputs=[
                buffer.particle_ids,
                buffer.particle_weights,
                buffer.affine_ids,
                buffer.affine_weights,
                buffer.directions,
                buffer.depths,
                buffer.mollifier_thresholds,
                buffer.mollifier_active,
                buffer.count,
                buffer.capacity,
                int(use_mollifier),
                self._particle_positions,
                self._particle_velocities,
                self.affine_positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                self._affine_velocities,
                self.stiffness,
                self.normal_damping,
                self.friction,
                self.friction_epsilon,
                self._dt,
            ],
            outputs=[buffer.forces, buffer.hessians],
            device=self.device,
        )

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before contact assembly")

    def _update_bounds(self, particle_q: wp.array[wp.vec3]) -> None:
        wp.launch(
            _update_affine_triangle_bounds,
            dim=len(self.cloth_triangles),
            inputs=[particle_q, self.cloth_triangles],
            outputs=[self.cloth_triangle_lower, self.cloth_triangle_upper],
            device=self.device,
        )
        wp.launch(
            _update_affine_edge_bounds,
            dim=len(self.cloth_edges),
            inputs=[particle_q, self.cloth_edges],
            outputs=[self.cloth_edge_lower, self.cloth_edge_upper],
            device=self.device,
        )
        wp.launch(
            _update_affine_triangle_bounds,
            dim=len(self.affine_triangles),
            inputs=[self.affine_positions, self.affine_triangles],
            outputs=[self.affine_triangle_lower, self.affine_triangle_upper],
            device=self.device,
        )

    def _validate_particle_vector(self, vector: wp.array[wp.vec3], name: str) -> None:
        if len(vector) != self.particle_count or vector.device != self.device or vector.dtype != wp.vec3:
            raise ValueError(f"{name} must contain {self.particle_count} vec3 values on {self.device}")

    def _validate_affine_vector(self, vector: wp.array[vec12], name: str) -> None:
        if len(vector) != self.body_count or vector.device != self.device or vector.dtype != vec12:
            raise ValueError(f"{name} must contain {self.body_count} vec12 values on {self.device}")
