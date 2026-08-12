# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free surface contact between LIMX affine bodies."""

from __future__ import annotations

import numpy as np
import warp as wp

from ....geometry.kernels import triangle_closest_point_barycentric
from ....utils.mesh import MeshAdjacency
from ..affine_body import AffineBodyModel
from ..affine_types import mat1212, vec12
from .affine_static_plane_contact import _affine_point, _lift_affine_vector
from .self_collision import (
    _edge_edge_gauss_newton_multiply,
    _edge_edge_mollified_residual_data,
    _edge_edge_mollified_residual_jacobian_transpose_multiply,
    _edge_pseudo_normal,
    _prepare_edge_edge_mollifier,
    _triangle_unit_normal,
)

_MIN_CONTACT_DISTANCE = 1.0e-7
_FEATURE_WEIGHT_EPSILON = 1.0e-6
_EE_MOLLIFIER_THRESHOLD_SCALE = 1.0e-3


@wp.func
def _contact_jacobian_axis(index: int) -> int:
    if index < 3:
        return index
    return (index - 3) // 3


@wp.func
def _contact_jacobian_weight(index: int, translation_weight: float, rest_weight: wp.vec3) -> float:
    if index < 3:
        return translation_weight
    return rest_weight[(index - 3) % 3]


@wp.func
def _lift_weighted_affine_matrix(
    world_matrix: wp.mat33,
    translation_weight: float,
    rest_weight: wp.vec3,
) -> mat1212:
    result = mat1212(0.0)
    for row in range(12):
        world_row = _contact_jacobian_axis(row)
        row_weight = _contact_jacobian_weight(row, translation_weight, rest_weight)
        for column in range(12):
            world_column = _contact_jacobian_axis(column)
            column_weight = _contact_jacobian_weight(column, translation_weight, rest_weight)
            result[row, column] = row_weight * world_matrix[world_row, world_column] * column_weight
    return result


@wp.func
def _affine_point_basis(rest_position: wp.vec3, column: int) -> wp.vec3:
    result = wp.vec3(0.0)
    if column < 3:
        result[column] = 1.0
    else:
        axis = (column - 3) // 3
        coordinate = (column - 3) % 3
        result[axis] = rest_position[coordinate]
    return result


@wp.func
def _vertex_pseudo_normal(
    vertex: int,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    vertex_triangle_offsets: wp.array[int],
    vertex_triangle_indices: wp.array[int],
) -> wp.vec3:
    normal = wp.vec3(0.0)
    for cursor in range(vertex_triangle_offsets[vertex], vertex_triangle_offsets[vertex + 1]):
        normal += _triangle_unit_normal(vertex_triangle_indices[cursor], positions, triangle_indices)
    normal_length = wp.length(normal)
    if normal_length <= _MIN_CONTACT_DISTANCE:
        return wp.vec3(0.0)
    return normal / normal_length


@wp.func
def _closest_triangle_feature_normal(
    triangle: int,
    barycentric: wp.vec3,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    triangle_edge_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
    vertex_triangle_offsets: wp.array[int],
    vertex_triangle_indices: wp.array[int],
) -> wp.vec3:
    zero_0 = barycentric[0] <= _FEATURE_WEIGHT_EPSILON
    zero_1 = barycentric[1] <= _FEATURE_WEIGHT_EPSILON
    zero_2 = barycentric[2] <= _FEATURE_WEIGHT_EPSILON
    zero_count = int(zero_0) + int(zero_1) + int(zero_2)
    if zero_count == 0:
        return _triangle_unit_normal(triangle, positions, triangle_indices)
    if zero_count == 1:
        local_edge = int(1)
        if zero_1:
            local_edge = 2
        elif zero_2:
            local_edge = 0
        return _edge_pseudo_normal(
            triangle_edge_indices[triangle, local_edge],
            positions,
            triangle_indices,
            edge_triangle_indices,
        )

    local_vertex = int(0)
    if barycentric[1] > barycentric[local_vertex]:
        local_vertex = 1
    if barycentric[2] > barycentric[local_vertex]:
        local_vertex = 2
    return _vertex_pseudo_normal(
        triangle_indices[triangle, local_vertex],
        positions,
        triangle_indices,
        vertex_triangle_offsets,
        vertex_triangle_indices,
    )


@wp.kernel
def _update_affine_triangle_bounds(
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    triangle = wp.tid()
    position_0 = positions[triangle_indices[triangle, 0]]
    position_1 = positions[triangle_indices[triangle, 1]]
    position_2 = positions[triangle_indices[triangle, 2]]
    lower_bounds[triangle] = wp.min(wp.min(position_0, position_1), position_2)
    upper_bounds[triangle] = wp.max(wp.max(position_0, position_1), position_2)


@wp.kernel
def _update_affine_edge_bounds(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    edge = wp.tid()
    position_0 = positions[edge_indices[edge, 2]]
    position_1 = positions[edge_indices[edge, 3]]
    lower_bounds[edge] = wp.min(position_0, position_1)
    upper_bounds[edge] = wp.max(position_0, position_1)


@wp.kernel
def _detect_affine_vertex_face_contacts(
    triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    triangle_indices: wp.array2d[int],
    triangle_edge_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
    vertex_triangle_offsets: wp.array[int],
    vertex_triangle_indices: wp.array[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    vertex = wp.tid()
    vertex_position = positions[vertex]
    query = wp.bvh_query_aabb(
        triangle_bvh_id,
        vertex_position - wp.vec3(thickness),
        vertex_position + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = triangle_indices[triangle, 0]
        index_1 = triangle_indices[triangle, 1]
        index_2 = triangle_indices[triangle, 2]
        if surface_ownership[vertex] == surface_ownership[index_0]:
            continue

        position_0 = positions[index_0]
        position_1 = positions[index_1]
        position_2 = positions[index_2]
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
            positions,
            triangle_indices,
            triangle_edge_indices,
            edge_triangle_indices,
            vertex_triangle_offsets,
            vertex_triangle_indices,
        )
        if wp.length(feature_normal) <= _MIN_CONTACT_DISTANCE:
            continue
        direction = separation / distance
        signed_distance = distance
        if wp.dot(separation, feature_normal) < 0.0:
            direction = -direction
            signed_distance = -distance

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        contact_ids[contact, 0] = vertex
        contact_ids[contact, 1] = index_0
        contact_ids[contact, 2] = index_1
        contact_ids[contact, 3] = index_2
        contact_weights[contact, 0] = 1.0
        contact_weights[contact, 1] = -barycentric[0]
        contact_weights[contact, 2] = -barycentric[1]
        contact_weights[contact, 3] = -barycentric[2]
        contact_directions[contact] = direction
        contact_depths[contact] = thickness - signed_distance


@wp.kernel
def _detect_affine_edge_edge_contacts(
    edge_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    triangle_indices: wp.array2d[int],
    edge_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    edge = wp.tid()
    index_0 = edge_indices[edge, 2]
    index_1 = edge_indices[edge, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    query = wp.bvh_query_aabb(
        edge_bvh_id,
        wp.min(position_0, position_1) - wp.vec3(thickness),
        wp.max(position_0, position_1) + wp.vec3(thickness),
    )
    other_edge = wp.int32(-1)
    while wp.bvh_query_next(query, other_edge):
        if other_edge <= edge:
            continue
        index_2 = edge_indices[other_edge, 2]
        index_3 = edge_indices[other_edge, 3]
        if surface_ownership[index_0] == surface_ownership[index_2]:
            continue

        position_2 = positions[index_2]
        position_3 = positions[index_3]
        parameters = wp.closest_point_edge_edge(position_0, position_1, position_2, position_3, 1.0e-5)
        parameter_0 = parameters[0]
        parameter_1 = parameters[1]
        if parameter_0 <= 0.0 or parameter_0 >= 1.0:
            continue
        if parameter_1 <= 0.0 or parameter_1 >= 1.0:
            continue

        closest_0 = wp.lerp(position_0, position_1, parameter_0)
        closest_1 = wp.lerp(position_2, position_3, parameter_1)
        separation = closest_0 - closest_1
        distance = wp.length(separation)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue

        pseudo_normal_0 = _edge_pseudo_normal(edge, positions, triangle_indices, edge_triangle_indices)
        pseudo_normal_1 = _edge_pseudo_normal(other_edge, positions, triangle_indices, edge_triangle_indices)
        direction_raw = pseudo_normal_1 - pseudo_normal_0
        direction_length = wp.length(direction_raw)
        if direction_length <= _MIN_CONTACT_DISTANCE:
            continue
        direction = direction_raw / direction_length
        signed_distance = wp.dot(separation, direction)

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        contact_ids[contact, 0] = index_0
        contact_ids[contact, 1] = index_1
        contact_ids[contact, 2] = index_2
        contact_ids[contact, 3] = index_3
        contact_weights[contact, 0] = 1.0 - parameter_0
        contact_weights[contact, 1] = parameter_0
        contact_weights[contact, 2] = -(1.0 - parameter_1)
        contact_weights[contact, 3] = -parameter_1
        contact_directions[contact] = direction
        contact_depths[contact] = thickness - signed_distance
        rest_edge_0 = rest_positions[index_1] - rest_positions[index_0]
        rest_edge_1 = rest_positions[index_3] - rest_positions[index_2]
        mollifier_thresholds[contact] = (
            _EE_MOLLIFIER_THRESHOLD_SCALE * wp.dot(rest_edge_0, rest_edge_0) * wp.dot(rest_edge_1, rest_edge_1)
        )
        mollifier_active[contact] = 0


@wp.kernel
def _prepare_affine_contact_response(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    count: wp.array[int],
    capacity: int,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    use_mollifier: int,
    velocities: wp.array[vec12],
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
    for local_index in range(4):
        particle = ids[contact, local_index]
        body = surface_ownership[particle]
        point_velocity = _affine_point(velocities[body], rest_positions[particle])
        relative_velocity += weights[contact, local_index] * point_velocity

    direction = directions[contact]
    depth = depths[contact]
    normal_outer = wp.outer(direction, direction)
    normal_scale = float(1.0)
    friction_load_scale = float(1.0)
    if use_mollifier != 0 and mollifier_active[contact] != 0:
        normal_scale = 0.0
        position_0 = positions[ids[contact, 0]]
        position_1 = positions[ids[contact, 1]]
        position_2 = positions[ids[contact, 2]]
        position_3 = positions[ids[contact, 3]]
        edge_0 = position_1 - position_0
        edge_1 = position_3 - position_2
        cross_product = wp.cross(edge_0, edge_1)
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
def _accumulate_mollified_affine_edge_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity) or mollifier_active[contact] == 0:
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    edge_0 = positions[index_1] - positions[index_0]
    edge_1 = positions[index_3] - positions[index_2]
    direction = directions[contact]
    depth = depths[contact]
    threshold = mollifier_thresholds[contact]
    contact_weights = wp.vec4(
        weights[contact, 0],
        weights[contact, 1],
        weights[contact, 2],
        weights[contact, 3],
    )
    cross_product, residual_scale, _scale_gradient = _edge_edge_mollified_residual_data(
        edge_0,
        edge_1,
        threshold,
    )
    edge_product_0, edge_product_1, depth_product = _edge_edge_mollified_residual_jacobian_transpose_multiply(
        edge_0,
        edge_1,
        depth,
        threshold,
        depth * residual_scale * cross_product,
    )
    gradient_0 = -edge_product_0 - contact_weights[0] * depth_product * direction
    gradient_1 = edge_product_0 - contact_weights[1] * depth_product * direction
    gradient_2 = -edge_product_1 - contact_weights[2] * depth_product * direction
    gradient_3 = edge_product_1 - contact_weights[3] * depth_product * direction
    gradients = gradient_0, gradient_1, gradient_2, gradient_3

    for local_index in range(4):
        particle = ids[contact, local_index]
        body = surface_ownership[particle]
        lifted_force = _lift_affine_vector(
            -stiffness * gradients[local_index],
            rest_positions[particle],
        )
        wp.atomic_add(output, body, lifted_force)


@wp.kernel
def _mollified_affine_edge_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    vector: wp.array[vec12],
    output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity) or mollifier_active[contact] == 0:
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    contact_weights = wp.vec4(
        weights[contact, 0],
        weights[contact, 1],
        weights[contact, 2],
        weights[contact, 3],
    )
    product_0, product_1, product_2, product_3 = _edge_edge_gauss_newton_multiply(
        positions[index_1] - positions[index_0],
        positions[index_3] - positions[index_2],
        contact_weights,
        directions[contact],
        depths[contact],
        mollifier_thresholds[contact],
        _affine_point(vector[surface_ownership[index_0]], rest_positions[index_0]),
        _affine_point(vector[surface_ownership[index_1]], rest_positions[index_1]),
        _affine_point(vector[surface_ownership[index_2]], rest_positions[index_2]),
        _affine_point(vector[surface_ownership[index_3]], rest_positions[index_3]),
    )
    products = product_0, product_1, product_2, product_3
    for local_index in range(4):
        particle = ids[contact, local_index]
        body = surface_ownership[particle]
        lifted_product = _lift_affine_vector(
            stiffness * products[local_index],
            rest_positions[particle],
        )
        wp.atomic_add(output, body, lifted_product)


@wp.kernel
def _accumulate_mollified_affine_edge_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    output: wp.array[mat1212],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity) or mollifier_active[contact] == 0:
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    body_0 = surface_ownership[index_0]
    body_1 = surface_ownership[index_2]
    contact_weights = wp.vec4(
        weights[contact, 0],
        weights[contact, 1],
        weights[contact, 2],
        weights[contact, 3],
    )

    for side in range(2):
        body = body_0
        if side == 1:
            body = body_1
        block = mat1212(0.0)
        for column in range(12):
            vector_0 = wp.vec3(0.0)
            vector_1 = wp.vec3(0.0)
            vector_2 = wp.vec3(0.0)
            vector_3 = wp.vec3(0.0)
            if surface_ownership[index_0] == body:
                vector_0 = _affine_point_basis(rest_positions[index_0], column)
            if surface_ownership[index_1] == body:
                vector_1 = _affine_point_basis(rest_positions[index_1], column)
            if surface_ownership[index_2] == body:
                vector_2 = _affine_point_basis(rest_positions[index_2], column)
            if surface_ownership[index_3] == body:
                vector_3 = _affine_point_basis(rest_positions[index_3], column)
            product_0, product_1, product_2, product_3 = _edge_edge_gauss_newton_multiply(
                positions[index_1] - positions[index_0],
                positions[index_3] - positions[index_2],
                contact_weights,
                directions[contact],
                depths[contact],
                mollifier_thresholds[contact],
                vector_0,
                vector_1,
                vector_2,
                vector_3,
            )
            generalized_product = vec12(0.0)
            if surface_ownership[index_0] == body:
                generalized_product += _lift_affine_vector(product_0, rest_positions[index_0])
            if surface_ownership[index_1] == body:
                generalized_product += _lift_affine_vector(product_1, rest_positions[index_1])
            if surface_ownership[index_2] == body:
                generalized_product += _lift_affine_vector(product_2, rest_positions[index_2])
            if surface_ownership[index_3] == body:
                generalized_product += _lift_affine_vector(product_3, rest_positions[index_3])
            for row in range(12):
                block[row, column] = stiffness * generalized_product[row]
        wp.atomic_add(output, body, block)


@wp.kernel
def _accumulate_affine_contact_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    forces: wp.array[wp.vec3],
    count: wp.array[int],
    capacity: int,
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    force = forces[contact]
    for local_index in range(4):
        particle = ids[contact, local_index]
        body = surface_ownership[particle]
        lifted_force = _lift_affine_vector(
            weights[contact, local_index] * force,
            rest_positions[particle],
        )
        wp.atomic_add(output, body, lifted_force)


@wp.kernel
def _affine_contact_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    hessians: wp.array[wp.mat33],
    count: wp.array[int],
    capacity: int,
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    vector: wp.array[vec12],
    output: wp.array[vec12],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    world_vector = wp.vec3(0.0)
    for local_index in range(4):
        particle = ids[contact, local_index]
        body = surface_ownership[particle]
        point_vector = _affine_point(vector[body], rest_positions[particle])
        world_vector += weights[contact, local_index] * point_vector
    world_product = hessians[contact] * world_vector

    for local_index in range(4):
        particle = ids[contact, local_index]
        body = surface_ownership[particle]
        lifted_product = _lift_affine_vector(
            weights[contact, local_index] * world_product,
            rest_positions[particle],
        )
        wp.atomic_add(output, body, lifted_product)


@wp.kernel
def _accumulate_affine_contact_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    hessians: wp.array[wp.mat33],
    count: wp.array[int],
    capacity: int,
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    output: wp.array[mat1212],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    body_0 = surface_ownership[ids[contact, 0]]
    body_1 = body_0
    for local_index in range(1, 4):
        candidate = surface_ownership[ids[contact, local_index]]
        if candidate != body_0:
            body_1 = candidate

    for side in range(2):
        body = body_0
        if side == 1:
            body = body_1
        translation_weight = float(0.0)
        rest_weight = wp.vec3(0.0)
        for local_index in range(4):
            particle = ids[contact, local_index]
            if surface_ownership[particle] == body:
                weight = weights[contact, local_index]
                translation_weight += weight
                rest_weight += weight * rest_positions[particle]
        block = _lift_weighted_affine_matrix(
            hessians[contact],
            translation_weight,
            rest_weight,
        )
        wp.atomic_add(output, body, block)


class _AffineContactBuffer:
    def __init__(self, capacity: int, device: wp.context.Device):
        self.capacity = capacity
        self.device = device
        self.ids = wp.empty((capacity, 4), dtype=wp.int32, device=device)
        self.weights = wp.empty((capacity, 4), dtype=wp.float32, device=device)
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


class _AffineEdgeEdgeContactBuffer(_AffineContactBuffer):
    pass


class ConstraintAffineBodyContact:
    """Detect frozen VF and strict-interior EE contact between affine bodies."""

    def __init__(
        self,
        body_model: AffineBodyModel,
        thickness: float,
        stiffness: float,
        normal_damping: float,
        friction: float,
        friction_epsilon: float,
        max_contacts: int = 262144,
    ):
        """Create affine-body surface contact.

        Args:
            body_model: Multi-body affine model supplying compact surface geometry.
            thickness: Two-surface contact activation distance [m].
            stiffness: Normal penalty stiffness per retained contact [N/m].
            normal_damping: Approaching normal damping coefficient [N·s/m].
            friction: Coulomb friction coefficient.
            friction_epsilon: Tangential displacement regularization [m].
            max_contacts: Maximum stored contacts for each contact type.
        """
        if not isinstance(body_model, AffineBodyModel):
            raise TypeError("body_model must be an AffineBodyModel")
        if body_model.body_count < 2:
            raise ValueError("body_model must contain at least two affine bodies")
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

        self.body_model = body_model
        self.body_count = body_model.body_count
        self.device = body_model.device
        self.thickness = float(thickness)
        self.stiffness = float(stiffness)
        self.normal_damping = float(normal_damping)
        self.friction = float(friction)
        self.friction_epsilon = float(friction_epsilon)
        self.max_contacts = int(max_contacts)

        triangles = np.asarray(body_model.surface_triangle_indices.numpy(), dtype=np.int32)
        ownership = np.asarray(body_model.surface_ownership.numpy(), dtype=np.int32)
        triangle_ownership = ownership[triangles[:, 0]]
        if not np.all(ownership[triangles] == triangle_ownership[:, None]):
            raise ValueError("Every surface triangle must belong to one affine body")
        adjacency = MeshAdjacency(triangles)
        edges = adjacency.edge_indices
        edge_ownership = ownership[edges[:, 2]]
        if not np.all(ownership[edges[:, 2:4]] == edge_ownership[:, None]):
            raise ValueError("Every surface edge must belong to one affine body")

        self.triangle_indices = wp.array(triangles, dtype=wp.int32, device=self.device)
        self.edge_indices = wp.array(edges, dtype=wp.int32, device=self.device)
        self.triangle_edge_indices = wp.array(adjacency.tri_edge_indices, dtype=wp.int32, device=self.device)
        self.edge_triangle_indices = wp.array(adjacency.edge_tri_indices, dtype=wp.int32, device=self.device)
        if np.any(adjacency.edge_tri_indices < 0):
            raise ValueError("Affine tetrahedral collision surfaces must be closed")
        vertex_triangles = [[] for _ in range(body_model.surface_vertex_count)]
        for triangle, indices in enumerate(triangles):
            for vertex in indices:
                vertex_triangles[int(vertex)].append(triangle)
        vertex_triangle_offsets = np.zeros(body_model.surface_vertex_count + 1, dtype=np.int32)
        vertex_triangle_offsets[1:] = np.cumsum(
            [len(incident) for incident in vertex_triangles],
            dtype=np.int32,
        )
        vertex_triangle_indices = np.fromiter(
            (triangle for incident in vertex_triangles for triangle in incident),
            dtype=np.int32,
            count=int(vertex_triangle_offsets[-1]),
        )
        self.vertex_triangle_offsets = wp.array(vertex_triangle_offsets, dtype=wp.int32, device=self.device)
        self.vertex_triangle_indices = wp.array(vertex_triangle_indices, dtype=wp.int32, device=self.device)
        self.triangle_count = len(triangles)
        self.edge_count = len(edges)
        self.surface_vertex_count = body_model.surface_vertex_count
        self.positions = wp.empty(self.surface_vertex_count, dtype=wp.vec3, device=self.device)
        self.triangle_lower_bounds = wp.empty(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_upper_bounds = wp.empty_like(self.triangle_lower_bounds)
        self.edge_lower_bounds = wp.empty(self.edge_count, dtype=wp.vec3, device=self.device)
        self.edge_upper_bounds = wp.empty_like(self.edge_lower_bounds)
        body_model.update_surface_positions(body_model.q, self.positions)
        self._update_bounds()
        self.triangle_bvh = wp.Bvh(self.triangle_lower_bounds, self.triangle_upper_bounds)
        self.edge_bvh = wp.Bvh(self.edge_lower_bounds, self.edge_upper_bounds)
        self.vertex_face_contacts = _AffineContactBuffer(self.max_contacts, self.device)
        self.edge_edge_contacts = _AffineEdgeEdgeContactBuffer(self.max_contacts, self.device)
        self._velocities: wp.array[vec12] | None = None
        self._dt = 0.0
        self._prepared = False

    def begin_step(self, q: wp.array[vec12], qd: wp.array[vec12], dt: float) -> None:
        """Cache step-start affine velocity for later contact response.

        Args:
            q: Step-start affine generalized states.
            qd: Step-start affine generalized velocities [m/s, 1/s].
            dt: Simulation time step [s].
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self._validate_affine_vectors((q, "q"), (qd, "qd"))
        self._velocities = qd
        self._dt = float(dt)
        self._prepared = False

    def prepare(self, q: wp.array[vec12]) -> None:
        """Reconstruct surfaces and freeze cross-body contacts.

        Args:
            q: Current affine generalized states.
        """
        if self._velocities is None:
            raise RuntimeError("begin_step() must be called before prepare()")
        self._validate_affine_vectors((q, "q"))
        self.body_model.update_surface_positions(q, self.positions)
        self._update_bounds()
        self.triangle_bvh.refit()
        self.edge_bvh.refit()
        self.vertex_face_contacts.clear()
        self.edge_edge_contacts.clear()
        wp.launch(
            _detect_affine_vertex_face_contacts,
            dim=self.surface_vertex_count,
            inputs=[
                self.triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                self.positions,
                self.body_model.surface_ownership,
                self.triangle_indices,
                self.triangle_edge_indices,
                self.edge_triangle_indices,
                self.vertex_triangle_offsets,
                self.vertex_triangle_indices,
            ],
            outputs=[
                self.vertex_face_contacts.ids,
                self.vertex_face_contacts.weights,
                self.vertex_face_contacts.directions,
                self.vertex_face_contacts.depths,
                self.vertex_face_contacts.count,
                self.vertex_face_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            _detect_affine_edge_edge_contacts,
            dim=self.edge_count,
            inputs=[
                self.edge_bvh.id,
                self.thickness,
                self.max_contacts,
                self.positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                self.triangle_indices,
                self.edge_indices,
                self.edge_triangle_indices,
            ],
            outputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
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
            _prepare_edge_edge_mollifier,
            dim=self.edge_edge_contacts.capacity,
            inputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.capacity,
                self.positions,
            ],
            outputs=[self.edge_edge_contacts.mollifier_active],
            device=self.device,
        )
        self._prepare_response(self.vertex_face_contacts, use_mollifier=False)
        self._prepare_response(self.edge_edge_contacts, use_mollifier=True)
        self._prepared = True

    def accumulate_force(self, q: wp.array[vec12], output: wp.array[vec12]) -> None:
        """Add frozen affine-body contact forces.

        Args:
            q: Current affine generalized states.
            output: Affine generalized force accumulation buffer [N, N·m].
        """
        self._require_prepared()
        self._validate_affine_vectors((q, "q"), (output, "output"))
        self._accumulate_buffer_force(self.vertex_face_contacts, output)
        self._accumulate_buffer_force(self.edge_edge_contacts, output)
        wp.launch(
            _accumulate_mollified_affine_edge_force,
            dim=self.edge_edge_contacts.capacity,
            inputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.mollifier_active,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.capacity,
                self.stiffness,
                self.positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
            ],
            outputs=[output],
            device=self.device,
        )

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Add complete matrix-free affine contact Hessian products."""
        self._require_prepared()
        self._validate_empty_particle_vector(particle_input, "particle_input")
        self._validate_empty_particle_vector(particle_output, "particle_output")
        self._validate_affine_vectors((affine_input, "affine_input"), (affine_output, "affine_output"))
        self._multiply_buffer(self.vertex_face_contacts, affine_input, affine_output)
        self._multiply_buffer(self.edge_edge_contacts, affine_input, affine_output)
        wp.launch(
            _mollified_affine_edge_hessian_multiply,
            dim=self.edge_edge_contacts.capacity,
            inputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.mollifier_active,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.capacity,
                self.stiffness,
                self.positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                affine_input,
            ],
            outputs=[affine_output],
            device=self.device,
        )

    def accumulate_diagonal(
        self,
        particle_diagonal: wp.array[wp.mat33],
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        """Add exact 12-by-12 contact blocks to the affine diagonal."""
        self._require_prepared()
        self._validate_empty_particle_diagonal(particle_diagonal)
        if len(affine_diagonal) != self.body_count:
            raise ValueError(f"affine_diagonal must contain {self.body_count} blocks")
        if affine_diagonal.device != self.device:
            raise ValueError(f"affine_diagonal must use device {self.device}")
        if affine_diagonal.dtype != mat1212:
            raise TypeError("affine_diagonal must have dtype mat1212")
        self._accumulate_buffer_diagonal(self.vertex_face_contacts, affine_diagonal)
        self._accumulate_buffer_diagonal(self.edge_edge_contacts, affine_diagonal)
        wp.launch(
            _accumulate_mollified_affine_edge_diagonal,
            dim=self.edge_edge_contacts.capacity,
            inputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.mollifier_active,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.capacity,
                self.stiffness,
                self.positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
            ],
            outputs=[affine_diagonal],
            device=self.device,
        )

    def _prepare_response(self, buffer: _AffineContactBuffer, use_mollifier: bool) -> None:
        if self._velocities is None:
            raise RuntimeError("begin_step() must be called before preparing contact response")
        wp.launch(
            _prepare_affine_contact_response,
            dim=buffer.capacity,
            inputs=[
                buffer.ids,
                buffer.weights,
                buffer.directions,
                buffer.depths,
                buffer.count,
                buffer.capacity,
                self.positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                buffer.mollifier_thresholds,
                buffer.mollifier_active,
                int(use_mollifier),
                self._velocities,
                self.stiffness,
                self.normal_damping,
                self.friction,
                self.friction_epsilon,
                self._dt,
            ],
            outputs=[buffer.forces, buffer.hessians],
            device=self.device,
        )

    def _accumulate_buffer_force(self, buffer: _AffineContactBuffer, output: wp.array[vec12]) -> None:
        wp.launch(
            _accumulate_affine_contact_force,
            dim=buffer.capacity,
            inputs=[
                buffer.ids,
                buffer.weights,
                buffer.forces,
                buffer.count,
                buffer.capacity,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
            ],
            outputs=[output],
            device=self.device,
        )

    def _multiply_buffer(
        self,
        buffer: _AffineContactBuffer,
        affine_input: wp.array[vec12],
        affine_output: wp.array[vec12],
    ) -> None:
        wp.launch(
            _affine_contact_hessian_multiply,
            dim=buffer.capacity,
            inputs=[
                buffer.ids,
                buffer.weights,
                buffer.hessians,
                buffer.count,
                buffer.capacity,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                affine_input,
            ],
            outputs=[affine_output],
            device=self.device,
        )

    def _accumulate_buffer_diagonal(
        self,
        buffer: _AffineContactBuffer,
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        wp.launch(
            _accumulate_affine_contact_diagonal,
            dim=buffer.capacity,
            inputs=[
                buffer.ids,
                buffer.weights,
                buffer.hessians,
                buffer.count,
                buffer.capacity,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
            ],
            outputs=[affine_diagonal],
            device=self.device,
        )

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before using affine contact contributions")

    def _update_bounds(self) -> None:
        wp.launch(
            _update_affine_triangle_bounds,
            dim=self.triangle_count,
            inputs=[self.positions, self.triangle_indices],
            outputs=[self.triangle_lower_bounds, self.triangle_upper_bounds],
            device=self.device,
        )
        wp.launch(
            _update_affine_edge_bounds,
            dim=self.edge_count,
            inputs=[self.positions, self.edge_indices],
            outputs=[self.edge_lower_bounds, self.edge_upper_bounds],
            device=self.device,
        )

    def _validate_affine_vectors(self, *arrays: tuple[wp.array[vec12], str]) -> None:
        for array, name in arrays:
            if len(array) != self.body_count:
                raise ValueError(f"{name} must contain {self.body_count} vectors")
            if array.device != self.device:
                raise ValueError(f"{name} must use device {self.device}")
            if array.dtype != vec12:
                raise TypeError(f"{name} must have dtype vec12")

    def _validate_empty_particle_vector(self, array: wp.array[wp.vec3], name: str) -> None:
        if len(array) != 0:
            raise ValueError(f"{name} must be empty for affine-only contact")
        if array.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")
        if array.dtype != wp.vec3:
            raise TypeError(f"{name} must have dtype wp.vec3")

    def _validate_empty_particle_diagonal(self, array: wp.array[wp.mat33]) -> None:
        if len(array) != 0:
            raise ValueError("particle_diagonal must be empty for affine-only contact")
        if array.device != self.device:
            raise ValueError(f"particle_diagonal must use device {self.device}")
        if array.dtype != wp.mat33:
            raise TypeError("particle_diagonal must have dtype wp.mat33")
