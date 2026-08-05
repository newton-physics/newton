# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free cloth self-collision for LIMX."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ....sim import Model
from ....utils.mesh import MeshAdjacency

_MIN_BARYCENTRIC_DENOMINATOR = 1.0e-12
_MIN_CONTACT_DISTANCE = 1.0e-7
_MIN_GEOMETRY_NORM = 1.0e-8
_MIN_STIFFNESS_DENOMINATOR = 1.0e-12


def _compute_geometry_aware_particle_radii(
    rest_positions: np.ndarray,
    triangle_indices: np.ndarray,
    nominal_radius: float,
    geometry_radius_scale: float,
) -> np.ndarray:
    if not np.isfinite(rest_positions).all():
        raise ValueError("geometry-aware collision requires finite rest positions")

    triangle_positions = rest_positions[triangle_indices]
    edge_01 = triangle_positions[:, 1] - triangle_positions[:, 0]
    edge_12 = triangle_positions[:, 2] - triangle_positions[:, 1]
    edge_20 = triangle_positions[:, 0] - triangle_positions[:, 2]
    maximum_edges = np.maximum.reduce(
        (
            np.linalg.norm(edge_01, axis=1),
            np.linalg.norm(edge_12, axis=1),
            np.linalg.norm(edge_20, axis=1),
        )
    )
    twice_areas = np.linalg.norm(np.cross(edge_01, -edge_20), axis=1)
    if np.any(maximum_edges <= _MIN_GEOMETRY_NORM):
        raise ValueError("geometry-aware collision requires non-degenerate rest triangles")
    triangle_scales = twice_areas / maximum_edges
    if np.any(triangle_scales <= _MIN_GEOMETRY_NORM) or not np.isfinite(triangle_scales).all():
        raise ValueError("geometry-aware collision requires non-degenerate rest triangles")

    local_scales = np.full(len(rest_positions), np.inf, dtype=np.float64)
    for corner in range(3):
        np.minimum.at(local_scales, triangle_indices[:, corner], triangle_scales)
    if not np.isfinite(local_scales).all():
        raise ValueError("geometry-aware collision requires every particle to be referenced by a triangle")

    return np.minimum(nominal_radius, geometry_radius_scale * local_scales).astype(np.float32)


@wp.func
def _triangle_barycentric(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    point: wp.vec3,
):
    edge_0 = position_0 - position_2
    edge_1 = position_1 - position_2
    relative = point - position_2
    dot_00 = wp.dot(edge_0, edge_0)
    dot_01 = wp.dot(edge_0, edge_1)
    dot_02 = wp.dot(edge_0, relative)
    dot_11 = wp.dot(edge_1, edge_1)
    dot_12 = wp.dot(edge_1, relative)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    if wp.abs(denominator) <= _MIN_BARYCENTRIC_DENOMINATOR:
        return wp.vec3(-1.0)
    inverse_denominator = 1.0 / denominator
    barycentric_0 = (dot_11 * dot_02 - dot_01 * dot_12) * inverse_denominator
    barycentric_1 = (dot_00 * dot_12 - dot_01 * dot_02) * inverse_denominator
    return wp.vec3(barycentric_0, barycentric_1, 1.0 - barycentric_0 - barycentric_1)


@wp.kernel
def _update_triangle_bounds(
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
def _update_edge_bounds(
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
def _detect_vertex_face_contacts(
    triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    positions: wp.array[wp.vec3],
    particle_world: wp.array[int],
    triangle_indices: wp.array2d[int],
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
        if vertex == index_0 or vertex == index_1 or vertex == index_2:
            continue
        if particle_world[vertex] != particle_world[index_0]:
            continue

        position_0 = positions[index_0]
        position_1 = positions[index_1]
        position_2 = positions[index_2]
        normal_raw = wp.cross(position_1 - position_0, position_2 - position_0)
        normal_length = wp.length(normal_raw)
        if normal_length <= _MIN_GEOMETRY_NORM:
            continue
        triangle_normal = normal_raw / normal_length
        signed_distance = wp.dot(vertex_position - position_0, triangle_normal)
        distance = wp.abs(signed_distance)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue

        projected = vertex_position - signed_distance * triangle_normal
        barycentric = _triangle_barycentric(position_0, position_1, position_2, projected)
        if barycentric[0] < 0.0 or barycentric[1] < 0.0 or barycentric[2] < 0.0:
            continue

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        direction = triangle_normal
        if signed_distance < 0.0:
            direction = -direction
        contact_ids[contact, 0] = vertex
        contact_ids[contact, 1] = index_0
        contact_ids[contact, 2] = index_1
        contact_ids[contact, 3] = index_2
        contact_weights[contact, 0] = 1.0
        contact_weights[contact, 1] = -barycentric[0]
        contact_weights[contact, 2] = -barycentric[1]
        contact_weights[contact, 3] = -barycentric[2]
        contact_directions[contact] = direction
        contact_depths[contact] = thickness - distance


@wp.kernel
def _detect_edge_edge_contacts(
    edge_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    positions: wp.array[wp.vec3],
    particle_world: wp.array[int],
    edge_indices: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
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
        if index_0 == index_2 or index_0 == index_3 or index_1 == index_2 or index_1 == index_3:
            continue
        if particle_world[index_0] != particle_world[index_2]:
            continue

        position_2 = positions[index_2]
        position_3 = positions[index_3]
        parameters = wp.closest_point_edge_edge(position_0, position_1, position_2, position_3, 1.0e-5)
        parameter_0 = parameters[0]
        parameter_1 = parameters[1]
        if (
            parameter_0 <= _MIN_CONTACT_DISTANCE
            or parameter_0 >= 1.0 - _MIN_CONTACT_DISTANCE
            or parameter_1 <= _MIN_CONTACT_DISTANCE
            or parameter_1 >= 1.0 - _MIN_CONTACT_DISTANCE
        ):
            continue

        closest_0 = wp.lerp(position_0, position_1, parameter_0)
        closest_1 = wp.lerp(position_2, position_3, parameter_1)
        separation = closest_0 - closest_1
        distance = wp.length(separation)
        limited_thickness = thickness
        if (
            edge_indices[edge, 0] == index_2
            or edge_indices[edge, 1] == index_2
            or edge_indices[edge, 0] == index_3
            or edge_indices[edge, 1] == index_3
            or edge_indices[other_edge, 0] == index_0
            or edge_indices[other_edge, 1] == index_0
            or edge_indices[other_edge, 0] == index_1
            or edge_indices[other_edge, 1] == index_1
        ):
            average_length = 0.5 * (wp.length(position_1 - position_0) + wp.length(position_3 - position_2))
            limited_thickness = wp.min(limited_thickness, 0.5 * average_length)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= limited_thickness:
            continue

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
        contact_directions[contact] = separation / distance
        contact_depths[contact] = limited_thickness - distance


@wp.func
def _intersection_gradient(intersection_direction: wp.vec3, edge_direction: wp.vec3, face_normal: wp.vec3):
    denominator = wp.dot(edge_direction, face_normal)
    if wp.abs(denominator) > 1.0e-6:
        return intersection_direction - (
            2.0 * face_normal * wp.dot(edge_direction, intersection_direction) / denominator
        )
    return intersection_direction


@wp.func
def _accumulate_adjacent_face_gradient(
    opposite: int,
    edge_position_0: wp.vec3,
    edge_position_1: wp.vec3,
    edge_direction: wp.vec3,
    hit_point: wp.vec3,
    face_normal: wp.vec3,
    positions: wp.array[wp.vec3],
):
    if opposite < 0:
        return wp.vec3(0.0)
    adjacent_normal_raw = wp.cross(edge_position_1 - edge_position_0, positions[opposite] - edge_position_0)
    adjacent_normal_length = wp.length(adjacent_normal_raw)
    if adjacent_normal_length <= _MIN_GEOMETRY_NORM:
        return wp.vec3(0.0)
    adjacent_normal = adjacent_normal_raw / adjacent_normal_length
    intersection_direction = wp.cross(face_normal, adjacent_normal)
    intersection_length = wp.length(intersection_direction)
    if intersection_length <= 1.0e-6:
        return wp.vec3(0.0)
    intersection_direction /= intersection_length
    if (
        wp.dot(
            wp.cross(edge_direction, intersection_direction),
            wp.cross(edge_direction, positions[opposite] - hit_point),
        )
        < 0.0
    ):
        intersection_direction = -intersection_direction
    return _intersection_gradient(intersection_direction, edge_direction, face_normal)


@wp.kernel
def _detect_edge_face_untangle_contacts(
    triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    positions: wp.array[wp.vec3],
    particle_world: wp.array[int],
    triangle_indices: wp.array2d[int],
    edge_indices: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    edge = wp.tid()
    edge_index_0 = edge_indices[edge, 2]
    edge_index_1 = edge_indices[edge, 3]
    edge_position_0 = positions[edge_index_0]
    edge_position_1 = positions[edge_index_1]
    edge_vector = edge_position_0 - edge_position_1
    edge_length = wp.length(edge_vector)
    if edge_length <= _MIN_GEOMETRY_NORM:
        return
    edge_direction = edge_vector / edge_length

    query = wp.bvh_query_aabb(
        triangle_bvh_id,
        wp.min(edge_position_0, edge_position_1),
        wp.max(edge_position_0, edge_position_1),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        face_index_0 = triangle_indices[triangle, 0]
        face_index_1 = triangle_indices[triangle, 1]
        face_index_2 = triangle_indices[triangle, 2]
        if (
            edge_index_0 == face_index_0
            or edge_index_0 == face_index_1
            or edge_index_0 == face_index_2
            or edge_index_1 == face_index_0
            or edge_index_1 == face_index_1
            or edge_index_1 == face_index_2
        ):
            continue
        if particle_world[edge_index_0] != particle_world[face_index_0]:
            continue

        face_position_0 = positions[face_index_0]
        face_position_1 = positions[face_index_1]
        face_position_2 = positions[face_index_2]
        face_normal_raw = wp.cross(face_position_1 - face_position_0, face_position_2 - face_position_0)
        face_normal_length = wp.length(face_normal_raw)
        if face_normal_length <= _MIN_GEOMETRY_NORM:
            continue
        face_normal = face_normal_raw / face_normal_length
        signed_distance_0 = wp.dot(face_normal, edge_position_0 - face_position_0)
        signed_distance_1 = wp.dot(face_normal, edge_position_1 - face_position_0)
        if signed_distance_0 * signed_distance_1 >= 0.0:
            continue

        distance_0 = wp.abs(signed_distance_0)
        distance_1 = wp.abs(signed_distance_1)
        distance_sum = distance_0 + distance_1
        if distance_sum <= _MIN_CONTACT_DISTANCE:
            continue
        edge_weight_0 = distance_1 / distance_sum
        edge_weight_1 = distance_0 / distance_sum
        hit_point = edge_weight_0 * edge_position_0 + edge_weight_1 * edge_position_1
        barycentric = _triangle_barycentric(face_position_0, face_position_1, face_position_2, hit_point)
        if barycentric[0] < 0.01 or barycentric[1] < 0.01 or barycentric[2] < 0.01:
            continue

        gradient = _accumulate_adjacent_face_gradient(
            edge_indices[edge, 0],
            edge_position_0,
            edge_position_1,
            edge_direction,
            hit_point,
            face_normal,
            positions,
        )
        gradient += _accumulate_adjacent_face_gradient(
            edge_indices[edge, 1],
            edge_position_0,
            edge_position_1,
            edge_direction,
            hit_point,
            face_normal,
            positions,
        )
        gradient_length = wp.length(gradient)
        if gradient_length <= _MIN_GEOMETRY_NORM:
            continue

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        contact_ids[contact, 0] = edge_index_0
        contact_ids[contact, 1] = edge_index_1
        contact_ids[contact, 2] = face_index_0
        contact_ids[contact, 3] = face_index_1
        contact_ids[contact, 4] = face_index_2
        contact_weights[contact, 0] = edge_weight_0
        contact_weights[contact, 1] = edge_weight_1
        contact_weights[contact, 2] = -barycentric[0]
        contact_weights[contact, 3] = -barycentric[1]
        contact_weights[contact, 4] = -barycentric[2]
        contact_directions[contact] = gradient / gradient_length
        contact_depths[contact] = 2.0 * thickness


@wp.kernel
def _accumulate_contact_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    scaled_direction = stiffness * depths[contact] * directions[contact]
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * scaled_direction)


@wp.kernel
def _contact_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    projected_sum = float(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        projected_sum += weights[contact, local_index] * wp.dot(direction, vector[particle])

    scaled_direction = stiffness * projected_sum * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * scaled_direction)


@wp.kernel
def _accumulate_contact_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    rank_one = stiffness * wp.outer(direction, direction)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * rank_one)


@wp.func
def _adaptive_contact_stiffness(
    ids: wp.array2d[int],
    directions: wp.array[wp.vec3],
    contact: int,
    arity: int,
    feature_split: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
):
    direction = directions[contact]
    scale_0 = float(0.0)
    scale_1 = float(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        directional_scale = wp.dot(direction, static_diagonal[particle] * direction)
        directional_scale += masses[particle] * inv_dt_squared
        if local_index < feature_split:
            scale_0 += directional_scale
        else:
            scale_1 += directional_scale

    scale_0 /= float(feature_split)
    scale_1 /= float(arity - feature_split)
    denominator = scale_0 + scale_1
    if denominator <= _MIN_STIFFNESS_DENOMINATOR:
        return float(0.0)
    return factor * scale_0 * scale_1 / denominator


@wp.kernel
def _accumulate_contact_force_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    count: wp.array[int],
    arity: int,
    feature_split: int,
    capacity: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    stiffness = _adaptive_contact_stiffness(
        ids,
        directions,
        contact,
        arity,
        feature_split,
        factor,
        static_diagonal,
        masses,
        inv_dt_squared,
    )
    scaled_direction = stiffness * depths[contact] * directions[contact]
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * scaled_direction)


@wp.kernel
def _contact_hessian_multiply_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    feature_split: int,
    capacity: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    projected_sum = float(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        projected_sum += weights[contact, local_index] * wp.dot(direction, vector[particle])

    stiffness = _adaptive_contact_stiffness(
        ids,
        directions,
        contact,
        arity,
        feature_split,
        factor,
        static_diagonal,
        masses,
        inv_dt_squared,
    )
    scaled_direction = stiffness * projected_sum * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * scaled_direction)


@wp.kernel
def _accumulate_contact_diagonal_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    feature_split: int,
    capacity: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    stiffness = _adaptive_contact_stiffness(
        ids,
        directions,
        contact,
        arity,
        feature_split,
        factor,
        static_diagonal,
        masses,
        inv_dt_squared,
    )
    rank_one = stiffness * wp.outer(direction, direction)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * rank_one)


class _ContactBuffer:
    """Fixed-capacity rank-one contact data and matrix-free operations."""

    def __init__(self, arity: int, capacity: int, device: Any, feature_split: int | None = None):
        if arity not in (4, 5):
            raise ValueError("contact arity must be four or five")
        if feature_split is not None and (feature_split <= 0 or feature_split >= arity):
            raise ValueError("feature_split must partition the contact particles")
        if capacity <= 0:
            raise ValueError("contact capacity must be positive")

        self.arity = arity
        self.feature_split = feature_split
        self.capacity = capacity
        self.device = wp.get_device(device)
        self.ids = wp.zeros((capacity, arity), dtype=wp.int32, device=self.device)
        self.weights = wp.zeros((capacity, arity), dtype=wp.float32, device=self.device)
        self.directions = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
        self.depths = wp.zeros(capacity, dtype=wp.float32, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow_count = wp.zeros(1, dtype=wp.int32, device=self.device)

    def clear(self) -> None:
        """Reset device-side contact and overflow counters."""
        self.count.zero_()
        self.overflow_count.zero_()

    def accumulate_force(self, stiffness: float, output: wp.array[wp.vec3]) -> None:
        """Add physical contact forces to ``output``."""
        self._validate_output(output, wp.vec3)
        wp.launch(
            _accumulate_contact_force,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
            ],
            outputs=[output],
            device=self.device,
        )

    def hessian_multiply(
        self,
        stiffness: float,
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add the full rank-one contact Hessian-vector products."""
        self._validate_output(vector, wp.vec3)
        self._validate_output(output, wp.vec3)
        if len(vector) != len(output):
            raise ValueError("vector and output must have the same length")
        wp.launch(
            _contact_hessian_multiply,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_diagonal(self, stiffness: float, output: wp.array[wp.mat33]) -> None:
        """Add exact diagonal blocks of the rank-one contact Hessians."""
        self._validate_output(output, wp.mat33)
        wp.launch(
            _accumulate_contact_diagonal,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_force_adaptive(
        self,
        factor: float,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
        inv_dt_squared: float,
        output: wp.array[wp.vec3],
    ) -> None:
        """Add contact forces using directional feature stiffness."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_output(output, wp.vec3)
        wp.launch(
            _accumulate_contact_force_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.count,
                self.arity,
                self.feature_split,
                self.capacity,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
            ],
            outputs=[output],
            device=self.device,
        )

    def hessian_multiply_adaptive(
        self,
        factor: float,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
        inv_dt_squared: float,
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add adaptive rank-one contact Hessian-vector products."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_output(vector, wp.vec3)
        self._validate_output(output, wp.vec3)
        if len(vector) != len(output):
            raise ValueError("vector and output must have the same length")
        wp.launch(
            _contact_hessian_multiply_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.count,
                self.arity,
                self.feature_split,
                self.capacity,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_diagonal_adaptive(
        self,
        factor: float,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
        inv_dt_squared: float,
        output: wp.array[wp.mat33],
    ) -> None:
        """Add exact diagonal blocks using adaptive contact stiffness."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_output(output, wp.mat33)
        wp.launch(
            _accumulate_contact_diagonal_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.count,
                self.arity,
                self.feature_split,
                self.capacity,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
            ],
            outputs=[output],
            device=self.device,
        )

    def _validate_adaptive_data(
        self,
        factor: float,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
        inv_dt_squared: float,
    ) -> None:
        if self.feature_split is None:
            raise RuntimeError("adaptive contact operations require feature_split")
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("adaptive contact factor must be finite and positive")
        if not np.isfinite(inv_dt_squared) or inv_dt_squared <= 0.0:
            raise ValueError("inv_dt_squared must be finite and positive")
        self._validate_output(static_diagonal, wp.mat33)
        self._validate_output(masses, wp.float32)
        if len(static_diagonal) != len(masses):
            raise ValueError("static_diagonal and masses must have the same length")

    def _validate_output(self, output: wp.array, dtype: Any) -> None:
        if output.device != self.device:
            raise ValueError(f"array must use device {self.device}")
        if output.dtype != dtype:
            raise TypeError(f"array must have dtype {dtype}")


class ConstraintSelfCollision:
    """Frictionless matrix-free cloth self-collision constraints.

    Attributes:
        particle_radii: One-sided collision radii [m], shape
            ``[particle_count]``.
    """

    particle_radii: wp.array[float]

    def __init__(
        self,
        model: Model,
        thickness: float,
        stiffness: float | None,
        untangle_stiffness: float | None = None,
        max_contacts: int = 32768,
        stiffness_factors: tuple[float, float, float] | None = None,
        geometry_radius_scale: float | None = None,
    ):
        """Create a fixed-capacity GPU cloth self-collision operator.

        Args:
            model: Particle triangle-mesh model whose topology remains fixed.
            thickness: Nominal two-surface collision activation distance [m].
            stiffness: Fixed vertex-face and edge-edge penalty stiffness [N/m].
                Set to ``None`` to use adaptive feature stiffness.
            untangle_stiffness: Edge-face recovery stiffness [N/m]. Defaults
                to three times fixed ``stiffness``. Must be ``None`` in
                adaptive mode.
            max_contacts: Maximum stored contacts for each contact type.
            stiffness_factors: Adaptive dimensionless ``(VF, EE, EF)``
                stiffness factors. Must be provided exactly when ``stiffness``
                is ``None``.
            geometry_radius_scale: Optional dimensionless rest-geometry radius
                scale. When set, each one-sided particle radius is capped by
                this value times its minimum incident triangle altitude. The
                initial recommended value is ``0.25``.
        """
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("thickness must be finite and positive")
        if geometry_radius_scale is not None:
            if not np.isfinite(geometry_radius_scale):
                raise ValueError("geometry_radius_scale must be finite")
            if geometry_radius_scale <= 0.0:
                raise ValueError("geometry_radius_scale must be positive")
            geometry_radius_scale = float(geometry_radius_scale)
        if stiffness_factors is None:
            if stiffness is None:
                raise ValueError("stiffness_factors must be provided when stiffness is None")
            if not np.isfinite(stiffness) or stiffness <= 0.0:
                raise ValueError("stiffness must be finite and positive")
            if untangle_stiffness is None:
                untangle_stiffness = 3.0 * stiffness
            if not np.isfinite(untangle_stiffness) or untangle_stiffness <= 0.0:
                raise ValueError("untangle_stiffness must be finite and positive")
            validated_stiffness = float(stiffness)
            validated_untangle_stiffness = float(untangle_stiffness)
            validated_stiffness_factors = None
        else:
            if stiffness is not None:
                raise ValueError("Specify either fixed stiffness or stiffness_factors, not both")
            if untangle_stiffness is not None:
                raise ValueError("untangle_stiffness must be None in adaptive mode")
            factors = np.asarray(stiffness_factors, dtype=np.float64)
            if factors.shape != (3,):
                raise ValueError("stiffness_factors must contain three values")
            if not np.isfinite(factors).all():
                raise ValueError("stiffness_factors must be finite")
            if np.any(factors <= 0.0):
                raise ValueError("stiffness_factors must be positive")
            validated_stiffness = None
            validated_untangle_stiffness = None
            validated_stiffness_factors = tuple(float(value) for value in factors)
        if max_contacts <= 0:
            raise ValueError("max_contacts must be positive")
        if model.particle_count <= 0 or model.tri_count <= 0 or model.tri_indices is None:
            raise ValueError("ConstraintSelfCollision requires a particle triangle mesh")

        self.device = wp.get_device(model.device)
        self.particle_count = model.particle_count
        self.thickness = float(thickness)
        self.stiffness = validated_stiffness
        self.untangle_stiffness = validated_untangle_stiffness
        self.stiffness_factors = validated_stiffness_factors
        self.max_contacts = int(max_contacts)
        self.particle_world = model.particle_world
        self._static_diagonal: wp.array[wp.mat33] | None = None
        self._masses: wp.array[float] | None = None
        self._inv_dt_squared = 0.0

        triangle_indices = np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape(-1, 3)
        if triangle_indices.shape != (model.tri_count, 3):
            raise ValueError("model triangle topology must have shape [triangle_count, 3]")
        if np.any(triangle_indices < 0) or np.any(triangle_indices >= model.particle_count):
            raise ValueError("model triangle topology contains an invalid particle index")
        if np.any(
            (triangle_indices[:, 0] == triangle_indices[:, 1])
            | (triangle_indices[:, 1] == triangle_indices[:, 2])
            | (triangle_indices[:, 2] == triangle_indices[:, 0])
        ):
            raise ValueError("model triangles must contain three distinct particle indices")

        nominal_radius = 0.5 * self.thickness
        if geometry_radius_scale is None:
            particle_radii = np.full(model.particle_count, nominal_radius, dtype=np.float32)
        else:
            rest_positions = np.asarray(model.particle_q.numpy(), dtype=np.float64)
            particle_radii = _compute_geometry_aware_particle_radii(
                rest_positions,
                triangle_indices,
                nominal_radius,
                geometry_radius_scale,
            )
        self.geometry_radius_scale = geometry_radius_scale
        self.particle_radii = wp.array(particle_radii, dtype=wp.float32, device=self.device)
        self._use_geometry_radii = int(geometry_radius_scale is not None)

        edge_indices = MeshAdjacency(triangle_indices).edge_indices
        if len(edge_indices) == 0:
            raise ValueError("ConstraintSelfCollision requires at least one mesh edge")
        self.triangle_indices = wp.array(triangle_indices, dtype=wp.int32, device=self.device)
        self.edge_indices = wp.array(edge_indices, dtype=wp.int32, device=self.device)
        self.triangle_count = len(triangle_indices)
        self.edge_count = len(edge_indices)

        self.triangle_lower_bounds = wp.empty(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_upper_bounds = wp.empty_like(self.triangle_lower_bounds)
        self.edge_lower_bounds = wp.empty(self.edge_count, dtype=wp.vec3, device=self.device)
        self.edge_upper_bounds = wp.empty_like(self.edge_lower_bounds)
        self._update_bounds(model.particle_q)
        self.triangle_bvh = wp.Bvh(self.triangle_lower_bounds, self.triangle_upper_bounds)
        self.edge_bvh = wp.Bvh(self.edge_lower_bounds, self.edge_upper_bounds)

        self.vertex_face_contacts = _ContactBuffer(4, max_contacts, self.device, feature_split=1)
        self.edge_edge_contacts = _ContactBuffer(4, max_contacts, self.device, feature_split=2)
        self.edge_face_contacts = _ContactBuffer(5, max_contacts, self.device, feature_split=2)

    def bind_static_system(
        self,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
    ) -> None:
        """Bind static diagonal blocks and masses used by adaptive contact.

        Args:
            static_diagonal: Current assembled elastic diagonal blocks [N/m],
                shape ``[particle_count, 3, 3]``.
            masses: Particle masses [kg], shape ``[particle_count]``.
        """
        if static_diagonal.device != self.device or masses.device != self.device:
            raise ValueError(f"static_diagonal and masses must use device {self.device}")
        if static_diagonal.dtype != wp.mat33 or len(static_diagonal) != self.particle_count:
            raise ValueError(f"static_diagonal must contain {self.particle_count} wp.mat33 values")
        if masses.dtype != wp.float32 or len(masses) != self.particle_count:
            raise ValueError(f"masses must contain {self.particle_count} float values")
        self._static_diagonal = static_diagonal
        self._masses = masses

    def begin_step(
        self,
        positions: wp.array[wp.vec3],
        velocities: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Cache the inverse squared time step for adaptive contact."""
        if self.stiffness_factors is None:
            return
        self._validate_positions(positions)
        if velocities.device != self.device:
            raise ValueError(f"velocities must use device {self.device}")
        if velocities.dtype != wp.vec3 or len(velocities) != self.particle_count:
            raise ValueError(f"velocities must contain {self.particle_count} wp.vec3 values")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self._static_diagonal is None or self._masses is None:
            raise RuntimeError("bind_static_system() must be called before begin_step() in adaptive mode")
        self._inv_dt_squared = 1.0 / (dt * dt)

    def prepare(self, positions: wp.array[wp.vec3]) -> None:
        """Detect and freeze contacts at the current Newton iterate."""
        self._validate_positions(positions)
        self._update_bounds(positions)
        self.triangle_bvh.refit()
        self.edge_bvh.refit()
        self.vertex_face_contacts.clear()
        self.edge_edge_contacts.clear()
        self.edge_face_contacts.clear()
        wp.launch(
            _detect_vertex_face_contacts,
            dim=self.particle_count,
            inputs=[
                self.triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                positions,
                self.particle_world,
                self.triangle_indices,
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
            _detect_edge_edge_contacts,
            dim=self.edge_count,
            inputs=[
                self.edge_bvh.id,
                self.thickness,
                self.max_contacts,
                positions,
                self.particle_world,
                self.edge_indices,
            ],
            outputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            _detect_edge_face_untangle_contacts,
            dim=self.edge_count,
            inputs=[
                self.triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                positions,
                self.particle_world,
                self.triangle_indices,
                self.edge_indices,
            ],
            outputs=[
                self.edge_face_contacts.ids,
                self.edge_face_contacts.weights,
                self.edge_face_contacts.directions,
                self.edge_face_contacts.depths,
                self.edge_face_contacts.count,
                self.edge_face_contacts.overflow_count,
            ],
            device=self.device,
        )

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add frozen-contact physical forces to ``output``."""
        self._validate_positions(positions)
        if self.stiffness_factors is None:
            self.vertex_face_contacts.accumulate_force(self.stiffness, output)
            self.edge_edge_contacts.accumulate_force(self.stiffness, output)
            self.edge_face_contacts.accumulate_force(self.untangle_stiffness, output)
            return

        static_diagonal, masses, inv_dt_squared = self._adaptive_system()
        self.vertex_face_contacts.accumulate_force_adaptive(
            self.stiffness_factors[0], static_diagonal, masses, inv_dt_squared, output
        )
        self.edge_edge_contacts.accumulate_force_adaptive(
            self.stiffness_factors[1], static_diagonal, masses, inv_dt_squared, output
        )
        self.edge_face_contacts.accumulate_force_adaptive(
            self.stiffness_factors[2], static_diagonal, masses, inv_dt_squared, output
        )

    def hessian_multiply(
        self,
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add full frozen-contact Hessian-vector products to ``output``."""
        self._validate_positions(positions)
        if self.stiffness_factors is None:
            self.vertex_face_contacts.hessian_multiply(self.stiffness, vector, output)
            self.edge_edge_contacts.hessian_multiply(self.stiffness, vector, output)
            self.edge_face_contacts.hessian_multiply(self.untangle_stiffness, vector, output)
            return

        static_diagonal, masses, inv_dt_squared = self._adaptive_system()
        self.vertex_face_contacts.hessian_multiply_adaptive(
            self.stiffness_factors[0], static_diagonal, masses, inv_dt_squared, vector, output
        )
        self.edge_edge_contacts.hessian_multiply_adaptive(
            self.stiffness_factors[1], static_diagonal, masses, inv_dt_squared, vector, output
        )
        self.edge_face_contacts.hessian_multiply_adaptive(
            self.stiffness_factors[2], static_diagonal, masses, inv_dt_squared, vector, output
        )

    def accumulate_diagonal(self, positions: wp.array[wp.vec3], output: wp.array[wp.mat33]) -> None:
        """Add frozen-contact diagonal Hessian blocks to ``output``."""
        self._validate_positions(positions)
        if self.stiffness_factors is None:
            self.vertex_face_contacts.accumulate_diagonal(self.stiffness, output)
            self.edge_edge_contacts.accumulate_diagonal(self.stiffness, output)
            self.edge_face_contacts.accumulate_diagonal(self.untangle_stiffness, output)
            return

        static_diagonal, masses, inv_dt_squared = self._adaptive_system()
        self.vertex_face_contacts.accumulate_diagonal_adaptive(
            self.stiffness_factors[0], static_diagonal, masses, inv_dt_squared, output
        )
        self.edge_edge_contacts.accumulate_diagonal_adaptive(
            self.stiffness_factors[1], static_diagonal, masses, inv_dt_squared, output
        )
        self.edge_face_contacts.accumulate_diagonal_adaptive(
            self.stiffness_factors[2], static_diagonal, masses, inv_dt_squared, output
        )

    def _adaptive_system(self) -> tuple[wp.array[wp.mat33], wp.array[float], float]:
        if self._static_diagonal is None or self._masses is None or self._inv_dt_squared <= 0.0:
            raise RuntimeError("bind_static_system() and begin_step() are required before adaptive contact evaluation")
        return self._static_diagonal, self._masses, self._inv_dt_squared

    def _update_bounds(self, positions: wp.array[wp.vec3]) -> None:
        wp.launch(
            _update_triangle_bounds,
            dim=self.triangle_count,
            inputs=[positions, self.triangle_indices],
            outputs=[self.triangle_lower_bounds, self.triangle_upper_bounds],
            device=self.device,
        )
        wp.launch(
            _update_edge_bounds,
            dim=self.edge_count,
            inputs=[positions, self.edge_indices],
            outputs=[self.edge_lower_bounds, self.edge_upper_bounds],
            device=self.device,
        )

    def _validate_positions(self, positions: wp.array[wp.vec3]) -> None:
        if positions.device != self.device:
            raise ValueError(f"positions must use device {self.device}")
        if positions.dtype != wp.vec3 or len(positions) != self.particle_count:
            raise ValueError(f"positions must contain {self.particle_count} wp.vec3 values")
