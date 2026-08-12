# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free cloth self-collision for LIMX."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ....geometry.kernels import triangle_closest_point_barycentric
from ....sim import Model
from ....utils.mesh import MeshAdjacency

_MIN_BARYCENTRIC_DENOMINATOR = 1.0e-12
_MIN_CONTACT_DISTANCE = 1.0e-7
_MIN_GEOMETRY_NORM = 1.0e-8
_MIN_STIFFNESS_DENOMINATOR = 1.0e-12
_FEATURE_WEIGHT_EPSILON = 1.0e-6
_EE_MOLLIFIER_THRESHOLD_SCALE = 1.0e-3
_AUTOMATIC_THICKNESS_ETA = 0.8
_AUTOMATIC_THICKNESS_MAX = 5.0e-3
_AUTOMATIC_THICKNESS_NO_BOUND = 1.0e30


def _pack_index_sets(index_sets: list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(index_sets) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(indices) for indices in index_sets], dtype=np.int32)
    values = np.fromiter(
        (index for indices in index_sets for index in sorted(indices)),
        dtype=np.int32,
        count=int(offsets[-1]),
    )
    return offsets, values


def _tetrahedral_triangle_orientation_signs(
    rest_positions: np.ndarray,
    tetrahedron_indices: np.ndarray,
    triangle_indices: np.ndarray,
) -> np.ndarray:
    """Mark tetrahedral boundary triangles with their outward winding sign."""
    signs = np.zeros(len(triangle_indices), dtype=np.int32)
    if len(tetrahedron_indices) == 0:
        return signs

    face_opposites: dict[tuple[int, int, int], list[int]] = {}
    for tetrahedron in tetrahedron_indices:
        for opposite_local in range(4):
            face = tuple(sorted(int(index) for index in np.delete(tetrahedron, opposite_local)))
            face_opposites.setdefault(face, []).append(int(tetrahedron[opposite_local]))
    boundary_opposites = {face: opposites[0] for face, opposites in face_opposites.items() if len(opposites) == 1}

    for triangle_index, triangle in enumerate(triangle_indices):
        face = tuple(sorted(int(index) for index in triangle))
        opposite_index = boundary_opposites.get(face)
        if opposite_index is None:
            continue
        position_0, position_1, position_2 = rest_positions[triangle]
        normal = np.cross(position_1 - position_0, position_2 - position_0)
        opposite = rest_positions[opposite_index]
        orientation = float(np.dot(normal, opposite - position_0))
        scale = float(np.linalg.norm(normal) * np.linalg.norm(opposite - position_0))
        if not np.isfinite(orientation) or scale <= 0.0 or abs(orientation) <= 64.0 * np.finfo(float).eps * scale:
            raise ValueError(f"tetrahedral boundary triangle {triangle_index} has degenerate orientation")
        signs[triangle_index] = 1 if orientation < 0.0 else -1
    return signs


def _compute_two_ring_collision_upper_bound(
    rest_positions: np.ndarray,
    triangle_indices: np.ndarray,
    edge_indices: np.ndarray,
    device: Any,
) -> float:
    """Return the smallest exact-two-ring interior VF/EE rest distance on the device."""
    if rest_positions.ndim != 2 or rest_positions.shape[1] != 3:
        raise ValueError("rest positions must have shape [particle_count, 3]")
    if not np.isfinite(rest_positions).all():
        raise ValueError("automatic collision thickness requires finite rest positions")

    particle_count = len(rest_positions)
    one_ring_neighbors = [set() for _ in range(particle_count)]
    vertex_triangles = [set() for _ in range(particle_count)]
    vertex_edges = [set() for _ in range(particle_count)]
    edge_vertices = edge_indices[:, 2:4]
    for edge, indices in enumerate(edge_vertices):
        index_0, index_1 = (int(index) for index in indices)
        one_ring_neighbors[index_0].add(index_1)
        one_ring_neighbors[index_1].add(index_0)
        vertex_edges[index_0].add(edge)
        vertex_edges[index_1].add(edge)
    for triangle, indices in enumerate(triangle_indices):
        for vertex in indices:
            vertex_triangles[int(vertex)].add(triangle)

    neighbor_offsets, neighbor_indices = _pack_index_sets(one_ring_neighbors)
    vertex_triangle_offsets, vertex_triangle_indices = _pack_index_sets(vertex_triangles)
    vertex_edge_offsets, vertex_edge_indices = _pack_index_sets(vertex_edges)
    device = wp.get_device(device)
    positions_device = wp.array(rest_positions, dtype=wp.vec3, device=device)
    triangles_device = wp.array(triangle_indices, dtype=wp.int32, device=device)
    edges_device = wp.array(edge_indices, dtype=wp.int32, device=device)
    neighbor_offsets_device = wp.array(neighbor_offsets, dtype=wp.int32, device=device)
    neighbor_indices_device = wp.array(neighbor_indices, dtype=wp.int32, device=device)
    vertex_triangle_offsets_device = wp.array(vertex_triangle_offsets, dtype=wp.int32, device=device)
    vertex_triangle_indices_device = wp.array(vertex_triangle_indices, dtype=wp.int32, device=device)
    vertex_edge_offsets_device = wp.array(vertex_edge_offsets, dtype=wp.int32, device=device)
    vertex_edge_indices_device = wp.array(vertex_edge_indices, dtype=wp.int32, device=device)
    upper_bound = wp.array([np.inf], dtype=wp.float32, device=device)
    wp.launch(
        _reduce_two_ring_vertex_face_upper_bound,
        dim=particle_count,
        inputs=[
            positions_device,
            triangles_device,
            neighbor_offsets_device,
            neighbor_indices_device,
            vertex_triangle_offsets_device,
            vertex_triangle_indices_device,
        ],
        outputs=[upper_bound],
        device=device,
    )
    wp.launch(
        _reduce_two_ring_edge_edge_upper_bound,
        dim=len(edge_indices),
        inputs=[
            positions_device,
            edges_device,
            neighbor_offsets_device,
            neighbor_indices_device,
            vertex_edge_offsets_device,
            vertex_edge_indices_device,
        ],
        outputs=[upper_bound],
        device=device,
    )
    result = float(upper_bound.numpy()[0])
    if result >= _AUTOMATIC_THICKNESS_NO_BOUND:
        return np.inf
    return result


def _compute_geometry_aware_particle_radii(
    rest_positions: np.ndarray,
    triangle_indices: np.ndarray,
    nominal_radius: float,
    geometry_radius_scale: float,
) -> np.ndarray:
    triangle_positions = rest_positions[triangle_indices]
    if not np.isfinite(triangle_positions).all():
        raise ValueError("geometry-aware collision requires finite surface rest positions")
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
    surface_mask = np.isfinite(local_scales)
    particle_radii = np.zeros(len(rest_positions), dtype=np.float32)
    particle_radii[surface_mask] = np.minimum(
        nominal_radius,
        geometry_radius_scale * local_scales[surface_mask],
    )
    return particle_radii


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


@wp.func
def _is_topology_local_edge_pair(
    edge: int,
    other_edge: int,
    index_0: int,
    index_1: int,
    index_2: int,
    index_3: int,
    edge_indices: wp.array2d[int],
):
    return (
        edge_indices[edge, 0] == index_2
        or edge_indices[edge, 1] == index_2
        or edge_indices[edge, 0] == index_3
        or edge_indices[edge, 1] == index_3
        or edge_indices[other_edge, 0] == index_0
        or edge_indices[other_edge, 1] == index_0
        or edge_indices[other_edge, 0] == index_1
        or edge_indices[other_edge, 1] == index_1
    )


@wp.func
def _is_vertex_topology_neighbor(
    vertex: int,
    candidate: int,
    vertex_neighbor_offsets: wp.array[int],
    vertex_neighbors: wp.array[int],
):
    for entry in range(vertex_neighbor_offsets[vertex], vertex_neighbor_offsets[vertex + 1]):
        if vertex_neighbors[entry] == candidate:
            return True
    return False


@wp.func
def _point_triangle_interior_distance_device(
    point: wp.vec3,
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
):
    normal = wp.cross(position_1 - position_0, position_2 - position_0)
    normal_squared = wp.dot(normal, normal)
    if normal_squared <= _MIN_BARYCENTRIC_DENOMINATOR:
        return float(_AUTOMATIC_THICKNESS_NO_BOUND)
    signed_scale = wp.dot(point - position_0, normal) / normal_squared
    projected = point - signed_scale * normal
    barycentric = _triangle_barycentric(position_0, position_1, position_2, projected)
    if barycentric[0] < 0.0 or barycentric[1] < 0.0 or barycentric[2] < 0.0:
        return float(_AUTOMATIC_THICKNESS_NO_BOUND)
    return wp.abs(signed_scale) * wp.sqrt(normal_squared)


@wp.func
def _edge_edge_interior_distance_device(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    position_3: wp.vec3,
):
    edge_0 = position_1 - position_0
    edge_1 = position_3 - position_2
    relative = position_0 - position_2
    dot_00 = wp.dot(edge_0, edge_0)
    dot_01 = wp.dot(edge_0, edge_1)
    dot_11 = wp.dot(edge_1, edge_1)
    dot_0r = wp.dot(edge_0, relative)
    dot_1r = wp.dot(edge_1, relative)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    if denominator <= _MIN_BARYCENTRIC_DENOMINATOR:
        return float(_AUTOMATIC_THICKNESS_NO_BOUND)
    parameter_0 = (dot_01 * dot_1r - dot_11 * dot_0r) / denominator
    parameter_1 = (dot_00 * dot_1r - dot_01 * dot_0r) / denominator
    if (
        parameter_0 <= _MIN_CONTACT_DISTANCE
        or parameter_0 >= 1.0 - _MIN_CONTACT_DISTANCE
        or parameter_1 <= _MIN_CONTACT_DISTANCE
        or parameter_1 >= 1.0 - _MIN_CONTACT_DISTANCE
    ):
        return float(_AUTOMATIC_THICKNESS_NO_BOUND)
    closest_0 = position_0 + parameter_0 * edge_0
    closest_1 = position_2 + parameter_1 * edge_1
    return wp.length(closest_0 - closest_1)


@wp.kernel
def _reduce_two_ring_vertex_face_upper_bound(
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    neighbor_offsets: wp.array[int],
    neighbor_indices: wp.array[int],
    vertex_triangle_offsets: wp.array[int],
    vertex_triangle_indices: wp.array[int],
    upper_bound: wp.array[float],
):
    vertex = wp.tid()
    for neighbor_entry in range(neighbor_offsets[vertex], neighbor_offsets[vertex + 1]):
        neighbor = neighbor_indices[neighbor_entry]
        for two_ring_entry in range(neighbor_offsets[neighbor], neighbor_offsets[neighbor + 1]):
            two_ring_vertex = neighbor_indices[two_ring_entry]
            if two_ring_vertex == vertex or _is_vertex_topology_neighbor(
                vertex,
                two_ring_vertex,
                neighbor_offsets,
                neighbor_indices,
            ):
                continue
            for triangle_entry in range(
                vertex_triangle_offsets[two_ring_vertex],
                vertex_triangle_offsets[two_ring_vertex + 1],
            ):
                triangle = vertex_triangle_indices[triangle_entry]
                index_0 = triangle_indices[triangle, 0]
                index_1 = triangle_indices[triangle, 1]
                index_2 = triangle_indices[triangle, 2]
                if vertex == index_0 or vertex == index_1 or vertex == index_2:
                    continue
                if (
                    _is_vertex_topology_neighbor(vertex, index_0, neighbor_offsets, neighbor_indices)
                    or _is_vertex_topology_neighbor(vertex, index_1, neighbor_offsets, neighbor_indices)
                    or _is_vertex_topology_neighbor(vertex, index_2, neighbor_offsets, neighbor_indices)
                ):
                    continue
                distance = _point_triangle_interior_distance_device(
                    positions[vertex],
                    positions[index_0],
                    positions[index_1],
                    positions[index_2],
                )
                if distance < upper_bound[0]:
                    wp.atomic_min(upper_bound, 0, distance)


@wp.kernel
def _reduce_two_ring_edge_edge_upper_bound(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array2d[int],
    neighbor_offsets: wp.array[int],
    neighbor_indices: wp.array[int],
    vertex_edge_offsets: wp.array[int],
    vertex_edge_indices: wp.array[int],
    upper_bound: wp.array[float],
):
    edge = wp.tid()
    index_0 = edge_indices[edge, 2]
    index_1 = edge_indices[edge, 3]
    for local_endpoint in range(2):
        endpoint = index_0
        if local_endpoint == 1:
            endpoint = index_1
        for neighbor_entry in range(neighbor_offsets[endpoint], neighbor_offsets[endpoint + 1]):
            neighbor = neighbor_indices[neighbor_entry]
            for two_ring_entry in range(neighbor_offsets[neighbor], neighbor_offsets[neighbor + 1]):
                two_ring_vertex = neighbor_indices[two_ring_entry]
                if two_ring_vertex == endpoint or _is_vertex_topology_neighbor(
                    endpoint,
                    two_ring_vertex,
                    neighbor_offsets,
                    neighbor_indices,
                ):
                    continue
                for other_edge_entry in range(
                    vertex_edge_offsets[two_ring_vertex],
                    vertex_edge_offsets[two_ring_vertex + 1],
                ):
                    other_edge = vertex_edge_indices[other_edge_entry]
                    if other_edge <= edge:
                        continue
                    index_2 = edge_indices[other_edge, 2]
                    index_3 = edge_indices[other_edge, 3]
                    if index_2 == index_0 or index_2 == index_1 or index_3 == index_0 or index_3 == index_1:
                        continue
                    if (
                        _is_vertex_topology_neighbor(index_0, index_2, neighbor_offsets, neighbor_indices)
                        or _is_vertex_topology_neighbor(index_0, index_3, neighbor_offsets, neighbor_indices)
                        or _is_vertex_topology_neighbor(index_1, index_2, neighbor_offsets, neighbor_indices)
                        or _is_vertex_topology_neighbor(index_1, index_3, neighbor_offsets, neighbor_indices)
                    ):
                        continue
                    distance = _edge_edge_interior_distance_device(
                        positions[index_0],
                        positions[index_1],
                        positions[index_2],
                        positions[index_3],
                    )
                    if distance < upper_bound[0]:
                        wp.atomic_min(upper_bound, 0, distance)


@wp.func
def _triangle_unit_normal(
    triangle: int,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
):
    if triangle < 0:
        return wp.vec3(0.0)
    position_0 = positions[triangle_indices[triangle, 0]]
    position_1 = positions[triangle_indices[triangle, 1]]
    position_2 = positions[triangle_indices[triangle, 2]]
    normal = wp.cross(position_1 - position_0, position_2 - position_0)
    length = wp.length(normal)
    if length <= _MIN_GEOMETRY_NORM:
        return wp.vec3(0.0)
    return normal / length


@wp.func
def _edge_pseudo_normal(
    edge: int,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
):
    normal = _triangle_unit_normal(edge_triangle_indices[edge, 0], positions, triangle_indices)
    normal += _triangle_unit_normal(edge_triangle_indices[edge, 1], positions, triangle_indices)
    length = wp.length(normal)
    if length <= _MIN_GEOMETRY_NORM:
        return wp.vec3(0.0)
    return normal / length


@wp.func
def _oriented_triangle_unit_normal(
    triangle: int,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    triangle_orientation_signs: wp.array[int],
) -> wp.vec3:
    if triangle < 0 or triangle_orientation_signs[triangle] == 0:
        return wp.vec3(0.0)
    return float(triangle_orientation_signs[triangle]) * _triangle_unit_normal(
        triangle,
        positions,
        triangle_indices,
    )


@wp.func
def _oriented_edge_pseudo_normal(
    edge: int,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
    triangle_orientation_signs: wp.array[int],
) -> wp.vec3:
    normal = _oriented_triangle_unit_normal(
        edge_triangle_indices[edge, 0],
        positions,
        triangle_indices,
        triangle_orientation_signs,
    )
    normal += _oriented_triangle_unit_normal(
        edge_triangle_indices[edge, 1],
        positions,
        triangle_indices,
        triangle_orientation_signs,
    )
    length = wp.length(normal)
    if length <= _MIN_GEOMETRY_NORM:
        return wp.vec3(0.0)
    return normal / length


@wp.func
def _oriented_vertex_pseudo_normal(
    vertex: int,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    triangle_orientation_signs: wp.array[int],
    vertex_triangle_offsets: wp.array[int],
    vertex_triangle_indices: wp.array[int],
) -> wp.vec3:
    normal = wp.vec3(0.0)
    for cursor in range(vertex_triangle_offsets[vertex], vertex_triangle_offsets[vertex + 1]):
        normal += _oriented_triangle_unit_normal(
            vertex_triangle_indices[cursor],
            positions,
            triangle_indices,
            triangle_orientation_signs,
        )
    length = wp.length(normal)
    if length <= _MIN_GEOMETRY_NORM:
        return wp.vec3(0.0)
    return normal / length


@wp.func
def _oriented_closest_feature_normal(
    triangle: int,
    barycentric: wp.vec3,
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    triangle_edge_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
    triangle_orientation_signs: wp.array[int],
    vertex_triangle_offsets: wp.array[int],
    vertex_triangle_indices: wp.array[int],
) -> wp.vec3:
    zero_0 = barycentric[0] <= _FEATURE_WEIGHT_EPSILON
    zero_1 = barycentric[1] <= _FEATURE_WEIGHT_EPSILON
    zero_2 = barycentric[2] <= _FEATURE_WEIGHT_EPSILON
    zero_count = int(zero_0) + int(zero_1) + int(zero_2)
    if zero_count == 0:
        return _oriented_triangle_unit_normal(
            triangle,
            positions,
            triangle_indices,
            triangle_orientation_signs,
        )
    if zero_count == 1:
        local_edge = int(1)
        if zero_1:
            local_edge = 2
        elif zero_2:
            local_edge = 0
        return _oriented_edge_pseudo_normal(
            triangle_edge_indices[triangle, local_edge],
            positions,
            triangle_indices,
            edge_triangle_indices,
            triangle_orientation_signs,
        )

    local_vertex = int(0)
    if barycentric[1] > barycentric[local_vertex]:
        local_vertex = 1
    if barycentric[2] > barycentric[local_vertex]:
        local_vertex = 2
    return _oriented_vertex_pseudo_normal(
        triangle_indices[triangle, local_vertex],
        positions,
        triangle_indices,
        triangle_orientation_signs,
        vertex_triangle_offsets,
        vertex_triangle_indices,
    )


@wp.func
def _edge_edge_mollified_residual_data(
    edge_0: wp.vec3,
    edge_1: wp.vec3,
    threshold: float,
):
    cross_product = wp.cross(edge_0, edge_1)
    root = wp.sqrt(wp.max(2.0 * threshold - wp.dot(cross_product, cross_product), threshold))
    scale = root / threshold
    scale_gradient = -0.5 / (threshold * root)
    return cross_product, scale, scale_gradient


@wp.func
def _edge_edge_mollified_residual_jacobian_multiply(
    edge_0: wp.vec3,
    edge_1: wp.vec3,
    depth: float,
    threshold: float,
    edge_delta_0: wp.vec3,
    edge_delta_1: wp.vec3,
    depth_delta: float,
):
    cross_product, scale, scale_gradient = _edge_edge_mollified_residual_data(edge_0, edge_1, threshold)
    cross_delta = wp.cross(edge_delta_0, edge_1) + wp.cross(edge_0, edge_delta_1)
    cross_squared_delta = 2.0 * wp.dot(cross_product, cross_delta)
    return (
        depth * scale * cross_delta
        + (depth * scale_gradient * cross_squared_delta + depth_delta * scale) * cross_product
    )


@wp.func
def _edge_edge_mollified_residual_jacobian_transpose_multiply(
    edge_0: wp.vec3,
    edge_1: wp.vec3,
    depth: float,
    threshold: float,
    residual_vector: wp.vec3,
):
    cross_product, scale, scale_gradient = _edge_edge_mollified_residual_data(edge_0, edge_1, threshold)
    cross_projection = wp.dot(cross_product, residual_vector)
    cross_squared_product = depth * scale_gradient * cross_projection
    edge_product_0 = depth * scale * wp.cross(edge_1, residual_vector)
    edge_product_0 += cross_squared_product * 2.0 * wp.cross(edge_1, cross_product)
    edge_product_1 = depth * scale * wp.cross(residual_vector, edge_0)
    edge_product_1 += cross_squared_product * 2.0 * wp.cross(cross_product, edge_0)
    depth_product = scale * cross_projection
    return edge_product_0, edge_product_1, depth_product


@wp.kernel
def _detect_vertex_face_contacts(
    triangle_bvh_id: wp.uint64,
    thickness: float,
    particle_radii: wp.array[float],
    use_geometry_radii: int,
    geometry_radius_topology_local_only: int,
    capacity: int,
    positions: wp.array[wp.vec3],
    surface_vertex_indices: wp.array[int],
    triangle_orientation_signs: wp.array[int],
    triangle_edge_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
    vertex_triangle_offsets: wp.array[int],
    vertex_triangle_indices: wp.array[int],
    vertex_neighbor_offsets: wp.array[int],
    vertex_neighbors: wp.array[int],
    particle_world: wp.array[int],
    triangle_indices: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    vertex = surface_vertex_indices[wp.tid()]
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

        topology_local = False
        if geometry_radius_topology_local_only != 0:
            topology_local = (
                _is_vertex_topology_neighbor(vertex, index_0, vertex_neighbor_offsets, vertex_neighbors)
                or _is_vertex_topology_neighbor(vertex, index_1, vertex_neighbor_offsets, vertex_neighbors)
                or _is_vertex_topology_neighbor(vertex, index_2, vertex_neighbor_offsets, vertex_neighbors)
            )

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

        effective_thickness = thickness
        if use_geometry_radii != 0 and (geometry_radius_topology_local_only == 0 or topology_local):
            face_radius = (
                barycentric[0] * particle_radii[index_0]
                + barycentric[1] * particle_radii[index_1]
                + barycentric[2] * particle_radii[index_2]
            )
            effective_thickness = particle_radii[vertex] + face_radius
        if distance >= effective_thickness:
            continue

        direction = separation / distance
        signed_distance = distance
        if triangle_orientation_signs[triangle] != 0:
            feature_normal = _oriented_closest_feature_normal(
                triangle,
                barycentric,
                positions,
                triangle_indices,
                triangle_edge_indices,
                edge_triangle_indices,
                triangle_orientation_signs,
                vertex_triangle_offsets,
                vertex_triangle_indices,
            )
            if wp.length(feature_normal) <= _MIN_GEOMETRY_NORM:
                continue
            if wp.dot(separation, feature_normal) < 0.0:
                direction = -direction
                signed_distance = -distance
        depth = effective_thickness - signed_distance

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
        contact_depths[contact] = depth


@wp.kernel
def _detect_edge_edge_contacts(
    edge_bvh_id: wp.uint64,
    thickness: float,
    particle_radii: wp.array[float],
    use_geometry_radii: int,
    geometry_radius_topology_local_only: int,
    capacity: int,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    particle_world: wp.array[int],
    triangle_indices: wp.array2d[int],
    edge_indices: wp.array2d[int],
    edge_triangle_indices: wp.array2d[int],
    triangle_orientation_signs: wp.array[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_mollifier_thresholds: wp.array[float],
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
        if parameter_0 <= 0.0 or parameter_0 >= 1.0 or parameter_1 <= 0.0 or parameter_1 >= 1.0:
            continue

        closest_0 = wp.lerp(position_0, position_1, parameter_0)
        closest_1 = wp.lerp(position_2, position_3, parameter_1)
        separation = closest_0 - closest_1
        distance = wp.length(separation)
        topology_local = _is_topology_local_edge_pair(
            edge,
            other_edge,
            index_0,
            index_1,
            index_2,
            index_3,
            edge_indices,
        )
        limited_thickness = thickness
        if use_geometry_radii != 0 and geometry_radius_topology_local_only == 0:
            radius_0 = (1.0 - parameter_0) * particle_radii[index_0]
            radius_0 += parameter_0 * particle_radii[index_1]
            radius_1 = (1.0 - parameter_1) * particle_radii[index_2]
            radius_1 += parameter_1 * particle_radii[index_3]
            limited_thickness = radius_0 + radius_1
        elif topology_local:
            if use_geometry_radii != 0:
                radius_0 = (1.0 - parameter_0) * particle_radii[index_0]
                radius_0 += parameter_0 * particle_radii[index_1]
                radius_1 = (1.0 - parameter_1) * particle_radii[index_2]
                radius_1 += parameter_1 * particle_radii[index_3]
                limited_thickness = wp.min(limited_thickness, radius_0 + radius_1)
            average_length = 0.5 * (wp.length(position_1 - position_0) + wp.length(position_3 - position_2))
            limited_thickness = wp.min(limited_thickness, 0.5 * average_length)
        if distance >= limited_thickness:
            continue

        direction = wp.vec3(0.0)
        depth = 0.0
        pseudo_normal_0 = _oriented_edge_pseudo_normal(
            edge,
            positions,
            triangle_indices,
            edge_triangle_indices,
            triangle_orientation_signs,
        )
        pseudo_normal_1 = _oriented_edge_pseudo_normal(
            other_edge,
            positions,
            triangle_indices,
            edge_triangle_indices,
            triangle_orientation_signs,
        )
        oriented_0 = wp.length(pseudo_normal_0) > _MIN_GEOMETRY_NORM
        oriented_1 = wp.length(pseudo_normal_1) > _MIN_GEOMETRY_NORM
        if oriented_0 or oriented_1:
            direction_raw = wp.vec3(0.0)
            if oriented_0 and oriented_1:
                direction_raw = pseudo_normal_1 - pseudo_normal_0
            elif oriented_0:
                direction_raw = -pseudo_normal_0
            else:
                direction_raw = pseudo_normal_1
            direction_length = wp.length(direction_raw)
            if direction_length <= _MIN_GEOMETRY_NORM:
                continue
            direction = direction_raw / direction_length
            signed_distance = wp.dot(separation, direction)
            if signed_distance >= limited_thickness:
                continue
            depth = limited_thickness - signed_distance
        else:
            if distance <= _MIN_CONTACT_DISTANCE:
                continue
            direction = separation / distance
            depth = limited_thickness - distance

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
        contact_depths[contact] = depth
        contact_mollifier_thresholds[contact] = 0.0
        if topology_local:
            rest_edge_0 = rest_positions[index_1] - rest_positions[index_0]
            rest_edge_1 = rest_positions[index_3] - rest_positions[index_2]
            contact_mollifier_thresholds[contact] = (
                _EE_MOLLIFIER_THRESHOLD_SCALE
                * wp.dot(
                    rest_edge_0,
                    rest_edge_0,
                )
                * wp.dot(
                    rest_edge_1,
                    rest_edge_1,
                )
            )


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
    projected_sum = float(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        projected_sum += weights[contact, local_index] * wp.dot(direction, vector[particle])
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


@wp.func
def _edge_edge_mollifier_is_active(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    position_3: wp.vec3,
    threshold: float,
):
    if threshold <= _MIN_GEOMETRY_NORM * _MIN_GEOMETRY_NORM:
        return False
    cross_product = wp.cross(position_1 - position_0, position_3 - position_2)
    return wp.dot(cross_product, cross_product) < threshold


@wp.func
def _edge_edge_friction_load_scale(
    ids: wp.array2d[int],
    contact: int,
    thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    positions: wp.array[wp.vec3],
):
    if mollifier_active[contact] == 0:
        return float(1.0)
    edge_0 = positions[ids[contact, 1]] - positions[ids[contact, 0]]
    edge_1 = positions[ids[contact, 3]] - positions[ids[contact, 2]]
    cross_product = wp.cross(edge_0, edge_1)
    cross_squared = wp.dot(cross_product, cross_product)
    threshold = thresholds[contact]
    if threshold <= _MIN_GEOMETRY_NORM * _MIN_GEOMETRY_NORM:
        return float(0.0)
    return wp.clamp(
        cross_squared * (2.0 * threshold - cross_squared) / (threshold * threshold),
        0.0,
        1.0,
    )


@wp.func
def _contact_relative_displacement(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    contact: int,
    arity: int,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
):
    relative_displacement = wp.vec3(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        relative_displacement += weights[contact, local_index] * (positions[particle] - anchor_positions[particle])
    return relative_displacement


@wp.func
def _regularized_friction_force_hessian(
    direction: wp.vec3,
    relative_displacement: wp.vec3,
    normal_load: float,
    friction: float,
    displacement_epsilon: float,
):
    tangent_displacement = relative_displacement - direction * wp.dot(direction, relative_displacement)
    tangent_length = wp.length(tangent_displacement)
    if tangent_length <= 0.0 or normal_load <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)

    friction_over_length = float(0.0)
    if tangent_length > displacement_epsilon:
        friction_over_length = 1.0 / tangent_length
    else:
        friction_over_length = (-tangent_length / displacement_epsilon + 2.0) / displacement_epsilon
    scale = friction * normal_load * friction_over_length
    tangent_projector = wp.identity(3, float) - wp.outer(direction, direction)
    return -scale * tangent_displacement, scale * tangent_projector


@wp.func
def _contact_friction_force_hessian(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    contact: int,
    arity: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    use_mollifier: int,
):
    load_scale = float(1.0)
    if use_mollifier != 0:
        load_scale = _edge_edge_friction_load_scale(
            ids,
            contact,
            mollifier_thresholds,
            mollifier_active,
            positions,
        )
    relative_displacement = _contact_relative_displacement(
        ids,
        weights,
        contact,
        arity,
        positions,
        anchor_positions,
    )
    return _regularized_friction_force_hessian(
        directions[contact],
        relative_displacement,
        stiffness * depths[contact] * load_scale,
        friction,
        displacement_epsilon,
    )


@wp.kernel
def _accumulate_contact_friction_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    arity: int,
    capacity: int,
    use_mollifier: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    friction_force, _friction_hessian = _contact_friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        positions,
        anchor_positions,
        mollifier_thresholds,
        mollifier_active,
        use_mollifier,
    )
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * friction_force)


@wp.kernel
def _contact_friction_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    arity: int,
    capacity: int,
    use_mollifier: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    _friction_force, friction_hessian = _contact_friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        positions,
        anchor_positions,
        mollifier_thresholds,
        mollifier_active,
        use_mollifier,
    )
    relative_vector = wp.vec3(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        relative_vector += weights[contact, local_index] * vector[particle]
    friction_product = friction_hessian * relative_vector
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * friction_product)


@wp.kernel
def _accumulate_contact_friction_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    arity: int,
    capacity: int,
    use_mollifier: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    _friction_force, friction_hessian = _contact_friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        positions,
        anchor_positions,
        mollifier_thresholds,
        mollifier_active,
        use_mollifier,
    )
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * friction_hessian)


@wp.kernel
def _accumulate_contact_friction_force_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    arity: int,
    feature_split: int,
    capacity: int,
    use_mollifier: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
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
    friction_force, _friction_hessian = _contact_friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        positions,
        anchor_positions,
        mollifier_thresholds,
        mollifier_active,
        use_mollifier,
    )
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * friction_force)


@wp.kernel
def _contact_friction_hessian_multiply_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    arity: int,
    feature_split: int,
    capacity: int,
    use_mollifier: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    vector: wp.array[wp.vec3],
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
    _friction_force, friction_hessian = _contact_friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        positions,
        anchor_positions,
        mollifier_thresholds,
        mollifier_active,
        use_mollifier,
    )
    relative_vector = wp.vec3(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        relative_vector += weights[contact, local_index] * vector[particle]
    friction_product = friction_hessian * relative_vector
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * friction_product)


@wp.kernel
def _accumulate_contact_friction_diagonal_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    arity: int,
    feature_split: int,
    capacity: int,
    use_mollifier: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    output: wp.array[wp.mat33],
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
    _friction_force, friction_hessian = _contact_friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        positions,
        anchor_positions,
        mollifier_thresholds,
        mollifier_active,
        use_mollifier,
    )
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * friction_hessian)


@wp.kernel
def _prepare_edge_edge_mollifier(
    ids: wp.array2d[int],
    mollifier_thresholds: wp.array[float],
    count: wp.array[int],
    capacity: int,
    positions: wp.array[wp.vec3],
    mollifier_active: wp.array[int],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    position_2 = positions[index_2]
    position_3 = positions[index_3]
    threshold = mollifier_thresholds[contact]
    active = _edge_edge_mollifier_is_active(position_0, position_1, position_2, position_3, threshold)
    mollifier_active[contact] = wp.int32(active)


@wp.func
def _edge_edge_gauss_newton_multiply(
    edge_0: wp.vec3,
    edge_1: wp.vec3,
    weights: wp.vec4,
    direction: wp.vec3,
    depth: float,
    threshold: float,
    vector_0: wp.vec3,
    vector_1: wp.vec3,
    vector_2: wp.vec3,
    vector_3: wp.vec3,
):
    edge_delta_0 = vector_1 - vector_0
    edge_delta_1 = vector_3 - vector_2
    depth_delta = -wp.dot(
        direction,
        weights[0] * vector_0 + weights[1] * vector_1 + weights[2] * vector_2 + weights[3] * vector_3,
    )
    residual_product = _edge_edge_mollified_residual_jacobian_multiply(
        edge_0,
        edge_1,
        depth,
        threshold,
        edge_delta_0,
        edge_delta_1,
        depth_delta,
    )
    edge_product_0, edge_product_1, depth_product = _edge_edge_mollified_residual_jacobian_transpose_multiply(
        edge_0,
        edge_1,
        depth,
        threshold,
        residual_product,
    )
    return (
        -edge_product_0 - weights[0] * depth_product * direction,
        edge_product_0 - weights[1] * depth_product * direction,
        -edge_product_1 - weights[2] * depth_product * direction,
        edge_product_1 - weights[3] * depth_product * direction,
    )


@wp.func
def _edge_edge_gauss_newton_diagonal_block(
    edge_0: wp.vec3,
    edge_1: wp.vec3,
    weight: float,
    direction: wp.vec3,
    depth: float,
    threshold: float,
    local_index: int,
):
    columns = wp.mat33(0.0)
    for axis in range(3):
        basis = wp.vec3(0.0)
        basis[axis] = 1.0
        edge_delta_0 = wp.vec3(0.0)
        edge_delta_1 = wp.vec3(0.0)
        if local_index == 0:
            edge_delta_0 = -basis
        elif local_index == 1:
            edge_delta_0 = basis
        elif local_index == 2:
            edge_delta_1 = -basis
        else:
            edge_delta_1 = basis
        residual_product = _edge_edge_mollified_residual_jacobian_multiply(
            edge_0,
            edge_1,
            depth,
            threshold,
            edge_delta_0,
            edge_delta_1,
            -weight * direction[axis],
        )
        edge_product_0, edge_product_1, depth_product = _edge_edge_mollified_residual_jacobian_transpose_multiply(
            edge_0,
            edge_1,
            depth,
            threshold,
            residual_product,
        )
        if local_index == 0:
            local_product = -edge_product_0
        elif local_index == 1:
            local_product = edge_product_0
        elif local_index == 2:
            local_product = -edge_product_1
        else:
            local_product = edge_product_1
        local_product -= weight * depth_product * direction
        columns[0, axis] = local_product[0]
        columns[1, axis] = local_product[1]
        columns[2, axis] = local_product[2]
    return columns


@wp.kernel
def _accumulate_mollified_edge_edge_force(
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
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    position_2 = positions[index_2]
    position_3 = positions[index_3]
    direction = directions[contact]
    depth = depths[contact]
    threshold = mollifier_thresholds[contact]
    if mollifier_active[contact] != 0:
        contact_weights = wp.vec4(
            weights[contact, 0],
            weights[contact, 1],
            weights[contact, 2],
            weights[contact, 3],
        )
        edge_0 = position_1 - position_0
        edge_1 = position_3 - position_2
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
        wp.atomic_add(output, index_0, -stiffness * gradient_0)
        wp.atomic_add(output, index_1, -stiffness * gradient_1)
        wp.atomic_add(output, index_2, -stiffness * gradient_2)
        wp.atomic_add(output, index_3, -stiffness * gradient_3)
        return

    scaled_direction = stiffness * depth * direction
    wp.atomic_add(output, index_0, weights[contact, 0] * scaled_direction)
    wp.atomic_add(output, index_1, weights[contact, 1] * scaled_direction)
    wp.atomic_add(output, index_2, weights[contact, 2] * scaled_direction)
    wp.atomic_add(output, index_3, weights[contact, 3] * scaled_direction)


@wp.kernel
def _mollified_edge_edge_hessian_multiply(
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
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    position_2 = positions[index_2]
    position_3 = positions[index_3]
    direction = directions[contact]
    if mollifier_active[contact] != 0:
        contact_weights = wp.vec4(weights[contact, 0], weights[contact, 1], weights[contact, 2], weights[contact, 3])
        product_0, product_1, product_2, product_3 = _edge_edge_gauss_newton_multiply(
            position_1 - position_0,
            position_3 - position_2,
            contact_weights,
            direction,
            depths[contact],
            mollifier_thresholds[contact],
            vector[index_0],
            vector[index_1],
            vector[index_2],
            vector[index_3],
        )
        wp.atomic_add(output, index_0, stiffness * product_0)
        wp.atomic_add(output, index_1, stiffness * product_1)
        wp.atomic_add(output, index_2, stiffness * product_2)
        wp.atomic_add(output, index_3, stiffness * product_3)
        return

    projected_sum = (
        weights[contact, 0] * wp.dot(direction, vector[index_0])
        + weights[contact, 1] * wp.dot(direction, vector[index_1])
        + weights[contact, 2] * wp.dot(direction, vector[index_2])
        + weights[contact, 3] * wp.dot(direction, vector[index_3])
    )
    scaled_direction = stiffness * projected_sum * direction
    wp.atomic_add(output, index_0, weights[contact, 0] * scaled_direction)
    wp.atomic_add(output, index_1, weights[contact, 1] * scaled_direction)
    wp.atomic_add(output, index_2, weights[contact, 2] * scaled_direction)
    wp.atomic_add(output, index_3, weights[contact, 3] * scaled_direction)


@wp.kernel
def _accumulate_mollified_edge_edge_diagonal(
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
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    position_2 = positions[index_2]
    position_3 = positions[index_3]
    direction = directions[contact]
    if mollifier_active[contact] != 0:
        edge_0 = position_1 - position_0
        edge_1 = position_3 - position_2
        for local_index in range(4):
            particle = ids[contact, local_index]
            block = _edge_edge_gauss_newton_diagonal_block(
                edge_0,
                edge_1,
                weights[contact, local_index],
                direction,
                depths[contact],
                mollifier_thresholds[contact],
                local_index,
            )
            wp.atomic_add(output, particle, stiffness * block)
        return

    rank_one = stiffness * wp.outer(direction, direction)
    for local_index in range(4):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * rank_one)


@wp.kernel
def _accumulate_mollified_edge_edge_force_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    positions: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    stiffness = _adaptive_contact_stiffness(
        ids, directions, contact, 4, 2, factor, static_diagonal, masses, inv_dt_squared
    )

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    position_2 = positions[index_2]
    position_3 = positions[index_3]
    direction = directions[contact]
    depth = depths[contact]
    threshold = mollifier_thresholds[contact]
    if mollifier_active[contact] != 0:
        contact_weights = wp.vec4(weights[contact, 0], weights[contact, 1], weights[contact, 2], weights[contact, 3])
        edge_0 = position_1 - position_0
        edge_1 = position_3 - position_2
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
        wp.atomic_add(output, index_0, -stiffness * gradient_0)
        wp.atomic_add(output, index_1, -stiffness * gradient_1)
        wp.atomic_add(output, index_2, -stiffness * gradient_2)
        wp.atomic_add(output, index_3, -stiffness * gradient_3)
        return

    scaled_direction = stiffness * depth * direction
    wp.atomic_add(output, index_0, weights[contact, 0] * scaled_direction)
    wp.atomic_add(output, index_1, weights[contact, 1] * scaled_direction)
    wp.atomic_add(output, index_2, weights[contact, 2] * scaled_direction)
    wp.atomic_add(output, index_3, weights[contact, 3] * scaled_direction)


@wp.kernel
def _mollified_edge_edge_hessian_multiply_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    positions: wp.array[wp.vec3],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    stiffness = _adaptive_contact_stiffness(
        ids, directions, contact, 4, 2, factor, static_diagonal, masses, inv_dt_squared
    )

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    position_2 = positions[index_2]
    position_3 = positions[index_3]
    direction = directions[contact]
    if mollifier_active[contact] != 0:
        contact_weights = wp.vec4(weights[contact, 0], weights[contact, 1], weights[contact, 2], weights[contact, 3])
        product_0, product_1, product_2, product_3 = _edge_edge_gauss_newton_multiply(
            position_1 - position_0,
            position_3 - position_2,
            contact_weights,
            direction,
            depths[contact],
            mollifier_thresholds[contact],
            vector[index_0],
            vector[index_1],
            vector[index_2],
            vector[index_3],
        )
        wp.atomic_add(output, index_0, stiffness * product_0)
        wp.atomic_add(output, index_1, stiffness * product_1)
        wp.atomic_add(output, index_2, stiffness * product_2)
        wp.atomic_add(output, index_3, stiffness * product_3)
        return

    projected_sum = (
        weights[contact, 0] * wp.dot(direction, vector[index_0])
        + weights[contact, 1] * wp.dot(direction, vector[index_1])
        + weights[contact, 2] * wp.dot(direction, vector[index_2])
        + weights[contact, 3] * wp.dot(direction, vector[index_3])
    )
    scaled_direction = stiffness * projected_sum * direction
    wp.atomic_add(output, index_0, weights[contact, 0] * scaled_direction)
    wp.atomic_add(output, index_1, weights[contact, 1] * scaled_direction)
    wp.atomic_add(output, index_2, weights[contact, 2] * scaled_direction)
    wp.atomic_add(output, index_3, weights[contact, 3] * scaled_direction)


@wp.kernel
def _accumulate_mollified_edge_edge_diagonal_adaptive(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    factor: float,
    static_diagonal: wp.array[wp.mat33],
    masses: wp.array[float],
    inv_dt_squared: float,
    positions: wp.array[wp.vec3],
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    stiffness = _adaptive_contact_stiffness(
        ids, directions, contact, 4, 2, factor, static_diagonal, masses, inv_dt_squared
    )

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    index_2 = ids[contact, 2]
    index_3 = ids[contact, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    position_2 = positions[index_2]
    position_3 = positions[index_3]
    direction = directions[contact]
    if mollifier_active[contact] != 0:
        edge_0 = position_1 - position_0
        edge_1 = position_3 - position_2
        for local_index in range(4):
            particle = ids[contact, local_index]
            block = _edge_edge_gauss_newton_diagonal_block(
                edge_0,
                edge_1,
                weights[contact, local_index],
                direction,
                depths[contact],
                mollifier_thresholds[contact],
                local_index,
            )
            wp.atomic_add(output, particle, stiffness * block)
        return

    rank_one = stiffness * wp.outer(direction, direction)
    for local_index in range(4):
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

    def accumulate_friction_force(
        self,
        stiffness: float,
        friction: float,
        displacement_epsilon: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add regularized Coulomb friction forces."""
        self._validate_friction_data(positions, anchor_positions, output, wp.vec3)
        mollifier_thresholds, mollifier_active, use_mollifier = self._friction_mollifier_data()
        wp.launch(
            _accumulate_contact_friction_force,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                mollifier_thresholds,
                mollifier_active,
                self.count,
                self.arity,
                self.capacity,
                use_mollifier,
                stiffness,
                friction,
                displacement_epsilon,
                positions,
                anchor_positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def friction_hessian_multiply(
        self,
        stiffness: float,
        friction: float,
        displacement_epsilon: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add regularized friction Hessian-vector products."""
        self._validate_friction_data(positions, anchor_positions, vector, wp.vec3)
        self._validate_output(output, wp.vec3)
        if len(output) != len(vector):
            raise ValueError("vector and output must have the same length")
        mollifier_thresholds, mollifier_active, use_mollifier = self._friction_mollifier_data()
        wp.launch(
            _contact_friction_hessian_multiply,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                mollifier_thresholds,
                mollifier_active,
                self.count,
                self.arity,
                self.capacity,
                use_mollifier,
                stiffness,
                friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_friction_diagonal(
        self,
        stiffness: float,
        friction: float,
        displacement_epsilon: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array[wp.mat33],
    ) -> None:
        """Add diagonal blocks of the regularized friction Hessian."""
        self._validate_friction_data(positions, anchor_positions, output, wp.mat33)
        mollifier_thresholds, mollifier_active, use_mollifier = self._friction_mollifier_data()
        wp.launch(
            _accumulate_contact_friction_diagonal,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                mollifier_thresholds,
                mollifier_active,
                self.count,
                self.arity,
                self.capacity,
                use_mollifier,
                stiffness,
                friction,
                displacement_epsilon,
                positions,
                anchor_positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_friction_force_adaptive(
        self,
        factor: float,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
        inv_dt_squared: float,
        friction: float,
        displacement_epsilon: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add adaptive-stiffness regularized Coulomb friction forces."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_friction_data(positions, anchor_positions, output, wp.vec3)
        mollifier_thresholds, mollifier_active, use_mollifier = self._friction_mollifier_data()
        wp.launch(
            _accumulate_contact_friction_force_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                mollifier_thresholds,
                mollifier_active,
                self.count,
                self.arity,
                self.feature_split,
                self.capacity,
                use_mollifier,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
                friction,
                displacement_epsilon,
                positions,
                anchor_positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def friction_hessian_multiply_adaptive(
        self,
        factor: float,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
        inv_dt_squared: float,
        friction: float,
        displacement_epsilon: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add adaptive-stiffness friction Hessian-vector products."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_friction_data(positions, anchor_positions, vector, wp.vec3)
        self._validate_output(output, wp.vec3)
        if len(output) != len(vector):
            raise ValueError("vector and output must have the same length")
        mollifier_thresholds, mollifier_active, use_mollifier = self._friction_mollifier_data()
        wp.launch(
            _contact_friction_hessian_multiply_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                mollifier_thresholds,
                mollifier_active,
                self.count,
                self.arity,
                self.feature_split,
                self.capacity,
                use_mollifier,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
                friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_friction_diagonal_adaptive(
        self,
        factor: float,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
        inv_dt_squared: float,
        friction: float,
        displacement_epsilon: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array[wp.mat33],
    ) -> None:
        """Add adaptive-stiffness friction diagonal blocks."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_friction_data(positions, anchor_positions, output, wp.mat33)
        mollifier_thresholds, mollifier_active, use_mollifier = self._friction_mollifier_data()
        wp.launch(
            _accumulate_contact_friction_diagonal_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                mollifier_thresholds,
                mollifier_active,
                self.count,
                self.arity,
                self.feature_split,
                self.capacity,
                use_mollifier,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
                friction,
                displacement_epsilon,
                positions,
                anchor_positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def _friction_mollifier_data(self) -> tuple[wp.array[float], wp.array[int], int]:
        return self.depths, self.count, 0

    def _validate_friction_data(
        self,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array,
        output_dtype: Any,
    ) -> None:
        self._validate_output(positions, wp.vec3)
        self._validate_output(anchor_positions, wp.vec3)
        self._validate_output(output, output_dtype)
        if len(positions) != len(anchor_positions) or len(positions) != len(output):
            raise ValueError("friction particle arrays must have the same length")

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


class _EdgeEdgeContactBuffer(_ContactBuffer):
    """Four-particle EE contacts with a topology-local IPC mollifier."""

    def __init__(self, capacity: int, device: Any):
        super().__init__(arity=4, capacity=capacity, device=device, feature_split=2)
        self.mollifier_thresholds = wp.zeros(capacity, dtype=wp.float32, device=self.device)
        self.mollifier_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)

    def prepare_hessian(self, positions: wp.array[wp.vec3]) -> None:
        """Mark EE contacts whose IPC mollifier is active."""
        self._validate_output(positions, wp.vec3)
        wp.launch(
            _prepare_edge_edge_mollifier,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.mollifier_thresholds,
                self.count,
                self.capacity,
                positions,
            ],
            outputs=[self.mollifier_active],
            device=self.device,
        )

    def _friction_mollifier_data(self) -> tuple[wp.array[float], wp.array[int], int]:
        return self.mollifier_thresholds, self.mollifier_active, 1

    def accumulate_force(
        self,
        stiffness: float,
        positions: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add exact forces of the IPC-mollified EE penalty energy."""
        self._validate_particle_vectors(positions, output)
        wp.launch(
            _accumulate_mollified_edge_edge_force,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                stiffness,
                positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def hessian_multiply(
        self,
        stiffness: float,
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add Gauss-Newton products of the mollified EE energy."""
        self._validate_particle_vectors(positions, vector, output)
        wp.launch(
            _mollified_edge_edge_hessian_multiply,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                stiffness,
                positions,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_diagonal(
        self,
        stiffness: float,
        positions: wp.array[wp.vec3],
        output: wp.array[wp.mat33],
    ) -> None:
        """Add exact diagonal blocks of the Gauss-Newton EE operator."""
        self._validate_output(positions, wp.vec3)
        self._validate_output(output, wp.mat33)
        if len(positions) != len(output):
            raise ValueError("positions and output must have the same length")
        wp.launch(
            _accumulate_mollified_edge_edge_diagonal,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                stiffness,
                positions,
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
        positions: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add mollified EE forces using adaptive directional stiffness."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_particle_vectors(positions, output)
        wp.launch(
            _accumulate_mollified_edge_edge_force_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
                positions,
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
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add adaptive PSD products of the mollified EE energy."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_particle_vectors(positions, vector, output)
        wp.launch(
            _mollified_edge_edge_hessian_multiply_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
                positions,
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
        positions: wp.array[wp.vec3],
        output: wp.array[wp.mat33],
    ) -> None:
        """Add adaptive diagonal blocks of the mollified EE operator."""
        self._validate_adaptive_data(factor, static_diagonal, masses, inv_dt_squared)
        self._validate_output(positions, wp.vec3)
        self._validate_output(output, wp.mat33)
        if len(positions) != len(output):
            raise ValueError("positions and output must have the same length")
        wp.launch(
            _accumulate_mollified_edge_edge_diagonal_adaptive,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                factor,
                static_diagonal,
                masses,
                inv_dt_squared,
                positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def _validate_particle_vectors(self, *arrays: wp.array[wp.vec3]) -> None:
        for array in arrays:
            self._validate_output(array, wp.vec3)
        if any(len(array) != len(arrays[0]) for array in arrays[1:]):
            raise ValueError("particle vectors must have the same length")


class ConstraintSelfCollision:
    """Matrix-free cloth self-collision constraints.

    Attributes:
        particle_radii: One-sided collision radii [m], shape
            ``[particle_count]``.
    """

    particle_radii: wp.array[float]

    def __init__(
        self,
        model: Model,
        thickness: float | None,
        stiffness: float | None,
        untangle_stiffness: float | None = None,
        max_contacts: int = 32768,
        stiffness_factors: tuple[float, float, float] | None = None,
        geometry_radius_scale: float | None = None,
        geometry_radius_topology_local_only: bool = False,
        friction: float = 0.0,
        friction_epsilon: float = 1.0e-2,
        enable_edge_face: bool = True,
        use_outward_normals: bool = False,
    ):
        """Create a fixed-capacity GPU cloth self-collision operator.

        Args:
            model: Particle triangle-mesh model whose topology remains fixed.
            thickness: Nominal two-surface collision activation distance [m].
                Pass ``None`` to estimate ``min(0.8 * two-ring rest
                clearance, 0.005 m)`` from the surface geometry.
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
                initial recommended value is ``0.25``. Particles absent from
                the collision surface receive zero radius.
            geometry_radius_topology_local_only: Whether geometry-aware radii
                apply only to topology-local VF/EE pairs. Nonlocal pairs keep
                the nominal ``thickness``. Requires ``geometry_radius_scale``.
            friction: Coulomb friction coefficient for vertex-face and
                edge-edge contacts.
            friction_epsilon: Relative-velocity regularization threshold [m/s].
            enable_edge_face: Whether to detect and assemble edge-face
                intersection recovery contacts.
            use_outward_normals: Whether to use oriented signed VF/EE contact
                for non-tetrahedral triangles, assuming outward winding.
                Tetrahedral boundary features are always oriented outward;
                leave this false to keep other triangles two-sided.
        """
        thickness_was_estimated = thickness is None
        if thickness is not None and (not np.isfinite(thickness) or thickness <= 0.0):
            raise ValueError("thickness must be finite and positive")
        if not np.isfinite(friction):
            raise ValueError("friction must be finite")
        if friction < 0.0:
            raise ValueError("friction must be nonnegative")
        if not np.isfinite(friction_epsilon):
            raise ValueError("friction_epsilon must be finite")
        if friction_epsilon <= 0.0:
            raise ValueError("friction_epsilon must be positive")
        if geometry_radius_scale is not None:
            if not np.isfinite(geometry_radius_scale):
                raise ValueError("geometry_radius_scale must be finite")
            if geometry_radius_scale <= 0.0:
                raise ValueError("geometry_radius_scale must be positive")
            geometry_radius_scale = float(geometry_radius_scale)
        if geometry_radius_topology_local_only and geometry_radius_scale is None:
            raise ValueError("geometry_radius_topology_local_only requires geometry_radius_scale")
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
        self.stiffness = validated_stiffness
        self.untangle_stiffness = validated_untangle_stiffness
        self.stiffness_factors = validated_stiffness_factors
        self.max_contacts = int(max_contacts)
        self.particle_world = model.particle_world
        self.rest_positions = wp.clone(model.particle_q)
        self.friction = float(friction)
        self.friction_epsilon = float(friction_epsilon)
        self.enable_edge_face = bool(enable_edge_face)
        self.use_outward_normals = bool(use_outward_normals)
        self._friction_positions: wp.array[wp.vec3] | None = None
        if self.friction > 0.0:
            self._friction_positions = wp.empty_like(model.particle_q)
        self._friction_displacement_epsilon = 0.0
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

        surface_vertex_indices = np.unique(triangle_indices.reshape(-1)).astype(np.int32)
        if len(surface_vertex_indices) == 0:
            raise ValueError("ConstraintSelfCollision requires at least one surface vertex")

        mesh_adjacency = MeshAdjacency(triangle_indices)
        edge_indices = mesh_adjacency.edge_indices
        if len(edge_indices) == 0:
            raise ValueError("ConstraintSelfCollision requires at least one mesh edge")
        rest_positions = np.asarray(model.particle_q.numpy(), dtype=np.float64)
        tetrahedron_indices = np.empty((0, 4), dtype=np.int32)
        if model.tet_count > 0 and model.tet_indices is not None:
            tetrahedron_indices = np.asarray(model.tet_indices.numpy(), dtype=np.int32).reshape(-1, 4)
        tetrahedral_orientation_signs = _tetrahedral_triangle_orientation_signs(
            rest_positions,
            tetrahedron_indices,
            triangle_indices,
        )
        self.tetrahedral_triangle_count = int(np.count_nonzero(tetrahedral_orientation_signs))
        triangle_orientation_signs = tetrahedral_orientation_signs.copy()
        if self.use_outward_normals:
            triangle_orientation_signs[triangle_orientation_signs == 0] = 1
        if thickness is None:
            two_ring_upper_bound = _compute_two_ring_collision_upper_bound(
                rest_positions,
                triangle_indices,
                edge_indices,
                self.device,
            )
            if np.isfinite(two_ring_upper_bound):
                if two_ring_upper_bound <= 0.0:
                    raise ValueError("automatic thickness requires positive two-ring rest clearance")
                thickness = min(
                    _AUTOMATIC_THICKNESS_ETA * two_ring_upper_bound,
                    _AUTOMATIC_THICKNESS_MAX,
                )
            else:
                thickness = _AUTOMATIC_THICKNESS_MAX
        self.thickness = float(thickness)
        self.thickness_was_estimated = thickness_was_estimated

        nominal_radius = 0.5 * self.thickness
        if geometry_radius_scale is None:
            particle_radii = np.full(model.particle_count, nominal_radius, dtype=np.float32)
        else:
            particle_radii = _compute_geometry_aware_particle_radii(
                rest_positions,
                triangle_indices,
                nominal_radius,
                geometry_radius_scale,
            )
        self.geometry_radius_scale = geometry_radius_scale
        self.geometry_radius_topology_local_only = bool(geometry_radius_topology_local_only)
        self.particle_radii = wp.array(particle_radii, dtype=wp.float32, device=self.device)
        self._use_geometry_radii = int(geometry_radius_scale is not None)

        vertex_neighbor_offsets = np.zeros(model.particle_count + 1, dtype=np.int32)
        vertex_neighbors = np.empty(0, dtype=np.int32)
        if self.geometry_radius_topology_local_only:
            one_ring_neighbors = [set() for _ in range(model.particle_count)]
            for index_0, index_1 in edge_indices[:, 2:4]:
                one_ring_neighbors[index_0].add(int(index_1))
                one_ring_neighbors[index_1].add(int(index_0))
            vertex_neighbor_offsets[1:] = np.cumsum(
                [len(neighbors) for neighbors in one_ring_neighbors],
                dtype=np.int32,
            )
            vertex_neighbors = np.fromiter(
                (neighbor for neighbors in one_ring_neighbors for neighbor in sorted(neighbors)),
                dtype=np.int32,
                count=int(vertex_neighbor_offsets[-1]),
            )
        self.triangle_indices = wp.array(triangle_indices, dtype=wp.int32, device=self.device)
        self.surface_vertex_indices = wp.array(surface_vertex_indices, dtype=wp.int32, device=self.device)
        self.edge_indices = wp.array(edge_indices, dtype=wp.int32, device=self.device)
        self.triangle_orientation_signs = wp.array(
            triangle_orientation_signs,
            dtype=wp.int32,
            device=self.device,
        )
        self.triangle_edge_indices = wp.array(
            mesh_adjacency.tri_edge_indices,
            dtype=wp.int32,
            device=self.device,
        )
        self.edge_triangle_indices = wp.array(
            mesh_adjacency.edge_tri_indices,
            dtype=wp.int32,
            device=self.device,
        )
        vertex_triangles = [set() for _ in range(model.particle_count)]
        for triangle, indices in enumerate(triangle_indices):
            for vertex in indices:
                vertex_triangles[int(vertex)].add(triangle)
        vertex_triangle_offsets, vertex_triangle_indices = _pack_index_sets(vertex_triangles)
        self.vertex_triangle_offsets = wp.array(vertex_triangle_offsets, dtype=wp.int32, device=self.device)
        self.vertex_triangle_indices = wp.array(vertex_triangle_indices, dtype=wp.int32, device=self.device)
        self.vertex_neighbor_offsets = wp.array(vertex_neighbor_offsets, dtype=wp.int32, device=self.device)
        self.vertex_neighbors = wp.array(vertex_neighbors, dtype=wp.int32, device=self.device)
        self.triangle_count = len(triangle_indices)
        self.surface_vertex_count = len(surface_vertex_indices)
        self.edge_count = len(edge_indices)

        self.triangle_lower_bounds = wp.empty(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_upper_bounds = wp.empty_like(self.triangle_lower_bounds)
        self.edge_lower_bounds = wp.empty(self.edge_count, dtype=wp.vec3, device=self.device)
        self.edge_upper_bounds = wp.empty_like(self.edge_lower_bounds)
        self._update_bounds(model.particle_q)
        self.triangle_bvh = wp.Bvh(self.triangle_lower_bounds, self.triangle_upper_bounds)
        self.edge_bvh = wp.Bvh(self.edge_lower_bounds, self.edge_upper_bounds)

        self.vertex_face_contacts = _ContactBuffer(4, max_contacts, self.device, feature_split=1)
        self.edge_edge_contacts = _EdgeEdgeContactBuffer(max_contacts, self.device)
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
        """Cache step data for adaptive stiffness and friction."""
        if self.stiffness_factors is None and self.friction == 0.0:
            return
        self._validate_positions(positions)
        if velocities.device != self.device:
            raise ValueError(f"velocities must use device {self.device}")
        if velocities.dtype != wp.vec3 or len(velocities) != self.particle_count:
            raise ValueError(f"velocities must contain {self.particle_count} wp.vec3 values")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.stiffness_factors is not None:
            if self._static_diagonal is None or self._masses is None:
                raise RuntimeError("bind_static_system() must be called before begin_step() in adaptive mode")
            self._inv_dt_squared = 1.0 / (dt * dt)
        if self.friction > 0.0:
            if self._friction_positions is None:
                raise RuntimeError("friction anchor storage is unavailable")
            self._friction_positions.assign(positions)
            self._friction_displacement_epsilon = self.friction_epsilon * dt

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
            dim=self.surface_vertex_count,
            inputs=[
                self.triangle_bvh.id,
                self.thickness,
                self.particle_radii,
                self._use_geometry_radii,
                int(self.geometry_radius_topology_local_only),
                self.max_contacts,
                positions,
                self.surface_vertex_indices,
                self.triangle_orientation_signs,
                self.triangle_edge_indices,
                self.edge_triangle_indices,
                self.vertex_triangle_offsets,
                self.vertex_triangle_indices,
                self.vertex_neighbor_offsets,
                self.vertex_neighbors,
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
                self.particle_radii,
                self._use_geometry_radii,
                int(self.geometry_radius_topology_local_only),
                self.max_contacts,
                positions,
                self.rest_positions,
                self.particle_world,
                self.triangle_indices,
                self.edge_indices,
                self.edge_triangle_indices,
                self.triangle_orientation_signs,
            ],
            outputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.overflow_count,
            ],
            device=self.device,
        )
        self.edge_edge_contacts.prepare_hessian(positions)
        if self.enable_edge_face:
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
            self.edge_edge_contacts.accumulate_force(self.stiffness, positions, output)
            if self.enable_edge_face:
                self.edge_face_contacts.accumulate_force(self.untangle_stiffness, output)
            if self.friction > 0.0:
                anchor_positions, displacement_epsilon = self._friction_state()
                self.vertex_face_contacts.accumulate_friction_force(
                    self.stiffness,
                    self.friction,
                    displacement_epsilon,
                    positions,
                    anchor_positions,
                    output,
                )
                self.edge_edge_contacts.accumulate_friction_force(
                    self.stiffness,
                    self.friction,
                    displacement_epsilon,
                    positions,
                    anchor_positions,
                    output,
                )
            return

        static_diagonal, masses, inv_dt_squared = self._adaptive_system()
        self.vertex_face_contacts.accumulate_force_adaptive(
            self.stiffness_factors[0], static_diagonal, masses, inv_dt_squared, output
        )
        self.edge_edge_contacts.accumulate_force_adaptive(
            self.stiffness_factors[1], static_diagonal, masses, inv_dt_squared, positions, output
        )
        if self.enable_edge_face:
            self.edge_face_contacts.accumulate_force_adaptive(
                self.stiffness_factors[2], static_diagonal, masses, inv_dt_squared, output
            )
        if self.friction > 0.0:
            anchor_positions, displacement_epsilon = self._friction_state()
            self.vertex_face_contacts.accumulate_friction_force_adaptive(
                self.stiffness_factors[0],
                static_diagonal,
                masses,
                inv_dt_squared,
                self.friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                output,
            )
            self.edge_edge_contacts.accumulate_friction_force_adaptive(
                self.stiffness_factors[1],
                static_diagonal,
                masses,
                inv_dt_squared,
                self.friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                output,
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
            self.edge_edge_contacts.hessian_multiply(self.stiffness, positions, vector, output)
            if self.enable_edge_face:
                self.edge_face_contacts.hessian_multiply(self.untangle_stiffness, vector, output)
            if self.friction > 0.0:
                anchor_positions, displacement_epsilon = self._friction_state()
                self.vertex_face_contacts.friction_hessian_multiply(
                    self.stiffness,
                    self.friction,
                    displacement_epsilon,
                    positions,
                    anchor_positions,
                    vector,
                    output,
                )
                self.edge_edge_contacts.friction_hessian_multiply(
                    self.stiffness,
                    self.friction,
                    displacement_epsilon,
                    positions,
                    anchor_positions,
                    vector,
                    output,
                )
            return

        static_diagonal, masses, inv_dt_squared = self._adaptive_system()
        self.vertex_face_contacts.hessian_multiply_adaptive(
            self.stiffness_factors[0], static_diagonal, masses, inv_dt_squared, vector, output
        )
        self.edge_edge_contacts.hessian_multiply_adaptive(
            self.stiffness_factors[1], static_diagonal, masses, inv_dt_squared, positions, vector, output
        )
        if self.enable_edge_face:
            self.edge_face_contacts.hessian_multiply_adaptive(
                self.stiffness_factors[2], static_diagonal, masses, inv_dt_squared, vector, output
            )
        if self.friction > 0.0:
            anchor_positions, displacement_epsilon = self._friction_state()
            self.vertex_face_contacts.friction_hessian_multiply_adaptive(
                self.stiffness_factors[0],
                static_diagonal,
                masses,
                inv_dt_squared,
                self.friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                vector,
                output,
            )
            self.edge_edge_contacts.friction_hessian_multiply_adaptive(
                self.stiffness_factors[1],
                static_diagonal,
                masses,
                inv_dt_squared,
                self.friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                vector,
                output,
            )

    def accumulate_diagonal(self, positions: wp.array[wp.vec3], output: wp.array[wp.mat33]) -> None:
        """Add frozen-contact diagonal Hessian blocks to ``output``."""
        self._validate_positions(positions)
        if self.stiffness_factors is None:
            self.vertex_face_contacts.accumulate_diagonal(self.stiffness, output)
            self.edge_edge_contacts.accumulate_diagonal(self.stiffness, positions, output)
            if self.enable_edge_face:
                self.edge_face_contacts.accumulate_diagonal(self.untangle_stiffness, output)
            if self.friction > 0.0:
                anchor_positions, displacement_epsilon = self._friction_state()
                self.vertex_face_contacts.accumulate_friction_diagonal(
                    self.stiffness,
                    self.friction,
                    displacement_epsilon,
                    positions,
                    anchor_positions,
                    output,
                )
                self.edge_edge_contacts.accumulate_friction_diagonal(
                    self.stiffness,
                    self.friction,
                    displacement_epsilon,
                    positions,
                    anchor_positions,
                    output,
                )
            return

        static_diagonal, masses, inv_dt_squared = self._adaptive_system()
        self.vertex_face_contacts.accumulate_diagonal_adaptive(
            self.stiffness_factors[0], static_diagonal, masses, inv_dt_squared, output
        )
        self.edge_edge_contacts.accumulate_diagonal_adaptive(
            self.stiffness_factors[1], static_diagonal, masses, inv_dt_squared, positions, output
        )
        if self.enable_edge_face:
            self.edge_face_contacts.accumulate_diagonal_adaptive(
                self.stiffness_factors[2], static_diagonal, masses, inv_dt_squared, output
            )
        if self.friction > 0.0:
            anchor_positions, displacement_epsilon = self._friction_state()
            self.vertex_face_contacts.accumulate_friction_diagonal_adaptive(
                self.stiffness_factors[0],
                static_diagonal,
                masses,
                inv_dt_squared,
                self.friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                output,
            )
            self.edge_edge_contacts.accumulate_friction_diagonal_adaptive(
                self.stiffness_factors[1],
                static_diagonal,
                masses,
                inv_dt_squared,
                self.friction,
                displacement_epsilon,
                positions,
                anchor_positions,
                output,
            )

    def _adaptive_system(self) -> tuple[wp.array[wp.mat33], wp.array[float], float]:
        if self._static_diagonal is None or self._masses is None or self._inv_dt_squared <= 0.0:
            raise RuntimeError("bind_static_system() and begin_step() are required before adaptive contact evaluation")
        return self._static_diagonal, self._masses, self._inv_dt_squared

    def _friction_state(self) -> tuple[wp.array[wp.vec3], float]:
        if self._friction_positions is None or self._friction_displacement_epsilon <= 0.0:
            raise RuntimeError("begin_step() is required before friction evaluation")
        return self._friction_positions, self._friction_displacement_epsilon

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
