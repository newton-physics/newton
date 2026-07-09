# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse-volume kernels for particle surface extraction."""

import math

import warp as wp

from .flags import ParticleFlags

wp.set_module_options({"enable_backward": False})

_TILE_SIZE = wp.constant(8)
_GRID_NODE_COUNT = wp.constant(0)
_GRID_CELL_COUNT = wp.constant(1)
_GRID_ACTIVE_COUNT = wp.constant(2)
_GRID_OVERFLOW = wp.constant(3)
_GRID_DIM_X = wp.constant(4)
_GRID_DIM_Y = wp.constant(5)
_GRID_DIM_Z = wp.constant(6)
_GRID_COUNT_STRIDE = wp.constant(7)
_MESH_VERTEX_COUNT = wp.constant(0)
_MESH_INDEX_COUNT = wp.constant(1)
_MESH_COUNT_STRIDE = wp.constant(3)
_SUPPORT_VOXEL_COUNT = 27


@wp.func
def _grid_count_index(world: int, entry: int) -> int:
    return world * _GRID_COUNT_STRIDE + entry


@wp.func
def _mesh_count_index(world: int, entry: int) -> int:
    return world * _MESH_COUNT_STRIDE + entry


@wp.func
def _particle_world(worlds: wp.array[wp.int32], use_worlds: int, world_count: int, particle: int) -> int:
    world = int(0)
    if use_worlds != 0:
        world = worlds[particle]
    if world < 0 or world >= world_count:
        return -1
    return world


@wp.func
def _is_active(flags: wp.array[wp.int32], use_flags: int, particle: int) -> bool:
    if use_flags != 0:
        return (flags[particle] & ParticleFlags.ACTIVE) != wp.int32(0)
    return True


@wp.func
def _is_finite(position: wp.vec3) -> bool:
    return wp.isfinite(position[0]) and wp.isfinite(position[1]) and wp.isfinite(position[2])


@wp.func
def _floor_tile_coordinate(coordinate: int) -> int:
    if coordinate < 0:
        return ((coordinate - (_TILE_SIZE - 1)) // _TILE_SIZE) * _TILE_SIZE
    return (coordinate // _TILE_SIZE) * _TILE_SIZE


@wp.func
def _tile_origin(point: wp.vec3i) -> wp.vec3i:
    return wp.vec3i(
        _floor_tile_coordinate(point[0]),
        _floor_tile_coordinate(point[1]),
        _floor_tile_coordinate(point[2]),
    )


@wp.func
def _field_value(
    node_grid: wp.uint64,
    field: wp.array[float],
    coordinate: wp.vec3i,
    outside_value: float,
) -> float:
    node = wp.volume_lookup_index(node_grid, coordinate[0], coordinate[1], coordinate[2])
    if node < 0:
        return outside_value
    return field[node]


@wp.func
def _coordinate_world(
    coordinate: wp.vec3i,
    packed_lower: wp.array[wp.vec3i],
    packed_upper: wp.array[wp.vec3i],
    world_count: int,
    include_upper: int,
) -> int:
    for world in range(world_count):
        lower = packed_lower[world]
        upper = packed_upper[world] + wp.vec3i(include_upper)
        if (
            coordinate[0] >= lower[0]
            and coordinate[1] >= lower[1]
            and coordinate[2] >= lower[2]
            and coordinate[0] < upper[0]
            and coordinate[1] < upper[1]
            and coordinate[2] < upper[2]
        ):
            return world
    return -1


@wp.func
def _cubic_bspline(q: float) -> float:
    if q < 1.0:
        return 1.0 - 1.5 * q * q + 0.75 * q * q * q
    if q < 2.0:
        t = 2.0 - q
        return 0.25 * t * t * t
    return 0.0


@wp.kernel
def finalize_sparse_grids(
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    grid_counts: wp.array[wp.int32],
    grid_origin: wp.array[wp.vec3],
    grid_dims: wp.array[wp.vec3i],
    env_offsets: wp.array[wp.vec3i],
    packed_lower: wp.array[wp.vec3i],
    packed_upper: wp.array[wp.vec3i],
    active_particle_count: wp.array[wp.int32],
    world_count: int,
    voxel_size: float,
    padding: int,
):
    if wp.tid() != 0:
        return
    inv_voxel_size = 1.0 / voxel_size
    packed_x = int(0)
    total_active = int(0)
    for world in range(world_count):
        counts_base = _grid_count_index(world, 0)
        total_active += grid_counts[counts_base + _GRID_ACTIVE_COUNT]
        if grid_counts[counts_base + _GRID_ACTIVE_COUNT] == 0 or lower[world][0] > upper[world][0]:
            grid_origin[world] = wp.vec3(0.0)
            grid_dims[world] = wp.vec3i(0)
            env_offsets[world] = wp.vec3i(0)
            packed_lower[world] = wp.vec3i(0)
            packed_upper[world] = wp.vec3i(0)
            continue

        grid_min = wp.vec3i(
            int(wp.floor(lower[world][0] * inv_voxel_size)) - padding,
            int(wp.floor(lower[world][1] * inv_voxel_size)) - padding,
            int(wp.floor(lower[world][2] * inv_voxel_size)) - padding,
        )
        grid_max = wp.vec3i(
            int(wp.ceil(upper[world][0] * inv_voxel_size)) + padding,
            int(wp.ceil(upper[world][1] * inv_voxel_size)) + padding,
            int(wp.ceil(upper[world][2] * inv_voxel_size)) + padding,
        )
        dims = grid_max - grid_min + wp.vec3i(1)
        tile_min = _tile_origin(grid_min)
        tile_upper = _tile_origin(grid_max) + wp.vec3i(_TILE_SIZE)
        extent = tile_upper - tile_min
        offset = wp.vec3i(0)
        packed_min = tile_min
        packed_max = tile_upper
        if world_count > 1:
            offset = wp.vec3i(packed_x, 0, 0) - tile_min
            packed_min = wp.vec3i(packed_x, 0, 0)
            packed_max = wp.vec3i(packed_x + extent[0], extent[1], extent[2])
        grid_origin[world] = voxel_size * wp.vec3(float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
        grid_dims[world] = dims
        env_offsets[world] = offset
        packed_lower[world] = packed_min
        packed_upper[world] = packed_max
        grid_counts[counts_base + _GRID_DIM_X] = dims[0]
        grid_counts[counts_base + _GRID_DIM_Y] = dims[1]
        grid_counts[counts_base + _GRID_DIM_Z] = dims[2]
        if world_count > 1:
            packed_x += extent[0] + _TILE_SIZE
    active_particle_count[0] = total_active


@wp.kernel
def emit_particle_support_voxels(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    particle_world: wp.array[wp.int32],
    use_worlds: int,
    world_count: int,
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    particle_sdf_radius_scale: float,
    particle_sdf_band: float,
    particle_sdf: int,
    anisotropic_sdf: int,
    stencil_voxels: int,
    env_offsets: wp.array[wp.vec3i],
    inv_voxel_size: float,
    points: wp.array[wp.vec3i],
    point_mask: wp.array[wp.int32],
):
    slot = wp.tid()
    particle = slot // _SUPPORT_VOXEL_COUNT
    sample = slot - _SUPPORT_VOXEL_COUNT * particle
    world = _particle_world(particle_world, use_worlds, world_count, particle)
    position = positions[particle]
    radius = radii[particle]
    valid = (
        world >= 0
        and _is_active(flags, use_flags, particle)
        and _is_finite(position)
        and radius > 0.0
        and wp.isfinite(radius)
    )
    if valid:
        reach = density_reach[particle]
        if particle_sdf != 0 and anisotropic_sdf == 0:
            reach = wp.vec3(radius * particle_sdf_radius_scale * particle_sdf_band)
        elif particle_sdf != 0:
            scaled_radius = wp.max(radius * particle_sdf_radius_scale, 1.0e-8)
            det_root = wp.pow(wp.max(det_G[particle], 1.0e-24), 1.0 / 3.0)
            reach *= 0.5 * particle_sdf_band * det_root * scaled_radius
        reach += wp.vec3(float(stencil_voxels) / inv_voxel_size)
        dx = sample // 9 - 1
        remainder = sample - (dx + 1) * 9
        dy = remainder // 3 - 1
        dz = remainder - (dy + 1) * 3 - 1
        sample_position = position + wp.cw_mul(reach, wp.vec3(float(dx), float(dy), float(dz)))
        coordinate = wp.vec3i(
            int(wp.floor(sample_position[0] * inv_voxel_size)),
            int(wp.floor(sample_position[1] * inv_voxel_size)),
            int(wp.floor(sample_position[2] * inv_voxel_size)),
        )
        points[slot] = coordinate + env_offsets[world]
    else:
        points[slot] = wp.vec3i(0)
    point_mask[slot] = wp.where(valid, wp.int32(1), wp.int32(0))


@wp.kernel
def classify_sparse_topology_single_world(
    grid: wp.uint64,
    voxel_world: wp.array[wp.int32],
    grid_counts: wp.array[wp.int32],
    stride: int,
):
    voxel = wp.tid()
    voxel_count = wp.volume_voxel_count(grid)
    while voxel < voxel_count:
        voxel_world[voxel] = wp.int32(0)
        voxel += stride
    if wp.tid() == 0:
        grid_counts[_grid_count_index(0, _GRID_CELL_COUNT)] = voxel_count
        grid_counts[_grid_count_index(0, _GRID_NODE_COUNT)] = voxel_count


@wp.kernel
def classify_sparse_topology(
    grid: wp.uint64,
    voxel_ijk: wp.array[wp.vec3i],
    packed_lower: wp.array[wp.vec3i],
    packed_upper: wp.array[wp.vec3i],
    voxel_world: wp.array[wp.int32],
    grid_counts: wp.array[wp.int32],
    world_count: int,
    stride: int,
):
    voxel = wp.tid()
    voxel_count = wp.volume_voxel_count(grid)
    while voxel < voxel_count:
        world = _coordinate_world(voxel_ijk[voxel], packed_lower, packed_upper, world_count, 0)
        voxel_world[voxel] = world
        if world >= 0:
            wp.atomic_add(grid_counts, _grid_count_index(world, _GRID_CELL_COUNT), wp.int32(1))
            wp.atomic_add(grid_counts, _grid_count_index(world, _GRID_NODE_COUNT), wp.int32(1))
        voxel += stride


@wp.kernel
def finalize_sparse_topology(
    statuses: wp.array[wp.uint32],
    grid_counts: wp.array[wp.int32],
    node_world_start: wp.array[wp.int32],
    cell_world_start: wp.array[wp.int32],
    world_count: int,
):
    if wp.tid() != 0:
        return
    overflow = int(0)
    for status in range(statuses.shape[0]):
        if statuses[status] != wp.uint32(0):
            overflow = 1
    node_start = int(0)
    cell_start = int(0)
    node_world_start[0] = wp.int32(0)
    cell_world_start[0] = wp.int32(0)
    for world in range(world_count):
        node_start += grid_counts[_grid_count_index(world, _GRID_NODE_COUNT)]
        cell_start += grid_counts[_grid_count_index(world, _GRID_CELL_COUNT)]
        node_world_start[world + 1] = node_start
        cell_world_start[world + 1] = cell_start
        grid_counts[_grid_count_index(world, _GRID_OVERFLOW)] = overflow


@wp.kernel
def fill_field(node_grid: wp.uint64, field: wp.array[float], value: float, stride: int):
    node = wp.tid()
    node_count = wp.volume_voxel_count(node_grid)
    while node < node_count:
        field[node] = value
        node += stride


@wp.kernel
def copy_field_to_nanogrid(
    node_grid: wp.uint64,
    field: wp.array[float],
    nanogrid_node_ijk: wp.array[wp.vec3i],
    nanogrid_field: wp.array[float],
    outside_value: float,
):
    node = wp.tid()
    coordinate = nanogrid_node_ijk[node]
    source = wp.volume_lookup_index(node_grid, coordinate[0], coordinate[1], coordinate[2])
    nanogrid_field[node] = wp.where(source >= 0, field[source], outside_value)


@wp.kernel
def evaluate_density(
    node_grid: wp.uint64,
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    particle_world: wp.array[wp.int32],
    use_worlds: int,
    world_count: int,
    G_matrices: wp.array[wp.mat33],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    env_offsets: wp.array[wp.vec3i],
    inv_voxel_size: float,
    field: wp.array[float],
):
    particle = wp.tid()
    world = _particle_world(particle_world, use_worlds, world_count, particle)
    if world < 0 or not _is_active(flags, use_flags, particle):
        return
    position = smoothed[particle]
    radius = radii[particle]
    if not _is_finite(position) or radius <= 0.0 or not wp.isfinite(radius):
        return
    reach = density_reach[particle]
    scaled_lower = (position - reach) * inv_voxel_size
    scaled_upper = (position + reach) * inv_voxel_size
    lower = wp.vec3i(int(wp.ceil(scaled_lower[0])), int(wp.ceil(scaled_lower[1])), int(wp.ceil(scaled_lower[2])))
    upper = wp.vec3i(int(wp.floor(scaled_upper[0])), int(wp.floor(scaled_upper[1])), int(wp.floor(scaled_upper[2])))
    G = G_matrices[particle]
    voxel_size = 1.0 / inv_voxel_size
    delta_i = voxel_size * wp.vec3(lower) - position
    transformed_i = G * delta_i
    step_x = voxel_size * wp.vec3(G[0, 0], G[1, 0], G[2, 0])
    step_y = voxel_size * wp.vec3(G[0, 1], G[1, 1], G[2, 1])
    step_z = voxel_size * wp.vec3(G[0, 2], G[1, 2], G[2, 2])
    weight = 8.0 * radius * radius * radius * wp.static(1.0 / math.pi) * det_G[particle]
    offset = env_offsets[world]
    for i in range(lower[0], upper[0] + 1):
        transformed_j = transformed_i
        for j in range(lower[1], upper[1] + 1):
            transformed = transformed_j
            for k in range(lower[2], upper[2] + 1):
                q_sq = wp.dot(transformed, transformed)
                if q_sq < 4.0:
                    coordinate = wp.vec3i(i, j, k) + offset
                    node = wp.volume_lookup_index(node_grid, coordinate[0], coordinate[1], coordinate[2])
                    if node >= 0:
                        wp.atomic_add(field, node, weight * _cubic_bspline(wp.sqrt(q_sq)))
                transformed += step_z
            transformed_j += step_y
        transformed_i += step_x


@wp.kernel
def evaluate_particle_sdf_isotropic(
    node_grid: wp.uint64,
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    particle_world: wp.array[wp.int32],
    use_worlds: int,
    world_count: int,
    radius_scale: float,
    band: float,
    env_offsets: wp.array[wp.vec3i],
    inv_voxel_size: float,
    field: wp.array[float],
):
    particle = wp.tid()
    world = _particle_world(particle_world, use_worlds, world_count, particle)
    if world < 0 or not _is_active(flags, use_flags, particle):
        return
    position = smoothed[particle]
    radius = radii[particle] * radius_scale
    if not _is_finite(position) or radius <= 0.0 or not wp.isfinite(radius):
        return
    reach = band * radius
    scaled_lower = (position - wp.vec3(reach)) * inv_voxel_size
    scaled_upper = (position + wp.vec3(reach)) * inv_voxel_size
    lower = wp.vec3i(int(wp.ceil(scaled_lower[0])), int(wp.ceil(scaled_lower[1])), int(wp.ceil(scaled_lower[2])))
    upper = wp.vec3i(int(wp.floor(scaled_upper[0])), int(wp.floor(scaled_upper[1])), int(wp.floor(scaled_upper[2])))
    voxel_size = 1.0 / inv_voxel_size
    offset = env_offsets[world]
    for i in range(lower[0], upper[0] + 1):
        x = voxel_size * float(i) - position[0]
        for j in range(lower[1], upper[1] + 1):
            y = voxel_size * float(j) - position[1]
            for k in range(lower[2], upper[2] + 1):
                z = voxel_size * float(k) - position[2]
                distance_sq = x * x + y * y + z * z
                if distance_sq <= reach * reach:
                    coordinate = wp.vec3i(i, j, k) + offset
                    node = wp.volume_lookup_index(node_grid, coordinate[0], coordinate[1], coordinate[2])
                    if node >= 0:
                        wp.atomic_min(field, node, wp.sqrt(distance_sq) - radius)


@wp.kernel
def evaluate_particle_sdf_anisotropic(
    node_grid: wp.uint64,
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    particle_world: wp.array[wp.int32],
    use_worlds: int,
    world_count: int,
    G_matrices: wp.array[wp.mat33],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    radius_scale: float,
    band: float,
    env_offsets: wp.array[wp.vec3i],
    inv_voxel_size: float,
    field: wp.array[float],
):
    particle = wp.tid()
    world = _particle_world(particle_world, use_worlds, world_count, particle)
    if world < 0 or not _is_active(flags, use_flags, particle):
        return
    position = smoothed[particle]
    radius = radii[particle] * radius_scale
    if not _is_finite(position) or radius <= 0.0 or not wp.isfinite(radius):
        return
    G = G_matrices[particle]
    det_root = wp.pow(wp.max(det_G[particle], 1.0e-24), 1.0 / 3.0)
    radius_normalization = det_root * radius
    H = G * (1.0 / radius_normalization)
    reach = density_reach[particle] * (0.5 * band * radius_normalization)
    scaled_lower = (position - reach) * inv_voxel_size
    scaled_upper = (position + reach) * inv_voxel_size
    lower = wp.vec3i(int(wp.ceil(scaled_lower[0])), int(wp.ceil(scaled_lower[1])), int(wp.ceil(scaled_lower[2])))
    upper = wp.vec3i(int(wp.floor(scaled_upper[0])), int(wp.floor(scaled_upper[1])), int(wp.floor(scaled_upper[2])))
    voxel_size = 1.0 / inv_voxel_size
    transformed_i = H * (voxel_size * wp.vec3(lower) - position)
    step_x = voxel_size * wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    step_y = voxel_size * wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    step_z = voxel_size * wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    H_transpose = wp.transpose(H)
    band_sq = band * band
    offset = env_offsets[world]
    for i in range(lower[0], upper[0] + 1):
        transformed_j = transformed_i
        for j in range(lower[1], upper[1] + 1):
            transformed = transformed_j
            for k in range(lower[2], upper[2] + 1):
                q_sq = wp.dot(transformed, transformed)
                if q_sq <= band_sq:
                    sdf = -radius
                    if q_sq > 1.0e-16:
                        q = wp.sqrt(q_sq)
                        gradient_norm_times_q = wp.length(H_transpose * transformed)
                        sdf = (q - 1.0) * q / wp.max(gradient_norm_times_q, 1.0e-8 * q)
                    coordinate = wp.vec3i(i, j, k) + offset
                    node = wp.volume_lookup_index(node_grid, coordinate[0], coordinate[1], coordinate[2])
                    if node >= 0:
                        wp.atomic_min(field, node, sdf)
                transformed += step_z
            transformed_j += step_y
        transformed_i += step_x


@wp.kernel
def density_to_sdf(node_grid: wp.uint64, field: wp.array[float], threshold: float, stride: int):
    node = wp.tid()
    node_count = wp.volume_voxel_count(node_grid)
    while node < node_count:
        field[node] = threshold - field[node]
        node += stride


@wp.kernel
def blur_field_axis(
    node_grid: wp.uint64,
    node_ijk: wp.array[wp.vec3i],
    source: wp.array[float],
    destination: wp.array[float],
    weights: wp.array[float],
    half_width: int,
    axis: int,
    outside_value: float,
    stride: int,
):
    node = wp.tid()
    node_count = wp.volume_voxel_count(node_grid)
    while node < node_count:
        coordinate = node_ijk[node]
        value = source[node] * weights[0]
        for offset in range(1, half_width + 1):
            lower = coordinate
            upper = coordinate
            lower[axis] -= offset
            upper[axis] += offset
            value += weights[offset] * (
                _field_value(node_grid, source, lower, outside_value)
                + _field_value(node_grid, source, upper, outside_value)
            )
        destination[node] = value
        node += stride


@wp.kernel
def redistance_step(
    node_grid: wp.uint64,
    node_ijk: wp.array[wp.vec3i],
    sdf: wp.array[float],
    sdf_out: wp.array[float],
    outside_value: float,
    inv_voxel_size: float,
    stride: int,
):
    node = wp.tid()
    node_count = wp.volume_voxel_count(node_grid)
    while node < node_count:
        coordinate = node_ijk[node]
        value = sdf[node]
        sign = wp.sign(value)
        dx_m = _field_value(node_grid, sdf, coordinate - wp.vec3i(1, 0, 0), outside_value)
        dx_p = _field_value(node_grid, sdf, coordinate + wp.vec3i(1, 0, 0), outside_value)
        dy_m = _field_value(node_grid, sdf, coordinate - wp.vec3i(0, 1, 0), outside_value)
        dy_p = _field_value(node_grid, sdf, coordinate + wp.vec3i(0, 1, 0), outside_value)
        dz_m = _field_value(node_grid, sdf, coordinate - wp.vec3i(0, 0, 1), outside_value)
        dz_p = _field_value(node_grid, sdf, coordinate + wp.vec3i(0, 0, 1), outside_value)
        ax = wp.max(wp.max(sign * (value - dx_m), 0.0), wp.max(-sign * (dx_p - value), 0.0)) * inv_voxel_size
        ay = wp.max(wp.max(sign * (value - dy_m), 0.0), wp.max(-sign * (dy_p - value), 0.0)) * inv_voxel_size
        az = wp.max(wp.max(sign * (value - dz_m), 0.0), wp.max(-sign * (dz_p - value), 0.0)) * inv_voxel_size
        gradient = wp.sqrt(ax * ax + ay * ay + az * az)
        voxel_size_sq = 1.0 / (inv_voxel_size * inv_voxel_size)
        smooth_sign = value / wp.sqrt(value * value + gradient * gradient * voxel_size_sq + 1.0e-20)
        sdf_out[node] = value - 0.5 / inv_voxel_size * smooth_sign * (gradient - 1.0)
        node += stride


@wp.kernel
def reset_edge_indices(node_grid: wp.uint64, edge_indices: wp.array[wp.int32], stride: int):
    edge = wp.tid()
    edge_count = 3 * wp.volume_voxel_count(node_grid)
    while edge < edge_count:
        edge_indices[edge] = wp.int32(-1)
        edge += stride


@wp.kernel
def extract_mesh_vertices(
    node_grid: wp.uint64,
    node_ijk: wp.array[wp.vec3i],
    node_world: wp.array[wp.int32],
    env_offsets: wp.array[wp.vec3i],
    field: wp.array[float],
    threshold: float,
    voxel_size: float,
    world_count: int,
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.int32],
    mesh_counts: wp.array[wp.int32],
    vertex_world_start: wp.array[wp.int32],
    write_output: int,
    stride: int,
):
    edge_slot = wp.tid()
    edge_count = 3 * wp.volume_voxel_count(node_grid)
    while edge_slot < edge_count:
        node = edge_slot // 3
        axis = edge_slot - 3 * node
        world = node_world[node]
        coordinate = node_ijk[node]
        neighbor_coordinate = coordinate
        neighbor_coordinate[axis] += 1
        neighbor = wp.volume_lookup_index(
            node_grid, neighbor_coordinate[0], neighbor_coordinate[1], neighbor_coordinate[2]
        )
        if world >= 0 and neighbor >= 0 and node_world[neighbor] == world:
            value = field[node]
            neighbor_value = field[neighbor]
            if (value >= threshold and neighbor_value < threshold) or (
                value < threshold and neighbor_value >= threshold
            ):
                count_index = wp.where(
                    world_count == 1, _MESH_VERTEX_COUNT, _mesh_count_index(world, _MESH_VERTEX_COUNT)
                )
                local_output = wp.atomic_add(mesh_counts, count_index, wp.int32(1))
                if write_output != 0:
                    output = local_output + wp.where(world_count == 1, 0, vertex_world_start[world])
                    interpolation = wp.clamp((threshold - value) / (neighbor_value - value), 0.0, 1.0)
                    local = coordinate - env_offsets[world]
                    position = voxel_size * wp.vec3(float(local[0]), float(local[1]), float(local[2]))
                    position[axis] += voxel_size * interpolation
                    vertices[output] = position
                    edge_indices[edge_slot] = output
        edge_slot += stride


@wp.kernel
def extract_mesh_indices(
    cell_grid: wp.uint64,
    node_grid: wp.uint64,
    cell_ijk: wp.array[wp.vec3i],
    cell_world: wp.array[wp.int32],
    field: wp.array[float],
    threshold: float,
    world_count: int,
    case_ranges: wp.array[wp.int32],
    local_edges: wp.array[wp.int32],
    corner_offsets: wp.array[wp.vec3i],
    edge_offsets: wp.array[wp.vec3i],
    edge_axes: wp.array[wp.int32],
    edge_indices: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    mesh_counts: wp.array[wp.int32],
    index_world_start: wp.array[wp.int32],
    write_output: int,
    stride: int,
):
    cell = wp.tid()
    cell_count = wp.volume_voxel_count(cell_grid)
    while cell < cell_count:
        coordinate = cell_ijk[cell]
        world = cell_world[cell]
        case = int(0)
        valid = world >= 0
        for corner in range(8):
            corner_coordinate = coordinate + corner_offsets[corner]
            corner_node = wp.volume_lookup_index(
                node_grid, corner_coordinate[0], corner_coordinate[1], corner_coordinate[2]
            )
            if corner_node < 0:
                valid = False
            elif field[corner_node] >= threshold:
                case |= 1 << corner
        if valid:
            local_begin = case_ranges[case]
            local_end = case_ranges[case + 1]
            local_count = local_end - local_begin
            if local_count > 0:
                count_index = wp.where(world_count == 1, _MESH_INDEX_COUNT, _mesh_count_index(world, _MESH_INDEX_COUNT))
                local_output = wp.atomic_add(mesh_counts, count_index, local_count)
                if write_output != 0:
                    output = local_output + wp.where(world_count == 1, 0, index_world_start[world])
                    for local_index in range(local_begin, local_end):
                        edge = local_edges[local_index]
                        edge_coordinate = coordinate + edge_offsets[edge]
                        edge_node = wp.volume_lookup_index(
                            node_grid, edge_coordinate[0], edge_coordinate[1], edge_coordinate[2]
                        )
                        indices[output + local_index - local_begin] = edge_indices[3 * edge_node + edge_axes[edge]]
        cell += stride
