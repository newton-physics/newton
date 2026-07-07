# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp functions and kernels for particle surface extraction."""

import math

import warp as wp

from .flags import ParticleFlags

wp.set_module_options({"enable_backward": False})

_GRID_NODE_COUNT = wp.constant(0)
_GRID_CELL_COUNT = wp.constant(1)
_GRID_ACTIVE_COUNT = wp.constant(2)
_GRID_OVERFLOW = wp.constant(3)
_GRID_DIM_X = wp.constant(4)
_GRID_DIM_Y = wp.constant(5)
_GRID_DIM_Z = wp.constant(6)

_MESH_VERTEX_COUNT = wp.constant(0)
_MESH_INDEX_COUNT = wp.constant(1)

_MAX_CAPACITY_LAUNCH_THREADS = 4_194_304


@wp.func
def _is_active(flags: wp.array[wp.int32], use_flags: int, i: int) -> bool:
    if use_flags != 0:
        return (flags[i] & ParticleFlags.ACTIVE) != wp.int32(0)
    return True


@wp.func
def _node_index(i: int, j: int, k: int, dims: wp.vec3i) -> int:
    return (i * dims[1] + j) * dims[2] + k


@wp.func
def _unravel_node(index: int, dims: wp.vec3i) -> wp.vec3i:
    yz = dims[1] * dims[2]
    i = index // yz
    remainder = index - i * yz
    j = remainder // dims[2]
    k = remainder - j * dims[2]
    return wp.vec3i(i, j, k)


@wp.func
def _unravel_cell(index: int, cell_dims: wp.vec3i) -> wp.vec3i:
    yz = cell_dims[1] * cell_dims[2]
    i = index // yz
    remainder = index - i * yz
    j = remainder // cell_dims[2]
    k = remainder - j * cell_dims[2]
    return wp.vec3i(i, j, k)


@wp.kernel
def reset_bounds_and_counts(
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    grid_counts: wp.array[wp.int32],
    mesh_counts: wp.array[wp.int32],
):
    lower[0] = wp.vec3(1.0e30)
    upper[0] = wp.vec3(-1.0e30)
    for i in range(7):
        grid_counts[i] = wp.int32(0)
    mesh_counts[_MESH_VERTEX_COUNT] = wp.int32(0)
    mesh_counts[_MESH_INDEX_COUNT] = wp.int32(0)
    mesh_counts[2] = wp.int32(0)


@wp.kernel
def reset_bounds(lower: wp.array[wp.vec3], upper: wp.array[wp.vec3]):
    lower[0] = wp.vec3(1.0e30)
    upper[0] = wp.vec3(-1.0e30)


@wp.kernel
def reset_mesh_counts(mesh_counts: wp.array[wp.int32], grid_counts: wp.array[wp.int32]):
    mesh_counts[_MESH_VERTEX_COUNT] = wp.int32(0)
    mesh_counts[_MESH_INDEX_COUNT] = wp.int32(0)
    mesh_counts[2] = grid_counts[_GRID_OVERFLOW]


@wp.kernel
def clear_grid_overflow(grid_counts: wp.array[wp.int32]):
    grid_counts[_GRID_OVERFLOW] = wp.int32(0)


@wp.kernel
def compute_particle_bounds(
    positions: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    use_flags: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
):
    i = wp.tid()
    if not _is_active(flags, use_flags, i):
        return

    p = positions[i]
    if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        return

    wp.atomic_min(lower, 0, p)
    wp.atomic_max(upper, 0, p)
    wp.atomic_add(active_count, _GRID_ACTIVE_COUNT, wp.int32(1))


@wp.kernel
def finalize_particle_bounds(
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    inactive_position: wp.array[wp.vec3],
    sentinel_distance: float,
):
    if lower[0][0] > upper[0][0]:
        lower[0] = wp.vec3(0.0)
        upper[0] = wp.vec3(0.0)
    inactive_position[0] = lower[0] - wp.vec3(sentinel_distance)


@wp.kernel
def copy_active_or_sentinel_positions(
    positions: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    use_flags: int,
    inactive_position: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    if _is_active(flags, use_flags, i):
        out[i] = positions[i]
    else:
        out[i] = inactive_position[0]


@wp.kernel
def smooth_positions_flagged(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    inactive_position: wp.array[wp.vec3],
    search_radius: float,
    smooth_lambda: float,
    smoothed: wp.array[wp.vec3],
):
    i = wp.tid()
    if not _is_active(flags, 1, i):
        smoothed[i] = inactive_position[0]
        return

    xi = positions[i]
    offset_sum = wp.vec3(0.0)
    weight_sum = float(0.0)
    radius_sq = search_radius * search_radius
    inv_radius_sq = 1.0 / radius_sq

    query = wp.hash_grid_query(grid, xi, search_radius)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if _is_active(flags, 1, index):
            offset = positions[index] - xi
            dist_sq = wp.dot(offset, offset)
            if dist_sq < radius_sq:
                q_sq = dist_sq * inv_radius_sq
                weight = 1.0 - q_sq * wp.sqrt(q_sq)
                offset_sum += weight * offset
                weight_sum += weight

    if weight_sum > 0.0:
        smoothed[i] = xi + (smooth_lambda / weight_sum) * offset_sum
    else:
        smoothed[i] = xi


@wp.func
def _kernel_reach(
    i: int,
    radii: wp.array[float],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    particle_sdf_radius_scale: float,
    particle_sdf_band: float,
    particle_sdf: int,
    anisotropic: int,
) -> wp.vec3:
    reach = density_reach[i]
    if particle_sdf != 0 and anisotropic == 0:
        radius = radii[i] * particle_sdf_radius_scale * particle_sdf_band
        reach = wp.vec3(radius)
    elif particle_sdf != 0:
        radius = wp.max(radii[i] * particle_sdf_radius_scale, 1.0e-8)
        det_root = wp.pow(wp.max(det_G[i], 1.0e-24), 1.0 / 3.0)
        reach = density_reach[i] * (0.5 * particle_sdf_band * det_root * radius)
    return reach


@wp.kernel
def compute_kernel_bounds(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    particle_sdf_radius_scale: float,
    particle_sdf_band: float,
    particle_sdf: int,
    anisotropic: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    i = wp.tid()
    if not _is_active(flags, use_flags, i):
        return

    radius = radii[i]
    if radius <= 0.0 or not wp.isfinite(radius):
        return

    position = positions[i]
    reach = _kernel_reach(
        i,
        radii,
        det_G,
        density_reach,
        particle_sdf_radius_scale,
        particle_sdf_band,
        particle_sdf,
        anisotropic,
    )
    wp.atomic_min(lower, 0, position - reach)
    wp.atomic_max(upper, 0, position + reach)


@wp.kernel
def finalize_grid(
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    grid_origin: wp.array[wp.vec3],
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    voxel_size: float,
    padding: int,
    max_grid_cells: int,
    max_grid_nodes: int,
):
    if active_count[_GRID_ACTIVE_COUNT] == 0 or lower[0][0] > upper[0][0]:
        grid_origin[0] = wp.vec3(0.0)
        grid_dims[0] = wp.vec3i(0)
        return

    inv_voxel_size = 1.0 / voxel_size
    pad = float(padding) * voxel_size
    padded_lower = (lower[0] - wp.vec3(pad)) * inv_voxel_size
    padded_upper = (upper[0] + wp.vec3(pad)) * inv_voxel_size
    grid_min = voxel_size * wp.vec3(
        wp.floor(padded_lower[0]),
        wp.floor(padded_lower[1]),
        wp.floor(padded_lower[2]),
    )
    grid_max = voxel_size * wp.vec3(
        wp.ceil(padded_upper[0]),
        wp.ceil(padded_upper[1]),
        wp.ceil(padded_upper[2]),
    )
    cell_dims = wp.vec3i(
        wp.max(int(wp.round((grid_max[0] - grid_min[0]) * inv_voxel_size)), 1),
        wp.max(int(wp.round((grid_max[1] - grid_min[1]) * inv_voxel_size)), 1),
        wp.max(int(wp.round((grid_max[2] - grid_min[2]) * inv_voxel_size)), 1),
    )
    dims = cell_dims + wp.vec3i(1)
    cell_count = cell_dims[0] * cell_dims[1] * cell_dims[2]
    node_count = dims[0] * dims[1] * dims[2]

    grid_origin[0] = grid_min
    grid_dims[0] = dims
    grid_counts[_GRID_CELL_COUNT] = cell_count
    grid_counts[_GRID_NODE_COUNT] = node_count
    grid_counts[_GRID_DIM_X] = dims[0]
    grid_counts[_GRID_DIM_Y] = dims[1]
    grid_counts[_GRID_DIM_Z] = dims[2]
    if cell_count > max_grid_cells or node_count > max_grid_nodes:
        grid_counts[_GRID_OVERFLOW] = wp.int32(1)


@wp.kernel
def fill_field(
    field: wp.array[float],
    grid_counts: wp.array[wp.int32],
    value: float,
    stride: int,
):
    index = wp.tid()
    node_count = grid_counts[_GRID_NODE_COUNT]
    if grid_counts[_GRID_OVERFLOW] != 0:
        node_count = 0
    while index < node_count:
        field[index] = value
        index += stride


@wp.func
def _cubic_bspline(q: float) -> float:
    if q < 1.0:
        return 1.0 - 1.5 * q * q + 0.75 * q * q * q
    if q < 2.0:
        t = 2.0 - q
        return 0.25 * t * t * t
    return 0.0


@wp.kernel
def evaluate_density(
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    G_matrices: wp.array[wp.mat33],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    grid_origin: wp.array[wp.vec3],
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    inv_voxel_size: float,
    field: wp.array[float],
):
    particle = wp.tid()
    if not _is_active(flags, use_flags, particle) or grid_counts[_GRID_OVERFLOW] != 0:
        return

    position = smoothed[particle]
    radius = radii[particle]
    if radius <= 0.0 or not wp.isfinite(radius):
        return

    volume = 8.0 * radius * radius * radius
    weight = volume * wp.static(1.0 / math.pi) * det_G[particle]
    G = G_matrices[particle]
    reach = density_reach[particle]
    origin = grid_origin[0]
    dims = grid_dims[0]

    lower = wp.vec3i(
        wp.max(int(wp.ceil((position[0] - reach[0] - origin[0]) * inv_voxel_size)), 0),
        wp.max(int(wp.ceil((position[1] - reach[1] - origin[1]) * inv_voxel_size)), 0),
        wp.max(int(wp.ceil((position[2] - reach[2] - origin[2]) * inv_voxel_size)), 0),
    )
    upper = wp.vec3i(
        wp.min(int(wp.floor((position[0] + reach[0] - origin[0]) * inv_voxel_size)), dims[0] - 1),
        wp.min(int(wp.floor((position[1] + reach[1] - origin[1]) * inv_voxel_size)), dims[1] - 1),
        wp.min(int(wp.floor((position[2] + reach[2] - origin[2]) * inv_voxel_size)), dims[2] - 1),
    )

    voxel_size = 1.0 / inv_voxel_size
    delta_i = origin + voxel_size * wp.vec3(float(lower[0]), float(lower[1]), float(lower[2])) - position
    step_x = voxel_size * wp.vec3(G[0, 0], G[1, 0], G[2, 0])
    step_y = voxel_size * wp.vec3(G[0, 1], G[1, 1], G[2, 1])
    step_z = voxel_size * wp.vec3(G[0, 2], G[1, 2], G[2, 2])
    transformed_i = G * delta_i
    for i in range(lower[0], upper[0] + 1):
        transformed_j = transformed_i
        for j in range(lower[1], upper[1] + 1):
            transformed = transformed_j
            for k in range(lower[2], upper[2] + 1):
                q_sq = wp.dot(transformed, transformed)
                if q_sq < 4.0:
                    wp.atomic_add(field, _node_index(i, j, k, dims), weight * _cubic_bspline(wp.sqrt(q_sq)))
                transformed += step_z
            transformed_j += step_y
        transformed_i += step_x


@wp.kernel
def evaluate_particle_sdf_isotropic(
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    radius_scale: float,
    band: float,
    grid_origin: wp.array[wp.vec3],
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    inv_voxel_size: float,
    field: wp.array[float],
):
    particle = wp.tid()
    if not _is_active(flags, use_flags, particle) or grid_counts[_GRID_OVERFLOW] != 0:
        return

    position = smoothed[particle]
    radius = radii[particle] * radius_scale
    if radius <= 0.0 or not wp.isfinite(radius):
        return
    reach = band * radius
    origin = grid_origin[0]
    dims = grid_dims[0]

    lower = wp.vec3i(
        wp.max(int(wp.ceil((position[0] - reach - origin[0]) * inv_voxel_size)), 0),
        wp.max(int(wp.ceil((position[1] - reach - origin[1]) * inv_voxel_size)), 0),
        wp.max(int(wp.ceil((position[2] - reach - origin[2]) * inv_voxel_size)), 0),
    )
    upper = wp.vec3i(
        wp.min(int(wp.floor((position[0] + reach - origin[0]) * inv_voxel_size)), dims[0] - 1),
        wp.min(int(wp.floor((position[1] + reach - origin[1]) * inv_voxel_size)), dims[1] - 1),
        wp.min(int(wp.floor((position[2] + reach - origin[2]) * inv_voxel_size)), dims[2] - 1),
    )

    voxel_size = 1.0 / inv_voxel_size
    for i in range(lower[0], upper[0] + 1):
        x = origin[0] + voxel_size * float(i) - position[0]
        for j in range(lower[1], upper[1] + 1):
            y = origin[1] + voxel_size * float(j) - position[1]
            for k in range(lower[2], upper[2] + 1):
                z = origin[2] + voxel_size * float(k) - position[2]
                distance_sq = x * x + y * y + z * z
                if distance_sq <= reach * reach:
                    wp.atomic_min(field, _node_index(i, j, k, dims), wp.sqrt(distance_sq) - radius)


@wp.kernel
def evaluate_particle_sdf_anisotropic(
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[wp.int32],
    use_flags: int,
    G_matrices: wp.array[wp.mat33],
    det_G: wp.array[float],
    density_reach: wp.array[wp.vec3],
    radius_scale: float,
    band: float,
    grid_origin: wp.array[wp.vec3],
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    inv_voxel_size: float,
    field: wp.array[float],
):
    particle = wp.tid()
    if not _is_active(flags, use_flags, particle) or grid_counts[_GRID_OVERFLOW] != 0:
        return

    position = smoothed[particle]
    radius = radii[particle] * radius_scale
    if radius <= 0.0 or not wp.isfinite(radius):
        return

    G = G_matrices[particle]
    det_root = wp.pow(wp.max(det_G[particle], 1.0e-24), 1.0 / 3.0)
    radius_normalization = det_root * radius
    H = G * (1.0 / radius_normalization)
    reach = density_reach[particle] * (0.5 * band * radius_normalization)
    origin = grid_origin[0]
    dims = grid_dims[0]

    lower = wp.vec3i(
        wp.max(int(wp.ceil((position[0] - reach[0] - origin[0]) * inv_voxel_size)), 0),
        wp.max(int(wp.ceil((position[1] - reach[1] - origin[1]) * inv_voxel_size)), 0),
        wp.max(int(wp.ceil((position[2] - reach[2] - origin[2]) * inv_voxel_size)), 0),
    )
    upper = wp.vec3i(
        wp.min(int(wp.floor((position[0] + reach[0] - origin[0]) * inv_voxel_size)), dims[0] - 1),
        wp.min(int(wp.floor((position[1] + reach[1] - origin[1]) * inv_voxel_size)), dims[1] - 1),
        wp.min(int(wp.floor((position[2] + reach[2] - origin[2]) * inv_voxel_size)), dims[2] - 1),
    )

    voxel_size = 1.0 / inv_voxel_size
    delta_i = origin + voxel_size * wp.vec3(float(lower[0]), float(lower[1]), float(lower[2])) - position
    transformed_i = H * delta_i
    step_x = voxel_size * wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    step_y = voxel_size * wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    step_z = voxel_size * wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    H_transpose = wp.transpose(H)
    band_sq = band * band
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
                    wp.atomic_min(field, _node_index(i, j, k, dims), sdf)
                transformed += step_z
            transformed_j += step_y
        transformed_i += step_x


@wp.kernel
def density_to_sdf(
    field: wp.array[float],
    grid_counts: wp.array[wp.int32],
    threshold: float,
    stride: int,
):
    index = wp.tid()
    node_count = grid_counts[_GRID_NODE_COUNT]
    if grid_counts[_GRID_OVERFLOW] != 0:
        node_count = 0
    while index < node_count:
        field[index] = threshold - field[index]
        index += stride


@wp.kernel
def blur_field_axis(
    source: wp.array[float],
    destination: wp.array[float],
    weights: wp.array[float],
    half_width: int,
    axis: int,
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    stride: int,
):
    index = wp.tid()
    node_count = grid_counts[_GRID_NODE_COUNT]
    if grid_counts[_GRID_OVERFLOW] != 0:
        node_count = 0
    dims = grid_dims[0]
    while index < node_count:
        coordinate = _unravel_node(index, dims)
        value = source[index] * weights[0]
        for offset in range(1, half_width + 1):
            lower = coordinate
            upper = coordinate
            if axis == 0:
                lower[0] = wp.max(coordinate[0] - offset, 0)
                upper[0] = wp.min(coordinate[0] + offset, dims[0] - 1)
            elif axis == 1:
                lower[1] = wp.max(coordinate[1] - offset, 0)
                upper[1] = wp.min(coordinate[1] + offset, dims[1] - 1)
            else:
                lower[2] = wp.max(coordinate[2] - offset, 0)
                upper[2] = wp.min(coordinate[2] + offset, dims[2] - 1)
            value += weights[offset] * (
                source[_node_index(lower[0], lower[1], lower[2], dims)]
                + source[_node_index(upper[0], upper[1], upper[2], dims)]
            )
        destination[index] = value
        index += stride


@wp.kernel
def redistance_step(
    sdf: wp.array[float],
    sdf_out: wp.array[float],
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    inv_voxel_size: float,
    stride: int,
):
    index = wp.tid()
    node_count = grid_counts[_GRID_NODE_COUNT]
    if grid_counts[_GRID_OVERFLOW] != 0:
        node_count = 0
    dims = grid_dims[0]
    while index < node_count:
        coordinate = _unravel_node(index, dims)
        i = coordinate[0]
        j = coordinate[1]
        k = coordinate[2]
        value = sdf[index]
        sign = wp.sign(value)

        dx_m = sdf[_node_index(wp.max(i - 1, 0), j, k, dims)]
        dx_p = sdf[_node_index(wp.min(i + 1, dims[0] - 1), j, k, dims)]
        dy_m = sdf[_node_index(i, wp.max(j - 1, 0), k, dims)]
        dy_p = sdf[_node_index(i, wp.min(j + 1, dims[1] - 1), k, dims)]
        dz_m = sdf[_node_index(i, j, wp.max(k - 1, 0), dims)]
        dz_p = sdf[_node_index(i, j, wp.min(k + 1, dims[2] - 1), dims)]

        ax = wp.max(wp.max(sign * (value - dx_m), 0.0), wp.max(-sign * (dx_p - value), 0.0)) * inv_voxel_size
        ay = wp.max(wp.max(sign * (value - dy_m), 0.0), wp.max(-sign * (dy_p - value), 0.0)) * inv_voxel_size
        az = wp.max(wp.max(sign * (value - dz_m), 0.0), wp.max(-sign * (dz_p - value), 0.0)) * inv_voxel_size
        gradient_magnitude = wp.sqrt(ax * ax + ay * ay + az * az)

        voxel_size_sq = 1.0 / (inv_voxel_size * inv_voxel_size)
        smooth_sign = value / wp.sqrt(value * value + gradient_magnitude * gradient_magnitude * voxel_size_sq + 1.0e-20)
        sdf_out[index] = value - 0.5 / inv_voxel_size * smooth_sign * (gradient_magnitude - 1.0)
        index += stride


@wp.kernel
def reset_edge_indices(
    edge_indices: wp.array[wp.int32],
    grid_counts: wp.array[wp.int32],
    stride: int,
):
    index = wp.tid()
    edge_count = 3 * grid_counts[_GRID_NODE_COUNT]
    if grid_counts[_GRID_OVERFLOW] != 0:
        edge_count = 0
    while index < edge_count:
        edge_indices[index] = wp.int32(-1)
        index += stride


@wp.kernel
def extract_mesh_vertices(
    field: wp.array[float],
    threshold: float,
    grid_origin: wp.array[wp.vec3],
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    voxel_size: float,
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.int32],
    mesh_counts: wp.array[wp.int32],
    write_output: int,
    stride: int,
):
    edge_slot = wp.tid()
    edge_count = 3 * grid_counts[_GRID_NODE_COUNT]
    if grid_counts[_GRID_OVERFLOW] != 0:
        edge_count = 0
    dims = grid_dims[0]
    while edge_slot < edge_count:
        node_index = edge_slot // 3
        axis = edge_slot - 3 * node_index
        coordinate = _unravel_node(node_index, dims)
        neighbor = coordinate
        neighbor[axis] += 1
        if neighbor[axis] < dims[axis]:
            neighbor_index = _node_index(neighbor[0], neighbor[1], neighbor[2], dims)
            value = field[node_index]
            neighbor_value = field[neighbor_index]
            if (value >= threshold and neighbor_value < threshold) or (
                value < threshold and neighbor_value >= threshold
            ):
                output_index = wp.atomic_add(mesh_counts, _MESH_VERTEX_COUNT, wp.int32(1))
                if write_output != 0:
                    interpolation = wp.clamp((threshold - value) / (neighbor_value - value), 0.0, 1.0)
                    position = wp.vec3(float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))
                    position[axis] += interpolation
                    vertices[output_index] = grid_origin[0] + voxel_size * position
                    edge_indices[edge_slot] = output_index
        edge_slot += stride


@wp.kernel
def extract_mesh_indices(
    field: wp.array[float],
    threshold: float,
    grid_dims: wp.array[wp.vec3i],
    grid_counts: wp.array[wp.int32],
    case_ranges: wp.array[wp.int32],
    local_edges: wp.array[wp.int32],
    corner_offsets: wp.array[wp.vec3i],
    edge_offsets: wp.array[wp.vec3i],
    edge_axes: wp.array[wp.int32],
    edge_indices: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    mesh_counts: wp.array[wp.int32],
    write_output: int,
    stride: int,
):
    cell_index = wp.tid()
    cell_count = grid_counts[_GRID_CELL_COUNT]
    if grid_counts[_GRID_OVERFLOW] != 0:
        cell_count = 0
    dims = grid_dims[0]
    cell_dims = dims - wp.vec3i(1)
    while cell_index < cell_count:
        coordinate = _unravel_cell(cell_index, cell_dims)
        case = int(0)
        for corner in range(8):
            corner_coordinate = coordinate + corner_offsets[corner]
            value = field[_node_index(corner_coordinate[0], corner_coordinate[1], corner_coordinate[2], dims)]
            if value >= threshold:
                case |= 1 << corner

        local_begin = case_ranges[case]
        local_end = case_ranges[case + 1]
        local_count = local_end - local_begin
        if local_count > 0:
            output_begin = wp.atomic_add(mesh_counts, _MESH_INDEX_COUNT, local_count)
            if write_output != 0:
                for local_index in range(local_begin, local_end):
                    edge = local_edges[local_index]
                    edge_coordinate = coordinate + edge_offsets[edge]
                    edge_node = _node_index(edge_coordinate[0], edge_coordinate[1], edge_coordinate[2], dims)
                    indices[output_begin + local_index - local_begin] = edge_indices[3 * edge_node + edge_axes[edge]]
        cell_index += stride


@wp.kernel
def flip_mesh_winding(indices: wp.array[wp.int32], mesh_counts: wp.array[wp.int32], stride: int):
    triangle = wp.tid()
    triangle_count = mesh_counts[_MESH_INDEX_COUNT] // 3
    while triangle < triangle_count:
        base = 3 * triangle
        first = indices[base]
        indices[base] = indices[base + 1]
        indices[base + 1] = first
        triangle += stride


@wp.kernel
def clear_mesh_neighbors(
    neighbor_sum: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    mesh_counts: wp.array[wp.int32],
    stride: int,
):
    vertex = wp.tid()
    vertex_count = mesh_counts[_MESH_VERTEX_COUNT]
    while vertex < vertex_count:
        neighbor_sum[vertex] = wp.vec3(0.0)
        valence[vertex] = wp.int32(0)
        vertex += stride


@wp.kernel
def scatter_mesh_neighbors(
    vertices: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    neighbor_sum: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    mesh_counts: wp.array[wp.int32],
    stride: int,
):
    index = wp.tid()
    index_count = mesh_counts[_MESH_INDEX_COUNT]
    while index < index_count:
        triangle = index // 3
        local = index - 3 * triangle
        base = 3 * triangle
        vertex = indices[base + local]
        neighbor_1 = indices[base + (local + 1) % 3]
        neighbor_2 = indices[base + (local + 2) % 3]
        wp.atomic_add(neighbor_sum, vertex, vertices[neighbor_1] + vertices[neighbor_2])
        wp.atomic_add(valence, vertex, wp.int32(2))
        index += stride


@wp.kernel
def apply_mesh_smoothing(
    vertices: wp.array[wp.vec3],
    neighbor_sum: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    factor: float,
    smoothed: wp.array[wp.vec3],
    mesh_counts: wp.array[wp.int32],
    stride: int,
):
    vertex = wp.tid()
    vertex_count = mesh_counts[_MESH_VERTEX_COUNT]
    while vertex < vertex_count:
        count = valence[vertex]
        if count > 0:
            average = neighbor_sum[vertex] / float(count)
            smoothed[vertex] = vertices[vertex] + factor * (average - vertices[vertex])
        else:
            smoothed[vertex] = vertices[vertex]
        vertex += stride


@wp.kernel
def clear_mesh_normals(normals: wp.array[wp.vec3], mesh_counts: wp.array[wp.int32], stride: int):
    vertex = wp.tid()
    vertex_count = mesh_counts[_MESH_VERTEX_COUNT]
    while vertex < vertex_count:
        normals[vertex] = wp.vec3(0.0)
        vertex += stride


@wp.kernel
def accumulate_mesh_normals(
    vertices: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    mesh_counts: wp.array[wp.int32],
    stride: int,
):
    triangle = wp.tid()
    triangle_count = mesh_counts[_MESH_INDEX_COUNT] // 3
    while triangle < triangle_count:
        base = 3 * triangle
        i0 = indices[base]
        i1 = indices[base + 1]
        i2 = indices[base + 2]
        normal = wp.cross(vertices[i1] - vertices[i0], vertices[i2] - vertices[i0])
        wp.atomic_add(normals, i0, normal)
        wp.atomic_add(normals, i1, normal)
        wp.atomic_add(normals, i2, normal)
        triangle += stride


@wp.kernel
def normalize_mesh_normals(normals: wp.array[wp.vec3], mesh_counts: wp.array[wp.int32], stride: int):
    vertex = wp.tid()
    vertex_count = mesh_counts[_MESH_VERTEX_COUNT]
    while vertex < vertex_count:
        normals[vertex] = wp.normalize(normals[vertex])
        vertex += stride


_DENSITY_KERNEL_SUPPORT = 2.0


@wp.func
def _is_active_particle(flags: wp.array[wp.int32], use_flags: int, i: int) -> bool:
    if use_flags != 0:
        return (flags[i] & ParticleFlags.ACTIVE) != wp.int32(0)
    return True


@wp.func
def _weight_squared(dist_sq: float, inv_radius_sq: float) -> float:
    """Cubic falloff weight: w = (1 - (d/r)^3) for d < r."""
    q_sq = dist_sq * inv_radius_sq
    if q_sq >= 1.0:
        return 0.0
    return 1.0 - q_sq * wp.sqrt(q_sq)


# ---------------------------------------------------------------------------
# Pass 1: Smooth particle centers (Eq. 6 of Yu & Turk 2010)
# ---------------------------------------------------------------------------


@wp.kernel
def _smooth_positions(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    search_radius: float,
    smooth_lambda: float,
    smoothed: wp.array[wp.vec3],
):
    i = wp.tid()
    xi = positions[i]
    offset_sum = wp.vec3(0.0)
    w_sum = float(0.0)
    radius_sq = search_radius * search_radius
    inv_radius_sq = 1.0 / radius_sq

    query = wp.hash_grid_query(grid, xi, search_radius)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        offset = positions[idx] - xi
        dist_sq = wp.dot(offset, offset)
        if dist_sq < radius_sq:
            w = _weight_squared(dist_sq, inv_radius_sq)
            offset_sum += w * offset
            w_sum += w

    if w_sum > 0.0:
        smoothed[i] = xi + (smooth_lambda / w_sum) * offset_sum
    else:
        smoothed[i] = xi


# ---------------------------------------------------------------------------
# Pass 2: Per-particle anisotropy via Weighted PCA (Eqs. 9-16)
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_anisotropy(
    grid: wp.uint64,
    smoothed: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    use_flags: int,
    search_radius: float,
    anisotropy_ratio: float,
    anisotropy_scale: float,
    kernel_scale: float,
    anisotropy_min_neighbors: int,
    anisotropy_strength: float,
    G_out: wp.array[wp.mat33],
    det_G_out: wp.array[float],
    density_reach_out: wp.array[wp.vec3],
):
    i = wp.tid()

    if not _is_active_particle(flags, use_flags, i):
        # Keep inactive slots recognizable if internal buffers are inspected.
        G_out[i] = wp.mat33(0.0)
        det_G_out[i] = 0.0
        density_reach_out[i] = wp.vec3(0.0)
        return

    xi = smoothed[i]
    h = search_radius

    mean_offset = wp.vec3(0.0)
    moment_xx = float(0.0)
    moment_xy = float(0.0)
    moment_xz = float(0.0)
    moment_yy = float(0.0)
    moment_yz = float(0.0)
    moment_zz = float(0.0)
    w_sum = float(0.0)
    count = int(0)
    inv_radius_sq = 1.0 / (search_radius * search_radius)

    query = wp.hash_grid_query(grid, xi, search_radius)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        if _is_active_particle(flags, use_flags, idx):
            offset = smoothed[idx] - xi
            w = _weight_squared(wp.dot(offset, offset), inv_radius_sq)
            mean_offset += w * offset
            moment_xx += w * offset[0] * offset[0]
            moment_xy += w * offset[0] * offset[1]
            moment_xz += w * offset[0] * offset[2]
            moment_yy += w * offset[1] * offset[1]
            moment_yz += w * offset[1] * offset[2]
            moment_zz += w * offset[2] * offset[2]
            w_sum += w
            if w > 0.0:
                count += 1

    inv_h = 1.0 / h
    G = wp.identity(n=3, dtype=float) * inv_h
    det_g = inv_h * inv_h * inv_h

    # ``count`` includes the particle itself, so this requires at least
    # ``anisotropy_min_neighbors`` other particles.
    if count > anisotropy_min_neighbors and w_sum > 0.0:
        mean_offset = mean_offset / w_sum
        inv_w_sum = 1.0 / w_sum
        C = wp.mat33(
            moment_xx * inv_w_sum - mean_offset[0] * mean_offset[0],
            moment_xy * inv_w_sum - mean_offset[0] * mean_offset[1],
            moment_xz * inv_w_sum - mean_offset[0] * mean_offset[2],
            moment_xy * inv_w_sum - mean_offset[0] * mean_offset[1],
            moment_yy * inv_w_sum - mean_offset[1] * mean_offset[1],
            moment_yz * inv_w_sum - mean_offset[1] * mean_offset[2],
            moment_xz * inv_w_sum - mean_offset[0] * mean_offset[2],
            moment_yz * inv_w_sum - mean_offset[1] * mean_offset[2],
            moment_zz * inv_w_sum - mean_offset[2] * mean_offset[2],
        )

        U, sigma, _V = wp.svd3(C)

        # Erratum fix: covariance eigenvalues are variances; sqrt for axis lengths
        s1 = wp.sqrt(wp.max(sigma[0], 1.0e-10))
        s2 = wp.sqrt(wp.max(sigma[1], 1.0e-10))
        s3 = wp.sqrt(wp.max(sigma[2], 1.0e-10))

        # Clamp minimum eigenvalue ratio (Eq. 14)
        s2 = wp.max(s2, s1 / anisotropy_ratio)
        s3 = wp.max(s3, s1 / anisotropy_ratio)

        # Match the geometric-mean radius of anisotropic kernels to the
        # isotropic kernel scale, then apply an optional relative multiplier.
        s_geo = wp.pow(s1 * s2 * s3, 1.0 / 3.0)
        anisotropic_axis_scale = kernel_scale * anisotropy_scale / wp.max(s_geo, 1.0e-10)

        inv_s1 = 1.0 / (anisotropic_axis_scale * s1)
        inv_s2 = 1.0 / (anisotropic_axis_scale * s2)
        inv_s3 = 1.0 / (anisotropic_axis_scale * s3)

        # Blend inverse kernel radii in the PCA frame.  Full strength
        # preserves the WPCA ellipsoid for every particle with enough
        # neighbors; sparse particles still use the isotropic fallback below.
        blend = anisotropy_strength

        iso_scale = 1.0 / (kernel_scale * h)
        if blend <= 0.0:
            G = wp.identity(n=3, dtype=float) * iso_scale
            det_g = iso_scale * iso_scale * iso_scale
        else:
            g1 = (1.0 - blend) * iso_scale + blend * inv_h * inv_s1
            g2 = (1.0 - blend) * iso_scale + blend * inv_h * inv_s2
            g3 = (1.0 - blend) * iso_scale + blend * inv_h * inv_s3
            G = U @ wp.diag(wp.vec3(g1, g2, g3)) @ wp.transpose(U)
            det_g = g1 * g2 * g3
    elif count <= anisotropy_min_neighbors and count > 0:
        scale = 1.0 / (kernel_scale * h)
        G = wp.identity(n=3, dtype=float) * scale
        det_g = scale * scale * scale

    G_out[i] = G
    det_G_out[i] = det_g
    G_inv = wp.inverse(G)
    density_reach_out[i] = _DENSITY_KERNEL_SUPPORT * wp.vec3(
        wp.length(wp.vec3(G_inv[0, 0], G_inv[1, 0], G_inv[2, 0])),
        wp.length(wp.vec3(G_inv[0, 1], G_inv[1, 1], G_inv[2, 1])),
        wp.length(wp.vec3(G_inv[0, 2], G_inv[1, 2], G_inv[2, 2])),
    )


@wp.kernel
def _fill_isotropic_G(
    kernel_radius: float,
    kernel_scale: float,
    flags: wp.array[wp.int32],
    use_flags: int,
    G_out: wp.array[wp.mat33],
    det_G_out: wp.array[float],
    density_reach_out: wp.array[wp.vec3],
):
    """Fill active particles with isotropic G and zero inactive slots."""
    i = wp.tid()
    if not _is_active_particle(flags, use_flags, i):
        G_out[i] = wp.mat33(0.0)
        det_G_out[i] = 0.0
        density_reach_out[i] = wp.vec3(0.0)
        return

    scale = 1.0 / (kernel_scale * kernel_radius)
    G_out[i] = wp.identity(n=3, dtype=float) * scale
    det_G_out[i] = scale * scale * scale
    density_reach_out[i] = wp.vec3(_DENSITY_KERNEL_SUPPORT / scale)
