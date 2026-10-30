# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Particle packing measurements for density feedback and critical packing.

Each particle gets a measured volume fraction ``phi`` and an available fraction ``A``, the part of
the same kernel outside colliders. Both come from either a ``"grid"`` measurement, B-splines
centered at cell centers and read back linearly at the particle, or a ``"particle"`` measurement,
a cubic B-spline centered at the particle. The two fractions are independent, and the packing laws
only combine them per particle.
"""

import warp as wp
import warp.fem as fem
from warp.fem import cache

import newton

from .rasterized_collisions import Collider, collision_sdf

wp.set_module_options({"enable_backward": False, "max_unroll": 0})

PARTICLE_KERNEL_RADIUS = 1.0
"""Radius of the particle-centered cubic B-spline kernel, in voxels."""

_PARTICLE_KERNEL_SCALE = wp.constant(PARTICLE_KERNEL_RADIUS / 2.0)
"""Scale of the cubic B-spline, whose support radius is two."""


@wp.func
def quadratic_bspline(x: float):
    x = wp.abs(x)
    if x < 0.5:
        return 0.75 - x * x
    if x < 1.5:
        return 0.5 * (1.5 - x) * (1.5 - x)
    return 0.0


@wp.func
def quadratic_bspline_integral(x: float):
    """Antiderivative of :func:`quadratic_bspline`, from 0 below its support to 1 above it."""
    if x <= -1.5:
        return 0.0
    if x <= -0.5:
        return (x + 1.5) * (x + 1.5) * (x + 1.5) / 6.0
    if x <= 0.5:
        return 1.0 / 6.0 + 0.75 * (x + 0.5) - (x * x * x + 0.125) / 3.0
    if x <= 1.5:
        return 1.0 - (1.5 - x) * (1.5 - x) * (1.5 - x) / 6.0
    return 1.0


@wp.func
def cubic_bspline(x: float):
    x = wp.abs(x)
    if x < 1.0:
        return 2.0 / 3.0 - x * x + 0.5 * x * x * x
    if x < 2.0:
        return (2.0 - x) * (2.0 - x) * (2.0 - x) / 6.0
    return 0.0


@wp.func
def cubic_bspline_integral(x: float):
    """Antiderivative of :func:`cubic_bspline`, from 0 below its support to 1 above it."""
    u = -wp.abs(x)
    c = float(0.0)
    if u > -2.0:
        if u <= -1.0:
            c = (2.0 + u) * (2.0 + u) * (2.0 + u) * (2.0 + u) / 24.0
        else:
            c = 1.0 / 24.0 + 2.0 / 3.0 * (u + 1.0) - (u * u * u + 1.0) / 3.0 - (u * u * u * u - 1.0) / 8.0
    return wp.where(x > 0.0, 1.0 - c, c)


@wp.func
def half_box_lower(k: int):
    """Lower corner of the half-cell box ``k`` relative to its cell center, in voxels."""
    return wp.vec3(
        wp.where(k // 4 == 0, -0.5, 0.0),
        wp.where((k // 2) % 2 == 0, -0.5, 0.0),
        wp.where(k % 2 == 0, -0.5, 0.0),
    )


@wp.func
def free_kernel_mass(offset: wp.vec3, mask: wp.uint32, scale: float, particle_kernel: bool):
    """Kernel mass over the free half-cell boxes of one cell.

    ``offset`` is the kernel center relative to the cell center, in voxels. The particle kernel is
    the cubic B-spline of scale ``scale``, the grid kernel the quadratic B-spline of unit scale.
    """
    mass = float(0.0)
    for k in range(8):
        if mask & (wp.uint32(1) << wp.uint32(k)):
            lo = half_box_lower(k)
            w = float(1.0)
            for d in range(3):
                a = (offset[d] - lo[d]) / scale
                b = (offset[d] - lo[d] - 0.5) / scale
                if particle_kernel:
                    w *= cubic_bspline_integral(a) - cubic_bspline_integral(b)
                else:
                    w *= quadratic_bspline_integral(a) - quadratic_bspline_integral(b)
            mass += w
    return mass


@wp.kernel
def deformed_particle_volumes(
    volumes: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    deformed_volumes: wp.array[float],
):
    """Scale reference volumes by the elastic volume ratio ``det(F_e)``."""
    i = wp.tid()
    deformed_volumes[i] = volumes[i] * wp.determinant(elastic_strain[i])


@wp.func
def free_box_mask(
    center: wp.vec3,
    collider_environment: int,
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    h: float,
):
    """Bit ``k`` marks the half-cell box ``k`` of the cell at ``center`` as outside colliders."""
    sdf, _n, _v, _cid, _mid = collision_sdf(center, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0)
    if sdf > 0.866026 * h:
        return wp.uint32(255)
    if sdf < -0.866026 * h:
        return wp.uint32(0)
    mask = wp.uint32(0)
    for k in range(8):
        q = wp.vec3(float(k // 4) - 0.5, float((k // 2) % 2) - 0.5, float(k % 2) - 0.5)
        phi, _normal, _velocity, _collider, _material = collision_sdf(
            center + 0.5 * h * q, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0
        )
        if phi >= 0.0:
            mask |= wp.uint32(1) << wp.uint32(k)
    return mask


def make_cell_locator(grid):
    """Functions locating a position's cell and its neighbor cells, on dense or sparse grids."""

    @cache.dynamic_func(suffix=grid.name)
    def containing_cell(geo: grid.CellArg, x: wp.vec3, environment: int, h: float):
        """Lattice coordinates and center of the cell containing ``x``, and whether it exists."""
        if wp.static(isinstance(grid, fem.Nanogrid)):
            uvw = wp.volume_world_to_index(geo.cell_grid, x) + wp.vec3(0.5)
            local_cell = wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))
            return local_cell, wp.volume_index_to_world(geo.cell_grid, wp.vec3(local_cell)), True
        sample = grid.cell_lookup(geo, x, 0.0, environment)
        if sample.element_index < 0:
            return wp.vec3i(0), wp.vec3(0.0), False
        sample.element_coords = wp.vec3(0.5)
        return wp.vec3i(0), grid.cell_position(geo, sample), True

    @cache.dynamic_func(suffix=grid.name)
    def neighbor_cell(
        geo: grid.CellArg, local_cell: wp.vec3i, center: wp.vec3, offset: wp.vec3i, environment: int, h: float
    ):
        """Index of the cell ``offset`` away from the containing cell, or -1 outside the grid."""
        if wp.static(isinstance(grid, fem.Nanogrid)):
            ijk = local_cell + offset + geo.env_offsets[environment]
            cell = wp.volume_lookup_index(geo.cell_grid, ijk[0], ijk[1], ijk[2])
            if cell >= 0:
                if geo.cell_env[cell] != environment:
                    cell = -1
            return cell
        target_center = center + h * wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
        target = grid.cell_lookup(geo, target_center, 0.0, environment)
        if target.element_index < 0:
            return -1
        # Grid3D lookups clamp at their boundary: reject clamped cells.
        actual = grid.cell_position(geo, target)
        if wp.length_sq(actual - target_center) >= 1.0e-8 * h * h:
            return -1
        return target.element_index

    return containing_cell, neighbor_cell


def make_free_box_mask_kernel(grid):
    """Free half-cell boxes of every grid cell, sampled at their centers."""

    @cache.dynamic_kernel(suffix=grid.name)
    def free_box_masks(
        geo: grid.CellArg,
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        h: float,
        output: wp.array[wp.uint32],
    ):
        cell = wp.tid()
        if wp.static(isinstance(grid, fem.Nanogrid)):
            if cell >= wp.volume_voxel_count(geo.cell_grid):
                output[cell] = wp.uint32(0)
                return
        sample = fem.make_free_sample(cell, wp.vec3(0.5))
        center = grid.cell_position(geo, sample)
        environment = grid.cell_environment_index(geo, sample)
        output[cell] = free_box_mask(
            center, wp.where(separate_worlds, environment, -2), collider, body_q, body_qd, body_q_prev, h
        )

    return free_box_masks


def make_grid_volume_kernel(grid):
    """Scatter particle volumes to cell centers with quadratic B-spline weights, as cell volume fractions."""
    containing_cell, neighbor_cell = make_cell_locator(grid)

    @cache.dynamic_kernel(suffix=grid.name)
    def grid_volume(
        geo: grid.CellArg,
        positions: wp.array[wp.vec3],
        volumes: wp.array[float],
        flags: wp.array[int],
        environments: wp.array[int],
        h: float,
        output: wp.array[float],
    ):
        p = wp.tid()
        if (flags[p] & newton.ParticleFlags.ACTIVE) == 0 or volumes[p] == 0.0:
            return
        environment = int(0)
        if environments:
            environment = environments[p]
        x = positions[p]
        local_cell, containing_center, found = containing_cell(geo, x, environment, h)
        if not found:
            return
        for k in range(27):
            offset = wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            q = (x - containing_center) / h - wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
            weight = quadratic_bspline(q[0]) * quadratic_bspline(q[1]) * quadratic_bspline(q[2])
            if weight > 0.0:
                cell = neighbor_cell(geo, local_cell, containing_center, offset, environment, h)
                if cell >= 0 and cell < output.shape[0]:
                    wp.atomic_add(output, cell, weight * volumes[p] / (h * h * h))

    return grid_volume


def make_grid_capacity_kernel(grid):
    """Mass of each cell-centered quadratic B-spline outside colliders, from the free half-cell boxes."""
    containing_cell, neighbor_cell = make_cell_locator(grid)

    @cache.dynamic_kernel(suffix=grid.name)
    def grid_capacity(
        geo: grid.CellArg,
        masks: wp.array[wp.uint32],
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        h: float,
        output: wp.array[float],
    ):
        cell = wp.tid()
        if wp.static(isinstance(grid, fem.Nanogrid)):
            if cell >= wp.volume_voxel_count(geo.cell_grid):
                return
        sample = fem.make_free_sample(cell, wp.vec3(0.5))
        center = grid.cell_position(geo, sample)
        environment = grid.cell_environment_index(geo, sample)
        # Cells always contain their own center
        local_cell, containing_center, _found = containing_cell(geo, center, environment, h)
        mass = float(0.0)
        for k in range(27):
            offset = wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            neighbor = neighbor_cell(geo, local_cell, containing_center, offset, environment, h)
            mask = wp.uint32(0)
            if neighbor >= 0:
                mask = masks[neighbor]
            else:
                # Cells outside the grid hold no particles, but their free space still counts
                neighbor_center = center + h * wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                mask = free_box_mask(
                    neighbor_center,
                    wp.where(separate_worlds, environment, -2),
                    collider,
                    body_q,
                    body_qd,
                    body_q_prev,
                    h,
                )
            relative = -wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
            mass += free_kernel_mass(relative, mask, 1.0, False)
        output[cell] = mass

    return grid_capacity


def make_grid_readback_kernel(grid):
    """Linear interpolation of cell-center values at each particle.

    Cells outside the grid take no part, and the weights renormalize over the rest, so measured
    and available fractions read back with the same weights.
    """
    containing_cell, neighbor_cell = make_cell_locator(grid)

    @cache.dynamic_kernel(suffix=grid.name)
    def grid_readback(
        geo: grid.CellArg,
        positions: wp.array[wp.vec3],
        flags: wp.array[int],
        environments: wp.array[int],
        cell_values: wp.array[float],
        h: float,
        values: wp.array[float],
    ):
        p = wp.tid()
        values[p] = 0.0
        if (flags[p] & newton.ParticleFlags.ACTIVE) == 0:
            return
        environment = int(0)
        if environments:
            environment = environments[p]
        x = positions[p]
        local_cell, containing_center, found = containing_cell(geo, x, environment, h)
        if not found:
            return
        # Position on the lattice of cell centers, relative to the containing cell
        t = (x - containing_center) / h
        base = wp.vec3i(int(wp.floor(t[0])), int(wp.floor(t[1])), int(wp.floor(t[2])))
        value = float(0.0)
        total = float(0.0)
        for k in range(8):
            offset = base + wp.vec3i(k // 4, (k // 2) % 2, k % 2)
            weight = float(1.0)
            for d in range(3):
                weight *= wp.max(1.0 - wp.abs(t[d] - float(offset[d])), 0.0)
            cell = neighbor_cell(geo, local_cell, containing_center, offset, environment, h)
            if cell >= 0 and cell < cell_values.shape[0]:
                value += weight * cell_values[cell]
                total += weight
        if total > 0.0:
            values[p] = value / total

    return grid_readback


def make_particle_volume_kernel(pic):
    """Volume fraction at each particle from its neighbors, with the particle-centered cubic B-spline.

    Neighbors come from the PIC cell lists, one point per particle center.
    """
    domain = pic.domain
    grid = domain.geometry
    domain_indices = pic._use_domain_element_indices
    containing_cell, neighbor_cell = make_cell_locator(grid)

    @cache.dynamic_kernel(suffix=f"{grid.name}_{domain.name}_{domain_indices}")
    def particle_volume(
        geo: grid.CellArg,
        domain_arg: domain.ElementIndexArg,
        particle_offsets: wp.array[int],
        particle_indices: wp.array[int],
        positions: wp.array[wp.vec3],
        volumes: wp.array[float],
        flags: wp.array[int],
        environments: wp.array[int],
        h: float,
        values: wp.array[float],
    ):
        p = wp.tid()
        values[p] = 0.0
        if (flags[p] & newton.ParticleFlags.ACTIVE) == 0 or volumes[p] == 0.0:
            return
        environment = int(0)
        if environments:
            environment = environments[p]
        x = positions[p]
        local_cell, containing_center, found = containing_cell(geo, x, environment, h)
        if not found:
            return
        scale = h * _PARTICLE_KERNEL_SCALE
        inv_volume = 1.0 / (scale * scale * scale)
        rho = float(0.0)
        for k in range(27):
            offset = wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            cell = neighbor_cell(geo, local_cell, containing_center, offset, environment, h)
            qp_cell = cell
            if wp.static(domain_indices):
                if cell >= 0:
                    qp_cell = domain.element_partition_index(domain_arg, cell)
            if qp_cell >= 0 and qp_cell + 1 < particle_offsets.shape[0]:
                for entry in range(particle_offsets[qp_cell], particle_offsets[qp_cell + 1]):
                    q = particle_indices[entry]
                    if q >= 0 and q < volumes.shape[0]:
                        if flags[q] & newton.ParticleFlags.ACTIVE and volumes[q] > 0.0:
                            separation = (x - positions[q]) / scale
                            rho += (
                                volumes[q]
                                * inv_volume
                                * cubic_bspline(separation[0])
                                * cubic_bspline(separation[1])
                                * cubic_bspline(separation[2])
                            )
        values[p] = rho

    return particle_volume


def make_particle_capacity_kernel(grid):
    """Mass of the particle-centered cubic B-spline outside colliders, from the free half-cell boxes."""
    containing_cell, neighbor_cell = make_cell_locator(grid)

    @cache.dynamic_kernel(suffix=grid.name)
    def particle_capacity(
        geo: grid.CellArg,
        positions: wp.array[wp.vec3],
        flags: wp.array[int],
        environments: wp.array[int],
        masks: wp.array[wp.uint32],
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        h: float,
        values: wp.array[float],
    ):
        p = wp.tid()
        values[p] = 0.0
        if (flags[p] & newton.ParticleFlags.ACTIVE) == 0:
            return
        environment = int(0)
        if environments:
            environment = environments[p]
        x = positions[p]
        local_cell, containing_center, found = containing_cell(geo, x, environment, h)
        if not found:
            return
        available = float(0.0)
        for k in range(27):
            offset = wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            center = containing_center + h * wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
            cell = neighbor_cell(geo, local_cell, containing_center, offset, environment, h)
            mask = wp.uint32(0)
            if cell >= 0 and cell < masks.shape[0]:
                mask = masks[cell]
            else:
                # Cells outside the grid hold no particles, but their free space still counts
                mask = free_box_mask(
                    center, wp.where(separate_worlds, environment, -2), collider, body_q, body_qd, body_q_prev, h
                )
            available += free_kernel_mass((x - center) / h, mask, _PARTICLE_KERNEL_SCALE, True)
        values[p] = available

    return particle_capacity


@wp.kernel
def feedback_offset_values(
    measured: wp.array[float],
    available: wp.array[float],
    flags: wp.array[int],
    fraction: float,
    values: wp.array[float],
):
    """Push-only feedback, the volumetric strain bringing overfilled particles back to the available fraction.

    The particle quadrature weighs offsets and volumetric strain alike by the measured fraction, so
    ``dt div(u) = available / measured - 1`` brings the fraction to ``available``.
    """
    p = wp.tid()
    values[p] = 0.0
    if (flags[p] & newton.ParticleFlags.ACTIVE) == 0 or measured[p] <= 0.0:
        return
    values[p] = fraction * wp.min(available[p] / measured[p] - 1.0, 0.0)


@wp.kernel
def critical_offset_values(
    measured: wp.array[float],
    available: wp.array[float],
    flags: wp.array[int],
    critical_fraction: float,
    values: wp.array[float],
):
    """Compression that underfilled particles may undergo before reaching the critical fraction."""
    p = wp.tid()
    values[p] = 0.0
    if (flags[p] & newton.ParticleFlags.ACTIVE) == 0 or measured[p] <= 0.0:
        return
    values[p] = wp.max(critical_fraction * available[p] / measured[p] - 1.0, 0.0)


@fem.integrand
def point_density_offset(
    s: fem.Sample,
    tau: fem.Field,
    values: wp.array[float],
    flags: wp.array[int],
    inv_cell_volume: float,
):
    """Integrate offsets held at the strain quadrature points, with the weights of the strain mass."""
    if (flags[s.qp_index] & newton.ParticleFlags.ACTIVE) == 0:
        return 0.0
    return values[s.qp_index] * tau(s) * inv_cell_volume
