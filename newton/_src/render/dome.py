# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diffuse HDRI dome lighting via order-2 spherical harmonics.

The unshadowed diffuse response of a surface to an environment is the
cosine-weighted hemisphere integral of incoming radiance around the surface
normal. Following Ramamoorthi & Hanrahan (2001), "An Efficient Representation
for Irradiance Environment Maps", this is captured almost exactly by 9
spherical-harmonic coefficients per color channel: the equirectangular HDRI is
projected onto the SH basis once (:func:`compute_dome_sh9`), and shading
evaluates a 9-term polynomial in the world-space normal
(:func:`eval_dome_irradiance`) with no texture sampling at render time.

The convolution constants (``A_0 = pi``, ``A_1 = 2*pi/3``, ``A_2 = pi/4``) and
the Lambert ``1/pi`` factor are baked into the coefficients so that evaluation
returns the diffuse reflectance factor directly: a constant white environment
of value ``L`` yields ``L`` for every normal, so a diffuse surface is lit to
``albedo * L``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core import Axis
from ..core.types import AxisType
from . import raytrace
from .types import DomeLight, RenderConfig, RenderState

if TYPE_CHECKING:
    from ..geometry.optix import OptixShapes

_PI = 3.141592653589793
_TWO_PI = 6.283185307179586

# Real spherical-harmonic basis normalization constants (bands l = 0, 1, 2).
_K0 = 0.282095
_K1 = 0.488603
_K2 = 1.092548  # l=2, m = -2, -1, +1
_K3 = 0.315392  # l=2, m = 0
_K4 = 0.546274  # l=2, m = +2

# Convolution constants A_l divided by the Lambert pi, per SH band.
_A_OVER_PI = np.array(
    [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25],
    dtype=np.float64,
)


def equirect_frame(up_axis: AxisType, rotation: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal world-space basis for the equirectangular mapping.

    Returns ``(forward, bitangent, up)`` such that a texel at polar angle
    ``theta`` (from ``up``) and azimuth ``phi`` maps to the world direction
    ``up*cos(theta) + sin(theta)*(forward*cos(phi) + bitangent*sin(phi))``. The
    ``rotation`` yaw is baked into ``forward``/``bitangent`` so the same frame
    reconstructs directions from environment samples (see
    :func:`~newton._src.render.dome.sample_env`).

    Args:
        up_axis: Scene up axis (the equirectangular pole).
        rotation: Azimuth offset [rad] about ``up_axis``.

    Returns:
        Three ``(3,)`` float64 unit vectors ``(forward, bitangent, up)``.
    """
    up = np.asarray(Axis.from_any(up_axis).to_vector(), dtype=np.float64)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    forward0 = np.cross(ref, up)
    forward0 /= np.linalg.norm(forward0)
    bitangent0 = np.cross(up, forward0)
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    forward = forward0 * cos_r + bitangent0 * sin_r
    bitangent = -forward0 * sin_r + bitangent0 * cos_r
    return forward, bitangent, up


def compute_dome_sh9(
    equirect_rgb: np.ndarray,
    up_axis: AxisType,
    rotation: float = 0.0,
) -> np.ndarray:
    """Project an equirectangular HDRI onto 9 diffuse-irradiance SH coefficients.

    Args:
        equirect_rgb: Equirectangular environment image of linear radiance,
            shape ``(H, W, C)`` with ``C >= 3``. Row 0 maps to the ``+up``
            pole; columns sweep azimuth over ``[0, 2*pi)``.
        up_axis: Scene up axis; sets the pole of the equirectangular mapping so
            the coefficients match world-space normals.
        rotation: Azimuth offset [rad] applied about ``up_axis`` (yaw), rotating
            the environment around the up axis.

    Returns:
        Diffuse-irradiance SH coefficients, shape ``(9, 3)`` float32. Dotting
        these with the SH basis at a normal yields the diffuse reflectance
        factor for that normal (see module docstring).
    """
    rgb = np.asarray(equirect_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or min(rgb.shape[:2]) == 0:
        raise ValueError(f"equirect image must have nonempty shape (H, W, C>=3), got {rgb.shape}")
    rgb = rgb[:, :, :3]
    height, width = rgb.shape[:2]
    theta = np.linspace(0.0, np.pi, height + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, width + 1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    dtheta = np.diff(theta)
    dphi = np.diff(phi)

    # Integrate each texel's constant radiance over its full solid angle.
    # The SH basis needs only the constant, first, and second direction moments.
    # Separable row/column integrals avoid a (9, H, W) basis allocation.
    theta_0 = -np.diff(cos_theta)
    theta_z = -np.diff(cos_theta**2) / 2.0
    theta_xy = dtheta / 2.0 - np.diff(np.sin(2.0 * theta)) / 4.0
    theta_zz = -np.diff(cos_theta**3) / 3.0
    theta_zxy = np.diff(sin_theta**3) / 3.0
    theta_xyxy = theta_0 - theta_zz
    phi_x = np.diff(np.sin(phi))
    phi_y = -np.diff(np.cos(phi))
    phi_xx = dphi / 2.0 + np.diff(np.sin(2.0 * phi)) / 4.0
    phi_xy = np.diff(np.sin(phi) ** 2) / 2.0
    phi_yy = dphi - phi_xx

    column_integrals = np.einsum(
        "hwc,kw->hkc", rgb, np.stack((dphi, phi_x, phi_y, phi_xx, phi_xy, phi_yy)), optimize=True
    )

    def integrate(rows, column):
        return rows @ column_integrals[:, column]

    integral = integrate(theta_0, 0)
    first = np.stack((integrate(theta_z, 0), integrate(theta_xy, 1), integrate(theta_xy, 2)))
    second = np.empty((3, 3, 3), dtype=np.float64)
    second[0, 0] = integrate(theta_zz, 0)
    second[0, 1] = second[1, 0] = integrate(theta_zxy, 1)
    second[0, 2] = second[2, 0] = integrate(theta_zxy, 2)
    second[1, 1] = integrate(theta_xyxy, 3)
    second[1, 2] = second[2, 1] = integrate(theta_xyxy, 4)
    second[2, 2] = integrate(theta_xyxy, 5)

    forward, bitangent, up = equirect_frame(up_axis, rotation)
    frame = np.column_stack((up, forward, bitangent))
    first = frame @ first
    second = np.einsum("ia,abc,jb->ijc", frame, second, frame)
    coefficients = np.stack(
        (
            _K0 * integral,
            _K1 * first[1],
            _K1 * first[2],
            _K1 * first[0],
            _K2 * second[0, 1],
            _K2 * second[1, 2],
            _K3 * (3.0 * second[2, 2] - integral),
            _K2 * second[0, 2],
            _K4 * (second[0, 0] - second[1, 1]),
        )
    )
    return (coefficients * _A_OVER_PI[:, None]).astype(np.float32)


@wp.kernel(enable_backward=False)
def advance_dome_seed(seed: wp.array[wp.int32]):
    seed[0] = seed[0] + 1


@wp.func
def eval_dome_irradiance(dome: DomeLight, n: wp.vec3f) -> wp.vec3f:
    """Evaluate the diffuse dome reflectance factor for a world-space normal.

    Args:
        dome: Dome light data; uses :attr:`DomeLight.spherical_harmonics` (SH
            coefficients from :func:`compute_dome_sh9`).
        n: Unit surface normal in world space.

    Returns:
        Diffuse reflectance factor per channel, clamped to non-negative.
    """
    spherical_harmonics = dome.spherical_harmonics
    dx = n[0]
    dy = n[1]
    dz = n[2]
    b0 = 0.282095
    b1 = 0.488603 * dy
    b2 = 0.488603 * dz
    b3 = 0.488603 * dx
    b4 = 1.092548 * dx * dy
    b5 = 1.092548 * dy * dz
    b6 = 0.315392 * (3.0 * dz * dz - 1.0)
    b7 = 1.092548 * dx * dz
    b8 = 0.546274 * (dx * dx - dy * dy)
    irr = (
        spherical_harmonics[0] * b0
        + spherical_harmonics[1] * b1
        + spherical_harmonics[2] * b2
        + spherical_harmonics[3] * b3
        + spherical_harmonics[4] * b4
        + spherical_harmonics[5] * b5
        + spherical_harmonics[6] * b6
        + spherical_harmonics[7] * b7
        + spherical_harmonics[8] * b8
    )
    # SH reconstruction can ring slightly negative; clamp to keep color valid.
    return wp.vec3f(wp.max(irr[0], 0.0), wp.max(irr[1], 0.0), wp.max(irr[2], 0.0))


@wp.func
def _sample_cumulative_distribution_1d(
    cumulative_distribution: wp.array[wp.float32], count: wp.int32, value: wp.float32
) -> wp.int32:
    """Return the first index whose cumulative value reaches ``value`` (binary search)."""
    lo = wp.int32(0)
    hi = count
    while lo < hi:
        mid = (lo + hi) >> 1
        if cumulative_distribution[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return wp.clamp(lo, 0, count - 1)


@wp.func
def _sample_cumulative_distribution_row(
    cumulative_distribution: wp.array2d[wp.float32], row: wp.int32, count: wp.int32, value: wp.float32
) -> wp.int32:
    """Binary search a single row of a 2D per-row cumulative distribution."""
    lo = wp.int32(0)
    hi = count
    while lo < hi:
        mid = (lo + hi) >> 1
        if cumulative_distribution[row, mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return wp.clamp(lo, 0, count - 1)


@wp.func
def sample_env(
    dome: DomeLight,
    r1: wp.float32,
    r2: wp.float32,
    r3: wp.float32,
    r4: wp.float32,
) -> tuple[wp.vec3f, wp.vec3f, wp.float32]:
    """Importance-sample the environment proportionally to radiance x solid angle.

    Draws a texel from the precomputed marginal/conditional CDFs (built weighted
    by luminance and texel solid angle), jitters within the texel, and returns the
    world-space direction, that texel's radiance, and the sampling pdf in
    solid-angle measure. Aiming rays at bright directions (e.g. the sun) is what
    keeps a concentrated source from producing fireflies / undersampling.

    Args:
        dome: Dome light data; uses :attr:`DomeLight.row_cumulative_distribution`,
            :attr:`DomeLight.column_cumulative_distribution`, :attr:`DomeLight.environment_map`,
            :attr:`DomeLight.forward`, :attr:`DomeLight.bitangent`,
            :attr:`DomeLight.up`, and :attr:`DomeLight.probability_density_scale`.
        r1: Uniform value selecting the row.
        r2: Uniform value selecting the column.
        r3: Uniform value jittering within the texel (polar).
        r4: Uniform value jittering within the texel (azimuth).

    Returns:
        ``(direction, radiance, pdf)``.
    """
    height = dome.environment_map.shape[0]
    width = dome.environment_map.shape[1]
    y = _sample_cumulative_distribution_1d(dome.row_cumulative_distribution, height, r1)
    x = _sample_cumulative_distribution_row(dome.column_cumulative_distribution, y, width, r2)

    # Uniform solid angle within the selected texel, including coarse maps at the poles.
    cos_top = wp.cos(float(y) / float(height) * float(_PI))
    cos_bottom = wp.cos(float(y + 1) / float(height) * float(_PI))
    cos_theta = cos_top + r3 * (cos_bottom - cos_top)
    phi = (float(x) + r4) / float(width) * float(_TWO_PI)
    sin_t = wp.sqrt(wp.max(0.0, 1.0 - cos_theta * cos_theta))
    direction = dome.up * cos_theta + sin_t * (dome.forward * wp.cos(phi) + dome.bitangent * wp.sin(phi))

    radiance = dome.environment_map[y, x]
    lum = 0.2126 * radiance[0] + 0.7152 * radiance[1] + 0.0722 * radiance[2]
    return direction, radiance, wp.max(lum, 1.0e-8) * dome.probability_density_scale


@wp.func
def sample_env_direction(dome: DomeLight, direction: wp.vec3f) -> wp.vec3f:
    """Look up the environment radiance visible along a world-space direction.

    Inverts the ``(theta, phi) -> direction`` mapping used by :func:`sample_env`
    to find the texel a camera ray would see, for displaying the dome as a
    background for rays that miss all geometry.

    Args:
        dome: Dome light data; uses :attr:`DomeLight.environment_map`,
            :attr:`DomeLight.forward`, :attr:`DomeLight.bitangent`, and
            :attr:`DomeLight.up`.
        direction: Unit ray direction in world space.

    Returns:
        Radiance of the texel visible along ``direction``.
    """
    height = dome.environment_map.shape[0]
    width = dome.environment_map.shape[1]
    cos_theta = wp.clamp(wp.dot(direction, dome.up), -1.0, 1.0)
    theta = wp.acos(cos_theta)
    phi = wp.atan2(wp.dot(direction, dome.bitangent), wp.dot(direction, dome.forward))
    if phi < 0.0:
        phi += float(_TWO_PI)

    y = wp.clamp(wp.int32(theta / float(_PI) * float(height)), 0, height - 1)
    x = wp.clamp(wp.int32(phi / float(_TWO_PI) * float(width)), 0, width - 1)
    return dome.environment_map[y, x]


def create_sample_dome_shadow_function(
    config: RenderConfig, state: RenderState, optix: OptixShapes | None = None
) -> wp.Function:
    first_hit = raytrace.create_first_hit_function(config, state, optix)
    dome_shadow_max_distance = (
        config.dome_shadow_max_distance if config.dome_shadow_max_distance > 0.0 else config.max_distance
    )

    @wp.func
    def sample_dome_shadow(
        dome: DomeLight,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        world_index: wp.int32,
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        shadow_origin: wp.vec3f,
        n: wp.vec3f,
        r1: wp.float32,
        r2: wp.float32,
        r3: wp.float32,
        r4: wp.float32,
    ) -> wp.vec3f:
        """Trace one dome shadow ray and return its (unoccluded) radiance contribution.

        Kept as a standalone function, rather than inlined into the shadow-sample
        loop in :func:`~newton._src.render.render.create_kernel`,
        so the loop body's compiled size stays constant regardless of
        :attr:`~newton.sensors.SensorCamera.RenderConfig.dome_shadow_samples`.
        Inlining it let the Warp compiler's loop unroller occasionally blow up the
        generated code for specific sample counts, corrupting the compiled kernel
        (illegal memory access at launch).
        """
        sample_dir, sample_radiance, sample_pdf = sample_env(dome, r1, r2, r3, r4)
        cos_n = wp.dot(sample_dir, n)
        if cos_n <= 0.0:
            return wp.vec3f(0.0)

        occluded = first_hit(
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            world_index,
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_transforms,
            shape_source_ptr,
            particles_position,
            particles_radius,
            topology_particle_mask,
            triangle_mesh_id,
            triangle_mesh_group_roots,
            shadow_origin,
            sample_dir,
            wp.static(dome_shadow_max_distance),
        )
        if occluded:
            return wp.vec3f(0.0)

        return sample_radiance * (cos_n / (3.141592653589793 * sample_pdf))

    return sample_dome_shadow
