"""Utilities for hydroelastic pressure field construction."""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from newton._src.core.types import MAXVAL, Devicelike

from .kernels import (
    sdf_box,
    sdf_capsule,
    sdf_cone,
    sdf_cylinder,
    sdf_ellipsoid,
    sdf_mesh,
    sdf_sphere,
)
from .sdf_utils import SDFData, sample_sdf_extrapolated
from .types import Axis, GeoType

_POISSON_MAX_ITERS = 512
_POISSON_TOL = 1.0e-5
_MIN_PRESSURE_GRID_DIM = 4
PRESSURE_PROFILE_POISSON = 0
PRESSURE_PROFILE_LAYER = 1


@wp.struct
class PressureFieldData:
    """Per-shape immutable pressure field volume handle."""

    pressure_ptr: wp.uint64
    pressure_max: wp.float32


@wp.kernel
def sample_sdf_grid_kernel(
    sdf_data: wp.array[SDFData],
    sdf_idx: wp.int32,
    lower: wp.vec3,
    voxel_size: wp.vec3,
    dims: wp.vec3i,
    out_sdf: wp.array[wp.float32],
):
    """Sample extrapolated SDF values on a dense regular grid."""
    tid = wp.tid()
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    total = nx * ny * nz
    if tid >= total:
        return

    x = tid % nx
    y = (tid // nx) % ny
    z = tid // (nx * ny)

    p = lower + wp.vec3(float(x) * voxel_size[0], float(y) * voxel_size[1], float(z) * voxel_size[2])
    out_sdf[tid] = sample_sdf_extrapolated(sdf_data[sdf_idx], p)


@wp.kernel
def sample_geometry_sdf_grid_kernel(
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    shape_source_ptr: wp.uint64,
    lower: wp.vec3,
    voxel_size: wp.vec3,
    dims: wp.vec3i,
    max_query_dist: wp.float32,
    out_sdf: wp.array[wp.float32],
):
    """Sample geometry-based signed distances on a dense regular grid."""
    tid = wp.tid()
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    total = nx * ny * nz
    if tid >= total:
        return

    x = tid % nx
    y = (tid // nx) % ny
    z = tid // (nx * ny)
    p = lower + wp.vec3(float(x) * voxel_size[0], float(y) * voxel_size[1], float(z) * voxel_size[2])

    sd = wp.float32(MAXVAL)
    if shape_type == int(GeoType.SPHERE):
        sd = sdf_sphere(p, shape_scale[0])
    elif shape_type == int(GeoType.BOX):
        sd = sdf_box(p, shape_scale[0], shape_scale[1], shape_scale[2])
    elif shape_type == int(GeoType.CAPSULE):
        sd = sdf_capsule(p, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == int(GeoType.CYLINDER):
        sd = sdf_cylinder(p, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == int(GeoType.CONE):
        sd = sdf_cone(p, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == int(GeoType.ELLIPSOID):
        sd = sdf_ellipsoid(p, shape_scale)
    elif shape_type == int(GeoType.MESH) or shape_type == int(GeoType.CONVEX_MESH):
        if shape_source_ptr != wp.uint64(0):
            sd = sdf_mesh(shape_source_ptr, p, max_query_dist)

    out_sdf[tid] = sd


def create_empty_pressure_field_data() -> PressureFieldData:
    """Create an empty pressure field payload."""
    pressure_data = PressureFieldData()
    pressure_data.pressure_ptr = wp.uint64(0)
    pressure_data.pressure_max = 0.0
    return pressure_data


def _entry_field_names(sdf_entry: np.void) -> tuple[str, ...]:
    """Return structured-field names for either legacy or texture-backed SDF entries."""
    names = getattr(getattr(sdf_entry, "dtype", None), "names", None)
    return tuple(names or ())


def _entry_has_field(sdf_entry: np.void, field_name: str) -> bool:
    """Check whether a structured SDF entry exposes a given field."""
    return field_name in _entry_field_names(sdf_entry)


def _entry_center(sdf_entry: np.void) -> np.ndarray:
    """Return SDF center [m] for legacy or texture-backed entries."""
    if _entry_has_field(sdf_entry, "center"):
        return np.asarray(sdf_entry["center"], dtype=np.float32)

    lower = np.asarray(sdf_entry["sdf_box_lower"], dtype=np.float32)
    upper = np.asarray(sdf_entry["sdf_box_upper"], dtype=np.float32)
    return 0.5 * (lower + upper)


def _entry_half_extents(sdf_entry: np.void) -> np.ndarray:
    """Return SDF half extents [m] for legacy or texture-backed entries."""
    if _entry_has_field(sdf_entry, "half_extents"):
        return np.asarray(sdf_entry["half_extents"], dtype=np.float32)

    lower = np.asarray(sdf_entry["sdf_box_lower"], dtype=np.float32)
    upper = np.asarray(sdf_entry["sdf_box_upper"], dtype=np.float32)
    return 0.5 * (upper - lower)


def _entry_voxel_size(sdf_entry: np.void) -> np.ndarray:
    """Return the effective sparse voxel spacing [m] for an SDF entry."""
    if _entry_has_field(sdf_entry, "sparse_voxel_size"):
        return np.asarray(sdf_entry["sparse_voxel_size"], dtype=np.float32)
    return np.asarray(sdf_entry["voxel_size"], dtype=np.float32)


def _entry_has_legacy_volume_data(sdf_entry: np.void) -> bool:
    """Return whether a legacy volume-backed SDF entry has sampleable sparse data."""
    if not _entry_has_field(sdf_entry, "sparse_sdf_ptr"):
        return False
    return int(sdf_entry["sparse_sdf_ptr"]) != 0


def _compute_pressure_grid_spec(sdf_entry: np.void) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute dense-grid sampling parameters for pressure field construction."""
    center = _entry_center(sdf_entry)
    half_extents = _entry_half_extents(sdf_entry)
    lower = center - half_extents
    upper = center + half_extents
    extent = np.maximum(upper - lower, 1.0e-6)

    sparse_voxel_size = _entry_voxel_size(sdf_entry)
    # Warp's load_from_numpy currently expects an isotropic voxel size.
    # Use the max sparse spacing to keep conservative coverage and avoid upsampling artifacts.
    voxel_size = float(np.max(np.maximum(sparse_voxel_size, 1.0e-6)))
    dims = np.rint(extent / voxel_size).astype(np.int32) + 1
    dims = np.maximum(dims, _MIN_PRESSURE_GRID_DIM)
    return lower.astype(np.float32), voxel_size, dims.astype(np.int32)


def _solve_poisson_pressure_extent(
    sdf_grid: np.ndarray,
    voxel_size: np.ndarray,
    max_iters: int = _POISSON_MAX_ITERS,
    tol: float = _POISSON_TOL,
) -> np.ndarray:
    """Solve ``-Laplace(e) = 1`` inside the object with ``e = 0`` on boundary.

    Args:
        sdf_grid: Signed-distance samples on a regular grid.
        voxel_size: Grid spacing [m] as ``(hx, hy, hz)``.
        max_iters: Maximum Jacobi iterations.
        tol: Infinity-norm convergence threshold.

    Returns:
        Normalized extent-like pressure field in ``[0, 1]`` over interior voxels
        and zero outside.
    """
    # Solve directly on the signed-distance interior. Do not apply morphological
    # post-processing; continuity comes from the PDE, not hole-filling.
    inside = sdf_grid < 0.0
    field = np.zeros_like(sdf_grid, dtype=np.float32)
    if not np.any(inside):
        warnings.warn(RuntimeWarning("Poisson pressure solve skipped because the SDF grid has no interior voxels"))
        return field

    padded_outside = np.pad(~inside, 1, mode="constant", constant_values=True)
    touches_outside = (
        padded_outside[:-2, 1:-1, 1:-1]
        | padded_outside[2:, 1:-1, 1:-1]
        | padded_outside[1:-1, :-2, 1:-1]
        | padded_outside[1:-1, 2:, 1:-1]
        | padded_outside[1:-1, 1:-1, :-2]
        | padded_outside[1:-1, 1:-1, 2:]
    )
    boundary = inside & touches_outside
    free = inside & (~boundary)
    if not np.any(free):
        return field

    hx, hy, hz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    cx = 1.0 / max(hx * hx, 1.0e-12)
    cy = 1.0 / max(hy * hy, 1.0e-12)
    cz = 1.0 / max(hz * hz, 1.0e-12)
    denom = 2.0 * (cx + cy + cz)

    next_field = np.zeros_like(field)
    neighbor_sum = np.zeros_like(field)

    delta = float("inf")
    iteration = -1
    for iteration in range(max_iters):
        work = np.where(inside, field, 0.0)
        neighbor_sum.fill(0.0)
        neighbor_sum[1:, :, :] += cx * work[:-1, :, :]
        neighbor_sum[:-1, :, :] += cx * work[1:, :, :]
        neighbor_sum[:, 1:, :] += cy * work[:, :-1, :]
        neighbor_sum[:, :-1, :] += cy * work[:, 1:, :]
        neighbor_sum[:, :, 1:] += cz * work[:, :, :-1]
        neighbor_sum[:, :, :-1] += cz * work[:, :, 1:]

        next_field.fill(0.0)
        next_field[free] = (neighbor_sum[free] + 1.0) / denom

        delta = float(np.max(np.abs(next_field[free] - field[free])))
        field, next_field = next_field, field
        if delta <= tol:
            break

    if iteration == max_iters - 1 and delta > tol * 10.0:
        warnings.warn(
            RuntimeWarning(
                f"Poisson pressure solve reached max_iters={max_iters} with residual {delta:.6g} "
                f"above tolerance {tol:.6g}"
            )
        )

    max_val = float(np.max(field[inside]))
    if max_val > 1.0e-8:
        field[inside] /= max_val
    else:
        field[inside] = 0.0
    field[~inside] = 0.0
    return field


def _solve_layer_pressure_extent(
    sdf_grid: np.ndarray,
    voxel_size: np.ndarray,
    layer_params: np.ndarray | tuple[float, float, float, float] | list[float] | None = None,
) -> np.ndarray:
    """Build a one-sided local-Z layer pressure extent from an SDF grid.

    The field approximates compression of vertical foam columns. For each
    ``(x, y)`` column, pressure is zero at the top interior sample and increases
    with normalized depth toward the local bottom. This is intended for layer-like
    objects such as shoe midsoles under top-down loading, not generic solids.

    Args:
        sdf_grid: Signed-distance samples on a regular grid.
        voxel_size: Grid spacing [m] as ``(hx, hy, hz)``.
        layer_params: ``(eta_lock, q, cubic, quintic)``. ``eta_lock`` is the
            normalized densification lock-up strain, ``q`` is the densification
            exponent, and ``cubic``/``quintic`` are smooth polynomial stiffening
            terms.

    Returns:
        Distance-like pressure extent over interior voxels and zero outside.
    """
    params = np.asarray(
        layer_params if layer_params is not None else (0.65, 0.0, 0.0, 0.0),
        dtype=np.float32,
    ).reshape(-1)
    if params.shape[0] != 4:
        raise ValueError(f"Expected 4 layer pressure parameters, got {params.shape[0]}")

    eta_lock_raw = float(params[0])
    if eta_lock_raw < 0.01 or eta_lock_raw > 0.98:
        warnings.warn(
            RuntimeWarning(
                f"Layer pressure eta_lock={eta_lock_raw:.6g} is near its stable limits; expected roughly [0.01, 0.98]"
            )
        )
    eta_lock = float(np.clip(eta_lock_raw, 1.0e-3, 0.999))
    densification_power = float(max(params[1], 0.0))
    cubic = float(params[2])
    quintic = float(params[3])

    inside = sdf_grid < 0.0
    field = np.zeros_like(sdf_grid, dtype=np.float32)
    if not np.any(inside):
        return field

    dz = float(voxel_size[2])
    if dz <= 0.0:
        raise ValueError(f"Layer pressure thickness must be positive, got voxel_size[2]={dz}")
    nx, ny, _nz = sdf_grid.shape
    z_indices = np.arange(sdf_grid.shape[2], dtype=np.float32)

    for ix in range(nx):
        for iy in range(ny):
            column_inside = inside[ix, iy, :]
            if not np.any(column_inside):
                continue

            column_z = z_indices[column_inside]
            z_bottom = float(np.min(column_z))
            z_top = float(np.max(column_z))
            thickness = max((z_top - z_bottom) * dz, dz)
            eta = ((z_top - column_z) * dz / thickness).astype(np.float32)
            eta_eval = np.minimum(eta, np.float32(eta_lock * 0.999))
            phi = eta_eval + np.float32(cubic) * eta_eval**3 + np.float32(quintic) * eta_eval**5
            if densification_power > 0.0:
                remaining = np.maximum(1.0 - eta_eval / np.float32(eta_lock), 1.0e-3)
                phi = phi / np.power(remaining, np.float32(densification_power))
            field[ix, iy, column_inside] = np.maximum(phi, 0.0) * np.float32(thickness)

    field[~inside] = 0.0
    return field


def _normalize_axis_triplet(
    value: np.ndarray | tuple[float, float, float] | list[float] | None,
    default: tuple[float, float, float],
) -> np.ndarray:
    """Convert axis-triplet input to float32 numpy array."""
    if value is None:
        return np.asarray(default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Expected 3 values, got {arr.shape[0]}")
    return arr


def _apply_pressure_axis_sine_modulation_to_grid(
    pressure_grid: np.ndarray,
    inside: np.ndarray,
    lower: np.ndarray,
    voxel_size_scalar: float,
    center: np.ndarray,
    half_extents: np.ndarray,
    amplitude: np.ndarray,
    cycles: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Apply per-axis sine modulation to a dense pressure grid."""
    if not np.any(inside):
        return pressure_grid

    out = pressure_grid.astype(np.float32, copy=True)
    two_pi = np.float32(2.0 * np.pi)
    nx, ny, nz = out.shape
    axis_sizes = (nx, ny, nz)

    for axis in range(3):
        amp = float(amplitude[axis])
        if abs(amp) <= 1.0e-8:
            continue
        cyc = float(max(cycles[axis], 1.0e-6))
        ph = float(phase[axis])
        count = axis_sizes[axis]
        coords = lower[axis] + np.arange(count, dtype=np.float32) * np.float32(voxel_size_scalar)
        half_extent = float(max(abs(half_extents[axis]), 1.0e-8))
        coord_norm = (coords - float(center[axis])) / half_extent
        coord01 = 0.5 * (coord_norm + 1.0)
        wave = np.sin(two_pi * np.float32(cyc) * coord01 + np.float32(ph)).astype(np.float32)
        modulation = np.maximum(1.0 + np.float32(amp) * wave, 0.0).astype(np.float32)

        if axis == 0:
            out *= modulation[:, None, None]
        elif axis == 1:
            out *= modulation[None, :, None]
        else:
            out *= modulation[None, None, :]

    out[~inside] = 0.0
    return out


def _is_geometry_sampling_supported(shape_type: int, shape_source_ptr: int) -> bool:
    """Return whether pressure construction can sample geometry directly."""
    if shape_type in {
        int(GeoType.SPHERE),
        int(GeoType.BOX),
        int(GeoType.CAPSULE),
        int(GeoType.CYLINDER),
        int(GeoType.CONE),
        int(GeoType.ELLIPSOID),
    }:
        return True
    if shape_type in {int(GeoType.MESH), int(GeoType.CONVEX_MESH)}:
        return shape_source_ptr != 0
    return False


def build_immutable_pressure_field_from_sdf(
    sdf_data,
    sdf_entry: np.void,
    sdf_idx: int,
    device: Devicelike,
    shape_type: int | None = None,
    shape_scale: np.ndarray | None = None,
    shape_source_ptr: int | None = None,
    pressure_sine_amplitude: np.ndarray | tuple[float, float, float] | None = None,
    pressure_sine_cycles: np.ndarray | tuple[float, float, float] | None = None,
    pressure_sine_phase: np.ndarray | tuple[float, float, float] | None = None,
    pressure_profile: int = PRESSURE_PROFILE_POISSON,
    pressure_layer_params: np.ndarray | tuple[float, float, float, float] | None = None,
) -> tuple[PressureFieldData, wp.Volume | None]:
    """Build immutable pressure field for one compact SDF entry."""
    if (not _entry_has_legacy_volume_data(sdf_entry)) and (
        shape_type is None or shape_scale is None or shape_source_ptr is None
    ):
        return create_empty_pressure_field_data(), None

    if _entry_has_field(sdf_entry, "sparse_sdf_ptr") and int(sdf_entry["sparse_sdf_ptr"]) == 0:
        return create_empty_pressure_field_data(), None

    lower, voxel_size_scalar, dims = _compute_pressure_grid_spec(sdf_entry)
    voxel_size = np.array([voxel_size_scalar, voxel_size_scalar, voxel_size_scalar], dtype=np.float32)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    total = nx * ny * nz

    dense_sdf = wp.empty(total, dtype=wp.float32, device=device)
    use_geometry_sampling = (
        shape_type is not None
        and shape_scale is not None
        and shape_source_ptr is not None
        and _is_geometry_sampling_supported(shape_type, int(shape_source_ptr))
    )

    if use_geometry_sampling:
        shape_scale_arr = np.asarray(shape_scale, dtype=np.float32)
        if shape_scale_arr.shape != (3,):
            shape_scale_arr = np.ones(3, dtype=np.float32)
        half_extents = _entry_half_extents(sdf_entry)
        max_query_dist = float(np.linalg.norm(half_extents) + 4.0 * voxel_size_scalar)
        wp.launch(
            kernel=sample_geometry_sdf_grid_kernel,
            dim=total,
            inputs=[
                wp.int32(shape_type),
                wp.vec3(float(shape_scale_arr[0]), float(shape_scale_arr[1]), float(shape_scale_arr[2])),
                wp.uint64(int(shape_source_ptr)),
                wp.vec3(float(lower[0]), float(lower[1]), float(lower[2])),
                wp.vec3(float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])),
                wp.vec3i(nx, ny, nz),
                wp.float32(max_query_dist),
            ],
            outputs=[dense_sdf],
            device=device,
        )
    else:
        wp.launch(
            kernel=sample_sdf_grid_kernel,
            dim=total,
            inputs=[
                sdf_data,
                wp.int32(sdf_idx),
                wp.vec3(float(lower[0]), float(lower[1]), float(lower[2])),
                wp.vec3(float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])),
                wp.vec3i(nx, ny, nz),
            ],
            outputs=[dense_sdf],
            device=device,
        )

    sdf_grid_zyx = dense_sdf.numpy().reshape((nz, ny, nx))
    # Warp load_from_numpy expects axis order (x, y, z).
    sdf_grid = np.transpose(sdf_grid_zyx, (2, 1, 0))
    if int(pressure_profile) == PRESSURE_PROFILE_LAYER:
        pressure_grid = _solve_layer_pressure_extent(sdf_grid, voxel_size, pressure_layer_params)
    else:
        pressure_grid = _solve_poisson_pressure_extent(sdf_grid, voxel_size)
    inside = sdf_grid < 0.0
    if np.any(inside) and int(pressure_profile) != PRESSURE_PROFILE_LAYER:
        # Convert normalized extent back to a distance-like scale so existing
        # hydroelastic stiffness calibration (k * depth * area) remains coherent.
        depth_scale = float(np.max(-sdf_grid[inside]))
        pressure_grid[inside] *= depth_scale

    amplitude = _normalize_axis_triplet(pressure_sine_amplitude, (0.0, 0.0, 0.0))
    cycles = _normalize_axis_triplet(pressure_sine_cycles, (1.0, 1.0, 1.0))
    phase = _normalize_axis_triplet(pressure_sine_phase, (0.0, 0.0, 0.0))
    if np.any(np.abs(amplitude) > 1.0e-8):
        center = _entry_center(sdf_entry)
        half_extents = _entry_half_extents(sdf_entry)
        pressure_grid = _apply_pressure_axis_sine_modulation_to_grid(
            pressure_grid,
            inside,
            lower,
            voxel_size_scalar,
            center,
            half_extents,
            amplitude,
            cycles,
            phase,
        )

    pressure_volume = wp.Volume.load_from_numpy(
        pressure_grid.astype(np.float32),
        min_world=(float(lower[0]), float(lower[1]), float(lower[2])),
        voxel_size=float(voxel_size_scalar),
        bg_value=0.0,
        device=device,
    )

    pressure_data = PressureFieldData()
    pressure_data.pressure_ptr = pressure_volume.id
    pressure_data.pressure_max = float(np.max(pressure_grid)) if pressure_grid.size > 0 else 0.0
    return pressure_data, pressure_volume
