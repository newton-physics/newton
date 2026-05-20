# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable vertical elastic-foundation model for Digital Instron v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf


@dataclass(frozen=True, init=False)
class FoundationMaterial:
    """Shared vertical foundation material parameters."""

    stiffness_pa: float
    ogden_alpha: float
    lock_strain: float
    damping_pa_s: float
    damping_power: float = 1.0
    per_cylinder_area: bool = False
    prony_stiffness_pa: float = 0.0
    prony_damping_pa_s: float = 0.0
    state_warmup_cycles: int = 0
    pasternak_stiffness_n_per_m: float = 0.0

    def __init__(
        self,
        stiffness_pa: float,
        ogden_alpha: float,
        lock_strain: float,
        damping_pa_s: float,
        damping_power: float = 1.0,
        per_cylinder_area: bool = False,
        prony_stiffness_pa: float = 0.0,
        prony_damping_pa_s: float = 0.0,
        state_warmup_cycles: int = 0,
        pasternak_stiffness_n_per_m: float = 0.0,
        shear_modulus_pa: float | None = None,
    ):
        if shear_modulus_pa is not None:
            pasternak_stiffness_n_per_m = float(shear_modulus_pa)
        object.__setattr__(self, "stiffness_pa", float(stiffness_pa))
        object.__setattr__(self, "ogden_alpha", float(ogden_alpha))
        object.__setattr__(self, "lock_strain", float(lock_strain))
        object.__setattr__(self, "damping_pa_s", float(damping_pa_s))
        object.__setattr__(self, "damping_power", float(damping_power))
        object.__setattr__(self, "per_cylinder_area", bool(per_cylinder_area))
        object.__setattr__(self, "prony_stiffness_pa", float(prony_stiffness_pa))
        object.__setattr__(self, "prony_damping_pa_s", float(prony_damping_pa_s))
        object.__setattr__(self, "state_warmup_cycles", int(state_warmup_cycles))
        object.__setattr__(self, "pasternak_stiffness_n_per_m", float(pasternak_stiffness_n_per_m))

    @property
    def shear_modulus_pa(self) -> float:
        """Deprecated alias for ``pasternak_stiffness_n_per_m``."""

        return self.pasternak_stiffness_n_per_m


@dataclass(frozen=True)
class FoundationResult:
    """Net force, wrench, and scalar loss from one replay evaluation."""

    force_n: float
    wrench: np.ndarray
    loss: float


@dataclass(frozen=True)
class FoundationGradientResult:
    """Forward result plus ``d loss / d material``."""

    force_n: float
    wrench: np.ndarray
    loss: float
    gradient: np.ndarray


@dataclass(frozen=True)
class FoundationFitSample:
    """One force sample for shared material fitting."""

    current_length_m: np.ndarray
    slack_length_m: np.ndarray
    velocity_mps: np.ndarray
    measured_force_n: float
    weight: float = 1.0
    cell_area_m2: np.ndarray | None = None
    neighbors: np.ndarray | None = None
    spacing_m: float | None = None


@dataclass(frozen=True)
class FoundationTrialBatch:
    """One averaged-cycle trial batch for GPU material fitting."""

    name: str
    current_length_m: np.ndarray
    slack_length_m: np.ndarray
    velocity_mps: np.ndarray
    measured_force_n: np.ndarray
    sample_weight: np.ndarray
    cell_area_m2: np.ndarray
    time_s: np.ndarray
    dt_s: np.ndarray
    displacement_m: np.ndarray
    phase: tuple[str, ...]
    force_zero_n: float = 0.0
    neighbors: np.ndarray | None = None
    spacing_m: float | None = None


@dataclass(frozen=True)
class FoundationTrialBatchResult:
    """Batched force/loss result for one trial."""

    trial: str
    predicted_force_n: np.ndarray
    loss: float
    gradient: np.ndarray


@dataclass(frozen=True)
class FoundationFitResult:
    """Result of the first v2 autodiff material fit."""

    material: FoundationMaterial
    history: tuple[dict[str, float], ...]


@wp.kernel
def _foundation_kernel(
    compression_m: wp.array[float],
    velocity_mps: wp.array[float],
    xy_m: wp.array[wp.vec2],
    area_m2: float,
    thickness_m: float,
    params: wp.array[float],
    measured_force_n: float,
    neighbors: wp.array2d[int],
    spacing_m: float,
    force_out: wp.array[float],
    wrench_out: wp.array[float],
    loss_out: wp.array[float],
):
    i = wp.tid()
    comp = wp.max(compression_m[i], 0.0)
    strain = comp / thickness_m
    lock = wp.max(params[2], 1.0e-4)
    normalized = wp.min(strain / lock, 0.999)
    alpha = wp.max(params[1], 1.0e-4)
    ogden_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha

    h2 = spacing_m * spacing_m
    laplacian = float(0.0)
    if h2 > 1.0e-12:
        n_left = neighbors[i, 0]
        val_left = comp
        if n_left != -1:
            val_left = wp.max(compression_m[n_left], 0.0)
        n_right = neighbors[i, 1]
        val_right = comp
        if n_right != -1:
            val_right = wp.max(compression_m[n_right], 0.0)
        n_bottom = neighbors[i, 2]
        val_bottom = comp
        if n_bottom != -1:
            val_bottom = wp.max(compression_m[n_bottom], 0.0)
        n_top = neighbors[i, 3]
        val_top = comp
        if n_top != -1:
            val_top = wp.max(compression_m[n_top], 0.0)
        laplacian = (val_left + val_right + val_bottom + val_top - 4.0 * comp) / h2

    elastic_stress = ogden_stress - params[7] * laplacian
    damping_strain = wp.max(strain, 1.0e-8)
    damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))
    viscous_stress = params[3] * damping_weight * velocity_mps[i]
    fz = area_m2 * wp.max(elastic_stress + viscous_stress, 0.0)
    xy = xy_m[i]
    wp.atomic_add(force_out, 0, fz)
    wp.atomic_add(wrench_out, 2, fz)
    wp.atomic_add(wrench_out, 3, xy[1] * fz)
    wp.atomic_add(wrench_out, 4, -xy[0] * fz)
    wp.atomic_add(loss_out, 0, 0.0)

    # Loss is filled by every lane with the same expression after the force
    # atomic. This keeps the force path differentiable without a second kernel.
    wp.atomic_add(loss_out, 0, 0.0 * measured_force_n)


@wp.kernel
def _foundation_lengths_kernel(
    current_length_m: wp.array[float],
    slack_length_m: wp.array[float],
    velocity_mps: wp.array[float],
    xy_m: wp.array[wp.vec2],
    cell_area_m2: wp.array[float],
    params: wp.array[float],
    neighbors: wp.array2d[int],
    spacing_m: float,
    force_out: wp.array[float],
    wrench_out: wp.array[float],
):
    i = wp.tid()
    slack = wp.max(slack_length_m[i], 1.0e-6)
    comp = wp.max(slack - current_length_m[i], 0.0)
    strain = comp / slack
    lock = wp.max(params[2], 1.0e-4)
    normalized = wp.min(strain / lock, 0.999)
    alpha = wp.max(params[1], 1.0e-4)
    ogden_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha

    h2 = spacing_m * spacing_m
    laplacian = float(0.0)
    if h2 > 1.0e-12:
        n_left = neighbors[i, 0]
        val_left = comp
        if n_left != -1:
            val_left = wp.max(slack_length_m[n_left] - current_length_m[n_left], 0.0)
        n_right = neighbors[i, 1]
        val_right = comp
        if n_right != -1:
            val_right = wp.max(slack_length_m[n_right] - current_length_m[n_right], 0.0)
        n_bottom = neighbors[i, 2]
        val_bottom = comp
        if n_bottom != -1:
            val_bottom = wp.max(slack_length_m[n_bottom] - current_length_m[n_bottom], 0.0)
        n_top = neighbors[i, 3]
        val_top = comp
        if n_top != -1:
            val_top = wp.max(slack_length_m[n_top] - current_length_m[n_top], 0.0)
        laplacian = (val_left + val_right + val_bottom + val_top - 4.0 * comp) / h2

    elastic_stress = ogden_stress - params[7] * laplacian
    damping_strain = wp.max(strain, 1.0e-8)
    damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))
    compression_velocity = -velocity_mps[i]
    viscous_stress = params[3] * damping_weight * compression_velocity
    fz = cell_area_m2[i] * wp.max(elastic_stress + viscous_stress, 0.0)
    xy = xy_m[i]
    wp.atomic_add(force_out, 0, fz)
    wp.atomic_add(wrench_out, 2, fz)
    wp.atomic_add(wrench_out, 3, xy[1] * fz)
    wp.atomic_add(wrench_out, 4, -xy[0] * fz)


@wp.func
def get_sdf_deflection(
    j: int,
    xy_m: wp.array[wp.vec2],
    top_z_m: wp.array[float],
    indenter_sdf: TextureSDFData,
    indenter_inv_transform: wp.transform,
    missing_value: float,
) -> float:
    if j == -1:
        return missing_value
    xy = xy_m[j]
    world_pt = wp.vec3(xy[0], xy[1], top_z_m[j])
    local_pt = wp.transform_point(indenter_inv_transform, world_pt)
    sdf_val = texture_sample_sdf(indenter_sdf, local_pt)
    return wp.max(-sdf_val, 0.0)


@wp.kernel
def _foundation_sdf_kernel(
    top_z_m: wp.array[float],
    slack_length_m: wp.array[float],
    velocity_mps: wp.array[float],
    xy_m: wp.array[wp.vec2],
    cell_area_m2: wp.array[float],
    indenter_sdf: TextureSDFData,
    indenter_pos: wp.vec3,
    indenter_quat: wp.quat,
    params: wp.array[float],
    neighbors: wp.array2d[int],
    spacing_m: float,
    force_out: wp.array[float],
    wrench_out: wp.array[float],
):
    i = wp.tid()
    xy = xy_m[i]
    world_pt = wp.vec3(xy[0], xy[1], top_z_m[i])

    indenter_inv_pos = wp.quat_rotate_inv(indenter_quat, -indenter_pos)
    indenter_inv_quat = wp.quat_inverse(indenter_quat)
    indenter_inv_transform = wp.transform(indenter_inv_pos, indenter_inv_quat)
    local_pt = wp.transform_point(indenter_inv_transform, world_pt)

    sdf_val = texture_sample_sdf(indenter_sdf, local_pt)
    comp = wp.max(-sdf_val, 0.0)
    slack = wp.max(slack_length_m[i], 1.0e-6)
    strain = comp / slack
    lock = wp.max(params[2], 1.0e-4)
    normalized = wp.min(strain / lock, 0.999)
    alpha = wp.max(params[1], 1.0e-4)
    ogden_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha

    h2 = spacing_m * spacing_m
    laplacian = float(0.0)
    if h2 > 1.0e-12:
        val_left = get_sdf_deflection(neighbors[i, 0], xy_m, top_z_m, indenter_sdf, indenter_inv_transform, comp)
        val_right = get_sdf_deflection(neighbors[i, 1], xy_m, top_z_m, indenter_sdf, indenter_inv_transform, comp)
        val_bottom = get_sdf_deflection(neighbors[i, 2], xy_m, top_z_m, indenter_sdf, indenter_inv_transform, comp)
        val_top = get_sdf_deflection(neighbors[i, 3], xy_m, top_z_m, indenter_sdf, indenter_inv_transform, comp)
        laplacian = (val_left + val_right + val_bottom + val_top - 4.0 * comp) / h2

    elastic_stress = ogden_stress - params[7] * laplacian
    damping_strain = wp.max(strain, 1.0e-8)
    damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))
    compression_velocity = -velocity_mps[i]
    viscous_stress = params[3] * damping_weight * compression_velocity
    fz = cell_area_m2[i] * wp.max(elastic_stress + viscous_stress, 0.0)

    wp.atomic_add(force_out, 0, fz)
    wp.atomic_add(wrench_out, 2, fz)
    wp.atomic_add(wrench_out, 3, xy[1] * fz)
    wp.atomic_add(wrench_out, 4, -xy[0] * fz)


@wp.kernel
def _loss_kernel(force_out: wp.array[float], measured_force_n: float, loss_out: wp.array[float]):
    residual = force_out[0] - measured_force_n
    loss_out[0] = 0.5 * residual * residual


@wp.kernel
def _foundation_lengths_batch_kernel(
    current_length_m: wp.array[float],
    slack_length_m: wp.array[float],
    velocity_mps: wp.array[float],
    xy_m: wp.array[wp.vec2],
    cell_area_m2: wp.array[float],
    params: wp.array[float],
    neighbors: wp.array2d[int],
    spacing_m: float,
    spring_count: int,
    force_out: wp.array[float],
):
    tid = wp.tid()
    spring = tid - (tid / spring_count) * spring_count
    slack = wp.max(slack_length_m[spring], 1.0e-6)
    comp = wp.max(slack - current_length_m[tid], 0.0)
    strain = comp / slack
    lock = wp.max(params[2], 1.0e-4)
    normalized = wp.min(strain / lock, 0.999)
    alpha = wp.max(params[1], 1.0e-4)
    ogden_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha

    h2 = spacing_m * spacing_m
    laplacian = float(0.0)
    if h2 > 1.0e-12:
        n_left = neighbors[spring, 0]
        val_left = comp
        if n_left != -1:
            val_left = wp.max(slack_length_m[n_left] - current_length_m[(tid / spring_count) * spring_count + n_left], 0.0)
        n_right = neighbors[spring, 1]
        val_right = comp
        if n_right != -1:
            val_right = wp.max(slack_length_m[n_right] - current_length_m[(tid / spring_count) * spring_count + n_right], 0.0)
        n_bottom = neighbors[spring, 2]
        val_bottom = comp
        if n_bottom != -1:
            val_bottom = wp.max(slack_length_m[n_bottom] - current_length_m[(tid / spring_count) * spring_count + n_bottom], 0.0)
        n_top = neighbors[spring, 3]
        val_top = comp
        if n_top != -1:
            val_top = wp.max(slack_length_m[n_top] - current_length_m[(tid / spring_count) * spring_count + n_top], 0.0)
        laplacian = (val_left + val_right + val_bottom + val_top - 4.0 * comp) / h2

    elastic_stress = ogden_stress - params[7] * laplacian
    damping_strain = wp.max(strain, 1.0e-8)
    damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))
    compression_velocity = -velocity_mps[tid]
    viscous_stress = params[3] * damping_weight * compression_velocity
    fz = cell_area_m2[spring] * wp.max(elastic_stress + viscous_stress, 0.0)
    frame = tid / spring_count
    wp.atomic_add(force_out, frame, fz)


@wp.kernel
def _foundation_lengths_stateful_batch_kernel(
    current_length_m: wp.array[float],
    slack_length_m: wp.array[float],
    velocity_mps: wp.array[float],
    dt_s: wp.array[float],
    xy_m: wp.array[wp.vec2],
    cell_area_m2: wp.array[float],
    params: wp.array[float],
    neighbors: wp.array2d[int],
    spacing_m: float,
    frame_count: int,
    spring_count: int,
    warmup_cycles: int,
    state_history: wp.array2d[float],
    force_out: wp.array[float],
):
    spring = wp.tid()
    slack = wp.max(slack_length_m[spring], 1.0e-6)
    total_cycles = warmup_cycles + 1
    ep = wp.max(params[5], 0.0)
    e0 = wp.max(params[0], 1.0e-4)
    beta = wp.min(ep / e0, 0.99)
    etap = wp.max(params[6], 0.0)
    tau = wp.max(etap / wp.max(ep, 1.0e-6), 1.0e-6)

    for cycle in range(total_cycles):
        for frame in range(frame_count):
            step = cycle * frame_count + frame
            offset = frame * spring_count + spring
            comp = wp.max(slack - current_length_m[offset], 0.0)

            # 1. Compute Ogden elastic stress
            strain = comp / slack
            lock = wp.max(params[2], 1.0e-4)
            normalized = wp.min(strain / lock, 0.999)
            ogden_alpha = wp.max(params[1], 1.0e-4)
            ogden_stress = params[0] * (wp.pow(1.0 - normalized, -ogden_alpha) - 1.0) / ogden_alpha

            # 2. Compute Pasternak Laplacian and shear stress
            h2 = spacing_m * spacing_m
            laplacian = float(0.0)
            if h2 > 1.0e-12:
                n_left = neighbors[spring, 0]
                val_left = comp
                if n_left != -1:
                    val_left = wp.max(slack_length_m[n_left] - current_length_m[frame * spring_count + n_left], 0.0)
                n_right = neighbors[spring, 1]
                val_right = comp
                if n_right != -1:
                    val_right = wp.max(slack_length_m[n_right] - current_length_m[frame * spring_count + n_right], 0.0)
                n_bottom = neighbors[spring, 2]
                val_bottom = comp
                if n_bottom != -1:
                    val_bottom = wp.max(slack_length_m[n_bottom] - current_length_m[frame * spring_count + n_bottom], 0.0)
                n_top = neighbors[spring, 3]
                val_top = comp
                if n_top != -1:
                    val_top = wp.max(slack_length_m[n_top] - current_length_m[frame * spring_count + n_top], 0.0)
                laplacian = (val_left + val_right + val_bottom + val_top - 4.0 * comp) / h2

            elastic_stress_total = ogden_stress - params[7] * laplacian

            # 3. Update QLV Prony Series State
            decay = wp.exp(-wp.max(dt_s[frame], 0.0) / tau)
            prev_state = float(0.0)
            if step > 0:
                prev_state = state_history[spring, step - 1]
            curr_state = prev_state * decay + (1.0 - decay) * beta * elastic_stress_total
            state_history[spring, step] = curr_state

            # 4. viscoelastic stress (only aggregate force during the last cycle)
            if cycle == warmup_cycles:
                viscoelastic_stress = elastic_stress_total - curr_state

                damping_strain = wp.max(strain, 1.0e-8)
                damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))
                compression_velocity = -velocity_mps[offset]
                viscous_stress = params[3] * damping_weight * compression_velocity
                fz = cell_area_m2[spring] * wp.max(viscoelastic_stress + viscous_stress, 0.0)
                wp.atomic_add(force_out, frame, fz)


@wp.kernel
def _weighted_normalized_loss_kernel(
    force_out: wp.array[float],
    measured_force_n: wp.array[float],
    sample_weight: wp.array[float],
    force_scale_n: float,
    loss_out: wp.array[float],
):
    frame = wp.tid()
    scale = wp.max(force_scale_n, 1.0)
    residual = (force_out[frame] - measured_force_n[frame]) / scale
    wp.atomic_add(loss_out, 0, 0.5 * sample_weight[frame] * residual * residual)


@wp.kernel
def _add_loop_loss_kernel(
    force_out: wp.array[float],
    displacement_m: wp.array[float],
    measured_loop: float,
    force_zero: float,
    loop_weight: float,
    loss_out: wp.array[float],
    frame_count: int,
):
    if wp.tid() == 0:
        pred_loop = float(0.0)
        for i in range(frame_count - 1):
            f_i = force_out[i] + force_zero
            f_next = force_out[i + 1] + force_zero
            dx = displacement_m[i + 1] - displacement_m[i]
            pred_loop += 0.5 * (f_i + f_next) * dx
        
        meas_loop = wp.max(measured_loop, 1.0e-3)
        loop_residual = (pred_loop - meas_loop) / meas_loop
        loss_val = 0.5 * loop_weight * loop_residual * loop_residual
        wp.atomic_add(loss_out, 0, loss_val)


def infer_spacing(xy_m: np.ndarray) -> float:
    if len(xy_m) <= 1:
        return 0.0
    # Just take the first point and find the minimum distance to other points
    diff = xy_m[1:] - xy_m[0]
    dists = np.linalg.norm(diff, axis=1)
    dists = dists[dists > 1e-6]
    if len(dists) == 0:
        return 0.0
    return float(np.min(dists))


def _prepare_neighbors_and_spacing(
    xy_m: np.ndarray,
    device: str | wp.context.Device | None,
    neighbors: np.ndarray | None = None,
    spacing_m: float | None = None,
) -> tuple[wp.array, float]:
    from .geometry import compute_grid_neighbors

    xy = np.asarray(xy_m)
    h = infer_spacing(xy) if spacing_m is None else float(spacing_m)
    if h < 0.0:
        raise ValueError("spacing_m must be non-negative")
    if neighbors is None:
        neighbor_array = compute_grid_neighbors(xy, h)
    else:
        neighbor_array = np.asarray(neighbors, dtype=np.int32)
        if neighbor_array.shape != (xy.shape[0], 4):
            raise ValueError("neighbors must have shape (len(xy_m), 4)")
    neighbors_wp = wp.array(neighbor_array, dtype=wp.int32, device=device)
    return neighbors_wp, h


def _as_vec2_array(xy_m: np.ndarray, device: str | wp.context.Device | None, requires_grad: bool) -> wp.array:
    xy = np.asarray(xy_m, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_m must have shape (n, 2)")
    return wp.array(
        [wp.vec2(float(x), float(y)) for x, y in xy], dtype=wp.vec2, device=device, requires_grad=requires_grad
    )


def _cell_area_array(cell_area_m2: np.ndarray | float, count: int) -> np.ndarray:
    if isinstance(cell_area_m2, (int, float)):
        return np.full(count, float(cell_area_m2), dtype=np.float32)
    cell_array = np.asarray(cell_area_m2, dtype=np.float32)
    if cell_array.shape != (count,):
        raise ValueError("cell_area_m2 must be a scalar or a 1D array matching spring count")
    return cell_array


def evaluate_foundation(
    xy_m: np.ndarray,
    compression_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    cell_area_m2: np.ndarray | float,
    thickness_m: float,
    material: FoundationMaterial,
    measured_force_n: float = 0.0,
    neighbors: np.ndarray | None = None,
    spacing_m: float | None = None,
    device: str | wp.context.Device | None = "cpu",
) -> FoundationResult:
    """Evaluate the differentiable foundation replay for one frame."""

    if thickness_m <= 0.0:
        raise ValueError("thickness_m must be positive")
    compression = np.asarray(compression_m, dtype=np.float32)
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    if compression.shape != velocity.shape:
        raise ValueError("compression_m and velocity_mps must have the same shape")
    if compression.ndim != 1 or compression.shape[0] != np.asarray(xy_m).shape[0]:
        raise ValueError("compression_m must be a 1D array matching xy_m rows")

    wp.init()
    wp_compression = wp.array(compression, dtype=float, device=device)
    wp_velocity = wp.array(velocity, dtype=float, device=device)
    wp_xy = _as_vec2_array(xy_m, device, requires_grad=False)
    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
    )
    neighbors_wp, h = _prepare_neighbors_and_spacing(xy_m, device, neighbors, spacing_m)
    force_out = wp.zeros(1, dtype=float, device=device)
    wrench_out = wp.zeros(6, dtype=float, device=device)
    loss_out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _foundation_kernel,
        dim=compression.shape[0],
        inputs=[
            wp_compression,
            wp_velocity,
            wp_xy,
            float(cell_area_m2),
            float(thickness_m),
            wp_params,
            float(measured_force_n),
            neighbors_wp,
            h,
            force_out,
            wrench_out,
            loss_out,
        ],
        device=device,
    )
    wp.launch(_loss_kernel, dim=1, inputs=[force_out, float(measured_force_n), loss_out], device=device)
    return FoundationResult(
        force_n=float(force_out.numpy()[0]),
        wrench=wrench_out.numpy().astype(np.float64),
        loss=float(loss_out.numpy()[0]),
    )


def evaluate_foundation_lengths(
    xy_m: np.ndarray,
    current_length_m: np.ndarray,
    slack_length_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    cell_area_m2: np.ndarray | float,
    material: FoundationMaterial,
    measured_force_n: float = 0.0,
    neighbors: np.ndarray | None = None,
    spacing_m: float | None = None,
    device: str | wp.context.Device | None = "cpu",
) -> FoundationResult:
    """Evaluate springs from current length relative to raycast slack length."""

    current = np.asarray(current_length_m, dtype=np.float32)
    slack = np.asarray(slack_length_m, dtype=np.float32)
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    if current.shape != slack.shape or current.shape != velocity.shape:
        raise ValueError("current_length_m, slack_length_m, and velocity_mps must have the same shape")
    if current.ndim != 1 or current.shape[0] != np.asarray(xy_m).shape[0]:
        raise ValueError("spring length arrays must be 1D and match xy_m rows")
    if np.any(slack <= 0.0):
        raise ValueError("slack_length_m values must be positive")

    wp.init()
    cell_array = _cell_area_array(cell_area_m2, current.shape[0])
    wp_cell_area = wp.array(cell_array, dtype=float, device=device)
    wp_current = wp.array(current, dtype=float, device=device)
    wp_slack = wp.array(slack, dtype=float, device=device)
    wp_velocity = wp.array(velocity, dtype=float, device=device)
    wp_xy = _as_vec2_array(xy_m, device, requires_grad=False)
    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
    )
    neighbors_wp, h = _prepare_neighbors_and_spacing(xy_m, device, neighbors, spacing_m)
    force_out = wp.zeros(1, dtype=float, device=device)
    wrench_out = wp.zeros(6, dtype=float, device=device)
    wp.launch(
        _foundation_lengths_kernel,
        dim=current.shape[0],
        inputs=[
            wp_current,
            wp_slack,
            wp_velocity,
            wp_xy,
            wp_cell_area,
            wp_params,
            neighbors_wp,
            h,
            force_out,
            wrench_out,
        ],
        device=device,
    )
    force_n = float(force_out.numpy()[0])
    residual = force_n - float(measured_force_n)
    return FoundationResult(
        force_n=force_n,
        wrench=wrench_out.numpy().astype(np.float64),
        loss=0.5 * residual * residual,
    )


def evaluate_foundation_sdf(
    xy_m: np.ndarray,
    top_z_m: np.ndarray,
    *,
    slack_length_m: np.ndarray,
    velocity_mps: np.ndarray,
    cell_area_m2: np.ndarray | float,
    material: FoundationMaterial,
    indenter_sdf: TextureSDFData,
    indenter_pos: wp.vec3 | tuple[float, float, float],
    indenter_quat: wp.quat | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    measured_force_n: float = 0.0,
    neighbors: np.ndarray | None = None,
    spacing_m: float | None = None,
    device: str | wp.context.Device | None = "cpu",
) -> FoundationResult:
    """Evaluate the differentiable foundation replay for one frame using an
    indenter SDF to compute penetration.

    Args:
        xy_m: ``(N, 2)`` array of in-plane cell positions [m].
        top_z_m: ``(N,)`` array of top-surface Z coordinates [m].
        slack_length_m: Raycast slack length for each spring [m].
        velocity_mps: Current spring-length velocity for each spring [m/s].
        cell_area_m2: Per-cell area [m²]. A single ``float`` is broadcast to
            all cells; an ``(N,)`` array supports per-cell variation.
        material: Shared foundation material parameters.
        indenter_sdf: Texture SDF of the indenter geometry.
        indenter_pos: Indenter position as ``(x, y, z)`` [m].
        indenter_quat: Indenter orientation quaternion ``(x, y, z, w)``.
            Defaults to identity.
        measured_force_n: Measured reference force [N]. When > 0 the loss is
            the squared residual; otherwise zero.
        device: Warp device. Must be a CUDA device when using a texture SDF.

    Returns:
        :class:`FoundationResult` with net force, spatial wrench, and scalar
        loss.
    """
    top_z = np.asarray(top_z_m, dtype=np.float32)
    slack = np.asarray(slack_length_m, dtype=np.float32)
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    n = top_z.shape[0]
    if top_z.ndim != 1 or n != np.asarray(xy_m).shape[0]:
        raise ValueError("top_z_m must be a 1D array matching xy_m rows")
    if slack.shape != top_z.shape or velocity.shape != top_z.shape:
        raise ValueError("slack_length_m and velocity_mps must match top_z_m shape")
    if np.any(slack <= 0.0):
        raise ValueError("slack_length_m values must be positive")
    wp.init()
    wp_top_z = wp.array(top_z, dtype=float, device=device)
    wp_slack = wp.array(slack, dtype=float, device=device)
    wp_velocity = wp.array(velocity, dtype=float, device=device)
    wp_xy = _as_vec2_array(xy_m, device, requires_grad=False)

    cell_array = _cell_area_array(cell_area_m2, n)
    wp_cell_area = wp.array(cell_array, dtype=float, device=device)

    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
    )
    neighbors_wp, h = _prepare_neighbors_and_spacing(xy_m, device, neighbors, spacing_m)
    force_out = wp.zeros(1, dtype=float, device=device)
    wrench_out = wp.zeros(6, dtype=float, device=device)

    if isinstance(indenter_pos, tuple):
        indenter_pos = wp.vec3(*indenter_pos)
    if isinstance(indenter_quat, tuple):
        indenter_quat = wp.quat(*indenter_quat)

    wp.launch(
        _foundation_sdf_kernel,
        dim=n,
        inputs=[
            wp_top_z,
            wp_slack,
            wp_velocity,
            wp_xy,
            wp_cell_area,
            indenter_sdf,
            indenter_pos,
            indenter_quat,
            wp_params,
            neighbors_wp,
            h,
            force_out,
            wrench_out,
        ],
        device=device,
    )

    force_n = float(force_out.numpy()[0])
    residual = force_n - float(measured_force_n)
    return FoundationResult(
        force_n=force_n,
        wrench=wrench_out.numpy().astype(np.float64),
        loss=0.5 * residual * residual,
    )


def foundation_lengths_loss_gradient(
    xy_m: np.ndarray,
    current_length_m: np.ndarray,
    slack_length_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    cell_area_m2: np.ndarray | float,
    material: FoundationMaterial,
    measured_force_n: float,
    neighbors: np.ndarray | None = None,
    spacing_m: float | None = None,
    device: str | wp.context.Device | None = "cpu",
) -> FoundationGradientResult:
    """Return force, loss, and material gradients for raycast-slack springs."""

    current = np.asarray(current_length_m, dtype=np.float32)
    slack = np.asarray(slack_length_m, dtype=np.float32)
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    if current.shape != slack.shape or current.shape != velocity.shape:
        raise ValueError("current_length_m, slack_length_m, and velocity_mps must have the same shape")
    if current.ndim != 1 or current.shape[0] != np.asarray(xy_m).shape[0]:
        raise ValueError("spring length arrays must be 1D and match xy_m rows")
    if np.any(slack <= 0.0):
        raise ValueError("slack_length_m values must be positive")

    wp.init()
    cell_array = _cell_area_array(cell_area_m2, current.shape[0])
    wp_cell_area = wp.array(cell_array, dtype=float, device=device)
    wp_current = wp.array(current, dtype=float, device=device)
    wp_slack = wp.array(slack, dtype=float, device=device)
    wp_velocity = wp.array(velocity, dtype=float, device=device)
    wp_xy = _as_vec2_array(xy_m, device, requires_grad=False)
    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    neighbors_wp, h = _prepare_neighbors_and_spacing(xy_m, device, neighbors, spacing_m)
    force_out = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    wrench_out = wp.zeros(6, dtype=float, device=device, requires_grad=True)
    loss_out = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(
            _foundation_lengths_kernel,
            dim=current.shape[0],
            inputs=[
                wp_current,
                wp_slack,
                wp_velocity,
                wp_xy,
                wp_cell_area,
                wp_params,
                neighbors_wp,
                h,
                force_out,
                wrench_out,
            ],
            device=device,
        )
        wp.launch(_loss_kernel, dim=1, inputs=[force_out, float(measured_force_n), loss_out], device=device)
    tape.backward(loss=loss_out)
    return FoundationGradientResult(
        force_n=float(force_out.numpy()[0]),
        wrench=wrench_out.numpy().astype(np.float64),
        loss=float(loss_out.numpy()[0]),
        gradient=wp_params.grad.numpy().astype(np.float64),
    )


def _validate_trial_batch(batch: FoundationTrialBatch, xy_m: np.ndarray) -> tuple[int, int]:
    current = np.asarray(batch.current_length_m)
    velocity = np.asarray(batch.velocity_mps)
    measured = np.asarray(batch.measured_force_n)
    weights = np.asarray(batch.sample_weight)
    slack = np.asarray(batch.slack_length_m)
    cell_area = np.asarray(batch.cell_area_m2)
    if current.ndim != 2:
        raise ValueError(f"Trial batch {batch.name!r} current_length_m must be 2D")
    if velocity.shape != current.shape:
        raise ValueError(f"Trial batch {batch.name!r} velocity_mps must match current_length_m")
    frame_count, spring_count = current.shape
    if spring_count != np.asarray(xy_m).shape[0]:
        raise ValueError(f"Trial batch {batch.name!r} spring count must match xy_m")
    dt = np.asarray(batch.dt_s)
    if measured.shape != (frame_count,) or weights.shape != (frame_count,) or dt.shape != (frame_count,):
        raise ValueError(f"Trial batch {batch.name!r} measured force and weights must match frame count")
    if slack.shape != (spring_count,) or cell_area.shape != (spring_count,):
        raise ValueError(f"Trial batch {batch.name!r} slack and cell area must match spring count")
    if np.any(slack <= 0.0):
        raise ValueError(f"Trial batch {batch.name!r} slack_length_m values must be positive")
    if frame_count <= 0 or spring_count <= 0:
        raise ValueError(f"Trial batch {batch.name!r} must not be empty")
    return frame_count, spring_count


def foundation_lengths_batch_loss_gradient(
    xy_m: np.ndarray,
    batch: FoundationTrialBatch,
    *,
    material: FoundationMaterial,
    loop_weight: float = 0.0,
    device: str | wp.context.Device | None = "cuda:0",
) -> FoundationTrialBatchResult:
    """Return weighted normalized loss and material gradient for one trial batch."""

    frame_count, spring_count = _validate_trial_batch(batch, xy_m)
    current = np.asarray(batch.current_length_m, dtype=np.float32)
    velocity = np.asarray(batch.velocity_mps, dtype=np.float32)
    measured = np.asarray(batch.measured_force_n, dtype=np.float32)
    weights = np.asarray(batch.sample_weight, dtype=np.float32)
    slack = np.asarray(batch.slack_length_m, dtype=np.float32)
    cell_area = np.asarray(batch.cell_area_m2, dtype=np.float32)
    if np.any(weights < 0.0) or not np.isclose(float(np.sum(weights)), 1.0, rtol=1.0e-4, atol=1.0e-6):
        raise ValueError(f"Trial batch {batch.name!r} sample weights must be non-negative and sum to 1")

    wp.init()
    wp_current = wp.array(current.reshape(-1), dtype=float, device=device)
    wp_velocity = wp.array(velocity.reshape(-1), dtype=float, device=device)
    wp_dt = wp.array(np.asarray(batch.dt_s, dtype=np.float32), dtype=float, device=device)
    wp_slack = wp.array(slack, dtype=float, device=device)
    wp_cell_area = wp.array(cell_area, dtype=float, device=device)
    wp_xy = _as_vec2_array(xy_m, device, requires_grad=False)
    wp_measured = wp.array(measured, dtype=float, device=device)
    wp_weights = wp.array(weights, dtype=float, device=device)
    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    neighbors_wp, h = _prepare_neighbors_and_spacing(xy_m, device, batch.neighbors, batch.spacing_m)
    force_out = wp.zeros(frame_count, dtype=float, device=device, requires_grad=True)
    loss_out = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    force_scale = float(max(np.max(np.abs(measured)), 1.0))

    total_steps = (int(material.state_warmup_cycles) + 1) * frame_count
    state_history = wp.zeros((spring_count, total_steps), dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        if material.state_warmup_cycles >= 0:
            wp.launch(
                _foundation_lengths_stateful_batch_kernel,
                dim=spring_count,
                inputs=[
                    wp_current,
                    wp_slack,
                    wp_velocity,
                    wp_dt,
                    wp_xy,
                    wp_cell_area,
                    wp_params,
                    neighbors_wp,
                    h,
                    frame_count,
                    spring_count,
                    int(material.state_warmup_cycles),
                    state_history,
                    force_out,
                ],
                device=device,
            )
        else:
            wp.launch(
                _foundation_lengths_batch_kernel,
                dim=frame_count * spring_count,
                inputs=[
                    wp_current,
                    wp_slack,
                    wp_velocity,
                    wp_xy,
                    wp_cell_area,
                    wp_params,
                    neighbors_wp,
                    h,
                    spring_count,
                    force_out,
                ],
                device=device,
            )
        wp.launch(
            _weighted_normalized_loss_kernel,
            dim=frame_count,
            inputs=[force_out, wp_measured, wp_weights, force_scale, loss_out],
            device=device,
        )

        if loop_weight > 0.0:
            try:
                from numpy import trapezoid as np_trapezoid
            except ImportError:
                from numpy import trapz as np_trapezoid

            measured_machine = measured + float(getattr(batch, "force_zero_n", 0.0))
            displacement = np.asarray(batch.displacement_m, dtype=np.float64)
            measured_loop = float(np_trapezoid(measured_machine, displacement))

            wp_disp = wp.array(np.asarray(displacement, dtype=np.float32), dtype=float, device=device)
            wp.launch(
                _add_loop_loss_kernel,
                dim=1,
                inputs=[
                    force_out,
                    wp_disp,
                    measured_loop,
                    float(getattr(batch, "force_zero_n", 0.0)),
                    float(loop_weight),
                    loss_out,
                    int(frame_count),
                ],
                device=device,
            )

    tape.backward(loss=loss_out)
    return FoundationTrialBatchResult(
        trial=batch.name,
        predicted_force_n=force_out.numpy().astype(np.float64),
        loss=float(loss_out.numpy()[0]),
        gradient=wp_params.grad.numpy().astype(np.float64),
    )


def evaluate_foundation_lengths_batch(
    xy_m: np.ndarray,
    batch: FoundationTrialBatch,
    *,
    material: FoundationMaterial,
    device: str | wp.context.Device | None = "cuda:0",
) -> FoundationTrialBatchResult:
    """Evaluate one trial batch without material gradients."""

    frame_count, spring_count = _validate_trial_batch(batch, xy_m)
    current = np.asarray(batch.current_length_m, dtype=np.float32)
    velocity = np.asarray(batch.velocity_mps, dtype=np.float32)
    measured = np.asarray(batch.measured_force_n, dtype=np.float32)
    weights = np.asarray(batch.sample_weight, dtype=np.float32)
    slack = np.asarray(batch.slack_length_m, dtype=np.float32)
    cell_area = np.asarray(batch.cell_area_m2, dtype=np.float32)

    wp.init()
    wp_current = wp.array(current.reshape(-1), dtype=float, device=device)
    wp_velocity = wp.array(velocity.reshape(-1), dtype=float, device=device)
    wp_dt = wp.array(np.asarray(batch.dt_s, dtype=np.float32), dtype=float, device=device)
    wp_slack = wp.array(slack, dtype=float, device=device)
    wp_cell_area = wp.array(cell_area, dtype=float, device=device)
    wp_xy = _as_vec2_array(xy_m, device, requires_grad=False)
    wp_measured = wp.array(measured, dtype=float, device=device)
    wp_weights = wp.array(weights, dtype=float, device=device)
    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
    )
    neighbors_wp, h = _prepare_neighbors_and_spacing(xy_m, device, batch.neighbors, batch.spacing_m)
    force_out = wp.zeros(frame_count, dtype=float, device=device)
    loss_out = wp.zeros(1, dtype=float, device=device)
    if material.state_warmup_cycles >= 0:
        total_steps = (int(material.state_warmup_cycles) + 1) * frame_count
        state_history = wp.zeros((spring_count, total_steps), dtype=float, device=device)
        wp.launch(
            _foundation_lengths_stateful_batch_kernel,
            dim=spring_count,
            inputs=[
                wp_current,
                wp_slack,
                wp_velocity,
                wp_dt,
                wp_xy,
                wp_cell_area,
                wp_params,
                neighbors_wp,
                h,
                frame_count,
                spring_count,
                int(material.state_warmup_cycles),
                state_history,
                force_out,
            ],
            device=device,
        )
    else:
        wp.launch(
            _foundation_lengths_batch_kernel,
            dim=frame_count * spring_count,
            inputs=[
                wp_current,
                wp_slack,
                wp_velocity,
                wp_xy,
                wp_cell_area,
                wp_params,
                neighbors_wp,
                h,
                spring_count,
                force_out,
            ],
            device=device,
        )
    wp.launch(
        _weighted_normalized_loss_kernel,
        dim=frame_count,
        inputs=[force_out, wp_measured, wp_weights, float(max(np.max(np.abs(measured)), 1.0)), loss_out],
        device=device,
    )
    return FoundationTrialBatchResult(
        trial=batch.name,
        predicted_force_n=force_out.numpy().astype(np.float64),
        loss=float(loss_out.numpy()[0]),
        gradient=np.zeros(8, dtype=np.float64),
    )


def _material_to_array(material: FoundationMaterial, include_state: bool = False) -> np.ndarray:
    return np.asarray(
        [
            material.stiffness_pa,
            material.ogden_alpha,
            material.lock_strain,
            material.damping_pa_s,
            material.damping_power,
            material.prony_stiffness_pa,
            material.prony_damping_pa_s,
            material.pasternak_stiffness_n_per_m,
        ],
        dtype=np.float64,
    )


def _array_to_material(params: np.ndarray, base: FoundationMaterial | None = None) -> FoundationMaterial:
    per_cylinder_area = False if base is None else base.per_cylinder_area
    state_warmup_cycles = 0 if base is None else base.state_warmup_cycles

    stiffness_pa = float(max(params[0], 1.0))
    prony_stiffness_pa = float(np.clip(params[5], 0.0, stiffness_pa))
    prony_damping_pa_s = float(max(params[6], 0.0))
    pasternak_stiffness_n_per_m = float(max(params[7], 0.0))

    return FoundationMaterial(
        stiffness_pa=stiffness_pa,
        ogden_alpha=float(max(params[1], 1.0e-3)),
        lock_strain=float(np.clip(params[2], 1.0e-3, 0.999)),
        damping_pa_s=float(max(params[3], 0.0)),
        damping_power=float(max(params[4], 0.0)),
        per_cylinder_area=per_cylinder_area,
        prony_stiffness_pa=prony_stiffness_pa,
        prony_damping_pa_s=prony_damping_pa_s,
        state_warmup_cycles=state_warmup_cycles,
        pasternak_stiffness_n_per_m=pasternak_stiffness_n_per_m,
    )


def fit_foundation_material_autodiff(
    xy_m: np.ndarray,
    samples: list[FoundationFitSample],
    *,
    cell_area_m2: float,
    initial_material: FoundationMaterial,
    iterations: int = 25,
    learning_rates: tuple[float, float, float, float, float] = (5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2),
    per_cylinder_area: bool = False,
    device: str | wp.context.Device | None = "cpu",
) -> FoundationFitResult:
    """Fit shared material parameters with Warp autodiff gradients.

    Updates are loss-normalized relative parameter steps over the shared
    material parameters. Stateful history parameters are only included in the
    batched averaged-cycle fitter.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not samples:
        raise ValueError("At least one fit sample is required")
    params = _material_to_array(initial_material)
    rates = np.zeros(8, dtype=np.float64)
    rates[:5] = learning_rates
    rates[7] = 5.0e-2

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_params = params.copy()
    for iteration in range(iterations):
        material = _array_to_material(params, initial_material)
        loss_sum = 0.0
        grad_sum = np.zeros(8, dtype=np.float64)
        force_sum = 0.0
        weight_sum = 0.0
        for sample in samples:
            sample_area = sample.cell_area_m2 if sample.cell_area_m2 is not None else cell_area_m2
            result = foundation_lengths_loss_gradient(
                xy_m,
                sample.current_length_m,
                sample.slack_length_m,
                sample.velocity_mps,
                cell_area_m2=sample_area,
                material=material,
                measured_force_n=sample.measured_force_n,
                neighbors=sample.neighbors,
                spacing_m=sample.spacing_m,
                device=device,
            )
            weight = float(sample.weight)
            loss_sum += weight * result.loss
            grad_sum += weight * result.gradient
            force_sum += weight * result.force_n
            weight_sum += weight

        scale = max(weight_sum, 1.0)
        mean_loss = loss_sum / scale
        mean_grad = grad_sum / scale
        mean_force = force_sum / scale
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = params.copy()
        history.append(
            {
                "iteration": float(iteration),
                "loss": float(mean_loss),
                "mean_force_n": float(mean_force),
                "stiffness_pa": float(material.stiffness_pa),
                "ogden_alpha": float(material.ogden_alpha),
                "lock_strain": float(material.lock_strain),
                "damping_pa_s": float(material.damping_pa_s),
                "damping_power": float(material.damping_power),
                "prony_stiffness_pa": float(material.prony_stiffness_pa),
                "prony_damping_pa_s": float(material.prony_damping_pa_s),
                "state_warmup_cycles": float(material.state_warmup_cycles),
                "pasternak_stiffness_n_per_m": float(material.pasternak_stiffness_n_per_m),
                "grad_stiffness_pa": float(mean_grad[0]),
                "grad_damping_pa_s": float(mean_grad[3]),
                "grad_prony_stiffness_pa": float(mean_grad[5]),
                "grad_prony_damping_pa_s": float(mean_grad[6]),
                "grad_pasternak_stiffness_n_per_m": float(mean_grad[7]),
            }
        )
        safe_grad = np.where(rates != 0.0, np.nan_to_num(mean_grad, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        log_step = rates * safe_grad * np.maximum(np.abs(params), 1.0) / max(mean_loss, 1.0)
        log_step = np.clip(log_step, -0.25, 0.25)
        active = rates != 0.0
        params[active] = params[active] * np.exp(-log_step[active])
        params = _material_to_array(_array_to_material(params, initial_material))

    material = _array_to_material(best_params, initial_material)
    result_material = FoundationMaterial(
        stiffness_pa=material.stiffness_pa,
        ogden_alpha=material.ogden_alpha,
        lock_strain=material.lock_strain,
        damping_pa_s=material.damping_pa_s,
        damping_power=material.damping_power,
        per_cylinder_area=per_cylinder_area,
        prony_stiffness_pa=material.prony_stiffness_pa,
        prony_damping_pa_s=material.prony_damping_pa_s,
        state_warmup_cycles=material.state_warmup_cycles,
        pasternak_stiffness_n_per_m=material.pasternak_stiffness_n_per_m,
    )
    return FoundationFitResult(material=result_material, history=tuple(history))


def fit_foundation_material_batches_autodiff(
    xy_m: np.ndarray,
    batches: list[FoundationTrialBatch],
    *,
    initial_material: FoundationMaterial,
    iterations: int = 25,
    learning_rates: tuple[float, float, float, float, float, float, float] = (
        5.0e-2,
        1.0e-2,
        1.0e-2,
        5.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
    ),
    per_cylinder_area: bool = True,
    loop_weight: float = 0.0,
    device: str | wp.context.Device | None = "cuda:0",
) -> FoundationFitResult:
    """Fit shared material parameters from one GPU batch per trial."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not batches:
        raise ValueError("At least one trial batch is required")

    from scipy.optimize import minimize

    # Get initial parameters
    params = _material_to_array(initial_material, include_state=True)

    # If pasternak_stiffness_n_per_m is initialized to 0.0, but we have learning rates,
    # initialize it to 500.0 so the log-space optimization can run and find positive values.
    # We do this only if rates[7] (Pasternak stiffness rate) will be non-zero.
    rates = np.zeros(8, dtype=np.float64)
    rates[:7] = learning_rates
    rates[7] = 1.0e-2

    if rates[7] != 0.0 and params[7] == 0.0:
        params[7] = 500.0

    # Define physical bounds
    bounds_phys = [
        (1000.0, 1.0e7),    # stiffness_pa
        (0.0001, 5.0),      # ogden_alpha
        (0.1, 0.99),        # lock_strain
        (1.0, 1.0e6),       # damping_pa_s
        (0.01, 5.0),        # damping_power
        (1.0, 1.0e7),       # prony_stiffness_pa
        (1.0, 1.0e6),       # prony_damping_pa_s
        (0.1, 1.0e5),       # pasternak_stiffness_n_per_m
    ]

    # For any parameter with rates[i] == 0, lock it at its initial value
    for i in range(8):
        if rates[i] == 0.0:
            val = float(params[i])
            bounds_phys[i] = (val, val)

    # Convert to safe log-space coordinates
    x0_safe = np.maximum(params, 1.0e-5)
    y0 = np.log(x0_safe)

    bounds_log = []
    for i, (low, high) in enumerate(bounds_phys):
        if low == 0.0 and high == 0.0:
            bounds_log.append((0.0, 0.0))
        else:
            low_log = np.log(max(low, 1.0e-5))
            high_log = np.log(max(high, 1.0e-5))
            bounds_log.append((low_log, high_log))

    history: list[dict[str, float]] = []

    def loss_and_grad(y):
        x = np.exp(y)
        for i, (low, high) in enumerate(bounds_phys):
            if low == 0.0 and high == 0.0:
                x[i] = 0.0

        material = _array_to_material(x, initial_material)
        loss_sum = 0.0
        grad_sum_x = np.zeros(8, dtype=np.float64)
        force_sum = 0.0
        frame_sum = 0

        for batch in batches:
            result = foundation_lengths_batch_loss_gradient(
                xy_m,
                batch,
                material=material,
                loop_weight=loop_weight,
                device=device,
            )
            loss_sum += result.loss
            grad_sum_x += result.gradient
            force_sum += float(np.sum(result.predicted_force_n))
            frame_sum += len(result.predicted_force_n)

        scale = float(len(batches))
        mean_loss = loss_sum / scale
        mean_grad_x = grad_sum_x / scale
        mean_grad_y = mean_grad_x * x

        for i, (low, high) in enumerate(bounds_phys):
            if low == 0.0 and high == 0.0:
                mean_grad_y[i] = 0.0

        mean_force = force_sum / max(frame_sum, 1)
        history.append(
            {
                "iteration": float(len(history)),
                "loss": float(mean_loss),
                "mean_force_n": float(mean_force),
                "stiffness_pa": float(material.stiffness_pa),
                "ogden_alpha": float(material.ogden_alpha),
                "lock_strain": float(material.lock_strain),
                "damping_pa_s": float(material.damping_pa_s),
                "damping_power": float(material.damping_power),
                "prony_stiffness_pa": float(material.prony_stiffness_pa),
                "prony_damping_pa_s": float(material.prony_damping_pa_s),
                "state_warmup_cycles": float(material.state_warmup_cycles),
                "pasternak_stiffness_n_per_m": float(material.pasternak_stiffness_n_per_m),
                "grad_stiffness_pa": float(mean_grad_x[0]),
                "grad_ogden_alpha": float(mean_grad_x[1]),
                "grad_lock_strain": float(mean_grad_x[2]),
                "grad_damping_pa_s": float(mean_grad_x[3]),
                "grad_damping_power": float(mean_grad_x[4]),
                "grad_prony_stiffness_pa": float(mean_grad_x[5]),
                "grad_prony_damping_pa_s": float(mean_grad_x[6]),
                "grad_pasternak_stiffness_n_per_m": float(mean_grad_x[7]),
            }
        )
        return mean_loss, mean_grad_y

    opt_res = minimize(
        loss_and_grad,
        y0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_log,
        options={"maxiter": iterations},
    )

    best_x = np.exp(opt_res.x)
    for i, (low, high) in enumerate(bounds_phys):
        if low == 0.0 and high == 0.0:
            best_x[i] = 0.0

    best_material = _array_to_material(best_x, initial_material)
    result_material = FoundationMaterial(
        stiffness_pa=best_material.stiffness_pa,
        ogden_alpha=best_material.ogden_alpha,
        lock_strain=best_material.lock_strain,
        damping_pa_s=best_material.damping_pa_s,
        damping_power=best_material.damping_power,
        per_cylinder_area=per_cylinder_area,
        prony_stiffness_pa=best_material.prony_stiffness_pa,
        prony_damping_pa_s=best_material.prony_damping_pa_s,
        state_warmup_cycles=best_material.state_warmup_cycles,
        pasternak_stiffness_n_per_m=best_material.pasternak_stiffness_n_per_m,
    )
    return FoundationFitResult(material=result_material, history=tuple(history))


def finite_difference_loss_gradient(
    xy_m: np.ndarray,
    compression_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    cell_area_m2: float,
    thickness_m: float,
    material: FoundationMaterial,
    measured_force_n: float,
    stiffness_eps: float = 1.0,
) -> float:
    """Small test helper for the stiffness loss gradient."""

    plus = FoundationMaterial(
        material.stiffness_pa + stiffness_eps,
        material.ogden_alpha,
        material.lock_strain,
        material.damping_pa_s,
        material.damping_power,
    )
    minus = FoundationMaterial(
        material.stiffness_pa - stiffness_eps,
        material.ogden_alpha,
        material.lock_strain,
        material.damping_pa_s,
        material.damping_power,
    )
    loss_plus = evaluate_foundation(
        xy_m,
        compression_m,
        velocity_mps,
        cell_area_m2=cell_area_m2,
        thickness_m=thickness_m,
        material=plus,
        measured_force_n=measured_force_n,
    ).loss
    loss_minus = evaluate_foundation(
        xy_m,
        compression_m,
        velocity_mps,
        cell_area_m2=cell_area_m2,
        thickness_m=thickness_m,
        material=minus,
        measured_force_n=measured_force_n,
    ).loss
    return (loss_plus - loss_minus) / (2.0 * stiffness_eps)


def warp_loss_gradient(
    xy_m: np.ndarray,
    compression_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    cell_area_m2: float,
    thickness_m: float,
    material: FoundationMaterial,
    measured_force_n: float,
    neighbors: np.ndarray | None = None,
    spacing_m: float | None = None,
    device: str | wp.context.Device | None = "cpu",
) -> np.ndarray:
    """Return ``d loss / d material`` from Warp autodiff."""

    compression = np.asarray(compression_m, dtype=np.float32)
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    wp.init()
    wp_compression = wp.array(compression, dtype=float, device=device)
    wp_velocity = wp.array(velocity, dtype=float, device=device)
    wp_xy = _as_vec2_array(xy_m, device, requires_grad=False)
    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    neighbors_wp, h = _prepare_neighbors_and_spacing(xy_m, device, neighbors, spacing_m)
    force_out = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    wrench_out = wp.zeros(6, dtype=float, device=device, requires_grad=True)
    loss_out = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(
            _foundation_kernel,
            dim=compression.shape[0],
            inputs=[
                wp_compression,
                wp_velocity,
                wp_xy,
                float(cell_area_m2),
                float(thickness_m),
                wp_params,
                float(measured_force_n),
                neighbors_wp,
                h,
                force_out,
                wrench_out,
                loss_out,
            ],
            device=device,
        )
        wp.launch(_loss_kernel, dim=1, inputs=[force_out, float(measured_force_n), loss_out], device=device)
    tape.backward(loss=loss_out)
    return wp_params.grad.numpy().astype(np.float64)
