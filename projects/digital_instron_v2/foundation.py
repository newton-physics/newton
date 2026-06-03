# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable vertical elastic-foundation model for Digital Instron v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .geometry import BakedMidsoleGeometry


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
    spatial_slope: float = 0.0

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
        spatial_slope: float = 0.0,
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
        object.__setattr__(self, "spatial_slope", float(spatial_slope))

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


@dataclass(frozen=True)
class ContactFieldResult:
    """Per-cell contact pressure field plus derived net force, CoP, and wrench.

    This is the general, manifest-free contact evaluation used to attach a
    learned midsole material and baked mesh geometry to an arbitrary indenter
    displacement schedule (for example a digital runner's foot strike).

    Attributes:
        cell_xy_m: Sample cell centroids in the surface-map plane, shape ``(cells, 2)`` ``[m]``.
        cell_area_m2: Per-cell tributary area ``[m^2]``.
        cell_force_n: Per-frame, per-cell vertical contact force, shape ``(frames, cells)`` ``[N]``.
        cell_pressure_pa: Per-frame, per-cell contact pressure ``cell_force_n / cell_area_m2`` ``[Pa]``.
        net_force_n: Per-frame total vertical force, shape ``(frames,)`` ``[N]``.
        cop_xy_m: Per-frame center of pressure in the plane, shape ``(frames, 2)`` ``[m]``;
            rows where ``net_force_n`` is near zero are ``NaN``.
        wrench: Per-frame ``[Fz, Mx, My]`` about the plane origin, shape ``(frames, 3)``
            with ``Fz`` ``[N]`` and moments ``[N·m]`` (``Mx = sum(y * fz)``, ``My = sum(-x * fz)``).
    """

    cell_xy_m: np.ndarray
    cell_area_m2: np.ndarray
    cell_force_n: np.ndarray
    cell_pressure_pa: np.ndarray
    net_force_n: np.ndarray
    cop_xy_m: np.ndarray
    wrench: np.ndarray


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
    dt_s: float,
    longitudinal_axis: int,
    x_min: float,
    x_max: float,
    state_in: wp.array[float],
    state_out: wp.array[float],
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

    xy = xy_m[i]
    coord = float(0.0)
    if longitudinal_axis == 0:
        coord = xy[0] - x_min
    else:
        coord = xy[1] - x_min
    bar_x = coord / x_max
    spatial_slope = params[8] - 1.0
    scale = wp.max(1.0 + spatial_slope * bar_x, 0.01)
    ogden_stress = ogden_stress * scale

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

    # QLV Prony Viscoelastic Stress
    ep = wp.max(params[5], 0.0)
    e0 = wp.max(params[0], 1.0e-4)
    beta = wp.min(ep / e0, 0.99)
    etap = wp.max(params[6], 0.0)
    tau = wp.max(etap / wp.max(ep, 1.0e-6), 1.0e-6)

    decay = wp.exp(-wp.max(dt_s, 0.0) / tau)
    prev_state = state_in[i]
    curr_state = prev_state * decay + (1.0 - decay) * beta * elastic_stress
    state_out[i] = curr_state

    viscoelastic_stress = elastic_stress - curr_state

    damping_strain = wp.max(strain, 1.0e-8)
    damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))
    compression_velocity = -velocity_mps[i]
    viscous_stress = params[3] * damping_weight * compression_velocity
    fz = cell_area_m2[i] * wp.max(viscoelastic_stress + viscous_stress, 0.0)
    wp.atomic_add(force_out, 0, fz)
    wp.atomic_add(wrench_out, 2, fz)
    wp.atomic_add(wrench_out, 3, xy[1] * fz)
    wp.atomic_add(wrench_out, 4, -xy[0] * fz)


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
    longitudinal_axis: int,
    x_min: float,
    x_max: float,
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

    xy = xy_m[spring]
    coord = float(0.0)
    if longitudinal_axis == 0:
        coord = xy[0] - x_min
    else:
        coord = xy[1] - x_min
    bar_x = coord / x_max
    spatial_slope = params[8] - 1.0
    scale = wp.max(1.0 + spatial_slope * bar_x, 0.01)
    ogden_stress = ogden_stress * scale

    h2 = spacing_m * spacing_m
    laplacian = float(0.0)
    if h2 > 1.0e-12:
        n_left = neighbors[spring, 0]
        val_left = comp
        if n_left != -1:
            val_left = wp.max(
                slack_length_m[n_left] - current_length_m[(tid / spring_count) * spring_count + n_left], 0.0
            )
        n_right = neighbors[spring, 1]
        val_right = comp
        if n_right != -1:
            val_right = wp.max(
                slack_length_m[n_right] - current_length_m[(tid / spring_count) * spring_count + n_right], 0.0
            )
        n_bottom = neighbors[spring, 2]
        val_bottom = comp
        if n_bottom != -1:
            val_bottom = wp.max(
                slack_length_m[n_bottom] - current_length_m[(tid / spring_count) * spring_count + n_bottom], 0.0
            )
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
    longitudinal_axis: int,
    x_min: float,
    x_max: float,
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

    xy = xy_m[spring]
    coord = float(0.0)
    if longitudinal_axis == 0:
        coord = xy[0] - x_min
    else:
        coord = xy[1] - x_min
    bar_x = coord / x_max
    spatial_slope = params[8] - 1.0
    scale = wp.max(1.0 + spatial_slope * bar_x, 0.01)

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
            ogden_stress = ogden_stress * scale

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
                    val_bottom = wp.max(
                        slack_length_m[n_bottom] - current_length_m[frame * spring_count + n_bottom], 0.0
                    )
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
def _add_positive_hysteresis_loss_kernel(
    force_out: wp.array[float],
    displacement_m: wp.array[float],
    active_mask: wp.array[float],
    measured_hysteresis: float,
    loop_weight: float,
    loss_out: wp.array[float],
    frame_count: int,
):
    if wp.tid() == 0:
        max_disp = float(-1.0e10)
        local_peak = int(-1)
        for i in range(frame_count):
            if active_mask[i] > 0.5:
                disp = displacement_m[i]
                if disp > max_disp:
                    max_disp = disp
                    local_peak = i

        if local_peak != -1:
            pred_loading_work = float(0.0)
            pred_unloading_work = float(0.0)

            prev_loading_idx = int(-1)
            for i in range(local_peak + 1):
                if active_mask[i] > 0.5:
                    if prev_loading_idx != -1:
                        f_prev = force_out[prev_loading_idx]
                        f_curr = force_out[i]
                        dx = displacement_m[i] - displacement_m[prev_loading_idx]
                        pred_loading_work += 0.5 * (f_prev + f_curr) * dx
                    prev_loading_idx = i

            prev_unloading_idx = int(-1)
            for i in range(local_peak, frame_count):
                if active_mask[i] > 0.5:
                    if prev_unloading_idx != -1:
                        f_prev = force_out[prev_unloading_idx]
                        f_curr = force_out[i]
                        dx = displacement_m[i] - displacement_m[prev_unloading_idx]
                        pred_unloading_work += 0.5 * (f_prev + f_curr) * dx
                    prev_unloading_idx = i

            pred_unloading_work = -pred_unloading_work
            pred_hysteresis = pred_loading_work - pred_unloading_work

            meas_hyst = wp.max(measured_hysteresis, 1.0e-3)
            hyst_residual = (pred_hysteresis - meas_hyst) / meas_hyst
            loss_val = 0.5 * loop_weight * hyst_residual * hyst_residual
            wp.atomic_add(loss_out, 0, loss_val)


def _infer_longitudinal_axis_and_x_max(xy_m: np.ndarray) -> tuple[int, float, float]:
    xy = np.asarray(xy_m)
    if len(xy) == 0:
        return 0, 0.0, 1.0
    x_min = float(np.min(xy[:, 0]))
    x_max = float(np.max(xy[:, 0]))
    y_min = float(np.min(xy[:, 1]))
    y_max = float(np.max(xy[:, 1]))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span >= y_span:
        return 0, x_min, max(x_span, 1.0e-5)
    else:
        return 1, y_min, max(y_span, 1.0e-5)


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
    from .geometry import compute_grid_neighbors  # noqa: PLC0415

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


def evaluate_foundation_lengths(
    xy_m: np.ndarray,
    current_length_m: np.ndarray,
    slack_length_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    cell_area_m2: np.ndarray | float,
    material: FoundationMaterial,
    dt_s: float = 0.001,
    state_in: wp.array | None = None,
    state_out: wp.array | None = None,
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
    longitudinal_axis, x_min, x_max = _infer_longitudinal_axis_and_x_max(xy_m)

    spring_count = current.shape[0]
    if state_in is None:
        state_in = wp.zeros(spring_count, dtype=float, device=device)
    if state_out is None:
        state_out = wp.zeros(spring_count, dtype=float, device=device)

    force_out = wp.zeros(1, dtype=float, device=device)
    wrench_out = wp.zeros(6, dtype=float, device=device)
    wp.launch(
        _foundation_lengths_kernel,
        dim=spring_count,
        inputs=[
            wp_current,
            wp_slack,
            wp_velocity,
            wp_xy,
            wp_cell_area,
            wp_params,
            neighbors_wp,
            h,
            float(dt_s),
            int(longitudinal_axis),
            float(x_min),
            float(x_max),
            state_in,
            state_out,
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
    longitudinal_axis, x_min, x_max = _infer_longitudinal_axis_and_x_max(xy_m)
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
                int(longitudinal_axis),
                float(x_min),
                float(x_max),
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
                int(longitudinal_axis),
                float(x_min),
                float(x_max),
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
        gradient=np.zeros(9, dtype=np.float64),
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
            material.spatial_slope + 1.0,
        ],
        dtype=np.float64,
    )


def _array_to_material(params: np.ndarray, base: FoundationMaterial | None = None) -> FoundationMaterial:
    per_cylinder_area = False if base is None else base.per_cylinder_area
    state_warmup_cycles = 0 if base is None else base.state_warmup_cycles

    stiffness_pa = float(max(params[0], 50000.0))
    prony_stiffness_pa = float(np.clip(params[5], 0.0, stiffness_pa))
    prony_damping_pa_s = float(max(params[6], 0.0))
    pasternak_stiffness_n_per_m = float(max(params[7], 0.0))
    spatial_slope_shifted = float(np.clip(params[8], 0.01, 1.99))
    spatial_slope = spatial_slope_shifted - 1.0

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
        spatial_slope=spatial_slope,
    )


def fit_foundation_material_baked_batches_autodiff(
    xy_m: np.ndarray,
    baked_geometry: BakedMidsoleGeometry,
    indenter_maps_by_trial: dict[str, tuple[np.ndarray, np.ndarray]],
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
    top_fractions_by_trial: dict[str, float] | None = None,
    bottom_fractions_by_trial: dict[str, float] | None = None,
    use_equilibrium: bool = True,
    use_subcell_coverage: bool = False,
    device: str | wp.context.Device | None = "cuda:0",
) -> FoundationFitResult:
    """Fit shared material parameters from baked (spatially-invariant) foundation batches using Warp gradients."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not batches:
        raise ValueError("At least one trial batch is required")

    from scipy.optimize import minimize

    # Get initial parameters
    params = _material_to_array(initial_material, include_state=True)

    rates = np.zeros(9, dtype=np.float64)
    rates[: len(learning_rates)] = learning_rates
    rates[7] = 0.0
    if len(learning_rates) <= 7:
        rates[7] = 1.0e-2
    if len(learning_rates) <= 8:
        rates[8] = 1.0e-2

    params[7] = 0.0

    # Define physical bounds
    bounds_phys = [
        (50000.0, 1.0e7),  # stiffness_pa
        (0.0001, 5.0),  # ogden_alpha
        (0.1, 0.99),  # lock_strain
        (1.0, 1.0e6),  # damping_pa_s
        (0.01, 5.0),  # damping_power
        (1.0, 1.0e7),  # prony_stiffness_pa
        (1.0, 1.0e6),  # prony_damping_pa_s
        (0.0, 0.0),  # pasternak_stiffness_n_per_m disabled for baked contact
        (0.01, 1.99),  # spatial_slope_shifted
    ]

    for i in range(9):
        if rates[i] == 0.0:
            val = float(params[i])
            bounds_phys[i] = (val, val)

    x0_safe = np.maximum(params, 1.0e-5)
    y0 = np.log(x0_safe)

    bounds_log = []
    for _, (low, high) in enumerate(bounds_phys):
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
        grad_sum_x = np.zeros(9, dtype=np.float64)
        force_sum = 0.0
        frame_sum = 0

        for batch in batches:
            ind_map, ind_valid_map = indenter_maps_by_trial[batch.name]
            top_frac = top_fractions_by_trial.get(batch.name, 1.0) if top_fractions_by_trial is not None else 1.0
            bottom_frac = (
                bottom_fractions_by_trial.get(batch.name, 0.0) if bottom_fractions_by_trial is not None else 0.0
            )

            result = foundation_baked_batch_loss_gradient(
                xy_m,
                baked_geometry,
                ind_map,
                ind_valid_map,
                batch,
                material=material,
                loop_weight=loop_weight,
                top_fraction=top_frac,
                bottom_fraction=bottom_frac,
                use_equilibrium=use_equilibrium,
                use_subcell_coverage=use_subcell_coverage,
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
                "spatial_slope": float(material.spatial_slope),
                "grad_stiffness_pa": float(mean_grad_x[0]),
                "grad_ogden_alpha": float(mean_grad_x[1]),
                "grad_lock_strain": float(mean_grad_x[2]),
                "grad_damping_pa_s": float(mean_grad_x[3]),
                "grad_damping_power": float(mean_grad_x[4]),
                "grad_prony_stiffness_pa": float(mean_grad_x[5]),
                "grad_prony_damping_pa_s": float(mean_grad_x[6]),
                "grad_pasternak_stiffness_n_per_m": float(mean_grad_x[7]),
                "grad_spatial_slope": float(mean_grad_x[8]),
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
        spatial_slope=best_material.spatial_slope,
    )
    return FoundationFitResult(material=result_material, history=tuple(history))


# =============================================================================
# Baked (Spatially-Invariant) Foundation Kernels & Python Wrappers
# =============================================================================


def _baked_min_bottom(baked_geometry: BakedMidsoleGeometry) -> float:
    valid_map = baked_geometry.valid_map
    if valid_map is None:
        return float(np.min(baked_geometry.bottom_map))
    valid = np.asarray(valid_map, dtype=np.float64) > 0.5
    if not np.any(valid):
        raise ValueError("Baked midsole geometry has no valid pixels")
    return float(np.min(baked_geometry.bottom_map[valid]))


@wp.func
def sample_2d_map_bilinear(
    texture_map: wp.array2d[float],
    u: float,
    v: float,
) -> float:
    h = float(texture_map.shape[0])
    w = float(texture_map.shape[1])

    # Map normalized coords to pixel indices [0, w-1] and [0, h-1]
    px = u * (w - 1.0)
    py = v * (h - 1.0)

    x0 = wp.clamp(int(wp.floor(px)), 0, int(w) - 1)
    y0 = wp.clamp(int(wp.floor(py)), 0, int(h) - 1)
    x1 = wp.clamp(x0 + 1, 0, int(w) - 1)
    y1 = wp.clamp(y0 + 1, 0, int(h) - 1)

    tx = px - float(x0)
    ty = py - float(y0)

    val00 = texture_map[y0, x0]
    val10 = texture_map[y0, x1]
    val01 = texture_map[y1, x0]
    val11 = texture_map[y1, x1]

    val_top = val00 + tx * (val10 - val00)
    val_bot = val01 + tx * (val11 - val01)

    return val_top + ty * (val_bot - val_top)


@wp.func
def _baked_compression(
    use_equilibrium: int,
    use_subcell_coverage: int,
    ind_val: float,
    disp: float,
    indenter_z: float,
    z_top_undeformed: float,
    z_bottom_undeformed: float,
    min_bottom: float,
    slack: float,
    top_fraction: float,
    bottom_fraction: float,
) -> float:
    comp = float(0.0)
    # When sub-cell coverage is enabled the hard `ind_val > 0.5` gate is dropped:
    # boundary cells are kept and weighted by their fractional indenter coverage
    # at the force-accumulation step instead, which removes the resolution-
    # dependent jump at sharp indenter edges. Off-footprint cells still produce
    # zero force because their coverage weight vanishes.
    do_compute = int(0)
    if use_subcell_coverage != 0:
        # Keep fractional boundary cells, but skip cells with no indenter
        # coverage at all. Otherwise off-footprint cells would compute a full
        # compression that explodes near the Ogden locking singularity during
        # optimisation (their force is zero-weighted, but the steep stress makes
        # the loss landscape ill-conditioned and breaks the line search).
        if ind_val > 1.0e-6:
            do_compute = 1
    elif ind_val > 0.5:
        do_compute = 1
    if do_compute != 0:
        if use_equilibrium != 0:
            # Through-thickness pressure equilibrium: a homogeneous foam column
            # squeezed between the descending indenter and its fixed local bottom
            # support carries a single uniform compressive stress (1-D series
            # column), so measured top travel shortens the column after the
            # indenter-to-top gap closes. The baked bottom map defines local
            # column thickness; it is not an additional air gap to a global plane.
            gap_top = wp.max(indenter_z - z_top_undeformed, 0.0)
            comp = wp.clamp(disp - gap_top, 0.0, slack)
        else:
            top_travel = top_fraction * disp
            z_contact = indenter_z - top_travel
            top_comp = wp.max(z_top_undeformed - z_contact, 0.0)
            bottom_travel = bottom_fraction * disp
            bottom_comp = wp.max(min_bottom + bottom_travel - z_bottom_undeformed, 0.0)
            comp = wp.min(top_comp + bottom_comp, slack)
    return comp


@wp.kernel
def _foundation_baked_batch_kernel(
    sample_uv_m: wp.array[wp.vec2],
    cell_area_m2: float,
    thickness_map: wp.array2d[float],
    top_map: wp.array2d[float],
    bottom_map: wp.array2d[float],
    indenter_map: wp.array2d[float],
    indenter_valid_map: wp.array2d[float],
    mins_uv: wp.vec2,
    maxs_uv: wp.vec2,
    min_bottom: float,
    displacement_m: wp.array[float],
    displacement_velocity_mps: wp.array[float],
    top_fraction: float,
    bottom_fraction: float,
    use_equilibrium: int,
    use_subcell_coverage: int,
    params: wp.array[float],
    spring_count: int,
    longitudinal_axis: int,
    x_min: float,
    x_max: float,
    force_out: wp.array[float],
):
    tid = wp.tid()
    frame = tid / spring_count
    spring = tid - frame * spring_count

    xy = sample_uv_m[spring]

    # Compute normalized UV footprint coordinates
    u = (xy[0] - mins_uv[0]) / (maxs_uv[0] - mins_uv[0])
    v = (xy[1] - mins_uv[1]) / (maxs_uv[1] - mins_uv[1])

    u = wp.clamp(u, 0.0, 1.0)
    v = wp.clamp(v, 0.0, 1.0)

    # Sample baked properties
    slack = wp.max(sample_2d_map_bilinear(thickness_map, u, v), 1.0e-6)
    z_top_undeformed = sample_2d_map_bilinear(top_map, u, v)
    z_bottom_undeformed = sample_2d_map_bilinear(bottom_map, u, v)

    ind_val = sample_2d_map_bilinear(indenter_valid_map, u, v)

    disp = displacement_m[frame]
    disp_vel = displacement_velocity_mps[frame]

    indenter_z = sample_2d_map_bilinear(indenter_map, u, v)
    comp = _baked_compression(
        use_equilibrium,
        use_subcell_coverage,
        ind_val,
        disp,
        indenter_z,
        z_top_undeformed,
        z_bottom_undeformed,
        min_bottom,
        slack,
        top_fraction,
        bottom_fraction,
    )
    strain = comp / slack

    lock = wp.max(params[2], 1.0e-4)
    alpha = wp.max(params[1], 1.0e-4)
    normalized = wp.min(strain / lock, 0.999)
    ogden_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha

    coord = float(0.0)
    if longitudinal_axis == 0:
        coord = xy[0] - x_min
    else:
        coord = xy[1] - x_min
    bar_x = coord / x_max
    spatial_slope = params[8] - 1.0
    scale = wp.max(1.0 + spatial_slope * bar_x, 0.01)
    ogden_stress = ogden_stress * scale

    damping_strain = wp.max(strain, 1.0e-8)
    damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))

    comp_vel = float(0.0)
    if comp > 0.0:
        comp_vel = disp_vel

    viscous_stress = params[3] * damping_weight * comp_vel

    coverage = float(1.0)
    if use_subcell_coverage != 0:
        coverage = wp.clamp(ind_val, 0.0, 1.0)
    fz = coverage * cell_area_m2 * wp.max(ogden_stress + viscous_stress, 0.0)

    wp.atomic_add(force_out, frame, fz)


@wp.kernel
def _foundation_baked_stateful_batch_kernel(
    sample_uv_m: wp.array[wp.vec2],
    cell_area_m2: float,
    thickness_map: wp.array2d[float],
    top_map: wp.array2d[float],
    bottom_map: wp.array2d[float],
    indenter_map: wp.array2d[float],
    indenter_valid_map: wp.array2d[float],
    mins_uv: wp.vec2,
    maxs_uv: wp.vec2,
    min_bottom: float,
    displacement_m: wp.array[float],
    displacement_velocity_mps: wp.array[float],
    dt_s: wp.array[float],
    top_fraction: float,
    bottom_fraction: float,
    use_equilibrium: int,
    use_subcell_coverage: int,
    params: wp.array[float],
    frame_count: int,
    spring_count: int,
    warmup_cycles: int,
    longitudinal_axis: int,
    x_min: float,
    x_max: float,
    state_history: wp.array2d[float],
    force_out: wp.array[float],
):
    spring = wp.tid()

    xy = sample_uv_m[spring]
    u = (xy[0] - mins_uv[0]) / (maxs_uv[0] - mins_uv[0])
    v = (xy[1] - mins_uv[1]) / (maxs_uv[1] - mins_uv[1])
    u = wp.clamp(u, 0.0, 1.0)
    v = wp.clamp(v, 0.0, 1.0)

    slack = wp.max(sample_2d_map_bilinear(thickness_map, u, v), 1.0e-6)
    z_top_undeformed = sample_2d_map_bilinear(top_map, u, v)
    z_bottom_undeformed = sample_2d_map_bilinear(bottom_map, u, v)
    ind_val = sample_2d_map_bilinear(indenter_valid_map, u, v)

    total_cycles = warmup_cycles + 1
    ep = wp.max(params[5], 0.0)
    e0 = wp.max(params[0], 1.0e-4)
    beta = wp.min(ep / e0, 0.99)
    etap = wp.max(params[6], 0.0)
    tau = wp.max(etap / wp.max(ep, 1.0e-6), 1.0e-6)

    coord = float(0.0)
    if longitudinal_axis == 0:
        coord = xy[0] - x_min
    else:
        coord = xy[1] - x_min
    bar_x = coord / x_max
    spatial_slope = params[8] - 1.0
    scale = wp.max(1.0 + spatial_slope * bar_x, 0.01)

    for cycle in range(total_cycles):
        for frame in range(frame_count):
            step = cycle * frame_count + frame

            disp = displacement_m[frame]
            disp_vel = displacement_velocity_mps[frame]

            indenter_z = sample_2d_map_bilinear(indenter_map, u, v)
            comp = _baked_compression(
                use_equilibrium,
                use_subcell_coverage,
                ind_val,
                disp,
                indenter_z,
                z_top_undeformed,
                z_bottom_undeformed,
                min_bottom,
                slack,
                top_fraction,
                bottom_fraction,
            )
            strain = comp / slack
            lock = wp.max(params[2], 1.0e-4)
            normalized = wp.min(strain / lock, 0.999)
            ogden_alpha = wp.max(params[1], 1.0e-4)
            ogden_stress = params[0] * (wp.pow(1.0 - normalized, -ogden_alpha) - 1.0) / ogden_alpha
            ogden_stress = ogden_stress * scale

            decay = wp.exp(-wp.max(dt_s[frame], 0.0) / tau)
            prev_state = float(0.0)
            if step > 0:
                prev_state = state_history[spring, step - 1]
            curr_state = prev_state * decay + (1.0 - decay) * beta * ogden_stress
            state_history[spring, step] = curr_state

            if cycle == warmup_cycles:
                viscoelastic_stress = ogden_stress - curr_state

                damping_strain = wp.max(strain, 1.0e-8)
                damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))

                comp_vel = float(0.0)
                if comp > 0.0:
                    comp_vel = disp_vel

                viscous_stress = params[3] * damping_weight * comp_vel
                coverage = float(1.0)
                if use_subcell_coverage != 0:
                    coverage = wp.clamp(ind_val, 0.0, 1.0)
                fz = coverage * cell_area_m2 * wp.max(viscoelastic_stress + viscous_stress, 0.0)
                wp.atomic_add(force_out, frame, fz)


@wp.kernel
def _foundation_baked_pressure_field_kernel(
    sample_uv_m: wp.array[wp.vec2],
    cell_area_m2: float,
    thickness_map: wp.array2d[float],
    top_map: wp.array2d[float],
    bottom_map: wp.array2d[float],
    indenter_map: wp.array2d[float],
    indenter_valid_map: wp.array2d[float],
    mins_uv: wp.vec2,
    maxs_uv: wp.vec2,
    min_bottom: float,
    displacement_m: wp.array[float],
    displacement_velocity_mps: wp.array[float],
    dt_s: wp.array[float],
    top_fraction: float,
    bottom_fraction: float,
    use_equilibrium: int,
    use_subcell_coverage: int,
    params: wp.array[float],
    frame_count: int,
    spring_count: int,
    warmup_cycles: int,
    longitudinal_axis: int,
    x_min: float,
    x_max: float,
    state_history: wp.array2d[float],
    cell_force_out: wp.array2d[float],
):
    spring = wp.tid()

    xy = sample_uv_m[spring]
    u = (xy[0] - mins_uv[0]) / (maxs_uv[0] - mins_uv[0])
    v = (xy[1] - mins_uv[1]) / (maxs_uv[1] - mins_uv[1])
    u = wp.clamp(u, 0.0, 1.0)
    v = wp.clamp(v, 0.0, 1.0)

    slack = wp.max(sample_2d_map_bilinear(thickness_map, u, v), 1.0e-6)
    z_top_undeformed = sample_2d_map_bilinear(top_map, u, v)
    z_bottom_undeformed = sample_2d_map_bilinear(bottom_map, u, v)
    ind_val = sample_2d_map_bilinear(indenter_valid_map, u, v)

    total_cycles = warmup_cycles + 1
    ep = wp.max(params[5], 0.0)
    e0 = wp.max(params[0], 1.0e-4)
    beta = wp.min(ep / e0, 0.99)
    etap = wp.max(params[6], 0.0)
    tau = wp.max(etap / wp.max(ep, 1.0e-6), 1.0e-6)

    coord = float(0.0)
    if longitudinal_axis == 0:
        coord = xy[0] - x_min
    else:
        coord = xy[1] - x_min
    bar_x = coord / x_max
    spatial_slope = params[8] - 1.0
    scale = wp.max(1.0 + spatial_slope * bar_x, 0.01)

    for cycle in range(total_cycles):
        for frame in range(frame_count):
            step = cycle * frame_count + frame

            disp = displacement_m[frame]
            disp_vel = displacement_velocity_mps[frame]

            indenter_z = sample_2d_map_bilinear(indenter_map, u, v)
            comp = _baked_compression(
                use_equilibrium,
                use_subcell_coverage,
                ind_val,
                disp,
                indenter_z,
                z_top_undeformed,
                z_bottom_undeformed,
                min_bottom,
                slack,
                top_fraction,
                bottom_fraction,
            )
            strain = comp / slack
            lock = wp.max(params[2], 1.0e-4)
            normalized = wp.min(strain / lock, 0.999)
            ogden_alpha = wp.max(params[1], 1.0e-4)
            ogden_stress = params[0] * (wp.pow(1.0 - normalized, -ogden_alpha) - 1.0) / ogden_alpha
            ogden_stress = ogden_stress * scale

            decay = wp.exp(-wp.max(dt_s[frame], 0.0) / tau)
            prev_state = float(0.0)
            if step > 0:
                prev_state = state_history[spring, step - 1]
            curr_state = prev_state * decay + (1.0 - decay) * beta * ogden_stress
            state_history[spring, step] = curr_state

            if cycle == warmup_cycles:
                viscoelastic_stress = ogden_stress - curr_state

                damping_strain = wp.max(strain, 1.0e-8)
                damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))

                comp_vel = float(0.0)
                if comp > 0.0:
                    comp_vel = disp_vel

                viscous_stress = params[3] * damping_weight * comp_vel
                coverage = float(1.0)
                if use_subcell_coverage != 0:
                    coverage = wp.clamp(ind_val, 0.0, 1.0)
                fz = coverage * cell_area_m2 * wp.max(viscoelastic_stress + viscous_stress, 0.0)
                cell_force_out[frame, spring] = fz


def evaluate_contact_field(
    sample_uv_m: np.ndarray,
    baked_geometry: BakedMidsoleGeometry,
    indenter_map: np.ndarray,
    indenter_valid_map: np.ndarray,
    batch: FoundationTrialBatch,
    *,
    material: FoundationMaterial,
    top_fraction: float = 1.0,
    bottom_fraction: float = 0.0,
    use_equilibrium: bool = True,
    use_subcell_coverage: bool = True,
    device: str | wp.context.Device | None = "cuda:0",
) -> ContactFieldResult:
    """Evaluate the per-cell vertical contact pressure field, net force, CoP, and wrench.

    This is the general, manifest-free contact evaluator: given a baked midsole
    mesh geometry, a learned material, and an indenter displacement schedule
    (the ``batch`` carries the per-frame ``displacement_m``, ``velocity_mps``,
    ``dt_s``, and ``cell_area_m2``), it replays the same viscoelastic foundation
    model used for fitting and returns the spatially resolved contact field.

    Args:
        sample_uv_m: Cell centroids in the surface-map plane, shape ``(cells, 2)`` ``[m]``.
        baked_geometry: Baked midsole thickness/top/bottom maps.
        indenter_map: Indenter Z height-map sampled on the surface-map grid ``[m]``.
        indenter_valid_map: Indenter coverage/validity map in ``[0, 1]``.
        batch: Per-frame displacement schedule and tributary cell area.
        material: Learned foundation material parameters.
        top_fraction: Fraction of the cell measured from the top surface.
        bottom_fraction: Fraction of the cell measured from the bottom surface.
        use_equilibrium: Enable through-thickness pressure equilibrium.
        use_subcell_coverage: Weight each cell by its fractional indenter coverage.
        device: Warp device to run on.

    Returns:
        ContactFieldResult: Per-cell force/pressure field plus net force, CoP, and wrench.
    """
    frame_count = len(batch.displacement_m)
    spring_count = sample_uv_m.shape[0]

    wp.init()
    wp_sample_uv = _as_vec2_array(sample_uv_m, device, requires_grad=False)
    wp_thickness = wp.array2d(baked_geometry.thickness_map, dtype=float, device=device)
    wp_top = wp.array2d(baked_geometry.top_map, dtype=float, device=device)
    wp_bottom = wp.array2d(baked_geometry.bottom_map, dtype=float, device=device)
    wp_indenter = wp.array2d(indenter_map, dtype=float, device=device)
    wp_indenter_valid = wp.array2d(indenter_valid_map, dtype=float, device=device)

    wp_displacement = wp.array(np.asarray(batch.displacement_m, dtype=np.float32), dtype=float, device=device)
    wp_velocity = wp.array(np.asarray(batch.velocity_mps, dtype=np.float32), dtype=float, device=device)
    wp_dt = wp.array(np.asarray(batch.dt_s, dtype=np.float32), dtype=float, device=device)

    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
    )
    longitudinal_axis, x_min, x_max = _infer_longitudinal_axis_and_x_max(sample_uv_m)

    mins_uv = wp.vec2(float(baked_geometry.mins_uv[0]), float(baked_geometry.mins_uv[1]))
    maxs_uv = wp.vec2(float(baked_geometry.maxs_uv[0]), float(baked_geometry.maxs_uv[1]))
    min_bottom = _baked_min_bottom(baked_geometry)
    cell_area_val = float(batch.cell_area_m2[0])

    warmup_cycles = max(int(material.state_warmup_cycles), 0)
    total_steps = (warmup_cycles + 1) * frame_count
    state_history = wp.zeros((spring_count, total_steps), dtype=float, device=device)
    cell_force_out = wp.zeros((frame_count, spring_count), dtype=float, device=device)

    wp.launch(
        _foundation_baked_pressure_field_kernel,
        dim=spring_count,
        inputs=[
            wp_sample_uv,
            cell_area_val,
            wp_thickness,
            wp_top,
            wp_bottom,
            wp_indenter,
            wp_indenter_valid,
            mins_uv,
            maxs_uv,
            min_bottom,
            wp_displacement,
            wp_velocity,
            wp_dt,
            float(top_fraction),
            float(bottom_fraction),
            int(use_equilibrium),
            int(use_subcell_coverage),
            wp_params,
            frame_count,
            spring_count,
            warmup_cycles,
            int(longitudinal_axis),
            float(x_min),
            float(x_max),
            state_history,
            cell_force_out,
        ],
        device=device,
    )

    cell_force_n = cell_force_out.numpy().astype(np.float64)
    cell_xy = np.asarray(sample_uv_m, dtype=np.float64).reshape(spring_count, 2)
    cell_area = np.full(spring_count, cell_area_val, dtype=np.float64)

    net_force_n = cell_force_n.sum(axis=1)
    x = cell_xy[:, 0]
    y = cell_xy[:, 1]
    fx = cell_force_n  # (frames, cells)
    mx = (fx * y[None, :]).sum(axis=1)
    my = (fx * (-x[None, :])).sum(axis=1)
    wrench = np.stack([net_force_n, mx, my], axis=1)

    safe = net_force_n > 1.0e-9
    cop_xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
    cop_xy[safe, 0] = (fx[safe] * x[None, :]).sum(axis=1) / net_force_n[safe]
    cop_xy[safe, 1] = (fx[safe] * y[None, :]).sum(axis=1) / net_force_n[safe]

    cell_pressure_pa = cell_force_n / max(cell_area_val, 1.0e-12)

    return ContactFieldResult(
        cell_xy_m=cell_xy,
        cell_area_m2=cell_area,
        cell_force_n=cell_force_n,
        cell_pressure_pa=cell_pressure_pa,
        net_force_n=net_force_n,
        cop_xy_m=cop_xy,
        wrench=wrench,
    )


def evaluate_foundation_baked_batch(
    sample_uv_m: np.ndarray,
    baked_geometry: BakedMidsoleGeometry,
    indenter_map: np.ndarray,
    indenter_valid_map: np.ndarray,
    batch: FoundationTrialBatch,
    *,
    material: FoundationMaterial,
    top_fraction: float = 1.0,
    bottom_fraction: float = 0.0,
    use_equilibrium: bool = True,
    use_subcell_coverage: bool = False,
    device: str | wp.context.Device | None = "cuda:0",
) -> FoundationTrialBatchResult:
    """Evaluate a batch of frames of the baked foundation replay without material gradients."""
    frame_count = len(batch.measured_force_n)
    spring_count = sample_uv_m.shape[0]

    wp.init()
    wp_sample_uv = _as_vec2_array(sample_uv_m, device, requires_grad=False)
    wp_thickness = wp.array2d(baked_geometry.thickness_map, dtype=float, device=device)
    wp_top = wp.array2d(baked_geometry.top_map, dtype=float, device=device)
    wp_bottom = wp.array2d(baked_geometry.bottom_map, dtype=float, device=device)
    wp_indenter = wp.array2d(indenter_map, dtype=float, device=device)
    wp_indenter_valid = wp.array2d(indenter_valid_map, dtype=float, device=device)

    wp_displacement = wp.array(np.asarray(batch.displacement_m, dtype=np.float32), dtype=float, device=device)
    wp_velocity = wp.array(np.asarray(batch.velocity_mps, dtype=np.float32), dtype=float, device=device)
    wp_dt = wp.array(np.asarray(batch.dt_s, dtype=np.float32), dtype=float, device=device)
    wp_weights = wp.array(np.asarray(batch.sample_weight, dtype=np.float32), dtype=float, device=device)
    wp_measured = wp.array(np.asarray(batch.measured_force_n, dtype=np.float32), dtype=float, device=device)

    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
    )
    longitudinal_axis, x_min, x_max = _infer_longitudinal_axis_and_x_max(sample_uv_m)
    force_out = wp.zeros(frame_count, dtype=float, device=device)
    loss_out = wp.zeros(1, dtype=float, device=device)
    force_scale = float(max(np.max(np.abs(batch.measured_force_n)), 1.0))

    mins_uv = wp.vec2(float(baked_geometry.mins_uv[0]), float(baked_geometry.mins_uv[1]))
    maxs_uv = wp.vec2(float(baked_geometry.maxs_uv[0]), float(baked_geometry.maxs_uv[1]))

    min_bottom = _baked_min_bottom(baked_geometry)

    cell_area_val = float(batch.cell_area_m2[0])

    if material.state_warmup_cycles >= 0:
        total_steps = (int(material.state_warmup_cycles) + 1) * frame_count
        state_history = wp.zeros((spring_count, total_steps), dtype=float, device=device)
        wp.launch(
            _foundation_baked_stateful_batch_kernel,
            dim=spring_count,
            inputs=[
                wp_sample_uv,
                cell_area_val,
                wp_thickness,
                wp_top,
                wp_bottom,
                wp_indenter,
                wp_indenter_valid,
                mins_uv,
                maxs_uv,
                min_bottom,
                wp_displacement,
                wp_velocity,
                wp_dt,
                float(top_fraction),
                float(bottom_fraction),
                int(use_equilibrium),
                int(use_subcell_coverage),
                wp_params,
                frame_count,
                spring_count,
                int(material.state_warmup_cycles),
                int(longitudinal_axis),
                float(x_min),
                float(x_max),
                state_history,
                force_out,
            ],
            device=device,
        )
    else:
        wp.launch(
            _foundation_baked_batch_kernel,
            dim=frame_count * spring_count,
            inputs=[
                wp_sample_uv,
                cell_area_val,
                wp_thickness,
                wp_top,
                wp_bottom,
                wp_indenter,
                wp_indenter_valid,
                mins_uv,
                maxs_uv,
                min_bottom,
                wp_displacement,
                wp_velocity,
                float(top_fraction),
                float(bottom_fraction),
                int(use_equilibrium),
                int(use_subcell_coverage),
                wp_params,
                spring_count,
                int(longitudinal_axis),
                float(x_min),
                float(x_max),
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
    return FoundationTrialBatchResult(
        trial=batch.name,
        predicted_force_n=force_out.numpy().astype(np.float64),
        loss=float(loss_out.numpy()[0]),
        gradient=np.zeros(9, dtype=np.float64),
    )


def foundation_baked_batch_loss_gradient(
    sample_uv_m: np.ndarray,
    baked_geometry: BakedMidsoleGeometry,
    indenter_map: np.ndarray,
    indenter_valid_map: np.ndarray,
    batch: FoundationTrialBatch,
    *,
    material: FoundationMaterial,
    loop_weight: float = 0.0,
    top_fraction: float = 1.0,
    bottom_fraction: float = 0.0,
    use_equilibrium: bool = True,
    use_subcell_coverage: bool = False,
    device: str | wp.context.Device | None = "cuda:0",
) -> FoundationTrialBatchResult:
    """Evaluate a batch of frames of the baked foundation replay and compute material parameter gradients."""
    frame_count = len(batch.measured_force_n)
    spring_count = sample_uv_m.shape[0]

    wp.init()
    wp_sample_uv = _as_vec2_array(sample_uv_m, device, requires_grad=False)
    wp_thickness = wp.array2d(baked_geometry.thickness_map, dtype=float, device=device)
    wp_top = wp.array2d(baked_geometry.top_map, dtype=float, device=device)
    wp_bottom = wp.array2d(baked_geometry.bottom_map, dtype=float, device=device)
    wp_indenter = wp.array2d(indenter_map, dtype=float, device=device)
    wp_indenter_valid = wp.array2d(indenter_valid_map, dtype=float, device=device)

    wp_displacement = wp.array(np.asarray(batch.displacement_m, dtype=np.float32), dtype=float, device=device)
    wp_velocity = wp.array(np.asarray(batch.velocity_mps, dtype=np.float32), dtype=float, device=device)
    wp_dt = wp.array(np.asarray(batch.dt_s, dtype=np.float32), dtype=float, device=device)
    wp_weights = wp.array(np.asarray(batch.sample_weight, dtype=np.float32), dtype=float, device=device)
    wp_measured = wp.array(np.asarray(batch.measured_force_n, dtype=np.float32), dtype=float, device=device)

    wp_params = wp.array(
        _material_to_array(material, include_state=True),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    longitudinal_axis, x_min, x_max = _infer_longitudinal_axis_and_x_max(sample_uv_m)
    force_out = wp.zeros(frame_count, dtype=float, device=device, requires_grad=True)
    loss_out = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    force_scale = float(max(np.max(np.abs(batch.measured_force_n)), 1.0))

    mins_uv = wp.vec2(float(baked_geometry.mins_uv[0]), float(baked_geometry.mins_uv[1]))
    maxs_uv = wp.vec2(float(baked_geometry.maxs_uv[0]), float(baked_geometry.maxs_uv[1]))

    min_bottom = _baked_min_bottom(baked_geometry)

    cell_area_val = float(batch.cell_area_m2[0])

    total_steps = (int(material.state_warmup_cycles) + 1) * frame_count
    state_history = wp.zeros((spring_count, total_steps), dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        if material.state_warmup_cycles >= 0:
            wp.launch(
                _foundation_baked_stateful_batch_kernel,
                dim=spring_count,
                inputs=[
                    wp_sample_uv,
                    cell_area_val,
                    wp_thickness,
                    wp_top,
                    wp_bottom,
                    wp_indenter,
                    wp_indenter_valid,
                    mins_uv,
                    maxs_uv,
                    min_bottom,
                    wp_displacement,
                    wp_velocity,
                    wp_dt,
                    float(top_fraction),
                    float(bottom_fraction),
                    int(use_equilibrium),
                    int(use_subcell_coverage),
                    wp_params,
                    frame_count,
                    spring_count,
                    int(material.state_warmup_cycles),
                    int(longitudinal_axis),
                    float(x_min),
                    float(x_max),
                    state_history,
                    force_out,
                ],
                device=device,
            )
        else:
            wp.launch(
                _foundation_baked_batch_kernel,
                dim=frame_count * spring_count,
                inputs=[
                    wp_sample_uv,
                    cell_area_val,
                    wp_thickness,
                    wp_top,
                    wp_bottom,
                    wp_indenter,
                    wp_indenter_valid,
                    mins_uv,
                    maxs_uv,
                    min_bottom,
                    wp_displacement,
                    wp_velocity,
                    float(top_fraction),
                    float(bottom_fraction),
                    int(use_equilibrium),
                    int(use_subcell_coverage),
                    wp_params,
                    spring_count,
                    int(longitudinal_axis),
                    float(x_min),
                    float(x_max),
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
            from .validation import active_force_mask, positive_hysteresis_work  # noqa: PLC0415

            active = active_force_mask(batch.measured_force_n, active_fraction=0.05, top_count=5)
            active_float = active.astype(np.float32)
            wp_active_mask = wp.array(active_float, dtype=float, device=device)
            displacement = np.asarray(batch.displacement_m, dtype=np.float64)
            measured_hyst = float(positive_hysteresis_work(displacement, batch.measured_force_n, active))

            wp_disp = wp.array(np.asarray(displacement, dtype=np.float32), dtype=float, device=device)
            wp.launch(
                _add_positive_hysteresis_loss_kernel,
                dim=1,
                inputs=[
                    force_out,
                    wp_disp,
                    wp_active_mask,
                    measured_hyst,
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
