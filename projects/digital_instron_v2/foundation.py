# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable vertical elastic-foundation model for Digital Instron v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf


@dataclass(frozen=True)
class FoundationMaterial:
    """Shared vertical foundation material parameters."""

    stiffness_pa: float
    ogden_alpha: float
    lock_strain: float
    damping_pa_s: float
    damping_power: float = 1.0
    per_cylinder_area: bool = False
    state_beta: float = 0.0
    state_tau_s: float = 0.05
    state_warmup_cycles: int = 0


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
    elastic_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha
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
    elastic_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha
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
    elastic_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha
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
    elastic_stress = params[0] * (wp.pow(1.0 - normalized, -alpha) - 1.0) / alpha
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
    frame_count: int,
    spring_count: int,
    warmup_cycles: int,
    force_out: wp.array[float],
):
    spring = wp.tid()
    slack = wp.max(slack_length_m[spring], 1.0e-6)
    state = float(0.0)
    total_cycles = warmup_cycles + 1
    tau = wp.max(params[6], 1.0e-6)
    beta = wp.max(wp.min(params[5], 1.0), 0.0)

    for cycle in range(total_cycles):
        for frame in range(frame_count):
            offset = frame * spring_count + spring
            comp = wp.max(slack - current_length_m[offset], 0.0)
            alpha_state = 1.0 - wp.exp(-wp.max(dt_s[frame], 0.0) / tau)
            state = state + alpha_state * (comp - state)

            if cycle == warmup_cycles:
                effective_comp = comp - beta * (comp - state)
                effective_comp = wp.max(effective_comp, 0.0)
                strain = effective_comp / slack
                lock = wp.max(params[2], 1.0e-4)
                normalized = wp.min(strain / lock, 0.999)
                ogden_alpha = wp.max(params[1], 1.0e-4)
                elastic_stress = params[0] * (wp.pow(1.0 - normalized, -ogden_alpha) - 1.0) / ogden_alpha

                instant_strain = comp / slack
                damping_strain = wp.max(instant_strain, 1.0e-8)
                damping_weight = wp.pow(damping_strain, wp.max(params[4], 0.0))
                compression_velocity = -velocity_mps[offset]
                viscous_stress = params[3] * damping_weight * compression_velocity
                fz = cell_area_m2[spring] * wp.max(elastic_stress + viscous_stress, 0.0)
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
        [
            material.stiffness_pa,
            material.ogden_alpha,
            material.lock_strain,
            material.damping_pa_s,
            material.damping_power,
        ],
        dtype=float,
        device=device,
    )
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
        [
            material.stiffness_pa,
            material.ogden_alpha,
            material.lock_strain,
            material.damping_pa_s,
            material.damping_power,
        ],
        dtype=float,
        device=device,
    )
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
        [
            material.stiffness_pa,
            material.ogden_alpha,
            material.lock_strain,
            material.damping_pa_s,
            material.damping_power,
        ],
        dtype=float,
        device=device,
    )

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
        [
            material.stiffness_pa,
            material.ogden_alpha,
            material.lock_strain,
            material.damping_pa_s,
            material.damping_power,
        ],
        dtype=float,
        device=device,
        requires_grad=True,
    )
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
    force_out = wp.zeros(frame_count, dtype=float, device=device, requires_grad=True)
    loss_out = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    force_scale = float(max(np.max(np.abs(measured)), 1.0))

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
                    frame_count,
                    spring_count,
                    int(material.state_warmup_cycles),
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
    force_out = wp.zeros(frame_count, dtype=float, device=device)
    loss_out = wp.zeros(1, dtype=float, device=device)
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
                frame_count,
                spring_count,
                int(material.state_warmup_cycles),
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
        gradient=np.zeros(7, dtype=np.float64),
    )


def _material_to_array(material: FoundationMaterial, include_state: bool = False) -> np.ndarray:
    values = [
        material.stiffness_pa,
        material.ogden_alpha,
        material.lock_strain,
        material.damping_pa_s,
        material.damping_power,
    ]
    if include_state:
        values += [material.state_beta, material.state_tau_s]
    return np.asarray(values, dtype=np.float64)


def _array_to_material(params: np.ndarray, base: FoundationMaterial | None = None) -> FoundationMaterial:
    per_cylinder_area = False if base is None else base.per_cylinder_area
    state_warmup_cycles = 0 if base is None else base.state_warmup_cycles

    if len(params) >= 7:
        state_beta = float(np.clip(params[5], 0.0, 1.0))
        state_tau_s = float(max(params[6], 1e-6))
    else:
        state_beta = 0.0 if base is None else base.state_beta
        state_tau_s = 0.05 if base is None else base.state_tau_s

    return FoundationMaterial(
        stiffness_pa=float(max(params[0], 1.0)),
        ogden_alpha=float(max(params[1], 1.0e-3)),
        lock_strain=float(np.clip(params[2], 1.0e-3, 0.999)),
        damping_pa_s=float(max(params[3], 0.0)),
        damping_power=float(max(params[4], 0.0)),
        per_cylinder_area=per_cylinder_area,
        state_beta=state_beta,
        state_tau_s=state_tau_s,
        state_warmup_cycles=state_warmup_cycles,
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
    rates = np.asarray(learning_rates, dtype=np.float64)
    if rates.shape != (5,):
        raise ValueError("learning_rates must have five entries")

    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        material = _array_to_material(params, initial_material)
        loss_sum = 0.0
        grad_sum = np.zeros(5, dtype=np.float64)
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
                "state_beta": float(material.state_beta),
                "state_tau_s": float(material.state_tau_s),
                "state_warmup_cycles": float(material.state_warmup_cycles),
                "grad_stiffness_pa": float(mean_grad[0]),
                "grad_damping_pa_s": float(mean_grad[3]),
            }
        )
        safe_grad = np.where(rates != 0.0, np.nan_to_num(mean_grad, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        log_step = rates * safe_grad * np.maximum(np.abs(params), 1.0) / max(mean_loss, 1.0)
        log_step = np.clip(log_step, -0.25, 0.25)
        active = rates != 0.0
        params[active] = params[active] * np.exp(-log_step[active])
        params = _material_to_array(_array_to_material(params, initial_material))

    material = _array_to_material(params, initial_material)
    result_material = FoundationMaterial(
        stiffness_pa=material.stiffness_pa,
        ogden_alpha=material.ogden_alpha,
        lock_strain=material.lock_strain,
        damping_pa_s=material.damping_pa_s,
        damping_power=material.damping_power,
        per_cylinder_area=per_cylinder_area,
        state_beta=material.state_beta,
        state_tau_s=material.state_tau_s,
        state_warmup_cycles=material.state_warmup_cycles,
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
    device: str | wp.context.Device | None = "cuda:0",
) -> FoundationFitResult:
    """Fit shared material parameters from one GPU batch per trial."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not batches:
        raise ValueError("At least one trial batch is required")
    params = _material_to_array(initial_material, include_state=True)
    rates = np.asarray(learning_rates, dtype=np.float64)
    if rates.shape != (7,):
        raise ValueError("learning_rates must have seven entries")

    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        material = _array_to_material(params, initial_material)
        loss_sum = 0.0
        grad_sum = np.zeros(7, dtype=np.float64)
        force_sum = 0.0
        frame_sum = 0
        for batch in batches:
            result = foundation_lengths_batch_loss_gradient(xy_m, batch, material=material, device=device)
            loss_sum += result.loss
            grad_sum += result.gradient
            force_sum += float(np.sum(result.predicted_force_n))
            frame_sum += len(result.predicted_force_n)

        scale = float(len(batches))
        mean_loss = loss_sum / scale
        mean_grad = grad_sum / scale
        mean_force = force_sum / max(frame_sum, 1)
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
                "state_beta": float(material.state_beta),
                "state_tau_s": float(material.state_tau_s),
                "state_warmup_cycles": float(material.state_warmup_cycles),
                "grad_stiffness_pa": float(mean_grad[0]),
                "grad_ogden_alpha": float(mean_grad[1]),
                "grad_lock_strain": float(mean_grad[2]),
                "grad_damping_pa_s": float(mean_grad[3]),
                "grad_damping_power": float(mean_grad[4]),
                "grad_state_beta": float(mean_grad[5]),
                "grad_state_tau_s": float(mean_grad[6]),
            }
        )
        safe_grad = np.where(rates != 0.0, np.nan_to_num(mean_grad, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        log_step = rates * safe_grad * np.maximum(np.abs(params), 1.0) / max(mean_loss, 1.0e-12)
        log_step = np.clip(log_step, -0.25, 0.25)
        active = rates != 0.0
        params[active] = params[active] * np.exp(-log_step[active])
        params = _material_to_array(_array_to_material(params, initial_material), include_state=True)

    material = _array_to_material(params, initial_material)
    result_material = FoundationMaterial(
        stiffness_pa=material.stiffness_pa,
        ogden_alpha=material.ogden_alpha,
        lock_strain=material.lock_strain,
        damping_pa_s=material.damping_pa_s,
        damping_power=material.damping_power,
        per_cylinder_area=per_cylinder_area,
        state_beta=material.state_beta,
        state_tau_s=material.state_tau_s,
        state_warmup_cycles=material.state_warmup_cycles,
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
        [
            material.stiffness_pa,
            material.ogden_alpha,
            material.lock_strain,
            material.damping_pa_s,
            material.damping_power,
        ],
        dtype=float,
        device=device,
        requires_grad=True,
    )
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
                force_out,
                wrench_out,
                loss_out,
            ],
            device=device,
        )
        wp.launch(_loss_kernel, dim=1, inputs=[force_out, float(measured_force_n), loss_out], device=device)
    tape.backward(loss=loss_out)
    return wp_params.grad.numpy().astype(np.float64)
