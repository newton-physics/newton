# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gradient-based foam material identification from a force-displacement curve.

This is the flagship payoff of the differentiable elastic foundation
(:mod:`projects.digital_instron_v2.dynamics_diff`): fitting the calibrated foam
material to a measured reaction-force curve with *exact* gradients instead of a
derivative-free sweep. A digital Instron drives the column bed through a
prescribed compression cycle and records the total reaction force at every
sample; :func:`fit_material_to_force_curve` differentiates that whole cycle --
including the generalized-Maxwell loading/unloading hysteresis -- with respect to
the foam material and descends a force-matching loss.

The problem is well conditioned because the reaction force scales directly with
the constitutive parameters (unlike a free-settling height, which is only weakly
sensitive to stiffness). The parameters span very different magnitudes, so the
optimizer works in a dimensionless *scale* space ``material = scale * reference``
and takes Adam steps on the scale vector.

Run the self-identification demo (recover a perturbed material from its own
synthetic curve)::

    uv run python -m projects.digital_instron_v2.inverse_id
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import dynamics
from .dynamics import FoundationParams
from .dynamics_diff import (
    MAT_PASTERNAK,
    foundation_pressure_diff,
)


@wp.kernel
def _accumulate_normal_force(
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    area: wp.array[wp.float32],
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    step: wp.int32,
    force_hist: wp.array[wp.float32],
):
    """Sum the Pasternak-coupled column normal forces for one Instron sample into ``force_hist[step]``."""
    i = wp.tid()
    pasternak = material_params[MAT_PASTERNAK]
    ci = compression[i]
    lap = -4.0 * ci
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            lap += compression[j]
        elif j == -1:
            lap += ci
    lap *= params.inv_h2
    pressure = base_pressure[i] - pasternak * lap
    if pressure < 0.0:
        pressure = 0.0
    wp.atomic_add(force_hist, step, pressure * area[i])


@wp.kernel
def _force_mse(force_hist: wp.array[wp.float32], target: wp.array[wp.float32], loss: wp.array[wp.float32]):
    """Accumulate the mean-square force-matching residual across all samples."""
    k = wp.tid()
    d = force_hist[k] - target[k]
    wp.atomic_add(loss, 0, d * d)


@dataclass
class FitResult:
    """Outcome of a force-matching material identification.

    Attributes:
        material_params: Fitted ``[g_eq, alpha, overstress, pasternak]`` vector
            ([Pa], [-], [-], [N/m]).
        scale: Fitted dimensionless scale of each parameter relative to the
            reference used to start the fit.
        loss_history: Mean-square force residual [N^2] at each iteration.
        force: Fitted reaction-force curve [N], one value per Instron sample.
    """

    material_params: np.ndarray
    scale: np.ndarray
    loss_history: np.ndarray
    force: np.ndarray


class InstronReplay:
    """Differentiable digital-Instron replay of the elastic-foundation column bed.

    Drives the calibrated column bed through a prescribed uniform-compression
    schedule and records the total reaction force at every sample as a
    differentiable function of the foam ``material_params``. The generalized-
    Maxwell overstress recurrence is unrolled with per-sample history so the whole
    loading/unloading cycle can be recorded on one :class:`warp.Tape`.

    Args:
        displacement_m: Prescribed platen compression at each sample [m], shape
            ``[sample_count]``.
        dt_s: Sample duration [s], scalar or per-sample array.
        material: Calibrated :class:`~projects.digital_instron_v2.core.Material`.
        geometry: Column bed from
            :func:`~projects.digital_instron_v2.dynamics.build_foundation_geometry`.
        device: Warp device.
    """

    def __init__(self, displacement_m, dt_s, material, geometry, device=None):
        self.device = device
        disp = np.ascontiguousarray(displacement_m, np.float32)
        self.nsteps = int(len(disp))
        self.dt = (
            np.full(self.nsteps, float(dt_s), np.float32)
            if np.isscalar(dt_s)
            else np.ascontiguousarray(dt_s, np.float32)
        )
        geo = geometry
        m = int(len(geo.slack_m))
        self.column_count = m

        params = FoundationParams()
        self.reference = np.array(
            [
                material.instantaneous_shear_modulus_pa * material.equilibrium_fraction,
                material.hyperfoam_exponent,
                (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction,
                material.pasternak_n_per_m,
            ],
            np.float32,
        )
        params.beta = dynamics.EFFECTIVE_POISSON_RATIO / (1.0 - 2.0 * dynamics.EFFECTIVE_POISSON_RATIO)
        params.inv_h2 = 1.0 / geo.spacing_m**2
        params.stretch_floor = 0.05
        self.params = params

        # Prescribed poses: pure downward translation by the platen displacement, so
        # every column sees a uniform compression equal to displacement[k].
        poses = np.zeros((self.nsteps, 7), np.float32)
        poses[:, 2] = -disp
        poses[:, 6] = 1.0
        self.pose = wp.array(poses, dtype=wp.transform, device=device)

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.slack_m])
        self.anchor_local = wp.array(np.ascontiguousarray(anchor, np.float32), dtype=wp.vec3, device=device)
        self.z_free = wp.array(np.ascontiguousarray(geo.slack_m, np.float32), dtype=wp.float32, device=device)
        self.rest_len = wp.array(np.ascontiguousarray(geo.slack_m, np.float32), dtype=wp.float32, device=device)
        self.area = wp.array(np.full(m, geo.area_m2, np.float32), dtype=wp.float32, device=device)
        self.neighbors = wp.array(np.ascontiguousarray(geo.neighbors, np.int32), dtype=wp.int32, device=device)

        self.material_params = wp.array(self.reference.copy(), dtype=wp.float32, device=device, requires_grad=True)

        def grad_zeros():
            return wp.zeros(m, dtype=wp.float32, device=device, requires_grad=True)

        self.compression = [grad_zeros() for _ in range(self.nsteps)]
        self.base_pressure = [grad_zeros() for _ in range(self.nsteps)]
        self.q_state = [grad_zeros() for _ in range(self.nsteps)]
        self.peq_hist = [grad_zeros() for _ in range(self.nsteps)]
        self.q_init = grad_zeros()
        self.peq_init = grad_zeros()
        self.force = wp.zeros(self.nsteps, dtype=wp.float32, device=device, requires_grad=True)

    def set_material(self, material_params: np.ndarray) -> None:
        """Overwrite the differentiable material vector with host values."""
        self.material_params.assign(np.ascontiguousarray(material_params, np.float32))

    def forward(self) -> wp.array:
        """Replay the compression cycle and return the per-sample reaction force [N]."""
        self.force.zero_()
        for k in range(self.nsteps):
            q_prev = self.q_init if k == 0 else self.q_state[k - 1]
            peq_prev = self.peq_init if k == 0 else self.peq_hist[k - 1]
            wp.launch(
                foundation_pressure_diff,
                dim=self.column_count,
                inputs=[
                    k,
                    float(self.dt[k]),
                    self.pose,
                    self.anchor_local,
                    self.z_free,
                    self.rest_len,
                    self.params,
                    self.material_params,
                    q_prev,
                    peq_prev,
                    self.q_state[k],
                    self.peq_hist[k],
                    self.compression[k],
                    self.base_pressure[k],
                ],
                device=self.device,
            )
            wp.launch(
                _accumulate_normal_force,
                dim=self.column_count,
                inputs=[
                    self.compression[k],
                    self.base_pressure[k],
                    self.neighbors,
                    self.area,
                    self.params,
                    self.material_params,
                    k,
                    self.force,
                ],
                device=self.device,
            )
        return self.force

    def zero_grad(self) -> None:
        """Zero the gradients on every differentiable buffer."""
        self.material_params.grad.zero_()
        self.force.grad.zero_()
        for buf in (*self.compression, *self.base_pressure, *self.q_state, *self.peq_hist):
            buf.grad.zero_()
        self.q_init.grad.zero_()
        self.peq_init.grad.zero_()


def fit_material_to_force_curve(
    target_force,
    displacement_m,
    dt_s,
    material,
    geometry,
    scale0=None,
    fit_mask=None,
    iterations: int = 300,
    learning_rate: float = 0.02,
    device=None,
) -> FitResult:
    """Fit the foam material to a measured reaction-force curve by gradient descent.

    Minimizes the mean-square force residual over the compression cycle with Adam,
    optimizing a dimensionless per-parameter scale ``material = scale * reference``
    so the very different parameter magnitudes stay comparable. Gradients come from
    a single backward pass over the differentiable Instron replay.

    Args:
        target_force: Measured reaction force at each sample [N].
        displacement_m: Prescribed platen compression at each sample [m].
        dt_s: Sample duration [s], scalar or per-sample array.
        material: Reference :class:`~projects.digital_instron_v2.core.Material`
            supplying the parameter scales and the fixed constitutive constants.
        geometry: Column bed geometry.
        scale0: Initial parameter scale (defaults to ones).
        fit_mask: Optional length-4 mask selecting which parameters to optimize;
            zeros hold a parameter fixed at its ``scale0`` value (defaults to all
            ones). Pin the Maxwell overstress ratio here when it is measured
            separately from a stress-relaxation test -- the force-displacement
            curve constrains stiffness and overstress along a strongly correlated
            direction, so freeing both makes the fit ill-conditioned.
        iterations: Number of Adam iterations.
        learning_rate: Adam step size in scale space.
        device: Warp device.

    Returns:
        The fitted material vector, scale, loss history, and fitted force curve.
    """
    replay = InstronReplay(displacement_m, dt_s, material, geometry, device=device)
    ref = replay.reference
    target = wp.array(np.ascontiguousarray(target_force, np.float32), dtype=wp.float32, device=device)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    scale = np.ones(4, np.float32) if scale0 is None else np.ascontiguousarray(scale0, np.float32).copy()
    mask = np.ones(4, np.float64) if fit_mask is None else np.ascontiguousarray(fit_mask, np.float64)
    m1 = np.zeros(4, np.float64)
    m2 = np.zeros(4, np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    history = np.empty(iterations, np.float32)

    for it in range(iterations):
        replay.set_material(scale * ref)
        replay.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            force = replay.forward()
            wp.launch(_force_mse, dim=replay.nsteps, inputs=[force, target, loss], device=device)
        tape.backward(loss)
        history[it] = float(loss.numpy()[0])
        grad = replay.material_params.grad.numpy() * ref  # chain rule into scale space
        tape.zero()

        g = grad.astype(np.float64) * mask  # freeze masked-out parameters
        m1 = beta1 * m1 + (1.0 - beta1) * g
        m2 = beta2 * m2 + (1.0 - beta2) * g * g
        m1_hat = m1 / (1.0 - beta1 ** (it + 1))
        m2_hat = m2 / (1.0 - beta2 ** (it + 1))
        scale = (scale - learning_rate * m1_hat / (np.sqrt(m2_hat) + eps)).astype(np.float32)

    replay.set_material(scale * ref)
    force = replay.forward().numpy().copy()
    return FitResult(material_params=scale * ref, scale=scale, loss_history=history, force=force)


def _triangular_cycle(peak_m: float, samples: int) -> np.ndarray:
    """Symmetric load/unload compression ramp from 0 to ``peak_m`` and back [m]."""
    half = samples // 2
    up = np.linspace(0.0, peak_m, half, endpoint=False)
    down = np.linspace(peak_m, 0.0, samples - half)
    return np.concatenate([up, down]).astype(np.float32)


def _demo() -> None:
    """Recover a perturbed foam material from its own synthetic Instron curve."""
    MANIFEST = "DigitalInstron/manifest_v2.json"
    wp.init()
    device = wp.get_device("cuda:0") if wp.get_cuda_device_count() else wp.get_device("cpu")
    material = dynamics.load_fitted_material(MANIFEST)
    geometry = dynamics.build_foundation_geometry(MANIFEST)

    displacement = _triangular_cycle(peak_m=0.006, samples=120)
    dt = 1.0 / 200.0  # 200 Hz Instron sampling
    names = ["g_eq", "alpha", "overstress", "pasternak"]

    # Synthesize the "measured" load/unload curve from the true material (scale = 1).
    truth = InstronReplay(displacement, dt, material, geometry, device=device)
    target = truth.forward().numpy().copy()
    print(f"columns={truth.column_count}  samples={truth.nsteps}  peak force={target.max():.2f} N")

    # Realistic workflow: the Maxwell overstress ratio comes from a separate
    # stress-relaxation test, so pin it and fit the elastic/geometry parameters
    # (equilibrium modulus, Hyperfoam exponent, Pasternak coupling) to the curve.
    scale0 = np.array([1.3, 0.85, 1.0, 0.7], np.float32)  # 15-30% error; overstress known
    fit_mask = np.array([1.0, 1.0, 0.0, 1.0], np.float32)
    print(f"\nfit (overstress pinned)  start scale = {scale0}")
    result = fit_material_to_force_curve(
        target,
        displacement,
        dt,
        material,
        geometry,
        scale0=scale0,
        fit_mask=fit_mask,
        iterations=600,
        device=device,
    )
    print(f"  final loss = {result.loss_history[-1]:.3e} N^2  (start {result.loss_history[0]:.3e})")
    print("  recovered scale (true 1.0):")
    for name, s, fit_it in zip(names, result.scale, fit_mask, strict=True):
        tag = "" if fit_it else "  (pinned)"
        print(f"    {name:10s} {s:.4f}{tag}")
    rms = float(np.sqrt(np.mean((result.force - target) ** 2)))
    print(f"  force RMS residual = {rms:.4e} N  ({rms / target.max() * 100:.4f}% of peak)")

    # Freeing all four parameters matches the force curve just as well but leaves a
    # strongly correlated stiffness/overstress direction: the curve is fit but those
    # two parameters are not separately identifiable from a single-rate cycle.
    joint = fit_material_to_force_curve(
        target,
        displacement,
        dt,
        material,
        geometry,
        scale0=np.array([1.3, 0.85, 1.2, 0.7], np.float32),
        iterations=600,
        device=device,
    )
    jrms = float(np.sqrt(np.mean((joint.force - target) ** 2)))
    print(
        f"\njoint 4-parameter fit  final loss = {joint.loss_history[-1]:.3e} N^2  "
        f"(force residual {jrms / target.max() * 100:.4f}% of peak)"
    )
    print("  recovered scale (true 1.0):")
    for name, s in zip(names, joint.scale, strict=True):
        print(f"    {name:10s} {s:.4f}")
    print(
        "  -> g_eq and overstress trade off along an ill-conditioned valley; pin overstress (above) for a unique fit."
    )


if __name__ == "__main__":
    _demo()
