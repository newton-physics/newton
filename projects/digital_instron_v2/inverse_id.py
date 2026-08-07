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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from . import dynamics, workflow
from .core import EFFECTIVE_POISSON_RATIO, MAXWELL_RELAXATION_TIME_S, Material, metrics, predict
from .dynamics import FoundationParams
from .dynamics_diff import (
    MAT_ALPHA,
    MAT_G_EQ,
    MAT_OVERSTRESS,
    MAT_PASTERNAK,
    _hyperfoam_pressure_diff,
    foundation_pressure_diff,
)
from .geometry import build_column_grid, load_mesh


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


# ---------------------------------------------------------------------------
# Fitting the foam material to measured Instron trials (shaped indenters).
#
# Unlike :class:`InstronReplay`, which drives a synthetic *uniform* platen, the
# real digital-Instron fixtures are shaped indenters (a spherical rearfoot punch
# and a full-foot shoe last), so each column sees its own compression history.
# :class:`DifferentiableTrial` is the autodiff sibling of
# :func:`~projects.digital_instron_v2.core.predict`: it reads the per-column
# strain history baked into a :class:`~projects.digital_instron_v2.core.Trial`
# and reproduces that forward model -- Hyperfoam equilibrium pressure, the exact
# periodic generalized-Maxwell overstress fixed point, and the precomputed
# Pasternak coupling -- as a differentiable function of the foam material.
# ---------------------------------------------------------------------------


@wp.kernel
def _trial_equilibrium_pressure(
    strain: wp.array2d[wp.float32],
    frame: wp.int32,
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    peq_out: wp.array[wp.float32],
):
    """Hyperfoam equilibrium pressure for every column at one measured frame."""
    i = wp.tid()
    g_eq = material_params[MAT_G_EQ]
    alpha = material_params[MAT_ALPHA]
    peq_out[i] = _hyperfoam_pressure_diff(strain[frame, i], g_eq, alpha, params)


@wp.kernel
def _trial_maxwell_step(
    peq_cur: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    decay: wp.float32,
    ramp: wp.float32,
    material_params: wp.array[wp.float32],
    q_prev: wp.array[wp.float32],
    q_out: wp.array[wp.float32],
):
    """One tape-safe linear-overstress recurrence step (writes ``q_out`` from ``q_prev``)."""
    i = wp.tid()
    overstress = material_params[MAT_OVERSTRESS]
    q_out[i] = decay * q_prev[i] + overstress * ramp * (peq_cur[i] - peq_prev[i])


@wp.kernel
def _scale_state(state: wp.array[wp.float32], factor: wp.float32, out: wp.array[wp.float32]):
    """Scale the transient end state into the periodic fixed-point initial state."""
    i = wp.tid()
    out[i] = state[i] * factor


@wp.kernel
def _trial_frame_force(
    peq_cur: wp.array[wp.float32],
    q_cur: wp.array[wp.float32],
    laplacian: wp.array2d[wp.float32],
    frame: wp.int32,
    area: wp.array[wp.float32],
    material_params: wp.array[wp.float32],
    force_hist: wp.array[wp.float32],
):
    """Sum the Pasternak-coupled column forces for one measured frame into ``force_hist[frame]``."""
    i = wp.tid()
    pasternak = material_params[MAT_PASTERNAK]
    pressure = peq_cur[i] + q_cur[i] - pasternak * laplacian[frame, i]
    if pressure < 0.0:
        pressure = 0.0
    wp.atomic_add(force_hist, frame, pressure * area[i])


@wp.kernel
def _weighted_force_mse(
    force_hist: wp.array[wp.float32],
    target: wp.array[wp.float32],
    weight: wp.float32,
    loss: wp.array[wp.float32],
):
    """Accumulate a per-trial-weighted mean-square force residual into ``loss``."""
    k = wp.tid()
    d = force_hist[k] - target[k]
    wp.atomic_add(loss, 0, weight * d * d)


def _params_from_material(material: Material) -> np.ndarray:
    """Pack a :class:`~projects.digital_instron_v2.core.Material` into ``[g_eq, alpha, overstress, pasternak]``."""
    return np.array(
        [
            material.instantaneous_shear_modulus_pa * material.equilibrium_fraction,
            material.hyperfoam_exponent,
            (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction,
            material.pasternak_n_per_m,
        ],
        np.float32,
    )


def _material_from_params(material_params: np.ndarray) -> Material:
    """Unpack a ``[g_eq, alpha, overstress, pasternak]`` vector back into a :class:`~projects.digital_instron_v2.core.Material`."""
    g_eq, alpha, overstress, pasternak = (float(value) for value in material_params)
    equilibrium_fraction = 1.0 / (1.0 + overstress)
    return Material(
        instantaneous_shear_modulus_pa=g_eq / equilibrium_fraction,
        hyperfoam_exponent=alpha,
        equilibrium_fraction=equilibrium_fraction,
        pasternak_n_per_m=pasternak,
    )


@dataclass
class TrialFitResult:
    """Outcome of fitting one foam material to measured Instron trials.

    Attributes:
        material: Fitted :class:`~projects.digital_instron_v2.core.Material`.
        material_params: Fitted ``[g_eq, alpha, overstress, pasternak]`` vector
            ([Pa], [-], [-], [N/m]).
        scale: Fitted dimensionless scale of each parameter relative to the
            reference material used to start the fit.
        loss_history: Weighted mean-square force residual [N^2] at each iteration.
        rms_relative: Final per-trial force RMS residual as a fraction of that
            trial's peak measured force, keyed by trial name.
    """

    material: Material
    material_params: np.ndarray
    scale: np.ndarray
    loss_history: np.ndarray
    rms_relative: dict[str, float]


class DifferentiableTrial:
    """Autodiff force predictor for one measured Instron trial (shaped indenter).

    Reproduces :func:`~projects.digital_instron_v2.core.predict` as a
    differentiable function of the foam ``material_params``: the per-column strain
    history baked into the :class:`~projects.digital_instron_v2.core.Trial` drives
    the Hyperfoam equilibrium pressure, the exact periodic generalized-Maxwell
    overstress fixed point (a zero-state transient pass followed by a fixed-point
    pass, matching the cyclic steady state that ``predict`` evaluates in closed
    form), and the precomputed Pasternak coupling. Every buffer on the loss path
    is written exactly once so the whole cycle records on one :class:`warp.Tape`.

    Args:
        trial: Measured :class:`~projects.digital_instron_v2.core.Trial` with
            per-column ``lengths_m`` and ``compression_laplacian_m_inv`` histories.
        material: Reference :class:`~projects.digital_instron_v2.core.Material`
            supplying the parameter scales and fixed constitutive constants.
        device: Warp device.
        material_params: Optional shared ``requires_grad`` material vector; when
            given (e.g. by :func:`fit_material_to_trials` to fit one material to
            several trials at once) it is used instead of a private copy.
    """

    def __init__(self, trial, material: Material, device=None, material_params: wp.array | None = None):
        self.device = device
        self.name = trial.name
        slack = np.asarray(trial.slack_m, np.float32)
        lengths = np.ascontiguousarray(trial.lengths_m, np.float32)
        self.frame_count, self.column_count = lengths.shape
        strain = np.maximum(slack[None, :] - lengths, 0.0) / slack[None, :]
        self.strain = wp.array(np.ascontiguousarray(strain, np.float32), dtype=wp.float32, device=device)
        laplacian = trial.compression_laplacian_m_inv
        laplacian = np.zeros_like(lengths) if laplacian is None else np.ascontiguousarray(laplacian, np.float32)
        self.laplacian = wp.array(laplacian, dtype=wp.float32, device=device)
        area = (
            np.full(self.column_count, float(trial.area_m2), np.float32)
            if np.isscalar(trial.area_m2)
            else np.ascontiguousarray(trial.area_m2, np.float32)
        )
        self.area = wp.array(area, dtype=wp.float32, device=device)

        dt = trial.dt_s
        dt = np.full(self.frame_count, float(dt), np.float64) if np.isscalar(dt) else np.asarray(dt, np.float64)
        decay = np.exp(-dt / MAXWELL_RELAXATION_TIME_S)
        self.decay = decay
        self.ramp = MAXWELL_RELAXATION_TIME_S * (1.0 - decay) / dt
        self.fixed_point_gain = float(1.0 / (1.0 - float(np.prod(decay))))

        params = FoundationParams()
        params.beta = EFFECTIVE_POISSON_RATIO / (1.0 - 2.0 * EFFECTIVE_POISSON_RATIO)
        params.stretch_floor = 1.0e-3  # match core.predict's stretch clamp
        self.params = params

        self.reference = _params_from_material(material)
        self.owns_material = material_params is None
        self.material_params = (
            wp.array(self.reference.copy(), dtype=wp.float32, device=device, requires_grad=True)
            if self.owns_material
            else material_params
        )

        def grad_zeros():
            return wp.zeros(self.column_count, dtype=wp.float32, device=device, requires_grad=True)

        self.peq = [grad_zeros() for _ in range(self.frame_count)]
        self.q_transient = [grad_zeros() for _ in range(self.frame_count)]
        self.q_cycle = [grad_zeros() for _ in range(self.frame_count)]
        self.q_zero = grad_zeros()  # transient-pass initial state (stays zero)
        self.q_init = grad_zeros()  # fixed-point initial state for the cycle pass
        self.force = wp.zeros(self.frame_count, dtype=wp.float32, device=device, requires_grad=True)
        self.measured = np.ascontiguousarray(trial.force_n, np.float32)
        self.peak = max(float(np.max(np.abs(self.measured))), 1.0e-9)

    def set_material(self, material_params: np.ndarray) -> None:
        """Overwrite the differentiable material vector with host values."""
        self.material_params.assign(np.ascontiguousarray(material_params, np.float32))

    def forward(self) -> wp.array:
        """Replay the measured cycle and return the per-frame reaction force [N]."""
        self.force.zero_()
        for f in range(self.frame_count):
            wp.launch(
                _trial_equilibrium_pressure,
                dim=self.column_count,
                inputs=[self.strain, f, self.params, self.material_params, self.peq[f]],
                device=self.device,
            )
        # Transient pass from a zero overstress state to find the cycle end state.
        for f in range(self.frame_count):
            q_prev = self.q_zero if f == 0 else self.q_transient[f - 1]
            peq_prev = self.peq[self.frame_count - 1] if f == 0 else self.peq[f - 1]
            wp.launch(
                _trial_maxwell_step,
                dim=self.column_count,
                inputs=[
                    self.peq[f],
                    peq_prev,
                    float(self.decay[f]),
                    float(self.ramp[f]),
                    self.material_params,
                    q_prev,
                    self.q_transient[f],
                ],
                device=self.device,
            )
        wp.launch(
            _scale_state,
            dim=self.column_count,
            inputs=[self.q_transient[self.frame_count - 1], self.fixed_point_gain, self.q_init],
            device=self.device,
        )
        # Fixed-point pass: the same recurrence from the periodic initial state, summing force.
        for f in range(self.frame_count):
            q_prev = self.q_init if f == 0 else self.q_cycle[f - 1]
            peq_prev = self.peq[self.frame_count - 1] if f == 0 else self.peq[f - 1]
            wp.launch(
                _trial_maxwell_step,
                dim=self.column_count,
                inputs=[
                    self.peq[f],
                    peq_prev,
                    float(self.decay[f]),
                    float(self.ramp[f]),
                    self.material_params,
                    q_prev,
                    self.q_cycle[f],
                ],
                device=self.device,
            )
            wp.launch(
                _trial_frame_force,
                dim=self.column_count,
                inputs=[self.peq[f], self.q_cycle[f], self.laplacian, f, self.area, self.material_params, self.force],
                device=self.device,
            )
        return self.force

    def zero_grad(self) -> None:
        """Zero the gradients on every differentiable buffer owned by this trial."""
        if self.owns_material:
            self.material_params.grad.zero_()
        self.force.grad.zero_()
        for buf in (*self.peq, *self.q_transient, *self.q_cycle):
            buf.grad.zero_()
        self.q_zero.grad.zero_()
        self.q_init.grad.zero_()


def fit_material_to_trials(
    trials,
    initial: Material,
    scale0=None,
    fit_mask=None,
    per_trial_weights=None,
    iterations: int = 200,
    learning_rate: float = 0.03,
    device=None,
) -> TrialFitResult:
    """Fit one foam material to measured Instron trials with exact gradients.

    Descends a peak-normalized joint force-matching loss with Adam, sharing one
    differentiable material vector across every trial so a single backward pass
    yields the gradient with respect to all four parameters at once. By default
    each trial is weighted by ``1 / peak_force^2`` -- reproducing the per-trial
    residual normalization of the shipped scipy calibration
    (:func:`~projects.digital_instron_v2.core.fit_material`) -- so a low-force
    fixture is not swamped by a high-force one.

    Args:
        trials: Measured :class:`~projects.digital_instron_v2.core.Trial` objects.
        initial: Reference :class:`~projects.digital_instron_v2.core.Material`
            supplying the parameter scales and fixed constitutive constants.
        scale0: Initial per-parameter scale (defaults to ones).
        fit_mask: Optional length-4 mask selecting which of
            ``[g_eq, alpha, overstress, pasternak]`` to optimize; zeros hold a
            parameter fixed at its ``scale0`` value.
        per_trial_weights: Optional per-trial loss weights (defaults to
            ``1 / peak_force^2`` for a peak-normalized joint fit).
        iterations: Number of Adam iterations.
        learning_rate: Adam step size in scale space.
        device: Warp device.

    Returns:
        The fitted material, its scale, the loss history, and the final per-trial
        force RMS residual as a fraction of each trial's peak force.
    """
    reference = _params_from_material(initial)
    shared = wp.array(reference.copy(), dtype=wp.float32, device=device, requires_grad=True)
    predictors = [DifferentiableTrial(trial, initial, device=device, material_params=shared) for trial in trials]
    targets = [wp.array(p.measured, dtype=wp.float32, device=device) for p in predictors]
    if per_trial_weights is None:
        weights = np.array([1.0 / p.peak**2 for p in predictors], np.float64)
    else:
        weights = np.ascontiguousarray(per_trial_weights, np.float64)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    scale = np.ones(4, np.float32) if scale0 is None else np.ascontiguousarray(scale0, np.float32).copy()
    mask = np.ones(4, np.float64) if fit_mask is None else np.ascontiguousarray(fit_mask, np.float64)
    m1 = np.zeros(4, np.float64)
    m2 = np.zeros(4, np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    history = np.empty(iterations, np.float32)

    for it in range(iterations):
        shared.assign(scale * reference)
        shared.grad.zero_()
        loss.zero_()
        for p in predictors:
            p.zero_grad()
        tape = wp.Tape()
        with tape:
            for p, target, weight in zip(predictors, targets, weights, strict=True):
                force = p.forward()
                wp.launch(
                    _weighted_force_mse, dim=p.frame_count, inputs=[force, target, float(weight), loss], device=device
                )
        tape.backward(loss)
        history[it] = float(loss.numpy()[0])
        grad = shared.grad.numpy() * reference  # chain rule into scale space
        tape.zero()

        g = grad.astype(np.float64) * mask
        m1 = beta1 * m1 + (1.0 - beta1) * g
        m2 = beta2 * m2 + (1.0 - beta2) * g * g
        m1_hat = m1 / (1.0 - beta1 ** (it + 1))
        m2_hat = m2 / (1.0 - beta2 ** (it + 1))
        scale = (scale - learning_rate * m1_hat / (np.sqrt(m2_hat) + eps)).astype(np.float32)

    shared.assign(scale * reference)
    rms_relative = {}
    for p in predictors:
        force = p.forward().numpy()
        rms_relative[p.name] = float(np.sqrt(np.mean((force - p.measured) ** 2)) / p.peak)
    fitted = scale * reference
    return TrialFitResult(
        material=_material_from_params(fitted),
        material_params=fitted,
        scale=scale,
        loss_history=history,
        rms_relative=rms_relative,
    )


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


def _demo_measured() -> None:
    """Fit the foam material to the two measured Instron fixtures with exact gradients.

    Loads the averaged rearfoot-punch and full-foot shoe-last trials, reproduces
    the shipped scipy calibration by a peak-normalized *joint* fit, then shows the
    lower residual each fixture reaches on its own -- the single homogeneous
    material is in genuine tension between the two loading regions.
    """
    base = Path("DigitalInstron")
    if not (base / "manifest_v2.json").exists():
        print("DigitalInstron dataset not found; skipping measured-trial demo.")
        return
    config = json.loads((base / "manifest_v2.json").read_text())
    midsole = load_mesh(str(base / config["midsole_mesh"]), 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
    trials, _, _ = workflow.prepare_trials(base, config, grid, midsole)
    material = dynamics.load_fitted_material("DigitalInstron/manifest_v2.json")
    device = wp.get_device("cuda:0") if wp.get_cuda_device_count() else wp.get_device("cpu")

    print("\nshipped calibration (reference material):")
    for trial in trials:
        rmse = metrics(trial.force_n, predict(trial, material), trial.displacement_m)["force_rmse_relative"]
        print(f"    {trial.name:16s} force RMS = {rmse * 100:.3f}% of peak")

    joint = fit_material_to_trials(trials, material, iterations=250, learning_rate=0.01, device=device)
    print(f"\npeak-normalized joint fit (scale, true 1.0): {np.round(joint.scale, 4)}")
    for name, rms in joint.rms_relative.items():
        print(f"    {name:16s} force RMS = {rms * 100:.3f}% of peak")
    print("  -> reproduces the shipped scipy calibration; the shared material is the joint optimum.")

    for trial in trials:
        per_trial = fit_material_to_trials([trial], material, iterations=250, learning_rate=0.01, device=device)
        rms = per_trial.rms_relative[trial.name]
        print(
            f"  per-fixture fit {trial.name:16s} force RMS = {rms * 100:.3f}% of peak  scale = {np.round(per_trial.scale, 3)}"
        )
    print(
        "  -> each fixture reaches a lower residual alone but demands a different stiffness (single-material tension)."
    )


if __name__ == "__main__":
    _demo()
    _demo_measured()
