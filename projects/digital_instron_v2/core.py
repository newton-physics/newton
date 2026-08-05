# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Digital Instron material model and fit."""

from dataclasses import dataclass

import numpy as np

EFFECTIVE_POISSON_RATIO = 0.30
MAXWELL_RELAXATION_TIME_S = 0.08


@dataclass(frozen=True)
class Material:
    """Reduced Hyperfoam-Maxwell-Pasternak parameters."""

    instantaneous_shear_modulus_pa: float
    hyperfoam_exponent: float
    equilibrium_fraction: float
    pasternak_n_per_m: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(tuple(self.__dict__.values()))):
            raise ValueError("material parameters must be finite")
        if self.instantaneous_shear_modulus_pa <= 0.0 or self.hyperfoam_exponent <= 0.0:
            raise ValueError("shear modulus and Hyperfoam exponent must be positive")
        if not 0.0 < self.equilibrium_fraction <= 1.0:
            raise ValueError("equilibrium fraction must be in (0, 1]")
        if self.pasternak_n_per_m < 0.0:
            raise ValueError("Pasternak coupling must be nonnegative")


@dataclass(frozen=True)
class Trial:
    """Measured force and matching column lengths."""

    name: str
    slack_m: np.ndarray
    area_m2: np.ndarray | float
    lengths_m: np.ndarray
    dt_s: np.ndarray
    force_n: np.ndarray
    displacement_m: np.ndarray
    compression_laplacian_m_inv: np.ndarray | None = None


def _hyperfoam_pressure(strain: np.ndarray, material: Material) -> np.ndarray:
    """Return positive uniaxial compression pressure from first-order Hyperfoam."""

    stretch = np.clip(1.0 - strain, 1.0e-3, 1.0)
    poisson = EFFECTIVE_POISSON_RATIO
    beta = poisson / (1.0 - 2.0 * poisson)
    volume_ratio = stretch ** (1.0 - 2.0 * poisson)
    equilibrium_shear_modulus = material.instantaneous_shear_modulus_pa * material.equilibrium_fraction
    pressure = 2.0 * equilibrium_shear_modulus / (material.hyperfoam_exponent * stretch)
    pressure *= volume_ratio ** (-material.hyperfoam_exponent * beta) - stretch**material.hyperfoam_exponent
    return pressure


def _periodic_maxwell_branch(
    equilibrium_pressure: np.ndarray,
    dt_s: np.ndarray,
    fraction: float,
    relaxation_time_s: float,
) -> np.ndarray:
    """Evaluate one linear overstress branch at its exact cycle fixed point."""

    if fraction == 0.0:
        return np.zeros_like(equilibrium_pressure)
    decay = np.exp(-dt_s / relaxation_time_s)
    ramp = relaxation_time_s * (1.0 - decay) / dt_s
    pressure_increment = equilibrium_pressure - np.roll(equilibrium_pressure, 1, axis=0)
    state = np.zeros(equilibrium_pressure.shape[1])
    for frame in range(len(equilibrium_pressure)):
        state = decay[frame] * state + fraction * ramp[frame] * pressure_increment[frame]
    state /= 1.0 - float(np.prod(decay))
    result = np.empty_like(equilibrium_pressure)
    for frame in range(len(equilibrium_pressure)):
        state = decay[frame] * state + fraction * ramp[frame] * pressure_increment[frame]
        result[frame] = state
    return result


def predict(trial: Trial, material: Material) -> np.ndarray:
    """Predict a trial force history."""

    slack = np.asarray(trial.slack_m)
    strain = np.maximum(slack[None, :] - trial.lengths_m, 0.0) / slack[None, :]
    equilibrium = _hyperfoam_pressure(strain, material)
    pressure = np.array(equilibrium, copy=True)
    maxwell_fraction = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction
    pressure += _periodic_maxwell_branch(equilibrium, trial.dt_s, maxwell_fraction, MAXWELL_RELAXATION_TIME_S)
    if trial.compression_laplacian_m_inv is not None:
        pressure -= material.pasternak_n_per_m * trial.compression_laplacian_m_inv
    return np.sum(trial.area_m2 * np.maximum(pressure, 0.0), axis=1)


def fit_material(
    trials: list[Trial],
    initial: Material,
    evaluations: int,
    history: list[dict[str, float]] | None = None,
) -> Material:
    """Fit one material to all trials."""

    from scipy.optimize import least_squares

    def residual(values: np.ndarray) -> np.ndarray:
        material = Material(*values)
        return np.concatenate(
            [(predict(trial, material) - trial.force_n) / max(float(np.max(trial.force_n)), 1.0) for trial in trials]
        )

    def record(values: np.ndarray) -> None:
        if history is None:
            return
        residuals = residual(values)
        row = {
            "iteration": float(len(history)),
            "loss": float(np.mean(residuals**2)),
            **{name: float(value) for name, value in zip(Material.__dataclass_fields__, values, strict=True)},
        }
        offset = 0
        for trial in trials:
            count = len(trial.force_n)
            row[f"loss_{trial.name}"] = float(np.mean(residuals[offset : offset + count] ** 2))
            offset += count
        history.append(row)

    x0 = np.asarray(list(initial.__dict__.values()))
    record(x0)
    lower = [1.0e3, 0.1, 0.01, 0.0]
    upper = [1.0e8, 20.0, 1.0, 1.0e5]
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=evaluations,
        callback=lambda intermediate: record(np.asarray(getattr(intermediate, "x", intermediate))),
    )
    if history is not None and not np.array_equal(
        result.x, [history[-1][name] for name in Material.__dataclass_fields__]
    ):
        record(result.x)
    return Material(*result.x)


def metrics(measured: np.ndarray, predicted: np.ndarray, displacement: np.ndarray) -> dict[str, float | bool]:
    """Return peak, RMSE, and hysteresis errors."""

    peak = max(float(np.max(measured)), 1.0e-9)
    peak_frame = int(np.argmax(displacement))
    values = {
        "peak_force_error": abs(float(np.max(predicted)) - peak) / peak,
        "force_rmse_relative": float(np.sqrt(np.mean((predicted - measured) ** 2)) / peak),
        "loading_rmse_relative": float(
            np.sqrt(np.mean((predicted[: peak_frame + 1] - measured[: peak_frame + 1]) ** 2)) / peak
        ),
        "unloading_rmse_relative": float(
            np.sqrt(np.mean((predicted[peak_frame:] - measured[peak_frame:]) ** 2)) / peak
        ),
        "hysteresis_error": abs(float(np.trapezoid(predicted - measured, displacement)))
        / max(abs(float(np.trapezoid(measured, displacement))), 1.0e-9),
    }
    values["passed"] = all(value < 0.1 for value in values.values())
    return values
