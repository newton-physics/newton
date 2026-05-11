"""Objective functions for Digital Instron material calibration."""

from __future__ import annotations

from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]

from .diagnostics import residual_motion_metrics


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def loop_metrics(
    displacement_m: np.ndarray,
    measured_force_n: np.ndarray,
    sim_force_n: np.ndarray,
    hysteresis_tol: float = 1.0e-6,
) -> dict[str, float]:
    peak_idx = int(np.argmax(displacement_m))
    velocity = np.gradient(displacement_m)
    loading = velocity > hysteresis_tol
    unloading = velocity < -hysteresis_tol

    split_idx = peak_idx
    is_loading = True
    for idx, is_loading_sample in enumerate(loading):
        if is_loading_sample:
            is_loading = True
        elif unloading[idx] and is_loading:
            split_idx = idx if idx > 0 and np.isclose(displacement_m[idx], displacement_m[idx - 1]) else max(idx - 1, 0)
            is_loading = False
    measured_loop_signed = trapz(measured_force_n, displacement_m)
    sim_loop_signed = trapz(sim_force_n, displacement_m)
    force_error = sim_force_n - measured_force_n
    force_sse = float(np.sum(force_error * force_error))
    measured_centered = measured_force_n - float(np.mean(measured_force_n))
    force_sst = float(np.sum(measured_centered * measured_centered))
    force_r2 = 1.0 - force_sse / force_sst if force_sst > 1.0e-12 else float(force_sse <= 1.0e-12)

    measured_loading_work = trapz(measured_force_n[: split_idx + 1], displacement_m[: split_idx + 1])
    sim_loading_work = trapz(sim_force_n[: split_idx + 1], displacement_m[: split_idx + 1])
    measured_unloading_work = -trapz(measured_force_n[split_idx:], displacement_m[split_idx:])
    sim_unloading_work = -trapz(sim_force_n[split_idx:], displacement_m[split_idx:])

    measured_dissipated = measured_loading_work - measured_unloading_work
    sim_dissipated = sim_loading_work - sim_unloading_work

    return {
        "sim_peak_displacement_m": float(np.max(displacement_m)),
        "sim_peak_displacement_mm": float(np.max(displacement_m) * 1000.0),
        "measured_peak_displacement_m": float(np.max(displacement_m)),
        "measured_peak_displacement_mm": float(np.max(displacement_m) * 1000.0),
        "peak_displacement_error_m": 0.0,
        "peak_displacement_error_mm": 0.0,
        "peak_displacement_m": float(np.max(displacement_m)),
        "peak_displacement_mm": float(np.max(displacement_m) * 1000.0),
        "peak_displacement_index": float(peak_idx),
        "force_r2": float(force_r2),
        "measured_loop_signed_j": measured_loop_signed,
        "sim_loop_signed_j": sim_loop_signed,
        "measured_hysteresis_j": float(abs(measured_loop_signed)),
        "sim_hysteresis_j": float(abs(sim_loop_signed)),
        "hysteresis_error_j": float(abs(sim_loop_signed) - abs(measured_loop_signed)),
        "measured_loading_work_j": measured_loading_work,
        "sim_loading_work_j": sim_loading_work,
        "measured_unloading_work_j": measured_unloading_work,
        "sim_unloading_work_j": sim_unloading_work,
        "measured_dissipated_work_j": measured_dissipated,
        "sim_dissipated_work_j": sim_dissipated,
        "loading_rmse_n": float(
            np.sqrt(np.mean((sim_force_n[: split_idx + 1] - measured_force_n[: split_idx + 1]) ** 2))
        ),
        "unloading_rmse_n": float(np.sqrt(np.mean((sim_force_n[split_idx:] - measured_force_n[split_idx:]) ** 2))),
    }


def force_fit_objective(
    displacement_m: np.ndarray,
    measured_force_n: np.ndarray,
    sim_force_n: np.ndarray,
) -> tuple[float, float, float]:
    measured_peak = float(np.max(measured_force_n))
    sim_peak = float(np.max(sim_force_n))
    peak_error = abs(sim_peak - measured_peak) / max(measured_peak, 1.0e-9)
    error = sim_force_n - measured_force_n
    integrated_sse = abs(trapz(error * error, displacement_m))
    displacement_range = max(float(displacement_m[-1] - displacement_m[0]), 1.0e-9)
    loop_rmse = float(np.sqrt(integrated_sse / displacement_range))
    objective = 0.5 * peak_error + 0.5 * (loop_rmse / max(measured_peak, 1.0e-9))
    return objective, peak_error, loop_rmse


def trial_objective_components(response: Any, args: Any) -> dict[str, float | str]:
    force_objective, peak_error, loop_rmse = force_fit_objective(
        response.displacement_m,
        response.measured_force_n,
        response.raw_contact_force_n,
    )
    metrics = loop_metrics(response.displacement_m, response.measured_force_n, response.raw_contact_force_n)
    motion_metrics = residual_motion_metrics(response.contact_stats)

    measured_peak = max(float(np.max(response.measured_force_n)), 1.0e-9)
    measured_hysteresis = max(abs(float(metrics["measured_hysteresis_j"])), 1.0e-9)
    vz_target = max(float(getattr(args, "goal_shoe_vz_target", 1.0e-3)), 1.0e-12)
    omega_target = max(float(getattr(args, "goal_shoe_omega_target", 1.0e-2)), 1.0e-12)

    rmse_relative = float(loop_rmse / measured_peak)
    r2_loss = float(max(0.0, 1.0 - float(metrics["force_r2"])))
    hysteresis_relative_error = float(abs(float(metrics["hysteresis_error_j"])) / measured_hysteresis)
    shoe_vz_relative = float(motion_metrics["rms_shoe_vertical_velocity_m_s"] / vz_target)
    shoe_omega_relative = float(motion_metrics["rms_shoe_free_axis_angular_velocity_rad_s"] / omega_target)

    goal_objective = (
        float(getattr(args, "goal_rmse_weight", 1.0)) * rmse_relative
        + float(getattr(args, "goal_r2_weight", 1.0)) * r2_loss
        + float(getattr(args, "goal_hysteresis_weight", 1.0)) * hysteresis_relative_error
        + float(getattr(args, "goal_shoe_vz_weight", 0.25)) * shoe_vz_relative
        + float(getattr(args, "goal_shoe_omega_weight", 0.25)) * shoe_omega_relative
    )

    return {
        "test_case": response.test_case,
        "force_objective": float(force_objective),
        "goal_objective": float(goal_objective),
        "peak_relative_error": float(peak_error),
        "rmse_relative": rmse_relative,
        "loop_rmse_n": float(loop_rmse),
        "force_r2": float(metrics["force_r2"]),
        "r2_loss": r2_loss,
        "hysteresis_relative_error": hysteresis_relative_error,
        "hysteresis_error_j": float(metrics["hysteresis_error_j"]),
        "measured_hysteresis_j": float(metrics["measured_hysteresis_j"]),
        "sim_hysteresis_j": float(metrics["sim_hysteresis_j"]),
        "shoe_vz_relative": shoe_vz_relative,
        "shoe_omega_relative": shoe_omega_relative,
        "rms_shoe_vertical_velocity_m_s": float(motion_metrics["rms_shoe_vertical_velocity_m_s"]),
        "rms_shoe_free_axis_angular_velocity_rad_s": float(motion_metrics["rms_shoe_free_axis_angular_velocity_rad_s"]),
    }


def mean_numeric_components(components: list[dict[str, float | str]]) -> dict[str, float]:
    numeric_keys = {
        key for component in components for key, value in component.items() if isinstance(value, int | float)
    }
    return {
        key: float(np.mean([float(component[key]) for component in components if key in component]))
        for key in sorted(numeric_keys)
    }


def shared_material_objective_breakdown(responses: list[Any], args: Any) -> tuple[float, dict[str, Any]]:
    mode = str(getattr(args, "calibration_objective", "force"))
    components = [trial_objective_components(response, args) for response in responses]
    if not components:
        return float("inf"), {
            "mode": mode,
            "combined_objective": float("inf"),
            "trials": {},
            "mean": {},
        }

    mean_components = mean_numeric_components(components)
    objective_key = "goal_objective" if mode == "goal" else "force_objective"
    combined = float(mean_components[objective_key])
    breakdown = {
        "mode": mode,
        "combined_objective": combined,
        "selected_component": objective_key,
        "weights": {
            "rmse": float(getattr(args, "goal_rmse_weight", 1.0)),
            "r2": float(getattr(args, "goal_r2_weight", 1.0)),
            "hysteresis": float(getattr(args, "goal_hysteresis_weight", 1.0)),
            "shoe_vz": float(getattr(args, "goal_shoe_vz_weight", 0.25)),
            "shoe_omega": float(getattr(args, "goal_shoe_omega_weight", 0.25)),
        },
        "targets": {
            "shoe_vz_m_s": float(getattr(args, "goal_shoe_vz_target", 1.0e-3)),
            "shoe_omega_rad_s": float(getattr(args, "goal_shoe_omega_target", 1.0e-2)),
        },
        "mean": mean_components,
        "trials": {str(component["test_case"]): component for component in components},
    }
    return combined, breakdown


def shared_material_objective(responses: list[Any], args: Any | None = None) -> float:
    if args is not None:
        combined, _breakdown = shared_material_objective_breakdown(responses, args)
        return combined
    objectives = []
    for response in responses:
        objective, _peak_error, _loop_rmse = force_fit_objective(
            response.displacement_m,
            response.measured_force_n,
            response.raw_contact_force_n,
        )
        objectives.append(objective)
    return float(np.mean(objectives))
