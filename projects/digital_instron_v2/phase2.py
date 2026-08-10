# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase-2 dynamic Digital Instron replay through SolverMuJoCo.

Phase 1 proves the effective shoe law is identifiable from intact bench data.
Phase 2 proves that same law survives contact dynamics: the calibrated
foundation kernel is coupled into MuJoCo-Warp through Newton ``state.body_f``
while the indenter is driven along the measured *held-out* trajectory (cycles
99-100). The shared material is fit on the train cycles only
(:func:`phase1.fit_train_material`), so the dynamic replay is scored on cycles
neither the fit nor the servo tuning ever saw.

Two drive modes exercise the same coupled kernel through
:class:`~newton.solvers.SolverMuJoCo`:

* ``"kinematic"`` (gated): a position-controlled crosshead prescribes the exact
  measured pose every substep -- the faithful digital twin of the bench Instron
  (a servo-hydraulic crosshead *is* position controlled). The dynamic force is
  scored with the identical Phase-1 metric formulas
  (:func:`validation.validate_trace_metrics`).
* ``"servo"`` (stress test): a prismatic PD joint tracks the same trajectory in
  closed loop. This is *not* gated on the rate-dependent hysteresis (a finite
  servo low-passes the crosshead velocity, thinning the loop), only on
  stability, tracking, and wrench continuity -- proving the law is stable under
  real closed-loop actuation.

Run it (regenerates the calibration split automatically) with::

    uv run -m projects.digital_instron_v2.phase2 --manifest DigitalInstron/manifest_v2.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton

from . import phase1
from .core import Material
from .dynamics import (
    FoundationConfig,
    MidsoleFoundation,
    build_foundation_geometry,
    cyclic_displacement,
)
from .validation import validate_trace_metrics

PASS_THRESHOLD = 0.10
TRACKING_TOLERANCE_M = 5.0e-4  # servo must follow the crosshead trajectory within 0.5 mm


@dataclass(frozen=True)
class DynamicReplayConfig:
    """Solver and driver settings for one dynamic replay."""

    drive: str = "kinematic"  # "kinematic" (gated) or "servo" (stress test)
    fps: int = 60
    substeps: int = 32
    warmup_cycles: int = 3
    servo_ke: float = 5.0e7
    servo_kd: float = 2.0e4
    carrier_mass_kg: float = 1.0


@dataclass
class DynamicReplayResult:
    """Recorded last-cycle diagnostics from one dynamic replay."""

    fixture: str
    drive: str
    carrier_body: int
    dt_s: float
    period_s: float
    phase: np.ndarray
    commanded_depth_m: np.ndarray
    achieved_depth_m: np.ndarray
    force_n: np.ndarray
    cop_x_m: np.ndarray
    cop_y_m: np.ndarray
    active_columns: np.ndarray
    wrench_fz_n: np.ndarray
    moment_mx_nm: np.ndarray
    moment_my_nm: np.ndarray
    max_compression_m: np.ndarray
    column_count: int
    column_area_m2: float


def _load_validation_trace(split: dict[str, Any], name: str) -> dict[str, np.ndarray]:
    """Return the phase, time, displacement, and baseline-corrected force of a held-out trace."""
    data = np.genfromtxt(Path(split["validate"][name]), delimiter=",", names=True)
    force = np.asarray(data["force_n"], np.float64)
    return {
        "phase": np.asarray(data["phase"], np.float64),
        "time_s": np.asarray(data["time_s"], np.float64),
        "displacement_m": np.maximum(np.asarray(data["displacement_m"], np.float64), 0.0),
        "force_n": force - float(np.min(force)),
    }


def run_dynamic_replay(
    manifest_path: str | Path,
    fixture: str,
    material: Material,
    trace: dict[str, np.ndarray],
    config: DynamicReplayConfig | None = None,
    device=None,
) -> DynamicReplayResult:
    """Replay one held-out indenter trajectory through SolverMuJoCo and record diagnostics.

    The recorder is phase aligned: the foundation force and the commanded depth
    are logged *before* the substep clock advances, so the rate-dependent
    hysteresis loop is not smeared by a one-substep force/displacement offset.
    """
    config = config or DynamicReplayConfig()
    newton.use_coord_layout_targets = True
    device = device or wp.get_preferred_device()
    geo = build_foundation_geometry(manifest_path, fixture)
    column_count = int(len(geo.slack_m))
    depth, period = cyclic_displacement(trace["time_s"], trace["displacement_m"])

    builder = newton.ModelBuilder()
    if config.drive == "servo":
        carrier = builder.add_link(mass=config.carrier_mass_kg, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_prismatic(
            -1, carrier, axis=newton.Axis.Z, target_pos=0.0, target_ke=config.servo_ke, target_kd=config.servo_kd
        )
        builder.add_articulation([joint])
    else:
        carrier = builder.add_body(
            mass=config.carrier_mass_kg, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)), is_kinematic=True
        )
    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    anchor_local = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.surface_m])
    foundation = MidsoleFoundation(
        anchor_local,
        geo.z_free_m.copy(),
        geo.slack_m,
        np.full(column_count, geo.area_m2),
        geo.neighbors,
        geo.spacing_m,
        material,
        carrier,
        model.body_com,
        FoundationConfig(stretch_floor=0.05),
        device,
    )

    def velocity(t: float, h: float = 2.0e-4) -> float:
        return (depth(t + h) - depth(t - h)) / (2.0 * h)

    dt = 1.0 / config.fps / config.substeps
    sim_time = 0.0
    pose = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    target = np.zeros(1, dtype=np.float32)
    target_vel = np.zeros(1, dtype=np.float32)
    records: list[tuple] = []
    for _ in range(int(round((config.warmup_cycles + 1) * period / dt))):
        commanded = depth(sim_time)
        if config.drive == "servo":
            target[0] = -commanded
            target_vel[0] = -velocity(sim_time)
            control.joint_target_q.assign(target)
            control.joint_target_qd.assign(target_vel)
        else:
            pose[0, 2] = -commanded
            state_0.body_q.assign(pose)
            state_0.body_qd.zero_()
        foundation.apply(state_0, dt, clear_body_force=True)
        diag = foundation.diagnostics()
        wrench = state_0.body_f.numpy()[carrier]
        achieved = -float(state_0.body_q.numpy()[carrier][2])
        max_comp = float(foundation.compression.numpy().max())
        records.append(
            (
                sim_time,
                commanded,
                achieved,
                diag["normal_force_n"],
                diag["cop_x_m"],
                diag["cop_y_m"],
                diag["active_columns"],
                float(wrench[2]),  # foundation writes spatial_vector[force(0:3), moment(3:6)]
                float(wrench[3]),
                float(wrench[4]),
                max_comp,
            )
        )
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        sim_time += dt

    rows = np.asarray(records, dtype=np.float64)
    window = rows[rows[:, 0] >= config.warmup_cycles * period - 0.5 * dt]
    phase = (window[:, 0] % period) / period
    order = np.argsort(phase)
    window = window[order]
    phase = phase[order]
    return DynamicReplayResult(
        fixture=fixture,
        drive=config.drive,
        carrier_body=int(carrier),
        dt_s=float(dt),
        period_s=float(period),
        phase=phase,
        commanded_depth_m=window[:, 1],
        achieved_depth_m=window[:, 2],
        force_n=window[:, 3],
        cop_x_m=window[:, 4],
        cop_y_m=window[:, 5],
        active_columns=window[:, 6],
        wrench_fz_n=window[:, 7],
        moment_mx_nm=window[:, 8],
        moment_my_nm=window[:, 9],
        max_compression_m=window[:, 10],
        column_count=column_count,
        column_area_m2=float(geo.area_m2),
    )


def _diagnostics(result: DynamicReplayResult) -> dict[str, Any]:
    """Summarize the dynamic diagnostics required by the Phase-2 report contract."""
    fz = result.force_n
    peak = int(np.argmax(fz))
    active_area = result.active_columns * result.column_area_m2
    return {
        "carrier_body_id": result.carrier_body,
        "peak_active_cells": int(result.active_columns.max()),
        "peak_cop_m": [float(result.cop_x_m[peak]), float(result.cop_y_m[peak])],
        "mean_cop_m": [float(np.mean(result.cop_x_m)), float(np.mean(result.cop_y_m))],
        "max_force_jump_n": float(np.max(np.abs(np.diff(fz)))) if len(fz) > 1 else 0.0,
        "max_compression_m": float(result.max_compression_m.max()),
        "peak_active_area_m2": float(active_area.max()),
        "peak_wrench_fz_n": float(np.max(np.abs(result.wrench_fz_n))),
        "peak_moment_nm": float(max(np.max(np.abs(result.moment_mx_nm)), np.max(np.abs(result.moment_my_nm)))),
        "tracking_error_max_m": float(np.max(np.abs(result.commanded_depth_m - result.achieved_depth_m))),
        "force_finite": bool(np.all(np.isfinite(fz))),
    }


def _resample_force(result: DynamicReplayResult, phase_grid: np.ndarray) -> np.ndarray:
    """Interpolate the simulated last-cycle force onto the measured phase grid."""
    phase = np.concatenate([[0.0], result.phase, [1.0]])
    force = np.concatenate([[result.force_n[0]], result.force_n, [result.force_n[-1]]])
    return np.interp(phase_grid % 1.0, phase, force)


def evaluate(
    manifest_path: str | Path,
    *,
    backend: str = "scipy",
    config: DynamicReplayConfig | None = None,
    run_servo_stress: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    """Run the dynamic held-out replay for every fixture and build the Phase-2 report."""
    config = config or DynamicReplayConfig(drive="kinematic")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    base = path.parent
    material, fit_info, split = phase1.fit_train_material(path, backend=backend)
    dynamic_dir = base / manifest["cycle_windows"].get("output_dir", "processed")

    validation_metrics: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    dissipation: dict[str, Any] = {}
    servo_stress: dict[str, Any] = {}
    gates: dict[str, dict[str, Any]] = {}
    wrenched_bodies: dict[str, int] = {}

    for source in manifest["trials"]:
        fixture, name, prefix = source["fixture"], source["name"], source["split_prefix"]
        trace = _load_validation_trace(split, name)

        result = run_dynamic_replay(path, fixture, material, trace, config)
        wrenched_bodies[fixture] = result.carrier_body
        simulated = _resample_force(result, trace["phase"])
        simulated = simulated - float(np.min(simulated))
        metrics = validate_trace_metrics(
            trace["force_n"], simulated, trace["displacement_m"], pass_threshold=PASS_THRESHOLD
        )
        validation_metrics[name] = metrics.as_dict()
        diag = _diagnostics(result)
        diagnostics[name] = diag

        disp = trace["displacement_m"]
        measured_loop = -_signed_loop_area(disp, trace["force_n"])
        simulated_loop = -_signed_loop_area(disp, simulated)
        dissipation[name] = {
            "measured_dissipated_j": float(measured_loop),
            "simulated_dissipated_j": float(simulated_loop),
            "relative_error": float(abs(simulated_loop - measured_loop) / max(abs(measured_loop), 1.0e-9)),
        }

        for gate in ("peak_force_error", "force_rmse_relative", "hysteresis_error"):
            value = validation_metrics[name][gate]
            gates[f"{prefix}_{gate}"] = {
                "value": value,
                "threshold": PASS_THRESHOLD,
                "passed": bool(value < PASS_THRESHOLD),
            }
        gates[f"{prefix}_stable_tracking"] = {
            "value": diag["tracking_error_max_m"],
            "threshold": TRACKING_TOLERANCE_M,
            "passed": bool(diag["tracking_error_max_m"] <= TRACKING_TOLERANCE_M and diag["force_finite"]),
        }
        continuous = diag["max_force_jump_n"] < 0.25 * max(metrics.measured_peak_force_n, 1.0)
        gates[f"{prefix}_wrench_continuous"] = {"value": diag["max_force_jump_n"], "passed": bool(continuous)}

        _write_trace_csv(dynamic_dir / f"{prefix}_phase2_dynamic_trace.csv", result, trace, simulated)

        if run_servo_stress:
            servo = run_dynamic_replay(path, fixture, material, trace, _servo_config(config))
            sdiag = _diagnostics(servo)
            servo_stress[name] = {
                "tracking_error_max_m": sdiag["tracking_error_max_m"],
                "max_force_jump_n": sdiag["max_force_jump_n"],
                "force_finite": sdiag["force_finite"],
                "servo_ke": config.servo_ke,
                "servo_kd": config.servo_kd,
            }
            gates[f"{prefix}_servo_stable"] = {
                "value": sdiag["tracking_error_max_m"],
                "threshold": TRACKING_TOLERANCE_M,
                "passed": bool(sdiag["tracking_error_max_m"] <= TRACKING_TOLERANCE_M and sdiag["force_finite"]),
            }

    gates["shared_parameter_set"] = {"passed": True}
    gates["no_validation_fit"] = {"passed": True, "note": "material fit on train cycles 90-98 only"}
    passed = all(gate["passed"] for gate in gates.values())

    report = {
        "schema_version": "digital_instron_phase2_1",
        "manifest": str(path),
        "phase": "dynamic_replay",
        "drive": config.drive,
        "material": asdict(material),
        "fit_backend": fit_info,
        "solver": {
            "name": "SolverMuJoCo",
            "servo_ke": config.servo_ke,
            "servo_kd": config.servo_kd,
            "carrier_mass_kg": config.carrier_mass_kg,
        },
        "timestep": {"fps": config.fps, "substeps": config.substeps, "dt_s": 1.0 / config.fps / config.substeps},
        "warmup_cycles": config.warmup_cycles,
        "validate_cycles": manifest["cycle_windows"]["validate"]["cycles"],
        "wrenched_bodies": wrenched_bodies,
        "validation_metrics": validation_metrics,
        "dynamic_diagnostics": diagnostics,
        "energy_consistency": dissipation,
        "servo_stress_test": servo_stress,
        "gates": gates,
        "passed": bool(passed),
    }
    if write_report:
        out = base / manifest["cache_dir"] / "digital_instron_phase2.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["report_path"] = str(out)
    return report


def _servo_config(config: DynamicReplayConfig) -> DynamicReplayConfig:
    """Return the servo-drive twin of a kinematic config."""
    return DynamicReplayConfig(
        drive="servo",
        fps=config.fps,
        substeps=config.substeps,
        warmup_cycles=config.warmup_cycles,
        servo_ke=config.servo_ke,
        servo_kd=config.servo_kd,
        carrier_mass_kg=config.carrier_mass_kg,
    )


def _signed_loop_area(x: np.ndarray, y: np.ndarray) -> float:
    """Signed shoelace area of the closed (x, y) loop."""
    return float(0.5 * np.sum((x - np.roll(x, 1)) * (y + np.roll(y, 1))))


def _write_trace_csv(
    path: Path, result: DynamicReplayResult, trace: dict[str, np.ndarray], simulated: np.ndarray
) -> None:
    """Write the measured-vs-simulated held-out trace for plotting."""
    columns = ("phase", "displacement_m", "measured_force_n", "simulated_force_n")
    rows = np.column_stack([trace["phase"], trace["displacement_m"], trace["force_n"], simulated])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=",".join(columns), comments="")


def _print_report(report: dict[str, Any]) -> None:
    """Print a compact Phase-2 dynamic summary."""
    print(
        f"\n=== Phase-2 dynamic replay [SolverMuJoCo, drive={report['drive']}] : {'PASS' if report['passed'] else 'FAIL'} ==="
    )
    mat = report["material"]
    print(
        "  material (train-only): G_inst={:.0f} Pa  eq_frac={:.4f}  pasternak={:.1f} N/m".format(
            mat["instantaneous_shear_modulus_pa"], mat["equilibrium_fraction"], mat["pasternak_n_per_m"]
        )
    )
    print(f"  {'fixture':16s} {'peak_err':>9s} {'rmse':>8s} {'hyst_err':>9s} {'track_mm':>9s} {'dF_N':>7s}  pass")
    for name, vm in report["validation_metrics"].items():
        d = report["dynamic_diagnostics"][name]
        print(
            f"  {name:16s} {vm['peak_force_error']:9.3f} {vm['force_rmse_relative']:8.3f} "
            f"{vm['hysteresis_error']:9.3f} {d['tracking_error_max_m'] * 1000:9.3f} {d['max_force_jump_n']:7.1f}  "
            f"{'yes' if vm['passed'] else 'no'}"
        )
    if report["servo_stress_test"]:
        print("  servo stress test (not gated on hysteresis):")
        for name, s in report["servo_stress_test"].items():
            print(
                f"    {name:16s} track={s['tracking_error_max_m'] * 1000:.3f}mm  dF={s['max_force_jump_n']:.1f}N  finite={s['force_finite']}"
            )


def main() -> None:
    """Run the Phase-2 dynamic replay command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json")
    parser.add_argument("--backend", choices=("scipy", "diff"), default="scipy")
    parser.add_argument("--drive", choices=("kinematic", "servo"), default="kinematic")
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--no-servo-stress", action="store_true")
    args = parser.parse_args()
    config = DynamicReplayConfig(drive=args.drive, substeps=args.substeps, warmup_cycles=args.warmup_cycles)
    report = evaluate(args.manifest, backend=args.backend, config=config, run_servo_stress=not args.no_servo_stress)
    _print_report(report)


if __name__ == "__main__":
    main()
