# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 effective-property identification with a held-out cycle split.

Closes the Phase-1 loop described in ``plans/plan.md``: the raw Digital Instron
CSVs are the source of truth, so this module regenerates train (cycles ``90-98``)
and held-out validation (cycles ``99-100``) traces from them, fits one shared
shoe material to the train traces, and reports the official baseline-corrected
peak/RMSE/hysteresis metrics on the *held-out* cycles the fit never saw.

Two calibration backends share the identical train/validate protocol so exact
gradients can be compared against the derivative-free reference:

* ``"scipy"`` -- the shipped :func:`~projects.digital_instron_v2.core.fit_material`
  least-squares fit.
* ``"diff"`` -- the differentiable
  :func:`~projects.digital_instron_v2.inverse_id.fit_material_to_trials` Adam fit
  with exact gradients (imports Warp lazily).

Run the held-out evaluation and write the validation report::

    uv run -m projects.digital_instron_v2.phase1 --manifest DigitalInstron/manifest_v2.json --backend scipy
    uv run -m projects.digital_instron_v2.phase1 --manifest DigitalInstron/manifest_v2.json --backend diff
    uv run -m projects.digital_instron_v2.phase1 --manifest DigitalInstron/manifest_v2.json --compare
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .core import EFFECTIVE_POISSON_RATIO, MAXWELL_RELAXATION_TIME_S, Material, fit_material, predict
from .cycle_windows import build_cycle_window_trace, write_cycle_window_trace
from .frame_qc import FrameConfig, infer_frame_config
from .geometry import build_column_grid, load_mesh
from .validation import validate_trace_metrics
from .workflow import prepare_trials

PASS_THRESHOLD = 0.10


def _frame_config_for(base: Path, source: dict[str, Any]) -> FrameConfig:
    """Return the saved or inferred frame configuration for one trial."""
    saved = source.get("frame_config")
    if isinstance(saved, dict):
        return FrameConfig(
            time_column=saved["time_column"],
            position_column=saved["position_column"],
            force_column=saved["force_column"],
            position_sign=float(saved["position_sign"]),
            force_sign=float(saved["force_sign"]),
            displacement_zero=float(saved.get("displacement_zero", 0.0)),
        )
    return infer_frame_config(base / source["raw_csv_path"])


def generate_split_traces(manifest_path: str | Path) -> dict[str, Any]:
    """Generate train and held-out validation traces from the raw CSVs.

    Writes ``<prefix>_<train_suffix>.csv`` and ``<prefix>_<validate_suffix>.csv``
    for every trial plus a ``cycle_window_split.json`` provenance summary, and
    returns the resolved per-trial train/validate paths and provenance.
    """
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    base = path.parent
    windows = config["cycle_windows"]
    out_dir = base / windows.get("output_dir", "processed")
    phase_count = int(windows.get("phase_count", 501))
    policy = windows.get("displacement_zero_policy", "top_of_stroke")

    result: dict[str, Any] = {"train": {}, "validate": {}, "provenance": {}}
    for source in config["trials"]:
        frame = _frame_config_for(base, source)
        raw = base / source["raw_csv_path"]
        prefix = source["split_prefix"]
        result["provenance"][source["name"]] = {"frame_config": frame.as_dict()}
        for split in ("train", "validate"):
            trace = build_cycle_window_trace(
                raw,
                frame,
                cycles=windows[split]["cycles"],
                phase_count=phase_count,
                displacement_zero_policy=policy,
            )
            out = out_dir / f"{prefix}_{windows[split]['suffix']}.csv"
            write_cycle_window_trace(out, trace)
            result[split][source["name"]] = out
            result["provenance"][source["name"]][split] = {
                "csv": str(out.relative_to(base)),
                **trace.provenance,
            }

    summary = out_dir / "cycle_window_split.json"
    summary.write_text(json.dumps(result["provenance"], indent=2, sort_keys=True) + "\n")
    result["summary"] = summary
    return result


def _fit_backend(
    trials: list,
    initial: Material,
    backend: str,
    *,
    evaluations: int,
    iterations: int,
    learning_rate: float,
) -> tuple[Material, dict[str, Any]]:
    """Fit one shared material on the train trials with the selected backend."""
    if backend == "scipy":
        history: list[dict[str, float]] = []
        material = fit_material(trials, initial, evaluations, history)
        return material, {"backend": "scipy", "accepted_iterations": max(len(history) - 1, 0)}
    if backend == "diff":
        from .inverse_id import fit_material_to_trials  # noqa: PLC0415  # lazy: pulls in Warp

        fit = fit_material_to_trials(trials, initial, iterations=iterations, learning_rate=learning_rate)
        info = {
            "backend": "diff",
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "scale": [float(value) for value in fit.scale],
            "final_loss": float(fit.loss_history[-1]),
            "train_rms_relative": {name: float(value) for name, value in fit.rms_relative.items()},
        }
        return fit.material, info
    raise ValueError(f"unknown backend {backend!r} (use 'scipy' or 'diff')")


def _trace_metrics(trial, material: Material) -> dict[str, float | int | bool]:
    """Official held-out metrics for one trial under a fitted material."""
    measured = np.asarray(trial.force_n, np.float64)
    measured = measured - float(np.min(measured))  # inactive-baseline correction
    simulated = np.asarray(predict(trial, material), np.float64)
    simulated = simulated - float(np.min(simulated))
    displacement = np.asarray(trial.displacement_m, np.float64)
    return validate_trace_metrics(measured, simulated, displacement, pass_threshold=PASS_THRESHOLD).as_dict()


def _nuisance_parameters(source: dict[str, Any], policy: str) -> dict[str, Any]:
    """Record the allowed nuisance parameters used for one fixture."""
    used: dict[str, Any] = {
        "force_zero_policy": "inactive_baseline_min",
        "displacement_zero_policy": policy,
    }
    indenter = source.get("indenter", {})
    if source["fixture"] == "rearfoot_punch":
        used["indenter_radius_m"] = indenter.get("radius_m")
    else:
        used["geometry_alignment"] = {
            "pose_rotation_deg": indenter.get("pose_rotation_deg"),
            "pose_translation_m": indenter.get("pose_translation_m"),
            "contact_percentile": indenter.get("contact_percentile"),
            "height_offset_m": indenter.get("height_offset_m"),
        }
    return used


def fit_train_material(
    manifest_path: str | Path,
    *,
    backend: str = "scipy",
    evaluations: int = 100,
    iterations: int = 150,
    learning_rate: float = 0.03,
) -> tuple[Material, dict[str, Any], dict[str, Any]]:
    """Fit one shared material on the train cycles only and return the split provenance.

    Returns:
        The train-fitted :class:`~projects.digital_instron_v2.core.Material`, the
        backend fit-info mapping, and the generated train/validate split (paths and
        provenance) from :func:`generate_split_traces`. Phase 2 reuses this so its
        dynamic replay is scored against a material that never saw the held-out
        cycles.
    """
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    base = path.parent
    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
    split = generate_split_traces(path)
    train_trials, _, _ = prepare_trials(base, config, grid, midsole, trace_paths=split["train"])
    initial = Material(*config["fit"].values())
    material, fit_info = _fit_backend(
        train_trials, initial, backend, evaluations=evaluations, iterations=iterations, learning_rate=learning_rate
    )
    return material, fit_info, split


def evaluate(
    manifest_path: str | Path,
    *,
    backend: str = "scipy",
    evaluations: int = 100,
    iterations: int = 150,
    learning_rate: float = 0.03,
    write_report: bool = True,
) -> dict[str, Any]:
    """Fit on train cycles and validate on held-out cycles; return the report."""
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    base = path.parent
    policy = config["cycle_windows"].get("displacement_zero_policy", "top_of_stroke")

    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])

    material, fit_info, split = fit_train_material(
        path, backend=backend, evaluations=evaluations, iterations=iterations, learning_rate=learning_rate
    )
    train_trials, _, _ = prepare_trials(base, config, grid, midsole, trace_paths=split["train"])
    validate_trials, _, _ = prepare_trials(base, config, grid, midsole, trace_paths=split["validate"])

    by_name = {source["name"]: source for source in config["trials"]}
    train_metrics = {t.name: _trace_metrics(t, material) for t in train_trials}
    validation_metrics = {t.name: _trace_metrics(t, material) for t in validate_trials}

    gates: dict[str, dict[str, Any]] = {}
    for trial in validate_trials:
        prefix = by_name[trial.name]["split_prefix"]
        vm = validation_metrics[trial.name]
        for gate in ("peak_force_error", "force_rmse_relative", "hysteresis_error"):
            gates[f"{prefix}_{gate}"] = {
                "value": vm[gate],
                "threshold": PASS_THRESHOLD,
                "passed": bool(vm[gate] < PASS_THRESHOLD),
            }
    gates["shared_parameter_set"] = {"passed": True}
    passed = all(gate["passed"] for gate in gates.values())

    report = {
        "schema_version": "digital_instron_phase1_1",
        "manifest": str(path),
        "backend": fit_info,
        "model": {
            "type": "reduced_hyperfoam_maxwell_pasternak",
            "effective_poisson_ratio": EFFECTIVE_POISSON_RATIO,
            "maxwell_relaxation_time_s": MAXWELL_RELAXATION_TIME_S,
            "state_initialization": "periodic_cycle_fixed_point",
        },
        "train_cycles": config["cycle_windows"]["train"]["cycles"],
        "validate_cycles": config["cycle_windows"]["validate"]["cycles"],
        "material": asdict(material),
        "nuisance_parameters": {name: _nuisance_parameters(src, policy) for name, src in by_name.items()},
        "traces": split["provenance"],
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "gates": gates,
        "passed": bool(passed),
    }

    if write_report:
        out = base / config["cache_dir"] / f"digital_instron_phase1_{backend}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["report_path"] = str(out)
    return report


def compare_backends(
    manifest_path: str | Path, *, evaluations: int = 100, iterations: int = 150, learning_rate: float = 0.03
) -> dict[str, Any]:
    """Run both backends through the identical protocol and tabulate the gates."""
    reports = {
        backend: evaluate(
            manifest_path,
            backend=backend,
            evaluations=evaluations,
            iterations=iterations,
            learning_rate=learning_rate,
            write_report=True,
        )
        for backend in ("scipy", "diff")
    }
    return reports


def _print_report(report: dict[str, Any]) -> None:
    """Print a compact held-out summary for one backend."""
    backend = report["backend"]["backend"]
    print(f"\n=== Phase-1 held-out validation [{backend}] : {'PASS' if report['passed'] else 'FAIL'} ===")
    mat = report["material"]
    print(
        "  material: G_inst={:.0f} Pa  alpha={:.3f}  eq_frac={:.4f}  pasternak={:.1f} N/m".format(
            mat["instantaneous_shear_modulus_pa"],
            mat["hyperfoam_exponent"],
            mat["equilibrium_fraction"],
            mat["pasternak_n_per_m"],
        )
    )
    header = f"  {'fixture':16s} {'peak_err':>9s} {'rmse':>8s} {'hyst_err':>9s}  pass"
    for scope in ("train_metrics", "validation_metrics"):
        print(f"  -- {scope.replace('_metrics', '')} --")
        print(header)
        for name, vm in report[scope].items():
            print(
                f"  {name:16s} {vm['peak_force_error']:9.3f} {vm['force_rmse_relative']:8.3f} "
                f"{vm['hysteresis_error']:9.3f}  {'yes' if vm['passed'] else 'no'}"
            )


def main() -> None:
    """Run the Phase-1 held-out evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json")
    parser.add_argument("--backend", choices=("scipy", "diff"), default="scipy")
    parser.add_argument("--evaluations", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--compare", action="store_true", help="run both backends and tabulate")
    args = parser.parse_args()

    if args.compare:
        reports = compare_backends(
            args.manifest, evaluations=args.evaluations, iterations=args.iterations, learning_rate=args.learning_rate
        )
        for report in reports.values():
            _print_report(report)
    else:
        report = evaluate(
            args.manifest,
            backend=args.backend,
            evaluations=args.evaluations,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
        _print_report(report)


if __name__ == "__main__":
    main()
