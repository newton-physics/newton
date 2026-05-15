# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manual/script entrypoint for the experimental Digital Instron v2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .foundation import FoundationFitSample, FoundationMaterial, evaluate_foundation, fit_foundation_material_autodiff
from .frame_qc import infer_frame_config, load_trial_frame
from .geometry import _load_obj_mesh, build_raycast_spring_grid, condition_midsole_mesh, make_cylinder_grid
from .geometry import place_rearfoot_punch_grid
from .manifest import load_manifest
from .visualization import write_visualization_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json", help="Path to v2 trial manifest")
    parser.add_argument("--output-dir", default=None, help="Directory for QC and summary outputs")
    parser.add_argument("--step", choices=("qc", "fit-smoke", "fit-autodiff", "visualize"), default="qc")
    parser.add_argument("--autodiff-iterations", type=int, default=25, help="Iterations for --step fit-autodiff")
    parser.add_argument(
        "--autodiff-sample-count",
        type=int,
        default=8,
        help="Number of positive-force samples per trial for --step fit-autodiff",
    )
    return parser


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_qc(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    mesh_report = condition_midsole_mesh(
        manifest.midsole_mesh,
        output_dir,
        source_units=str(manifest.qc.get("mesh_source_units", "mm")),
        min_thickness_m=float(manifest.qc.get("min_midsole_thickness_m", 0.005)),
        max_thickness_m=float(manifest.qc.get("max_midsole_thickness_m", 0.08)),
    )

    frame_reports = []
    for trial in manifest.trials:
        frame = infer_frame_config(
            trial.csv_path,
            min_force_span_n=float(manifest.qc.get("min_force_span_n", 50.0)),
            min_position_span_mm=float(manifest.qc.get("min_position_span_mm", 1.0)),
        )
        frame_path = output_dir / f"{trial.name}.frame_config.json"
        _write_json(frame_path, frame.as_dict())
        frame_reports.append({"trial": trial.name, "frame_config": str(frame_path), **frame.as_dict()})

    report = {"manifest": str(manifest.path), "mesh": mesh_report, "frames": frame_reports}
    _write_json(output_dir / "digital_instron_v2_qc.json", report)
    return report


def run_fit_smoke(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    grid = make_cylinder_grid(
        radius_m=float(manifest.grid.get("cylinder_radius_m", 0.0225)),
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
    )
    material = FoundationMaterial(
        stiffness_pa=float(manifest.fit.get("initial_stiffness_pa", 2.0e6)),
        ogden_alpha=float(manifest.fit.get("initial_ogden_alpha", 2.0)),
        lock_strain=float(manifest.fit.get("initial_lock_strain", 0.65)),
        damping_pa_s=float(manifest.fit.get("initial_damping_pa_s", 1.0e4)),
        damping_power=float(manifest.fit.get("initial_damping_power", 1.0)),
    )

    summaries = []
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        frame_config = trial.frame
        if frame_config is None:
            frame_config_path = output_dir / f"{trial.name}.frame_config.json"
            frame_config = json.loads(frame_config_path.read_text())
        trace = load_trial_frame(trial.csv_path, frame_config)
        index = int(np.argmax(trace["force_n"]))
        compression = np.full(len(grid.xy_m), max(float(trace["displacement_m"][index]), 0.0), dtype=np.float64)
        velocity = np.zeros_like(compression)
        result = evaluate_foundation(
            grid.xy_m,
            compression,
            velocity,
            cell_area_m2=grid.cell_area_m2,
            thickness_m=float(manifest.fit.get("nominal_midsole_thickness_m", 0.03)),
            material=material,
            measured_force_n=float(trace["force_n"][index]),
        )
        summaries.append(
            {
                "trial": trial.name,
                "sample": index,
                "measured_force_n": float(trace["force_n"][index]),
                "predicted_force_n": result.force_n,
                "loss": result.loss,
            }
        )

    report = {"manifest": str(manifest.path), "material": material.__dict__, "fit_smoke": summaries}
    _write_json(output_dir / "digital_instron_v2_fit_smoke.json", report)
    return report


def _initial_material(manifest) -> FoundationMaterial:
    return FoundationMaterial(
        stiffness_pa=float(manifest.fit.get("initial_stiffness_pa", 2.0e6)),
        ogden_alpha=float(manifest.fit.get("initial_ogden_alpha", 2.0)),
        lock_strain=float(manifest.fit.get("initial_lock_strain", 0.65)),
        damping_pa_s=float(manifest.fit.get("initial_damping_pa_s", 1.0e4)),
        damping_power=float(manifest.fit.get("initial_damping_power", 1.0)),
    )


def _load_spring_grid(manifest, output_dir: Path):
    mesh_report = condition_midsole_mesh(
        manifest.midsole_mesh,
        output_dir,
        source_units=str(manifest.qc.get("mesh_source_units", "mm")),
        min_thickness_m=float(manifest.qc.get("min_midsole_thickness_m", 0.005)),
        max_thickness_m=float(manifest.qc.get("max_midsole_thickness_m", 0.08)),
    )
    vertices, faces = _load_obj_mesh(Path(str(mesh_report["repaired_mesh"])))
    spring_grid = build_raycast_spring_grid(
        vertices,
        faces,
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
        min_slack_length_m=float(manifest.qc.get("min_spring_slack_length_m", 0.001)),
    )
    return spring_grid, vertices


def _rearfoot_mask(manifest, spring_grid, vertices) -> np.ndarray:
    punch = place_rearfoot_punch_grid(
        vertices,
        radius_m=float(next((t.indenter.get("radius_m", 0.0225) for t in manifest.trials if t.fixture == "rearfoot_punch"), 0.0225)),
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
        frame=spring_grid.frame,
        heel_side=str(manifest.grid.get("rearfoot_heel_side", "min")),
        length_fraction=float(manifest.grid.get("rearfoot_length_fraction", 0.22)),
        lateral_fraction=float(manifest.grid.get("rearfoot_lateral_fraction", 0.5)),
        lateral_band_fraction=float(manifest.grid.get("rearfoot_lateral_band_fraction", 0.12)),
    )
    dist = np.linalg.norm(spring_grid.grid_uv_m - punch.center_uv_m, axis=1)
    return dist <= punch.radius_m + spring_grid.spacing_m * 0.5


def _autodiff_samples(
    manifest,
    output_dir: Path,
    spring_grid,
    vertices,
    *,
    sample_count: int,
) -> list[FoundationFitSample]:
    samples: list[FoundationFitSample] = []
    sample_count = max(int(sample_count), 1)
    rearfoot_mask = _rearfoot_mask(manifest, spring_grid, vertices)
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        frame_config = trial.frame
        if frame_config is None:
            frame_config_path = output_dir / f"{trial.name}.frame_config.json"
            if not frame_config_path.exists():
                frame = infer_frame_config(
                    trial.csv_path,
                    min_force_span_n=float(manifest.qc.get("min_force_span_n", 50.0)),
                    min_position_span_mm=float(manifest.qc.get("min_position_span_mm", 1.0)),
                )
                _write_json(frame_config_path, frame.as_dict())
            frame_config = json.loads(frame_config_path.read_text())
        trace = load_trial_frame(trial.csv_path, frame_config)
        displacement_velocity = np.gradient(trace["displacement_m"], trace["time_s"])
        positive = np.nonzero((trace["force_n"] > 0.0) & (trace["displacement_m"] > 0.0))[0]
        if len(positive) == 0:
            continue
        selected = positive[np.linspace(0, len(positive) - 1, min(sample_count, len(positive)), dtype=np.int64)]
        active = rearfoot_mask if trial.fixture == "rearfoot_punch" else np.ones_like(rearfoot_mask, dtype=bool)
        for index in selected:
            displacement = max(float(trace["displacement_m"][index]), 0.0)
            current_length = spring_grid.slack_length_m.copy()
            current_length[active] = np.maximum(spring_grid.slack_length_m[active] - displacement, 0.0)
            velocity = np.zeros_like(current_length)
            velocity[active] = -float(displacement_velocity[index])
            samples.append(
                FoundationFitSample(
                    current_length_m=current_length,
                    slack_length_m=spring_grid.slack_length_m,
                    velocity_mps=velocity,
                    measured_force_n=float(trace["force_n"][index]),
                    weight=1.0,
                )
            )
    return samples


def run_fit_autodiff(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    spring_grid, vertices = _load_spring_grid(manifest, output_dir)
    samples = _autodiff_samples(
        manifest,
        output_dir,
        spring_grid,
        vertices,
        sample_count=int(args.autodiff_sample_count),
    )
    result = fit_foundation_material_autodiff(
        spring_grid.xy_m,
        samples,
        cell_area_m2=spring_grid.cell_area_m2,
        initial_material=_initial_material(manifest),
        iterations=int(args.autodiff_iterations),
    )
    report = {
        "manifest": str(manifest.path),
        "sample_count": len(samples),
        "spring_grid_cells": int(len(spring_grid.xy_m)),
        "material": result.material.__dict__,
        "history": list(result.history),
    }
    _write_json(output_dir / "digital_instron_v2_autodiff_fit.json", report)
    return report


def run_visualize(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    return write_visualization_report(manifest, output_dir)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.step == "qc":
        report = run_qc(args)
    elif args.step == "fit-smoke":
        report = run_fit_smoke(args)
    elif args.step == "fit-autodiff":
        report = run_fit_autodiff(args)
    else:
        report = run_visualize(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
