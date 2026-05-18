# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manual/script entrypoint for the experimental Digital Instron v2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .foundation import (
    FoundationFitSample,
    FoundationMaterial,
    evaluate_foundation,
    evaluate_foundation_lengths,
    fit_foundation_material_autodiff,
)
from .frame_qc import infer_frame_config, load_trial_frame
from .geometry import (
    _load_obj_mesh,
    _ray_triangle_z_candidates,
    build_raycast_spring_grid,
    condition_midsole_mesh,
    make_cylinder_grid,
    place_rearfoot_punch_grid,
)
from .manifest import load_manifest
from .sdf_utils import _load_stl_mesh
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
    parser.add_argument(
        "--hysteresis-sample-count",
        type=int,
        default=250,
        help="Maximum frames per trial to replay for the fit-autodiff hysteresis plot",
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


def _initial_material(manifest, *, per_cylinder_area: bool = False) -> FoundationMaterial:
    return FoundationMaterial(
        stiffness_pa=float(manifest.fit.get("initial_stiffness_pa", 2.0e6)),
        ogden_alpha=float(manifest.fit.get("initial_ogden_alpha", 2.0)),
        lock_strain=float(manifest.fit.get("initial_lock_strain", 0.65)),
        damping_pa_s=float(manifest.fit.get("initial_damping_pa_s", 1.0e4)),
        damping_power=float(manifest.fit.get("initial_damping_power", 1.0)),
        per_cylinder_area=per_cylinder_area,
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
        thickness_axis=manifest.grid.get("force_thickness_axis"),
    )
    return spring_grid, vertices


def _rearfoot_mask(manifest, spring_grid, vertices) -> np.ndarray:
    punch = place_rearfoot_punch_grid(
        vertices,
        radius_m=float(
            next((t.indenter.get("radius_m", 0.0225) for t in manifest.trials if t.fixture == "rearfoot_punch"), 0.0225)
        ),
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
        frame=spring_grid.frame,
        heel_side=str(manifest.grid.get("rearfoot_heel_side", "min")),
        length_fraction=float(manifest.grid.get("rearfoot_length_fraction", 0.22)),
        lateral_fraction=float(manifest.grid.get("rearfoot_lateral_fraction", 0.5)),
        lateral_band_fraction=float(manifest.grid.get("rearfoot_lateral_band_fraction", 0.12)),
    )
    dist = np.linalg.norm(spring_grid.grid_uv_m - punch.center_uv_m, axis=1)
    return dist <= punch.radius_m + spring_grid.spacing_m * 0.5


def _rotation_matrix_xyz_deg(rotation_deg: object) -> np.ndarray:
    if rotation_deg is None:
        angles = np.zeros(3, dtype=np.float64)
    else:
        angles = np.asarray(rotation_deg, dtype=np.float64)
    if angles.shape != (3,):
        raise ValueError("indenter.rotation_deg must contain three XYZ angles")
    rx, ry, rz = np.deg2rad(angles)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _indenter_mesh_m(indenter: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    indenter_type = indenter.get("type")
    if indenter_type != "stl":
        raise ValueError(f"Only STL fullfoot indenters are supported here, got {indenter_type!r}")
    path = Path(str(indenter["path"]))
    if path.suffix.lower() == ".obj":
        vertices, faces = _load_obj_mesh(path)
    else:
        vertices, faces = _load_stl_mesh(path)
    units = str(indenter.get("units", "mm"))
    if units == "mm":
        vertices = vertices * 0.001
    elif units != "m":
        raise ValueError("indenter.units must be 'mm' or 'm'")
    rotation = _rotation_matrix_xyz_deg(indenter.get("rotation_deg"))
    vertices = vertices @ rotation.T
    return vertices, faces


def _indenter_contact_surface_m(spring_grid, trial) -> tuple[np.ndarray, np.ndarray]:
    """Return STL contact-surface coordinate along the spring thickness axis."""

    vertices, faces = _indenter_mesh_m(trial.indenter)
    frame = spring_grid.frame
    plane_axes = frame.plane_axes
    thickness_axis = frame.thickness_axis

    indenter_plane = vertices[:, plane_axes]
    spring_plane_center = 0.5 * (np.min(spring_grid.grid_uv_m, axis=0) + np.max(spring_grid.grid_uv_m, axis=0))
    indenter_plane_center = 0.5 * (np.min(indenter_plane, axis=0) + np.max(indenter_plane, axis=0))
    vertices = vertices.copy()
    vertices[:, plane_axes] += spring_plane_center - indenter_plane_center

    plane = vertices[:, plane_axes]
    thickness = vertices[:, thickness_axis]
    triangles_plane = plane[faces.reshape(-1, 3)]
    triangles_thickness = thickness[faces.reshape(-1, 3)]
    tri_min = np.min(triangles_plane, axis=1) - 1.0e-12
    tri_max = np.max(triangles_plane, axis=1) + 1.0e-12

    contact_raw = np.full(len(spring_grid.grid_uv_m), np.nan, dtype=np.float64)
    for i, point in enumerate(spring_grid.grid_uv_m):
        candidate_indices = np.nonzero(
            (tri_min[:, 0] <= point[0])
            & (point[0] <= tri_max[:, 0])
            & (tri_min[:, 1] <= point[1])
            & (point[1] <= tri_max[:, 1])
        )[0]
        hits = _ray_triangle_z_candidates(point, triangles_plane, triangles_thickness, candidate_indices)
        if hits:
            contact_raw[i] = hits[0]

    valid = np.isfinite(contact_raw)
    if not np.any(valid):
        raise ValueError(f"Fullfoot STL indenter {trial.name!r} does not overlap the spring grid")

    initial_clearance = float(trial.indenter.get("initial_clearance_m", 0.0))
    thickness_offset = float(np.max(spring_grid.top_m[valid] - contact_raw[valid]) + initial_clearance)
    return contact_raw + thickness_offset, valid


def _trial_contact_surface_cache(manifest, spring_grid) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trial in manifest.trials:
        if trial.include_in_fit and trial.fixture == "fullfoot_last" and trial.indenter.get("type") == "stl":
            surfaces[trial.name] = _indenter_contact_surface_m(spring_grid, trial)
    return surfaces


def _trial_frame_config(manifest, output_dir: Path, trial) -> dict[str, object]:
    frame_config = trial.frame
    if frame_config is not None:
        return frame_config
    frame_config_path = output_dir / f"{trial.name}.frame_config.json"
    if not frame_config_path.exists():
        frame = infer_frame_config(
            trial.csv_path,
            min_force_span_n=float(manifest.qc.get("min_force_span_n", 50.0)),
            min_position_span_mm=float(manifest.qc.get("min_position_span_mm", 1.0)),
        )
        _write_json(frame_config_path, frame.as_dict())
    return json.loads(frame_config_path.read_text())


def _spring_state_for_trial_frame(
    spring_grid,
    trial,
    rearfoot_mask: np.ndarray,
    contact_surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    displacement_m: float,
    displacement_velocity_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = max(float(displacement_m), 0.0)
    current_length = spring_grid.slack_length_m.copy()
    velocity = np.zeros_like(current_length)
    if trial.fixture == "rearfoot_punch":
        active = rearfoot_mask
        current_length[active] = np.maximum(spring_grid.slack_length_m[active] - displacement, 0.0)
        velocity[active] = -float(displacement_velocity_mps)
        return current_length, velocity

    if trial.fixture == "fullfoot_last" and trial.name in contact_surfaces:
        contact_surface_0, valid = contact_surfaces[trial.name]
        contact_surface = contact_surface_0 - displacement
        compression = np.zeros_like(current_length)
        compression[valid] = np.maximum(spring_grid.top_m[valid] - contact_surface[valid], 0.0)
        contact_active = valid & (compression > 0.0)
        current_length[valid] = np.maximum(spring_grid.slack_length_m[valid] - compression[valid], 0.0)
        velocity[contact_active] = -float(displacement_velocity_mps)
        return current_length, velocity

    active = np.ones_like(rearfoot_mask, dtype=bool)
    current_length[active] = np.maximum(spring_grid.slack_length_m[active] - displacement, 0.0)
    velocity[active] = -float(displacement_velocity_mps)
    return current_length, velocity


def _hysteresis_segments(displacement_m: np.ndarray) -> list[np.ndarray]:
    """Split sampled hysteresis points where downsampling jumps across cycles."""

    displacement = np.asarray(displacement_m, dtype=np.float64)
    if displacement.ndim != 1:
        raise ValueError("displacement_m must be 1D")
    if len(displacement) == 0:
        return []
    if len(displacement) == 1:
        return [np.asarray([0], dtype=np.int64)]

    step = np.abs(np.diff(displacement))
    nonzero = step[step > 1.0e-12]
    median_step = float(np.median(nonzero)) if len(nonzero) else 0.0
    jump_threshold = max(0.002, 8.0 * median_step)
    split_after = np.nonzero(step > jump_threshold)[0] + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), split_after))
    stops = np.concatenate((split_after, np.asarray([len(displacement)], dtype=np.int64)))
    return [np.arange(start, stop, dtype=np.int64) for start, stop in zip(starts, stops, strict=True) if stop > start]


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
    contact_surfaces = _trial_contact_surface_cache(manifest, spring_grid)
    # Per-cylinder cell area (uniform spacing broadcast to all cells)
    n_cells = len(spring_grid.xy_m)
    cell_area = np.full(n_cells, spring_grid.cell_area_m2, dtype=np.float64)
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        frame_config = _trial_frame_config(manifest, output_dir, trial)
        trace = load_trial_frame(trial.csv_path, frame_config)
        displacement_velocity = np.gradient(trace["displacement_m"], trace["time_s"])
        positive = np.nonzero((trace["force_n"] > 0.0) & (trace["displacement_m"] > 0.0))[0]
        if len(positive) == 0:
            continue
        selected = positive[np.linspace(0, len(positive) - 1, min(sample_count, len(positive)), dtype=np.int64)]
        for index in selected:
            current_length, velocity = _spring_state_for_trial_frame(
                spring_grid,
                trial,
                rearfoot_mask,
                contact_surfaces,
                float(trace["displacement_m"][index]),
                float(displacement_velocity[index]),
            )
            samples.append(
                FoundationFitSample(
                    current_length_m=current_length,
                    slack_length_m=spring_grid.slack_length_m,
                    velocity_mps=velocity,
                    measured_force_n=float(trace["force_n"][index]),
                    weight=1.0,
                    cell_area_m2=cell_area,
                )
            )
    return samples


def _write_autodiff_hysteresis_plot(
    manifest,
    output_dir: Path,
    spring_grid,
    vertices,
    material: FoundationMaterial,
    *,
    sample_count: int,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    sample_count = max(int(sample_count), 2)
    rearfoot_mask = _rearfoot_mask(manifest, spring_grid, vertices)
    contact_surfaces = _trial_contact_surface_cache(manifest, spring_grid)
    cell_area = np.full(len(spring_grid.xy_m), spring_grid.cell_area_m2, dtype=np.float64)
    rows: list[tuple[str, int, float, float, float, float, float]] = []
    trial_summaries: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        frame_config = _trial_frame_config(manifest, output_dir, trial)
        trace = load_trial_frame(trial.csv_path, frame_config)
        if len(trace["time_s"]) == 0:
            continue
        displacement_velocity = np.gradient(trace["displacement_m"], trace["time_s"])
        frame_indices = np.linspace(
            0,
            len(trace["time_s"]) - 1,
            min(sample_count, len(trace["time_s"])),
            dtype=np.int64,
        )
        measured = trace["force_n"][frame_indices].astype(np.float64)
        displacement = trace["displacement_m"][frame_indices].astype(np.float64)
        predicted = np.empty_like(measured)
        losses = np.empty_like(measured)

        for out_index, frame_index in enumerate(frame_indices):
            current_length, velocity = _spring_state_for_trial_frame(
                spring_grid,
                trial,
                rearfoot_mask,
                contact_surfaces,
                float(trace["displacement_m"][frame_index]),
                float(displacement_velocity[frame_index]),
            )
            result = evaluate_foundation_lengths(
                spring_grid.xy_m,
                current_length,
                spring_grid.slack_length_m,
                velocity,
                cell_area_m2=cell_area,
                material=material,
                measured_force_n=float(trace["force_n"][frame_index]),
            )
            predicted[out_index] = result.force_n
            losses[out_index] = result.loss
            rows.append(
                (
                    trial.name,
                    int(frame_index),
                    float(trace["time_s"][frame_index]),
                    float(trace["displacement_m"][frame_index]),
                    float(trace["force_n"][frame_index]),
                    result.force_n,
                    result.loss,
                )
            )

        segments = _hysteresis_segments(displacement)
        measured_points = ax.scatter(
            displacement * 1000.0,
            measured,
            s=10.0,
            alpha=0.75,
            label=f"{trial.name} measured",
        )
        predicted_points = ax.scatter(
            displacement * 1000.0,
            predicted,
            s=10.0,
            alpha=0.85,
            marker="x",
            label=f"{trial.name} predicted",
        )
        measured_color = measured_points.get_facecolors()[0]
        predicted_color = predicted_points.get_facecolors()[0]
        for segment in segments:
            if len(segment) < 4:
                continue
            ax.plot(
                displacement[segment] * 1000.0,
                measured[segment],
                linewidth=1.2,
                alpha=0.35,
                color=measured_color,
            )
            ax.plot(
                displacement[segment] * 1000.0,
                predicted[segment],
                linewidth=1.2,
                linestyle="--",
                alpha=0.45,
                color=predicted_color,
            )
        rmse = float(np.sqrt(np.mean((predicted - measured) ** 2)))
        trial_summaries.append(
            {
                "trial": trial.name,
                "frame_count": int(len(frame_indices)),
                "segment_count": int(len(segments)),
                "rmse_n": rmse,
                "measured_peak_force_n": float(np.max(measured)),
                "predicted_peak_force_n": float(np.max(predicted)),
            }
        )

    ax.set_title("fit-autodiff hysteresis replay")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("force [N]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")

    png_path = output_dir / "digital_instron_v2_autodiff_hysteresis.png"
    csv_path = output_dir / "digital_instron_v2_autodiff_hysteresis.csv"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    np.savetxt(
        csv_path,
        np.asarray([row[1:] for row in rows], dtype=np.float64) if rows else np.empty((0, 6), dtype=np.float64),
        delimiter=",",
        header="frame_index,time_s,displacement_m,measured_force_n,predicted_force_n,loss",
        comments="",
    )
    if rows:
        trial_path = output_dir / "digital_instron_v2_autodiff_hysteresis_trials.csv"
        trial_path.write_text(
            "trial,frame_index,time_s,displacement_m,measured_force_n,predicted_force_n,loss\n"
            + "\n".join(
                f"{trial},{frame},{time},{displacement},{measured},{predicted},{loss}"
                for trial, frame, time, displacement, measured, predicted, loss in rows
            )
            + "\n"
        )
    else:
        trial_path = output_dir / "digital_instron_v2_autodiff_hysteresis_trials.csv"
        trial_path.write_text("trial,frame_index,time_s,displacement_m,measured_force_n,predicted_force_n,loss\n")

    return {
        "hysteresis_png": str(png_path),
        "hysteresis_csv": str(csv_path),
        "hysteresis_trials_csv": str(trial_path),
        "trials": trial_summaries,
    }


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
        per_cylinder_area=True,
    )
    hysteresis = _write_autodiff_hysteresis_plot(
        manifest,
        output_dir,
        spring_grid,
        vertices,
        result.material,
        sample_count=int(getattr(args, "hysteresis_sample_count", 250)),
    )
    report = {
        "manifest": str(manifest.path),
        "sample_count": len(samples),
        "spring_grid_cells": int(len(spring_grid.xy_m)),
        "material": result.material.__dict__,
        "history": list(result.history),
        "hysteresis": hysteresis,
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
