# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manual/script entrypoint for the experimental Digital Instron v2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cycle_windows import build_cycle_window_trace, write_cycle_window_trace
from .foundation import (
    FoundationMaterial,
    FoundationTrialBatch,
    evaluate_foundation,
    evaluate_foundation_lengths,
    evaluate_foundation_lengths_batch,
    fit_foundation_material_batches_autodiff,
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
from .mujoco_adapter import apply_foundation_wrench_to_body_f
from .sdf_utils import _load_stl_mesh
from .validation import active_force_mask, validate_trace_metrics
from .visualization import write_visualization_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json", help="Path to v2 trial manifest")
    parser.add_argument("--output-dir", default=None, help="Directory for QC and summary outputs")
    parser.add_argument(
        "--step",
        choices=("qc", "split-cycles", "fit-smoke", "fit-autodiff", "fit-validate", "dynamic-replay", "visualize"),
        default="qc",
    )
    parser.add_argument("--train-cycles", default="90-98", help="Cycle window for split-cycles or fit-validate")
    parser.add_argument(
        "--validate-cycles", default="99-100", help="Held-out cycle window for split-cycles or fit-validate"
    )
    parser.add_argument("--cycle-phase-count", type=int, default=501, help="Phase samples per generated cycle trace")
    parser.add_argument("--autodiff-iterations", type=int, default=25, help="Iterations for --step fit-autodiff")
    parser.add_argument("--loop-weight", type=float, default=None, help="Override fit.loop_weight for autodiff fitting")
    parser.add_argument(
        "--autodiff-sample-count",
        type=int,
        default=8,
        help="Deprecated; fit-autodiff now uses every frame in each averaged-cycle CSV",
    )
    parser.add_argument("--autodiff-device", default="cuda:0", help="Warp device for --step fit-autodiff")
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
        pasternak_stiffness_n_per_m=float(
            manifest.fit.get(
                "initial_pasternak_stiffness_n_per_m",
                manifest.fit.get("initial_shear_modulus_pa", 0.0),
            )
        ),
        spatial_slope=float(manifest.fit.get("initial_spatial_slope", 0.0)),
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
        if trial.fixture == "rearfoot_punch":
            radius_m = float(trial.indenter.get("radius_m", 0.0225))
            analytical_area = np.pi * (radius_m**2)
            active_count = len(grid.xy_m)
            cell_area_val = analytical_area / active_count if active_count > 0 else grid.cell_area_m2
        else:
            cell_area_val = grid.cell_area_m2

        result = evaluate_foundation(
            grid.xy_m,
            compression,
            velocity,
            cell_area_m2=cell_area_val,
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
    stiffness = float(manifest.fit.get("initial_stiffness_pa", 2.0e6))
    if "initial_prony_stiffness_pa" in manifest.fit:
        prony_stiffness = float(manifest.fit.get("initial_prony_stiffness_pa", 0.0))
        prony_damping = float(manifest.fit.get("initial_prony_damping_pa_s", 0.0))
    elif "state_beta" in manifest.fit:
        beta = float(manifest.fit.get("state_beta", 0.0))
        tau = float(manifest.fit.get("state_tau_s", 0.05))
        prony_stiffness = beta * stiffness
        prony_damping = tau * prony_stiffness
    else:
        prony_stiffness = 0.0
        prony_damping = 0.0

    return FoundationMaterial(
        stiffness_pa=stiffness,
        ogden_alpha=float(manifest.fit.get("initial_ogden_alpha", 2.0)),
        lock_strain=float(manifest.fit.get("initial_lock_strain", 0.65)),
        damping_pa_s=float(manifest.fit.get("initial_damping_pa_s", 1.0e4)),
        damping_power=float(manifest.fit.get("initial_damping_power", 1.0)),
        per_cylinder_area=per_cylinder_area,
        prony_stiffness_pa=prony_stiffness,
        prony_damping_pa_s=prony_damping,
        state_warmup_cycles=int(manifest.fit.get("state_warmup_cycles", 0)),
        pasternak_stiffness_n_per_m=float(
            manifest.fit.get(
                "initial_pasternak_stiffness_n_per_m",
                manifest.fit.get("initial_shear_modulus_pa", 0.0),
            )
        ),
        spatial_slope=float(manifest.fit.get("initial_spatial_slope", 0.0)),
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
    height_offset = float(trial.indenter.get("height_offset_m", 0.0))
    contact_percentile = float(trial.indenter.get("contact_percentile", 75.0))
    clearance_raw = spring_grid.top_m[valid] - contact_raw[valid]
    thickness_offset = float(np.percentile(clearance_raw, contact_percentile) + initial_clearance + height_offset)
    contact_surface = contact_raw + thickness_offset
    contact_surface[valid] = np.maximum(contact_surface[valid], spring_grid.top_m[valid])
    return contact_surface, valid


def _trial_contact_surface_cache(manifest, spring_grid) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trial in manifest.trials:
        if trial.include_in_fit and trial.fixture == "fullfoot_last" and trial.indenter.get("type") == "stl":
            surfaces[trial.name] = _indenter_contact_surface_m(spring_grid, trial)
    return surfaces


def _trial_displacement_split(trial) -> tuple[float, float]:
    """Return top/bottom fractions of the measured platen displacement."""

    indenter = getattr(trial, "indenter", {}) or {}
    top_fraction = float(indenter.get("top_displacement_fraction", 0.5))
    bottom_fraction = float(indenter.get("bottom_displacement_fraction", 0.5))
    if top_fraction <= 0.0 or bottom_fraction <= 0.0:
        raise ValueError("top and bottom indenter displacement fractions must both be positive")
    return top_fraction, bottom_fraction


def _bottom_platen_compression(
    spring_grid,
    active_mask: np.ndarray,
    bottom_travel_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compression from a flat bottom platen moving toward the midsole."""

    compression = np.zeros_like(spring_grid.slack_length_m)
    active = np.asarray(active_mask, dtype=bool)
    if bottom_travel_m <= 0.0 or not np.any(active):
        return compression, np.zeros_like(active, dtype=bool)

    bottom_plane_m = float(np.min(spring_grid.bottom_m[active]) + bottom_travel_m)
    compression[active] = np.maximum(bottom_plane_m - spring_grid.bottom_m[active], 0.0)
    return compression, active & (compression > 0.0)


def _spring_compression_components_for_trial_frame(
    spring_grid,
    trial,
    rearfoot_mask: np.ndarray,
    contact_surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    displacement_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return top and bottom spring-compression components for one frame."""

    displacement = max(float(displacement_m), 0.0)
    top_fraction, bottom_fraction = _trial_displacement_split(trial)
    top_travel = top_fraction * displacement
    bottom_travel = bottom_fraction * displacement

    top_compression = np.zeros_like(spring_grid.slack_length_m)
    bottom_compression = np.zeros_like(spring_grid.slack_length_m)
    top_active = np.zeros_like(top_compression, dtype=bool)
    bottom_active = np.zeros_like(top_compression, dtype=bool)

    if trial.fixture == "rearfoot_punch":
        active = np.asarray(rearfoot_mask, dtype=bool)
        top_compression[active] = top_travel
        top_active = active & (top_compression > 0.0)
        bottom_compression, bottom_active = _bottom_platen_compression(spring_grid, active, bottom_travel)
        return top_compression, bottom_compression, top_active, bottom_active

    if trial.fixture == "fullfoot_last" and trial.name in contact_surfaces:
        contact_surface_0, valid = contact_surfaces[trial.name]
        contact_surface = contact_surface_0 - top_travel
        top_compression[valid] = np.maximum(spring_grid.top_m[valid] - contact_surface[valid], 0.0)
        top_active = valid & (top_compression > 0.0)
        bottom_compression, bottom_active = _bottom_platen_compression(spring_grid, valid, bottom_travel)
        return top_compression, bottom_compression, top_active, bottom_active

    active = np.ones_like(rearfoot_mask, dtype=bool)
    top_compression[active] = top_travel
    top_active = active & (top_compression > 0.0)
    bottom_compression, bottom_active = _bottom_platen_compression(spring_grid, active, bottom_travel)
    return top_compression, bottom_compression, top_active, bottom_active


def _fullfoot_contact_diagnostics(
    spring_grid,
    contact_surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    displacement_m: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Compute fullfoot STL contact coverage diagnostics at a displacement."""

    diagnostics: dict[str, dict[str, float]] = {}
    for trial_name, (contact_surface_0, valid) in contact_surfaces.items():
        contact_surface = contact_surface_0 - float(displacement_m)
        clearance_mm = (spring_grid.top_m[valid] - contact_surface[valid]) * 1000.0
        active_count = int(np.count_nonzero(clearance_mm > 0.0))
        valid_count = int(np.count_nonzero(valid))
        if valid_count:
            percentiles = np.percentile(clearance_mm, [1.0, 5.0, 50.0, 95.0, 99.0])
            contact_min_mm = float(np.min(clearance_mm))
            contact_max_mm = float(np.max(clearance_mm))
        else:
            percentiles = np.full(5, np.nan, dtype=np.float64)
            contact_min_mm = float("nan")
            contact_max_mm = float("nan")
        diagnostics[trial_name] = {
            "valid_count": float(valid_count),
            "active_count": float(active_count),
            "valid_area_mm2": float(valid_count * spring_grid.cell_area_m2 * 1.0e6),
            "active_area_mm2": float(active_count * spring_grid.cell_area_m2 * 1.0e6),
            "contact_min_mm": contact_min_mm,
            "contact_p01_mm": float(percentiles[0]),
            "contact_p05_mm": float(percentiles[1]),
            "contact_p50_mm": float(percentiles[2]),
            "contact_p95_mm": float(percentiles[3]),
            "contact_p99_mm": float(percentiles[4]),
            "contact_max_mm": contact_max_mm,
        }
    return diagnostics


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


def _parse_cycle_window(value: str) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        raise ValueError("Cycle window must not be empty")
    if "-" in text:
        start_text, stop_text = text.split("-", 1)
        start = int(start_text)
        stop = int(stop_text)
        if stop < start:
            raise ValueError(f"Cycle window {value!r} has stop before start")
        return tuple(range(start, stop + 1))
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _cycle_window_label(cycles: tuple[int, ...]) -> str:
    if not cycles:
        raise ValueError("cycles must not be empty")
    if len(cycles) == cycles[-1] - cycles[0] + 1:
        return f"{cycles[0]}_{cycles[-1]}"
    return "_".join(str(cycle) for cycle in cycles)


def run_split_cycles(args: argparse.Namespace) -> dict[str, object]:
    """Generate train/validation averaged-cycle traces from raw trial CSVs."""

    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cycles = _parse_cycle_window(str(args.train_cycles))
    validate_cycles = _parse_cycle_window(str(args.validate_cycles))
    phase_count = int(args.cycle_phase_count)

    generated = []
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        frame_config = _trial_frame_config(manifest, output_dir, trial)
        windows = {
            "train": train_cycles,
            "validate": validate_cycles,
        }
        trial_outputs = {}
        for split_name, cycles in windows.items():
            trace = build_cycle_window_trace(
                trial.csv_path,
                frame_config,
                cycles,
                phase_count=phase_count,
            )
            cycle_label = _cycle_window_label(cycles)
            path = output_dir / f"{trial.name}_{split_name}_cycles_{cycle_label}.csv"
            write_cycle_window_trace(path, trace)
            summary_path = path.with_suffix(".summary.json")
            summary = {
                "schema_version": "digital_instron_v2_cycle_window_1",
                "trial": trial.name,
                "split": split_name,
                "output_csv": str(path),
                "frame_config": frame_config,
                **trace.provenance,
            }
            _write_json(summary_path, summary)
            trial_outputs[split_name] = {
                "csv": str(path),
                "summary": str(summary_path),
                "cycles": list(cycles),
            }
        generated.append({"trial": trial.name, **trial_outputs})

    report = {
        "schema_version": "digital_instron_v2_cycle_windows_1",
        "manifest": str(manifest.path),
        "output_dir": str(output_dir),
        "train_cycles": list(train_cycles),
        "validate_cycles": list(validate_cycles),
        "phase_count": phase_count,
        "trials": generated,
    }
    _write_json(output_dir / "digital_instron_v2_cycle_windows.json", report)
    return report


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
    compression_velocity = np.zeros_like(current_length)
    top_fraction, bottom_fraction = _trial_displacement_split(trial)
    top_velocity = top_fraction * float(displacement_velocity_mps)
    bottom_velocity = bottom_fraction * float(displacement_velocity_mps)

    top_compression, bottom_compression, top_active, bottom_active = _spring_compression_components_for_trial_frame(
        spring_grid,
        trial,
        rearfoot_mask,
        contact_surfaces,
        displacement,
    )
    compression = np.minimum(top_compression + bottom_compression, spring_grid.slack_length_m)
    current_length = np.maximum(spring_grid.slack_length_m - compression, 0.0)
    compression_velocity[top_active] += top_velocity
    compression_velocity[bottom_active] += bottom_velocity
    velocity[compression > 0.0] = -compression_velocity[compression > 0.0]
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


_PHASE_WEIGHTS = {
    "baseline_pre": 0.25,
    "loading": 1.0,
    "peak": 2.0,
    "unloading": 1.5,
    "baseline_post": 0.25,
}


def _fit_phase_weights(fit: dict[str, object]) -> dict[str, float]:
    """Return phase weights from the manifest, falling back to conservative defaults."""

    raw = fit.get("phase_weights")
    if raw is None:
        return dict(_PHASE_WEIGHTS)
    if not isinstance(raw, dict):
        raise ValueError("fit.phase_weights must be an object when provided")
    weights = dict(_PHASE_WEIGHTS)
    for phase, value in raw.items():
        if phase not in weights:
            raise ValueError(f"fit.phase_weights has unsupported phase {phase!r}")
        phase_weight = float(value)
        if phase_weight < 0.0:
            raise ValueError(f"fit.phase_weights[{phase!r}] must be non-negative")
        weights[phase] = phase_weight
    if sum(weights.values()) <= 0.0:
        raise ValueError("fit.phase_weights must contain at least one positive weight")
    return weights


def _fit_loop_weight(args: argparse.Namespace, fit: dict[str, object]) -> float:
    override = getattr(args, "loop_weight", None)
    if override is not None:
        value = float(override)
        if value < 0.0:
            raise ValueError("--loop-weight must be >= 0")
        return value
    return _fit_float(fit, "loop_weight", 0.0, min_value=0.0)


def _fit_float(fit: dict[str, object], key: str, default: float, *, min_value: float | None = None) -> float:
    value = float(fit.get(key, default))
    if min_value is not None and value < min_value:
        raise ValueError(f"fit.{key} must be >= {min_value}")
    return value


def _fit_int(fit: dict[str, object], key: str, default: int, *, min_value: int | None = None) -> int:
    value = int(fit.get(key, default))
    if min_value is not None and value < min_value:
        raise ValueError(f"fit.{key} must be >= {min_value}")
    return value


_ACCEPTANCE_GATES = {
    "rearfoot_rmse_n": 150.0,
    "fullfoot_rmse_n": 400.0,
    "loop_area_relative_error": 0.35,
    "peak_force_ratio_upper": 1.5,
    "peak_force_ratio_lower": 0.67,
    "baseline_rmse_n": 50.0,
}


def _load_averaged_cycle(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    required = ("time_s", "displacement_m", "force_n", "velocity_m_s")
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Averaged cycle CSV {path} is missing columns: {', '.join(missing)}")
    trace = {name: np.asarray(data[name], dtype=np.float64) for name in required}
    finite = np.isfinite(trace["time_s"]) & np.isfinite(trace["displacement_m"])
    finite &= np.isfinite(trace["force_n"]) & np.isfinite(trace["velocity_m_s"])
    if int(np.count_nonzero(finite)) < 3:
        raise ValueError(f"Averaged cycle CSV {path} has fewer than three finite samples")
    trace = {name: values[finite] for name, values in trace.items()}
    if np.any(np.diff(trace["time_s"]) <= 0.0):
        raise ValueError(f"Averaged cycle CSV {path} time_s must be strictly increasing")
    gradient_velocity = np.gradient(trace["displacement_m"], trace["time_s"])
    candidates = (trace["velocity_m_s"], -trace["velocity_m_s"])
    errors = [np.abs(gradient_velocity - candidate) for candidate in candidates]
    best_index = int(np.argmin([np.percentile(error, 95.0) for error in errors]))
    best_error = errors[best_index]
    if float(np.percentile(best_error, 95.0)) > 0.03 or float(np.max(best_error)) > 0.05:
        raise ValueError(
            f"Averaged cycle velocity in {path} disagrees with displacement derivative: "
            f"p95={np.percentile(best_error, 95.0):.4g} m/s max={np.max(best_error):.4g} m/s"
        )
    trace["velocity_m_s"] = candidates[best_index]
    force_zero_n = float(np.min(trace["force_n"]))
    trace["raw_force_n"] = trace["force_n"].copy()
    trace["force_zero_n"] = np.full_like(trace["force_n"], force_zero_n)
    trace["force_n"] = trace["force_n"] - force_zero_n
    return trace


def _phase_labels_and_weights(
    displacement_m: np.ndarray,
    force_n: np.ndarray,
    *,
    low_force_limit_n: float = 20.0,
    peak_fraction: float = 0.95,
    phase_weights: dict[str, float] | None = None,
    displacement_shape_weight: float = 0.0,
    displacement_shape_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.asarray(displacement_m, dtype=np.float64)
    force = np.asarray(force_n, dtype=np.float64)
    if displacement.shape != force.shape or displacement.ndim != 1:
        raise ValueError("displacement_m and force_n must be matching 1D arrays")
    phases = np.full(len(force), "ignored", dtype=object)
    active = force > low_force_limit_n
    if not np.any(active):
        raise ValueError("Averaged cycle has no force-active frames")

    peak_force = float(np.max(force[active]))
    peak_displacement = float(np.max(displacement[active]))
    peak = active & ((force >= peak_fraction * peak_force) | (displacement >= peak_fraction * peak_displacement))
    if not np.any(peak):
        peak[np.argmax(force)] = True
    peak_start = int(np.nonzero(peak)[0][0])
    peak_stop = int(np.nonzero(peak)[0][-1])

    baseline = force <= low_force_limit_n
    phases[baseline & (np.arange(len(force)) < peak_start)] = "baseline_pre"
    phases[baseline & (np.arange(len(force)) > peak_stop)] = "baseline_post"
    phases[active & (np.arange(len(force)) < peak_start)] = "loading"
    phases[peak] = "peak"
    phases[active & (np.arange(len(force)) > peak_stop)] = "unloading"

    configured_phase_weights = dict(_PHASE_WEIGHTS if phase_weights is None else phase_weights)
    shape_blend = float(displacement_shape_weight)
    if not 0.0 <= shape_blend <= 1.0:
        raise ValueError("displacement_shape_weight must be in [0, 1]")
    if displacement_shape_bins <= 0:
        raise ValueError("displacement_shape_bins must be positive")

    weights = np.zeros(len(force), dtype=np.float64)
    present_weight_sum = 0.0
    for phase, phase_weight in configured_phase_weights.items():
        count = int(np.count_nonzero(phases == phase))
        if count == 0:
            continue
        present_weight_sum += phase_weight
        weights[phases == phase] = phase_weight / count
    if present_weight_sum <= 0.0:
        raise ValueError("Averaged cycle phase split produced no weighted frames")
    weights /= present_weight_sum

    if shape_blend > 0.0:
        shape_weights = np.zeros(len(force), dtype=np.float64)
        active_indices = np.nonzero(active)[0]
        active_displacement = displacement[active_indices]
        if len(active_indices):
            low = float(np.min(active_displacement))
            high = float(np.max(active_displacement))
            if high > low:
                edges = np.linspace(low, high, displacement_shape_bins + 1)
                bin_ids = np.searchsorted(edges, active_displacement, side="right") - 1
                bin_ids = np.clip(bin_ids, 0, displacement_shape_bins - 1)
                present_bins = np.unique(bin_ids)
                for bin_id in present_bins:
                    bin_indices = active_indices[bin_ids == bin_id]
                    shape_weights[bin_indices] = 1.0 / (len(present_bins) * len(bin_indices))
            else:
                shape_weights[active_indices] = 1.0 / len(active_indices)
        if np.sum(shape_weights) > 0.0:
            weights = (1.0 - shape_blend) * weights + shape_blend * shape_weights
            weights /= np.sum(weights)
    return phases, weights


def _trial_averaged_cycle_path(trial) -> Path:
    if trial.averaged_cycle_path is None:
        raise ValueError(f"Trial {trial.name!r} must define averaged_cycle_path for fit-autodiff")
    return trial.averaged_cycle_path


def _autodiff_batches(
    manifest,
    spring_grid,
    vertices,
    trace_paths_by_trial: dict[str, Path] | None = None,
) -> list[FoundationTrialBatch]:
    batches: list[FoundationTrialBatch] = []
    rearfoot_mask = _rearfoot_mask(manifest, spring_grid, vertices)
    contact_surfaces = _trial_contact_surface_cache(manifest, spring_grid)
    cell_area = np.full(len(spring_grid.xy_m), spring_grid.cell_area_m2, dtype=np.float64)
    phase_weights = _fit_phase_weights(manifest.fit)
    low_force_limit_n = _fit_float(manifest.fit, "low_force_limit_n", 20.0, min_value=0.0)
    peak_fraction = _fit_float(manifest.fit, "peak_fraction", 0.95, min_value=0.0)
    if peak_fraction > 1.0:
        raise ValueError("fit.peak_fraction must be <= 1")
    displacement_shape_weight = _fit_float(manifest.fit, "displacement_shape_weight", 0.0, min_value=0.0)
    if displacement_shape_weight > 1.0:
        raise ValueError("fit.displacement_shape_weight must be <= 1")
    displacement_shape_bins = _fit_int(manifest.fit, "displacement_shape_bins", 12, min_value=1)
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        trace_path = None if trace_paths_by_trial is None else trace_paths_by_trial.get(trial.name)
        if trace_path is None:
            trace_path = _trial_averaged_cycle_path(trial)
        trace = _load_averaged_cycle(trace_path)
        phases, weights = _phase_labels_and_weights(
            trace["displacement_m"],
            trace["force_n"],
            low_force_limit_n=low_force_limit_n,
            peak_fraction=peak_fraction,
            phase_weights=phase_weights,
            displacement_shape_weight=displacement_shape_weight,
            displacement_shape_bins=displacement_shape_bins,
        )
        current_rows = []
        velocity_rows = []
        for displacement, displacement_velocity in zip(trace["displacement_m"], trace["velocity_m_s"], strict=True):
            current_length, velocity = _spring_state_for_trial_frame(
                spring_grid,
                trial,
                rearfoot_mask,
                contact_surfaces,
                float(displacement),
                float(displacement_velocity),
            )
            current_rows.append(current_length)
            velocity_rows.append(velocity)
        if trial.fixture == "rearfoot_punch":
            active_mask = rearfoot_mask
            active_count = np.count_nonzero(active_mask)
            radius_m = float(trial.indenter.get("radius_m", 0.0225))
            analytical_area = np.pi * (radius_m**2)
            cell_area_val = analytical_area / active_count if active_count > 0 else spring_grid.cell_area_m2
            trial_cell_area = np.full(len(spring_grid.xy_m), cell_area_val, dtype=np.float64)
        else:
            trial_cell_area = cell_area

        batches.append(
            FoundationTrialBatch(
                name=trial.name,
                current_length_m=np.asarray(current_rows, dtype=np.float64),
                slack_length_m=spring_grid.slack_length_m,
                velocity_mps=np.asarray(velocity_rows, dtype=np.float64),
                measured_force_n=trace["force_n"],
                sample_weight=weights,
                cell_area_m2=trial_cell_area,
                time_s=trace["time_s"],
                dt_s=np.concatenate(
                    (
                        [trace["time_s"][1] - trace["time_s"][0] if len(trace["time_s"]) > 1 else 0.001],
                        np.diff(trace["time_s"]),
                    )
                ),
                displacement_m=trace["displacement_m"],
                phase=tuple(str(phase) for phase in phases),
                force_zero_n=float(trace["force_zero_n"][0]),
                neighbors=spring_grid.neighbors,
                spacing_m=spring_grid.spacing_m,
            )
        )
    return batches


def _trace_paths_from_cycle_report(report: dict[str, object], split: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for trial in report["trials"]:
        trial_report = dict(trial)
        split_report = dict(trial_report[split])
        paths[str(trial_report["trial"])] = Path(str(split_report["csv"]))
    return paths


def _safe_report_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _phase_diagnostics(batch: FoundationTrialBatch, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    phase_array = np.asarray(batch.phase, dtype=object)
    measured = np.asarray(batch.measured_force_n, dtype=np.float64)
    for phase in _PHASE_WEIGHTS:
        mask = phase_array == phase
        if not np.any(mask):
            diagnostics[phase] = {"frame_count": 0}
            continue
        residual = predicted[mask] - measured[mask]
        diagnostics[phase] = {
            "frame_count": int(np.count_nonzero(mask)),
            "rmse_n": float(np.sqrt(np.mean(residual**2))),
            "mean_loss": float(np.mean(0.5 * residual**2)),
        }
    return diagnostics


def _acceptance_summary(trial_summaries: list[dict[str, object]]) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for summary in trial_summaries:
        trial = str(summary["trial"])
        rmse_limit = (
            _ACCEPTANCE_GATES["rearfoot_rmse_n"] if "rearfoot" in trial else _ACCEPTANCE_GATES["fullfoot_rmse_n"]
        )
        rmse = float(summary["rmse_n"])
        checks.append(
            {
                "trial": trial,
                "metric": "rmse_n",
                "value": rmse,
                "limit": rmse_limit,
                "passed": rmse < rmse_limit,
            }
        )

        measured_area = abs(float(summary["measured_loop_area_j"]))
        loop_area_relative_error = abs(float(summary["loop_area_error_j"])) / max(measured_area, 1.0e-9)
        checks.append(
            {
                "trial": trial,
                "metric": "loop_area_relative_error",
                "value": loop_area_relative_error,
                "limit": _ACCEPTANCE_GATES["loop_area_relative_error"],
                "passed": loop_area_relative_error < _ACCEPTANCE_GATES["loop_area_relative_error"],
            }
        )

        peak_ratio = float(summary["predicted_machine_peak_force_n"]) / max(
            float(summary["measured_machine_peak_force_n"]), 1.0
        )
        checks.append(
            {
                "trial": trial,
                "metric": "peak_force_ratio_upper",
                "value": peak_ratio,
                "limit": _ACCEPTANCE_GATES["peak_force_ratio_upper"],
                "passed": peak_ratio < _ACCEPTANCE_GATES["peak_force_ratio_upper"],
            }
        )
        checks.append(
            {
                "trial": trial,
                "metric": "peak_force_ratio_lower",
                "value": peak_ratio,
                "limit": _ACCEPTANCE_GATES["peak_force_ratio_lower"],
                "passed": peak_ratio > _ACCEPTANCE_GATES["peak_force_ratio_lower"],
            }
        )

        phases = summary["phases"]
        baseline_rmse = 0.0
        baseline_counts = 0
        for phase in ("baseline_pre", "baseline_post"):
            phase_summary = phases[phase]
            count = int(phase_summary.get("frame_count", 0))
            if count == 0:
                continue
            baseline_rmse += float(phase_summary["rmse_n"]) * count
            baseline_counts += count
        if baseline_counts > 0:
            baseline_rmse /= baseline_counts
            checks.append(
                {
                    "trial": trial,
                    "metric": "baseline_rmse_n",
                    "value": baseline_rmse,
                    "limit": _ACCEPTANCE_GATES["baseline_rmse_n"],
                    "passed": baseline_rmse < _ACCEPTANCE_GATES["baseline_rmse_n"],
                }
            )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "gates": dict(_ACCEPTANCE_GATES),
        "checks": checks,
    }


def _write_foundation_material_artifact(
    output_dir: Path,
    manifest,
    spring_grid,
    material: FoundationMaterial,
    hysteresis: dict[str, object],
    acceptance: dict[str, object],
    *,
    fit_source: str = "averaged_cycle",
) -> Path:
    preload = {
        str(trial_summary["trial"]): {"force_zero_n": float(trial_summary["force_zero_n"])}
        for trial_summary in hysteresis["trials"]
    }
    contact_trials = {}
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        top_fraction, bottom_fraction = _trial_displacement_split(trial)
        contact_trials[trial.name] = {
            "fixture": trial.fixture,
            "indenter_type": str(trial.indenter.get("type", "")),
            "top_displacement_fraction": float(top_fraction),
            "bottom_displacement_fraction": float(bottom_fraction),
        }
    trial_envelopes = {}
    for trial_summary in hysteresis["trials"]:
        trial_name = str(trial_summary["trial"])
        contact_trial = contact_trials.get(trial_name, {})
        top_fraction = float(contact_trial.get("top_displacement_fraction", 0.5))
        bottom_fraction = float(contact_trial.get("bottom_displacement_fraction", 0.5))
        fraction_total = max(top_fraction + bottom_fraction, 1.0e-12)
        peak_max_compression_m = float(trial_summary.get("peak_max_compression_m", 0.0))
        trial_envelopes[trial_name] = {
            "max_displacement_m": float(trial_summary.get("max_displacement_m", 0.0)),
            "peak_displacement_m": float(trial_summary.get("peak_displacement_m", 0.0)),
            "max_compression_m": float(trial_summary.get("max_compression_m", 0.0)),
            "peak_max_compression_m": peak_max_compression_m,
            "peak_top_compression_m": peak_max_compression_m * top_fraction / fraction_total,
            "peak_bottom_compression_m": peak_max_compression_m * bottom_fraction / fraction_total,
            "peak_active_area_m2": float(trial_summary.get("peak_active_area_m2", 0.0)),
            "measured_peak_force_n": float(trial_summary.get("measured_peak_force_n", 0.0)),
            "predicted_peak_force_n": float(trial_summary.get("predicted_peak_force_n", 0.0)),
        }

    preferred_trial_name = ""
    preferred_trial_envelope: dict[str, float] = {}
    preferred_stack = -1.0
    for trial_name, envelope in trial_envelopes.items():
        fixture = contact_trials.get(trial_name, {}).get("fixture")
        if fixture != "fullfoot_last":
            continue
        stack = float(envelope["peak_max_compression_m"])
        if stack > preferred_stack:
            preferred_stack = stack
            preferred_trial_name = trial_name
            preferred_trial_envelope = envelope
    if not preferred_trial_envelope:
        for trial_name, envelope in trial_envelopes.items():
            stack = float(envelope["peak_max_compression_m"])
            if stack > preferred_stack:
                preferred_stack = stack
                preferred_trial_name = trial_name
                preferred_trial_envelope = envelope

    preferred_peak_displacement_m = float(preferred_trial_envelope.get("peak_displacement_m", 0.0))
    preferred_peak_top_compression_m = float(preferred_trial_envelope.get("peak_top_compression_m", 0.0))
    preferred_peak_bottom_compression_m = float(preferred_trial_envelope.get("peak_bottom_compression_m", 0.0))
    preferred_peak_stack_compression_m = float(preferred_trial_envelope.get("peak_max_compression_m", 0.0))
    preferred_one_sided_hydro_shoe_stroke_m = preferred_peak_top_compression_m + 0.5 * preferred_peak_bottom_compression_m
    artifact = {
        "schema_version": "digital_instron_v2_foundation_material_1",
        "manifest": str(manifest.path),
        "fit_source": fit_source,
        "material": material.__dict__,
        "contact_model": {
            "type": "two_sided_spring_grid",
            "compression_components": "top_plus_bottom",
            "top_contact": "manifest indenter or flat active fixture region",
            "bottom_contact": "flat bottom platen over the active fixture region",
            "trials": contact_trials,
        },
        "calibration_envelope": {
            "preferred_peak_displacement_m": float(preferred_peak_displacement_m),
            "preferred_peak_top_compression_m": preferred_peak_top_compression_m,
            "preferred_peak_bottom_compression_m": preferred_peak_bottom_compression_m,
            "preferred_peak_stack_compression_m": preferred_peak_stack_compression_m,
            "preferred_one_sided_hydro_shoe_stroke_m": preferred_one_sided_hydro_shoe_stroke_m,
            "preferred_trial": preferred_trial_name,
            "basis": (
                "fullfoot_last peak envelope when available, otherwise max fitted trial peak stack compression; "
                "one_sided_hydro_shoe_stroke is top compression plus half bottom compression"
            ),
            "trials": trial_envelopes,
        },
        "grid": {
            "spring_count": int(len(spring_grid.xy_m)),
            "cell_area_m2": float(spring_grid.cell_area_m2),
            "spacing_m": float(spring_grid.spacing_m),
        },
        "preload_policy": {
            "type": "per_trial_force_zero_subtraction",
            "trials": preload,
        },
        "acceptance": acceptance,
        "hysteresis": hysteresis,
    }
    path = output_dir / "digital_instron_v2_foundation_material.json"
    _write_json(path, artifact)
    return path


def _write_autodiff_hysteresis_plot(
    output_dir: Path,
    xy_m: np.ndarray,
    material: FoundationMaterial,
    batches: list[FoundationTrialBatch],
    *,
    device: str,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    rows: list[tuple[str, int, float, float, str, float, float, float, float, float, float]] = []
    trial_summaries: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for batch in batches:
        result = evaluate_foundation_lengths_batch(
            xy_m,
            batch,
            material=material,
            device=device,
        )
        predicted = result.predicted_force_n
        measured = np.asarray(batch.measured_force_n, dtype=np.float64)
        force_zero = float(batch.force_zero_n)
        measured_machine = measured + force_zero
        predicted_machine = predicted + force_zero
        displacement = np.asarray(batch.displacement_m, dtype=np.float64)
        losses = 0.5 * (predicted - measured) ** 2

        for frame_index, (
            time,
            disp,
            phase,
            measured_force,
            predicted_force,
            measured_machine_force,
            predicted_machine_force,
            loss,
            weight,
        ) in enumerate(
            zip(
                batch.time_s,
                displacement,
                batch.phase,
                measured,
                predicted,
                measured_machine,
                predicted_machine,
                losses,
                batch.sample_weight,
                strict=True,
            )
        ):
            rows.append(
                (
                    batch.name,
                    int(frame_index),
                    float(time),
                    float(disp),
                    str(phase),
                    float(measured_force),
                    float(predicted_force),
                    float(measured_machine_force),
                    float(predicted_machine_force),
                    float(loss),
                    float(weight),
                )
            )

        segments = _hysteresis_segments(displacement)
        measured_points = ax.scatter(
            displacement * 1000.0,
            measured_machine,
            s=10.0,
            alpha=0.75,
            label=f"{batch.name} measured",
        )
        predicted_points = ax.scatter(
            displacement * 1000.0,
            predicted_machine,
            s=10.0,
            alpha=0.85,
            marker="x",
            label=f"{batch.name} predicted",
        )
        measured_color = measured_points.get_facecolors()[0]
        predicted_color = predicted_points.get_facecolors()[0]
        for segment in segments:
            if len(segment) < 4:
                continue
            ax.plot(
                displacement[segment] * 1000.0,
                measured_machine[segment],
                linewidth=1.2,
                alpha=0.35,
                color=measured_color,
            )
            ax.plot(
                displacement[segment] * 1000.0,
                predicted_machine[segment],
                linewidth=1.2,
                linestyle="--",
                alpha=0.45,
                color=predicted_color,
            )
        rmse = float(np.sqrt(np.mean((predicted - measured) ** 2)))
        measured_loop_area = float(np.trapezoid(measured_machine, displacement))
        predicted_loop_area = float(np.trapezoid(predicted_machine, displacement))
        compression = np.maximum(batch.slack_length_m[None, :] - batch.current_length_m, 0.0)
        peak_index = int(np.argmax(measured))
        peak_active = compression[peak_index] > 0.0
        trial_summaries.append(
            {
                "trial": batch.name,
                "frame_count": int(len(measured)),
                "force_zero_n": float(getattr(batch, "force_zero_n", 0.0)),
                "segment_count": int(len(segments)),
                "max_displacement_m": float(np.max(displacement)) if len(displacement) else 0.0,
                "peak_displacement_m": float(displacement[peak_index]) if len(displacement) else 0.0,
                "max_compression_m": float(np.max(compression)) if compression.size else 0.0,
                "peak_max_compression_m": float(np.max(compression[peak_index])) if compression.size else 0.0,
                "peak_active_area_m2": float(np.sum(batch.cell_area_m2[peak_active])),
                "rmse_n": rmse,
                "normalized_rmse": float(rmse / max(float(np.max(np.abs(measured))), 1.0)),
                "measured_peak_force_n": float(np.max(measured)),
                "predicted_peak_force_n": float(np.max(predicted)),
                "peak_force_error_n": float(np.max(predicted) - np.max(measured)),
                "measured_machine_peak_force_n": float(np.max(measured_machine)),
                "predicted_machine_peak_force_n": float(np.max(predicted_machine)),
                "measured_loop_area_j": measured_loop_area,
                "predicted_loop_area_j": predicted_loop_area,
                "loop_area_error_j": predicted_loop_area - measured_loop_area,
                "phases": _phase_diagnostics(batch, predicted),
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
        np.asarray([row[1:4] + row[5:] for row in rows], dtype=np.float64)
        if rows
        else np.empty((0, 9), dtype=np.float64),
        delimiter=",",
        header=(
            "frame_index,time_s,displacement_m,measured_force_n,predicted_force_n,"
            "measured_machine_force_n,predicted_machine_force_n,loss,sample_weight"
        ),
        comments="",
    )
    if rows:
        trial_path = output_dir / "digital_instron_v2_autodiff_hysteresis_trials.csv"
        trial_path.write_text(
            "trial,frame_index,time_s,displacement_m,phase,measured_force_n,predicted_force_n,"
            "measured_machine_force_n,predicted_machine_force_n,loss,sample_weight\n"
            + "\n".join(
                f"{trial},{frame},{time},{displacement},{phase},{measured},{predicted},"
                f"{measured_machine},{predicted_machine},{loss},{weight}"
                for (
                    trial,
                    frame,
                    time,
                    displacement,
                    phase,
                    measured,
                    predicted,
                    measured_machine,
                    predicted_machine,
                    loss,
                    weight,
                ) in rows
            )
            + "\n"
        )
    else:
        trial_path = output_dir / "digital_instron_v2_autodiff_hysteresis_trials.csv"
        trial_path.write_text(
            "trial,frame_index,time_s,displacement_m,phase,measured_force_n,predicted_force_n,"
            "measured_machine_force_n,predicted_machine_force_n,loss,sample_weight\n"
        )

    hysteresis = {
        "hysteresis_png": str(png_path),
        "hysteresis_csv": str(csv_path),
        "hysteresis_trials_csv": str(trial_path),
        "trials": trial_summaries,
    }
    hysteresis["acceptance"] = _acceptance_summary(trial_summaries)
    return hysteresis


def _write_autodiff_loss_plot(
    output_dir: Path,
    history: list[dict[str, float]],
) -> str:
    """Write a loss-by-iteration plot from the autodiff fit history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [h["iteration"] for h in history]
    losses = [h["loss"] for h in history]

    # Check if state_beta/state_tau_s or prony_stiffness_pa/prony_damping_pa_s exist in history (for dual-plot)
    has_state = ("state_beta" in history[0] or "prony_stiffness_pa" in history[0]) if history else False

    if has_state:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.0), constrained_layout=True, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)

    ax1.plot(iterations, losses, "b-", linewidth=0.8)
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.set_title("Autodiff Fit — Loss per Iteration")
    ax1.grid(True, alpha=0.3)

    if has_state:
        if "prony_stiffness_pa" in history[0]:
            stiffnesses = [h["prony_stiffness_pa"] for h in history]
            dampings = [h["prony_damping_pa_s"] for h in history]
            ax2.plot(iterations, stiffnesses, "r-", label="prony_stiffness_pa", linewidth=0.8)
            ax2.plot(iterations, dampings, "g-", label="prony_damping_pa_s", linewidth=0.8)
        else:
            betas = [h["state_beta"] for h in history]
            taus = [h["state_tau_s"] for h in history]
            ax2.plot(iterations, betas, "r-", label="state_beta", linewidth=0.8)
            ax2.plot(iterations, taus, "g-", label="state_tau_s", linewidth=0.8)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Parameter value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    png_path = output_dir / "digital_instron_v2_autodiff_loss.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return str(png_path)


def _material_from_history_row(row: dict[str, float]) -> FoundationMaterial:
    if "prony_stiffness_pa" in row:
        prony_stiffness = float(row["prony_stiffness_pa"])
        prony_damping = float(row["prony_damping_pa_s"])
    else:
        # Convert state parameters
        stiffness = float(row["stiffness_pa"])
        beta = float(row.get("state_beta", 0.0))
        tau = float(row.get("state_tau_s", 0.05))
        prony_stiffness = beta * stiffness
        prony_damping = tau * prony_stiffness

    return FoundationMaterial(
        stiffness_pa=float(row["stiffness_pa"]),
        ogden_alpha=float(row["ogden_alpha"]),
        lock_strain=float(row["lock_strain"]),
        damping_pa_s=float(row["damping_pa_s"]),
        damping_power=float(row["damping_power"]),
        per_cylinder_area=True,
        prony_stiffness_pa=prony_stiffness,
        prony_damping_pa_s=prony_damping,
        state_warmup_cycles=int(row["state_warmup_cycles"]),
        pasternak_stiffness_n_per_m=float(row.get("pasternak_stiffness_n_per_m", row.get("shear_modulus_pa", 0.0))),
        spatial_slope=float(row.get("spatial_slope", 0.0)),
    )


def _history_selection_score(
    xy_m: np.ndarray,
    batches: list[FoundationTrialBatch],
    material: FoundationMaterial,
    *,
    device: str,
) -> dict[str, object]:
    trial_scores = []
    for batch in batches:
        result = evaluate_foundation_lengths_batch(xy_m, batch, material=material, device=device)
        measured = np.asarray(batch.measured_force_n, dtype=np.float64)
        predicted = np.asarray(result.predicted_force_n, dtype=np.float64)
        metrics = validate_trace_metrics(measured, predicted, batch.displacement_m)
        force_scale = max(float(metrics.measured_peak_force_n), 1.0)
        rmse = float(np.sqrt(np.mean((predicted - measured) ** 2)))
        normalized_rmse = rmse / force_scale
        # This selector is deliberately aligned with the official gates. It is
        # not the differentiable objective; it chooses the checkpoint that best
        # satisfies the locked train metrics before held-out validation.
        score = metrics.peak_force_error + metrics.force_rmse_relative + min(metrics.hysteresis_error, 10.0)
        trial_scores.append(
            {
                "trial": batch.name,
                "score": float(score),
                "rmse_n": rmse,
                "normalized_rmse": normalized_rmse,
                "peak_force_error": float(metrics.peak_force_error),
                "force_rmse_relative": float(metrics.force_rmse_relative),
                "hysteresis_error": float(metrics.hysteresis_error),
                "measured_hysteresis_j": float(metrics.measured_hysteresis_j),
                "predicted_hysteresis_j": float(metrics.simulated_hysteresis_j),
            }
        )
    return {
        "score": float(np.mean([trial["score"] for trial in trial_scores])),
        "trials": trial_scores,
    }


def _select_history_material(
    xy_m: np.ndarray,
    batches: list[FoundationTrialBatch],
    history: list[dict[str, float]],
    *,
    device: str,
) -> tuple[FoundationMaterial, dict[str, object]]:
    if not history:
        raise ValueError("Cannot select fit history material without history rows")
    best_material = _material_from_history_row(history[0])
    best_selection: dict[str, object] | None = None
    for row in history:
        material = _material_from_history_row(row)
        selection = _history_selection_score(xy_m, batches, material, device=device)
        selection["iteration"] = int(row["iteration"])
        selection["loss"] = float(row["loss"])
        if best_selection is None or float(selection["score"]) < float(best_selection["score"]):
            best_material = material
            best_selection = selection
    if best_selection is None:
        raise ValueError("Fit history selection produced no candidates")
    return best_material, best_selection


def _locked_metrics_report(
    xy_m: np.ndarray,
    batches: list[FoundationTrialBatch],
    material: FoundationMaterial,
    *,
    device: str,
    output_dir: Path | None = None,
    split: str | None = None,
    active_fraction: float = 0.05,
    top_count: int = 5,
    pass_threshold: float = 0.10,
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for batch in batches:
        result = evaluate_foundation_lengths_batch(
            xy_m,
            batch,
            material=material,
            device=device,
        )
        metrics = validate_trace_metrics(
            batch.measured_force_n,
            result.predicted_force_n,
            batch.displacement_m,
            active_fraction=active_fraction,
            top_count=top_count,
            pass_threshold=pass_threshold,
        )
        metric_values = metrics.as_dict()
        active = active_force_mask(
            batch.measured_force_n,
            active_fraction=active_fraction,
            top_count=top_count,
        )
        residual_diagnostics = _residual_diagnostics(
            batch.displacement_m,
            batch.measured_force_n,
            result.predicted_force_n,
            active,
            measured_peak_force_n=metrics.measured_peak_force_n,
        )
        artifacts = {}
        if output_dir is not None and split is not None:
            artifacts = _write_locked_metric_artifacts(
                output_dir,
                split,
                batch,
                result.predicted_force_n,
                active,
            )
        trials.append(
            {
                "trial": batch.name,
                "frame_count": int(len(batch.measured_force_n)),
                "force_zero_n": float(batch.force_zero_n),
                "metrics": metric_values,
                "residual_diagnostics": residual_diagnostics,
                "artifacts": artifacts,
            }
        )
        for metric_name in ("peak_force_error", "force_rmse_relative", "hysteresis_error"):
            checks.append(
                {
                    "trial": batch.name,
                    "metric": metric_name,
                    "value": float(metric_values[metric_name]),
                    "limit": pass_threshold,
                    "passed": float(metric_values[metric_name]) < pass_threshold,
                }
            )
    return {
        "schema_version": "digital_instron_v2_locked_metrics_1",
        "active_fraction": active_fraction,
        "top_count": top_count,
        "pass_threshold": pass_threshold,
        "passed": all(bool(trial["metrics"]["passed"]) for trial in trials),
        "checks": checks,
        "trials": trials,
    }


def _rmse_summary(residual: np.ndarray, scale: float) -> dict[str, float | int]:
    if len(residual) == 0:
        return {
            "frame_count": 0,
            "rmse_n": float("nan"),
            "rmse_relative": float("nan"),
            "mean_residual_n": float("nan"),
            "max_abs_residual_n": float("nan"),
        }
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "frame_count": int(len(residual)),
        "rmse_n": rmse,
        "rmse_relative": float(rmse / max(abs(scale), 1.0e-9)),
        "mean_residual_n": float(np.mean(residual)),
        "max_abs_residual_n": float(np.max(np.abs(residual))),
    }


def _residual_diagnostics(
    displacement_m: np.ndarray,
    measured_force_n: np.ndarray,
    predicted_force_n: np.ndarray,
    active_mask: np.ndarray,
    *,
    measured_peak_force_n: float,
) -> dict[str, object]:
    displacement = np.asarray(displacement_m, dtype=np.float64)
    measured = np.asarray(measured_force_n, dtype=np.float64)
    predicted = np.asarray(predicted_force_n, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    residual = predicted - measured
    active_indices = np.nonzero(active & np.isfinite(displacement) & np.isfinite(residual))[0]
    if len(active_indices) == 0:
        return {
            "active": _rmse_summary(np.asarray([], dtype=np.float64), measured_peak_force_n),
            "loading": _rmse_summary(np.asarray([], dtype=np.float64), measured_peak_force_n),
            "unloading": _rmse_summary(np.asarray([], dtype=np.float64), measured_peak_force_n),
        }
    peak_index = int(active_indices[np.argmax(displacement[active_indices])])
    loading_indices = active_indices[active_indices <= peak_index]
    unloading_indices = active_indices[active_indices >= peak_index]
    return {
        "active": _rmse_summary(residual[active_indices], measured_peak_force_n),
        "loading": _rmse_summary(residual[loading_indices], measured_peak_force_n),
        "unloading": _rmse_summary(residual[unloading_indices], measured_peak_force_n),
        "peak_displacement_frame": peak_index,
    }


def _write_locked_metric_artifacts(
    output_dir: Path,
    split: str,
    batch: FoundationTrialBatch,
    predicted_force_n: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_trial = _safe_report_name(batch.name)
    prefix = output_dir / f"digital_instron_v2_{split}_{safe_trial}"
    trace_csv = prefix.with_name(f"{prefix.name}_measured_vs_predicted.csv")
    fd_png = prefix.with_name(f"{prefix.name}_force_displacement.png")
    hysteresis_png = prefix.with_name(f"{prefix.name}_hysteresis.png")
    output_dir.mkdir(parents=True, exist_ok=True)

    measured = np.asarray(batch.measured_force_n, dtype=np.float64)
    predicted = np.asarray(predicted_force_n, dtype=np.float64)
    displacement = np.asarray(batch.displacement_m, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    force_zero = float(batch.force_zero_n)
    residual = predicted - measured

    rows = [
        "frame_index,time_s,displacement_m,phase,active,measured_force_n,predicted_force_n,"
        "residual_force_n,measured_machine_force_n,predicted_machine_force_n,sample_weight"
    ]
    for index, (time_s, disp, phase, is_active, measured_force, predicted_force, residual_force, weight) in enumerate(
        zip(
            batch.time_s,
            displacement,
            batch.phase,
            active,
            measured,
            predicted,
            residual,
            batch.sample_weight,
            strict=True,
        )
    ):
        rows.append(
            f"{index},{time_s},{disp},{phase},{int(is_active)},{measured_force},{predicted_force},"
            f"{residual_force},{measured_force + force_zero},{predicted_force + force_zero},{weight}"
        )
    trace_csv.write_text("\n".join(rows) + "\n")

    fig, (ax_force, ax_residual) = plt.subplots(2, 1, figsize=(7.5, 6.0), constrained_layout=True, sharex=True)
    x_mm = displacement * 1000.0
    ax_force.plot(x_mm, measured, label="measured", linewidth=1.5)
    ax_force.plot(x_mm, predicted, label="predicted", linewidth=1.2, linestyle="--")
    ax_force.scatter(x_mm[active], measured[active], s=12.0, alpha=0.45, label="active")
    ax_force.set_ylabel("force [N]")
    ax_force.set_title(f"{split} {batch.name} force-displacement")
    ax_force.grid(True, alpha=0.3)
    ax_force.legend(fontsize="small")
    ax_residual.plot(x_mm, residual, color="tab:red", linewidth=1.2)
    ax_residual.axhline(0.0, color="black", linewidth=0.8)
    ax_residual.set_xlabel("displacement [mm]")
    ax_residual.set_ylabel("residual [N]")
    ax_residual.grid(True, alpha=0.3)
    fig.savefig(fd_png, dpi=150)
    plt.close(fig)

    active_indices = np.nonzero(active)[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    if len(active_indices):
        peak_index = int(active_indices[np.argmax(displacement[active_indices])])
        loading = active_indices[active_indices <= peak_index]
        unloading = active_indices[active_indices >= peak_index]
        ax.plot(x_mm[loading], measured[loading], label="measured loading", linewidth=1.5)
        ax.plot(x_mm[unloading], measured[unloading], label="measured unloading", linewidth=1.5)
        ax.plot(x_mm[loading], predicted[loading], label="predicted loading", linewidth=1.2, linestyle="--")
        ax.plot(x_mm[unloading], predicted[unloading], label="predicted unloading", linewidth=1.2, linestyle="--")
    else:
        ax.plot(x_mm, measured, label="measured", linewidth=1.5)
        ax.plot(x_mm, predicted, label="predicted", linewidth=1.2, linestyle="--")
    ax.set_title(f"{split} {batch.name} hysteresis")
    ax.set_xlabel("displacement [mm]")
    ax.set_ylabel("force [N]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.savefig(hysteresis_png, dpi=150)
    plt.close(fig)

    return {
        "measured_vs_predicted_csv": str(trace_csv),
        "force_displacement_png": str(fd_png),
        "hysteresis_png": str(hysteresis_png),
    }


def _nuisance_parameter_report(
    manifest,
    output_dir: Path,
    train_batches: list[FoundationTrialBatch],
    validate_batches: list[FoundationTrialBatch],
) -> dict[str, object]:
    train_by_name = {batch.name: batch for batch in train_batches}
    validate_by_name = {batch.name: batch for batch in validate_batches}
    trials: dict[str, object] = {}
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        frame_config = _trial_frame_config(manifest, output_dir, trial)
        train_batch = train_by_name.get(trial.name)
        validate_batch = validate_by_name.get(trial.name)
        trials[trial.name] = {
            "fixture": trial.fixture,
            "force_zero_policy": "per_generated_trace_min_force_subtraction",
            "force_zero_n": {
                "train": None if train_batch is None else float(train_batch.force_zero_n),
                "validate": None if validate_batch is None else float(validate_batch.force_zero_n),
            },
            "displacement_zero_policy": "frame_config_displacement_zero",
            "displacement_zero": float(frame_config.get("displacement_zero", 0.0)),
            "geometry_alignment_policy": "manifest_fixture_geometry",
            "indenter": trial.indenter,
        }
    return {
        "allowed_only": True,
        "disallowed_parameters": [
            "fixture_specific_stiffness",
            "fixture_specific_damping",
            "fixture_specific_hysteresis_state",
            "arbitrary_force_scaling",
            "per_trial_pressure_multiplier",
            "manual_cop_shift",
        ],
        "trials": trials,
    }


def run_fit_validate(args: argparse.Namespace) -> dict[str, object]:
    """Fit on training cycles and validate on held-out cycles with locked metrics."""

    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    spring_grid, vertices = _load_spring_grid(manifest, output_dir)
    device = str(getattr(args, "autodiff_device", "cuda:0"))

    cycle_report = run_split_cycles(args)
    train_paths = _trace_paths_from_cycle_report(cycle_report, "train")
    validate_paths = _trace_paths_from_cycle_report(cycle_report, "validate")
    train_batches = _autodiff_batches(
        manifest,
        spring_grid,
        vertices,
        trace_paths_by_trial=train_paths,
    )
    validate_batches = _autodiff_batches(
        manifest,
        spring_grid,
        vertices,
        trace_paths_by_trial=validate_paths,
    )

    loop_weight = _fit_loop_weight(args, manifest.fit)
    result = fit_foundation_material_batches_autodiff(
        spring_grid.xy_m,
        train_batches,
        initial_material=_initial_material(manifest),
        iterations=int(args.autodiff_iterations),
        per_cylinder_area=True,
        loop_weight=loop_weight,
        device=device,
    )
    selected_material, selected_history = _select_history_material(
        spring_grid.xy_m,
        train_batches,
        list(result.history),
        device=device,
    )
    loss_plot_path = _write_autodiff_loss_plot(output_dir, list(result.history))
    train_metrics = _locked_metrics_report(
        spring_grid.xy_m,
        train_batches,
        selected_material,
        device=device,
        output_dir=output_dir,
        split="train",
    )
    validation_metrics = _locked_metrics_report(
        spring_grid.xy_m,
        validate_batches,
        selected_material,
        device=device,
        output_dir=output_dir,
        split="validate",
    )
    validation_acceptance = {
        "schema_version": "digital_instron_v2_phase1_acceptance_1",
        "passed": bool(validation_metrics["passed"]),
        "basis": "held_out_cycle_windows",
        "checks": validation_metrics["checks"],
    }
    nuisance_parameters = _nuisance_parameter_report(
        manifest,
        output_dir,
        train_batches,
        validate_batches,
    )
    material_artifact_path = _write_foundation_material_artifact(
        output_dir,
        manifest,
        spring_grid,
        selected_material,
        validation_metrics,
        validation_acceptance,
        fit_source="cycle_window_train_validate",
    )
    report = {
        "schema_version": "digital_instron_v2_phase1_validation_1",
        "manifest": str(manifest.path),
        "fit_source": "cycle_window_train_validate",
        "train_cycles": cycle_report["train_cycles"],
        "validate_cycles": cycle_report["validate_cycles"],
        "cycle_windows": cycle_report,
        "sample_count": {
            "train": int(sum(len(batch.measured_force_n) for batch in train_batches)),
            "validate": int(sum(len(batch.measured_force_n) for batch in validate_batches)),
        },
        "sample_weight_config": {
            "phase_weights": _fit_phase_weights(manifest.fit),
            "low_force_limit_n": _fit_float(manifest.fit, "low_force_limit_n", 20.0, min_value=0.0),
            "peak_fraction": _fit_float(manifest.fit, "peak_fraction", 0.95, min_value=0.0),
            "displacement_shape_weight": _fit_float(manifest.fit, "displacement_shape_weight", 0.0, min_value=0.0),
            "displacement_shape_bins": _fit_int(manifest.fit, "displacement_shape_bins", 12, min_value=1),
            "loop_weight": loop_weight,
        },
        "autodiff_device": device,
        "spring_grid_cells": int(len(spring_grid.xy_m)),
        "material": selected_material.__dict__,
        "nuisance_parameters": nuisance_parameters,
        "foundation_material_json": str(material_artifact_path),
        "acceptance": validation_acceptance,
        "loss_plot": loss_plot_path,
        "selected_iteration": int(selected_history["iteration"]),
        "selected_loss": float(selected_history["loss"]),
        "selected_score": float(selected_history["score"]),
        "selected_score_trials": selected_history["trials"],
        "history": list(result.history),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    _write_json(output_dir / "digital_instron_v2_phase1_validation.json", report)
    return report


def _cop_from_wrench(wrench: np.ndarray) -> tuple[float, float]:
    fz = float(wrench[2])
    if abs(fz) <= 1.0e-9:
        return float("nan"), float("nan")
    return float(-wrench[4] / fz), float(wrench[3] / fz)


def _make_body_f_solver(device: str):
    import newton
    import warp as wp

    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    for label in ("shoe", "fixture"):
        body = builder.add_link(
            mass=1000.0,
            inertia=wp.mat33(1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0),
            label=label,
            lock_inertia=True,
        )
        builder.add_shape_box(
            body,
            hx=0.20,
            hy=0.10,
            hz=0.04,
            cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0.0),
        )
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
            ],
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
            ],
        )
        builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, njmax=16)
    return model, solver, model.state(), model.state()


def _phase2_dynamic_replay(
    xy_m: np.ndarray,
    batches: list[FoundationTrialBatch],
    material: FoundationMaterial,
    *,
    output_dir: Path,
    device: str,
    shoe_body_index: int = 0,
    fixture_body_index: int = 1,
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    model, solver, state_0, state_1 = _make_body_f_solver(device)
    for batch in batches:
        state_0, state_1 = model.state(), model.state()
        safe_trial = _safe_report_name(batch.name)
        csv_path = output_dir / f"digital_instron_v2_dynamic_{safe_trial}.csv"
        predicted_forces = []
        rows = [
            "frame_index,time_s,displacement_m,measured_force_n,predicted_force_n,"
            "cop_x_m,cop_y_m,active_cell_count,active_area_m2,max_compression_m,"
            "wrench_fx,wrench_fy,wrench_fz,wrench_tx,wrench_ty,wrench_tz,"
            "shoe_body_f_fx,shoe_body_fy,shoe_body_fz,shoe_body_tx,shoe_body_ty,shoe_body_tz,"
            "fixture_body_f_fx,fixture_body_fy,fixture_body_fz,fixture_body_tx,fixture_body_ty,fixture_body_tz,"
            "shoe_body_qd_vx,shoe_body_qd_vy,shoe_body_qd_vz,shoe_body_qd_wx,shoe_body_qd_wy,shoe_body_qd_wz,"
            "fixture_body_qd_vx,fixture_body_qd_vy,fixture_body_qd_vz,"
            "fixture_body_qd_wx,fixture_body_qd_wy,fixture_body_qd_wz"
        ]
        cop_rows = []
        active_counts = []
        active_areas = []
        max_compressions = []
        wrench_rows = []
        body_force_rows = []
        body_qd_rows = []
        for frame_index in range(len(batch.measured_force_n)):
            current_length = batch.current_length_m[frame_index]
            velocity = batch.velocity_mps[frame_index]
            result = evaluate_foundation_lengths(
                xy_m,
                current_length,
                batch.slack_length_m,
                velocity,
                cell_area_m2=batch.cell_area_m2,
                material=material,
                measured_force_n=float(batch.measured_force_n[frame_index]),
                neighbors=batch.neighbors,
                spacing_m=batch.spacing_m,
                device=device,
            )
            predicted_forces.append(result.force_n)
            wrench = np.asarray(result.wrench, dtype=np.float64)
            wrench_rows.append(wrench)
            cop_x, cop_y = _cop_from_wrench(wrench)
            cop_rows.append((cop_x, cop_y))
            compression = np.maximum(batch.slack_length_m - current_length, 0.0)
            active = compression > 0.0
            active_counts.append(int(np.count_nonzero(active)))
            active_areas.append(float(np.sum(batch.cell_area_m2[active])))
            max_compressions.append(float(np.max(compression)) if len(compression) else 0.0)

            body_f = np.zeros((2, 6), dtype=np.float64)
            apply_foundation_wrench_to_body_f(body_f, shoe_body_index, wrench)
            apply_foundation_wrench_to_body_f(body_f, fixture_body_index, -wrench)
            body_force_rows.append(body_f.copy())
            state_0.body_f.assign(body_f.astype(np.float32).reshape(-1))
            dt_s = float(batch.dt_s[frame_index])
            if dt_s <= 0.0 and frame_index + 1 < len(batch.time_s):
                dt_s = float(batch.time_s[frame_index + 1] - batch.time_s[frame_index])
            if dt_s <= 0.0:
                dt_s = 1.0e-4
            solver.step(state_0, state_1, None, None, dt_s)
            body_qd = np.asarray(state_1.body_qd.numpy(), dtype=np.float64)
            body_qd_rows.append(body_qd.copy())
            state_0, state_1 = state_1, state_0
            rows.append(
                f"{frame_index},{batch.time_s[frame_index]},{batch.displacement_m[frame_index]},"
                f"{batch.measured_force_n[frame_index]},{result.force_n},{cop_x},{cop_y},"
                f"{active_counts[-1]},{active_areas[-1]},{max_compressions[-1]},"
                + ",".join(str(float(value)) for value in wrench)
                + ","
                + ",".join(str(float(value)) for value in body_f[shoe_body_index])
                + ","
                + ",".join(str(float(value)) for value in body_f[fixture_body_index])
                + ","
                + ",".join(str(float(value)) for value in body_qd[shoe_body_index])
                + ","
                + ",".join(str(float(value)) for value in body_qd[fixture_body_index])
            )
        csv_path.write_text("\n".join(rows) + "\n")

        predicted = np.asarray(predicted_forces, dtype=np.float64)
        metrics = validate_trace_metrics(batch.measured_force_n, predicted, batch.displacement_m)
        cop = np.asarray(cop_rows, dtype=np.float64)
        finite_cop = np.isfinite(cop[:, 0]) & np.isfinite(cop[:, 1])
        force_jump = np.abs(np.diff(predicted))
        force_scale = max(float(metrics.measured_peak_force_n), 1.0)
        max_force_jump = float(np.max(force_jump)) if len(force_jump) else 0.0
        max_force_jump_relative = float(max_force_jump / force_scale)
        wrench_array = np.asarray(wrench_rows, dtype=np.float64)
        body_force_array = np.asarray(body_force_rows, dtype=np.float64)
        body_qd_array = np.asarray(body_qd_rows, dtype=np.float64)
        equal_opposite_error = float(
            np.max(np.abs(body_force_array[:, shoe_body_index, :] + body_force_array[:, fixture_body_index, :]))
        )
        solver_advanced = bool(len(body_qd_array) and np.any(np.abs(body_qd_array) > 0.0))
        peak_body_qd_abs = [float(value) for value in np.max(np.abs(body_qd_array), axis=(0, 1))]
        body_qd_bound = 1.0e6
        body_qd_plausible = bool(np.max(np.abs(body_qd_array)) < body_qd_bound)
        stable = bool(
            np.all(np.isfinite(predicted))
            and np.all(np.isfinite(wrench_array))
            and np.all(np.isfinite(body_qd_array))
            and equal_opposite_error < 1.0e-9
            and solver_advanced
            and body_qd_plausible
        )
        trial_report = {
            "trial": batch.name,
            "trace_csv": str(csv_path),
            "frame_count": int(len(batch.measured_force_n)),
            "body_ids": {
                "shoe": shoe_body_index,
                "fixture": fixture_body_index,
            },
            "metrics": metrics.as_dict(),
            "dynamic_diagnostics": {
                "stable": stable,
                "peak_active_cell_count": int(np.max(active_counts)) if active_counts else 0,
                "peak_active_area_m2": float(np.max(active_areas)) if active_areas else 0.0,
                "mean_active_area_m2": float(np.mean(active_areas)) if active_areas else 0.0,
                "max_compression_m": float(np.max(max_compressions)) if max_compressions else 0.0,
                "max_force_jump_n": max_force_jump,
                "max_force_jump_relative": max_force_jump_relative,
                "mean_cop_x_m": float(np.mean(cop[finite_cop, 0])) if np.any(finite_cop) else float("nan"),
                "mean_cop_y_m": float(np.mean(cop[finite_cop, 1])) if np.any(finite_cop) else float("nan"),
                "peak_cop_x_m": float(cop[np.nanargmax(np.abs(cop[:, 0])), 0])
                if np.any(np.isfinite(cop[:, 0]))
                else float("nan"),
                "peak_cop_y_m": float(cop[np.nanargmax(np.abs(cop[:, 1])), 1])
                if np.any(np.isfinite(cop[:, 1]))
                else float("nan"),
                "equal_opposite_wrench_error": equal_opposite_error,
                "solver_advanced_state": solver_advanced,
                "solver_step_count": int(len(body_qd_rows)),
                "peak_body_qd_abs": peak_body_qd_abs,
                "body_qd_bound": body_qd_bound,
                "body_qd_plausible": body_qd_plausible,
                "predicted_signed_work_j": float(np.trapezoid(predicted, batch.displacement_m)),
                "measured_signed_work_j": float(np.trapezoid(batch.measured_force_n, batch.displacement_m)),
                "net_wrench_peak_abs": [float(value) for value in np.max(np.abs(wrench_array), axis=0)],
            },
        }
        trials.append(trial_report)
        for metric_name in ("peak_force_error", "force_rmse_relative", "hysteresis_error"):
            value = float(trial_report["metrics"][metric_name])
            checks.append(
                {
                    "trial": batch.name,
                    "metric": metric_name,
                    "value": value,
                    "limit": 0.10,
                    "passed": value < 0.10,
                }
            )
        checks.append(
            {
                "trial": batch.name,
                "metric": "dynamic_stability",
                "value": stable,
                "limit": True,
                "passed": stable,
            }
        )
    return {
        "schema_version": "digital_instron_v2_phase2_dynamic_replay_1",
        "solver": {
            "name": "SolverMuJoCo",
            "coupling": "Newton state.body_f wrench application",
            "mujoco_warp_status": "measured trajectory drives contact law; SolverMuJoCo advances body_f response state",
            "model_body_count": int(model.body_count),
        },
        "body_ids": {
            "shoe": shoe_body_index,
            "fixture": fixture_body_index,
        },
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "trials": trials,
    }


def run_dynamic_replay(args: argparse.Namespace) -> dict[str, object]:
    """Run the Phase 2 measured-trajectory body_f replay report."""

    phase1_report = run_fit_validate(args)
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    spring_grid, vertices = _load_spring_grid(manifest, output_dir)
    device = str(getattr(args, "autodiff_device", "cuda:0"))
    validate_paths = _trace_paths_from_cycle_report(phase1_report["cycle_windows"], "validate")
    validate_batches = _autodiff_batches(
        manifest,
        spring_grid,
        vertices,
        trace_paths_by_trial=validate_paths,
    )
    material = FoundationMaterial(**phase1_report["material"])
    dynamic = _phase2_dynamic_replay(
        spring_grid.xy_m,
        validate_batches,
        material,
        output_dir=output_dir,
        device=device,
    )
    report = {
        "schema_version": "digital_instron_v2_phase2_report_1",
        "manifest": str(manifest.path),
        "phase1_validation_json": str(output_dir / "digital_instron_v2_phase1_validation.json"),
        "material": material.__dict__,
        "train_cycles": phase1_report["train_cycles"],
        "validate_cycles": phase1_report["validate_cycles"],
        "dynamic_replay": dynamic,
        "acceptance": {
            "passed": bool(dynamic["passed"]),
            "basis": "held_out_cycle_windows_dynamic_replay",
            "checks": dynamic["checks"],
        },
    }
    _write_json(output_dir / "digital_instron_v2_phase2_dynamic_replay.json", report)
    return report


def run_fit_autodiff(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    spring_grid, vertices = _load_spring_grid(manifest, output_dir)
    device = str(getattr(args, "autodiff_device", "cuda:0"))
    batches = _autodiff_batches(
        manifest,
        spring_grid,
        vertices,
    )
    contact_surfaces = _trial_contact_surface_cache(manifest, spring_grid)
    contact_diagnostics: dict[str, dict[str, float]] = {}
    for batch in batches:
        if batch.name not in contact_surfaces:
            continue
        peak_index = int(np.argmax(batch.measured_force_n))
        displacement_m = float(batch.displacement_m[peak_index])
        trial_diagnostics = _fullfoot_contact_diagnostics(
            spring_grid,
            {batch.name: contact_surfaces[batch.name]},
            displacement_m=displacement_m,
        )
        contact_diagnostics.update(trial_diagnostics)
        stats = trial_diagnostics[batch.name]
        print(
            f"Contact diagnostics {batch.name}: "
            f"active_area={stats['active_area_mm2']:.0f} mm² "
            f"valid_area={stats['valid_area_mm2']:.0f} mm² "
            f"contact_mm min={stats['contact_min_mm']:.1f} "
            f"p50={stats['contact_p50_mm']:.1f} "
            f"max={stats['contact_max_mm']:.1f}"
        )
    _write_json(output_dir / "digital_instron_v2_contact_diagnostics.json", contact_diagnostics)
    loop_weight = _fit_loop_weight(args, manifest.fit)
    result = fit_foundation_material_batches_autodiff(
        spring_grid.xy_m,
        batches,
        initial_material=_initial_material(manifest),
        iterations=int(args.autodiff_iterations),
        per_cylinder_area=True,
        loop_weight=loop_weight,
        device=device,
    )
    selected_material, selected_history = _select_history_material(
        spring_grid.xy_m,
        batches,
        list(result.history),
        device=device,
    )
    loss_plot_path = _write_autodiff_loss_plot(output_dir, list(result.history))
    hysteresis = _write_autodiff_hysteresis_plot(
        output_dir,
        spring_grid.xy_m,
        selected_material,
        batches,
        device=device,
    )
    material_artifact_path = _write_foundation_material_artifact(
        output_dir,
        manifest,
        spring_grid,
        selected_material,
        hysteresis,
        hysteresis["acceptance"],
    )
    report = {
        "manifest": str(manifest.path),
        "sample_count": int(sum(len(batch.measured_force_n) for batch in batches)),
        "fit_source": "averaged_cycle",
        "sample_weight_config": {
            "phase_weights": _fit_phase_weights(manifest.fit),
            "low_force_limit_n": _fit_float(manifest.fit, "low_force_limit_n", 20.0, min_value=0.0),
            "peak_fraction": _fit_float(manifest.fit, "peak_fraction", 0.95, min_value=0.0),
            "displacement_shape_weight": _fit_float(manifest.fit, "displacement_shape_weight", 0.0, min_value=0.0),
            "displacement_shape_bins": _fit_int(manifest.fit, "displacement_shape_bins", 12, min_value=1),
            "loop_weight": loop_weight,
        },
        "autodiff_device": device,
        "spring_grid_cells": int(len(spring_grid.xy_m)),
        "material": selected_material.__dict__,
        "foundation_material_json": str(material_artifact_path),
        "acceptance": hysteresis["acceptance"],
        "contact_diagnostics": contact_diagnostics,
        "loss_plot": loss_plot_path,
        "selected_iteration": int(selected_history["iteration"]),
        "selected_loss": float(selected_history["loss"]),
        "selected_score": float(selected_history["score"]),
        "selected_score_trials": selected_history["trials"],
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
    elif args.step == "split-cycles":
        report = run_split_cycles(args)
    elif args.step == "fit-smoke":
        report = run_fit_smoke(args)
    elif args.step == "fit-autodiff":
        report = run_fit_autodiff(args)
    elif args.step == "fit-validate":
        report = run_fit_validate(args)
    elif args.step == "dynamic-replay":
        report = run_dynamic_replay(args)
    else:
        report = run_visualize(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
