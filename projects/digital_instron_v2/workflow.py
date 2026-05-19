# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manual/script entrypoint for the experimental Digital Instron v2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .foundation import (
    FoundationMaterial,
    FoundationTrialBatch,
    evaluate_foundation,
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
        state_beta=float(manifest.fit.get("state_beta", 0.0)),
        state_tau_s=float(manifest.fit.get("state_tau_s", 0.05)),
        state_warmup_cycles=int(manifest.fit.get("state_warmup_cycles", 0)),
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
        trace = _load_averaged_cycle(_trial_averaged_cycle_path(trial))
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
        batches.append(
            FoundationTrialBatch(
                name=trial.name,
                current_length_m=np.asarray(current_rows, dtype=np.float64),
                slack_length_m=spring_grid.slack_length_m,
                velocity_mps=np.asarray(velocity_rows, dtype=np.float64),
                measured_force_n=trace["force_n"],
                sample_weight=weights,
                cell_area_m2=cell_area,
                time_s=trace["time_s"],
                dt_s=np.concatenate(([0.0], np.diff(trace["time_s"]))),
                displacement_m=trace["displacement_m"],
                phase=tuple(str(phase) for phase in phases),
                force_zero_n=float(trace["force_zero_n"][0]),
            )
        )
    return batches


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
) -> Path:
    preload = {
        str(trial_summary["trial"]): {"force_zero_n": float(trial_summary["force_zero_n"])}
        for trial_summary in hysteresis["trials"]
    }
    artifact = {
        "schema_version": "digital_instron_v2_foundation_material_1",
        "manifest": str(manifest.path),
        "fit_source": "averaged_cycle",
        "material": material.__dict__,
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
        trial_summaries.append(
            {
                "trial": batch.name,
                "frame_count": int(len(measured)),
                "force_zero_n": float(getattr(batch, "force_zero_n", 0.0)),
                "segment_count": int(len(segments)),
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

    # Check if state_beta/state_tau_s exist in history (for dual-plot)
    has_state = "state_beta" in history[0] if history else False

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
    return FoundationMaterial(
        stiffness_pa=float(row["stiffness_pa"]),
        ogden_alpha=float(row["ogden_alpha"]),
        lock_strain=float(row["lock_strain"]),
        damping_pa_s=float(row["damping_pa_s"]),
        damping_power=float(row["damping_power"]),
        per_cylinder_area=True,
        state_beta=float(row["state_beta"]),
        state_tau_s=float(row["state_tau_s"]),
        state_warmup_cycles=int(row["state_warmup_cycles"]),
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
        force_scale = max(float(np.max(np.abs(measured))), 1.0)
        rmse = float(np.sqrt(np.mean((predicted - measured) ** 2)))
        measured_peak = float(np.max(measured))
        predicted_peak = float(np.max(predicted))
        peak_error = abs(predicted_peak - measured_peak) / force_scale
        measured_machine = measured + float(getattr(batch, "force_zero_n", 0.0))
        predicted_machine = predicted + float(getattr(batch, "force_zero_n", 0.0))
        measured_loop = float(np.trapezoid(measured_machine, batch.displacement_m))
        predicted_loop = float(np.trapezoid(predicted_machine, batch.displacement_m))
        loop_relative_error = abs(predicted_loop - measured_loop) / max(abs(measured_loop), 1.0e-9)
        normalized_rmse = rmse / force_scale
        # This is a validation selector over the optimizer history, not the
        # differentiable training objective. Keep it simple and physical.
        score = normalized_rmse + 0.5 * peak_error + 0.25 * min(loop_relative_error, 2.0)
        trial_scores.append(
            {
                "trial": batch.name,
                "score": float(score),
                "rmse_n": rmse,
                "normalized_rmse": normalized_rmse,
                "peak_error_n": float(predicted_peak - measured_peak),
                "peak_error_fraction": float(peak_error),
                "measured_loop_area_j": measured_loop,
                "predicted_loop_area_j": predicted_loop,
                "loop_area_relative_error": float(loop_relative_error),
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
    result = fit_foundation_material_batches_autodiff(
        spring_grid.xy_m,
        batches,
        initial_material=_initial_material(manifest),
        iterations=int(args.autodiff_iterations),
        per_cylinder_area=True,
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
    elif args.step == "fit-smoke":
        report = run_fit_smoke(args)
    elif args.step == "fit-autodiff":
        report = run_fit_autodiff(args)
    else:
        report = run_visualize(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
