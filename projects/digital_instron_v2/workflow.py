# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manual/script entrypoint for the experimental Digital Instron v2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from .cycle_windows import build_cycle_window_trace, write_cycle_window_trace
from .foundation import (
    FoundationMaterial,
    FoundationTrialBatch,
    evaluate_foundation_baked_batch,
    evaluate_foundation_lengths_batch,
    fit_foundation_material_baked_batches_autodiff,
)
from .frame_qc import infer_frame_config
from .geometry import (
    BakedMidsoleGeometry,
    _load_obj_mesh,
    _ray_triangle_z_candidates,
    build_baked_midsole_geometry,
    build_raycast_spring_grid,
    condition_midsole_mesh,
    place_rearfoot_punch_grid,
)
from .manifest import load_manifest
from .sdf_utils import _load_stl_mesh
from .validation import validate_trace_metrics
from .visualization import write_visualization_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json", help="Path to v2 trial manifest")
    parser.add_argument("--output-dir", default=None, help="Directory for QC and summary outputs")
    parser.add_argument(
        "--step",
        choices=(
            "qc",
            "split-cycles",
            "fit-autodiff",
            "visualize",
            "surface-scene",
        ),
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
        "--shape-weight",
        type=float,
        default=None,
        help="Override fit.displacement_shape_weight (0..1) to emphasise full-curve shape matching",
    )
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
    parser.add_argument(
        "--use-surfacemaps",
        dest="use_surfacemaps",
        action="store_true",
        help="Use the surface-map hydroelastic foundation calibration",
    )
    parser.add_argument(
        "--use-baked",
        dest="use_baked",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-equilibrium",
        dest="use_equilibrium",
        action="store_false",
        help="Disable through-thickness pressure equilibrium (use the legacy top/bottom travel split)",
    )
    parser.set_defaults(use_equilibrium=True)
    parser.add_argument(
        "--no-subcell-coverage",
        dest="use_subcell_coverage",
        action="store_false",
        help="Disable analytic sub-cell contact coverage (use the legacy binary cell gate)",
    )
    parser.set_defaults(use_subcell_coverage=True)
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl", help="Viewer for --step surface-scene")
    parser.add_argument("--scene-trial", default=None, help="Trial name for --step surface-scene")
    parser.add_argument(
        "--scene-max-frames", type=int, default=501, help="Maximum replay frames for --step surface-scene"
    )
    return parser


def _use_surfacemaps(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_surfacemaps", False) or getattr(args, "use_baked", False))


def _use_equilibrium(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_equilibrium", True))


def _use_subcell_coverage(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_subcell_coverage", True))


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


def _load_midsole_mesh(manifest, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh_report = condition_midsole_mesh(
        manifest.midsole_mesh,
        output_dir,
        source_units=str(manifest.qc.get("mesh_source_units", "mm")),
        min_thickness_m=float(manifest.qc.get("min_midsole_thickness_m", 0.005)),
        max_thickness_m=float(manifest.qc.get("max_midsole_thickness_m", 0.08)),
    )
    vertices, faces = _load_obj_mesh(Path(str(mesh_report["repaired_mesh"])))
    return vertices, faces


def _load_spring_grid(manifest, output_dir: Path):
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    spring_grid = build_raycast_spring_grid(
        vertices,
        faces,
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
        min_slack_length_m=float(manifest.qc.get("min_spring_slack_length_m", 0.001)),
        thickness_axis=manifest.grid.get("force_thickness_axis"),
    )
    return spring_grid, vertices, faces


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


def bake_indenter_maps(
    baked_geometry: BakedMidsoleGeometry,
    trial,
    manifest,
    vertices: np.ndarray,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bake the indenter contact height map and valid mask at 0 displacement.

    Returns:
        indenter_map: 2D array of shape (V, U) containing contact surface heights.
        indenter_valid_map: 2D array of shape (V, U) containing 1.0 where valid, 0.0 otherwise.
    """
    u = np.arange(baked_geometry.mins_uv[0], baked_geometry.maxs_uv[0] + spacing_m * 0.5, spacing_m, dtype=np.float64)
    v = np.arange(baked_geometry.mins_uv[1], baked_geometry.maxs_uv[1] + spacing_m * 0.5, spacing_m, dtype=np.float64)
    U, V = np.meshgrid(u, v, indexing="xy")
    shape_2d = U.shape
    grid_uv_m = np.stack([U.ravel(), V.ravel()], axis=1)
    valid_midsole_flat = (
        np.ones(len(grid_uv_m), dtype=bool)
        if baked_geometry.valid_map is None
        else np.asarray(baked_geometry.valid_map, dtype=np.float64).ravel() > 0.5
    )

    frame = baked_geometry.frame

    if trial.fixture == "rearfoot_punch":
        punch = place_rearfoot_punch_grid(
            vertices,
            radius_m=float(
                next(
                    (t.indenter.get("radius_m", 0.0225) for t in manifest.trials if t.fixture == "rearfoot_punch"),
                    0.0225,
                )
            ),
            spacing_m=spacing_m,
            frame=frame,
            heel_side=str(manifest.grid.get("rearfoot_heel_side", "min")),
            length_fraction=float(manifest.grid.get("rearfoot_length_fraction", 0.22)),
            lateral_fraction=float(manifest.grid.get("rearfoot_lateral_fraction", 0.5)),
            lateral_band_fraction=float(manifest.grid.get("rearfoot_lateral_band_fraction", 0.12)),
        )
        dist = np.linalg.norm(grid_uv_m - punch.center_uv_m, axis=1)
        # Analytic sub-cell coverage: linear ramp of the signed distance to the
        # disk boundary over one cell width. Cells fully inside the punch get
        # 1.0, fully outside get 0.0, and boundary cells get the fraction of the
        # cell width that lies inside the circle. This makes the integrated
        # contact area converge at O(h^2) instead of staircasing at O(h).
        coverage_flat = np.clip(0.5 + (punch.radius_m - dist) / spacing_m, 0.0, 1.0)
        coverage_flat[~valid_midsole_flat] = 0.0

        indenter_map = baked_geometry.top_map.copy()
        indenter_valid_map = coverage_flat.reshape(shape_2d).astype(np.float64)
        return indenter_map, indenter_valid_map

    elif trial.fixture == "fullfoot_last" and trial.indenter.get("type") == "stl":
        indenter_vertices, indenter_faces = _indenter_mesh_m(trial.indenter)
        plane_axes = frame.plane_axes
        thickness_axis = frame.thickness_axis

        indenter_plane = indenter_vertices[:, plane_axes]
        spring_plane_center = 0.5 * (np.min(grid_uv_m, axis=0) + np.max(grid_uv_m, axis=0))
        indenter_plane_center = 0.5 * (np.min(indenter_plane, axis=0) + np.max(indenter_plane, axis=0))
        indenter_vertices = indenter_vertices.copy()
        indenter_vertices[:, plane_axes] += spring_plane_center - indenter_plane_center

        plane = indenter_vertices[:, plane_axes]
        thickness = indenter_vertices[:, thickness_axis]
        triangles_plane = plane[indenter_faces.reshape(-1, 3)]
        triangles_thickness = thickness[indenter_faces.reshape(-1, 3)]
        tri_min = np.min(triangles_plane, axis=1) - 1.0e-12
        tri_max = np.max(triangles_plane, axis=1) + 1.0e-12

        contact_raw = np.full(len(grid_uv_m), np.nan, dtype=np.float64)
        for i, point in enumerate(grid_uv_m):
            candidate_indices = np.nonzero(
                (tri_min[:, 0] <= point[0])
                & (point[0] <= tri_max[:, 0])
                & (tri_min[:, 1] <= point[1])
                & (point[1] <= tri_max[:, 1])
            )[0]
            hits = _ray_triangle_z_candidates(point, triangles_plane, triangles_thickness, candidate_indices)
            if hits:
                contact_raw[i] = hits[0]

        valid_flat = np.isfinite(contact_raw) & valid_midsole_flat
        if not np.any(valid_flat):
            raise ValueError(f"Fullfoot STL indenter {trial.name!r} does not overlap the footprint grid")

        initial_clearance = float(trial.indenter.get("initial_clearance_m", 0.0))
        height_offset = float(trial.indenter.get("height_offset_m", 0.0))
        contact_percentile = float(trial.indenter.get("contact_percentile", 75.0))

        top_map_flat = baked_geometry.top_map.ravel()
        clearance_raw = top_map_flat[valid_flat] - contact_raw[valid_flat]
        thickness_offset = float(np.percentile(clearance_raw, contact_percentile) + initial_clearance + height_offset)
        contact_surface = contact_raw + thickness_offset
        contact_surface[valid_flat] = np.maximum(contact_surface[valid_flat], top_map_flat[valid_flat])

        contact_surface[~valid_flat] = top_map_flat[~valid_flat]

        indenter_map = contact_surface.reshape(shape_2d)
        # The fullfoot last is a smooth, large-area indenter: its contact patch
        # has soft edges, so a binary cell mask already integrates the area at
        # O(h^2). (Sub-cell area supersampling does not help here and the
        # resolution variation is dominated by the curved height-map resampling.)
        indenter_valid_map = np.zeros(shape_2d, dtype=np.float64)
        indenter_valid_map.flat[valid_flat] = 1.0
        return indenter_map, indenter_valid_map

    else:
        indenter_map = baked_geometry.top_map.copy()
        indenter_valid_map = valid_midsole_flat.reshape(shape_2d).astype(np.float64)
        return indenter_map, indenter_valid_map


def compute_baked_compression(
    xy_m: np.ndarray,
    baked_geometry: BakedMidsoleGeometry,
    indenter_map: np.ndarray,
    indenter_valid_map: np.ndarray,
    displacement_m: float,
    top_fraction: float = 1.0,
    bottom_fraction: float = 0.0,
) -> np.ndarray:
    """Evaluate compression at xy_m using the continuous baked maps."""
    u = (xy_m[:, 0] - baked_geometry.mins_uv[0]) / (baked_geometry.maxs_uv[0] - baked_geometry.mins_uv[0])
    v = (xy_m[:, 1] - baked_geometry.mins_uv[1]) / (baked_geometry.maxs_uv[1] - baked_geometry.mins_uv[1])
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    def sample_numpy(tex_map, u_arr, v_arr):
        h, w = tex_map.shape
        px = u_arr * (w - 1.0)
        py = v_arr * (h - 1.0)
        x0 = np.clip(np.floor(px).astype(np.int32), 0, w - 1)
        y0 = np.clip(np.floor(py).astype(np.int32), 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        tx = px - x0
        ty = py - y0
        val00 = tex_map[y0, x0]
        val10 = tex_map[y0, x1]
        val01 = tex_map[y1, x0]
        val11 = tex_map[y1, x1]
        val_top = val00 + tx * (val10 - val00)
        val_bot = val01 + tx * (val11 - val01)
        return val_top + ty * (val_bot - val_top)

    slack = np.maximum(sample_numpy(baked_geometry.thickness_map, u, v), 1.0e-6)
    z_top_undeformed = sample_numpy(baked_geometry.top_map, u, v)
    z_bottom_undeformed = sample_numpy(baked_geometry.bottom_map, u, v)
    ind_val = sample_numpy(indenter_valid_map, u, v)
    ind_map_val = sample_numpy(indenter_map, u, v)

    top_comp = np.zeros_like(slack)
    bottom_comp = np.zeros_like(slack)

    valid = ind_val > 0.5
    if np.any(valid):
        closure_m = max(displacement_m, 0.0)
        top_travel = top_fraction * closure_m
        z_contact = ind_map_val[valid] - top_travel
        top_comp[valid] = np.maximum(z_top_undeformed[valid] - z_contact, 0.0)

        bottom_travel = bottom_fraction * closure_m
        if baked_geometry.valid_map is None:
            min_bottom = float(np.min(baked_geometry.bottom_map))
        else:
            valid_map = np.asarray(baked_geometry.valid_map, dtype=np.float64) > 0.5
            min_bottom = float(np.min(baked_geometry.bottom_map[valid_map]))
        bottom_comp[valid] = np.maximum(min_bottom + bottom_travel - z_bottom_undeformed[valid], 0.0)

    return np.minimum(top_comp + bottom_comp, slack)


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
    "trace_rmse_relative": 0.05,
    "loop_area_relative_error": 0.35,
    "peak_force_ratio_upper": 1.5,
    "peak_force_ratio_lower": 0.67,
    "baseline_rmse_n": 50.0,
}


def _baked_quadrature(baked_geometry: BakedMidsoleGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if baked_geometry.grid_uv_m is None or baked_geometry.xy_m is None:
        raise ValueError("Baked geometry must define valid quadrature points")
    cell_area = float(baked_geometry.cell_area_m2)
    if cell_area <= 0.0:
        cell_area = float(baked_geometry.spacing_m * baked_geometry.spacing_m)
    if cell_area <= 0.0:
        raise ValueError("Baked geometry must define a positive cell area")
    cell_area_m2 = np.full(len(baked_geometry.grid_uv_m), cell_area, dtype=np.float64)
    return baked_geometry.grid_uv_m, baked_geometry.xy_m, cell_area_m2


def _write_surface_map_plot(output_dir: Path, baked_geometry: BakedMidsoleGeometry) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = (
        np.ones_like(baked_geometry.thickness_map, dtype=bool)
        if baked_geometry.valid_map is None
        else np.asarray(baked_geometry.valid_map, dtype=np.float64) > 0.5
    )
    maps = [
        ("thickness [mm]", baked_geometry.thickness_map * 1000.0),
        ("top surface [mm]", baked_geometry.top_map * 1000.0),
        ("bottom surface [mm]", baked_geometry.bottom_map * 1000.0),
        ("valid footprint", valid.astype(np.float64)),
    ]
    extent = [
        float(baked_geometry.mins_uv[0] * 1000.0),
        float(baked_geometry.maxs_uv[0] * 1000.0),
        float(baked_geometry.mins_uv[1] * 1000.0),
        float(baked_geometry.maxs_uv[1] * 1000.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), constrained_layout=True)
    for ax, (title, values) in zip(axes.ravel(), maps, strict=True):
        masked = np.ma.array(values, mask=(~valid if title != "valid footprint" else np.zeros_like(valid)))
        image = ax.imshow(masked, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("u [mm]")
        ax.set_ylabel("v [mm]")
        fig.colorbar(image, ax=ax, shrink=0.82)
    path = output_dir / "digital_instron_v2_surfacemaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _with_baked_cell_area(
    batches: list[FoundationTrialBatch],
    cell_area_m2: np.ndarray,
    spacing_m: float,
) -> list[FoundationTrialBatch]:
    return [
        FoundationTrialBatch(
            name=batch.name,
            current_length_m=batch.current_length_m,
            slack_length_m=batch.slack_length_m,
            velocity_mps=batch.velocity_mps,
            measured_force_n=batch.measured_force_n,
            sample_weight=batch.sample_weight,
            cell_area_m2=np.asarray(cell_area_m2, dtype=np.float64),
            time_s=batch.time_s,
            dt_s=batch.dt_s,
            displacement_m=batch.displacement_m,
            phase=batch.phase,
            force_zero_n=batch.force_zero_n,
            neighbors=None,
            spacing_m=float(spacing_m),
        )
        for batch in batches
    ]


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
    use_baked: bool = False,
    shape_weight_override: float | None = None,
) -> list[FoundationTrialBatch]:
    batches: list[FoundationTrialBatch] = []
    rearfoot_mask = None if use_baked else _rearfoot_mask(manifest, spring_grid, vertices)
    contact_surfaces = {}
    if not use_baked:
        contact_surfaces = _trial_contact_surface_cache(manifest, spring_grid)
    cell_area = (
        np.ones(1, dtype=np.float64)
        if use_baked
        else np.full(len(spring_grid.xy_m), spring_grid.cell_area_m2, dtype=np.float64)
    )
    phase_weights = _fit_phase_weights(manifest.fit)
    low_force_limit_n = _fit_float(manifest.fit, "low_force_limit_n", 20.0, min_value=0.0)
    peak_fraction = _fit_float(manifest.fit, "peak_fraction", 0.95, min_value=0.0)
    if peak_fraction > 1.0:
        raise ValueError("fit.peak_fraction must be <= 1")
    displacement_shape_weight = _fit_float(manifest.fit, "displacement_shape_weight", 0.0, min_value=0.0)
    if shape_weight_override is not None:
        if shape_weight_override < 0.0:
            raise ValueError("--shape-weight must be >= 0")
        displacement_shape_weight = float(shape_weight_override)
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
        if use_baked:
            current_rows = np.zeros((len(trace["displacement_m"]), 1), dtype=np.float64)
            slack_length_m = np.zeros(1, dtype=np.float64)
            velocity_mps = trace["velocity_m_s"]
            neighbors = None
        else:
            current_rows_list = []
            velocity_rows_list = []
            for displacement, displacement_velocity in zip(trace["displacement_m"], trace["velocity_m_s"], strict=True):
                current_length, velocity = _spring_state_for_trial_frame(
                    spring_grid,
                    trial,
                    rearfoot_mask,
                    contact_surfaces,
                    float(displacement),
                    float(displacement_velocity),
                )
                current_rows_list.append(current_length)
                velocity_rows_list.append(velocity)
            current_rows = np.asarray(current_rows_list, dtype=np.float64)
            slack_length_m = spring_grid.slack_length_m
            velocity_mps = np.asarray(velocity_rows_list, dtype=np.float64)
            neighbors = spring_grid.neighbors

        if use_baked:
            trial_cell_area = cell_area
        elif trial.fixture == "rearfoot_punch":
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
                current_length_m=current_rows,
                slack_length_m=slack_length_m,
                velocity_mps=velocity_mps,
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
                neighbors=neighbors,
                spacing_m=float(manifest.grid.get("baked_spacing_m", 0.002)) if use_baked else spring_grid.spacing_m,
            )
        )
    return batches


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
        rmse = float(summary["rmse_n"])
        normalized_rmse = float(summary["normalized_rmse"])
        checks.append(
            {
                "trial": trial,
                "metric": "trace_rmse_relative",
                "value": normalized_rmse,
                "rmse_n": rmse,
                "limit": _ACCEPTANCE_GATES["trace_rmse_relative"],
                "passed": normalized_rmse < _ACCEPTANCE_GATES["trace_rmse_relative"],
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
    use_baked: bool = False,
    baked_geometry: BakedMidsoleGeometry | None = None,
) -> Path:
    preload = {
        str(trial_summary["trial"]): {"force_zero_n": float(trial_summary["force_zero_n"])}
        for trial_summary in hysteresis["trials"]
    }
    contact_trials = {}
    for trial in manifest.trials:
        if not trial.include_in_fit:
            continue
        contact_trials[trial.name] = {
            "fixture": trial.fixture,
            "indenter_type": str(trial.indenter.get("type", "")),
        }
        if not use_baked:
            top_fraction, bottom_fraction = _trial_displacement_split(trial)
            contact_trials[trial.name]["top_displacement_fraction"] = float(top_fraction)
            contact_trials[trial.name]["bottom_displacement_fraction"] = float(bottom_fraction)
    trial_envelopes = {}
    for trial_summary in hysteresis["trials"]:
        trial_name = str(trial_summary["trial"])
        contact_trial = contact_trials.get(trial_name, {})
        peak_max_compression_m = float(trial_summary.get("peak_max_compression_m", 0.0))
        if use_baked:
            peak_top_compression_m = peak_max_compression_m
            peak_bottom_compression_m = 0.0
        else:
            top_fraction = float(contact_trial.get("top_displacement_fraction", 0.5))
            bottom_fraction = float(contact_trial.get("bottom_displacement_fraction", 0.5))
            fraction_total = max(top_fraction + bottom_fraction, 1.0e-12)
            peak_top_compression_m = peak_max_compression_m * top_fraction / fraction_total
            peak_bottom_compression_m = peak_max_compression_m * bottom_fraction / fraction_total
        trial_envelopes[trial_name] = {
            "max_displacement_m": float(trial_summary.get("max_displacement_m", 0.0)),
            "peak_displacement_m": float(trial_summary.get("peak_displacement_m", 0.0)),
            "max_compression_m": float(trial_summary.get("max_compression_m", 0.0)),
            "peak_max_compression_m": peak_max_compression_m,
            "peak_top_compression_m": peak_top_compression_m,
            "peak_bottom_compression_m": peak_bottom_compression_m,
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
    preferred_one_sided_hydro_shoe_stroke_m = (
        preferred_peak_top_compression_m + 0.5 * preferred_peak_bottom_compression_m
    )
    artifact = {
        "schema_version": "digital_instron_v2_foundation_material_1",
        "manifest": str(manifest.path),
        "fit_source": fit_source,
        "material": material.__dict__,
        "contact_model": {
            "type": "surface_map_hydroelastic" if use_baked else "two_sided_spring_grid",
            "compression_components": "top_moving_against_fixed_bottom_support" if use_baked else "top_plus_bottom",
            "top_contact": "manifest indenter or flat active fixture region",
            "bottom_contact": "fixed flat ground support"
            if use_baked
            else "flat bottom platen over the active fixture region",
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
                "surface-map replay uses a fixed bottom support and the moving upper contact surface carries the stroke"
            ),
            "trials": trial_envelopes,
        },
        "grid": (
            {
                "surface_map_cell_count": 0
                if baked_geometry is None or baked_geometry.grid_uv_m is None
                else int(len(baked_geometry.grid_uv_m)),
                "cell_area_m2": 0.0 if baked_geometry is None else float(baked_geometry.cell_area_m2),
                "spacing_m": 0.0 if baked_geometry is None else float(baked_geometry.spacing_m),
            }
            if use_baked
            else {
                "spring_count": int(len(spring_grid.xy_m)),
                "cell_area_m2": float(spring_grid.cell_area_m2),
                "spacing_m": float(spring_grid.spacing_m),
            }
        ),
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
    baked_geometry: BakedMidsoleGeometry | None = None,
    indenter_maps_by_trial: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    top_fractions_by_trial: dict[str, float] | None = None,
    bottom_fractions_by_trial: dict[str, float] | None = None,
    use_equilibrium: bool = True,
    use_subcell_coverage: bool = True,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    rows: list[tuple[str, int, float, float, str, float, float, float, float, float, float]] = []
    trial_summaries: list[dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for batch in batches:
        if baked_geometry is not None and indenter_maps_by_trial is not None:
            ind_map, ind_valid_map = indenter_maps_by_trial[batch.name]
            top_frac = top_fractions_by_trial.get(batch.name, 1.0) if top_fractions_by_trial is not None else 1.0
            bottom_frac = (
                bottom_fractions_by_trial.get(batch.name, 0.0) if bottom_fractions_by_trial is not None else 0.0
            )
            result = evaluate_foundation_baked_batch(
                xy_m,
                baked_geometry,
                ind_map,
                ind_valid_map,
                batch,
                material=material,
                top_fraction=top_frac,
                bottom_fraction=bottom_frac,
                use_equilibrium=use_equilibrium,
                device=device,
            )
        else:
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
        if baked_geometry is not None and indenter_maps_by_trial is not None:
            ind_map, ind_valid_map = indenter_maps_by_trial[batch.name]
            top_frac = top_fractions_by_trial.get(batch.name, 1.0) if top_fractions_by_trial is not None else 1.0
            bottom_frac = (
                bottom_fractions_by_trial.get(batch.name, 0.0) if bottom_fractions_by_trial is not None else 0.0
            )
            compression_list = []
            for disp in displacement:
                comp_frame = compute_baked_compression(
                    xy_m,
                    baked_geometry,
                    ind_map,
                    ind_valid_map,
                    float(disp),
                    top_frac,
                    bottom_frac,
                )
                compression_list.append(comp_frame)
            compression = np.asarray(compression_list, dtype=np.float64)
        else:
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
    baked_geometry: BakedMidsoleGeometry | None = None,
    indenter_maps_by_trial: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    top_fractions_by_trial: dict[str, float] | None = None,
    bottom_fractions_by_trial: dict[str, float] | None = None,
    use_equilibrium: bool = True,
    use_subcell_coverage: bool = True,
) -> dict[str, object]:
    trial_scores = []
    for batch in batches:
        if baked_geometry is not None and indenter_maps_by_trial is not None:
            ind_map, ind_valid_map = indenter_maps_by_trial[batch.name]
            top_frac = top_fractions_by_trial.get(batch.name, 1.0) if top_fractions_by_trial is not None else 1.0
            bottom_frac = (
                bottom_fractions_by_trial.get(batch.name, 0.0) if bottom_fractions_by_trial is not None else 0.0
            )
            result = evaluate_foundation_baked_batch(
                xy_m,
                baked_geometry,
                ind_map,
                ind_valid_map,
                batch,
                material=material,
                top_fraction=top_frac,
                bottom_fraction=bottom_frac,
                use_equilibrium=use_equilibrium,
                use_subcell_coverage=use_subcell_coverage,
                device=device,
            )
        else:
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
    baked_geometry: BakedMidsoleGeometry | None = None,
    indenter_maps_by_trial: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    top_fractions_by_trial: dict[str, float] | None = None,
    bottom_fractions_by_trial: dict[str, float] | None = None,
    use_equilibrium: bool = True,
    use_subcell_coverage: bool = True,
) -> tuple[FoundationMaterial, dict[str, object]]:
    if not history:
        raise ValueError("Cannot select fit history material without history rows")
    best_material = _material_from_history_row(history[0])
    best_selection: dict[str, object] | None = None
    for row in history:
        material = _material_from_history_row(row)
        selection = _history_selection_score(
            xy_m,
            batches,
            material,
            device=device,
            baked_geometry=baked_geometry,
            indenter_maps_by_trial=indenter_maps_by_trial,
            top_fractions_by_trial=top_fractions_by_trial,
            bottom_fractions_by_trial=bottom_fractions_by_trial,
            use_equilibrium=use_equilibrium,
            use_subcell_coverage=use_subcell_coverage,
        )
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
    device = str(getattr(args, "autodiff_device", "cuda:0"))
    use_baked = _use_surfacemaps(args)
    if not use_baked:
        raise SystemExit("fit-autodiff requires --use-surfacemaps")
    use_equilibrium = _use_equilibrium(args)
    use_subcell_coverage = _use_subcell_coverage(args)
    spring_grid = None
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    batches = _autodiff_batches(
        manifest,
        spring_grid,
        vertices,
        use_baked=use_baked,
        shape_weight_override=getattr(args, "shape_weight", None),
    )

    contact_diagnostics: dict[str, dict[str, float]] = {}
    if not use_baked:
        contact_surfaces = _trial_contact_surface_cache(manifest, spring_grid)
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

    baked_geometry = None
    indenter_maps_by_trial = None
    fit_xy_m = spring_grid.xy_m if spring_grid is not None else None
    surface_map_plot_path = None

    if use_baked:
        baked_spacing = float(manifest.grid.get("baked_spacing_m", 0.002))
        baked_geometry = build_baked_midsole_geometry(
            vertices,
            faces,
            spacing_m=baked_spacing,
            thickness_axis=manifest.grid.get("force_thickness_axis"),
        )
        surface_map_plot_path = _write_surface_map_plot(output_dir, baked_geometry)
        baked_uv_m, _baked_xy_m, baked_cell_area_m2 = _baked_quadrature(baked_geometry)
        fit_xy_m = baked_uv_m
        batches = _with_baked_cell_area(batches, baked_cell_area_m2, baked_geometry.spacing_m)
        indenter_maps_by_trial = {}
        for batch in batches:
            trial = next(t for t in manifest.trials if t.name == batch.name)
            ind_map, ind_valid_map = bake_indenter_maps(
                baked_geometry,
                trial,
                manifest,
                vertices,
                baked_spacing,
            )
            indenter_maps_by_trial[batch.name] = (ind_map, ind_valid_map)

        result = fit_foundation_material_baked_batches_autodiff(
            fit_xy_m,
            baked_geometry,
            indenter_maps_by_trial,
            batches,
            initial_material=_initial_material(manifest),
            iterations=int(args.autodiff_iterations),
            per_cylinder_area=True,
            loop_weight=loop_weight,
            use_equilibrium=use_equilibrium,
            use_subcell_coverage=use_subcell_coverage,
            device=device,
        )
        selected_material, selected_history = _select_history_material(
            fit_xy_m,
            batches,
            list(result.history),
            device=device,
            baked_geometry=baked_geometry,
            indenter_maps_by_trial=indenter_maps_by_trial,
            use_equilibrium=use_equilibrium,
            use_subcell_coverage=use_subcell_coverage,
        )

    loss_plot_path = _write_autodiff_loss_plot(output_dir, list(result.history))
    hysteresis = _write_autodiff_hysteresis_plot(
        output_dir,
        fit_xy_m,
        selected_material,
        batches,
        device=device,
        baked_geometry=baked_geometry,
        indenter_maps_by_trial=indenter_maps_by_trial,
        use_equilibrium=use_equilibrium,
        use_subcell_coverage=use_subcell_coverage,
    )
    material_artifact_path = _write_foundation_material_artifact(
        output_dir,
        manifest,
        spring_grid,
        selected_material,
        hysteresis,
        hysteresis["acceptance"],
        use_baked=use_baked,
        baked_geometry=baked_geometry,
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
        "spring_grid_cells": 0 if spring_grid is None else int(len(spring_grid.xy_m)),
        "surface_map_cells": 0
        if baked_geometry is None or baked_geometry.grid_uv_m is None
        else int(len(baked_geometry.grid_uv_m)),
        "surface_map_plot": surface_map_plot_path,
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


def _z_up_points_from_uv_z(baked_geometry: BakedMidsoleGeometry, uv_m: np.ndarray, z_values: np.ndarray) -> np.ndarray:
    """Map surface-map coordinates into a z-up viewer frame."""
    frame = baked_geometry.frame
    center = frame.center_m.copy()
    plane_axes = frame.plane_axes
    thickness_axis = frame.thickness_axis
    uv = np.asarray(uv_m, dtype=np.float64)
    points = np.zeros((len(uv), 3), dtype=np.float32)
    points[:, 0] = (uv[:, 0] - center[plane_axes[0]]).astype(np.float32)
    points[:, 1] = (uv[:, 1] - center[plane_axes[1]]).astype(np.float32)
    points[:, 2] = (np.asarray(z_values, dtype=np.float64) - center[thickness_axis]).astype(np.float32)
    return points


def _mesh_vertices_z_up(vertices: np.ndarray, baked_geometry: BakedMidsoleGeometry) -> np.ndarray:
    """Map conditioned mesh vertices into the same z-up viewer frame as surface maps."""
    frame = baked_geometry.frame
    center = frame.center_m.copy()
    plane_axes = frame.plane_axes
    thickness_axis = frame.thickness_axis
    vertices = np.asarray(vertices, dtype=np.float64)
    points = np.zeros((len(vertices), 3), dtype=np.float32)
    points[:, 0] = (vertices[:, plane_axes[0]] - center[plane_axes[0]]).astype(np.float32)
    points[:, 1] = (vertices[:, plane_axes[1]] - center[plane_axes[1]]).astype(np.float32)
    points[:, 2] = (vertices[:, thickness_axis] - center[thickness_axis]).astype(np.float32)
    return points


def _surface_map_mesh(
    baked_geometry: BakedMidsoleGeometry,
    z_map: np.ndarray,
    valid_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = z_map.shape
    u = np.linspace(float(baked_geometry.mins_uv[0]), float(baked_geometry.maxs_uv[0]), w, dtype=np.float64)
    v = np.linspace(float(baked_geometry.mins_uv[1]), float(baked_geometry.maxs_uv[1]), h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    uv_flat = np.column_stack((uu.ravel(), vv.ravel()))
    points = _z_up_points_from_uv_z(baked_geometry, uv_flat, np.asarray(z_map, dtype=np.float64).ravel())
    uvs = np.column_stack(
        (
            np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0).ravel(),
            np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1).ravel(),
        )
    ).astype(np.float32)

    valid = np.isfinite(z_map)
    if baked_geometry.valid_map is not None:
        valid &= np.asarray(baked_geometry.valid_map, dtype=np.float64) > 0.5
    if valid_map is not None:
        valid &= np.asarray(valid_map, dtype=np.float64) > 0.5

    triangles: list[tuple[int, int, int]] = []
    for y in range(h - 1):
        for x in range(w - 1):
            i00 = y * w + x
            i10 = y * w + x + 1
            i01 = (y + 1) * w + x
            i11 = (y + 1) * w + x + 1
            if valid[y, x] and valid[y, x + 1] and valid[y + 1, x] and valid[y + 1, x + 1]:
                triangles.append((i00, i10, i11))
                triangles.append((i00, i11, i01))
    return points, np.asarray(triangles, dtype=np.int32).reshape(-1), uvs


def _plane_mesh(
    baked_geometry: BakedMidsoleGeometry,
    z_m: float,
    *,
    padding_m: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u0 = float(baked_geometry.mins_uv[0] - padding_m)
    u1 = float(baked_geometry.maxs_uv[0] + padding_m)
    v0 = float(baked_geometry.mins_uv[1] - padding_m)
    v1 = float(baked_geometry.maxs_uv[1] + padding_m)
    uv = np.asarray([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
    points = _z_up_points_from_uv_z(baked_geometry, uv, np.full(4, float(z_m), dtype=np.float64))
    indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
    uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    return points, indices, uvs


def _sample_map_numpy(
    texture_map: np.ndarray, baked_geometry: BakedMidsoleGeometry, sample_uv_m: np.ndarray
) -> np.ndarray:
    u = (sample_uv_m[:, 0] - baked_geometry.mins_uv[0]) / (baked_geometry.maxs_uv[0] - baked_geometry.mins_uv[0])
    v = (sample_uv_m[:, 1] - baked_geometry.mins_uv[1]) / (baked_geometry.maxs_uv[1] - baked_geometry.mins_uv[1])
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    h, w = texture_map.shape
    px = u * (w - 1.0)
    py = v * (h - 1.0)
    x0 = np.clip(np.floor(px).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(py).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    tx = px - x0
    ty = py - y0
    val00 = texture_map[y0, x0]
    val10 = texture_map[y0, x1]
    val01 = texture_map[y1, x0]
    val11 = texture_map[y1, x1]
    val_top = val00 + tx * (val10 - val00)
    val_bot = val01 + tx * (val11 - val01)
    return val_top + ty * (val_bot - val_top)


def _surface_contact_components(
    baked_geometry: BakedMidsoleGeometry,
    indenter_map: np.ndarray,
    indenter_valid_map: np.ndarray,
    displacement_m: float,
    *,
    top_fraction: float = 1.0,
    bottom_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closure_m = max(float(displacement_m), 0.0)
    valid = np.isfinite(baked_geometry.top_map) & np.isfinite(baked_geometry.bottom_map)
    if baked_geometry.valid_map is not None:
        valid &= np.asarray(baked_geometry.valid_map, dtype=np.float64) > 0.5
    valid &= np.asarray(indenter_valid_map, dtype=np.float64) > 0.5

    top_contact_z = np.asarray(indenter_map, dtype=np.float64) - top_fraction * closure_m
    if baked_geometry.valid_map is None:
        min_bottom = float(np.min(baked_geometry.bottom_map))
    else:
        valid_midsole = np.asarray(baked_geometry.valid_map, dtype=np.float64) > 0.5
        min_bottom = float(np.min(baked_geometry.bottom_map[valid_midsole]))
    bottom_contact_z = min_bottom + bottom_fraction * closure_m

    top_comp = np.zeros_like(baked_geometry.top_map, dtype=np.float64)
    bottom_comp = np.zeros_like(baked_geometry.bottom_map, dtype=np.float64)
    top_comp[valid] = np.maximum(baked_geometry.top_map[valid] - top_contact_z[valid], 0.0)
    bottom_comp[valid] = np.maximum(bottom_contact_z - baked_geometry.bottom_map[valid], 0.0)
    slack = np.maximum(np.asarray(baked_geometry.thickness_map, dtype=np.float64), 1.0e-6)
    total = np.minimum(top_comp + bottom_comp, slack)
    return top_comp, bottom_comp, total


def _compression_heat_texture(values_m: np.ndarray, valid_map: np.ndarray | None, vmax_m: float) -> np.ndarray:
    values = np.asarray(values_m, dtype=np.float64)
    scale = max(float(vmax_m), 1.0e-6)
    t = np.clip(values / scale, 0.0, 1.0)
    stops = np.asarray(
        [
            [26.0, 37.0, 68.0],
            [38.0, 135.0, 118.0],
            [245.0, 203.0, 92.0],
            [224.0, 79.0, 57.0],
        ],
        dtype=np.float64,
    )
    seg = np.clip(t * (len(stops) - 1), 0.0, len(stops) - 1.0)
    lo = np.floor(seg).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (seg - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[hi] * frac
    if valid_map is not None:
        invalid = np.asarray(valid_map, dtype=np.float64) <= 0.5
        rgb[invalid] = np.asarray([30.0, 30.0, 32.0], dtype=np.float64)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def run_surface_scene(args: argparse.Namespace) -> dict[str, object]:
    import newton.viewer  # noqa: PLC0415

    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.cache_dir
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    baked_spacing = float(manifest.grid.get("baked_spacing_m", 0.002))
    baked_geometry = build_baked_midsole_geometry(
        vertices,
        faces,
        spacing_m=baked_spacing,
        thickness_axis=manifest.grid.get("force_thickness_axis"),
    )
    surface_map_plot = _write_surface_map_plot(output_dir, baked_geometry)
    sample_uv_m, _moment_xy_m, _cell_area_m2 = _baked_quadrature(baked_geometry)

    trial = None
    if args.scene_trial is not None:
        trial = next((candidate for candidate in manifest.trials if candidate.name == args.scene_trial), None)
        if trial is None:
            raise ValueError(f"Unknown scene trial {args.scene_trial!r}")
    else:
        trial = next((candidate for candidate in manifest.trials if candidate.include_in_fit), None)
    if trial is None:
        raise ValueError("No included trial is available for the surface scene")

    ind_map, ind_valid_map = bake_indenter_maps(baked_geometry, trial, manifest, vertices, baked_spacing)
    trace_csv = output_dir / f"digital_instron_v2_dynamic_{_safe_report_name(trial.name)}.csv"
    if trace_csv.exists():
        data = np.genfromtxt(trace_csv, delimiter=",", names=True, dtype=np.float64)
        if data.shape == ():
            data = np.asarray([data], dtype=data.dtype)
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        displacement_m = np.asarray(data["displacement_m"], dtype=np.float64)
    else:
        trace = _load_averaged_cycle(_trial_averaged_cycle_path(trial))
        time_s = trace["time_s"]
        displacement_m = trace["displacement_m"]

    frame_count = min(int(args.scene_max_frames), len(displacement_m))
    if frame_count <= 0:
        raise ValueError("Surface scene needs at least one replay frame")

    bottom_z = _sample_map_numpy(baked_geometry.bottom_map, baked_geometry, sample_uv_m)
    midsole_mesh_points = _mesh_vertices_z_up(vertices, baked_geometry)
    midsole_mesh_indices = np.asarray(faces, dtype=np.int32).reshape(-1)
    top_mesh_points, top_mesh_indices, top_mesh_uvs = _surface_map_mesh(
        baked_geometry,
        baked_geometry.top_map + 0.0008,
    )
    bottom_mesh_points, bottom_mesh_indices, bottom_mesh_uvs = _surface_map_mesh(
        baked_geometry,
        baked_geometry.bottom_map - 0.0008,
    )
    _foot_mesh_points0, foot_mesh_indices, foot_mesh_uvs = _surface_map_mesh(baked_geometry, ind_map, ind_valid_map)
    min_bottom_z = float(np.min(bottom_z))
    ground_z = min_bottom_z
    ground_points, ground_indices, ground_uvs = _plane_mesh(baked_geometry, ground_z, padding_m=0.03)
    ground_heat_z0 = np.full_like(baked_geometry.bottom_map, min_bottom_z, dtype=np.float64)
    _ground_heat_points0, ground_heat_indices, ground_heat_uvs = _surface_map_mesh(
        baked_geometry,
        ground_heat_z0,
        ind_valid_map,
    )

    viewer = newton.viewer.ViewerNull(num_frames=frame_count) if args.viewer == "null" else newton.viewer.ViewerGL()
    if hasattr(viewer, "set_camera"):
        extents = np.asarray(baked_geometry.frame.extents_m, dtype=np.float64)
        span = max(float(np.max(extents)), 0.20)
        viewer.set_camera(wp.vec3(0.55 * span, -1.75 * span, 0.85 * span), pitch=-26.0, yaw=-18.0)
    viewer_device = viewer.device
    midsole_mesh_points_wp = wp.array(midsole_mesh_points, dtype=wp.vec3, device=viewer_device)
    midsole_mesh_indices_wp = wp.array(midsole_mesh_indices, dtype=wp.int32, device=viewer_device)
    top_mesh_points_wp = wp.array(top_mesh_points, dtype=wp.vec3, device=viewer_device)
    top_mesh_indices_wp = wp.array(top_mesh_indices, dtype=wp.int32, device=viewer_device)
    top_mesh_uvs_wp = wp.array(top_mesh_uvs, dtype=wp.vec2, device=viewer_device)
    bottom_mesh_points_wp = wp.array(bottom_mesh_points, dtype=wp.vec3, device=viewer_device)
    bottom_mesh_indices_wp = wp.array(bottom_mesh_indices, dtype=wp.int32, device=viewer_device)
    bottom_mesh_uvs_wp = wp.array(bottom_mesh_uvs, dtype=wp.vec2, device=viewer_device)
    foot_mesh_indices_wp = wp.array(foot_mesh_indices, dtype=wp.int32, device=viewer_device)
    foot_mesh_uvs_wp = wp.array(foot_mesh_uvs, dtype=wp.vec2, device=viewer_device)
    ground_heat_indices_wp = wp.array(ground_heat_indices, dtype=wp.int32, device=viewer_device)
    ground_heat_uvs_wp = wp.array(ground_heat_uvs, dtype=wp.vec2, device=viewer_device)
    ground_points_wp = wp.array(ground_points, dtype=wp.vec3, device=viewer_device)
    ground_indices_wp = wp.array(ground_indices, dtype=wp.int32, device=viewer_device)
    ground_uvs_wp = wp.array(ground_uvs, dtype=wp.vec2, device=viewer_device)
    heat_vmax_m = max(float(np.nanpercentile(baked_geometry.thickness_map, 95.0)) * 0.30, 0.002)
    frame_index = 0
    rendered_frames = 0
    while viewer.is_running():
        disp = float(displacement_m[frame_index])
        sim_time = float(time_s[frame_index]) if frame_index < len(time_s) else float(frame_index) / 60.0
        top_comp, bottom_comp, total_comp = _surface_contact_components(baked_geometry, ind_map, ind_valid_map, disp)
        active = total_comp > 0.0
        active_any = bool(np.any(active))
        frame_vmax = max(heat_vmax_m, float(np.max(total_comp)) if active_any else 0.0)
        top_texture = _compression_heat_texture(top_comp, ind_valid_map, frame_vmax)
        bottom_texture = _compression_heat_texture(bottom_comp, ind_valid_map, frame_vmax)
        support_texture = _compression_heat_texture(total_comp, ind_valid_map, frame_vmax)
        foot_mesh_points, _foot_mesh_indices, _foot_mesh_uvs = _surface_map_mesh(
            baked_geometry,
            ind_map - max(disp, 0.0),
            ind_valid_map,
        )
        ground_heat_z = np.full_like(baked_geometry.bottom_map, min_bottom_z, dtype=np.float64)
        ground_heat_points, _ground_heat_indices, _ground_heat_uvs = _surface_map_mesh(
            baked_geometry,
            ground_heat_z,
            ind_valid_map,
        )
        platen_points, platen_indices, _platen_uvs = _plane_mesh(
            baked_geometry,
            min_bottom_z,
            padding_m=0.01,
        )

        viewer.begin_frame(sim_time)
        viewer.log_mesh(
            "/digital_instron/actual_midsole_mesh",
            midsole_mesh_points_wp,
            midsole_mesh_indices_wp,
            color=(0.56, 0.58, 0.62),
            roughness=0.72,
            backface_culling=False,
        )
        viewer.log_mesh(
            "/digital_instron/midsole_top_contact_heatmap",
            top_mesh_points_wp,
            top_mesh_indices_wp,
            uvs=top_mesh_uvs_wp,
            texture=np.flipud(top_texture),
            roughness=0.55,
            hidden=not active_any,
            backface_culling=False,
        )
        viewer.log_mesh(
            "/digital_instron/midsole_bottom_contact_heatmap",
            bottom_mesh_points_wp,
            bottom_mesh_indices_wp,
            uvs=bottom_mesh_uvs_wp,
            texture=np.flipud(bottom_texture),
            roughness=0.65,
            hidden=not active_any,
            backface_culling=False,
        )
        viewer.log_mesh(
            "/digital_instron/ground_plane",
            ground_points_wp,
            ground_indices_wp,
            uvs=ground_uvs_wp,
            color=(0.22, 0.22, 0.24),
            roughness=0.85,
            backface_culling=False,
        )
        viewer.log_mesh(
            "/digital_instron/bottom_platen",
            wp.array(platen_points, dtype=wp.vec3, device=viewer_device),
            wp.array(platen_indices, dtype=wp.int32, device=viewer_device),
            color=(0.52, 0.52, 0.56),
            roughness=0.35,
            backface_culling=False,
        )
        viewer.log_mesh(
            "/digital_instron/foot_resolved_contact_heatmap",
            wp.array(foot_mesh_points, dtype=wp.vec3, device=viewer_device),
            foot_mesh_indices_wp,
            uvs=foot_mesh_uvs_wp,
            texture=np.flipud(top_texture),
            roughness=0.35,
            hidden=(len(foot_mesh_indices) == 0) or (not active_any),
            backface_culling=False,
        )
        viewer.log_mesh(
            "/digital_instron/ground_resolved_contact_heatmap",
            wp.array(ground_heat_points, dtype=wp.vec3, device=viewer_device),
            ground_heat_indices_wp,
            uvs=ground_heat_uvs_wp,
            texture=np.flipud(support_texture),
            roughness=0.35,
            hidden=not np.any(active),
            backface_culling=False,
        )
        viewer.end_frame()
        rendered_frames += 1
        frame_index = (frame_index + 1) % frame_count

    viewer.close()
    return {
        "schema_version": "digital_instron_v2_surface_scene_1",
        "viewer": args.viewer,
        "trial": trial.name,
        "trace_frame_count": int(frame_count),
        "rendered_frame_count": int(rendered_frames),
        "looped_until_closed": True,
        "surface_map_plot": surface_map_plot,
        "source_trace": str(trace_csv if trace_csv.exists() else trial.averaged_cycle_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.step == "qc":
        report = run_qc(args)
    elif args.step == "split-cycles":
        report = run_split_cycles(args)
    elif args.step == "fit-autodiff":
        report = run_fit_autodiff(args)
    elif args.step == "visualize":
        report = run_visualize(args)
    else:
        report = run_surface_scene(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
