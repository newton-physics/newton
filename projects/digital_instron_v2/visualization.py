# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Offline visual diagnostics for the Digital Instron v2 workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .foundation import FoundationMaterial, evaluate_foundation_lengths
from .geometry import (
    SpringSurfaceGrid,
    _load_obj_mesh,
    build_raycast_spring_grid,
    condition_midsole_mesh,
    place_rearfoot_punch_grid,
)
from .manifest import TrialManifest


def _rearfoot_punch_radius_m(manifest: TrialManifest) -> float:
    for trial in manifest.trials:
        if trial.fixture == "rearfoot_punch":
            indenter = trial.indenter
            if "diameter_m" in indenter:
                return float(indenter["diameter_m"]) * 0.5
            if "radius_m" in indenter:
                return float(indenter["radius_m"])
    if "rearfoot_punch_diameter_m" in manifest.grid:
        return float(manifest.grid["rearfoot_punch_diameter_m"]) * 0.5
    return float(manifest.grid.get("cylinder_radius_m", 0.0225))


def _load_conditioned_mesh(manifest: TrialManifest, output_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    report = condition_midsole_mesh(
        manifest.midsole_mesh,
        output_dir,
        source_units=str(manifest.qc.get("mesh_source_units", "mm")),
        min_thickness_m=float(manifest.qc.get("min_midsole_thickness_m", 0.005)),
        max_thickness_m=float(manifest.qc.get("max_midsole_thickness_m", 0.08)),
    )
    vertices, faces = _load_obj_mesh(Path(str(report["repaired_mesh"])))
    return vertices, faces, report


def _spring_forces(
    spring_grid: SpringSurfaceGrid,
    material: FoundationMaterial,
    displacement_m: float,
) -> np.ndarray:
    compression = np.clip(displacement_m, 0.0, spring_grid.slack_length_m)
    strain = compression / spring_grid.slack_length_m
    normalized = np.minimum(strain / max(material.lock_strain, 1.0e-4), 0.999)
    alpha = max(material.ogden_alpha, 1.0e-4)
    elastic_stress = material.stiffness_pa * ((1.0 - normalized) ** (-alpha) - 1.0) / alpha
    return spring_grid.cell_area_m2 * np.maximum(elastic_stress, 0.0)


def write_visualization_report(manifest: TrialManifest, output_dir: str | Path) -> dict[str, Any]:
    """Write mesh, raycast, and spring diagnostics as PNGs plus JSON."""

    import matplotlib.pyplot as plt  # noqa: PLC0415

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vertices, faces, mesh_report = _load_conditioned_mesh(manifest, output)
    footprint_grid = build_raycast_spring_grid(
        vertices,
        faces,
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
        min_slack_length_m=float(manifest.qc.get("min_spring_slack_length_m", 0.001)),
    )
    frame = footprint_grid.frame
    rearfoot_grid = place_rearfoot_punch_grid(
        vertices,
        radius_m=_rearfoot_punch_radius_m(manifest),
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
        frame=frame,
        heel_side=str(manifest.grid.get("rearfoot_heel_side", "min")),
        length_fraction=float(manifest.grid.get("rearfoot_length_fraction", 0.25)),
        lateral_fraction=float(manifest.grid.get("rearfoot_lateral_fraction", 0.5)),
        lateral_band_fraction=float(manifest.grid.get("rearfoot_lateral_band_fraction", 0.12)),
    )
    nominal_thickness_m = float(manifest.fit.get("nominal_midsole_thickness_m", mesh_report["thickness_m"]))
    material = FoundationMaterial(
        stiffness_pa=float(manifest.fit.get("initial_stiffness_pa", 2.0e6)),
        ogden_alpha=float(manifest.fit.get("initial_ogden_alpha", 2.0)),
        lock_strain=float(manifest.fit.get("initial_lock_strain", 0.85)),
        damping_pa_s=float(manifest.fit.get("initial_damping_pa_s", 1.0e4)),
        damping_power=float(manifest.fit.get("initial_damping_power", 1.0)),
    )

    plane = vertices[:, frame.plane_axes]
    thickness = vertices[:, frame.thickness_axis]
    sampled = np.linspace(0, len(vertices) - 1, min(len(vertices), 6000), dtype=np.int64)
    orientation_png = output / "digital_instron_v2_mesh_orientation.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    scatter = axes[0].scatter(plane[sampled, 0], plane[sampled, 1], c=thickness[sampled], s=1.0, cmap="viridis")
    axes[0].set_title("mesh projection")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(f"axis {frame.plane_axes[0]} [m]")
    axes[0].set_ylabel(f"axis {frame.plane_axes[1]} [m]")
    fig.colorbar(scatter, ax=axes[0], label=f"axis {frame.thickness_axis} [m]")
    axes[1].hist(thickness, bins=80, color="#4c78a8")
    axes[1].set_title("thickness-axis distribution")
    axes[1].set_xlabel(f"axis {frame.thickness_axis} [m]")
    axes[1].set_ylabel("vertex count")
    axes[2].bar(["x", "y", "z"], frame.extents_m, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[2].set_title("axis extents")
    axes[2].set_ylabel("extent [m]")
    axes[2].axhline(nominal_thickness_m, color="#e45756", linestyle="--", label="nominal thickness")
    axes[2].legend()
    fig.savefig(orientation_png, dpi=180)
    plt.close(fig)

    ray_png = output / "digital_instron_v2_raycast_grid.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    sc = axes[0].scatter(
        footprint_grid.grid_uv_m[:, 0],
        footprint_grid.grid_uv_m[:, 1],
        c=footprint_grid.slack_length_m,
        s=14,
        cmap="magma",
    )
    axes[0].scatter(plane[sampled, 0], plane[sampled, 1], c="0.7", s=0.2, alpha=0.2)
    axes[0].set_title("full-footprint raycast slack length")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(f"axis {frame.plane_axes[0]} [m]")
    axes[0].set_ylabel(f"axis {frame.plane_axes[1]} [m]")
    fig.colorbar(sc, ax=axes[0], label="spring slack length [m]")
    axes[1].scatter(
        footprint_grid.grid_uv_m[:, 0],
        footprint_grid.grid_uv_m[:, 1],
        c=footprint_grid.hit_count,
        s=14,
        cmap="plasma",
    )
    axes[1].scatter(
        rearfoot_grid.grid_uv_m[:, 0],
        rearfoot_grid.grid_uv_m[:, 1],
        facecolors="none",
        edgecolors="#e45756",
        s=28,
        linewidths=0.7,
    )
    axes[1].scatter(
        [rearfoot_grid.center_uv_m[0]],
        [rearfoot_grid.center_uv_m[1]],
        c="#e45756",
        marker="x",
        s=50,
    )
    axes[1].set_title("hit count with rearfoot punch overlay")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel(f"axis {frame.plane_axes[0]} [m]")
    axes[1].set_ylabel(f"axis {frame.plane_axes[1]} [m]")
    fig.savefig(ray_png, dpi=180)
    plt.close(fig)

    max_displacement = min(nominal_thickness_m * float(material.lock_strain) * 0.8, 0.04)
    displacement = np.linspace(0.0, max_displacement, 80)
    force = np.empty_like(displacement)
    for i, value in enumerate(displacement):
        current_length = np.maximum(footprint_grid.slack_length_m - value, 0.0)
        result = evaluate_foundation_lengths(
            footprint_grid.xy_m,
            current_length,
            footprint_grid.slack_length_m,
            np.zeros_like(current_length),
            cell_area_m2=footprint_grid.cell_area_m2,
            material=material,
        )
        force[i] = result.force_n

    response_png = output / "digital_instron_v2_spring_response.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(displacement * 1000.0, force, color="#4c78a8")
    ax.set_title("raycast-slack 1D spring response")
    ax.set_xlabel("compression [mm]")
    ax.set_ylabel("aggregate force [N]")
    ax.grid(True, alpha=0.3)
    fig.savefig(response_png, dpi=180)
    plt.close(fig)

    snapshot_displacement_m = max_displacement * 0.5
    cell_force = _spring_forces(footprint_grid, material, snapshot_displacement_m)
    snapshot_png = output / "digital_instron_v2_spring_snapshot.png"
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sc = ax.scatter(footprint_grid.grid_uv_m[:, 0], footprint_grid.grid_uv_m[:, 1], c=cell_force, s=18, cmap="cividis")
    ax.set_title(f"per-cell spring force at {snapshot_displacement_m * 1000.0:.1f} mm")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"axis {frame.plane_axes[0]} [m]")
    ax.set_ylabel(f"axis {frame.plane_axes[1]} [m]")
    fig.colorbar(sc, ax=ax, label="cell force [N]")
    fig.savefig(snapshot_png, dpi=180)
    plt.close(fig)

    spring_grid_csv = output / "digital_instron_v2_spring_grid.csv"
    spring_grid_npz = output / "digital_instron_v2_spring_grid.npz"
    spring_grid_table = np.column_stack(
        (
            footprint_grid.xy_m[:, 0],
            footprint_grid.xy_m[:, 1],
            footprint_grid.grid_uv_m[:, 0],
            footprint_grid.grid_uv_m[:, 1],
            footprint_grid.slack_length_m,
            footprint_grid.bottom_m,
            footprint_grid.top_m,
            footprint_grid.hit_count,
        )
    )
    np.savetxt(
        spring_grid_csv,
        spring_grid_table,
        delimiter=",",
        header="local_x_m,local_y_m,mesh_u_m,mesh_v_m,slack_length_m,bottom_m,top_m,hit_count",
        comments="",
    )
    np.savez(
        spring_grid_npz,
        xy_m=footprint_grid.xy_m,
        grid_uv_m=footprint_grid.grid_uv_m,
        slack_length_m=footprint_grid.slack_length_m,
        bottom_m=footprint_grid.bottom_m,
        top_m=footprint_grid.top_m,
        hit_count=footprint_grid.hit_count,
        cell_area_m2=np.asarray([footprint_grid.cell_area_m2], dtype=np.float64),
        spacing_m=np.asarray([footprint_grid.spacing_m], dtype=np.float64),
        plane_axes=np.asarray(frame.plane_axes, dtype=np.int32),
        thickness_axis=np.asarray([frame.thickness_axis], dtype=np.int32),
    )

    report: dict[str, Any] = {
        "manifest": str(manifest.path),
        "mesh_report": mesh_report,
        "detected_thickness_axis": int(frame.thickness_axis),
        "detected_plane_axes": [int(axis) for axis in frame.plane_axes],
        "axis_extents_m": frame.extents_m.tolist(),
        "footprint_grid_cells": int(len(footprint_grid.xy_m)),
        "rearfoot_punch_grid_cells": int(len(rearfoot_grid.local_xy_m)),
        "rearfoot_punch_radius_m": float(rearfoot_grid.radius_m),
        "rearfoot_punch_center_uv_m": rearfoot_grid.center_uv_m.tolist(),
        "rearfoot_heel_side": str(manifest.grid.get("rearfoot_heel_side", "min")),
        "rearfoot_length_fraction": float(manifest.grid.get("rearfoot_length_fraction", 0.25)),
        "rearfoot_lateral_fraction": float(manifest.grid.get("rearfoot_lateral_fraction", 0.5)),
        "rearfoot_lateral_band_fraction": float(manifest.grid.get("rearfoot_lateral_band_fraction", 0.12)),
        "spring_slack_min_m": float(np.min(footprint_grid.slack_length_m)),
        "spring_slack_median_m": float(np.median(footprint_grid.slack_length_m)),
        "spring_slack_max_m": float(np.max(footprint_grid.slack_length_m)),
        "spring_snapshot_displacement_m": float(snapshot_displacement_m),
        "spring_snapshot_total_force_n": float(np.sum(cell_force)),
        "outputs": {
            "mesh_orientation_png": str(orientation_png),
            "raycast_grid_png": str(ray_png),
            "spring_response_png": str(response_png),
            "spring_snapshot_png": str(snapshot_png),
            "spring_grid_csv": str(spring_grid_csv),
            "spring_grid_npz": str(spring_grid_npz),
        },
    }
    report_path = output / "digital_instron_v2_visualization.summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["outputs"]["summary_json"] = str(report_path)
    return report
