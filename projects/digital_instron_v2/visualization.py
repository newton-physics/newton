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


def _trial_peak_displacement_m(trial) -> float | None:
    if trial.averaged_cycle_path is None or not trial.averaged_cycle_path.exists():
        return None
    data = np.genfromtxt(trial.averaged_cycle_path, delimiter=",", names=True, dtype=np.float64)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    if data.dtype.names is None or "displacement_m" not in data.dtype.names:
        return None
    displacement = np.asarray(data["displacement_m"], dtype=np.float64)
    finite = np.isfinite(displacement)
    if not np.any(finite):
        return None
    return float(np.max(displacement[finite]))


def _load_trial_impact_trace(trial) -> dict[str, np.ndarray] | None:
    if trial.averaged_cycle_path is None or not trial.averaged_cycle_path.exists():
        return None
    data = np.genfromtxt(trial.averaged_cycle_path, delimiter=",", names=True, dtype=np.float64)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    if data.dtype.names is None or "displacement_m" not in data.dtype.names:
        return None
    displacement = np.asarray(data["displacement_m"], dtype=np.float64)
    if "time_s" in data.dtype.names:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
    else:
        time_s = np.arange(len(displacement), dtype=np.float64)
    if "velocity_m_s" in data.dtype.names:
        velocity_mps = np.asarray(data["velocity_m_s"], dtype=np.float64)
    else:
        velocity_mps = np.gradient(displacement, time_s)
    finite = np.isfinite(time_s) & np.isfinite(displacement) & np.isfinite(velocity_mps)
    if int(np.count_nonzero(finite)) < 2:
        return None
    return {
        "time_s": time_s[finite],
        "displacement_m": displacement[finite],
        "velocity_mps": velocity_mps[finite],
    }


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def write_visualization_report(manifest: TrialManifest, output_dir: str | Path) -> dict[str, Any]:
    """Write mesh, raycast, and spring diagnostics as PNGs plus JSON."""

    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vertices, faces, mesh_report = _load_conditioned_mesh(manifest, output)
    footprint_grid = build_raycast_spring_grid(
        vertices,
        faces,
        spacing_m=float(manifest.grid.get("coarse_spacing_m", 0.005)),
        min_slack_length_m=float(manifest.qc.get("min_spring_slack_length_m", 0.001)),
        thickness_axis=manifest.grid.get("force_thickness_axis"),
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

    material = FoundationMaterial(
        stiffness_pa=stiffness,
        ogden_alpha=float(manifest.fit.get("initial_ogden_alpha", 2.0)),
        lock_strain=float(manifest.fit.get("initial_lock_strain", 0.85)),
        damping_pa_s=float(manifest.fit.get("initial_damping_pa_s", 1.0e4)),
        damping_power=float(manifest.fit.get("initial_damping_power", 1.0)),
        prony_stiffness_pa=prony_stiffness,
        prony_damping_pa_s=prony_damping,
        pasternak_stiffness_n_per_m=float(
            manifest.fit.get(
                "initial_pasternak_stiffness_n_per_m",
                manifest.fit.get("initial_shear_modulus_pa", 0.0),
            )
        ),
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

    force_heatmap_png = output / "digital_instron_v2_force_heatmap.png"
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sc = ax.scatter(
        footprint_grid.xy_m[:, 0],
        footprint_grid.xy_m[:, 1],
        c=cell_force,
        s=18,
        cmap="coolwarm",
    )
    ax.set_title("Per-Cylinder Force Distribution")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"axis {frame.plane_axes[0]} [m]")
    ax.set_ylabel(f"axis {frame.plane_axes[1]} [m]")
    fig.colorbar(sc, ax=ax, label="Force [N]")
    fig.savefig(force_heatmap_png, dpi=180)
    plt.close(fig)

    from .workflow import _rearfoot_mask, _spring_state_for_trial_frame, _trial_contact_surface_cache

    contact_surfaces = _trial_contact_surface_cache(manifest, footprint_grid)
    rearfoot_mask = _rearfoot_mask(manifest, footprint_grid, vertices)
    squish_trials = [trial for trial in manifest.trials if trial.include_in_fit]
    squish_png = output / "digital_instron_v2_contact_squish.png"
    compression_map_png = output / "digital_instron_v2_max_compression_map.png"
    squish_summary: list[dict[str, float | str]] = []
    impact_animation_outputs: dict[str, str] = {}
    impact_animation_frames: dict[str, int] = {}
    if squish_trials:
        fig, axes = plt.subplots(
            len(squish_trials),
            1,
            figsize=(8.5, 3.8 * len(squish_trials)),
            constrained_layout=True,
            squeeze=False,
        )
        length_axis = int(np.argmax(frame.extents_m[list(frame.plane_axes)]))
        x_m = footprint_grid.grid_uv_m[:, length_axis]
        x_mm = (x_m - float(np.min(x_m))) * 1000.0
        xlim_map = (
            float(np.min(footprint_grid.grid_uv_m[:, 0]) * 1000.0),
            float(np.max(footprint_grid.grid_uv_m[:, 0]) * 1000.0),
        )
        ylim_map = (
            float(np.min(footprint_grid.grid_uv_m[:, 1]) * 1000.0),
            float(np.max(footprint_grid.grid_uv_m[:, 1]) * 1000.0),
        )
        z_min_mm = float(np.min(footprint_grid.bottom_m) * 1000.0)
        z_max_mm = float(np.max(footprint_grid.top_m) * 1000.0)
        fig_map, map_axes = plt.subplots(
            1,
            len(squish_trials),
            figsize=(5.5 * len(squish_trials), 5.0),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, trial in zip(axes[:, 0], squish_trials, strict=True):
            displacement_m = _trial_peak_displacement_m(trial)
            if displacement_m is None:
                displacement_m = snapshot_displacement_m
            current_length, velocity = _spring_state_for_trial_frame(
                footprint_grid,
                trial,
                rearfoot_mask,
                contact_surfaces,
                displacement_m,
                0.0,
            )
            compression = np.maximum(footprint_grid.slack_length_m - current_length, 0.0)
            deformed_top = footprint_grid.top_m - compression
            active = compression > 0.0

            ax.scatter(x_mm, footprint_grid.bottom_m * 1000.0, s=7, c="#595959", alpha=0.35, label="ground side")
            ax.scatter(x_mm, footprint_grid.top_m * 1000.0, s=7, c="#9ecae1", alpha=0.28, label="uncompressed top")
            sc = ax.scatter(
                x_mm,
                deformed_top * 1000.0,
                c=compression * 1000.0,
                s=np.where(active, 18, 8),
                cmap="inferno",
                alpha=np.where(active, 0.9, 0.25),
                label="compressed top",
            )
            if trial.fixture == "fullfoot_last" and trial.name in contact_surfaces:
                contact_surface_0, valid = contact_surfaces[trial.name]
                contact_surface = contact_surface_0 - displacement_m
                ax.scatter(
                    x_mm[valid],
                    contact_surface[valid] * 1000.0,
                    s=10,
                    c="#2ca02c",
                    alpha=0.45,
                    label="last contact surface",
                )
            elif trial.fixture == "rearfoot_punch":
                platen = footprint_grid.top_m - displacement_m
                ax.scatter(
                    x_mm[rearfoot_mask],
                    platen[rearfoot_mask] * 1000.0,
                    s=12,
                    c="#2ca02c",
                    alpha=0.55,
                    label="flat punch platen",
                )

            ax.set_title(f"{trial.name}: peak squish at {displacement_m * 1000.0:.2f} mm")
            ax.set_xlabel("footprint length [mm]")
            ax.set_ylabel(f"axis {frame.thickness_axis} [mm]")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize="small")
            fig.colorbar(sc, ax=ax, label="compression [mm]")
            squish_summary.append(
                {
                    "trial": trial.name,
                    "fixture": trial.fixture,
                    "displacement_m": float(displacement_m),
                    "active_cells": int(np.count_nonzero(active)),
                    "max_compression_m": float(np.max(compression)) if len(compression) else 0.0,
                    "mean_active_compression_m": float(np.mean(compression[active])) if np.any(active) else 0.0,
                }
            )
            map_ax = map_axes[0, squish_trials.index(trial)]
            sc_map = map_ax.scatter(
                footprint_grid.grid_uv_m[:, 0] * 1000.0,
                footprint_grid.grid_uv_m[:, 1] * 1000.0,
                c=compression * 1000.0,
                s=np.where(active, 24, 8),
                cmap="inferno",
                alpha=np.where(active, 0.9, 0.22),
            )
            map_ax.scatter(
                footprint_grid.grid_uv_m[active, 0] * 1000.0,
                footprint_grid.grid_uv_m[active, 1] * 1000.0,
                facecolors="none",
                edgecolors="#2ca02c",
                s=32,
                linewidths=0.45,
            )
            map_ax.set_title(f"{trial.name}: max compression")
            map_ax.set_aspect("equal", adjustable="box")
            map_ax.set_xlabel(f"axis {frame.plane_axes[0]} [mm]")
            map_ax.set_ylabel(f"axis {frame.plane_axes[1]} [mm]")
            map_ax.grid(True, alpha=0.2)
            fig_map.colorbar(sc_map, ax=map_ax, label="compression [mm]")
        fig.savefig(squish_png, dpi=180)
        plt.close(fig)
        fig_map.savefig(compression_map_png, dpi=180)
        plt.close(fig_map)

        from matplotlib.animation import FuncAnimation, PillowWriter

        max_animation_frames = int(manifest.fit.get("impact_animation_max_frames", 120))
        for trial in squish_trials:
            trace = _load_trial_impact_trace(trial)
            if trace is None:
                continue
            frame_count = len(trace["displacement_m"])
            if max_animation_frames > 0 and frame_count > max_animation_frames:
                frame_indices = np.linspace(0, frame_count - 1, max_animation_frames, dtype=np.int64)
            else:
                frame_indices = np.arange(frame_count, dtype=np.int64)

            compressions = []
            deformed_tops = []
            contact_lines = []
            for frame_index in frame_indices:
                displacement_m = float(trace["displacement_m"][frame_index])
                velocity_mps = float(trace["velocity_mps"][frame_index])
                current_length, _ = _spring_state_for_trial_frame(
                    footprint_grid,
                    trial,
                    rearfoot_mask,
                    contact_surfaces,
                    displacement_m,
                    velocity_mps,
                )
                compression = np.maximum(footprint_grid.slack_length_m - current_length, 0.0)
                compressions.append(compression)
                deformed_tops.append(footprint_grid.top_m - compression)
                if trial.fixture == "fullfoot_last" and trial.name in contact_surfaces:
                    contact_surface_0, valid = contact_surfaces[trial.name]
                    contact_lines.append((x_mm[valid], (contact_surface_0[valid] - displacement_m) * 1000.0))
                elif trial.fixture == "rearfoot_punch":
                    contact_lines.append(
                        (x_mm[rearfoot_mask], (footprint_grid.top_m[rearfoot_mask] - displacement_m) * 1000.0)
                    )
                else:
                    contact_lines.append((np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)))

            compression_vmax_mm = max(float(np.max([np.max(c) for c in compressions]) * 1000.0), 1.0)
            fig_anim, (ax_map, ax_side) = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
            top_scatter = ax_map.scatter(
                footprint_grid.grid_uv_m[:, 0] * 1000.0,
                footprint_grid.grid_uv_m[:, 1] * 1000.0,
                c=compressions[0] * 1000.0,
                s=20,
                cmap="inferno",
                vmin=0.0,
                vmax=compression_vmax_mm,
            )
            ax_map.set_aspect("equal", adjustable="box")
            ax_map.set_xlim(*xlim_map)
            ax_map.set_ylim(*ylim_map)
            ax_map.set_xlabel(f"axis {frame.plane_axes[0]} [mm]")
            ax_map.set_ylabel(f"axis {frame.plane_axes[1]} [mm]")
            fig_anim.colorbar(top_scatter, ax=ax_map, label="compression [mm]")

            ax_side.scatter(x_mm, footprint_grid.bottom_m * 1000.0, s=6, c="#595959", alpha=0.35, label="ground side")
            ax_side.scatter(x_mm, footprint_grid.top_m * 1000.0, s=6, c="#9ecae1", alpha=0.25, label="uncompressed top")
            side_scatter = ax_side.scatter(
                x_mm,
                deformed_tops[0] * 1000.0,
                c=compressions[0] * 1000.0,
                s=14,
                cmap="inferno",
                vmin=0.0,
                vmax=compression_vmax_mm,
                label="compressed top",
            )
            contact_x, contact_y = contact_lines[0]
            contact_scatter = ax_side.scatter(contact_x, contact_y, s=9, c="#2ca02c", alpha=0.5, label="indenter")
            ax_side.set_xlim(float(np.min(x_mm)), float(np.max(x_mm)))
            ax_side.set_ylim(z_min_mm - 5.0, z_max_mm + 5.0)
            ax_side.set_xlabel("footprint length [mm]")
            ax_side.set_ylabel(f"axis {frame.thickness_axis} [mm]")
            ax_side.grid(True, alpha=0.25)
            ax_side.legend(loc="best", fontsize="small")
            title = fig_anim.suptitle("")

            def update(frame_number: int):
                compression = compressions[frame_number]
                top_scatter.set_array(compression * 1000.0)
                side_scatter.set_offsets(np.column_stack((x_mm, deformed_tops[frame_number] * 1000.0)))
                side_scatter.set_array(compression * 1000.0)
                contact_x_frame, contact_y_frame = contact_lines[frame_number]
                contact_scatter.set_offsets(np.column_stack((contact_x_frame, contact_y_frame)))
                source_index = frame_indices[frame_number]
                title.set_text(
                    f"{trial.name} impact | "
                    f"t={trace['time_s'][source_index]:.4f} s | "
                    f"disp={trace['displacement_m'][source_index] * 1000.0:.2f} mm"
                )
                return top_scatter, side_scatter, contact_scatter, title

            animation_path = output / f"digital_instron_v2_impact_{_safe_name(trial.name)}.gif"
            animation = FuncAnimation(fig_anim, update, frames=len(frame_indices), interval=80, blit=False)
            animation.save(animation_path, writer=PillowWriter(fps=12), dpi=110)
            plt.close(fig_anim)
            impact_animation_outputs[trial.name] = str(animation_path)
            impact_animation_frames[trial.name] = int(len(frame_indices))

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
        "contact_squish": squish_summary,
        "impact_animation_frames": impact_animation_frames,
        "outputs": {
            "mesh_orientation_png": str(orientation_png),
            "raycast_grid_png": str(ray_png),
            "spring_response_png": str(response_png),
            "spring_snapshot_png": str(snapshot_png),
            "force_heatmap_png": str(force_heatmap_png),
            "contact_squish_png": str(squish_png),
            "max_compression_map_png": str(compression_map_png),
            "impact_animation_gif": impact_animation_outputs,
            "spring_grid_csv": str(spring_grid_csv),
            "spring_grid_npz": str(spring_grid_npz),
        },
    }
    report_path = output / "digital_instron_v2_visualization.summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["outputs"]["summary_json"] = str(report_path)
    return report
