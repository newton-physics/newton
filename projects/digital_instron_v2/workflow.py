# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit the Digital Instron material model."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import (
    EFFECTIVE_POISSON_RATIO,
    MAXWELL_RELAXATION_TIME_S,
    Material,
    Trial,
    fit_material,
    metrics,
    predict,
)
from .geometry import build_column_grid, load_mesh, raycast_surface, rearfoot_center, transform_mesh


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, displacement, and baseline-corrected force."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    time = np.asarray(data["time_s"])
    displacement = np.maximum(np.asarray(data["displacement_m"]), 0.0)
    force = np.asarray(data["force_n"])
    return time, displacement, force - np.min(force)


def compression_laplacian(
    compression_m: np.ndarray, uv_m: np.ndarray, grid_uv_m: np.ndarray, spacing_m: float
) -> np.ndarray:
    """Return the lateral compression Laplacian with natural outer boundaries."""

    cells = [tuple(np.rint(point / spacing_m).astype(int)) for point in uv_m]
    trial_index = {cell: index for index, cell in enumerate(cells)}
    full_cells = {tuple(np.rint(point / spacing_m).astype(int)) for point in grid_uv_m}
    neighbors = np.zeros((compression_m.shape[1], 4), dtype=np.int32)
    neighbor_is_trial = np.zeros_like(neighbors, dtype=bool)
    neighbor_is_boundary = np.zeros_like(neighbors, dtype=bool)
    for index, (u, v) in enumerate(cells):
        for side, (du, dv) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            cell = (u + du, v + dv)
            if cell in trial_index:
                neighbors[index, side] = trial_index[cell]
                neighbor_is_trial[index, side] = True
            elif cell not in full_cells:
                neighbor_is_boundary[index, side] = True

    neighbor_compression = np.zeros((len(compression_m), compression_m.shape[1], 4))
    for side in range(4):
        active = neighbor_is_trial[:, side]
        neighbor_compression[:, active, side] = compression_m[:, neighbors[active, side]]
        boundary = neighbor_is_boundary[:, side]
        neighbor_compression[:, boundary, side] = compression_m[:, boundary]
    return (np.sum(neighbor_compression, axis=2) - 4.0 * compression_m) / spacing_m**2


def prepare_trials(
    base: Path,
    config: dict,
    grid,
    midsole,
    trace_paths: dict[str, str | Path] | None = None,
) -> tuple[list[Trial], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Convert the rearfoot and full-foot traces to column histories.

    Args:
        base: Manifest directory used to resolve relative asset paths.
        config: Parsed manifest.
        grid: Column grid from :func:`build_column_grid`.
        midsole: Loaded midsole mesh.
        trace_paths: Optional mapping of trial name to force-displacement trace
            path. When given (for example the generated train or held-out split),
            it overrides each trial's ``averaged_cycle_path`` so the same geometry
            can be driven by a different cycle window.
    """

    trials = []
    displacement_by_name = {}
    uv_by_name = {}
    for source in config["trials"]:
        if trace_paths is not None:
            trace_path = Path(trace_paths[source["name"]])
        else:
            trace_path = base / source["averaged_cycle_path"]
        time, displacement, force = load_trace(trace_path)
        dt = np.diff(time, prepend=time[0] - (time[1] - time[0]))
        if source["fixture"] == "rearfoot_punch":
            radius = source["indenter"]["radius_m"]
            center = rearfoot_center(midsole, grid, config["grid"]["rearfoot_length_fraction"])
            active = np.linalg.norm(grid.uv_m - center, axis=1) <= radius
            slack = grid.slack_m[active]
            area = np.pi * radius**2 / np.count_nonzero(active)
            lengths = np.maximum(slack - displacement[:, None], 0.0)
        else:
            last = load_mesh(
                base / source["indenter"]["path"],
                0.001,
                source["indenter"]["rotation_deg"],
                source["indenter"]["crop_height_m"],
            )
            transform_mesh(
                last,
                source["indenter"].get("pose_rotation_deg", [0.0, 0.0, 0.0]),
                source["indenter"].get("pose_translation_m", [0.0, 0.0, 0.0]),
            )
            surface = raycast_surface(last, grid.uv_m, grid.thickness_axis, source["indenter"]["contact_side"])
            active = np.isfinite(surface)
            offset = np.percentile(grid.top_m[active] - surface[active], source["indenter"]["contact_percentile"])
            surface += offset + source["indenter"]["height_offset_m"]
            surface[active] = np.maximum(surface[active], grid.top_m[active])
            top = np.minimum(grid.top_m[active], surface[active] - displacement[:, None])
            slack = grid.slack_m[active]
            lengths = np.maximum(top - grid.bottom_m[active], 0.0)
            area = grid.area_m2
        compression = np.maximum(slack[None, :] - lengths, 0.0)
        laplacian = compression_laplacian(compression, grid.uv_m[active], grid.uv_m, grid.spacing_m)
        trials.append(Trial(source["name"], slack, area, lengths, dt, force, displacement, laplacian))
        displacement_by_name[source["name"]] = displacement
        uv_by_name[source["name"]] = grid.uv_m[active]
    return trials, displacement_by_name, uv_by_name


def write_plots(output: Path, trials: list[Trial], displacement: dict, uv: dict, grid, material: Material) -> None:
    """Plot fitted hysteresis loops and peak-compression footprints."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(trials), figsize=(12, 4), constrained_layout=True)
    for ax, trial in zip(axes, trials, strict=True):
        ax.plot(displacement[trial.name] * 1000.0, trial.force_n, color="black", label="measured")
        ax.plot(displacement[trial.name] * 1000.0, predict(trial, material), label="column model")
        ax.set(title=trial.name, xlabel="displacement [mm]", ylabel="force [N]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(output / "digital_instron_hysteresis_current.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(trials), figsize=(10, 4), constrained_layout=True)
    for ax, trial in zip(axes, trials, strict=True):
        frame = int(np.argmax(displacement[trial.name]))
        compression = np.maximum(trial.slack_m - trial.lengths_m[frame], 0.0) * 1000.0
        active = compression > 0.0
        ax.scatter(
            grid.uv_m[:, 0] * 1000.0,
            grid.uv_m[:, 1] * 1000.0,
            s=12,
            color="lightgray",
            label="midsole column",
        )
        ax.scatter(
            uv[trial.name][:, 0] * 1000.0,
            uv[trial.name][:, 1] * 1000.0,
            s=20,
            facecolors="none",
            edgecolors="black",
            linewidths=0.3,
            label="fixture overlap",
        )
        points = ax.scatter(
            uv[trial.name][active, 0] * 1000.0,
            uv[trial.name][active, 1] * 1000.0,
            c=compression[active],
            label="compressed",
        )
        ax.set(
            title=f"{trial.name}: peak kinematic compression",
            xlabel="footprint axis [mm]",
            ylabel="footprint axis [mm]",
            aspect="equal",
        )
        fig.colorbar(points, ax=ax, label="compression [mm]")
        ax.legend(fontsize="x-small")
    fig.savefig(output / "digital_instron_contact_current.png", dpi=180)
    plt.close(fig)


def write_convergence_plot(output: Path, history: list[dict[str, float]]) -> None:
    """Plot pointwise loss and fitted parameters at accepted optimizer iterations."""

    import matplotlib.pyplot as plt

    iteration = [row["iteration"] for row in history]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    loss_ax = axes[0, 0]
    loss_ax.semilogy(iteration, [row["loss"] for row in history], label="total")
    for name in ("rearfoot_140ms", "fullfoot_185ms"):
        loss_ax.semilogy(iteration, [row[f"loss_{name}"] for row in history], label=name)
    loss_ax.set(title="Peak-normalized pointwise MSE", xlabel="accepted iteration", ylabel="MSE")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)

    parameter_axes = list(axes.flat[1:])
    for ax, name in zip(parameter_axes, Material.__dataclass_fields__, strict=False):
        ax.plot(iteration, [row[name] for row in history])
        ax.set(title=name, xlabel="accepted iteration", ylabel="value")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        ax.grid(alpha=0.3)
    for ax in parameter_axes[len(Material.__dataclass_fields__) :]:
        ax.set_visible(False)
    fig.savefig(output / "digital_instron_convergence.png", dpi=180)
    plt.close(fig)


def write_geometry_plot(output: Path, base: Path, config: dict, grid) -> None:
    """Plot the cropped shoe-last and the lower surface used for contact."""

    import matplotlib.pyplot as plt

    source = next(trial for trial in config["trials"] if trial["fixture"] == "fullfoot_last")
    raw = load_mesh(base / source["indenter"]["path"], 0.001, source["indenter"]["rotation_deg"])
    cropped = load_mesh(
        base / source["indenter"]["path"],
        0.001,
        source["indenter"]["rotation_deg"],
        source["indenter"]["crop_height_m"],
    )
    transform_mesh(
        cropped,
        source["indenter"].get("pose_rotation_deg", [0.0, 0.0, 0.0]),
        source["indenter"].get("pose_translation_m", [0.0, 0.0, 0.0]),
    )
    near = raycast_surface(cropped, grid.uv_m, grid.thickness_axis, "near")
    far = raycast_surface(cropped, grid.uv_m, grid.thickness_axis, "far")
    selected_side = source["indenter"]["contact_side"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].scatter(raw.vertices[:, 0] * 1000.0, raw.vertices[:, 2] * 1000.0, s=0.2, label="raw STL")
    axes[0].scatter(cropped.vertices[:, 0] * 1000.0, cropped.vertices[:, 2] * 1000.0, s=0.2, label="retained last")
    axes[0].axhline(source["indenter"]["crop_height_m"] * 1000.0, color="red", linestyle="--", label="crop")
    axes[0].set(title="Shoe-last crop (side view)", xlabel="length [mm]", ylabel="height [mm]")
    axes[0].legend(markerscale=10)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        grid.uv_m[np.isfinite(far), 0] * 1000.0,
        far[np.isfinite(far)] * 1000.0,
        s=8,
        label=f"far{' (selected)' if selected_side == 'far' else ''}",
    )
    axes[1].scatter(
        grid.uv_m[np.isfinite(near), 0] * 1000.0,
        near[np.isfinite(near)] * 1000.0,
        s=8,
        label=f"near{' (selected)' if selected_side == 'near' else ''}",
    )
    axes[1].set(title="Cropped STL sides", xlabel="length [mm]", ylabel="surface height [mm]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(output / "digital_instron_geometry.png", dpi=180)
    plt.close(fig)


def run(manifest_path: str | Path, evaluations: int, plots: bool = False) -> dict:
    """Fit both configured trials and write the material artifact."""

    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    base = path.parent
    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
    trials, displacement, uv = prepare_trials(base, config, grid, midsole)
    initial = Material(*config["fit"].values())
    history: list[dict[str, float]] = []
    material = fit_material(trials, initial, evaluations, history)
    report = {
        "schema_version": "digital_instron_material_1",
        "model": {
            "type": "reduced_hyperfoam_maxwell_pasternak",
            "effective_poisson_ratio": EFFECTIVE_POISSON_RATIO,
            "maxwell_relaxation_time_s": MAXWELL_RELAXATION_TIME_S,
            "state_initialization": "periodic_cycle_fixed_point",
            "fit_objective": "per_trial_peak_normalized_pointwise_force_rmse",
        },
        "material": asdict(material),
        "optimization": {"accepted_iterations": len(history) - 1, "history": history},
        "trials": {
            trial.name: metrics(trial.force_n, predict(trial, material), displacement[trial.name]) for trial in trials
        },
    }
    output = base / config["cache_dir"] / "digital_instron_material.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if plots:
        write_plots(output.parent, trials, displacement, uv, grid, material)
        write_convergence_plot(output.parent, history)
        write_geometry_plot(output.parent, base, config, grid)
    return report


def main() -> None:
    """Run the fit command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json")
    parser.add_argument("--evaluations", type=int, default=100)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.evaluations, args.plots), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
