# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark replicate modes with bundled quadruped URDF and one shared global terrain."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import warp as wp

import newton
import newton.examples
from newton.geometry import create_mesh_terrain

ReplicateMode = Literal["auto", "fast", "legacy"]


@dataclass(frozen=True)
class RunRecord:
    num_worlds: int
    mode: ReplicateMode
    run_index: int
    robot_build_seconds: float
    replicate_seconds: float
    terrain_build_once_seconds: float
    finalize_seconds: float
    replicate_path_only_seconds: float
    total_startup_seconds: float


def _synchronize() -> None:
    wp.synchronize_device()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _p95(values: list[float]) -> float:
    # "inclusive" avoids over/undershooting for small samples such as 5-run sweeps.
    return float(statistics.quantiles(values, n=100, method="inclusive")[94])


def _build_quadruped_template() -> newton.ModelBuilder:
    quadruped = newton.ModelBuilder()

    quadruped.default_body_armature = 0.01
    quadruped.default_joint_cfg.armature = 0.01
    quadruped.default_joint_cfg.target_ke = 2000.0
    quadruped.default_joint_cfg.target_kd = 1.0
    quadruped.default_shape_cfg.ke = 1.0e4
    quadruped.default_shape_cfg.kd = 1.0e2
    quadruped.default_shape_cfg.kf = 1.0e2
    quadruped.default_shape_cfg.mu = 1.0

    quadruped.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
    )

    quadruped.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
    quadruped.joint_target_pos[-12:] = quadruped.joint_q[-12:]

    return quadruped


def _add_global_procedural_terrain(scene: newton.ModelBuilder, seed: int) -> None:
    vertices, indices = create_mesh_terrain(
        grid_size=(8, 3),
        block_size=(3.0, 3.0),
        terrain_types=["random_grid", "flat", "wave", "gap", "pyramid_stairs"],
        terrain_params={
            "pyramid_stairs": {"step_width": 0.3, "step_height": 0.02, "platform_width": 0.6},
            "random_grid": {"grid_width": 0.3, "grid_height_range": (0.0, 0.02)},
            "wave": {"wave_amplitude": 0.1, "wave_frequency": 2.0},
        },
        seed=seed,
    )
    terrain_mesh = newton.Mesh(vertices, indices)
    terrain_offset = wp.transform(p=wp.vec3(-5.0, -2.0, 0.01), q=wp.quat_identity())
    scene.add_shape_mesh(body=-1, mesh=terrain_mesh, xform=terrain_offset)


def _run_single(
    *,
    num_worlds: int,
    mode: ReplicateMode,
    spacing: tuple[float, float, float],
    terrain_seed: int,
    run_index: int,
) -> RunRecord:
    total_start = time.perf_counter()

    robot_build_start = time.perf_counter()
    quadruped = _build_quadruped_template()
    _synchronize()
    robot_build_seconds = time.perf_counter() - robot_build_start

    scene = newton.ModelBuilder()

    replicate_start = time.perf_counter()
    scene.replicate(quadruped, num_worlds=num_worlds, spacing=spacing, mode=mode)
    _synchronize()
    replicate_seconds = time.perf_counter() - replicate_start

    terrain_start = time.perf_counter()
    _add_global_procedural_terrain(scene, seed=terrain_seed)
    _synchronize()
    terrain_build_once_seconds = time.perf_counter() - terrain_start

    finalize_start = time.perf_counter()
    scene.finalize()
    _synchronize()
    finalize_seconds = time.perf_counter() - finalize_start

    total_startup_seconds = time.perf_counter() - total_start
    replicate_path_only_seconds = robot_build_seconds + replicate_seconds + finalize_seconds

    return RunRecord(
        num_worlds=num_worlds,
        mode=mode,
        run_index=run_index,
        robot_build_seconds=robot_build_seconds,
        replicate_seconds=replicate_seconds,
        terrain_build_once_seconds=terrain_build_once_seconds,
        finalize_seconds=finalize_seconds,
        replicate_path_only_seconds=replicate_path_only_seconds,
        total_startup_seconds=total_startup_seconds,
    )


def _resolve_modes(mode_arg: str) -> list[ReplicateMode]:
    if mode_arg == "all":
        return ["legacy", "auto", "fast"]
    return [mode_arg]  # type: ignore[return-value]


def _aggregate(records: list[RunRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str], list[RunRecord]] = {}
    for record in records:
        key = (record.num_worlds, record.mode)
        grouped.setdefault(key, []).append(record)

    summaries: list[dict[str, float | int | str]] = []
    for (num_worlds, mode), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        robot_build = [r.robot_build_seconds for r in rows]
        replicate = [r.replicate_seconds for r in rows]
        terrain = [r.terrain_build_once_seconds for r in rows]
        finalize = [r.finalize_seconds for r in rows]
        replicate_path = [r.replicate_path_only_seconds for r in rows]
        total = [r.total_startup_seconds for r in rows]

        summaries.append(
            {
                "num_worlds": num_worlds,
                "mode": mode,
                "runs": len(rows),
                "robot_build_median_seconds": _median(robot_build),
                "robot_build_p95_seconds": _p95(robot_build),
                "replicate_median_seconds": _median(replicate),
                "replicate_p95_seconds": _p95(replicate),
                "terrain_build_once_median_seconds": _median(terrain),
                "terrain_build_once_p95_seconds": _p95(terrain),
                "finalize_median_seconds": _median(finalize),
                "finalize_p95_seconds": _p95(finalize),
                "replicate_path_only_median_seconds": _median(replicate_path),
                "replicate_path_only_p95_seconds": _p95(replicate_path),
                "total_startup_median_seconds": _median(total),
                "total_startup_p95_seconds": _p95(total),
            }
        )

    return summaries


def _add_legacy_effects(summary_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    by_world: dict[int, dict[str, dict[str, float | int | str]]] = {}
    for row in summary_rows:
        num_worlds = int(row["num_worlds"])
        mode = str(row["mode"])
        by_world.setdefault(num_worlds, {})[mode] = row

    metric_keys = [
        "robot_build_median_seconds",
        "replicate_median_seconds",
        "terrain_build_once_median_seconds",
        "finalize_median_seconds",
        "replicate_path_only_median_seconds",
        "total_startup_median_seconds",
    ]
    phase_keys = [
        "robot_build_median_seconds",
        "replicate_median_seconds",
        "terrain_build_once_median_seconds",
        "finalize_median_seconds",
    ]

    enriched_rows: list[dict[str, float | int | str]] = []
    for row in summary_rows:
        num_worlds = int(row["num_worlds"])
        mode = str(row["mode"])
        if "legacy" not in by_world[num_worlds]:
            raise ValueError(f"missing legacy row for num_worlds={num_worlds}")
        legacy = by_world[num_worlds]["legacy"]

        enriched = dict(row)
        total_delta = float(row["total_startup_median_seconds"]) - float(legacy["total_startup_median_seconds"])

        for key in metric_keys:
            base = key.replace("_median_seconds", "")
            current_value = float(row[key])
            legacy_value = float(legacy[key])
            if legacy_value == 0.0:
                raise ValueError(f"legacy baseline for {key} is zero at num_worlds={num_worlds}")
            delta_seconds = current_value - legacy_value
            delta_percent = (delta_seconds / legacy_value) * 100.0
            enriched[f"{base}_delta_vs_legacy_seconds"] = delta_seconds
            enriched[f"{base}_delta_vs_legacy_percent"] = delta_percent

        if float(row["replicate_path_only_median_seconds"]) == 0.0:
            raise ValueError(f"replicate_path_only_median_seconds is zero at num_worlds={num_worlds}, mode={mode}")
        if float(row["total_startup_median_seconds"]) == 0.0:
            raise ValueError(f"total_startup_median_seconds is zero at num_worlds={num_worlds}, mode={mode}")
        enriched["replicate_path_only_speedup_vs_legacy_x"] = float(
            legacy["replicate_path_only_median_seconds"]
        ) / float(row["replicate_path_only_median_seconds"])
        enriched["total_startup_speedup_vs_legacy_x"] = float(legacy["total_startup_median_seconds"]) / float(
            row["total_startup_median_seconds"]
        )

        for key in phase_keys:
            base = key.replace("_median_seconds", "")
            if mode == "legacy":
                contribution_percent = 0.0
            else:
                if total_delta == 0.0:
                    raise ValueError(
                        f"total_startup delta is zero for non-legacy row at num_worlds={num_worlds}, mode={mode}"
                    )
                phase_delta = float(row[key]) - float(legacy[key])
                contribution_percent = (phase_delta / total_delta) * 100.0
            enriched[f"{base}_contribution_to_total_delta_percent"] = contribution_percent

        enriched_rows.append(enriched)

    return enriched_rows


def _write_outputs(
    out_dir: Path,
    records: list[RunRecord],
    summary_rows: list[dict[str, float | int | str]],
    summary_effect_rows: list[dict[str, float | int | str]],
) -> None:
    out_dir.mkdir(parents=True)

    run_records_path = out_dir / "run_records.json"
    run_records_path.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")

    summary_json_path = out_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    summary_csv_path = out_dir / "summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_effect_json_path = out_dir / "summary_effects_vs_legacy.json"
    summary_effect_json_path.write_text(json.dumps(summary_effect_rows, indent=2), encoding="utf-8")

    summary_effect_csv_path = out_dir / "summary_effects_vs_legacy.csv"
    effect_fieldnames = list(summary_effect_rows[0].keys())
    with summary_effect_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=effect_fieldnames)
        writer.writeheader()
        writer.writerows(summary_effect_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark quadruped replicate modes with one shared global procedural terrain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-worlds", type=int, nargs="+", required=True, help="World counts to benchmark.")
    parser.add_argument(
        "--mode",
        choices=["all", "auto", "fast", "legacy"],
        default="all",
        help="Replicate mode(s) to benchmark.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        required=True,
        metavar=("SX", "SY", "SZ"),
        help="World spacing passed to replicate().",
    )
    parser.add_argument("--runs", type=int, default=5, help="Repeats per mode/world combination.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Warp device.")
    parser.add_argument("--terrain-seed", type=int, default=42, help="Seed for procedural terrain generation.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSON/CSV artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 2:
        raise ValueError(f"--runs must be >= 2 (p95 requires at least two samples), got {args.runs}")
    if any(num_worlds <= 0 for num_worlds in args.num_worlds):
        raise ValueError(f"all --num-worlds values must be > 0, got {args.num_worlds}")
    if args.out_dir.exists():
        raise FileExistsError(f"--out-dir already exists: {args.out_dir}")

    wp.set_device(args.device)
    _synchronize()

    spacing = (float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2]))
    modes = _resolve_modes(args.mode)

    records: list[RunRecord] = []
    for num_worlds in args.num_worlds:
        for mode in modes:
            for run_index in range(args.runs):
                record = _run_single(
                    num_worlds=num_worlds,
                    mode=mode,
                    spacing=spacing,
                    terrain_seed=args.terrain_seed,
                    run_index=run_index,
                )
                records.append(record)
                print(
                    f"num_worlds={record.num_worlds:5d} mode={record.mode:7s} "
                    f"run={record.run_index + 1}/{args.runs} total={record.total_startup_seconds:.4f}s"
                )

    summary_rows = _aggregate(records)
    summary_effect_rows = _add_legacy_effects(summary_rows)
    _write_outputs(args.out_dir, records, summary_rows, summary_effect_rows)
    summary_effect_map = {(int(row["num_worlds"]), str(row["mode"])): row for row in summary_effect_rows}

    print("\nSummary (median seconds)")
    for row in summary_rows:
        num_worlds = int(row["num_worlds"])
        mode = str(row["mode"])
        effects = summary_effect_map[(num_worlds, mode)]
        print(
            f"num_worlds={num_worlds:5d} mode={mode:7s} "
            f"replicate_path_only={row['replicate_path_only_median_seconds']:.4f}s "
            f"total_startup={row['total_startup_median_seconds']:.4f}s "
            f"delta_vs_legacy={effects['total_startup_delta_vs_legacy_percent']:+.1f}% "
            f"speedup={effects['total_startup_speedup_vs_legacy_x']:.3f}x"
        )

    print(f"\nArtifacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
