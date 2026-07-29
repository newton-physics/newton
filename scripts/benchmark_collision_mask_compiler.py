# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark collision-mask lowering on a MuJoCo Menagerie checkout."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import numpy as np

from newton._src.solvers.mujoco.collision_masks import (
    compile_collision_masks,
    compile_newton_collision_graph,
    verify_collision_mask_compilation,
    verify_newton_collision_graph_compilation,
)


def _structural_filter_pairs(model) -> np.ndarray:
    """Return geom pairs rejected before MuJoCo's mask predicate."""
    import mujoco

    exclude_signatures = set(map(int, model.exclude_signature))
    parent_filter_disabled = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    pairs = []
    for geom_a in range(model.ngeom - 1):
        body_a = int(model.geom_bodyid[geom_a])
        weld_a = int(model.body_weldid[body_a])
        parent_weld_a = int(model.body_weldid[int(model.body_parentid[weld_a])])
        for geom_b in range(geom_a + 1, model.ngeom):
            body_b = int(model.geom_bodyid[geom_b])
            weld_b = int(model.body_weldid[body_b])
            if weld_a == weld_b:
                pairs.append((geom_a, geom_b))
                continue
            parent_weld_b = int(model.body_weldid[int(model.body_parentid[weld_b])])
            if (
                not parent_filter_disabled
                and weld_a != 0
                and weld_b != 0
                and (weld_a == parent_weld_b or weld_b == parent_weld_a)
            ):
                pairs.append((geom_a, geom_b))
                continue
            signature = (min(body_a, body_b) << 16) + max(body_a, body_b)
            if signature in exclude_signatures:
                pairs.append((geom_a, geom_b))
    return np.asarray(pairs, dtype=np.int32).reshape((-1, 2))


def _mask_blocked_pair_count(
    collision_type: np.ndarray,
    collision_affinity: np.ndarray,
    prefiltered_pairs: np.ndarray,
) -> int:
    """Count mask-incompatible pairs not already structurally filtered."""
    shape_count = collision_type.shape[0]
    prefiltered = set(map(tuple, prefiltered_pairs.tolist()))
    count = 0
    for shape_a in range(shape_count - 1):
        if collision_type[shape_a] == 0 and collision_affinity[shape_a] == 0:
            continue
        for shape_b in range(shape_a + 1, shape_count):
            if (shape_a, shape_b) in prefiltered:
                continue
            if collision_type[shape_b] == 0 and collision_affinity[shape_b] == 0:
                continue
            allowed = bool(
                (collision_type[shape_a] & collision_affinity[shape_b])
                or (collision_type[shape_b] & collision_affinity[shape_a])
            )
            count += not allowed
    return count


def _compile_timed(
    collision_type: np.ndarray,
    collision_affinity: np.ndarray,
    prefiltered_pairs: np.ndarray,
    *,
    max_exact_classes: int,
    repeats: int,
):
    """Compile repeatedly and return the result with median runtime."""
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = compile_collision_masks(
            collision_type,
            collision_affinity,
            prefiltered_pairs=prefiltered_pairs,
            force_nonzero=True,
            max_exact_classes=max_exact_classes,
            max_positive_groups=1,
        )
        timings.append(time.perf_counter_ns() - start)
    assert result is not None
    verify_collision_mask_compilation(
        collision_type,
        collision_affinity,
        result,
        prefiltered_pairs=prefiltered_pairs,
    )
    return result, statistics.median(timings) / 1_000.0


def _compile_reverse_timed(groups: np.ndarray, excluded_pairs: np.ndarray, *, repeats: int):
    """Compile Newton filtering repeatedly and return median runtime."""
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = compile_newton_collision_graph(groups, excluded_pairs=excluded_pairs)
        timings.append(time.perf_counter_ns() - start)
    assert result is not None
    if result.exact:
        verify_newton_collision_graph_compilation(
            groups,
            result,
            excluded_pairs=excluded_pairs,
        )
    return result, statistics.median(timings) / 1_000.0


def _load_model(xml_path: Path):
    """Compile one XML and return its mask metadata."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    masks = collections.Counter(
        (int(model.geom_contype[index]), int(model.geom_conaffinity[index])) for index in range(model.ngeom)
    )
    noncanonical = any(mask not in {(0, 0), (1, 1)} for mask in masks)
    return (
        noncanonical,
        len(masks),
        model.ngeom,
        model.npair,
        xml_path,
        model,
        masks,
    )


def _models(root: Path, *, all_models: bool, stats: dict):
    """Yield every compilable XML or one representative per asset directory."""
    xml_paths = sorted(root.rglob("*.xml"))
    stats["xml_file_count"] = len(xml_paths)
    stats["compile_failure_count"] = 0
    stats["compilable_xml_count"] = 0

    if all_models:
        for xml_path in xml_paths:
            try:
                record = _load_model(xml_path)
            except Exception:
                stats["compile_failure_count"] += 1
                continue
            stats["compilable_xml_count"] += 1
            yield record
        return

    for asset_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        candidates = []
        for xml_path in asset_dir.rglob("*.xml"):
            try:
                record = _load_model(xml_path)
            except Exception:
                stats["compile_failure_count"] += 1
                continue
            stats["compilable_xml_count"] += 1
            candidates.append(record)
        if candidates:
            yield max(candidates, key=lambda item: item[:4])


def main() -> None:
    """Run the Menagerie collision-mask compiler benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument("menagerie", type=Path, help="Path to a mujoco_menagerie checkout")
    parser.add_argument("--repeats", type=int, default=7, help="Compiler timing repetitions per asset")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Verify every compilable XML instead of one representative per asset directory",
    )
    args = parser.parse_args()
    root = args.menagerie.resolve()
    if not root.is_dir():
        parser.error(f"Menagerie directory does not exist: {root}")

    rows = []
    corpus_stats = {}
    for noncanonical, _mask_count, _ngeom, _npair, xml_path, model, masks in _models(
        root,
        all_models=args.all_models,
        stats=corpus_stats,
    ):
        collision_type = np.asarray(model.geom_contype, dtype=np.uint32)
        collision_affinity = np.asarray(model.geom_conaffinity, dtype=np.uint32)
        structural_pairs = _structural_filter_pairs(model)
        baseline_exclusions = _mask_blocked_pair_count(
            collision_type,
            collision_affinity,
            structural_pairs,
        )
        greedy, greedy_us = _compile_timed(
            collision_type,
            collision_affinity,
            structural_pairs,
            max_exact_classes=1,
            repeats=args.repeats,
        )
        optimized, optimized_us = _compile_timed(
            collision_type,
            collision_affinity,
            structural_pairs,
            max_exact_classes=8,
            repeats=args.repeats,
        )
        reverse, reverse_us = _compile_reverse_timed(
            optimized.groups,
            optimized.excluded_pairs,
            repeats=args.repeats,
        )
        row = {
            "asset": xml_path.relative_to(root).as_posix(),
            "ngeom": model.ngeom,
            "active_geoms": int(np.count_nonzero(collision_type | collision_affinity)),
            "mask_classes": len(masks),
            "noncanonical": noncanonical,
            "explicit_pairs": model.npair,
            "structural_pairs": structural_pairs.shape[0],
            "group_1_exclusions": baseline_exclusions,
            "greedy_groups": greedy.group_count,
            "greedy_exclusions": greedy.excluded_pairs.shape[0],
            "greedy_us": greedy_us,
            "optimized_groups": optimized.group_count,
            "optimized_exclusions": optimized.excluded_pairs.shape[0],
            "optimized_us": optimized_us,
            "search_nodes": optimized.search_nodes,
            "optimal": optimized.optimal,
            "reverse_exact": reverse.exact,
            "reverse_bits": reverse.bit_count,
            "reverse_uncovered_pairs": reverse.uncovered_pair_count,
            "reverse_us": reverse_us,
            "masks": {f"{mask[0]}/{mask[1]}": count for mask, count in sorted(masks.items())},
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    if not rows:
        raise RuntimeError(f"No compilable MJCF models found under {root}")
    noncanonical_rows = [row for row in rows if row["noncanonical"]]
    baseline_total = sum(row["group_1_exclusions"] for row in rows)
    greedy_total = sum(row["greedy_exclusions"] for row in rows)
    optimized_total = sum(row["optimized_exclusions"] for row in rows)
    reverse_exact_count = sum(row["reverse_exact"] for row in rows)
    summary = {
        **corpus_stats,
        "mode": "all_models" if args.all_models else "representatives",
        "asset_count": len(rows),
        "noncanonical_assets": len(noncanonical_rows),
        "group_1_exclusions": baseline_total,
        "greedy_exclusions": greedy_total,
        "optimized_exclusions": optimized_total,
        "greedy_reduction_percent": 100.0 * (baseline_total - greedy_total) / max(1, baseline_total),
        "optimized_reduction_percent": 100.0 * (baseline_total - optimized_total) / max(1, baseline_total),
        "median_greedy_us": statistics.median(row["greedy_us"] for row in rows),
        "median_optimized_us": statistics.median(row["optimized_us"] for row in rows),
        "max_optimized_us": max(row["optimized_us"] for row in rows),
        "reverse_exact_models": reverse_exact_count,
        "reverse_capacity_failures": len(rows) - reverse_exact_count,
        "median_reverse_us": statistics.median(row["reverse_us"] for row in rows),
        "max_reverse_us": max(row["reverse_us"] for row in rows),
        "max_reverse_bits": max((row["reverse_bits"] for row in rows if row["reverse_exact"]), default=0),
        "all_forward_compilations_verified": True,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
