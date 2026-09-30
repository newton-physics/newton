# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic peak-to-peak stance dataset from Visual3D exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .prepare_visual3d import _read_triplets, prepare
from .visual3d import load_visual3d_export

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data/F01"
DEFAULT_PREPARED = ROOT / "outputs/impedance_instron/three_trial_fresh_20260924"
DEFAULT_OUTPUT = ROOT / "outputs/impedance_instron/stance_dataset_peak_hip"
TRIALS = ("FR3_1", "FR3_2")
SEED = 20260929
SELECTED_SIDE = "right"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _share_bundle_assets(root: Path, records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Hard-link identical profile and shoe assets to save repeated bundle copies."""
    assets = root / "assets"
    assets.mkdir()
    first = root / records[0]["reference"]
    shared: dict[str, dict[str, str]] = {}
    for name in ("profile.json", "digital_shoe.json"):
        source = first.parent / name
        asset = assets / name
        shutil.copyfile(source, asset)
        digest = _sha256(asset)
        for record in records:
            member = root / record["reference"]
            duplicate = member.parent / name
            if _sha256(duplicate) != digest:
                raise ValueError(f"Dataset members do not share an identical {name}")
            duplicate.unlink()
            os.link(asset, duplicate)
        shared[name] = {"file": str(asset.relative_to(root)), "sha256": digest}
    return shared


def _peak_indices(values: np.ndarray, times: np.ndarray, min_spacing_s: float = 0.20) -> list[int]:
    """Find local maxima with deterministic minimum spacing."""
    candidates = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1
    peaks: list[int] = []
    for index in candidates:
        if not peaks or times[index] - times[peaks[-1]] >= min_spacing_s:
            peaks.append(int(index))
        elif values[index] > values[peaks[-1]]:
            peaks[-1] = int(index)
    return peaks


def _marker_centroid_on_analog_clock(times: np.ndarray, marker_times: np.ndarray, markers: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(markers), axis=1)
    result = np.full((len(times), 3), np.nan)
    if valid.sum() < 2:
        return result
    for axis in range(3):
        result[:, axis] = np.interp(times, marker_times[valid], markers[valid, axis])
    result[(times < marker_times[valid][0]) | (times > marker_times[valid][-1])] = np.nan
    return result


def _trial_candidates(trial_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trial = load_visual3d_export(trial_root)
    source_force_side = str(trial.manifest.get("force_side"))
    side = SELECTED_SIDE
    prefix = "R"
    center_labels, centers, center_valid = _read_triplets(trial_root / "motion_joint_centers.txt", trial.manifest)
    hip_name = f"{prefix}HIP"
    if hip_name not in center_labels:
        raise ValueError(f"{trial_root.name}: missing {hip_name} joint center")
    hip_id = center_labels.index(hip_name)
    hip_height = centers[:, hip_id, 2]
    usable = center_valid[:, hip_id] & np.isfinite(hip_height)
    if not np.all(usable):
        raise ValueError(f"{trial_root.name}: hip-center height contains missing values")
    peaks = _peak_indices(hip_height, trial.marker_time_s)

    marker_labels, markers, _ = _read_triplets(trial_root / "motion_processed_targets.txt", trial.manifest)
    foot_centroids = []
    for foot_prefix in ("L", "R"):
        names = [f"{foot_prefix}{name}" for name in ("CAL1", "CAL2", "CAL3", "MT1H", "MT5H", "TOE")]
        missing = [name for name in names if name not in marker_labels]
        if missing:
            raise ValueError(f"{trial_root.name}: missing foot markers {missing}")
        indices = [marker_labels.index(name) for name in names]
        with np.errstate(invalid="ignore"):
            centroid = np.nanmean(markers[:, indices, :], axis=1)
        foot_centroids.append(_marker_centroid_on_analog_clock(trial.analog_time_s, trial.marker_time_s, centroid))

    distances = [np.linalg.norm(trial.cop_m - centroid, axis=1) for centroid in foot_centroids]
    assigned_index = 0 if side == "left" else 1
    opposite_index = 1 - assigned_index
    own_distance, opposite_distance = distances[assigned_index], distances[opposite_index]
    force_loaded = (trial.force_n[:, 2] > 50.0) & np.isfinite(trial.cop_m).all(axis=1)
    assigned_contact = (
        force_loaded
        & np.isfinite(own_distance)
        & np.isfinite(opposite_distance)
        & (own_distance <= 0.15)
        & (own_distance + 0.04 < opposite_distance)
    )
    opposite_contact = (
        force_loaded
        & np.isfinite(own_distance)
        & np.isfinite(opposite_distance)
        & (opposite_distance <= 0.15)
        & (opposite_distance + 0.04 < own_distance)
    )

    # Detect whole contact events first, then bracket each event by the nearest
    # preceding and following hip-height peaks. Bridge only brief dropouts.
    def event_groups(mask: np.ndarray) -> list[list[int]]:
        groups: list[list[int]] = []
        for sample in np.flatnonzero(mask):
            if not groups or sample - groups[-1][-1] > 20:
                groups.append([int(sample)])
            else:
                groups[-1].append(int(sample))
        return groups

    minimum_contact_samples = int(0.08 * trial.analog_rate_hz)
    own_events = [group for group in event_groups(assigned_contact) if len(group) >= minimum_contact_samples]
    other_events = [group for group in event_groups(opposite_contact) if len(group) >= minimum_contact_samples]
    peak_times = trial.marker_time_s[peaks]
    cycles = []
    for group in own_events:
        contact_start_s = float(trial.analog_time_s[group[0]])
        contact_end_s = float(trial.analog_time_s[group[-1]])
        before = np.flatnonzero(peak_times < contact_start_s)
        after = np.flatnonzero(peak_times > contact_end_s)
        if len(before) == 0 or len(after) == 0:
            continue
        start, end = peaks[int(before[-1])], peaks[int(after[0])]
        start_s, end_s = float(trial.marker_time_s[start]), float(trial.marker_time_s[end])
        events_inside = [
            candidate
            for candidate in own_events
            if trial.analog_time_s[candidate[-1]] >= start_s and trial.analog_time_s[candidate[0]] <= end_s
        ]
        other_inside = [
            candidate
            for candidate in other_events
            if trial.analog_time_s[candidate[-1]] >= start_s and trial.analog_time_s[candidate[0]] <= end_s
        ]
        if len(events_inside) != 1 or events_inside[0] is not group or other_inside:
            continue
        cycles.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "start_peak_height_m": float(hip_height[start]),
                "end_peak_height_m": float(hip_height[end]),
                "start_peak_index": int(start),
                "end_peak_index": int(end),
                "contact_start_s": contact_start_s,
                "contact_end_s": contact_end_s,
                "contact_samples": len(group),
            }
        )
    consumed_paths = [Path(path) for path in trial.source_files]
    for name in (
        "visual3d_manifest.json",
        "motion_joint_centers.txt",
        "motion_processed_targets.txt",
        "static_all_targets.txt",
        "static_joint_centers.txt",
        "motion_joint_angles.txt",
    ):
        path = trial_root / name
        if path.is_file():
            consumed_paths.append(path)
    return {
        "side": side,
        "source_manifest_force_side": source_force_side,
        "source_hashes": {str(path): _sha256(path) for path in sorted(set(consumed_paths))},
        "source_start_s": float(trial.marker_time_s[0]),
        "source_end_s": float(trial.marker_time_s[-1]),
    }, cycles


def _split(cycles: list[dict[str, Any]], rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(cycles) < 57:
        return [], []
    # Evaluation is a contiguous source-time block. Keep two full cycles as a
    # guard interval at each edge before drawing the 50 training windows.
    block_start = int(rng.integers(2, len(cycles) - 7))
    eval_rows = cycles[block_start : block_start + 5]
    excluded = set(range(max(0, block_start - 2), min(len(cycles), block_start + 7)))
    train_pool = [row for index, row in enumerate(cycles) if index not in excluded]
    train_indices = np.sort(rng.choice(len(train_pool), size=50, replace=False))
    train_rows = [train_pool[int(index)] for index in train_indices]
    return train_rows, eval_rows


def build_dataset(
    data_root: Path = DEFAULT_DATA,
    prepared_root: Path = DEFAULT_PREPARED,
    output_root: Path = DEFAULT_OUTPUT,
    seed: int = SEED,
) -> Path:
    """Prepare 100 training and 10 evaluation peak-to-peak stance references."""
    rng = np.random.default_rng(seed)
    available: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for trial_name in TRIALS:
        available[trial_name] = _trial_candidates(data_root / trial_name)
    counts = {name: len(candidate[1]) for name, candidate in available.items()}
    if any(count < 57 for count in counts.values()):
        raise ValueError(f"Insufficient eligible cycles for separated 50/5 splits; exact counts: {counts}")

    selections: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for trial_name, (_, cycles) in available.items():
        train, evaluation = _split(cycles, rng)
        selections[trial_name] = {"train": train, "eval": evaluation}

    # Build into a fresh directory so manifests never describe a partial set.
    destination = output_root
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    output_root = destination.with_name(f".{destination.name}.{os.getpid()}.building")
    output_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for trial_name in TRIALS:
        trial_root = data_root / trial_name
        trial_info, _ = available[trial_name]
        # Both trial bundles carry byte-identical profile and shoe artifacts;
        # use the FR3_2 right-side bundle as the shared right-foot source.
        prepared = prepared_root / "FR3_2" / "prepared"
        profile = prepared / "profile.json"
        shoe = prepared / "digital_shoe.json"
        trial_info["source_hashes"][str(profile)] = _sha256(profile)
        trial_info["source_hashes"][str(shoe)] = _sha256(shoe)
        selection_path = trial_root / "stance_selection.json"
        if not selection_path.is_file():
            selection_path = data_root / TRIALS[0] / "stance_selection.json"
        trial_info["source_hashes"][str(selection_path)] = _sha256(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        for split_name in ("train", "eval"):
            for index, cycle in enumerate(selections[trial_name][split_name]):
                sample_id = f"{trial_name}_{split_name}_{index:03d}"
                bundle = output_root / split_name / trial_name / sample_id
                prepare(
                    trial_root,
                    trial_root,
                    bundle,
                    profile,
                    shoe,
                    side=trial_info["side"],
                    start_s=cycle["start_s"],
                    end_s=cycle["end_s"],
                    subject_mass_kg=float(selection["subject_mass_kg"]),
                    belt_speed_m_s=float(selection["belt_speed_m_s"]),
                    virtual_foot_reference="reconstructed_ground",
                    shoe_static_pitch_rad=float(
                        json.loads((prepared / "summary.json").read_text(encoding="utf-8"))["angle_convention"][
                            "shoe_static_pitch_rad"
                        ]
                    ),
                    allow_force_side_override=(trial_info["source_manifest_force_side"] != SELECTED_SIDE),
                )
                records.append(
                    {
                        "id": sample_id,
                        "trial": trial_name,
                        "split": split_name,
                        **cycle,
                        "hip_center_proxy": {
                            "label": "RHIP",
                            "quantity": "Visual3D model-based hip-center vertical position",
                            "units": "m in Newton coordinates",
                            "proxy_for": "center-of-mass height",
                            "provenance": "Right hip center is an explicitly labeled proxy; this is not a measured COM trajectory.",
                        },
                        "contact_assignment": {
                            "method": "COP within 0.15 m of assigned-side CAL1/CAL2/CAL3/MT1H/MT5H/TOE centroid and at least 0.04 m closer than opposite-side centroid; upward force > 50 N",
                            "side": trial_info["side"],
                            "event_start_s": cycle["contact_start_s"],
                            "event_end_s": cycle["contact_end_s"],
                            "analog_samples": cycle["contact_samples"],
                        },
                        "reference": f"{split_name}/{trial_name}/{sample_id}/reference.npz",
                        "source_hashes": trial_info["source_hashes"],
                        "source_manifest_force_side": trial_info["source_manifest_force_side"],
                        "force_side_override": trial_info["source_manifest_force_side"] != SELECTED_SIDE,
                    }
                )
    shared_assets = _share_bundle_assets(output_root, records)
    manifest = {
        "schema": "peak_hip_stance_dataset_1",
        "selection_seed": int(seed),
        "policy": {
            "trials": list(TRIALS),
            "selected_contact_side": SELECTED_SIDE,
            "excluded_trials": ["FR3_3"],
            "windows": "Each identified right-foot contact is bracketed by the preceding and following local maxima of right Visual3D hip-center height; endpoints included.",
            "minimum_peak_spacing_s": 0.2,
            "split_per_trial": {"train": 50, "eval": 5},
            "eval_block": "Five consecutive eligible contact windows, with two eligible windows of source-time guard on either side from training.",
            "eligible_cycle_counts": counts,
        },
        "hip_center_proxy_provenance": "Right hip-center height is used as a proxy for COM height per user instruction; it is not a measured or inferred whole-body COM.",
        "shared_assets": shared_assets,
        "members": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output_root.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(build_dataset(args.data_root, args.prepared_root, args.output, args.seed))


if __name__ == "__main__":
    main()
