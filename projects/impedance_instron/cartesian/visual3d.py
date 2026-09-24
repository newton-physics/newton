# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read validated Visual3D ASCII exports used by the impedance preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.gait_c3d.c3d_adapter import lab_to_newton_rotation

_SCHEMA = "visual3d_export_manifest_1"
_UNIT_SCALE = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3}
_FORCE_SCALE = {"n": 1.0, "newton": 1.0, "newtons": 1.0}
_MOMENT_SCALE = {"n*m": 1.0, "nm": 1.0, "nmm": 1.0e-3}


def _clock_rate(times: np.ndarray, declared: float | None, label: str) -> float:
    """Validate a uniform clock allowing for Visual3D float32 time rounding."""
    if len(times) < 2:
        raise ValueError(f"{label} needs at least two samples")
    if declared is not None and (
        isinstance(declared, bool)
        or not isinstance(declared, (int, float))
        or not np.isfinite(declared)
        or declared <= 0
    ):
        raise ValueError(f"{label} declared rate must be finite and positive")
    inferred = (len(times) - 1) / float(times[-1] - times[0])
    if declared is not None and not np.isclose(declared, inferred, rtol=1.0e-4):
        raise ValueError(f"{label} declared rate disagrees with exported clock")
    rate = float(declared) if declared is not None else inferred
    expected = times[0] + np.arange(len(times), dtype=np.float64) / rate
    rounding = np.spacing(np.maximum(np.abs(times), 1.0).astype(np.float32)).astype(np.float64)
    tolerance = np.minimum(rounding + 1.0e-8, 0.005 / rate)
    if np.any(np.abs(times - expected) > tolerance):
        raise ValueError(f"{label} must use a uniform native sample clock")
    return rate


@dataclass(frozen=True, slots=True)
class Visual3DTrial:
    """One Visual3D export with explicit clocks and SI/Newton-frame arrays."""

    marker_time_s: np.ndarray
    marker_names: tuple[str, ...]
    marker_positions_m: np.ndarray
    marker_valid: np.ndarray
    analog_time_s: np.ndarray
    force_n: np.ndarray
    free_moment_nm: np.ndarray
    cop_m: np.ndarray
    point_rate_hz: float
    analog_rate_hz: float
    manifest: dict[str, Any]
    source_files: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(path: Path, lines: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if len(lines) < 6:
        raise ValueError(f"{path} is not a Visual3D ASCII export (expected five header lines)")
    labels, kinds, folders = (tuple(line.split()) for line in lines[1:4])
    components = tuple(lines[4].split())
    if not components or components[0] != "ITEM":
        raise ValueError(f"{path} must have an ITEM component header")
    axes = components[1:]
    if not axes or any(len(row) != len(axes) for row in (labels, kinds, folders)):
        raise ValueError(f"{path} header columns do not match data components")
    return labels, kinds, folders, axes


def _check_signal(
    path: Path, kind: str, *, folder: str | None = None, axis: str | None = None, name: str | None = None
) -> None:
    with path.open(encoding="utf-8-sig") as stream:
        lines = [next(stream, "") for _ in range(6)]
    labels, kinds, folders, axes = _header(path, lines)
    if any(value != kind for value in kinds) or (folder is not None and any(value != folder for value in folders)):
        raise ValueError(f"{path} must contain {kind}/{folder or 'declared folder'} signals")
    if axis is not None and axes != (axis,):
        raise ValueError(f"{path} must contain component {axis}")
    if name is not None and labels != (name,):
        raise ValueError(f"{path} must contain signal {name}")


def _ascii(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Parse Visual3D headers, retaining missing values and validating XYZ groups."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    raw_labels, kinds, folders, axes = _header(path, lines)
    width = len(axes)
    if width > 1:
        if width % 3 or any(
            axes[i : i + 3] != ("X", "Y", "Z")
            or len(set(raw_labels[i : i + 3])) != 1
            or len(set(kinds[i : i + 3])) != 1
            or len(set(folders[i : i + 3])) != 1
            for i in range(0, width, 3)
        ):
            raise ValueError(f"{path} must contain matching label/type/folder X/Y/Z triplets")
        labels = raw_labels[::3]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{path} contains duplicate vector labels")
    else:
        if axes[0] not in {"X", "Y", "Z", "0"}:
            raise ValueError(f"{path} has unsupported component labels")
        labels = raw_labels
    rows: list[list[float]] = []
    for line_number, line in enumerate(lines[5:], 6):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != width + 1:
            raise ValueError(f"{path}:{line_number} has {len(fields) - 1} values; expected {width}")
        expected_index = len(rows) + 1
        if fields[0] != str(expected_index):
            raise ValueError(
                f"{path}:{line_number} has noncontiguous ITEM index {fields[0]!r}; expected {expected_index}"
            )
        try:
            rows.append([float(value) for value in fields[1:]])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} contains a nonnumeric value") from error
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"{path} contains no numeric samples")
    if np.isinf(values).any():
        raise ValueError(f"{path} contains infinite values")
    return labels, values


def _manifest(root: Path, manifest_path: Path | None) -> dict[str, Any]:
    path = manifest_path or root / "visual3d_manifest.json"
    if not path.exists():
        raise ValueError(
            f"missing {path}; the importer requires declared units and laboratory axes. "
            "Write visual3d_manifest.json (see load_visual3d_export docstring)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != _SCHEMA:
        raise ValueError(f"{path} must declare schema {_SCHEMA!r}")
    required = (
        "marker_units",
        "force_units",
        "moment_units",
        "cop_units",
        "up_axis",
        "forward_axis",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    if not isinstance(data["up_axis"], str) or not isinstance(data["forward_axis"], str):
        raise ValueError(f"{path} must declare string up_axis and forward_axis")
    return data


def _time_file(root: Path, name: str, count: int, rate: float | None, units: str) -> np.ndarray:
    path = root / name
    if path.exists():
        _check_signal(path, "FRAME_NUMBERS", folder="ORIGINAL")
        _, values = _ascii(path)
        if values.shape[1] != 1 or len(values) != count:
            raise ValueError(f"{path} must contain exactly one time value for each sample")
        times = values[:, 0]
        if not np.all(np.isfinite(times)):
            raise ValueError(f"{path} contains nonfinite time values")
    else:
        raise ValueError(f"missing required clock export: {path}")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path} time values must increase strictly")
    if units in {"frame", "frames", "frame_number", "frame_numbers"}:
        raise ValueError("Export TIME/ANALOGTIME in seconds; frame-number clock units are ambiguous")
    elif units not in {"s", "sec", "second", "seconds"}:
        raise ValueError(f"unsupported time_units {units!r}")
    return times


def _single_signal(root: Path, directory: str, axis: str, count: int | None = None) -> np.ndarray:
    path = root / directory / f"motion_FP1_{axis}.txt"
    if not path.exists():
        raise ValueError(f"missing required Visual3D signal: {path}")
    _check_signal(path, directory, folder="ORIGINAL", axis=axis, name="FP1")
    _, values = _ascii(path)
    if values.shape[1] != 1 or (count is not None and len(values) != count):
        raise ValueError(f"{path} must contain one column and {count} samples")
    return values[:, 0]


def load_visual3d_export(root: str | Path, *, manifest_path: str | Path | None = None) -> Visual3DTrial:
    """Load a Visual3D ASCII directory after checking its physical metadata.

    Explicit TIME and ANALOGTIME exports retain their common source clock.
    Rates are inferred from uniform clocks; optional declared rates are checked.
    The manifest declares length/force/moment units and forward/up lab axes.
    Missing marker and unloaded COP values remain NaN, with a marker validity mask.

    Args:
        root: Directory containing original targets and native FP1 exports.
        manifest_path: Optional manifest path outside ``root``.

    Returns:
        SI measurements transformed into the Newton coordinate system.
    """
    root = Path(root).resolve()
    manifest = _manifest(root, Path(manifest_path).resolve() if manifest_path else None)
    marker_file = root / "motion_all_targets.txt"
    _check_signal(marker_file, "TARGET", folder="ORIGINAL")
    labels, marker_values = _ascii(marker_file)
    if marker_values.shape[1] != 3 * len(labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{marker_file} must contain unique marker labels with X/Y/Z triplets")
    marker_positions = marker_values.reshape(len(marker_values), len(labels), 3)
    marker_scale = _UNIT_SCALE.get(str(manifest["marker_units"]).lower())
    if marker_scale is None:
        raise ValueError(f"unsupported marker_units {manifest['marker_units']!r}")
    force_scale = _FORCE_SCALE.get(str(manifest["force_units"]).lower())
    moment_scale = _MOMENT_SCALE.get(str(manifest["moment_units"]).lower().replace(" ", ""))
    cop_scale = _UNIT_SCALE.get(str(manifest["cop_units"]).lower())
    if force_scale is None or moment_scale is None or cop_scale is None:
        raise ValueError("force_units, moment_units, and cop_units must be N, N*m, or a supported length unit")
    point_rate = manifest.get("point_rate_hz")
    analog_rate = manifest.get("analog_rate_hz")
    time_units = str(manifest.get("time_units", "s")).lower()
    _check_signal(
        root / manifest.get("marker_time_file", "motion_time.txt"), "FRAME_NUMBERS", folder="ORIGINAL", name="TIME"
    )
    marker_time = _time_file(
        root, manifest.get("marker_time_file", "motion_time.txt"), len(marker_values), point_rate, time_units
    )
    force = np.stack([_single_signal(root, "FORCE", axis) for axis in "XYZ"], axis=1) * force_scale
    moment = np.stack([_single_signal(root, "FREEMOMENT", axis, len(force)) for axis in "XYZ"], axis=1) * moment_scale
    cop = np.stack([_single_signal(root, "COFP", axis, len(force)) for axis in "XYZ"], axis=1) * cop_scale
    if not np.all(np.isfinite(force)) or not np.all(np.isfinite(moment)):
        raise ValueError("force and free-moment exports must contain only finite values")
    _check_signal(
        root / manifest.get("analog_time_file", "motion_analog_time.txt"),
        "FRAME_NUMBERS",
        folder="ORIGINAL",
        name="ANALOGTIME",
    )
    analog_time = _time_file(
        root, manifest.get("analog_time_file", "motion_analog_time.txt"), len(force), analog_rate, time_units
    )

    point_rate = _clock_rate(marker_time, point_rate, "point clock")
    analog_rate = _clock_rate(analog_time, analog_rate, "analog clock")
    rotation = lab_to_newton_rotation(manifest["up_axis"], manifest["forward_axis"])
    consumed = (
        marker_file,
        *(root / folder / f"motion_FP1_{axis}.txt" for folder in ("FORCE", "FREEMOMENT", "COFP") for axis in "XYZ"),
        root / manifest.get("marker_time_file", "motion_time.txt"),
        root / manifest.get("analog_time_file", "motion_analog_time.txt"),
    )
    sources = set()
    for path in consumed:
        with path.open(encoding="utf-8-sig") as stream:
            sources.update(value.strip() for value in stream.readline().split("\t") if value.strip())
    if len(sources) != 1:
        raise ValueError("Visual3D channels must identify the same source recording in their headers")
    marker_valid = np.all(np.isfinite(marker_positions), axis=-1)
    marker_positions = marker_positions * marker_scale @ rotation.T
    force = force @ rotation.T
    moment = moment @ rotation.T
    cop = cop @ rotation.T
    return Visual3DTrial(
        marker_time,
        labels,
        marker_positions,
        marker_valid,
        analog_time,
        force,
        moment,
        cop,
        point_rate,
        analog_rate,
        manifest,
        tuple(
            str(path.resolve())
            for path in (
                marker_file,
                *(
                    root / folder / f"motion_FP1_{axis}.txt"
                    for folder in ("FORCE", "FREEMOMENT", "COFP")
                    for axis in "XYZ"
                ),
                root / manifest.get("marker_time_file", "motion_time.txt"),
                root / manifest.get("analog_time_file", "motion_analog_time.txt"),
                Path(manifest_path) if manifest_path else root / "visual3d_manifest.json",
            )
        ),
    )


def export_manifest_template(path: str | Path) -> None:
    """Write a conservative manifest template for a Visual3D export."""
    template = {
        "schema": _SCHEMA,
        "point_rate_hz": None,
        "analog_rate_hz": None,
        "marker_units": None,
        "force_units": None,
        "moment_units": None,
        "cop_units": None,
        "marker_time_file": "motion_time.txt",
        "analog_time_file": "motion_analog_time.txt",
        "time_units": "s",
        "up_axis": None,
        "forward_axis": None,
        "source_sha256": {},
    }
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(template, indent=2) + "\n")


def inspect_visual3d_root(root: str | Path) -> dict[str, Any]:
    """Audit Visual3D trial directories without assuming timing or units."""
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Visual3D root does not exist or is not a directory: {root}")
    trials = []
    candidates = (
        [root]
        if (root / "motion_all_targets.txt").exists()
        else sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    )
    for trial_root in candidates:
        expected = [
            trial_root / "motion_all_targets.txt",
            *(
                trial_root / name
                for name in (
                    "motion_processed_targets.txt",
                    "static_all_targets.txt",
                    "static_joint_centers.txt",
                    "motion_joint_centers.txt",
                    "static_time.txt",
                )
            ),
            *(
                trial_root / folder / f"motion_FP1_{axis}.txt"
                for folder in ("FORCE", "FREEMOMENT", "COFP")
                for axis in "XYZ"
            ),
        ]
        files = [path for path in expected if path.exists()]
        counts = {}
        missing_values = {}
        for path in files:
            try:
                _, values = _ascii(path)
                counts[str(path.relative_to(trial_root))] = len(values)
                missing_values[str(path.relative_to(trial_root))] = int(np.isnan(values).sum())
            except ValueError as error:
                counts[str(path.relative_to(trial_root))] = {"error": str(error)}
        mass_path = trial_root / "model_mass.txt"
        mass = None
        if mass_path.exists():
            try:
                _, values = _ascii(mass_path)
                if values.shape != (1, 1) or not np.isfinite(values).all():
                    raise ValueError("mass must be one finite scalar")
                mass = float(values[0, 0])
            except ValueError:
                mass = {"error": "invalid model_mass.txt"}
        trials.append(
            {
                "name": trial_root.name,
                "path": str(trial_root),
                "missing_files": [str(path.relative_to(trial_root)) for path in expected if not path.exists()],
                "sample_counts": counts,
                "missing_component_counts": missing_values,
                "preparation_validated": False,
                "model_mass_export": mass,
                "manifest": str(trial_root / "visual3d_manifest.json")
                if (trial_root / "visual3d_manifest.json").exists()
                else None,
                "missing_metadata": [
                    name
                    for name, path in (
                        ("visual3d_manifest.json", trial_root / "visual3d_manifest.json"),
                        ("motion_time.txt", trial_root / "motion_time.txt"),
                        ("motion_analog_time.txt", trial_root / "motion_analog_time.txt"),
                    )
                    if not path.exists()
                ],
                "warnings": (["model_mass_export_is_one_kg_verify_measured_mass"] if mass == 1.0 else [])
                + ["Physical metadata and selected-window quality require separate preparation validation"],
                "source_sha256": {str(path.relative_to(trial_root)): _sha256(path) for path in files},
            }
        )
    if not trials:
        raise ValueError(f"No Visual3D trial directories found in {root}")
    return {"schema": "visual3d_input_audit_1", "root": str(root), "trials": trials}


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="audit exported trials without physical assumptions")
    inspect.add_argument("root", type=Path)
    inspect.add_argument("--output", type=Path)
    normalize = subparsers.add_parser("normalize", help="validate and write one normalized NPZ")
    normalize.add_argument("root", type=Path)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--manifest", type=Path)
    template = subparsers.add_parser("manifest-template", help="write a blank manifest")
    template.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        payload = inspect_visual3d_root(args.root)
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
    elif args.command == "normalize":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
        trial = load_visual3d_export(args.root, manifest_path=args.manifest)
        source_hashes = {path: _sha256(Path(path)) for path in trial.source_files}
        if args.output.suffix != ".npz":
            raise ValueError("normalized output must use the .npz extension")
        np.savez_compressed(
            args.output,
            marker_time_s=trial.marker_time_s,
            marker_positions_m=trial.marker_positions_m,
            marker_valid=trial.marker_valid,
            force_time_s=trial.analog_time_s,
            force_n=trial.force_n,
            free_moment_nm=trial.free_moment_nm,
            cop_m=trial.cop_m,
            marker_names=np.asarray(trial.marker_names),
            metadata_json=json.dumps({"manifest": trial.manifest, "source_files": source_hashes}, allow_nan=False),
        )
        print(args.output)
    else:
        export_manifest_template(args.path)


if __name__ == "__main__":
    _main()
