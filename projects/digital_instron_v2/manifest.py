# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manifest parsing for the experimental Digital Instron v2 workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Trial:
    """One physical Instron trial declared by the v2 manifest."""

    name: str
    csv_path: Path
    fixture: str
    indenter: dict[str, Any]
    include_in_fit: bool
    averaged_cycle_path: Path | None = None
    frame_config_path: Path | None = None
    frame: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrialManifest:
    """Resolved v2 Digital Instron manifest."""

    path: Path
    midsole_mesh: Path
    cache_dir: Path
    qc: dict[str, Any]
    grid: dict[str, Any]
    fit: dict[str, Any]
    trials: tuple[Trial, ...]


def _resolve_path(base: Path, value: str, field: str) -> Path:
    if not value:
        raise ValueError(f"Manifest field {field!r} must not be empty")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _validate_grid(grid: dict[str, Any]) -> None:
    """Validate grid section fields."""
    axis = grid.get("force_thickness_axis")
    if axis is not None:
        if not isinstance(axis, int) or axis not in {0, 1, 2}:
            raise ValueError(f"grid.force_thickness_axis must be 0, 1, or 2, got {axis!r}")


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be an object")
    return value


_VALID_INDENTER_TYPES = frozenset({"flat_plate", "cylinder", "stl"})


def _validate_indenter(indenter: dict[str, Any], trial_name: str, base: Path) -> dict[str, Any]:
    """Validate indenter configuration fields by type."""
    resolved = dict(indenter)
    indenter_type = indenter.get("type")
    if not isinstance(indenter_type, str) or indenter_type not in _VALID_INDENTER_TYPES:
        raise ValueError(
            f"Trial {trial_name!r} indenter type {indenter_type!r} must be one of {sorted(_VALID_INDENTER_TYPES)}"
        )

    if indenter_type == "flat_plate":
        plate_height = indenter.get("plate_height")
        if not isinstance(plate_height, (int, float)):
            raise ValueError(f"Trial {trial_name!r} flat_plate indenter must define a numeric 'plate_height' field")

    elif indenter_type == "cylinder":
        radius_m = indenter.get("radius_m")
        if not isinstance(radius_m, (int, float)):
            raise ValueError(f"Trial {trial_name!r} cylinder indenter must define a numeric 'radius_m' field")

    elif indenter_type == "stl":
        path = indenter.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Trial {trial_name!r} stl indenter must define a non-empty string 'path' field")
        resolved_path = _resolve_path(base, path, f"Trial {trial_name!r} indenter.path")
        if not resolved_path.exists():
            raise FileNotFoundError(f"Trial {trial_name!r} indenter STL does not exist: {resolved_path}")
        resolved["path"] = str(resolved_path)
        height_offset_m = indenter.get("height_offset_m")
        if height_offset_m is not None and not isinstance(height_offset_m, (int, float)):
            raise ValueError(f"Trial {trial_name!r} stl indenter height_offset_m must be numeric")
        if height_offset_m is not None:
            resolved["height_offset_m"] = height_offset_m
        contact_percentile = indenter.get("contact_percentile")
        if contact_percentile is not None:
            if not isinstance(contact_percentile, (int, float)):
                raise ValueError(f"Trial {trial_name!r} stl indenter contact_percentile must be numeric")
            if not 0.0 < float(contact_percentile) <= 100.0:
                raise ValueError(f"Trial {trial_name!r} stl indenter contact_percentile must be in (0, 100]")
            resolved["contact_percentile"] = contact_percentile

    return resolved


def _require_trial(data: dict[str, Any], index: int, base: Path) -> Trial:
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Trial {index} must define a non-empty name")

    fixture = data.get("fixture")
    if fixture not in {"rearfoot_punch", "fullfoot_last", "localized_punch"}:
        raise ValueError(f"Trial {name!r} has unsupported fixture {fixture!r}")

    indenter = data.get("indenter")
    if not isinstance(indenter, dict):
        raise ValueError(f"Trial {name!r} must define an indenter object")
    indenter = _validate_indenter(indenter, name, base)

    averaged_cycle = data.get("averaged_cycle_path")
    averaged_cycle_path = (
        _resolve_path(base, averaged_cycle, f"trials[{index}].averaged_cycle_path")
        if isinstance(averaged_cycle, str)
        else None
    )
    if averaged_cycle_path is not None and not averaged_cycle_path.exists():
        raise FileNotFoundError(f"Trial {name!r} averaged cycle CSV does not exist: {averaged_cycle_path}")

    frame_config = data.get("frame_config_path")
    return Trial(
        name=name,
        csv_path=_resolve_path(base, str(data.get("csv_path", "")), f"trials[{index}].csv_path"),
        fixture=fixture,
        indenter=indenter,
        include_in_fit=bool(data.get("include_in_fit", True)),
        averaged_cycle_path=averaged_cycle_path,
        frame_config_path=_resolve_path(base, frame_config, f"trials[{index}].frame_config_path")
        if isinstance(frame_config, str)
        else None,
        frame=data.get("frame") if isinstance(data.get("frame"), dict) else None,
    )


def load_manifest(path: str | Path) -> TrialManifest:
    """Load and validate a v2 trial manifest.

    Paths inside the manifest are resolved relative to the manifest file.
    Missing CSV and mesh paths fail immediately because the workflow is meant
    to expose stale data references before fitting starts.
    """

    manifest_path = Path(path).expanduser().resolve()
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Digital Instron v2 manifest must be a JSON object")

    base = manifest_path.parent
    midsole_mesh = _resolve_path(base, str(data.get("midsole_mesh", "")), "midsole_mesh")
    if not midsole_mesh.exists():
        raise FileNotFoundError(f"Midsole mesh does not exist: {midsole_mesh}")

    raw_trials = data.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ValueError("Manifest must define at least one trial")
    trials = tuple(_require_trial(trial, i, base) for i, trial in enumerate(raw_trials))
    for trial in trials:
        if not trial.csv_path.exists():
            raise FileNotFoundError(f"Trial {trial.name!r} CSV does not exist: {trial.csv_path}")

    fit = _require_mapping(data, "fit")
    fit_trial_names = {trial.name for trial in trials if trial.include_in_fit}
    if not fit_trial_names:
        raise ValueError("Manifest must include at least one trial in the fit")

    grid = _require_mapping(data, "grid")
    _validate_grid(grid)

    return TrialManifest(
        path=manifest_path,
        midsole_mesh=midsole_mesh,
        cache_dir=_resolve_path(base, str(data.get("cache_dir", "processed/v2_cache")), "cache_dir"),
        qc=_require_mapping(data, "qc"),
        grid=grid,
        fit=fit,
        trials=trials,
    )


def validate_frame_config(config: dict[str, Any]) -> None:
    """Validate a saved frame configuration object."""

    required = ("time_column", "position_column", "force_column", "position_sign", "force_sign")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Frame config is missing required keys: {', '.join(missing)}")

    for key in ("time_column", "position_column", "force_column"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"Frame config {key!r} must be a non-empty string")

    for key in ("position_sign", "force_sign"):
        if float(config[key]) not in {-1.0, 1.0}:
            raise ValueError(f"Frame config {key!r} must be -1 or 1")
