# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manifest parsing for the experimental Digital Instron v2 workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
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


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be an object")
    return value


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

    frame_config = data.get("frame_config_path")
    return Trial(
        name=name,
        csv_path=_resolve_path(base, str(data.get("csv_path", "")), f"trials[{index}].csv_path"),
        fixture=fixture,
        indenter=indenter,
        include_in_fit=bool(data.get("include_in_fit", True)),
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

    return TrialManifest(
        path=manifest_path,
        midsole_mesh=midsole_mesh,
        cache_dir=_resolve_path(base, str(data.get("cache_dir", "processed/v2_cache")), "cache_dir"),
        qc=_require_mapping(data, "qc"),
        grid=_require_mapping(data, "grid"),
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
