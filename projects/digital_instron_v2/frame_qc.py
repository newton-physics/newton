# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame selection and QC for physical Digital Instron CSV traces."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import validate_frame_config


class FrameQCError(ValueError):
    """Raised when a trace cannot support a trustworthy frame choice."""


@dataclass(frozen=True)
class FrameConfig:
    """Column and sign choices for one Instron trace."""

    time_column: str
    position_column: str
    force_column: str
    position_sign: float
    force_sign: float
    displacement_zero: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "position_column": self.position_column,
            "force_column": self.force_column,
            "position_sign": self.position_sign,
            "force_sign": self.force_sign,
            "displacement_zero": self.displacement_zero,
        }


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise FrameQCError(f"CSV has no header: {path}")
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(np.nan)

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    if not arrays or min(len(values) for values in arrays.values()) == 0:
        raise FrameQCError(f"CSV has no numeric rows: {path}")
    return arrays


def _find_column(columns: dict[str, np.ndarray], candidates: tuple[str, ...], label: str) -> str:
    lowered = {name.lower(): name for name in columns}
    for candidate in candidates:
        for lowered_name, original in lowered.items():
            if candidate in lowered_name:
                return original
    raise FrameQCError(f"Could not infer {label} column from CSV columns: {', '.join(columns)}")


def infer_frame_config(path: str | Path, *, min_force_span_n: float = 50.0, min_position_span_mm: float = 1.0) -> FrameConfig:
    """Infer column/sign choices and fail on implausible traces."""

    csv_path = Path(path)
    columns = _read_numeric_csv(csv_path)
    time_col = _find_column(columns, ("total time", "elapsed time", "time"), "time")
    pos_col = _find_column(columns, ("position", "displacement"), "position")
    force_col = _find_column(columns, ("force",), "force")

    position = columns[pos_col]
    force = columns[force_col]
    finite = np.isfinite(position) & np.isfinite(force) & np.isfinite(columns[time_col])
    if int(np.count_nonzero(finite)) < 3:
        raise FrameQCError(f"Trace {csv_path} has fewer than three finite position/force samples")

    position = position[finite]
    force = force[finite]
    pos_span = float(np.nanmax(position) - np.nanmin(position))
    force_span = float(np.nanmax(force) - np.nanmin(force))
    if pos_span < min_position_span_mm:
        raise FrameQCError(f"Trace {csv_path} position span {pos_span:.3g} mm is too small")
    if force_span < min_force_span_n:
        raise FrameQCError(f"Trace {csv_path} force span {force_span:.3g} N is too small")

    force_sign = -1.0 if abs(float(np.nanmin(force))) > abs(float(np.nanmax(force))) else 1.0
    compressive_force = force_sign * force
    if len(position) > 1 and float(np.nanstd(position)) > 0.0 and float(np.nanstd(compressive_force)) > 0.0:
        corr = float(np.corrcoef(position, compressive_force)[0, 1])
        position_sign = 1.0 if corr >= 0.0 else -1.0
    else:
        position_sign = 1.0
    return FrameConfig(
        time_column=time_col,
        position_column=pos_col,
        force_column=force_col,
        position_sign=position_sign,
        force_sign=force_sign,
        displacement_zero=float(position[0]),
    )


def load_trial_frame(path: str | Path, config: FrameConfig | dict[str, Any]) -> dict[str, np.ndarray]:
    """Load one trace using an explicit saved frame configuration."""

    if isinstance(config, dict):
        validate_frame_config(config)
        frame = FrameConfig(
            time_column=config["time_column"],
            position_column=config["position_column"],
            force_column=config["force_column"],
            position_sign=float(config["position_sign"]),
            force_sign=float(config["force_sign"]),
            displacement_zero=float(config.get("displacement_zero", 0.0)),
        )
    else:
        frame = config

    columns = _read_numeric_csv(Path(path))
    missing = [name for name in (frame.time_column, frame.position_column, frame.force_column) if name not in columns]
    if missing:
        raise FrameQCError(f"CSV is missing saved frame columns: {', '.join(missing)}")

    time_s = columns[frame.time_column]
    displacement_m = frame.position_sign * (columns[frame.position_column] - frame.displacement_zero) * 0.001
    force_n = frame.force_sign * columns[frame.force_column]
    finite = np.isfinite(time_s) & np.isfinite(displacement_m) & np.isfinite(force_n)
    if int(np.count_nonzero(finite)) < 3:
        raise FrameQCError("Saved frame config produced fewer than three finite samples")
    return {
        "time_s": time_s[finite],
        "displacement_m": displacement_m[finite],
        "force_n": force_n[finite],
    }
