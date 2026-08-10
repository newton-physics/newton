# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame selection and QC for physical Digital Instron CSV traces.

A raw Instron export names its own columns and picks its own sign convention for
position and force. :func:`infer_frame_config` chooses the time/position/force
columns, resolves the compressive sign of each, and fails loudly on traces too
flat to trust, so the cycle-window averaging in
:mod:`~projects.digital_instron_v2.cycle_windows` has one audited source of truth
for how a raw CSV maps onto a physical loading trace.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
        """Return the frame configuration as a plain JSON-serializable mapping."""
        return {
            "time_column": self.time_column,
            "position_column": self.position_column,
            "force_column": self.force_column,
            "position_sign": self.position_sign,
            "force_sign": self.force_sign,
            "displacement_zero": self.displacement_zero,
        }


def validate_frame_config(config: dict[str, Any]) -> None:
    """Validate a saved frame-configuration object.

    Args:
        config: Mapping with column names and sign conventions.

    Raises:
        ValueError: If a required key is missing, a column name is not a
            non-empty string, or a sign is not exactly ``-1`` or ``1``.
    """
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


def read_numeric_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a numeric CSV into per-column float arrays, mapping blanks to NaN."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise FrameQCError(f"CSV has no header: {csv_path}")
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    columns[name].append(np.nan)

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    if not arrays or min(len(values) for values in arrays.values()) == 0:
        raise FrameQCError(f"CSV has no numeric rows: {csv_path}")
    return arrays


def _find_column(columns: dict[str, np.ndarray], candidates: tuple[str, ...], label: str) -> str:
    lowered = {name.lower(): name for name in columns}
    for candidate in candidates:
        for lowered_name, original in lowered.items():
            if candidate in lowered_name:
                return original
    raise FrameQCError(f"Could not infer {label} column from CSV columns: {', '.join(columns)}")


def infer_frame_config(
    path: str | Path, *, min_force_span_n: float = 50.0, min_position_span_mm: float = 1.0
) -> FrameConfig:
    """Infer column and sign choices for a raw trace and fail on implausible data.

    Args:
        path: Raw Instron CSV path.
        min_force_span_n: Minimum peak-to-peak force span for a usable trace [N].
        min_position_span_mm: Minimum peak-to-peak position span [mm].

    Returns:
        The inferred :class:`FrameConfig`. ``displacement_zero`` is the first
        finite position sample; the cycle-window generator applies its own
        displacement-zero policy on top of this raw reference.

    Raises:
        FrameQCError: If a required column cannot be found or the trace is too
            flat in force or position to be trusted.
    """
    csv_path = Path(path)
    columns = read_numeric_csv(csv_path)
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
