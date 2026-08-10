# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cycle-window averaging for raw Digital Instron traces.

The raw Instron CSV is the source of truth for train/validation splits. This
module builds a phase-gridded averaged trace over an explicit window of cycles
(for example train cycles ``90-98`` versus held-out cycles ``99-100``) and
records enough provenance -- source CSV, cycle column, cycle window, phase-grid
size, and displacement-zero policy -- to make each generated split unambiguous.

The emitted schema matches the shipped averaged-cycle files so the existing
fitting path can consume it directly::

    (
        phase,
        time_s,
        displacement_m,
        displacement_mm,
        force_n,
        position_mm_raw,
    )
    force_n_raw, velocity_m_s, cycle_energy_j
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .frame_qc import FrameConfig, read_numeric_csv, validate_frame_config

TRACE_COLUMNS = (
    "phase",
    "time_s",
    "displacement_m",
    "displacement_mm",
    "force_n",
    "position_mm_raw",
    "force_n_raw",
    "velocity_m_s",
    "cycle_energy_j",
)

DISPLACEMENT_ZERO_POLICIES = ("top_of_stroke", "frame_zero")


@dataclass(frozen=True)
class CycleWindowTrace:
    """Averaged trace generated from a raw cycle window."""

    data: dict[str, np.ndarray]
    provenance: dict[str, Any]


def _frame_from_config(config: FrameConfig | dict[str, Any]) -> FrameConfig:
    if isinstance(config, dict):
        validate_frame_config(config)
        return FrameConfig(
            time_column=config["time_column"],
            position_column=config["position_column"],
            force_column=config["force_column"],
            position_sign=float(config["position_sign"]),
            force_sign=float(config["force_sign"]),
            displacement_zero=float(config.get("displacement_zero", 0.0)),
        )
    return config


def infer_cycle_column(columns: dict[str, np.ndarray]) -> str:
    """Infer the cycle-index column from raw Instron columns.

    Prefers a repeating per-cycle counter (``Elapsed Cycles``/``Total Cycles``)
    and only accepts a column that actually takes more than one distinct integer
    value, so an all-zero placeholder counter is skipped.
    """
    lowered = {name.lower(): name for name in columns}
    priority = (
        "elapsed cycles",
        "total cycles",
        "total cycle count(electropuls",
        "total cycle count",
    )
    for needle in priority:
        for lowered_name, original in lowered.items():
            if needle in lowered_name:
                values = columns[original]
                finite = values[np.isfinite(values)]
                if len(finite) and len(np.unique(np.rint(finite).astype(np.int64))) > 1:
                    return original
    raise ValueError(f"Could not infer cycle column from CSV columns: {', '.join(columns)}")


def _infer_cycle_time_column(columns: dict[str, np.ndarray]) -> str | None:
    lowered = {name.lower(): name for name in columns}
    for needle in ("cycle elapsed time", "elapsed time"):
        for lowered_name, original in lowered.items():
            if needle in lowered_name:
                return original
    return None


def _interp_cycle(phase_grid: np.ndarray, phase: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(phase)
    phase_sorted = phase[order]
    values_sorted = values[order]
    unique_phase, unique_indices = np.unique(phase_sorted, return_index=True)
    unique_values = values_sorted[unique_indices]
    if len(unique_phase) < 2:
        raise ValueError("Cycle has fewer than two unique phase samples")
    return np.interp(phase_grid, unique_phase, unique_values)


def build_cycle_window_trace(
    csv_path: str | Path,
    frame_config: FrameConfig | dict[str, Any],
    cycles: Iterable[int],
    *,
    phase_count: int = 501,
    cycle_column: str | None = None,
    displacement_zero_policy: str = "top_of_stroke",
) -> CycleWindowTrace:
    """Build a phase-averaged trace for a raw CSV cycle window.

    Args:
        csv_path: Raw Instron CSV path.
        frame_config: Column/sign choices, as a :class:`FrameConfig` or a saved
            frame-config mapping.
        cycles: Cycle indices to average (for example ``range(90, 99)``).
        phase_count: Number of phase-grid samples per cycle.
        cycle_column: Explicit cycle-index column; inferred when ``None``.
        displacement_zero_policy: How penetration is zeroed. ``"top_of_stroke"``
            (default) measures penetration downward from the highest averaged
            position, matching the shipped averaged-cycle files; ``"frame_zero"``
            subtracts the frame config's ``displacement_zero`` per raw sample.

    Returns:
        A :class:`CycleWindowTrace` with the averaged per-column arrays and a
        provenance mapping.
    """
    if phase_count < 3:
        raise ValueError("phase_count must be at least 3")
    if displacement_zero_policy not in DISPLACEMENT_ZERO_POLICIES:
        raise ValueError(f"displacement_zero_policy must be one of {DISPLACEMENT_ZERO_POLICIES}")
    cycle_ids = tuple(int(cycle) for cycle in cycles)
    if not cycle_ids:
        raise ValueError("cycles must contain at least one cycle")

    path = Path(csv_path)
    frame = _frame_from_config(frame_config)
    columns = read_numeric_csv(path)
    selected_cycle_column = cycle_column or infer_cycle_column(columns)
    if selected_cycle_column not in columns:
        raise ValueError(f"Cycle column {selected_cycle_column!r} is missing from {path}")

    required = (frame.time_column, frame.position_column, frame.force_column, selected_cycle_column)
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    time_s = columns[frame.time_column]
    position_raw_mm = columns[frame.position_column]
    force_raw_n = columns[frame.force_column]
    force_n = frame.force_sign * force_raw_n
    cycle_numeric = columns[selected_cycle_column]

    finite = np.isfinite(time_s) & np.isfinite(position_raw_mm) & np.isfinite(force_raw_n) & np.isfinite(cycle_numeric)
    cycle_values = np.zeros_like(cycle_numeric, dtype=np.int64)
    cycle_values[finite] = np.rint(cycle_numeric[finite]).astype(np.int64)
    phase_grid = np.linspace(0.0, 1.0, phase_count, dtype=np.float64)
    cycle_time_column = _infer_cycle_time_column(columns)

    force_rows = []
    position_rows = []
    raw_force_rows = []
    duration_s = []
    energy_rows = []
    energy_column = next((name for name in columns if "cycle energy" in name.lower()), None)

    for cycle_id in cycle_ids:
        mask = finite & (cycle_values == cycle_id)
        if int(np.count_nonzero(mask)) < 3:
            raise ValueError(f"Cycle {cycle_id} in {path} has fewer than three finite samples")

        if cycle_time_column is not None:
            local_time = np.asarray(columns[cycle_time_column][mask], dtype=np.float64)
        else:
            local_time = np.asarray(time_s[mask] - float(time_s[mask][0]), dtype=np.float64)
        local_duration = float(np.nanmax(local_time) - np.nanmin(local_time))
        if local_duration <= 0.0:
            raise ValueError(f"Cycle {cycle_id} in {path} has non-positive duration")
        phase = (local_time - float(np.nanmin(local_time))) / local_duration

        force_rows.append(_interp_cycle(phase_grid, phase, force_n[mask]))
        position_rows.append(_interp_cycle(phase_grid, phase, position_raw_mm[mask]))
        raw_force_rows.append(_interp_cycle(phase_grid, phase, force_raw_n[mask]))
        duration_s.append(local_duration)
        if energy_column is not None:
            energy_rows.append(_interp_cycle(phase_grid, phase, columns[energy_column][mask]))

    mean_duration = float(np.mean(duration_s))
    mean_force = np.mean(np.asarray(force_rows), axis=0)
    mean_position = np.mean(np.asarray(position_rows), axis=0)
    mean_raw_force = np.mean(np.asarray(raw_force_rows), axis=0)

    if displacement_zero_policy == "top_of_stroke":
        signed_into_sample = frame.position_sign * mean_position
        displacement_m = (signed_into_sample - float(np.min(signed_into_sample))) * 0.001
    else:
        displacement_m = frame.position_sign * (mean_position - frame.displacement_zero) * 0.001

    time_grid = phase_grid * mean_duration
    velocity = np.gradient(displacement_m, time_grid)
    cycle_energy = np.mean(np.asarray(energy_rows), axis=0) if energy_rows else np.zeros_like(phase_grid)

    data = {
        "phase": phase_grid,
        "time_s": time_grid,
        "displacement_m": displacement_m,
        "displacement_mm": displacement_m * 1000.0,
        "force_n": mean_force,
        "position_mm_raw": mean_position,
        "force_n_raw": mean_raw_force,
        "velocity_m_s": velocity,
        "cycle_energy_j": cycle_energy,
    }
    return CycleWindowTrace(
        data=data,
        provenance={
            "source_csv": str(path),
            "frame_config": frame.as_dict(),
            "cycle_column": selected_cycle_column,
            "cycle_time_column": cycle_time_column,
            "cycles": list(cycle_ids),
            "phase_count": int(phase_count),
            "displacement_zero_policy": displacement_zero_policy,
            "force_zero_policy": "sign_flip_only",
            "mean_duration_s": mean_duration,
        },
    )


def write_cycle_window_trace(path: str | Path, trace: CycleWindowTrace) -> None:
    """Write a cycle-window trace CSV in the shared averaged-cycle schema."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([trace.data[name] for name in TRACE_COLUMNS])
    np.savetxt(out_path, rows, delimiter=",", header=",".join(TRACE_COLUMNS), comments="")
