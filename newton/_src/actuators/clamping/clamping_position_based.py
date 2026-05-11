# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import Clamping


@wp.func
def _interp_1d(
    x: float,
    xs: wp.array[float],
    ys: wp.array[float],
    n: int,
) -> float:
    """Linearly interpolate (x -> y) from sorted sample arrays, clamping at boundaries."""
    if n <= 0:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    for k in range(n - 1):
        if xs[k + 1] >= x:
            dx = xs[k + 1] - xs[k]
            if dx == 0.0:
                return ys[k]
            t = (x - xs[k]) / dx
            return ys[k] + t * (ys[k + 1] - ys[k])
    return ys[n - 1]


@wp.kernel
def _position_based_clamp_kernel(
    current_pos: wp.array[float],
    state_indices: wp.array[wp.uint32],
    lookup_positions: wp.array[float],
    lookup_efforts: wp.array[float],
    lookup_size: int,
    src: wp.array[float],
    dst: wp.array[float],
):
    """Position-dependent clamping via interpolated lookup table: read src, write dst."""
    i = wp.tid()
    state_idx = state_indices[i]
    limit = _interp_1d(current_pos[state_idx], lookup_positions, lookup_efforts, lookup_size)
    dst[i] = wp.clamp(src[i], -limit, limit)


class ClampingPositionBased(Clamping):
    """Position-dependent effort clamping via lookup table.

    Provides position-dependent effort limits interpolated from a
    lookup table to model actuators whose maximum output effort varies
    with joint position.

    The lookup table can be provided either as a file path
    (``lookup_table_path``) or as direct value lists
    (``lookup_positions`` + ``lookup_efforts``).  When a path is given,
    the file is read in :meth:`finalize`.

    Sampling rules:

    - Inputs strictly between adjacent entries are linearly interpolated.
    - Inputs at or below ``lookup_positions[0]`` clamp to ``lookup_efforts[0]``.
    - Inputs at or above ``lookup_positions[-1]`` clamp to ``lookup_efforts[-1]``.
    - For rotational actuators, positions do not wrap periodically.

    The lookup table is a shared parameter: all DOFs within one
    :class:`~newton.actuators.Actuator` group share the same table.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"lookup_table_path", "lookup_positions", "lookup_efforts"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Accepts either ``lookup_table_path`` (file) or
        ``lookup_positions`` + ``lookup_efforts`` (direct values).

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        has_path = "lookup_table_path" in args
        has_direct = "lookup_positions" in args or "lookup_efforts" in args

        if has_path and has_direct:
            raise ValueError("Provide either 'lookup_table_path' or 'lookup_positions'+'lookup_efforts', not both")
        if not has_path and not has_direct:
            raise ValueError(
                "ClampingPositionBased requires 'lookup_table_path' or 'lookup_positions'+'lookup_efforts'"
            )

        if has_path:
            return {"lookup_table_path": args["lookup_table_path"]}

        if "lookup_positions" not in args or "lookup_efforts" not in args:
            raise ValueError("Both 'lookup_positions' and 'lookup_efforts' are required")
        positions = tuple(args["lookup_positions"])
        efforts = tuple(args["lookup_efforts"])
        if len(positions) == 0:
            raise ValueError("lookup_positions/lookup_efforts must not be empty")
        if len(positions) != len(efforts):
            raise ValueError(
                f"lookup_positions length ({len(positions)}) must match lookup_efforts length ({len(efforts)})"
            )
        if any(v < 0 for v in efforts):
            raise ValueError("lookup_efforts must contain non-negative values for symmetric clamping")
        if not all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1)):
            raise ValueError("lookup_positions must be monotonically non-decreasing for interpolation")
        return {"lookup_positions": positions, "lookup_efforts": efforts}

    def __init__(
        self,
        lookup_table_path: str | None = None,
        lookup_positions: tuple[float, ...] | None = None,
        lookup_efforts: tuple[float, ...] | None = None,
    ):
        """Initialize position-based clamp.

        Provide *either* ``lookup_table_path`` *or* both
        ``lookup_positions`` and ``lookup_efforts``.

        Args:
            lookup_table_path: Path to a whitespace/comma-separated
                text file with two columns (position, effort).  Lines
                starting with ``#`` are comments.  The file is read
                in :meth:`finalize`.
            lookup_positions: Sorted joint positions [rad or m] for the
                effort lookup table.  Shape ``(K,)``.
            lookup_efforts: Max output efforts [N·m or N] corresponding
                to *lookup_positions*.  Shape ``(K,)``.
        """
        if lookup_table_path is None and (lookup_positions is None or lookup_efforts is None):
            raise ValueError("Provide either 'lookup_table_path' or both 'lookup_positions' and 'lookup_efforts'")
        if lookup_positions is not None and lookup_efforts is not None:
            if len(lookup_positions) != len(lookup_efforts):
                raise ValueError(
                    f"lookup_positions length ({len(lookup_positions)}) must match "
                    f"lookup_efforts length ({len(lookup_efforts)})"
                )
        self._lookup_table_path = lookup_table_path
        self._positions_tuple = lookup_positions
        self._efforts_tuple = lookup_efforts
        self.lookup_size: int = 0
        self.lookup_positions: wp.array[float] | None = None
        self.lookup_efforts: wp.array[float] | None = None

    def _read_lookup_table(self, path: str) -> tuple[list[float], list[float]]:
        """Parse a whitespace/comma-separated lookup table file.

        Args:
            path: File path to the lookup table.

        Returns:
            ``(positions, efforts)`` as lists of floats.
        """
        table_path = Path(path)
        if not table_path.is_file():
            raise ValueError(f"Lookup table file not found: {path}")
        positions: list[float] = []
        efforts: list[float] = []
        for raw_line in table_path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            positions.append(float(parts[0]))
            efforts.append(float(parts[1]))
        if not positions:
            raise ValueError(f"Lookup table file is empty: {path}")
        if any(v < 0 for v in efforts):
            raise ValueError(f"Lookup table efforts must be non-negative in: {path}")
        if not all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1)):
            raise ValueError(f"Lookup table positions must be monotonically non-decreasing in: {path}")
        return positions, efforts

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Called by :class:`Actuator` after construction.

        Reads the lookup table from file (if a path was given) and
        allocates device arrays.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs).
        """
        if self._lookup_table_path is not None:
            positions, efforts = self._read_lookup_table(self._lookup_table_path)
            self._lookup_table_path = None
        else:
            positions = list(self._positions_tuple)
            efforts = list(self._efforts_tuple)
            self._positions_tuple = None
            self._efforts_tuple = None

        self.lookup_size = len(positions)
        self.lookup_positions = wp.array(np.array(positions, dtype=np.float32), dtype=wp.float32, device=device)
        self.lookup_efforts = wp.array(np.array(efforts, dtype=np.float32), dtype=wp.float32, device=device)

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_position_based_clamp_kernel,
            dim=len(src_forces),
            inputs=[
                positions,
                pos_indices,
                self.lookup_positions,
                self.lookup_efforts,
                self.lookup_size,
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
