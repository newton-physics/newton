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
    lookup_angles: wp.array[float],
    lookup_torques: wp.array[float],
    lookup_size: int,
    src: wp.array[float],
    dst: wp.array[float],
):
    """Angle-dependent clamping via interpolated lookup table: read src, write dst."""
    i = wp.tid()
    state_idx = state_indices[i]
    limit = _interp_1d(current_pos[state_idx], lookup_angles, lookup_torques, lookup_size)
    dst[i] = wp.clamp(src[i], -limit, limit)


class ClampingPositionBased(Clamping):
    """Angle-dependent torque clamping via lookup table.

    Replaces a fixed ±max_force box clamp with angle-dependent torque
    limits interpolated from a lookup table.  Models actuators where
    the transmission ratio and thus maximum output torque vary with
    joint angle (e.g., linkage-driven joints).

    The lookup table can be provided either as a file path
    (``lookup_table_path``) or as direct value lists
    (``lookup_angles`` + ``lookup_torques``).  When a path is given,
    the file is read in :meth:`finalize`, mirroring the pattern used
    by the neural-network controllers.

    The lookup table is a shared parameter: all DOFs within one
    :class:`~newton.actuators.Actuator` group share the same table.

    Class Attributes:
        SHARED_PARAMS: Parameter names that are instance-level (shared across
            all DOFs). Different values require separate actuator instances.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"lookup_table_path", "lookup_angles", "lookup_torques"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Accepts either ``lookup_table_path`` (file) or
        ``lookup_angles`` + ``lookup_torques`` (direct values).

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        has_path = "lookup_table_path" in args
        has_direct = "lookup_angles" in args or "lookup_torques" in args

        if has_path and has_direct:
            raise ValueError("Provide either 'lookup_table_path' or 'lookup_angles'+'lookup_torques', not both")
        if not has_path and not has_direct:
            raise ValueError("ClampingPositionBased requires 'lookup_table_path' or 'lookup_angles'+'lookup_torques'")

        if has_path:
            return {"lookup_table_path": args["lookup_table_path"]}

        if "lookup_angles" not in args or "lookup_torques" not in args:
            raise ValueError("Both 'lookup_angles' and 'lookup_torques' are required")
        angles = tuple(args["lookup_angles"])
        torques = tuple(args["lookup_torques"])
        if len(angles) != len(torques):
            raise ValueError(f"lookup_angles length ({len(angles)}) must match lookup_torques length ({len(torques)})")
        if any(v < 0 for v in torques):
            raise ValueError("lookup_torques must contain non-negative values for symmetric clamping")
        if not all(angles[i] <= angles[i + 1] for i in range(len(angles) - 1)):
            raise ValueError("lookup_angles must be monotonically non-decreasing for interpolation")
        return {"lookup_angles": angles, "lookup_torques": torques}

    def __init__(
        self,
        lookup_table_path: str | None = None,
        lookup_angles: tuple[float, ...] | None = None,
        lookup_torques: tuple[float, ...] | None = None,
    ):
        """Initialize position-based clamp.

        Provide *either* ``lookup_table_path`` *or* both
        ``lookup_angles`` and ``lookup_torques``.

        Args:
            lookup_table_path: Path to a whitespace/comma-separated
                text file with two columns (angle, torque).  Lines
                starting with ``#`` are comments.  The file is read
                in :meth:`finalize`.
            lookup_angles: Sorted joint angles [rad] for the torque
                lookup table.  Shape ``(K,)``.
            lookup_torques: Max output torques [N·m] corresponding to
                *lookup_angles*.  Shape ``(K,)``.
        """
        if lookup_table_path is None and (lookup_angles is None or lookup_torques is None):
            raise ValueError("Provide either 'lookup_table_path' or both 'lookup_angles' and 'lookup_torques'")
        if lookup_angles is not None and lookup_torques is not None:
            if len(lookup_angles) != len(lookup_torques):
                raise ValueError(
                    f"lookup_angles length ({len(lookup_angles)}) must match "
                    f"lookup_torques length ({len(lookup_torques)})"
                )
        self._lookup_table_path = lookup_table_path
        self._angles_tuple = lookup_angles
        self._torques_tuple = lookup_torques
        self.lookup_size: int = 0
        self.lookup_angles: wp.array[float] | None = None
        self.lookup_torques: wp.array[float] | None = None

    def _read_lookup_table(self, path: str) -> tuple[list[float], list[float]]:
        """Parse a whitespace/comma-separated lookup table file.

        Args:
            path: File path to the lookup table.

        Returns:
            ``(angles, torques)`` as lists of floats.
        """
        table_path = Path(path)
        if not table_path.is_file():
            raise ValueError(f"Lookup table file not found: {path}")
        angles: list[float] = []
        torques: list[float] = []
        for raw_line in table_path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            angles.append(float(parts[0]))
            torques.append(float(parts[1]))
        if not angles:
            raise ValueError(f"Lookup table file is empty: {path}")
        if any(v < 0 for v in torques):
            raise ValueError(f"Lookup table torques must be non-negative in: {path}")
        if not all(angles[i] <= angles[i + 1] for i in range(len(angles) - 1)):
            raise ValueError(f"Lookup table angles must be monotonically non-decreasing in: {path}")
        return angles, torques

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Called by :class:`Actuator` after construction.

        Reads the lookup table from file (if a path was given) and
        allocates device arrays.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs).
        """
        if self._lookup_table_path is not None:
            angles, torques = self._read_lookup_table(self._lookup_table_path)
            self._lookup_table_path = None
        else:
            angles = list(self._angles_tuple)
            torques = list(self._torques_tuple)
            self._angles_tuple = None
            self._torques_tuple = None

        self.lookup_size = len(angles)
        self.lookup_angles = wp.array(np.array(angles, dtype=np.float32), dtype=wp.float32, device=device)
        self.lookup_torques = wp.array(np.array(torques, dtype=np.float32), dtype=wp.float32, device=device)

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
                self.lookup_angles,
                self.lookup_torques,
                self.lookup_size,
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
