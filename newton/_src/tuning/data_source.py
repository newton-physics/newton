# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Where a recording's drive trajectory comes from.

A :class:`CableDataSource` answers one question: give me the tool-centre pose at
these instants. It hides how. That is the seam which keeps container formats out
of the tuner -- an implementation may replay a robotics log, read a materialized
trajectory, or synthesize one, and nothing downstream changes.

Implementations that read recordings normally live outside this package, because
their dependencies do. An evidence bundle stores only
:class:`~.trajectory.MaterializedTrajectorySource`, so it replays without them.

Timestamps cross this interface as plain integer nanoseconds. Message types
belong to the formats being read, not to the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

import warp as wp

Pose: TypeAlias = tuple[tuple[float, float, float], tuple[float, float, float, float]]
"""A ``((x, y, z), (qx, qy, qz, qw))`` pose."""


class CableDataSource(ABC):
    """Interface for a recording's drive trajectory.

    Subclasses implement :meth:`tcp_transforms`.
    """

    source_id: str
    """Identity of the recording this source serves."""

    @abstractmethod
    def tcp_transforms(self, stamps_ns: Sequence[int]) -> list[Pose]:
        """Absolute tool-centre pose at each instant.

        Args:
            stamps_ns: Absolute timestamps [ns].

        Returns:
            One ``(translation, quaternion)`` pair per stamp, in the robot base
            frame, with the quaternion as ``(qx, qy, qz, qw)``.
        """

    def drive_buffer(
        self, start_ns: int, num_frames: int, substep_rate: int, sim_substeps: int
    ) -> wp.array[wp.transform]:
        """Per-substep drive transforms, shaped for the simulator.

        Samples ``num_frames * sim_substeps`` instants at ``substep_rate`` Hz
        from ``start_ns``.

        Args:
            start_ns: Absolute time of the first sample [ns].
            num_frames: Number of simulated frames.
            substep_rate: Sampling rate [Hz], normally the frame rate times
                ``sim_substeps``.
            sim_substeps: Simulation substeps per frame.

        Returns:
            One tool-centre pose per substep, in sampling order.

        Raises:
            ValueError: If :meth:`tcp_transforms` returns a different number of poses.
        """
        n = num_frames * sim_substeps
        stamps_ns = [start_ns + (k * 1_000_000_000) // substep_rate for k in range(n)]
        transforms = self.tcp_transforms(stamps_ns)
        if len(transforms) != n:
            raise ValueError(f"{type(self).__name__}: asked for {n} pose(s), got {len(transforms)}.")
        return wp.array([wp.transform(t, q) for t, q in transforms], dtype=wp.transform)
