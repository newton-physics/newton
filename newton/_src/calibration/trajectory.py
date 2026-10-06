# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A drive trajectory that carries its own poses.

Evidence bundles store drive trajectories in this form, so a bundle replays
without the recording it came from.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import warp as wp

from .data_source import CableDataSource, Pose
from .schema import SCHEMA_VERSION, check_fields, check_schema_version


@dataclass
class MaterializedTrajectorySource(CableDataSource):
    """A drive trajectory sampled once and stored, not read from a live source.

    Poses are ``(x, y, z)`` / ``(qx, qy, qz, qw)`` of the tool centre in the
    robot base frame, at :attr:`timestamps_ns`, strictly increasing. Between
    stored samples a query interpolates: linear in position, shortest-path
    SLERP in orientation. Outside them it holds the nearest end pose.
    """

    TYPE_NAME = "materialized_trajectory"
    """Value of the ``type`` field in the serialized form."""

    source_id: str
    """Identity of the recording this trajectory serves."""

    timestamps_ns: list[int] = field(default_factory=list)
    """Absolute sample times [ns], strictly increasing."""

    positions: list[list[float]] = field(default_factory=list)
    """Tool-centre position ``(x, y, z)`` at each sample [m]."""

    quaternions: list[list[float]] = field(default_factory=list)
    """Tool-centre orientation ``(qx, qy, qz, qw)`` at each sample."""

    def to_dict(self) -> dict[str, Any]:
        """Return the trajectory as a JSON-serializable mapping, with its type and schema version."""
        d = asdict(self)
        d["type"] = self.TYPE_NAME
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MaterializedTrajectorySource:
        """Build a trajectory from :meth:`to_dict` output.

        Raises:
            ValueError: If the schema version is not supported, the type is not
                :attr:`TYPE_NAME`, or ``d`` has a field this class does not define.
        """
        d = dict(d)
        check_schema_version(d.pop("schema_version", None), "MaterializedTrajectorySource")
        type_name = d.pop("type", None)
        if type_name != cls.TYPE_NAME:
            raise ValueError(f"expected a data source of type {cls.TYPE_NAME!r}, got {type_name!r}.")
        check_fields(cls, d, "MaterializedTrajectorySource")
        return cls(**d)

    def validate(self) -> None:
        """Check the samples can be interpolated.

        Raises:
            ValueError: If there are fewer than two samples, the arrays differ in
                length, a position does not have 3 values or a quaternion 4, or
                the timestamps are not strictly increasing.
        """
        n = len(self.timestamps_ns)
        if n < 2:
            raise ValueError(f"data source {self.source_id}: needs at least 2 samples, got {n}.")
        if len(self.positions) != n or len(self.quaternions) != n:
            raise ValueError(
                f"data source {self.source_id}: {n} timestamps_ns but "
                f"{len(self.positions)} positions and {len(self.quaternions)} quaternions."
            )
        for name, values, size in (("position", self.positions, 3), ("quaternion", self.quaternions, 4)):
            for i, value in enumerate(values):
                if len(value) != size:
                    raise ValueError(
                        f"data source {self.source_id}: {name} {i} has {len(value)} values, expected {size}."
                    )
        for prev, cur in zip(self.timestamps_ns, self.timestamps_ns[1:], strict=False):
            if cur <= prev:
                raise ValueError(
                    f"data source {self.source_id}: timestamps_ns must be strictly increasing, got {prev} then {cur}."
                )

    def coverage_ns(self) -> tuple[int, int]:
        """``(first_ns, last_ns)`` actually covered by stored samples."""
        return self.timestamps_ns[0], self.timestamps_ns[-1]

    def tcp_transforms(self, stamps_ns: Sequence[int]) -> list[Pose]:
        """Interpolated tool-centre pose at each instant.

        Args:
            stamps_ns: Absolute timestamps [ns].

        Returns:
            One ``(position, quaternion)`` pair of float tuples per stamp.
        """
        ts = self.timestamps_ns
        lo, hi = ts[0], ts[-1]
        out = []
        for stamp in stamps_ns:
            if stamp <= lo or stamp >= hi:
                end = 0 if stamp <= lo else -1
                pos = tuple(float(v) for v in self.positions[end])
                quat = tuple(float(v) for v in self.quaternions[end])
                out.append((pos, quat))
                continue
            i1 = bisect.bisect_right(ts, stamp)
            i0 = i1 - 1
            frac = (stamp - ts[i0]) / (ts[i1] - ts[i0])
            pos = wp.lerp(wp.vec3(*self.positions[i0]), wp.vec3(*self.positions[i1]), frac)
            quat = wp.quat_slerp(wp.quat(*self.quaternions[i0]), wp.quat(*self.quaternions[i1]), frac)
            out.append((tuple(float(v) for v in pos), tuple(float(v) for v in quat)))
        return out
