# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Normalized measurement evidence for cable calibration.

A :class:`CableEvidenceBundle` is the input boundary of the calibration workflow. It
holds one or more :class:`CableRecording` entries, each a camera's view of one
recording, with the RGB and mask sequences already materialized on disk and the
camera, timing and attachment data captured as plain JSON.

The point of the boundary is that a tuner reads a bundle and nothing else. It
never opens a bag, an MCAP file, a topic, or a video container. Producing a
bundle from those formats is an importer's job, and importers live outside this
package because their dependencies do.

A bundle is self-contained in the settings sense: ``applied_settings`` records
the importer settings that produced the frames and masks, so the evidence stays
interpretable after the tool that produced it changes its configuration.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import SCHEMA_VERSION, check_fields, check_schema_version


@dataclass
class CableRecording:
    """One camera's view of a recording, with its evidence materialized."""

    label: str
    """Short identifier, unique in the bundle. Used for artifact subfolders."""

    recording_key: str
    """Identifies the recording. Entries that share it are views of one event."""

    applied_settings: dict[str, Any]
    """Importer settings that produced the frames and masks, for example segmentation values, stored verbatim."""

    frame_timestamps: list[float]
    """Capture time of each kept frame [s], relative to :attr:`start_ns`. One per RGB frame.

    The first frame can be later than ``0``.
    """

    rgb_frames: list[str] = field(default_factory=list)
    """Paths to the RGB frames, relative to the bundle directory."""

    masks: list[str] = field(default_factory=list)
    """Paths to the binary masks, relative to the bundle directory. One per RGB frame."""

    sensor_pos: list[float] | None = None
    """Camera origin in the robot base frame [m]."""

    sensor_quat: list[float] | None = None
    """Camera orientation in the robot base frame, ``(qx, qy, qz, qw)``."""

    camera_intrinsics: list[float] | None = None
    """``[width, height, fx, fy, cx, cy]`` [px] of the stored frames and masks, or ``None`` when uncalibrated."""

    render_size: list[int] | None = None
    """``[width, height]`` [px] of the stored frames and masks, used when :attr:`camera_intrinsics` is ``None``."""

    fov_deg: float | None = None
    """Vertical field of view [deg], required when :attr:`camera_intrinsics` is ``None``. There is no default."""

    cable_start: list[float] | None = None
    """Cable attachment point in the robot base frame [m]."""

    driven: bool = False
    """Whether a recorded trajectory, stored at :attr:`trajectory_path`, drives the cable during simulation."""

    trajectory_path: str | None = None
    """Path to this recording's stored drive trajectory, relative to the bundle directory.

    See :class:`~.trajectory.MaterializedTrajectorySource`.
    """

    start_ns: int | None = None
    """Absolute timestamp [ns] that :attr:`frame_timestamps` are relative to. Required when :attr:`driven` is ``True``."""

    notes: list[str] = field(default_factory=list)
    """Free-form importer remarks, for example substituted calibration frames."""

    def to_dict(self) -> dict[str, Any]:
        """Return the recording as a JSON-serializable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CableRecording:
        """Build a recording from :meth:`to_dict` output.

        Raises:
            ValueError: If ``d`` has a field this class does not define.
        """
        check_fields(cls, d, "CableRecording")
        return cls(**d)


@dataclass
class CableEvidenceBundle:
    """A set of co-imported recordings, one per camera view.

    Distances are in meters. Poses are in the robot base frame (right-handed,
    z up), and quaternions are ``(qx, qy, qz, qw)``.
    """

    recordings: list[CableRecording] = field(default_factory=list)
    """The :class:`CableRecording` entries, one per camera view."""

    schema_version: int = SCHEMA_VERSION
    """Format version of the serialized bundle."""

    def to_dict(self) -> dict[str, Any]:
        """Return the bundle as a JSON-serializable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CableEvidenceBundle:
        """Build a bundle from :meth:`to_dict` output.

        Raises:
            ValueError: If the schema version is not supported, or ``d`` or one
                of its recordings has a field this format does not define.
        """
        d = dict(d)
        check_schema_version(d.get("schema_version"), "CableEvidenceBundle")
        check_fields(cls, d, "CableEvidenceBundle")
        recordings = [CableRecording.from_dict(r) for r in d.pop("recordings", [])]
        return cls(recordings=recordings, **d)

    def validate(self) -> None:
        """Check the bundle is usable before a tuner allocates anything.

        Raises:
            ValueError: with the offending recording named.
        """
        check_schema_version(self.schema_version, "CableEvidenceBundle")
        if not self.recordings:
            raise ValueError("CableEvidenceBundle has no recordings.")
        seen = set()
        for r in self.recordings:
            tag = f"recording '{r.label}'"
            if r.label in seen:
                raise ValueError(f"{tag}: duplicate label.")
            seen.add(r.label)
            if not r.rgb_frames:
                raise ValueError(f"{tag}: no rgb frames.")
            if len(r.masks) != len(r.rgb_frames):
                raise ValueError(
                    f"{tag}: {len(r.masks)} mask(s) but {len(r.rgb_frames)} rgb frame(s); "
                    "one mask per reference frame is required."
                )
            if len(r.frame_timestamps) != len(r.rgb_frames):
                raise ValueError(
                    f"{tag}: {len(r.frame_timestamps)} frame_timestamps but {len(r.rgb_frames)} rgb frame(s)."
                )
            if r.frame_timestamps[0] < 0.0:
                raise ValueError(f"{tag}: frame_timestamps must not precede start_ns, got {r.frame_timestamps[0]}.")
            for prev, cur in zip(r.frame_timestamps, r.frame_timestamps[1:], strict=False):
                if cur <= prev:
                    raise ValueError(f"{tag}: frame_timestamps must be strictly increasing, got {prev} then {cur}.")
            if not r.sensor_pos or len(r.sensor_pos) != 3:
                raise ValueError(f"{tag}: invalid sensor_pos {r.sensor_pos!r}; expected [x, y, z].")
            if not r.sensor_quat or len(r.sensor_quat) != 4:
                raise ValueError(f"{tag}: invalid sensor_quat {r.sensor_quat!r}; expected [qx, qy, qz, qw].")
            if r.camera_intrinsics is not None:
                if len(r.camera_intrinsics) != 6:
                    raise ValueError(
                        f"{tag}: invalid camera_intrinsics {r.camera_intrinsics!r}; expected [width, height, fx, fy, cx, cy]."
                    )
            else:
                if r.render_size is None:
                    raise ValueError(f"{tag}: needs camera_intrinsics or render_size.")
                if len(r.render_size) != 2:
                    raise ValueError(f"{tag}: invalid render_size {r.render_size!r}; expected [width, height].")
                if r.fov_deg is None:
                    raise ValueError(f"{tag}: no camera_intrinsics, so fov_deg is required to define the sim camera.")
            if not r.cable_start or len(r.cable_start) != 3:
                raise ValueError(f"{tag}: invalid cable_start {r.cable_start!r}.")
            if r.driven and not r.trajectory_path:
                raise ValueError(f"{tag}: driven=True but no trajectory_path, so there is no drive to replay.")
            if r.driven and r.start_ns is None:
                raise ValueError(f"{tag}: driven=True but no start_ns, so the frames cannot be aligned with the drive.")

    def save(self, bundle_dir: str | os.PathLike) -> str:
        """Write ``bundle.json`` to ``bundle_dir``, creating the directory if needed.

        Only the bundle description is written. The frames, masks and
        trajectories it refers to must already be in ``bundle_dir``.

        Returns:
            The path of the written ``bundle.json``.
        """
        os.makedirs(bundle_dir, exist_ok=True)
        path = os.path.join(bundle_dir, "bundle.json")
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, bundle_dir: str | os.PathLike) -> CableEvidenceBundle:
        """Read ``bundle.json`` from ``bundle_dir``. Does not call :meth:`validate`."""
        with open(os.path.join(bundle_dir, "bundle.json")) as fh:
            return cls.from_dict(json.load(fh))
