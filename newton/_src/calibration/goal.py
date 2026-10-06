# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""What a candidate is scored against, in memory.

A :class:`CableGoal` is one camera's view of one recording: the reference masks
plus the camera and crop needed to render a candidate the same way. It is the
in-memory counterpart of :class:`~.evidence.CableRecording`, holding decoded
pixels rather than paths.

Views of a single recording share their physics and differ only in where the
camera is, so they share one simulation. :func:`group_goals` collects them into
:class:`CableGoalGroup`, which is what an evaluation actually iterates over.

Sharing a simulation first requires a common clock. Each view's frame times are
anchored to *its own* first capture, and two cameras of one recording generally
start at different instants. Scored independently that is invisible; shared, it
would misalign the views against each other and defeat the point of using more
than one. So a multi-view group adopts the earliest view's origin, shifts the
others by their offset, and re-reads the drive from that origin over a span
covering the latest frame of any view.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import numpy as np
import warp as wp

from .data_source import CableDataSource, Pose

Vec3: TypeAlias = tuple[float, float, float]
"""An ``(x, y, z)`` vector."""


@dataclass
class CableGoal:
    """One camera's view of one recording."""

    label: str
    """Short identifier, unique within an evaluation."""

    recording_key: str
    """Identifies the recording. Goals sharing it are views of one event."""

    masks: list[np.ndarray]
    """Reference masks, one ``(height, width)`` uint8 array per scored frame."""

    frame_times: list[float]
    """Capture time of each scored frame [s] relative to :attr:`start_ns`. The
    simulated timeline ends at the frame of the last one; see :func:`group_goals`."""

    cable_start: Vec3
    """Cable attachment point in the robot base frame [m]."""

    sensor_pos: Vec3
    """Camera optical-frame origin in base [m]."""

    crop: list[int]
    """``[x0, y0, x1, y1]`` region scored in this view [px]."""

    attachment_transform: Pose
    """``T_tcp_attachment`` as ``((x, y, z), (qx, qy, qz, qw))``.

    Views of one recording must agree; see :func:`group_goals`.
    """

    clamp_position: float
    """Arc length [m] from the node-0 end of the cable to the TCP grasp point.

    0.0 is the end clamp; an interior value fixes an interior capsule and lets
    both sides hang. Views of one recording must agree; see :func:`group_goals`.
    """

    sensor_quat: list[float] | None = None
    """Camera optical-frame orientation in base, ``(qx, qy, qz, qw)``."""

    camera_intrinsics: tuple[float, ...] | None = None
    """``(width, height, fx, fy, cx, cy)`` [px], or ``None`` when uncalibrated."""

    render_size: tuple[int, int] | None = None
    """``(width, height)`` [px] to render at when :attr:`camera_intrinsics` is ``None``."""

    fov_deg: float | None = None
    """Vertical field of view [deg], required when :attr:`camera_intrinsics` is ``None``."""

    start_ns: int | None = None
    """Absolute timestamp [ns] :attr:`frame_times` is relative to.

    Needed to put co-recorded views on a common clock; see :func:`group_goals`.
    """

    driven: bool = False
    """Whether this recording drives the cable from a recorded trajectory."""

    data_source: CableDataSource | None = None
    """The :class:`~.data_source.CableDataSource` serving the drive, if driven."""

    cable_axis: Vec3 | None = None
    """Direction ``(x, y, z)`` the cable runs at the grasp, in the TCP (gripper)
    frame, or ``None`` to use the attachment rotation. Overrides the attachment
    rotation. Views of one recording must agree; see :func:`group_goals`.
    """

    def validate(self) -> None:
        """Check that this goal can be rendered and scored.

        Raises:
            ValueError: If the goal has no masks, a frame time per mask is
                missing, the camera is underspecified, or a driven goal has no
                data source or no :attr:`start_ns`.
        """
        if not self.masks:
            raise ValueError(f"goal '{self.label}': no masks to score against.")
        if len(self.frame_times) != len(self.masks):
            raise ValueError(
                f"goal '{self.label}': {len(self.frame_times)} frame time(s) but {len(self.masks)} mask(s)."
            )
        if self.camera_intrinsics is None:
            if self.render_size is None:
                raise ValueError(f"goal '{self.label}': needs camera_intrinsics or render_size.")
            if self.fov_deg is None:
                raise ValueError(
                    f"goal '{self.label}': no camera_intrinsics, so fov_deg is required to define the sim camera."
                )
        if self.driven and self.data_source is None:
            raise ValueError(f"goal '{self.label}': driven but no data_source to supply the trajectory.")
        if self.driven and self.start_ns is None:
            raise ValueError(
                f"goal '{self.label}': driven but no start_ns, so the frames cannot be aligned with the drive."
            )


@dataclass
class CableGoalGroup:
    """Views of one recording, scored against a single shared simulation."""

    views: list[CableGoal]
    """The :class:`CableGoal` entries, one per camera."""

    num_frames: int
    """Timeline length covering the latest frame time across all views."""

    cable_start: Vec3
    """Attachment point, taken from the view that owns the group's origin."""

    attachment_transform: Pose
    """``T_tcp_attachment``, agreed by every view; see :class:`CableGoal`."""

    clamp_position: float
    """Clamp position [m], agreed by every view; see :class:`CableGoal`."""

    cable_axis: Vec3 | None = None
    """Grasp direction (TCP frame), agreed by every view; see :class:`CableGoal`."""

    transform_buffer: wp.array[wp.transform] | None = None
    """Per-substep drive from the shared origin, or ``None`` when undriven."""

    weights: list[float] = field(default_factory=list)
    """Per-view objective weight. See :func:`group_goals`."""

    @property
    def label(self) -> str:
        """The recording key for a multi-view group, else the single view's label."""
        return self.views[0].recording_key if len(self.views) > 1 else self.views[0].label

    def cameras(self) -> list[dict[str, Any]]:
        """Per-view camera descriptors, shaped for ``CableWorld``."""
        return [
            {
                "sensor_pos": v.sensor_pos,
                "sensor_quat": v.sensor_quat,
                "camera_intrinsics": v.camera_intrinsics,
                "render_size": v.render_size,
                "fov_deg": v.fov_deg,
            }
            for v in self.views
        ]

    def run_args(self, group_reprs: Sequence[Any]) -> tuple[Any, Any, Any]:
        """``(goal_reprs, crop, frame_times)`` shaped for ``CableWorld.run_sequence``.

        Per-view lists for a multi-camera group, bare values for a single-camera
        one, matching run_sequence's rule that the shapes follow the camera count.
        """
        if len(self.views) == 1:
            v = self.views[0]
            return group_reprs[0], v.crop, v.frame_times
        return group_reprs, [v.crop for v in self.views], [v.frame_times for v in self.views]

    def per_view(self, result: Any) -> list[Any]:
        """A run_sequence return value as a per-view list, whatever the camera count."""
        return [result] if len(self.views) == 1 else result


def _as_tuple(value: Any) -> Any:
    """``value`` with every list or tuple in it converted to a tuple."""
    if isinstance(value, list | tuple):
        return tuple(_as_tuple(item) for item in value)
    return value


def _agreed(views: Sequence[CableGoal], name: str) -> Any:
    """The one value of the field ``name`` every view of a recording must share.

    The grasp is a property of the recording, not the camera. Taking the first
    view's value would silently ignore views that disagree, so raise instead,
    naming the recording. Lists compare equal to tuples with the same items.
    """
    values = [_as_tuple(getattr(v, name)) for v in views]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(
            f"recording '{views[0].recording_key}': views disagree on {name}: {sorted(set(values), key=repr)}."
        )
    return values[0]


def _agreed_drive(views: Sequence[CableGoal]) -> CableDataSource | None:
    """The one drive every view of a recording must share, or ``None`` when undriven.

    Like the grasp, the drive is a property of the recording, not the camera:
    every view must agree on :attr:`CableGoal.driven` and, when driven, use an
    equal data source.
    """
    if not _agreed(views, "driven"):
        return None
    source = views[0].data_source
    if any(v.data_source != source for v in views[1:]):
        raise ValueError(f"recording '{views[0].recording_key}': views disagree on data_source.")
    return source


def group_goals(goals: Sequence[CableGoal], fps: int, sim_substeps: int) -> list[CableGoalGroup]:
    """Collect goals into groups that share one simulation.

    Each *recording* contributes one unit of objective weight, split evenly over
    the cameras that observed it. Two views of one recording therefore weigh half
    each rather than one each: the extra view still contributes its complementary
    information, but pointing more cameras at a recording does not multiply that
    recording's say in the parameter trade-offs.

    Args:
        goals: The :class:`CableGoal` entries to group.
        fps: Simulation frame rate [Hz], for converting time offsets to frames.
        sim_substeps: Substeps per frame, for sizing a re-anchored drive.

    Returns:
        One :class:`CableGoalGroup` per recording, in the order the recordings
        first appear in ``goals``. A goal without :attr:`CableGoal.start_ns`
        gets a group of its own.

    Raises:
        ValueError: If the views of one recording disagree on
            ``attachment_transform``, ``clamp_position``, ``cable_axis``,
            ``driven`` or ``data_source``.
    """
    order, by_key = [], {}
    for goal in goals:
        # A goal groups only if it can be put on a common clock with its peers.
        groupable = goal.start_ns is not None
        key = goal.recording_key if groupable else id(goal)
        if key not in by_key:
            by_key[key] = []
            order.append(key)
        by_key[key].append(goal)

    view_counts = {}
    for goal in goals:
        view_counts[goal.recording_key] = view_counts.get(goal.recording_key, 0) + 1

    groups = []
    for key in order:
        views = by_key[key]
        weights = [1.0 / view_counts[v.recording_key] for v in views]
        if len(views) == 1:
            v = views[0]
            num_frames = 1 + round(max(v.frame_times) * fps)
            drive = None
            if v.driven:
                drive = v.data_source.drive_buffer(v.start_ns, num_frames, fps * sim_substeps, sim_substeps)
            groups.append(
                CableGoalGroup(
                    views=views,
                    num_frames=num_frames,
                    cable_start=v.cable_start,
                    attachment_transform=v.attachment_transform,
                    clamp_position=v.clamp_position,
                    cable_axis=v.cable_axis,
                    transform_buffer=drive,
                    weights=weights,
                )
            )
            continue

        t0 = min(v.start_ns for v in views)
        # The earliest view's cable_start already refers to t0, so adopting it
        # keeps whatever that view resolved.
        base = next(v for v in views if v.start_ns == t0)
        num_frames = 0
        anchored = []
        for v in views:
            dt = (v.start_ns - t0) * 1e-9
            times = [t + dt for t in v.frame_times]
            # Re-anchoring yields a new view rather than editing the caller's. The
            # same goal may be grouped more than once -- two evaluators over one
            # list, say -- and shifting in place would compound the offset, with
            # nothing to signal that the clock had moved twice.
            anchored.append(replace(v, frame_times=times))
            num_frames = max(num_frames, 1 + round(max(times) * fps))
        drive = None
        source = _agreed_drive(views)
        if source is not None:
            drive = source.drive_buffer(t0, num_frames, fps * sim_substeps, sim_substeps)
        groups.append(
            CableGoalGroup(
                views=anchored,
                num_frames=num_frames,
                cable_start=base.cable_start,
                attachment_transform=_agreed(views, "attachment_transform"),
                clamp_position=_agreed(views, "clamp_position"),
                cable_axis=_agreed(views, "cable_axis"),
                transform_buffer=drive,
                weights=weights,
            )
        )
    return groups
