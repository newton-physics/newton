# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read an evidence bundle into goals.

:func:`bundle_to_goals` reads a :class:`~.evidence.CableEvidenceBundle` from its
directory and returns one :class:`~.goal.CableGoal` per recording. It is the only
function that opens the files a bundle refers to. The code after it works with
decoded arrays, not with file paths.

It reads only the masks and the drive trajectories. The RGB frames are for
review, and no loss uses them, so they are not decoded.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np

from .evidence import CableEvidenceBundle
from .goal import CableGoal
from .trajectory import MaterializedTrajectorySource


def _load_mask(path: str) -> np.ndarray:
    """Read one binary mask as a ``(height, width)`` uint8 array.

    Pillow is imported here rather than at module level, because it is an
    optional dependency.
    """
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def bundle_to_goals(
    bundle: CableEvidenceBundle,
    bundle_dir: str | os.PathLike,
    *,
    attachment_transform: Sequence[Sequence[float]],
    clamp_position: float = 0.0,
    cable_axis: Sequence[float] | None = None,
) -> list[CableGoal]:
    """Read a bundle's evidence into goals an evaluator can score against.

    Everything comes from the bundle directory: masks, and for a driven
    recording the stored drive trajectory. There is nothing external to
    resolve, so a bundle copied to another machine reads the same.

    Args:
        bundle: The :class:`~.evidence.CableEvidenceBundle` to read.
        bundle_dir: Directory the bundle's relative paths resolve against.
        attachment_transform: ``T_tcp_attachment`` as ``((x, y, z), (qx, qy, qz, qw))``,
            applied to every recording.
        clamp_position: Arc length [m] of the TCP grasp from the node-0 end,
            applied to every recording. 0.0 is the end clamp.
        cable_axis: Grasp direction ``(x, y, z)`` in the TCP (gripper) frame, or
            ``None`` to use the attachment rotation; applied to every recording.

    Returns:
        One :class:`~.goal.CableGoal` per recording, in bundle order.

    Raises:
        ValueError: If the bundle is malformed, a referenced file is missing, or
            a recording's masks differ in size from each other or from the
            images its camera describes.
    """
    bundle.validate()
    sources = {}

    goals = []
    for rec in bundle.recordings:
        source = None
        if rec.driven:
            source = sources.get(rec.trajectory_path)
            if source is None:
                path = os.path.join(bundle_dir, rec.trajectory_path)
                if not os.path.exists(path):
                    raise ValueError(
                        f"recording '{rec.label}': trajectory {rec.trajectory_path!r} is missing from the bundle."
                    )
                with open(path) as fh:
                    source = MaterializedTrajectorySource.from_dict(json.load(fh))
                source.validate()
                sources[rec.trajectory_path] = source

        masks = []
        for rel in rec.masks:
            path = os.path.join(bundle_dir, rel)
            if not os.path.exists(path):
                raise ValueError(f"recording '{rec.label}': mask {rel!r} is missing from the bundle.")
            masks.append(_load_mask(path))

        # The camera data describes the stored masks, so the scored region is the whole mask.
        height, width = masks[0].shape
        for rel, mask in zip(rec.masks, masks, strict=True):
            if mask.shape != (height, width):
                raise ValueError(
                    f"recording '{rec.label}': mask {rel!r} is {mask.shape[1]}x{mask.shape[0]}, "
                    f"but the first mask is {width}x{height}."
                )
        size = rec.camera_intrinsics[:2] if rec.camera_intrinsics is not None else rec.render_size
        if (int(size[0]), int(size[1])) != (width, height):
            raise ValueError(
                f"recording '{rec.label}': the masks are {width}x{height}, "
                f"but the camera describes {int(size[0])}x{int(size[1])} images."
            )
        goal = CableGoal(
            label=rec.label,
            recording_key=rec.recording_key,
            masks=masks,
            frame_times=rec.frame_timestamps,
            cable_start=tuple(rec.cable_start),
            sensor_pos=tuple(rec.sensor_pos),
            crop=[0, 0, width, height],
            sensor_quat=list(rec.sensor_quat) if rec.sensor_quat is not None else None,
            camera_intrinsics=(tuple(rec.camera_intrinsics) if rec.camera_intrinsics is not None else None),
            render_size=tuple(rec.render_size) if rec.render_size is not None else None,
            fov_deg=rec.fov_deg,
            start_ns=rec.start_ns,
            driven=rec.driven,
            data_source=source,
            attachment_transform=tuple(tuple(part) for part in attachment_transform),
            clamp_position=clamp_position,
            cable_axis=tuple(cable_axis) if cable_axis is not None else None,
        )
        goal.validate()
        goals.append(goal)
    return goals
