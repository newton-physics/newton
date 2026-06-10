# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp


def set_viewer_camera(
    viewer,
    *,
    pos: wp.vec3,
    target: wp.vec3,
    fov: float = 32.0,
    show_joints: bool | None = None,
    joint_scale: float | None = None,
) -> None:
    """Set an example camera and optional joint-axis visualization."""
    if show_joints is not None:
        viewer.show_joints = show_joints

    if hasattr(viewer, "set_camera"):
        viewer.set_camera(pos=pos, pitch=0.0, yaw=0.0)
        if hasattr(viewer, "camera"):
            viewer.camera.look_at(target)
            viewer.camera.fov = fov

    if joint_scale is not None and hasattr(viewer, "renderer"):
        viewer.renderer.joint_scale = joint_scale
