# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q using the [x, y, z, w] convention."""
    x, y, z, w = q
    u = np.asarray([x, y, z], dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def node_xyz(body_q_row: np.ndarray, segment_length: float) -> np.ndarray:
    """Recover the start-node world position from a COM-centered capsule body state.

    With ``body_frame_origin="com"`` the body origin sits at the capsule midpoint.
    The start node is half a segment length behind that origin along local +Z.
    """
    p = np.asarray(body_q_row[:3], dtype=np.float64)
    q = np.asarray(body_q_row[3:7], dtype=np.float64)
    return p - _quat_rotate(q, np.array([0.0, 0.0, 0.5 * segment_length], dtype=np.float64))


def com_from_node(node: np.ndarray, q: np.ndarray, segment_length: float) -> np.ndarray:
    """Compute the capsule COM position from a start-node point and orientation.

    Inverse of :func:`node_xyz`: given the start-node world position and the
    body orientation quaternion, returns the COM-centered body origin.
    """
    return np.asarray(node, dtype=np.float64) + _quat_rotate(
        np.asarray(q, dtype=np.float64),
        np.array([0.0, 0.0, 0.5 * segment_length], dtype=np.float64),
    )


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
    if show_joints is not None and hasattr(viewer, "show_joints"):
        viewer.show_joints = show_joints

    if hasattr(viewer, "set_camera"):
        viewer.set_camera(pos=pos, pitch=0.0, yaw=0.0)
        if hasattr(viewer, "camera"):
            viewer.camera.look_at(target)
            viewer.camera.fov = fov

    if joint_scale is not None and hasattr(viewer, "renderer"):
        viewer.renderer.joint_scale = joint_scale
