# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera declarations stored on :class:`~newton.Model`.

A model camera describes what an asset authors about a camera: the frame it
is mounted in, the world it belongs to, its lens model and intrinsics, and its
clip range. Resolution, ray bundles and output images belong to the sensor that
renders the camera (see :class:`~newton.sensors.SensorCamera`).
"""

from __future__ import annotations

from enum import IntEnum


class CameraLens(IntEnum):
    """Lens model of a model camera.

    .. experimental::
    """

    PINHOLE = 0
    """Ideal pinhole projection defined by the camera's focal length, aperture
    and aperture offset."""

    CUSTOM = 1
    """No built-in projection. The rendering sensor supplies the rays, either as
    a ray bundle or as a ray function; the camera's intrinsics are ignored."""


CAMERA_DEFAULT_VERTICAL_APERTURE = 0.024
"""Vertical aperture [m] used when a camera is declared from a field of view."""
