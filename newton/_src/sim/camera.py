# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera declarations stored on :class:`~newton.Model`.

A model camera describes what an asset authors about a camera: the frame it
is mounted in, the world it belongs to, its lens type and intrinsics, its clip
range, and the nominal resolution its aperture maps onto. Ray bundles and output
images belong to the sensor that renders the camera (see
:class:`~newton.sensors.SensorCamera`), which may render at another size.
"""

from __future__ import annotations

from enum import IntEnum


class CameraLensType(IntEnum):
    """Lens type of a model camera."""

    PINHOLE = 0
    """Ideal pinhole projection defined by the camera's focal length, aperture
    and aperture offset."""

    CUSTOM = 1
    """No built-in projection. The rendering sensor supplies the rays as a ray
    bundle; the camera's intrinsics are ignored."""


CAMERA_DEFAULT_VERTICAL_APERTURE = 0.024
"""Vertical aperture [m] used when a camera is declared from a field of view."""
