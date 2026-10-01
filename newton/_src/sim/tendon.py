# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Massless routed tendons for cable-driven mechanisms.

Route geometry follows Müller et al., "Cable Joints" (SCA 2018), using rigid
body contact points such as pulleys, pinholes, and attachments.

Each tendon is an ordered sequence of waypoints on rigid bodies. Between
adjacent waypoints, a unilateral distance constraint enforces the tendon
length. Roller guides update the tangent geometry and can apply finite
capstan slip through their ``mu`` value; high ``mu`` recovers the no-slip
baseline. Pinholes are zero-radius waypoints on a rigid body and transfer rest
length between their adjacent spans subject to the same local capstan tension
ratio. ``mu=0`` gives a frictionless pinhole, while higher ``mu`` increasingly
resists slip through the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ..core.types import Vec3


class TendonGuideType(IntEnum):
    """Guide type in a massless tendon route.

    .. experimental::
    """

    ROLLER = 0
    """Cable wraps around the body surface. Attachment point moves to the
    tangent; rest length updated by arc length as the body rotates."""

    ANCHOR = 1
    """Cable is fixed to a body-local point. Material cannot pass through it."""

    PINHOLE = 2
    """Cable passes through a fixed point on the body.

    Attachment follows the body point, while rest length transfers between
    adjacent segments subject to ``mu`` and the local bend angle.
    """


class TendonGuideFlags(IntEnum):
    """Flags controlling tendon-guide routing.

    .. experimental::
    """

    DYNAMIC = 1 << 0
    """Guide activity is updated from the current route geometry."""


@dataclass(frozen=True, kw_only=True)
class TendonGuide:
    """Describe one point or circular guide in an experimental massless tendon route.

    Pass a complete sequence to :meth:`newton.ModelBuilder.add_tendon`. Geometry
    is body-local; the builder copies it and normalizes the axis. Material
    parameters describe the incoming straight span and are ignored for the
    first guide.

    .. experimental::
    """

    body: int
    """Index of the attached rigid body; world-body index ``-1`` is not supported."""
    guide_type: TendonGuideType = TendonGuideType.ANCHOR
    """Anchor, pinhole, or circular roller."""
    radius: float = 0.0
    """Contact radius [m]; must be positive for rollers."""
    orientation: int = 1
    """Winding side, either ``1`` or ``-1``."""
    mu: float = 0.0
    """Nonnegative capstan friction coefficient [dimensionless]."""
    dynamic: bool = False
    """Whether an internal roller can engage and disengage."""
    offset: Vec3 = (0.0, 0.0, 0.0)
    """Anchor point or roller center in the body's local frame [m]."""
    axis: Vec3 = (0.0, 0.0, 1.0)
    """Nonzero cable-plane normal in the body's local frame [dimensionless]."""
    compliance: float = 0.0
    """Incoming span compliance [m/N]."""
    damping: float = 0.0
    """Incoming span damping coefficient [N·s/m]."""
    rest_length: float = -1.0
    """Incoming span rest length [m]; negative measures it from initial body poses."""
