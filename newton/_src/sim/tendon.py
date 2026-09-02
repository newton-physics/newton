# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Massless routed tendons for cable-driven mechanisms.

Route geometry follows Müller et al., "Cable Joints" (SCA 2018), using rigid
body contact points such as pulleys, pinholes, and attachments.

Each tendon is an ordered sequence of waypoints on rigid bodies. Between
adjacent waypoints, a unilateral distance constraint enforces the tendon
length. Rolling links update the tangent geometry and can apply finite
capstan slip through their ``mu`` value; high ``mu`` recovers the no-slip
baseline. Pinholes are zero-radius waypoints on a rigid body and transfer rest
length between their adjacent spans subject to the same local capstan tension
ratio. ``mu=0`` gives a frictionless pinhole, while higher ``mu`` increasingly
resists slip through the point.
"""

from __future__ import annotations

from enum import IntEnum


class TendonLinkType(IntEnum):
    """Type of contact between a tendon and a rigid body."""

    ROLLING = 0
    """Cable wraps around the body surface. Attachment point moves to the
    tangent; rest length updated by arc length as the body rotates."""

    ATTACHMENT = 1
    """Cable is fixed to the body at a point. Neither attachment nor rest
    length changes."""

    PINHOLE = 2
    """Cable passes through a fixed point on the body.

    Attachment follows the body point, while rest length transfers between
    adjacent segments subject to ``mu`` and the local bend angle.
    """


class TendonLinkFlags(IntEnum):
    """Flags controlling tendon link routing."""

    DYNAMIC = 1 << 0
    """Link activity is updated from the current route geometry."""
