# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tendon system for cable-driven mechanisms.

Implements massless cable routing through rigid body contact points (pulleys,
pinholes, attachments) using the Cable Joints method [Müller et al. SCA 2018].

Each tendon is an ordered sequence of waypoints on rigid bodies. Between
adjacent waypoints, a unilateral distance constraint enforces the cable
length. Rolling links use the Cable Joints tangent update and can apply finite
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
