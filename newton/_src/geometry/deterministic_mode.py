# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolve the ``deterministic`` argument of the collision classes to a Warp mode."""

from __future__ import annotations

import warnings

import warp as wp

from ..exceptions import NewtonWarning
from ..utils.deprecation import warn_deprecation


def resolve_deterministic_mode(deterministic: wp.DeterministicMode | bool | None, owner: str) -> wp.DeterministicMode:
    """Return the determinism mode a collision class runs.

    ``None`` inherits ``wp.config.deterministic``. A ``bool`` maps to
    ``RUN_TO_RUN`` or ``NOT_GUARANTEED`` with a deprecation warning. Collision
    detection supports at most ``RUN_TO_RUN``: an explicit ``GPU_TO_GPU``
    raises, and an inherited one warns and runs ``RUN_TO_RUN``.

    Args:
        deterministic: The mode the caller passed.
        owner: Class name used in warning and error messages.

    Returns:
        ``NOT_GUARANTEED`` or ``RUN_TO_RUN``.
    """
    # DeterministicMode is an IntEnum, so check bool first: RUN_TO_RUN == True.
    if isinstance(deterministic, bool):
        mode = wp.DeterministicMode.RUN_TO_RUN if deterministic else wp.DeterministicMode.NOT_GUARANTEED
        warn_deprecation(
            f"{owner}(deterministic={deterministic}) is deprecated in Newton 1.9; pass "
            f"deterministic=wp.DeterministicMode.{mode.name} instead, or omit it to inherit "
            "wp.config.deterministic."
        )
        return mode
    if deterministic is None:
        mode = wp.config.deterministic
        if not isinstance(mode, wp.DeterministicMode):
            raise TypeError(f"wp.config.deterministic must be a warp.DeterministicMode, got {mode!r}")
        if mode == wp.DeterministicMode.GPU_TO_GPU:
            warnings.warn(
                f"wp.config.deterministic is GPU_TO_GPU; {owner} supports at most RUN_TO_RUN and runs that mode. "
                "Pass deterministic explicitly to select a mode without this warning.",
                NewtonWarning,
                stacklevel=3,
            )
            return wp.DeterministicMode.RUN_TO_RUN
        return mode
    if not isinstance(deterministic, wp.DeterministicMode):
        raise TypeError(f"deterministic must be a warp.DeterministicMode or None, got {deterministic!r}")
    if deterministic == wp.DeterministicMode.GPU_TO_GPU:
        raise ValueError(f"{owner} supports deterministic=RUN_TO_RUN or NOT_GUARANTEED, not GPU_TO_GPU")
    return deterministic
