# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structured selection views for model entities.

Use :class:`ArticulationView` for tree-specific roots, masks, and reduced-coordinate
operators. Use :class:`JointView` or :class:`BodyView` for topology-independent
attribute access, including closed-loop mechanisms without articulations.
"""

from ._src.utils.selection import ArticulationView, BodyView, JointView

__all__ = [
    "ArticulationView",
    "BodyView",
    "JointView",
]
