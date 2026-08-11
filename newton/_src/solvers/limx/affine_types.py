# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp types for affine generalized coordinates."""

import warp as wp


class vec12(wp.types.vector(length=12, dtype=wp.float32)):
    """Twelve-component affine generalized vector."""


class mat1212(wp.types.matrix(shape=(12, 12), dtype=wp.float32)):
    """Twelve-by-twelve affine generalized matrix."""
