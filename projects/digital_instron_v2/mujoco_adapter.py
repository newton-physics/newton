# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo validation adapter for learned foundation wrenches."""

from __future__ import annotations

import numpy as np


def apply_foundation_wrench_to_body_f(body_f: np.ndarray, body_index: int, wrench: np.ndarray) -> None:
    """Apply a learned foundation wrench through Newton ``state.body_f`` storage."""

    if body_f.ndim != 2 or body_f.shape[1] != 6:
        raise ValueError("body_f must have shape (body_count, 6)")
    if body_index < 0 or body_index >= body_f.shape[0]:
        raise IndexError(f"body_index {body_index} is outside body_f with {body_f.shape[0]} bodies")
    values = np.asarray(wrench, dtype=body_f.dtype)
    if values.shape != (6,):
        raise ValueError("wrench must have shape (6,)")
    body_f[body_index, :] += values
