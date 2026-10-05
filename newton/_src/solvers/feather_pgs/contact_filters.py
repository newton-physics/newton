# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share contact eligibility between patch selection and contact row allocation."""

import warp as wp


@wp.func
def contact_normal_gap_limit(
    a_non_free: bool,
    b_non_free: bool,
    same_articulation: bool,
    contact_gap: float,
    articulation_pair_gap: float,
    same_articulation_gap: float,
):
    limit = float(wp.inf)
    if contact_gap > 0.0:
        limit = contact_gap
    if a_non_free and b_non_free:
        if articulation_pair_gap > 0.0:
            limit = wp.min(limit, articulation_pair_gap)
        if same_articulation and same_articulation_gap > 0.0:
            limit = wp.min(limit, same_articulation_gap)
    return limit


@wp.func
def contact_friction_eligible(
    gap: float, a_non_free: bool, b_non_free: bool, articulation_pairs_only: int, friction_gap: float
):
    return (articulation_pairs_only != 0 and not (a_non_free and b_non_free)) or gap <= friction_gap
