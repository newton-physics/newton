# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect current soft-contact force selection in tests."""

import warp as wp

from newton import Contacts, Model, State
from newton._src.geometry.soft_contacts_mesh import (
    _make_soft_contact_filter_data,
    _soft_contact_force_enabled,
    _SoftContactFilterData,
)


@wp.kernel(enable_backward=False)
def _evaluate_force_mask(data: _SoftContactFilterData, count: wp.array[int], force_mask: wp.array[bool]):
    tid = wp.tid()
    force_mask[tid] = tid < count[0] and _soft_contact_force_enabled(tid, data)


def evaluate_soft_contact_force_mask(
    model: Model, state: State, contacts: Contacts, *, out: wp.array[bool] | None = None
) -> wp.array[bool]:
    """Return a test-owned eligibility snapshot; pass preallocated output for graph replay."""
    if out is None:
        out = wp.empty(contacts.soft_contact_max, dtype=bool, device=model.device)
    if contacts.soft_contact_max > 0:
        wp.launch(
            _evaluate_force_mask,
            dim=contacts.soft_contact_max,
            inputs=[
                _make_soft_contact_filter_data(model, contacts, state.particle_q, state.body_q),
                contacts.soft_contact_count,
            ],
            outputs=[out],
            device=model.device,
            record_tape=False,
        )
    return out
