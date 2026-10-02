# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact row allocation of SolverFeatherPGS under capacity overflow.

Slot counters are shared per world and reserved with atomics. When a reservation does not
fit, the finalized row count must still bound every accepted contact: a row that is built
but lies at or past the count is never solved.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import allocate_world_contact_slots, finalize_mf_constraint_counts
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_UNBOUNDED = 2**31 - 1


def _allocate_overflowing_contacts(device, contact_count, row_capacity, propagation=False):
    """Route ``contact_count`` free-body contacts into one world with ``row_capacity`` rows.

    With ``propagation`` the contacts go to the propagation rows of the propagation response.
    """
    contact_slot = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    contact_path = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    counters = [wp.zeros((1,), dtype=wp.int32, device=device) for _ in range(3)]
    dropped = [wp.zeros((1,), dtype=wp.int32, device=device) for _ in range(3)]
    first_rejected = [wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device) for _ in range(3)]
    family = 2 if propagation else 1
    count = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_world_contact_slots,
        dim=contact_count,
        inputs=[
            wp.array([contact_count], dtype=wp.int32, device=device),
            contact_count,
            wp.zeros((contact_count,), dtype=wp.int32, device=device),
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.array([6], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.ones((1,), dtype=wp.int32, device=device),
            wp.ones((1,), dtype=wp.int32, device=device),
            1,
            int(propagation),
            0,
            int(propagation),
            row_capacity,
            row_capacity,
            row_capacity,
            wp.zeros((1,), dtype=wp.int32, device=device),
        ],
        outputs=[
            wp.zeros((contact_count,), dtype=wp.int32, device=device),
            contact_slot,
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            counters[0],
            contact_path,
            counters[1],
            counters[2],
            wp.zeros((1,), dtype=wp.int32, device=device),
            dropped[0],
            dropped[1],
            dropped[2],
            first_rejected[0],
            first_rejected[1],
            first_rejected[2],
            wp.zeros((2,), dtype=wp.int32, device=device),
        ],
        device=device,
    )
    wp.launch(
        finalize_mf_constraint_counts,
        dim=1,
        inputs=[counters[family], row_capacity, 3, first_rejected[family]],
        outputs=[count],
        device=device,
    )
    return (
        contact_slot.numpy(),
        contact_path.numpy(),
        int(count.numpy()[0]),
        int(counters[family].numpy()[0]),
        int(dropped[family].numpy()[0]),
    )


def test_overflow_keeps_every_accepted_contact_below_the_count(test, device, propagation=False):
    """Keep every accepted contact's rows inside the finalized count while thousands overflow."""
    row_capacity = 48
    for _ in range(20):
        slot, path, count, counter, dropped = _allocate_overflowing_contacts(device, 3000, row_capacity, propagation)
        accepted = slot >= 0
        test.assertTrue(np.all(path[accepted] == (2 if propagation else 1)))
        test.assertTrue(np.all(path[~accepted] == -1))
        test.assertEqual(count, 3 * int(accepted.sum()))
        test.assertLessEqual(count, row_capacity)
        test.assertEqual(counter, 3 * int(accepted.sum()) + dropped)
        test.assertTrue(np.all(slot[accepted] + 3 <= count))
        np.testing.assert_array_equal(np.sort(slot[accepted]), np.arange(0, count, 3))


class TestFeatherPGSRowAllocation(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSRowAllocation,
    "test_overflow_keeps_every_accepted_contact_below_the_count",
    test_overflow_keeps_every_accepted_contact_below_the_count,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRowAllocation,
    "test_overflow_keeps_every_accepted_contact_below_the_count_propagation",
    test_overflow_keeps_every_accepted_contact_below_the_count,
    devices=get_cuda_test_devices(),
    propagation=True,
)


if __name__ == "__main__":
    unittest.main()
