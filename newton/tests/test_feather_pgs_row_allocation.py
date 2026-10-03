# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatherPGS contact row allocation under capacity overflow.

Slot counters are shared per world and reserved with atomics. When a reservation does
not fit, the row count must still bound every accepted contact: a row that is built but
lies at or past the count is never solved, yet friction patch links can chain into it,
and the patch-chain walk in the solve kernels then never terminates.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import allocate_world_contact_slots, finalize_mf_constraint_counts

_UNBOUNDED = 2**31 - 1


def _allocate_overflowing_contacts(device, contact_count, row_capacity):
    """Route ``contact_count`` contacts into one world with ``row_capacity`` rows.

    Contacts alternate between penetrating (three rows: normal plus two friction) and
    separated within the gap (one speculative normal row). Mixed reservation sizes are
    what let a rolled-back counter re-admit a contact behind a rejected one.
    """
    contact_shape0 = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    contact_shape1 = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    points = np.zeros((contact_count, 3), dtype=np.float32)
    points[:, 2] = np.where(np.arange(contact_count) % 2 == 0, -0.01, 0.01)
    contact_point0 = wp.array(points, dtype=wp.vec3, device=device)
    contact_point1 = wp.zeros((contact_count,), dtype=wp.vec3, device=device)
    contact_normal = wp.array(
        np.tile([0.0, 0.0, -1.0], (contact_count, 1)).astype(np.float32), dtype=wp.vec3, device=device
    )
    contact_slot = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    contact_path = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    contact_world = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    slots_needed = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    counter = wp.zeros((1,), dtype=wp.int32, device=device)
    dropped = wp.zeros((1,), dtype=wp.int32, device=device)
    first_rejected = wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device)
    count = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_world_contact_slots,
        dim=contact_count,
        inputs=[
            wp.array([contact_count], dtype=wp.int32, device=device),
            contact_count,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            wp.zeros((contact_count,), dtype=wp.float32, device=device),
            wp.zeros((contact_count,), dtype=wp.float32, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.array([6], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.ones((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            0,
            1,
            1,
            0,
            0.0,
            0.0,
            0.0,
            row_capacity,
            row_capacity,
            row_capacity,
            1,
            0.0,
            0,
            1,
            FrictionPatches(),
        ],
        outputs=[
            contact_world,
            contact_slot,
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            contact_path,
            wp.zeros((1,), dtype=wp.int32, device=device),
            counter,
            wp.zeros((1,), dtype=wp.int32, device=device),
            slots_needed,
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            dropped,
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            first_rejected,
        ],
        device=device,
    )
    wp.launch(
        finalize_mf_constraint_counts,
        dim=1,
        inputs=[counter, row_capacity, 3, first_rejected],
        outputs=[count],
        device=device,
    )
    return (
        contact_slot.numpy(),
        slots_needed.numpy(),
        int(count.numpy()[0]),
        int(counter.numpy()[0]),
        int(dropped.numpy()[0]),
    )


@unittest.skipUnless(wp.is_cuda_available(), "the allocation race needs concurrent GPU reservations")
class TestFeatherPGSRowAllocation(unittest.TestCase):
    def test_overflow_keeps_every_accepted_contact_below_the_count(self):
        """No accepted contact may hold rows at or past the finalized count.

        Rolling the shared counter back for a rejected reservation raced with concurrent
        reservations: a contact accepted at slot s could see the counter end below s, so
        its rows were built but never solved while patch links still chained into them.
        With 3000 mixed one- and three-row contacts into 48 rows nearly every reservation
        is rejected, which makes the race easy to hit.
        """
        device = wp.get_device()
        worst_excess = 0
        for _ in range(40):
            slot, needed, count, counter, dropped = _allocate_overflowing_contacts(device, 3000, 48)
            accepted = slot >= 0
            self.assertLessEqual(count, 48)
            self.assertEqual(
                counter,
                int(needed[accepted].sum()) + dropped,
                "counter must equal accepted rows plus rejected reservations",
            )
            excess = int(np.max(slot[accepted] + needed[accepted])) - count if accepted.any() else 0
            worst_excess = max(worst_excess, excess)
            self.assertEqual(int(np.sum(needed[accepted])), count, "accepted rows must fill exactly [0, count)")
        self.assertLessEqual(worst_excess, 0, f"accepted rows reach {worst_excess} past the row count")

    def test_dense_capacity_keeps_room_for_same_articulation_contacts(self):
        """Propagation modes must not shrink the dense budget when contacts can still route there.

        Contacts between two links of one articulation stay on the dense family unless
        ``propagation_same_articulation_rows`` routes them; a 16-row internal reserve
        drops most of them every step.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
        j_root = builder.add_joint_revolute(parent=-1, child=root, axis=wp.vec3(0.0, 1.0, 0.0))
        link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        j_link = builder.add_joint_revolute(
            parent=root,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
        )
        builder.add_articulation([j_root, j_link], label="chain")
        model = builder.finalize()
        model.rigid_contact_max = 256
        dense_routed = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-colored",
            dense_max_constraints=320,
            mf_max_constraints=256,
        )
        self.assertEqual(int(dense_routed.dense_max_constraints), 320)
        propagation_routed = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-colored",
            dense_max_constraints=320,
            mf_max_constraints=256,
            propagation_same_articulation_rows=True,
        )
        self.assertLess(int(propagation_routed.dense_max_constraints), 320)


if __name__ == "__main__":
    unittest.main()
