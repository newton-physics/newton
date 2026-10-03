# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS contact-capacity handling."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import allocate_world_contact_slots, collect_propagation_units
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    PROPAGATION_COLOR_TAIL,
    _get_color_propagation_prebuild_kernel,
)
from newton.solvers import SolverFeatherPGS


def _build_model():
    """Build one articulated body for contact-capacity tests."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    model = builder.finalize(device="cpu")
    model.rigid_contact_max = 1
    return model


class TestFeatherPGSContactCapacity(unittest.TestCase):
    def _assert_mixed_units_are_all_scheduled(
        self,
        *,
        contact_phi: tuple[float, ...],
        friction_gap_threshold: float,
        patch_friction: bool,
    ):
        """Build and color a row-capacity-filling mixture of contact units."""
        device = wp.get_device("cuda:0")
        contact_count = len(contact_phi)
        row_capacity = 6
        color_entries = contact_count + 2  # one color per unit, plus the tail and total entries
        order_stride = 8  # power of two >= row_capacity

        contact_count_array = wp.array([contact_count], dtype=wp.int32, device=device)
        contact_shape0 = wp.zeros((contact_count,), dtype=wp.int32, device=device)
        contact_shape1 = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
        contact_point0_np = np.zeros((contact_count, 3), dtype=np.float32)
        contact_point0_np[:, 2] = contact_phi
        contact_point0 = wp.array(contact_point0_np, dtype=wp.vec3, device=device)
        contact_point1 = wp.zeros((contact_count,), dtype=wp.vec3, device=device)
        contact_normal = wp.array(
            np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (contact_count, 1)),
            dtype=wp.vec3,
            device=device,
        )
        contact_slot = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
        contact_path = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
        contact_world = wp.zeros((contact_count,), dtype=wp.int32, device=device)
        contact_slots_needed = wp.zeros((contact_count,), dtype=wp.int32, device=device)
        propagation_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

        patches = FrictionPatches()
        if patch_friction:
            patches.enabled = 1
            patches.weight = wp.array([1.0] + [0.0] * (contact_count - 1), dtype=float, device=device)

        wp.launch(
            allocate_world_contact_slots,
            dim=contact_count,
            inputs=[
                contact_count_array,
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
                friction_gap_threshold,
                0,
                0,
                patches,
            ],
            outputs=[
                contact_world,
                contact_slot,
                wp.full((contact_count,), -1, dtype=wp.int32, device=device),
                wp.full((contact_count,), -1, dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                contact_path,
                wp.zeros((1,), dtype=wp.int32, device=device),
                propagation_slot_counter,
                wp.zeros((1,), dtype=wp.int32, device=device),
                contact_slots_needed,
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
            ],
            device=device,
        )

        unit_cursor = wp.zeros((1,), dtype=wp.int32, device=device)
        unit_contact = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        unit_body_a = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        unit_body_b = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        unit_len = wp.zeros((row_capacity,), dtype=wp.int32, device=device)
        wp.launch(
            collect_propagation_units,
            dim=contact_count,
            inputs=[
                contact_count_array,
                contact_path,
                contact_world,
                contact_shape0,
                contact_shape1,
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),  # body_prescribed: none here
                contact_slots_needed,
                row_capacity,
            ],
            outputs=[unit_cursor, unit_contact, unit_body_a, unit_body_b, unit_len],
            device=device,
        )

        row_order = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        color_offsets = wp.full((color_entries,), -1, dtype=wp.int32, device=device)
        sorted_contact = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        kernel = _get_color_propagation_prebuild_kernel(
            row_capacity,
            color_entries,
            64,
            order_stride,
            str(device.arch),
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                1,
                unit_cursor,
                unit_contact,
                unit_body_a,
                unit_body_b,
                unit_len,
                wp.zeros(((PROPAGATION_COLOR_TAIL + 31) // 32,), dtype=wp.int32, device=device),  # body 0 mask
                wp.zeros((order_stride,), dtype=wp.int32, device=device),
                wp.zeros((row_capacity,), dtype=wp.int32, device=device),
                contact_slot,
                row_order,
                color_offsets,
                sorted_contact,
            ],
            block_dim=64,
            device=device,
        )

        slots_needed_np = contact_slots_needed.numpy()
        self.assertEqual(sorted(slots_needed_np.tolist()), [1, 1, 1, 3])
        np.testing.assert_array_equal(contact_path.numpy(), np.full(contact_count, 2, dtype=np.int32))
        self.assertEqual(int(propagation_slot_counter.numpy()[0]), row_capacity)
        self.assertEqual(int(unit_cursor.numpy()[0]), contact_count)

        # Every unit is colored (no staging cap, so nothing spills into the tail). All
        # units share body 0, so each takes its own color, in contact-index order.
        offsets_np = color_offsets.numpy()
        self.assertEqual(int(offsets_np[-1]), contact_count)
        self.assertEqual(int(offsets_np[-1] - offsets_np[-2]), 0)
        self.assertEqual(offsets_np[: contact_count + 1].tolist(), list(range(contact_count + 1)))

        sorted_contact_np = sorted_contact.numpy()[:contact_count]
        self.assertEqual(sorted_contact_np.tolist(), list(range(contact_count)))
        contact_slot_np = contact_slot.numpy()
        row_order_np = row_order.numpy()
        row_acc = 0
        for position, contact in enumerate(sorted_contact_np):
            self.assertEqual(int(row_order_np[position]), row_acc)
            self.assertEqual(int(contact_slot_np[contact]), row_acc)
            row_acc += int(slots_needed_np[contact])
        self.assertEqual(row_acc, row_capacity)

    def test_allocator_rejects_incomplete_contact_frame(self):
        """Ignore a frame whose reported contact count exceeds its materialized capacity."""
        capacity = 2
        device = "cpu"
        contact_slot = wp.full((capacity,), -7, dtype=wp.int32, device=device)
        contact_path = wp.full((capacity,), -7, dtype=wp.int32, device=device)
        world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

        wp.launch(
            allocate_world_contact_slots,
            dim=capacity,
            inputs=[
                wp.array([capacity + 1], dtype=wp.int32, device=device),
                capacity,
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.full((capacity,), -1, dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.array([wp.vec3(0.0, 0.0, 1.0)] * capacity, dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([6], dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.ones((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                0,
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                8,
                8,
                8,
                0,
                0.0,
                0,
                0,
                FrictionPatches(),
            ],
            outputs=[
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                contact_slot,
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                world_slot_counter,
                contact_path,
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
            ],
            device=device,
        )

        np.testing.assert_array_equal(contact_slot.numpy(), np.full(capacity, -1, dtype=np.int32))
        np.testing.assert_array_equal(contact_path.numpy(), np.full(capacity, -1, dtype=np.int32))
        self.assertEqual(int(world_slot_counter.numpy()[0]), 0)

    def test_step_rejects_contacts_larger_than_scratch(self):
        """Reject oversized contact input before it can overwrite solver scratch."""
        model = _build_model()
        solver = SolverFeatherPGS(model)
        contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=0, device=model.device)

        with self.assertRaisesRegex(ValueError, "contact capacity"):
            solver.step(model.state(), model.state(), model.control(), contacts, 1.0 / 60.0)

    @unittest.skipUnless(wp.is_cuda_available(), "propagation-colored prebuild requires CUDA")
    def test_colored_prebuild_preserves_mixed_contact_units(self):
        """Schedule friction-gap and patch-selected one-row units alongside three-row units."""
        scenarios = {
            "friction_gap": ((-0.01, 0.01, 0.02, 0.03), 0.0, 0),
            "patch_friction": ((-0.01, -0.01, -0.01, -0.01), float("inf"), 1),
        }
        for name, (contact_phi, friction_gap_threshold, patch_friction) in scenarios.items():
            with self.subTest(name=name):
                self._assert_mixed_units_are_all_scheduled(
                    contact_phi=contact_phi,
                    friction_gap_threshold=friction_gap_threshold,
                    patch_friction=patch_friction,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
