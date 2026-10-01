# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-capacity handling of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import allocate_world_contact_slots
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_UNBOUNDED = 2**31 - 1


def test_allocator_rejects_incomplete_contact_frame(test, device):
    """Route no contact of a frame whose reported count exceeds its materialized capacity."""
    capacity = 2
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
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([6], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.ones((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            0,
            8,
            8,
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
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    np.testing.assert_array_equal(contact_slot.numpy(), np.full(capacity, -1, dtype=np.int32))
    np.testing.assert_array_equal(contact_path.numpy(), np.full(capacity, -1, dtype=np.int32))
    test.assertEqual(int(world_slot_counter.numpy()[0]), 0)


def test_step_rejects_contacts_larger_than_scratch(test, device):
    """Reject a contact buffer larger than the solver's contact scratch before it is read."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1
    solver = SolverFeatherPGS(model)
    contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=0, device=model.device)
    with test.assertRaisesRegex(ValueError, "contact capacity"):
        solver.step(model.state(), model.state(), model.control(), contacts, 1.0 / 60.0)


def test_overflowed_contact_count_invalidates_every_world(test, device):
    """Flag every world when the narrow phase reports more contacts than the buffer holds."""
    template = newton.ModelBuilder()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, True])
    np.testing.assert_array_equal(solver.contact_path.numpy()[: contacts.rigid_contact_max], -1)


class TestFeatherPGSContactCapacity(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSContactCapacity,
    "test_allocator_rejects_incomplete_contact_frame",
    test_allocator_rejects_incomplete_contact_frame,
    devices=devices,
)
add_function_test(
    TestFeatherPGSContactCapacity,
    "test_step_rejects_contacts_larger_than_scratch",
    test_step_rejects_contacts_larger_than_scratch,
    devices=devices,
)
add_function_test(
    TestFeatherPGSContactCapacity,
    "test_overflowed_contact_count_invalidates_every_world",
    test_overflowed_contact_count_invalidates_every_world,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
