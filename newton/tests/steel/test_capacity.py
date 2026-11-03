# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Row capacities, overflow reporting, watermarks and heterogeneous-world guards of SolverSteel."""

import sys
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.friction_patches import FrictionPatches
from newton._src.solvers.steel.kernels import allocate_world_contact_slots, finalize_mf_constraint_counts
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_cuda_test_devices, get_test_devices


def _box_on_ground(device, worlds=1):
    template = newton.ModelBuilder()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_capacity_failure_is_observable(test, device, solve_mode="matrix_free"):
    """Flag dropped free-body contact rows until reset, without optional telemetry."""
    model = _box_on_ground(device)
    solver = SolverSteel(model, solve_mode=solve_mode, free_body_constraint_max=1, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.check_constraint_capacity()
    test.assertTrue(solver.constraint_overflow.numpy()[0])
    test.assertGreater(int(solver._row_dropped_mf.numpy()[0]), 0)
    solver.reset(state_out)
    solver.check_constraint_capacity()


def test_dense_capacity_failure_is_world_local(test, device, solve_mode="matrix_free"):
    """Flag only the world whose dense rows overflow."""
    template = newton.ModelBuilder()
    link = template.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(link, hx=0.3, hy=0.3, hz=0.1)
    joint = template.add_joint_prismatic(-1, link, axis=newton.Axis.Z)
    template.add_articulation([joint])
    # The second world's slider starts high above the ground: no contacts there.
    lifted = newton.ModelBuilder()
    lifted_link = lifted.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()))
    lifted.add_shape_box(lifted_link, hx=0.3, hy=0.3, hz=0.1)
    lifted.add_articulation([lifted.add_joint_prismatic(-1, lifted_link, axis=newton.Axis.Z)])
    builder = newton.ModelBuilder()
    builder.add_world(template)
    builder.add_world(lifted)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = SolverSteel(model, solve_mode=solve_mode, articulation_constraint_max=3, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 1)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, False, False])
    test.assertLessEqual(int(solver._constraint_count.numpy()[0]), 3)


def test_overflow_warning_is_printed_once(test, device, solve_mode="matrix_free", response="immediate"):
    """Print one device-side warning per overflowing row family, inside this test."""
    if wp.get_device(device).is_cpu and sys.platform == "win32":
        test.skipTest("Warp CPU printf capture is unreliable on Windows")
    model = _box_on_ground(device, worlds=2)
    if response == "propagation":
        # The propagation rows hold every contact, in free_body_constraint_max + articulation_constraint_max rows.
        solver = SolverSteel(
            model, free_body_constraint_max=1, articulation_constraint_max=1, articulated_contact_response=response
        )
    else:
        solver = SolverSteel(model, solve_mode=solve_mode, free_body_constraint_max=1)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    step_output = []
    for _ in range(2):
        capture = StdOutCapture()
        capture.begin()
        try:
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        finally:
            # Flush the device printf into this capture, also when the step raises.
            try:
                wp.synchronize()
            finally:
                output = capture.end()
        step_output.append(
            [
                line
                for line in output.splitlines()
                if line.strip() and not (line.startswith("Module") and "load on device" in line)
            ]
        )
    family = "propagation" if response == "propagation" else "matrix-free"
    test.assertEqual(len(step_output[0]), 1, step_output[0])
    test.assertRegex(step_output[0][0], rf"^Warning: Steel {family} constraint-row overflow in world [01]:")
    test.assertEqual(step_output[1], [])
    expected = [0, 0, 0, 1] if response == "propagation" else [0, 1, 0, 0]
    np.testing.assert_array_equal(solver._row_overflow_warning_emitted.numpy(), expected)


class TestSteelCapacityStatus(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


add_function_test(
    TestSteelCapacityStatus,
    "test_capacity_failure_is_observable",
    test_capacity_failure_is_observable,
    devices=devices,
)


add_function_test(
    TestSteelCapacityStatus,
    "test_dense_capacity_failure_is_world_local",
    test_dense_capacity_failure_is_world_local,
    devices=devices,
)


# The warning tests check their own output: CheckOutput's capture would receive the device printf.
add_function_test(
    TestSteelCapacityStatus,
    "test_overflow_warning_is_printed_once",
    test_overflow_warning_is_printed_once,
    devices=devices,
    check_output=False,
)


for _name in (
    "test_capacity_failure_is_observable",
    "test_dense_capacity_failure_is_world_local",
    "test_overflow_warning_is_printed_once",
):
    add_function_test(
        TestSteelCapacityStatus,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        check_output=_name != "test_overflow_warning_is_printed_once",
        solve_mode="split",
    )


add_function_test(
    TestSteelCapacityStatus,
    "test_overflow_warning_is_printed_once_propagation",
    test_overflow_warning_is_printed_once,
    devices=devices,
    check_output=False,
    response="propagation",
)


def _driven_model(device, worlds=2):
    """One driven slider per world plus one global (world -1) free body, without gravity."""
    template = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_prismatic(
        -1, body, axis=newton.Axis.X, target_ke=10000.0, target_kd=0.0, armature=0.0, damping=0.0
    )
    template.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for _ in range(worlds):
        builder.add_world(template)
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    return builder.finalize(device=device)


def test_partial_reset_refreshes_only_selected_mass_factors(test, device, solve_mode="matrix_free"):
    """Refresh only the selected world's (or the global slot's) factors, eagerly and under graph replay."""
    model = _driven_model(device)
    solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
    state, output = model.state(), model.state()
    control = model.control()
    solver.step(state, output, control, None, 0.01)

    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(state, mask)
    solver.step(state, output, control, None, 0.01)
    np.testing.assert_array_equal(solver._mass_update_mask.numpy(), [1, 0, 0])
    if not wp.get_device(device).is_cuda:
        return

    first_mask = wp.empty_like(solver._mass_update_mask)
    with wp.ScopedCapture(device=device) as capture:
        solver.reset(state, mask)
        solver.step(state, output, control, None, 0.01)
        wp.copy(first_mask, solver._mass_update_mask)
        solver.step(state, output, control, None, 0.01)

    for selected, expected in (([False, True, False], [0, 1, 0]), ([False, False, True], [0, 0, 1])):
        with test.subTest(mask=selected):
            mask.assign(np.array(selected))
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(first_mask.numpy(), expected)
            # The request is consumed by the first step; the second reuses every factor.
            np.testing.assert_array_equal(solver._mass_update_mask.numpy(), [0, 0, 0])


def test_partial_warp_capacity_preserves_world_boundaries(test, device, pgs_kernel="tiled"):
    """Keep the dense split-mode solve's loads and stores inside each world's row storage.

    Capacities that are not a multiple of the warp width leave partial warps in the native
    kernel; an inactive world must stay untouched.
    """
    for capacity in (33, 35, 40):
        with test.subTest(capacity=capacity):
            with kernel_overrides({"pgs_kernel": pgs_kernel}):
                solver = SolverSteel(
                    _driven_model(device, worlds=3), solve_mode="split", articulation_constraint_max=capacity
                )
            test.assertEqual(
                solver._pgs_solve_tiled_row_kernel is not None, pgs_kernel == "tiled" and wp.get_device(device).is_cuda
            )
            initial = np.zeros((3, capacity), dtype=np.float32)
            initial[2] = 1234.0
            solver._constraint_count.assign(np.array([capacity, 1, 0], dtype=np.int32))
            solver._impulses.assign(initial)
            solver._diag.fill_(2.0)
            solver._C.assign(np.tile(2.0 * np.eye(capacity, dtype=np.float32), (3, 1, 1)))
            solver._rhs.fill_(-1.0)
            solver._row_type.zero_()
            solver._row_parent.zero_()
            solver._row_mu.zero_()
            solver._dense_pgs_solve(1)
            expected = initial.copy()
            expected[0] = 0.5
            expected[1, 0] = 0.5
            np.testing.assert_array_equal(solver._impulses.numpy(), expected)


def _cylinder_foot(builder, pos, num_cyl=7, radius=0.02, half_height=0.015):
    """Add one free body whose collision is ``num_cyl`` small cylinders in a row."""
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=1.0)
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            body, xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), radius=radius, half_height=half_height
        )
    return body


def test_unmatched_pipeline_clears_stale_contact_identity(test, device):
    """Cold-start a matched buffer when its current producer disables matching.

    Warm starting reads ``rigid_contact_match_index``; a stale index left by an earlier
    matching producer would seed impulses into unrelated contacts.
    """
    builder = newton.ModelBuilder()
    _cylinder_foot(builder, wp.vec3(0.0, 0.0, 0.015))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    matched = newton.CollisionPipeline(model, contact_matching="latest")
    unmatched = newton.CollisionPipeline(model, deterministic=True)
    state, contacts = model.state(), matched.contacts()
    matched.collide(state, contacts)
    matched.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertTrue(np.any(contacts.rigid_contact_match_index.numpy()[:count] >= 0))
    unmatched.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] == -1))


class TestSteelSafety(unittest.TestCase):
    pass


add_function_test(
    TestSteelSafety,
    "test_partial_reset_refreshes_only_selected_mass_factors",
    test_partial_reset_refreshes_only_selected_mass_factors,
    devices=get_cuda_test_devices(),
)


add_function_test(
    TestSteelSafety,
    "test_partial_reset_refreshes_only_selected_mass_factors_split",
    test_partial_reset_refreshes_only_selected_mass_factors,
    devices=get_test_devices(),
    solve_mode="split",
)


add_function_test(
    TestSteelSafety,
    "test_partial_warp_capacity_preserves_world_boundaries",
    test_partial_warp_capacity_preserves_world_boundaries,
    devices=get_cuda_test_devices(),
)


add_function_test(
    TestSteelSafety,
    "test_partial_warp_capacity_preserves_world_boundaries_scalar",
    test_partial_warp_capacity_preserves_world_boundaries,
    devices=get_test_devices(),
    pgs_kernel="loop",
)


add_function_test(
    TestSteelSafety,
    "test_unmatched_pipeline_clears_stale_contact_identity",
    test_unmatched_pipeline_clears_stale_contact_identity,
    devices=get_test_devices(),
)


_UNBOUNDED = 2**31 - 1


def _point_friction(device):
    """Return the disabled friction-patch view: every contact gets point friction rows."""
    patches = FrictionPatches()
    patches.enabled = 0
    patches.weight = wp.zeros(0, dtype=float, device=device)
    patches.next_contact = wp.zeros(0, dtype=int, device=device)
    patches.point_a = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.point_b = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.phi = wp.zeros(1, dtype=wp.vec2, device=device)
    return patches


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
            wp.zeros((capacity,), dtype=wp.vec3, device=device),
            wp.zeros((capacity,), dtype=wp.vec3, device=device),
            wp.zeros((capacity,), dtype=wp.vec3, device=device),
            wp.zeros((capacity,), dtype=wp.float32, device=device),
            wp.zeros((capacity,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.transform, device=device),
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
            8,
            8,
            8,
            wp.zeros((1,), dtype=wp.int32, device=device),
            float("inf"),
            float("inf"),
            _point_friction(device),
            0,
            wp.zeros((1,), dtype=wp.float32, device=device),
            0,
        ],
        outputs=[
            wp.zeros((capacity,), dtype=wp.int32, device=device),
            contact_slot,
            wp.zeros((capacity,), dtype=wp.int32, device=device),
            wp.zeros((capacity,), dtype=wp.int32, device=device),
            wp.zeros((capacity,), dtype=wp.int32, device=device),
            world_slot_counter,
            contact_path,
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
        ],
        device=device,
    )
    np.testing.assert_array_equal(contact_slot.numpy(), np.full(capacity, -1, dtype=np.int32))
    np.testing.assert_array_equal(contact_path.numpy(), np.full(capacity, -1, dtype=np.int32))
    test.assertEqual(int(world_slot_counter.numpy()[0]), 0)


def test_step_rejects_contacts_larger_than_scratch(test, device, solve_mode="matrix_free"):
    """Reject a contact buffer larger than the solver's contact scratch before it is read."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1
    solver = SolverSteel(model, solve_mode=solve_mode)
    contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=0, device=model.device)
    with test.assertRaisesRegex(ValueError, "contact capacity"):
        solver.step(model.state(), model.state(), model.control(), contacts, 1.0 / 60.0)


def test_overflowed_contact_count_invalidates_every_world(test, device, solve_mode="matrix_free"):
    """Flag every world when the narrow phase reports more contacts than the buffer holds."""
    template = newton.ModelBuilder()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, True, True])
    np.testing.assert_array_equal(solver._contact_path.numpy()[: contacts.rigid_contact_max], -1)


class TestSteelContactCapacity(unittest.TestCase):
    pass


add_function_test(
    TestSteelContactCapacity,
    "test_allocator_rejects_incomplete_contact_frame",
    test_allocator_rejects_incomplete_contact_frame,
    devices=get_test_devices(),
)


add_function_test(
    TestSteelContactCapacity,
    "test_step_rejects_contacts_larger_than_scratch",
    test_step_rejects_contacts_larger_than_scratch,
    devices=devices,
)


add_function_test(
    TestSteelContactCapacity,
    "test_overflowed_contact_count_invalidates_every_world",
    test_overflowed_contact_count_invalidates_every_world,
    devices=devices,
)


split_devices = get_test_devices()


for _name in (
    "test_step_rejects_contacts_larger_than_scratch",
    "test_overflowed_contact_count_invalidates_every_world",
):
    add_function_test(
        TestSteelContactCapacity, f"{_name}_split", globals()[_name], devices=split_devices, solve_mode="split"
    )


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
            wp.zeros((contact_count,), dtype=wp.vec3, device=device),
            wp.zeros((contact_count,), dtype=wp.vec3, device=device),
            wp.full((contact_count,), wp.vec3(0.0, 0.0, 1.0), dtype=wp.vec3, device=device),
            wp.zeros((contact_count,), dtype=wp.float32, device=device),
            wp.zeros((contact_count,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.transform, device=device),
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
            float("inf"),
            float("inf"),
            _point_friction(device),
            0,
            wp.zeros((1,), dtype=wp.float32, device=device),
            0,
        ],
        outputs=[
            wp.zeros((contact_count,), dtype=wp.int32, device=device),
            contact_slot,
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.zeros((contact_count,), dtype=wp.int32, device=device),
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


class TestSteelRowAllocation(unittest.TestCase):
    pass


add_function_test(
    TestSteelRowAllocation,
    "test_overflow_keeps_every_accepted_contact_below_the_count",
    test_overflow_keeps_every_accepted_contact_below_the_count,
    devices=get_test_devices(),
)


add_function_test(
    TestSteelRowAllocation,
    "test_overflow_keeps_every_accepted_contact_below_the_count_propagation",
    test_overflow_keeps_every_accepted_contact_below_the_count,
    devices=get_cuda_test_devices(),
    propagation=True,
)


_FAMILY_KEYS = ("high_water", "raw_high_water", "dropped_contact_rows_high_water", "overflow_excess_high_water")


def _scene(device):
    """Two worlds: a box on the ground next to a fixed-base slider resting on it, one world without contacts."""
    template = newton.ModelBuilder()
    box = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    link = template.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    template.add_articulation(
        [
            template.add_joint_prismatic(
                -1, link, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity())
            )
        ]
    )
    lifted = newton.ModelBuilder()
    lifted.add_shape_box(
        lifted.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())), hx=0.1, hy=0.1, hz=0.1
    )
    builder = newton.ModelBuilder()
    builder.add_world(template)
    builder.add_world(lifted)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _run(model, solver, steps):
    """Step and return the host-side per-step maxima of every row family and the contact count."""
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    observed = {"dense": 0, "dense_raw": 0, "mf": 0, "mf_raw": 0, "contact": 0}
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        observed["dense"] = max(observed["dense"], int(solver._constraint_count.numpy().max()))
        observed["dense_raw"] = max(observed["dense_raw"], int(solver._slot_counter.numpy().max()))
        observed["mf"] = max(observed["mf"], int(solver._mf_constraint_count.numpy().max()))
        observed["mf_raw"] = max(observed["mf_raw"], int(solver._mf_slot_counter.numpy().max()))
        observed["contact"] = max(observed["contact"], int(contacts.rigid_contact_count.numpy()[0]))
        if solver._propagation_active:
            observed["propagation"] = max(
                observed.get("propagation", 0), int(solver._propagation_constraint_count.numpy().max())
            )
    return observed, state_0


def test_watermarks_are_zero_when_disabled(test, device, solve_mode="matrix_free"):
    """Report zeros and allocate nothing without ``row_watermark``."""
    model = _scene(device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    _run(model, solver, 4)
    test.assertIsNone(solver._row_watermarks)
    marks = solver.constraint_row_watermarks()
    test.assertEqual(len(marks), 16)
    test.assertTrue(all(value == 0 for value in marks.values()))


def test_watermarks_track_the_largest_row_counts(test, device, solve_mode="matrix_free"):
    """Match the per-step maxima of the retained and requested rows and the contact count."""
    model = _scene(device)
    solver = SolverSteel(model, solve_mode=solve_mode, row_watermark=True, friction_anchor_beta=0.0)
    observed, state = _run(model, solver, 30)
    marks = solver.constraint_row_watermarks()
    test.assertGreater(observed["dense"], 0)
    test.assertGreater(observed["mf"], 0)
    test.assertEqual(marks["articulation_high_water"], observed["dense"])
    test.assertEqual(marks["articulation_raw_high_water"], observed["dense_raw"])
    test.assertEqual(marks["free_body_high_water"], observed["mf"])
    test.assertEqual(marks["free_body_raw_high_water"], observed["mf_raw"])
    test.assertEqual(marks["contact_high_water"], observed["contact"])
    for family in ("articulation", "free_body", "propagation"):
        with test.subTest(family=family):
            test.assertEqual(marks[f"{family}_dropped_contact_rows_high_water"], 0)
            test.assertEqual(marks[f"{family}_overflow_excess_high_water"], 0)
            test.assertEqual(marks[f"{family}_overflow_world_steps"], 0)
    # The marks survive a reset; they cover the whole run.
    solver.reset(state)
    test.assertEqual(solver.constraint_row_watermarks(), marks)


def test_watermarks_track_propagation_rows(test, device):
    """Track the propagation row family of the propagation response."""
    model = _scene(device)
    solver = SolverSteel(model, row_watermark=True, articulated_contact_response="propagation")
    observed, _ = _run(model, solver, 10)
    marks = solver.constraint_row_watermarks()
    test.assertGreater(observed["propagation"], 0)
    test.assertEqual(marks["propagation_high_water"], observed["propagation"])
    test.assertEqual(marks["propagation_overflow_world_steps"], 0)


def test_watermarks_record_capacity_overflow(test, device, solve_mode="matrix_free"):
    """Record the dropped rows, the excess over capacity and the overflowing world-steps."""
    model = _scene(device)
    capacity = 3
    solver = SolverSteel(
        model,
        solve_mode=solve_mode,
        row_watermark=True,
        friction_anchor_beta=0.0,
        free_body_constraint_max=capacity,
        warn_constraint_overflow=False,
    )
    steps = 5
    observed, _ = _run(model, solver, steps)
    marks = solver.constraint_row_watermarks()
    test.assertGreater(observed["mf_raw"], capacity)
    test.assertEqual(marks["free_body_high_water"], capacity)
    test.assertEqual(marks["free_body_raw_high_water"], observed["mf_raw"])
    test.assertEqual(marks["free_body_overflow_excess_high_water"], observed["mf_raw"] - capacity)
    test.assertGreater(marks["free_body_dropped_contact_rows_high_water"], 0)
    # Only the world with the resting box overflows, in every step.
    test.assertEqual(marks["free_body_overflow_world_steps"], steps)
    test.assertEqual(marks["articulation_overflow_world_steps"], 0)


def test_watermarks_accumulate_in_captured_steps(test, device):
    """Accumulate the same marks under CUDA graph replay as in eager steps."""
    results = []
    for capture in (False, True):
        model = _scene(device)
        solver = SolverSteel(model, row_watermark=True, friction_anchor_beta=0.0)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()

        def substep(
            state_0=state_0, state_1=state_1, contacts=contacts, solver=solver, control=control, pipeline=pipeline
        ):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            wp.copy(state_0.joint_q, state_1.joint_q)
            wp.copy(state_0.joint_qd, state_1.joint_qd)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        if capture:
            substep()
            with wp.ScopedCapture(device) as capture_scope:
                substep()
            for _ in range(9):
                wp.capture_launch(capture_scope.graph)
        else:
            for _ in range(11):
                substep()
        results.append(solver.constraint_row_watermarks())
    test.assertEqual(results[0], results[1])
    test.assertGreater(results[1]["articulation_high_water"], 0)


class TestSteelRowWatermark(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


for _fn in (
    test_watermarks_are_zero_when_disabled,
    test_watermarks_track_the_largest_row_counts,
    test_watermarks_record_capacity_overflow,
):
    add_function_test(TestSteelRowWatermark, _fn.__name__, _fn, devices=cuda_devices)
    add_function_test(
        TestSteelRowWatermark, f"{_fn.__name__}_split", _fn, devices=get_test_devices(), solve_mode="split"
    )


add_function_test(
    TestSteelRowWatermark,
    "test_watermarks_track_propagation_rows",
    test_watermarks_track_propagation_rows,
    devices=cuda_devices,
)


add_function_test(
    TestSteelRowWatermark,
    "test_watermarks_accumulate_in_captured_steps",
    test_watermarks_accumulate_in_captured_steps,
    devices=cuda_devices,
)


def _make_chain_world(n_links: int, root_height: float = 1.0, free_box: bool = False) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs), optionally with a free box."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for _ in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, root_height), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    if free_box:
        body = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder


def _build_model(world_link_counts: list[int], device, root_height: float = 1.0, free_box=None) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for world, n_links in enumerate(world_link_counts):
        scene.add_world(_make_chain_world(n_links, root_height, free_box is not None and free_box[world]))
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _final_joint_q(model, steps=60, solve_mode="matrix_free", response="immediate", from_fk=False, deterministic=False):
    solver = SolverSteel(
        model,
        solve_mode=solve_mode,
        articulation_constraint_max=64,
        articulated_contact_response=response,
        deterministic=wp.DeterministicMode.RUN_TO_RUN if deterministic else None,
    )
    pipeline = newton.CollisionPipeline(model, deterministic=deterministic)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    if from_fk:
        # Start from the joint-space pose instead of the builder's body placement.
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
    return state_0.joint_q.numpy()


def test_hetero_matrix_free_constructs(test, device):
    """Construct on worlds whose DOF counts differ."""
    solver = SolverSteel(_build_model([1, 3, 1, 3], device), solve_mode="matrix_free")
    np.testing.assert_array_equal(solver._world_dof_count.numpy(), [1, 3, 1, 3])


def test_homogeneous_matrix_free_constructs(test, device):
    """Construct on worlds with identical DOF counts."""
    solver = SolverSteel(_build_model([3, 3, 3, 3], device), solve_mode="matrix_free")
    np.testing.assert_array_equal(solver._world_dof_count.numpy(), [3, 3, 3, 3])


def test_hetero_split_constructs(test, device):
    """Construct the split solve on worlds whose DOF counts differ."""
    solver = SolverSteel(_build_model([1, 3, 1, 3], device), solve_mode="split")
    test.assertEqual(solver.solve_mode, "split")
    test.assertEqual(solver._size_groups, [3, 1])
    test.assertEqual(solver._C.shape, (4, solver.articulation_constraint_max, solver.articulation_constraint_max))


def test_homogeneous_split_constructs(test, device):
    """Construct the split solve on worlds with identical DOF counts."""
    solver = SolverSteel(_build_model([3, 3, 3, 3], device), solve_mode="split")
    test.assertEqual(solver.solve_mode, "split")
    test.assertEqual(solver._size_groups, [3])


def test_hetero_worlds_match_isolated_worlds(test, device, solve_mode="matrix_free"):
    """Simulate each world of a heterogeneous model exactly as it simulates alone."""
    combined = _final_joint_q(_build_model([1, 3], device), solve_mode=solve_mode)
    single = _final_joint_q(_build_model([1], device), solve_mode=solve_mode)
    triple = _final_joint_q(_build_model([3], device), solve_mode=solve_mode)
    np.testing.assert_allclose(combined, np.concatenate([single, triple]), rtol=0.0, atol=1.0e-5)


def test_homogeneous_propagation_constructs(test, device):
    """Construct both propagation responses on worlds with identical DOF counts."""
    model = _build_model([3, 3, 3, 3], device)
    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            solver = SolverSteel(model, articulated_contact_response=response)
            test.assertEqual(solver.articulated_contact_response, response)
            np.testing.assert_array_equal(solver._world_dof_count.numpy(), [3, 3, 3, 3])


def test_hetero_propagation_matches_isolated_worlds(test, device):
    """Simulate each world of a heterogeneous model with a propagation response exactly as it simulates alone.

    The chains hang low enough to hit the ground, so every world solves propagation contact rows.
    """

    def run(link_counts, free_box, response):
        model = _build_model(link_counts, device, 0.35, free_box)
        # Without a deterministic mode these chains can reach a different contact outcome from run to run.
        return _final_joint_q(model, response=response, from_fk=True, deterministic=True)

    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            combined = run([1, 3], None, response)
            single = run([1], None, response)
            triple = run([3], None, response)
            # The cached response sizes its reductions by the model's body capacity, so the sums reassociate.
            np.testing.assert_allclose(combined, np.concatenate([single, triple]), rtol=0.0, atol=1.0e-4)
            # One articulation size, with a free box that still makes the world DOF counts differ.
            combined = run([3, 3], [False, True], response)
            plain = run([3], [False], response)
            boxed = run([3], [True], response)
            np.testing.assert_allclose(combined, np.concatenate([plain, boxed]), rtol=0.0, atol=1.0e-4)


class TestSteelHeteroGuard(unittest.TestCase):
    pass


add_function_test(
    TestSteelHeteroGuard, "test_hetero_matrix_free_constructs", test_hetero_matrix_free_constructs, devices=devices
)


add_function_test(
    TestSteelHeteroGuard,
    "test_homogeneous_matrix_free_constructs",
    test_homogeneous_matrix_free_constructs,
    devices=devices,
)


add_function_test(
    TestSteelHeteroGuard,
    "test_hetero_worlds_match_isolated_worlds",
    test_hetero_worlds_match_isolated_worlds,
    devices=devices,
)


add_function_test(
    TestSteelHeteroGuard, "test_hetero_split_constructs", test_hetero_split_constructs, devices=split_devices
)


add_function_test(
    TestSteelHeteroGuard,
    "test_homogeneous_split_constructs",
    test_homogeneous_split_constructs,
    devices=split_devices,
)


add_function_test(
    TestSteelHeteroGuard,
    "test_hetero_worlds_match_isolated_worlds_split",
    test_hetero_worlds_match_isolated_worlds,
    devices=split_devices,
    solve_mode="split",
)


add_function_test(
    TestSteelHeteroGuard,
    "test_homogeneous_propagation_constructs",
    test_homogeneous_propagation_constructs,
    devices=devices,
)


add_function_test(
    TestSteelHeteroGuard,
    "test_hetero_propagation_matches_isolated_worlds",
    test_hetero_propagation_matches_isolated_worlds,
    devices=devices,
)


def test_propagation_constraint_max_sizes_the_propagation_family(test, device):
    """Size the propagation rows from ``propagation_constraint_max`` instead of the summed budgets."""
    model = _box_on_ground(device)
    options = {"solve_mode": "matrix_free", "articulated_contact_response": "propagation_colored"}
    summed = SolverSteel(model, articulation_constraint_max=32, free_body_constraint_max=256, **options)
    test.assertIsNone(summed.propagation_constraint_max)
    test.assertEqual(summed._propagation_max_constraints, 256 + 32)
    decoupled = SolverSteel(
        model, articulation_constraint_max=32, free_body_constraint_max=16, propagation_constraint_max=288, **options
    )
    test.assertEqual(decoupled.propagation_constraint_max, 288)
    test.assertEqual(decoupled._propagation_max_constraints, 288)
    test.assertEqual(decoupled.free_body_constraint_max, 16)
    with test.assertRaisesRegex(ValueError, "propagation_constraint_max must be >= 1"):
        SolverSteel(model, propagation_constraint_max=0, **options)
    with test.assertRaisesRegex(ValueError, "propagation_constraint_max requires"):
        SolverSteel(model, solve_mode="matrix_free", propagation_constraint_max=64)


def test_propagation_constraint_max_keeps_contact_rows_with_a_small_free_body_budget(test, device):
    """Keep every contact row when the propagation family is sized for them and the free-body budget is small."""
    model = _box_on_ground(device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)

    def step(response, free_body_constraint_max=1, **options):
        # A summed capacity of 1 + 1 rows is far below the rows of one resting box.
        solver = SolverSteel(
            model,
            solve_mode="matrix_free",
            articulated_contact_response=response,
            free_body_constraint_max=free_body_constraint_max,
            articulation_constraint_max=1,
            row_watermark=True,
            warn_constraint_overflow=False,
            **options,
        )
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        return solver.constraint_row_watermarks(), state_out.body_qd.numpy()

    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            summed, _ = step(response)
            test.assertGreater(summed["propagation_dropped_contact_rows_high_water"], 0)
            sized, sized_qd = step(response, propagation_constraint_max=16)
            test.assertEqual(sized["propagation_dropped_contact_rows_high_water"], 0)
            test.assertEqual(sized["propagation_high_water"], summed["propagation_raw_high_water"])
            # The retained rows reach the solve: it matches a summed budget large enough for them.
            _, ample_qd = step(response, free_body_constraint_max=15)
            np.testing.assert_allclose(sized_qd, ample_qd, atol=1e-6)


add_function_test(
    TestSteelRowAllocation,
    "test_propagation_constraint_max_sizes_the_propagation_family",
    test_propagation_constraint_max_sizes_the_propagation_family,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSteelRowAllocation,
    "test_propagation_constraint_max_keeps_contact_rows_with_a_small_free_body_budget",
    test_propagation_constraint_max_keeps_contact_rows_with_a_small_free_body_budget,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
