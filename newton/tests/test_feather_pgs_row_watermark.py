# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Row high-water marks of SolverFeatherPGS (``row_watermark``)."""

import unittest

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

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
        observed["dense"] = max(observed["dense"], int(solver.constraint_count.numpy().max()))
        observed["dense_raw"] = max(observed["dense_raw"], int(solver.slot_counter.numpy().max()))
        observed["mf"] = max(observed["mf"], int(solver.mf_constraint_count.numpy().max()))
        observed["mf_raw"] = max(observed["mf_raw"], int(solver.mf_slot_counter.numpy().max()))
        observed["contact"] = max(observed["contact"], int(contacts.rigid_contact_count.numpy()[0]))
        if solver._propagation_active:
            observed["propagation"] = max(
                observed.get("propagation", 0), int(solver.propagation_constraint_count.numpy().max())
            )
    return observed, state_0


def test_watermarks_are_zero_when_disabled(test, device, pgs_mode="matrix_free"):
    """Report zeros and allocate nothing without ``row_watermark``."""
    model = _scene(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    _run(model, solver, 4)
    test.assertIsNone(solver._row_watermarks)
    marks = solver.constraint_row_watermarks()
    test.assertEqual(len(marks), 16)
    test.assertTrue(all(value == 0 for value in marks.values()))


def test_watermarks_track_the_largest_row_counts(test, device, pgs_mode="matrix_free"):
    """Match the per-step maxima of the retained and requested rows and the contact count."""
    model = _scene(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, row_watermark=True, friction_anchor_beta=0.0)
    observed, state = _run(model, solver, 30)
    marks = solver.constraint_row_watermarks()
    test.assertGreater(observed["dense"], 0)
    test.assertGreater(observed["mf"], 0)
    test.assertEqual(marks["dense_high_water"], observed["dense"])
    test.assertEqual(marks["dense_raw_high_water"], observed["dense_raw"])
    test.assertEqual(marks["mf_high_water"], observed["mf"])
    test.assertEqual(marks["mf_raw_high_water"], observed["mf_raw"])
    test.assertEqual(marks["contact_high_water"], observed["contact"])
    for family in ("dense", "mf", "propagation"):
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
    solver = SolverFeatherPGS(model, row_watermark=True, articulated_contact_response="propagation")
    observed, _ = _run(model, solver, 10)
    marks = solver.constraint_row_watermarks()
    test.assertGreater(observed["propagation"], 0)
    test.assertEqual(marks["propagation_high_water"], observed["propagation"])
    test.assertEqual(marks["propagation_overflow_world_steps"], 0)


def test_watermarks_record_capacity_overflow(test, device, pgs_mode="matrix_free"):
    """Record the dropped rows, the excess over capacity and the overflowing world-steps."""
    model = _scene(device)
    capacity = 3
    solver = SolverFeatherPGS(
        model,
        pgs_mode=pgs_mode,
        row_watermark=True,
        friction_anchor_beta=0.0,
        mf_max_constraints=capacity,
        warn_constraint_overflow=False,
    )
    steps = 5
    observed, _ = _run(model, solver, steps)
    marks = solver.constraint_row_watermarks()
    test.assertGreater(observed["mf_raw"], capacity)
    test.assertEqual(marks["mf_high_water"], capacity)
    test.assertEqual(marks["mf_raw_high_water"], observed["mf_raw"])
    test.assertEqual(marks["mf_overflow_excess_high_water"], observed["mf_raw"] - capacity)
    test.assertGreater(marks["mf_dropped_contact_rows_high_water"], 0)
    # Only the world with the resting box overflows, in every step.
    test.assertEqual(marks["mf_overflow_world_steps"], steps)
    test.assertEqual(marks["dense_overflow_world_steps"], 0)


def test_watermarks_accumulate_in_captured_steps(test, device):
    """Accumulate the same marks under CUDA graph replay as in eager steps."""
    results = []
    for capture in (False, True):
        model = _scene(device)
        solver = SolverFeatherPGS(model, row_watermark=True, friction_anchor_beta=0.0)
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
    test.assertGreater(results[1]["dense_high_water"], 0)


class TestFeatherPGSRowWatermark(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _fn in (
    test_watermarks_are_zero_when_disabled,
    test_watermarks_track_the_largest_row_counts,
    test_watermarks_record_capacity_overflow,
):
    add_function_test(TestFeatherPGSRowWatermark, _fn.__name__, _fn, devices=cuda_devices)
    add_function_test(
        TestFeatherPGSRowWatermark, f"{_fn.__name__}_split", _fn, devices=get_test_devices(), pgs_mode="split"
    )
add_function_test(
    TestFeatherPGSRowWatermark,
    "test_watermarks_track_propagation_rows",
    test_watermarks_track_propagation_rows,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSRowWatermark,
    "test_watermarks_accumulate_in_captured_steps",
    test_watermarks_accumulate_in_captured_steps,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
