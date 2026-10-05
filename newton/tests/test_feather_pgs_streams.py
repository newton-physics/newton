# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parallel size-group streams and double buffering of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _scene(device, worlds=2):
    """Two articulation sizes (a 2-link and a 3-link chain) and a box, falling onto the ground."""
    template = newton.ModelBuilder()
    for x, links in ((0.0, 2), (0.8, 3)):
        parent = -1
        joints = []
        for i in range(links):
            link = template.add_link(xform=wp.transform(wp.vec3(x + 0.25 * i, 0.0, 0.3), wp.quat_identity()))
            template.add_shape_capsule(
                link,
                radius=0.04,
                half_height=0.1,
                xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)),
            )
            joints.append(
                template.add_joint_revolute(
                    parent,
                    link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(
                        wp.vec3(x, 0.0, 0.3) if parent < 0 else wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()
                    ),
                    child_xform=wp.transform_identity(),
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )
            )
            parent = link
        template.add_articulation(joints)
    box = template.add_body(xform=wp.transform(wp.vec3(0.4, 0.4, 0.3), wp.quat_rpy(0.2, 0.1, 0.0)))
    template.add_shape_box(box, hx=0.06, hy=0.06, hz=0.06)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _run(model, steps=60, capture=False, **solver_kwargs):
    solver = SolverFeatherPGS(
        model, enable_joint_limits=True, friction_anchor_beta=0.0, dense_max_constraints=96, **solver_kwargs
    )
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()

    def substep():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(state_0, name), getattr(state_1, name))

    if capture:
        # A graph of two steps cycles both buffer sets.
        with wp.ScopedCapture(model.device) as scope:
            solver.seed_double_buffer_events()
            substep()
            substep()
        for _ in range(steps // 2):
            wp.capture_launch(scope.graph)
    else:
        for _ in range(steps):
            substep()
    solver.check_constraint_capacity()
    return solver, state_0.joint_q.numpy().copy()


def test_streams_and_double_buffer_step_identically(test, device, pgs_mode="matrix_free"):
    """Match the serial single-buffer trajectory bitwise with streams and double buffering."""
    model = _scene(device)
    reference_solver, reference = _run(model, pgs_mode=pgs_mode)
    test.assertEqual(reference_solver._size_streams, {})
    test.assertFalse(reference_solver._double_buffered)
    test.assertGreater(len(reference_solver.size_groups), 1)
    for options in (
        {"use_parallel_streams": True},
        {"double_buffer": True},
        {"use_parallel_streams": True, "double_buffer": True},
        {"use_parallel_streams": True, "double_buffer": True, "update_mass_matrix_interval": 3},
    ):
        with test.subTest(**options):
            solver, joint_q = _run(model, pgs_mode=pgs_mode, **options)
            test.assertEqual(bool(solver._size_streams), options.get("use_parallel_streams", False))
            test.assertEqual(solver._double_buffered, options.get("double_buffer", False))
            if solver._double_buffered:
                # Both buffer sets were cleared on the memset stream after their last use.
                wp.synchronize_device(device)
                used = 1 - solver._buffer_index
                test.assertIs(solver._J_bufs[used], solver.J_by_size)
                for index in (0, 1):
                    for size in solver.size_groups:
                        test.assertFalse(solver._J_bufs[index][size].numpy().any())
                        if "update_mass_matrix_interval" not in options:
                            test.assertFalse(solver._H_bufs[index][size].numpy().any())
            expected = reference
            if "update_mass_matrix_interval" in options:
                _, expected = _run(model, pgs_mode=pgs_mode, update_mass_matrix_interval=3)
            np.testing.assert_array_equal(joint_q, expected)


def test_captured_double_buffer_matches_eager(test, device, pgs_mode="matrix_free"):
    """Replay a two-step graph of the double-buffered solve with the eager result."""
    model = _scene(device)
    _, eager = _run(model, pgs_mode=pgs_mode, use_parallel_streams=True, double_buffer=True)
    _, captured = _run(model, pgs_mode=pgs_mode, use_parallel_streams=True, double_buffer=True, capture=True)
    np.testing.assert_array_equal(captured, eager)


def test_capture_without_seeding_raises(test, device):
    """Explain the missing seed instead of failing the capture on an event from outside it."""
    model = _scene(device)
    solver = SolverFeatherPGS(model, double_buffer=True, friction_anchor_beta=0.0, dense_max_constraints=96)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, 1.0 / 240.0)
    solver.step(state_1, state_0, control, None, 1.0 / 240.0)
    with test.assertRaisesRegex(RuntimeError, "seed_double_buffer_events"):
        with wp.ScopedCapture(device):
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
    # Eager steps after a capture do not wait on the capture's events.
    solver.step(state_0, state_1, control, None, 1.0 / 240.0)
    test.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())


def test_flags_are_ignored_where_they_do_not_apply(test, device):
    """Keep one stream and one buffer set on CPU, with sparse factors and with one size group."""
    if wp.get_device(device).is_cpu:
        solver = SolverFeatherPGS(_scene(device), pgs_mode="split", use_parallel_streams=True, double_buffer=True)
        test.assertEqual(solver._size_streams, {})
        test.assertFalse(solver._double_buffered)
        solver.seed_double_buffer_events()
        return
    builder = newton.ModelBuilder()
    root = builder.add_link()
    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
    joints = [builder.add_joint_revolute(-1, root, axis=newton.Axis.Z)]
    for side in (-1.0, 1.0):
        leg = builder.add_link()
        builder.add_shape_box(leg, hx=0.05, hy=0.05, hz=0.05)
        joints.append(
            builder.add_joint_revolute(
                root, leg, axis=newton.Axis.Y, parent_xform=wp.transform(wp.vec3(side * 0.2, 0, 0), wp.quat_identity())
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, friction_anchor_beta=0.0, use_parallel_streams=True, double_buffer=True)
    test.assertIsNotNone(solver._sparse_mass_matrix_size)
    test.assertFalse(solver._double_buffered)
    # One size group needs no extra streams.
    test.assertEqual(solver._size_streams, {})


class TestFeatherPGSStreams(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSStreams,
    "test_capture_without_seeding_raises",
    test_capture_without_seeding_raises,
    devices=cuda_devices,
)
for _fn in (test_streams_and_double_buffer_step_identically, test_captured_double_buffer_matches_eager):
    add_function_test(TestFeatherPGSStreams, _fn.__name__, _fn, devices=cuda_devices)
    add_function_test(TestFeatherPGSStreams, f"{_fn.__name__}_split", _fn, devices=cuda_devices, pgs_mode="split")
add_function_test(
    TestFeatherPGSStreams,
    "test_flags_are_ignored_where_they_do_not_apply",
    test_flags_are_ignored_where_they_do_not_apply,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
