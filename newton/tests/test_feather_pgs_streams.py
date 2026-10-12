# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parallel size-group streams and double buffering of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_connect import _build_four_bar
from newton.tests.test_feather_pgs_contact_torsion import set_torsion_radius
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
        model, enable_joint_limits=True, friction_anchor_beta=0.0, articulation_constraint_max=96, **solver_kwargs
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
            substep()
            substep()
        for _ in range(steps // 2):
            wp.capture_launch(scope.graph)
    else:
        for _ in range(steps):
            substep()
    solver.check_constraint_capacity()
    return solver, state_0.joint_q.numpy().copy()


def test_streams_and_double_buffer_step_identically(test, device, solve_mode="matrix_free"):
    """Match the serial single-buffer trajectory bitwise with streams and double buffering."""
    model = _scene(device)
    reference_solver, reference = _run(model, solve_mode=solve_mode)
    test.assertEqual(reference_solver._size_streams, {})
    test.assertFalse(reference_solver._double_buffered)
    test.assertGreater(len(reference_solver._size_groups), 1)
    for options in (
        {"use_parallel_streams": True},
        {"double_buffer": True},
        {"use_parallel_streams": True, "double_buffer": True},
        {"use_parallel_streams": True, "double_buffer": True, "update_mass_matrix_interval": 3},
    ):
        with test.subTest(**options):
            solver, joint_q = _run(model, solve_mode=solve_mode, **options)
            test.assertEqual(bool(solver._size_streams), options.get("use_parallel_streams", False))
            test.assertEqual(solver._double_buffered, options.get("double_buffer", False))
            if solver._double_buffered:
                # Both buffer sets were cleared on the memset stream after their last use.
                wp.synchronize_device(device)
                used = 1 - solver._buffer_index
                test.assertIs(solver._J_bufs[used], solver._J_by_size)
                for index in (0, 1):
                    for size in solver._size_groups:
                        test.assertFalse(solver._J_bufs[index][size].numpy().any())
                        if "update_mass_matrix_interval" not in options:
                            test.assertFalse(solver._H_bufs[index][size].numpy().any())
            expected = reference
            if "update_mass_matrix_interval" in options:
                _, expected = _run(model, solve_mode=solve_mode, update_mass_matrix_interval=3)
            np.testing.assert_array_equal(joint_q, expected)


def test_captured_double_buffer_matches_eager(test, device, solve_mode="matrix_free"):
    """Replay a two-step graph of the double-buffered solve with the eager result."""
    model = _scene(device)
    _, eager = _run(model, solve_mode=solve_mode, use_parallel_streams=True, double_buffer=True)
    _, captured = _run(model, solve_mode=solve_mode, use_parallel_streams=True, double_buffer=True, capture=True)
    np.testing.assert_array_equal(captured, eager)


@wp.kernel
def _spin(iterations: int, sink: wp.array[float]):
    value = float(0.0)
    for i in range(iterations):
        value = wp.sin(value + float(i))
    sink[0] = value


def _run_sequence(model, plan, *, delay_clear=False, **solver_kwargs):
    """Run ``plan`` and return ``joint_q`` after each operation.

    Operations: ``("eager", n)``, ``("capture", name, n)`` (records ``n`` steps without
    running them), ``("replay", name, n)`` and ``("reset",)``. With ``delay_clear`` the
    double-buffer clearing of the last eager step before each capture first waits for a
    long kernel on its stream, so it is still pending when the graph is first launched.
    """
    solver = SolverFeatherPGS(model, friction_anchor_beta=0.0, articulation_constraint_max=96, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    sink = wp.zeros(1, dtype=float, device=model.device)

    def substep():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(state_0, name), getattr(state_1, name))

    graphs = {}
    results = []
    for index, (operation, *args) in enumerate(plan):
        if operation == "eager":
            for step in range(args[0]):
                last_before_capture = step == args[0] - 1 and index + 1 < len(plan) and plan[index + 1][0] == "capture"
                if delay_clear and solver._double_buffered and last_before_capture:
                    with wp.ScopedStream(solver._memset_stream, sync_enter=False):
                        wp.launch(_spin, dim=1, inputs=[20_000_000], outputs=[sink])
                substep()
        elif operation == "capture":
            with wp.ScopedCapture(model.device) as scope:
                for _ in range(args[1]):
                    substep()
            graphs[args[0]] = scope.graph
        elif operation == "replay":
            for _ in range(args[1]):
                wp.capture_launch(graphs[args[0]])
        else:
            solver.reset(state_0)
        results.append(state_0.joint_q.numpy().copy())
    solver.check_constraint_capacity()
    return solver, results


_SEQUENCES = {
    "cold capture": [("capture", "a", 2), ("replay", "a", 6)],
    "eager then capture": [("eager", 3), ("capture", "a", 2), ("replay", "a", 6), ("eager", 2)],
    "odd steps per graph": [("eager", 2), ("capture", "a", 3), ("replay", "a", 5), ("eager", 1)],
    "one step per graph": [("eager", 1), ("capture", "a", 1), ("replay", "a", 5)],
    "two graphs alternated": [
        ("eager", 2),
        ("capture", "a", 1),
        ("capture", "b", 2),
        ("replay", "a", 2),
        ("replay", "b", 2),
        ("replay", "a", 3),
        ("eager", 1),
        ("replay", "b", 1),
    ],
    "capture after reset": [("eager", 2), ("reset",), ("capture", "a", 2), ("replay", "a", 4), ("reset",)],
}


def test_double_buffer_capture_sequences_match_single_buffer(test, device):
    """Match the single-buffer trajectory bitwise across capture, replay, eager and reset sequences."""
    model = _scene(device)
    for name, plan in _SEQUENCES.items():
        _, expected = _run_sequence(model, plan)
        for streams in (False, True):
            with test.subTest(sequence=name, use_parallel_streams=streams):
                solver, actual = _run_sequence(model, plan, double_buffer=True, use_parallel_streams=streams)
                test.assertTrue(solver._double_buffered)
                for got, want in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(got, want)


def test_capture_after_eager_orders_the_pending_clear(test, device):
    """Order an eager step's still-running buffer clearing before the first replay of a later graph.

    The four-bar's loop-closure rows accumulate into the Jacobian, so a set that is not
    cleared yet changes the replayed trajectory.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(_build_four_bar(), 8)
    model = builder.finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().reshape(8, -1)
    targets[:, 0] = 0.6
    control.joint_target_q.assign(targets.reshape(-1))
    sink = wp.zeros(1, dtype=float, device=device)
    results = []
    for double_buffer in (False, True):
        solver = SolverFeatherPGS(
            model, solve_mode="matrix_free", iterations=16, baumgarte=0.1, double_buffer=double_buffer
        )
        test.assertEqual(solver._double_buffered, double_buffer)
        state_0, state_1 = model.state(), model.state()

        def step(solver=solver, state_0=state_0, state_1=state_1):
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_0, name), getattr(state_1, name))

        step()
        step()
        if double_buffer:
            # Hold the last eager step's clearing back until long after the graph launches.
            with wp.ScopedStream(solver._memset_stream, sync_enter=False):
                wp.launch(_spin, dim=1, inputs=[100_000_000], outputs=[sink])
        step()
        with wp.ScopedCapture(device) as scope:
            step()
            step()
        for _ in range(6):
            wp.capture_launch(scope.graph)
        results.append(state_0.joint_q.numpy())
    np.testing.assert_array_equal(results[1], results[0])


def test_reset_after_an_eager_error_steps_like_a_fresh_solver(test, device):
    """Recover from an eager step that raised mid-step: after reset, step bitwise like a fresh solver.

    The step raises after the closure rows were written into the active buffer set, and the
    four-bar starts with a closure error so those rows are nonzero.
    """
    results = []
    for fail_first in (False, True):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(_build_four_bar(), 2)
        model = builder.finalize(device=device)
        set_torsion_radius(model, 0.01)
        solver = SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=16,
            baumgarte=0.1,
            double_buffer=True,
            contact_torsion_device=True,
        )
        test.assertTrue(solver._double_buffered)
        state, output = model.state(), model.state()
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.2
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        if fail_first:
            # A latched torsion error makes the eager step raise during torsion preparation.
            solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
            with test.assertRaisesRegex(RuntimeError, "load ring"):
                solver.step(state, output, model.control(), None, 1.0 / 240.0)
            solver.reset(state)
            solver.check_constraint_capacity()
        for _ in range(3):
            solver.step(state, output, model.control(), None, 1.0 / 240.0)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state, name), getattr(output, name))
        results.append((state.joint_q.numpy(), state.joint_qd.numpy()))
    for got, want in zip(results[1], results[0], strict=True):
        np.testing.assert_array_equal(got, want)


def test_flags_are_ignored_where_they_do_not_apply(test, device):
    """Keep one stream and one buffer set on CPU, with sparse factors and with one size group."""
    if wp.get_device(device).is_cpu:
        solver = SolverFeatherPGS(_scene(device), solve_mode="split", use_parallel_streams=True, double_buffer=True)
        test.assertEqual(solver._size_streams, {})
        test.assertFalse(solver._double_buffered)
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
for _fn in (
    test_double_buffer_capture_sequences_match_single_buffer,
    test_capture_after_eager_orders_the_pending_clear,
    test_reset_after_an_eager_error_steps_like_a_fresh_solver,
):
    add_function_test(TestFeatherPGSStreams, _fn.__name__, _fn, devices=cuda_devices)
for _fn in (test_streams_and_double_buffer_step_identically, test_captured_double_buffer_matches_eager):
    add_function_test(TestFeatherPGSStreams, _fn.__name__, _fn, devices=cuda_devices)
    add_function_test(TestFeatherPGSStreams, f"{_fn.__name__}_split", _fn, devices=cuda_devices, solve_mode="split")
add_function_test(
    TestFeatherPGSStreams,
    "test_flags_are_ignored_where_they_do_not_apply",
    test_flags_are_ignored_where_they_do_not_apply,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
