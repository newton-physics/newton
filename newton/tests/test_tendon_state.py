# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that tendon material history follows simulation states, not solvers."""

import unittest
from functools import partial

import numpy as np
import warp as wp

import newton
from newton.examples.tendon.utils import (
    get_tendon_attachment_worlds,
    get_tendon_cable_lines,
    get_tendon_total_lengths,
)
from newton.tests.test_tendon_capstan import build_explicit_inactive_dynamic_route, build_kinematic_rolling_transport


class TestTendonState(unittest.TestCase):
    def test_world_guides_without_bodies(self):
        """Equilibrate and replay world-only tendons without requiring an unrelated rigid body."""

        def step_pair(solver, state, output, control):
            solver.step(state, output, control, None, 0.01)
            solver.step(output, state, control, None, 0.01)

        for unused_body in (False, True):
            builder = newton.ModelBuilder()
            if unused_body:
                builder.add_body(mass=0.0)
            builder.begin_world()
            builder.add_tendon(
                [
                    newton.TendonGuide(body=-1, offset=(-1.0, 0.0, 0.0)),
                    newton.TendonGuide(
                        body=-1,
                        guide_type=newton.TendonGuideType.PINHOLE,
                        compliance=0.01,
                        damping=0.3,
                        rest_length=0.8,
                    ),
                    newton.TendonGuide(body=-1, offset=(0.0, 1.0, 0.0), compliance=0.01, damping=0.3, rest_length=1.0),
                ]
            )
            builder.end_world()
            builder.color()
            for device in wp.get_devices():
                for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                    with self.subTest(device=device, solver=solver_type.__name__, unused_body=unused_body):
                        model = builder.finalize(device=device)
                        solver = solver_type(model, iterations=2)
                        state, output = model.state(), model.state()
                        control = model.control()

                        advance = partial(step_pair, solver, state, output, control)
                        advance()
                        if device.is_cuda:
                            with wp.ScopedCapture(device=device) as capture:
                                advance()

                            advance = partial(wp.capture_launch, capture.graph)

                        advance()
                        np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.9, 0.9], atol=1e-6)
                        np.testing.assert_allclose(
                            solver.tendon_segment_material_tension.numpy(), [10.0, 10.0], atol=1e-4
                        )
                        np.testing.assert_allclose(solver.tendon_segment_damping_tension.numpy(), 0.0, atol=1e-6)

                        # Drawing and length diagnostics must also work without body arrays.
                        for helper in (get_tendon_attachment_worlds, get_tendon_cable_lines):
                            with self.subTest(helper=helper.__name__):
                                starts, ends = helper(solver, model, state)
                                if helper is get_tendon_cable_lines:
                                    starts, ends = starts.numpy(), ends.numpy()
                                np.testing.assert_allclose(starts, [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                                np.testing.assert_allclose(ends, [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                        with self.subTest(helper="get_tendon_total_lengths"):
                            np.testing.assert_allclose(get_tendon_total_lengths(solver, model, state), [2.0])

                        # The next step must load the edited state, not reuse the solver's old history.
                        state.tendon_segment_rest_length.fill_(0.7)
                        advance()
                        np.testing.assert_allclose(
                            solver.tendon_segment_material_tension.numpy(), [30.0, 30.0], atol=1e-4
                        )
                        mask = wp.array([True, False], dtype=wp.bool, device=device)
                        solver.reset(state, world_mask=mask, flags=0)
                        np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.8, 1.0], atol=1e-6)
                        advance()
                        np.testing.assert_allclose(
                            solver.tendon_segment_material_tension.numpy(), [10.0, 10.0], atol=1e-4
                        )

    def test_world_guides_match_stationary_bodies(self):
        """Match frictional, damped anchor/roller/pinhole routes and body impulses against stationary guides."""

        def build(world_attached, guide_type, device):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            pose = wp.transform(wp.vec3(0.2, 0.3, 0.4), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3))
            fixed = builder.add_body(xform=pose, mass=0.0)
            moving = builder.add_body(
                xform=wp.transform(p=wp.transform_point(pose, wp.vec3(0.5, 0.0, -0.5))),
                mass=1.0,
                inertia=wp.mat33(np.eye(3)),
                com=(0.02, 0.0, 0.0),
            )
            axis = wp.vec3(0.0, 1.0, 0.0)
            guides = []
            for offset, kind in ((wp.vec3(-0.5, 0.0, -0.5), newton.TendonGuideType.ANCHOR), (wp.vec3(), guide_type)):
                guides.append(
                    newton.TendonGuide(
                        body=-1 if world_attached else fixed,
                        offset=wp.transform_point(pose, offset) if world_attached else offset,
                        axis=wp.transform_vector(pose, axis) if world_attached else axis,
                        guide_type=kind,
                        radius=0.1 if kind == newton.TendonGuideType.ROLLER else 0.0,
                        mu=0.2,
                        compliance=0.01 if guides else 0.0,
                        damping=0.3 if guides else 0.0,
                        rest_length=0.6 if guides else -1.0,
                    )
                )
            guides.append(newton.TendonGuide(body=moving, compliance=0.02, damping=0.5, rest_length=0.6))
            builder.add_tendon(guides)
            builder.color()
            return builder.finalize(device=device)

        for device in wp.get_devices():
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                for guide_type in (newton.TendonGuideType.ROLLER, newton.TendonGuideType.PINHOLE):
                    with self.subTest(device=device, solver=solver_type.__name__, guide_type=guide_type):
                        results = []
                        for world_attached in (False, True):
                            model = build(world_attached, guide_type, device)
                            solver = solver_type(model, iterations=4)
                            a, b = model.state(), model.state()
                            velocity = a.body_qd.numpy()
                            velocity[1] = [0.1, 0.0, -0.2, 0.0, 0.3, 0.1]
                            a.body_qd.assign(velocity)
                            for _ in range(8):
                                solver.step(a, b, model.control(), None, 0.01)
                                a, b = b, a
                            results.append(
                                [
                                    a.body_q.numpy(),
                                    a.body_qd.numpy(),
                                    a.tendon_segment_rest_length.numpy(),
                                    solver.tendon_segment_material_tension.numpy(),
                                    solver.tendon_segment_damping_tension.numpy(),
                                    get_tendon_total_lengths(solver, model, a),
                                    *(lines.numpy() for lines in get_tendon_cable_lines(solver, model, a)),
                                ]
                            )
                        self.assertGreater(np.max(results[0][3]), 0.1)
                        for reference, actual in zip(*results, strict=True):
                            np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)

    def test_world_dynamic_roller(self):
        """Activate and detach a world-fixed roller as its neighboring anchors move."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        endpoints = builder.add_body(mass=0.0)
        builder.add_tendon(
            [
                newton.TendonGuide(body=endpoints, offset=(0.0, 0.0, -0.5), axis=(0.0, 1.0, 0.0)),
                newton.TendonGuide(
                    body=-1,
                    guide_type=newton.TendonGuideType.ROLLER,
                    offset=(0.25, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                    radius=0.1,
                    dynamic=True,
                    mu=0.2,
                    compliance=0.01,
                ),
                newton.TendonGuide(body=endpoints, offset=(0.0, 0.0, 0.5), axis=(0.0, 1.0, 0.0), compliance=0.01),
            ]
        )
        builder.color()
        for device in wp.get_devices():
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                with self.subTest(device=device, solver=solver_type.__name__):
                    model = builder.finalize(device=device)
                    solver = solver_type(model, iterations=2)
                    a, b = model.state(), model.state()
                    self.assertFalse(solver.tendon_guide_active.numpy()[1])
                    initial_rest = solver.tendon_segment_rest_length.numpy().copy()
                    for x, active in ((0.2, True), (0.0, False)):
                        poses = a.body_q.numpy()
                        poses[0, 0] = x
                        a.body_q.assign(poses)
                        solver.step(a, b, model.control(), None, 0.01)
                        a, b = b, a
                        self.assertEqual(bool(a.tendon_guide_active.numpy()[1]), active)
                        self.assertTrue(np.isfinite(a.tendon_segment_rest_length.numpy()).all())
                    solver.reset(a)
                    np.testing.assert_allclose(a.tendon_segment_rest_length.numpy(), initial_rest)
                    self.assertFalse(a.tendon_guide_active.numpy()[1])

    def test_checkpoint_preserves_activation_hysteresis(self):
        """Restore the accepted activity even where the same pose permits either route."""
        for device in wp.get_devices():
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                with self.subTest(device=device, solver=solver_type.__name__), wp.ScopedDevice(device):
                    model, guide = build_explicit_inactive_dynamic_route(device)
                    model.body_color_groups = [wp.array([i], dtype=int) for i in range(model.body_count)]
                    roller = model.tendon_guide_body.numpy()[guide]
                    solver = solver_type(model, iterations=4, tendon_activation_tol=2.0e-3)
                    a, b, saved, expected = (model.state() for _ in range(4))
                    control = model.control()
                    # Activate first, then stop inside the hysteresis band.
                    for x in (0.05, 0.0999):
                        poses = a.body_q.numpy().copy()
                        poses[roller, 0] = x
                        a.body_q.assign(poses)
                        solver.step(a, b, control, None, 0.01)
                        a, b = b, a
                        self.assertTrue(a.tendon_guide_active.numpy()[guide])
                    saved.assign(a)
                    solver.step(a, b, control, None, 0.01)
                    expected.assign(b)

                    poses = a.body_q.numpy().copy()
                    poses[roller, 0] = 0.25
                    a.body_q.assign(poses)
                    solver.step(a, b, control, None, 0.01)
                    self.assertFalse(b.tendon_guide_active.numpy()[guide])
                    for replay_solver in (solver, solver_type(model, iterations=4, tendon_activation_tol=2.0e-3)):
                        replay_solver.reset(a, flags=0)
                        a.assign(saved)
                        replay_solver.step(a, b, control, None, 0.01)
                        np.testing.assert_array_equal(
                            b.tendon_guide_active.numpy(), expected.tendon_guide_active.numpy()
                        )
                        np.testing.assert_allclose(
                            b.tendon_segment_rest_length.numpy(), expected.tendon_segment_rest_length.numpy(), atol=1e-7
                        )

    def test_captured_reset_uses_updated_baseline(self):
        """Replay one graph after a property update without replacing its history buffers."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph capture requires a CUDA device")
        for device in devices:
            for world_attached in (False, True):
                for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                    with self.subTest(device=device, solver=solver_type.__name__, world_attached=world_attached):
                        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                        anchor = -1 if world_attached else builder.add_body(mass=0.0)
                        b = builder.add_body(xform=wp.transform(p=(1.0, 0.0, 0.0)), mass=0.0)
                        builder.add_tendon(
                            [
                                newton.TendonGuide(body=anchor),
                                newton.TendonGuide(body=b, compliance=0.01, rest_length=0.9),
                            ]
                        )
                        builder.color()
                        model = builder.finalize(device=device)
                        solver = solver_type(model, iterations=2)
                        state, output, saved = (model.state() for _ in range(3))
                        control = model.control()
                        solver.step(state, output, control, None, 0.01)
                        with wp.ScopedCapture(device=device) as capture:
                            solver.step(state, output, control, None, 0.01)
                            solver.step(output, state, control, None, 0.01)
                        wp.capture_launch(capture.graph)
                        np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.9], atol=1e-7)
                        saved.assign(state)
                        model.tendon_segment_rest_length.fill_(0.5)
                        model.tendon_segment_compliance.fill_(0.02)
                        solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
                        solver.reset(state)
                        wp.capture_launch(capture.graph)
                        np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.5], atol=1e-7)
                        np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [25.0], atol=1e-5)
                        solver.reset(state, flags=0)
                        state.assign(saved)
                        wp.capture_launch(capture.graph)
                        np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.9], atol=1e-7)
                        np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [5.0], atol=1e-5)

    def test_damped_dynamic_body_replay(self):
        """Replay body motion and damped material transfer from one saved trajectory."""
        for device in wp.get_devices():
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                with self.subTest(device=device, solver=solver_type.__name__):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    anchor = builder.add_body(xform=wp.transform(p=(-1.0, 0.0, 0.0)), mass=0.0)
                    pin = builder.add_body(mass=0.0)
                    moving = builder.add_body(
                        xform=wp.transform(p=(0.0, 1.0, 0.0)), mass=1.0, inertia=wp.mat33(np.eye(3))
                    )
                    builder.add_tendon(
                        [
                            newton.TendonGuide(body=anchor),
                            newton.TendonGuide(
                                body=pin,
                                guide_type=newton.TendonGuideType.PINHOLE,
                                mu=0.2,
                                compliance=0.01,
                                damping=0.1,
                                rest_length=0.9,
                            ),
                            newton.TendonGuide(body=moving, compliance=0.02, damping=0.1, rest_length=0.8),
                        ]
                    )
                    builder.color()
                    model = builder.finalize(device=device)
                    solver = solver_type(model, iterations=8)
                    a, b, saved, expected = (model.state() for _ in range(4))
                    control = model.control()
                    for _ in range(4):
                        solver.step(a, b, control, None, 0.01)
                        a, b = b, a
                    saved.assign(a)
                    solver.step(a, b, control, None, 0.01)
                    expected.assign(b)
                    for _ in range(5):
                        a, b = b, a
                        solver.step(a, b, control, None, 0.01)
                    for replay_solver in (solver, solver_type(model, iterations=8)):
                        # VBD also has non-tendon rigid-body history. Clear it
                        # before restoring the saved physical tendon history.
                        replay_solver.reset(a, flags=0)
                        a.assign(saved)
                        replay_solver.step(a, b, control, None, 0.01)
                        for name in ("body_q", "body_qd", "tendon_segment_rest_length", "tendon_guide_active"):
                            np.testing.assert_allclose(
                                getattr(b, name).numpy(), getattr(expected, name).numpy(), atol=1e-6
                            )

    def test_property_updates(self):
        """Refresh reset rest lengths without discarding history for compliance-only updates."""
        for device in wp.get_devices():
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                with self.subTest(device=device, solver=solver_type.__name__):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    a = builder.add_body(mass=0.0)
                    b = builder.add_body(xform=wp.transform(p=(1.0, 0.0, 0.0)), mass=0.0)
                    builder.add_tendon(
                        [newton.TendonGuide(body=a), newton.TendonGuide(body=b, compliance=0.01, rest_length=1.0)]
                    )
                    builder.color()
                    model = builder.finalize(device=device)
                    solver = solver_type(model, iterations=2)
                    state, output = model.state(), model.state()
                    solver.step(state, output, model.control(), None, 0.01)
                    state.assign(output)
                    state.tendon_segment_rest_length.fill_(0.8)
                    model.tendon_segment_compliance.fill_(0.02)
                    model.tendon_segment_damping.fill_(0.1)
                    solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
                    solver.step(state, output, model.control(), None, 0.01)
                    np.testing.assert_allclose(output.tendon_segment_rest_length.numpy(), [0.8], atol=1e-7)
                    np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [10.0], atol=1e-5)
                    model.tendon_segment_rest_length.fill_(0.5)
                    solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
                    solver.step(state, output, model.control(), None, 0.01)
                    np.testing.assert_allclose(output.tendon_segment_rest_length.numpy(), [0.8], atol=1e-7)
                    solver.reset(state)
                    np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.5], atol=1e-7)
                    solver.step(state, output, model.control(), None, 0.01)
                    np.testing.assert_allclose(output.tendon_segment_rest_length.numpy(), [0.5], atol=1e-7)

                    model.tendon_segment_damping.fill_(-1.0)
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
                    model.tendon_segment_damping.fill_(0.1)
                    model.tendon_guide_radius.fill_(0.1)
                    with self.assertRaisesRegex(ValueError, "reconstructing"):
                        solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)

    def test_checkpoint_replay(self):
        """Replay a saved moving, frictional route after advancing another trajectory."""
        for device in wp.get_devices():
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                with self.subTest(device=device, solver=solver_type.__name__), wp.ScopedDevice(device):
                    model, roller = build_kinematic_rolling_transport(mu=10.0)
                    model.body_color_groups = [wp.array([i], dtype=int) for i in range(model.body_count)]
                    solver = solver_type(model, iterations=4)
                    control = model.control()
                    a, b, saved, expected = (model.state() for _ in range(4))
                    initial_rest = solver.tendon_segment_rest_length.numpy().copy()
                    for step in range(3):
                        pose = a.body_q.numpy()
                        angle = (step + 1) * 0.05
                        pose[roller, 3:] = [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
                        a.body_q.assign(pose)
                        solver.step(a, b, control, None, 1.0 / 60.0)
                        a, b = b, a
                    self.assertGreater(np.max(np.abs(a.tendon_segment_rest_length.numpy() - initial_rest)), 1e-5)
                    saved.assign(a)
                    solver.step(a, b, control, None, 1.0 / 60.0)
                    expected.assign(b)
                    for step in range(4):
                        a, b = b, a
                        pose = a.body_q.numpy()
                        angle = 0.3 + step * 0.1
                        pose[roller, 3:] = [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
                        a.body_q.assign(pose)
                        solver.step(a, b, control, None, 1.0 / 60.0)
                    a.assign(saved)
                    solver.step(a, b, control, None, 1.0 / 60.0)
                    for name in (
                        "body_q",
                        "tendon_segment_rest_length",
                        "tendon_guide_active",
                        "tendon_segment_attachment_l_local",
                        "tendon_segment_attachment_r_local",
                    ):
                        np.testing.assert_allclose(getattr(b, name).numpy(), getattr(expected, name).numpy(), atol=1e-7)
                    fresh_solver = solver_type(model, iterations=4)
                    a.assign(saved)
                    fresh_solver.step(a, b, control, None, 1.0 / 60.0)
                    np.testing.assert_allclose(
                        b.tendon_segment_rest_length.numpy(), expected.tendon_segment_rest_length.numpy(), atol=1e-7
                    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
