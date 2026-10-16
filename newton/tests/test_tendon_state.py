# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that tendon material history follows simulation states, not solvers."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_tendon_capstan import build_explicit_inactive_dynamic_route, build_kinematic_rolling_transport


class TestTendonState(unittest.TestCase):
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
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                with self.subTest(device=device, solver=solver_type.__name__):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    a = builder.add_body(mass=0.0)
                    b = builder.add_body(xform=wp.transform(p=(1.0, 0.0, 0.0)), mass=0.0)
                    builder.add_tendon(
                        [newton.TendonGuide(body=a), newton.TendonGuide(body=b, compliance=0.01, rest_length=0.9)]
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
