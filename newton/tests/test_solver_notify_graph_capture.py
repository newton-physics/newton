# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_SINGLE_FLAGS = tuple(flag for flag in newton.ModelFlags if flag not in (newton.ModelFlags.NONE, newton.ModelFlags.ALL))


def _articulated_model(device):
    """A revolute pendulum, a free sphere, and a ground plane."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        -1, link, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity())
    )
    builder.add_articulation([joint])
    builder.add_shape_sphere(builder.add_body(xform=wp.transform((1.0, 0.0, 0.5), wp.quat_identity())), radius=0.1)
    builder.add_ground_plane()
    builder.color()
    return builder.finalize(device=device)


class TestNotifyModelChangedGraphable(unittest.TestCase):
    def test_default_reports(self):
        """Solvers that keep the base notification are graphable; overrides that do not report are not."""
        model = _articulated_model("cpu")
        semi_implicit = newton.solvers.SolverSemiImplicit(model)
        self.assertTrue(semi_implicit.is_notify_model_changed_graphable(newton.ModelFlags.ALL))

        class CachingSolver(newton.solvers.SolverBase):
            def notify_model_changed(self, flags):
                pass

        caching = CachingSolver(model)
        self.assertFalse(caching.is_notify_model_changed_graphable(newton.ModelFlags.SHAPE_PROPERTIES))
        self.assertTrue(caching.is_notify_model_changed_graphable(newton.ModelFlags.NONE))

        # Featherstone refreshes body inertia on the device but its kinematic state on the host.
        featherstone = newton.solvers.SolverFeatherstone(model)
        self.assertTrue(featherstone.is_notify_model_changed_graphable(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
        self.assertFalse(
            featherstone.is_notify_model_changed_graphable(
                newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.BODY_PROPERTIES
            )
        )


def test_graphable_notifications_capture(test, device):
    """Every flag a solver reports as graphable captures and replays in a CUDA graph."""
    if not wp.is_mempool_enabled(device):
        test.skipTest("Capturing allocations requires a memory pool")
    solvers = (
        newton.solvers.SolverXPBD,
        newton.solvers.SolverFeatherstone,
        newton.solvers.SolverVBD,
        newton.solvers.SolverMuJoCo,
        newton.solvers.SolverSteel,
        newton.solvers.SolverKamino,
    )
    for solver_class in solvers:
        solver = solver_class(_articulated_model(device))
        # Capturing a notification reported as not graphable can leave the device unusable, so only
        # the reported ones are captured.
        for flag in (flag for flag in _SINGLE_FLAGS if solver.is_notify_model_changed_graphable(flag)):
            with test.subTest(solver=solver_class.__name__, flag=flag.name):
                with wp.ScopedCapture(device) as capture:
                    solver.notify_model_changed(flag)
                wp.capture_launch(capture.graph)
    wp.synchronize_device(device)


add_function_test(
    TestNotifyModelChangedGraphable,
    "test_graphable_notifications_capture",
    test_graphable_notifications_capture,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
