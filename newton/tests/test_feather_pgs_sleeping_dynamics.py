# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Skipping the dynamics of sleeping SolverFeatherPGS articulations must not change any published state."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.test_feather_pgs_sleeping_production import PROFILE
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")
# Hard point friction: branched articulations of one topology select sparse mass factors.
_SPARSE_PROFILE = {
    "friction_anchor_beta": 0.0,
    "pgs_contact_regularization": 0.0,
    "contact_friction_gap_threshold": float("inf"),
}


def _compare(test, device, *, interval, velocity_limits, graph, arm=True, expect_sparse=False, **options):
    runs = [_Run(device, skip, interval, velocity_limits, graph, arm, options) for skip in (False, True)]
    test.assertTrue(runs[1].solver._sleep_skips_dynamics)
    test.assertFalse(runs[0].solver._sleep_skips_dynamics)
    test.assertEqual(runs[1].solver._sparse_mass_matrix_size is not None, expect_sparse)
    slept = woke = False
    for phase, steps, force in (("settle", 400, 0.0), ("push", 6, 20.0), ("resettle", 600, 0.0)):
        for step in range(steps):
            for run in runs:
                run.advance(force)
            for field in _FIELDS:
                np.testing.assert_array_equal(
                    runs[1].field(field), runs[0].field(field), err_msg=f"{phase} step {step} {field}"
                )
            awake = runs[1].solver.sleeping.art_awake.numpy()
            slept |= phase == "settle" and not awake[:2].any()
            woke |= phase == "push" and bool(awake[0])
    test.assertTrue(slept and woke)
    # The driven arm never sleeps, and everything else settles again.
    expected = [0, 0, 0, 1] if arm else [0, 0, 0]
    np.testing.assert_array_equal(runs[1].solver.sleeping.art_awake.numpy(), expected)
    test.assertFalse(runs[1].solver._dynamics_art_active.numpy()[:3].any())


def test_eager_trajectories_match(test, device):
    """Settle, sleep, force-wake and resettle identically with and without skipped dynamics."""
    for interval in (1, 3):
        for velocity_limits in (False, True):
            with test.subTest(interval=interval, velocity_limits=velocity_limits):
                _compare(test, device, interval=interval, velocity_limits=velocity_limits, graph=False)


def test_alternate_kernel_paths_match(test, device):
    """Cover tiled triangular solves, cooperative tree traversal and sparse mass factors."""
    for options in (
        {"kernel_overrides": {"trisolve_kernel": "tiled"}},
        {"parallel_tree": True},
        {"arm": False, "expect_sparse": True, **_SPARSE_PROFILE},
        {"arm": False, "expect_sparse": True, "parallel_tree": True, **_SPARSE_PROFILE},
    ):
        with test.subTest(options=options):
            _compare(test, device, interval=3, velocity_limits=False, graph=False, **options)


def test_graph_trajectories_match(test, device):
    """Replay captured steps identically with and without skipped dynamics."""
    for interval in (1, 2):
        with test.subTest(interval=interval):
            _compare(test, device, interval=interval, velocity_limits=True, graph=True)


class _Run:
    def __init__(self, device, skip, interval, velocity_limits, graph, arm, options):
        self.model = _scene(device, arm)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=512)
        options = dict(options)
        overrides = options.pop("kernel_overrides", {})
        with mock.patch.object(newton.solvers.SolverFeatherPGS, "_kernel_overrides", overrides):
            self.solver = newton.solvers.SolverFeatherPGS(
                self.model,
                pgs_mode="matrix_free",
                **{
                    **PROFILE,
                    "update_mass_matrix_interval": interval,
                    "enable_joint_velocity_limits": velocity_limits,
                    **options,
                },
            )
        self.solver.sleeping.skip_dynamics = skip
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.graph = None
        if graph:
            self._step()
            self._step()
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._step()
                self._step()
            self.graph = capture.graph
        self.force = np.zeros((self.model.body_count, 6), dtype=np.float32)

    def advance(self, force):
        self.force[0, 0] = force
        for state in self.states:
            state.body_f.assign(self.force)
        if self.graph is None:
            self._step()
        else:
            wp.capture_launch(self.graph)

    def _step(self):
        self.pipeline.collide(self.states[0], self.contacts)
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, 0.005)
        self.states.reverse()

    def field(self, name):
        return getattr(self.states[0], name).numpy()


def _scene(device, arm):
    """Two resting branching articulations, a resting box and, with ``arm``, a driven fixed-base arm."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 1.0):
        base = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
        joints = [builder.add_joint_free(child=base)]
        # Two tips make the articulation a branching tree.
        for side in (-1.0, 1.0):
            tip = builder.add_link(xform=wp.transform((x + side * 0.3, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
            hinge = wp.transform((side * 0.3, 0.0, 0.0), wp.quat_identity())
            joints.append(builder.add_joint_revolute(parent=base, child=tip, parent_xform=hinge, axis=(0.0, 1.0, 0.0)))
        builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform((3.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    if arm:
        root = builder.add_link(xform=wp.transform((5.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
        joints = [
            builder.add_joint_fixed(
                parent=-1, child=root, parent_xform=wp.transform((5.0, 0.0, 0.5), wp.quat_identity())
            )
        ]
        parent = root
        for k in range(3):
            link = builder.add_link(xform=wp.transform((5.0, 0.0, 0.3 - 0.2 * k), wp.quat_identity()))
            builder.add_shape_box(link, hx=0.03, hy=0.03, hz=0.08)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    parent_xform=wp.transform((0.0, 0.0, -0.2), wp.quat_identity()),
                    axis=(0.0, 1.0, 0.0),
                    target_ke=100.0,
                    target_kd=10.0,
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )
            )
            parent = link
        builder.add_articulation(joints)
    builder.joint_velocity_limit[:] = [5.0] * len(builder.joint_velocity_limit)
    return builder.finalize(device=device)


devices = get_cuda_test_devices()


class TestFeatherPGSSleepingDynamics(unittest.TestCase):
    pass


for _name in ("test_eager_trajectories_match", "test_alternate_kernel_paths_match", "test_graph_trajectories_match"):
    add_function_test(TestFeatherPGSSleepingDynamics, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
