# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test complete tendon routes and builder composition."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton


class TestTendonBuilder(unittest.TestCase):
    def test_collapse_preserves_tendons_and_particle_attachments(self):
        """Retain world-fixed bodies needed by either tendon guides or particle attachments."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        tendon_anchor = builder.add_link(mass=1.0)
        particle_anchor = builder.add_link(xform=wp.transform(p=(1.0, 0.0, 0.0)), mass=1.0)
        endpoint = builder.add_body(xform=wp.transform(p=(2.0, 0.0, 0.0)), mass=1.0)
        for body in (tendon_anchor, particle_anchor):
            joint = builder.add_joint_fixed(-1, body, parent_xform=builder.body_q[body])
            builder.add_articulation([joint])
        particle = builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(), mass=1.0)
        builder.add_attachment_body_particle(particle_anchor, particle)
        builder.add_tendon(
            [newton.TendonGuide(body=tendon_anchor), newton.TendonGuide(body=endpoint, compliance=1.0e-3)]
        )

        builder.collapse_fixed_joints()
        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.tendon_guide_body, [0, 2])
        self.assertEqual(builder.attachment_body_particle_body, [1])
        builder.color()
        for device in wp.get_devices():
            with self.subTest(device=device):
                model = builder.finalize(device=device)
                solver = newton.solvers.SolverVBD(model, iterations=2)
                state, output = model.state(), model.state()
                solver.step(state, output, model.control(), None, 1.0 / 240.0)
                np.testing.assert_allclose(output.body_q.numpy(), state.body_q.numpy(), atol=1.0e-6)
                np.testing.assert_allclose(output.particle_q.numpy(), state.particle_q.numpy(), atol=1.0e-6)
                self.assertEqual(model.tendon_count, 1)
                self.assertEqual(model.attachment_body_particle_count, 1)

    def test_collapse_preserves_guides_across_fixed_joint_chain(self):
        """Compose local guide transforms across more than one collapsed joint."""
        builder = newton.ModelBuilder()
        inertia = wp.mat33(np.eye(3))
        root = builder.add_link(mass=1.0, inertia=inertia)
        first = wp.transform(wp.vec3(1.0, 0.5, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5))
        second = wp.transform(wp.vec3(0.0, 0.2, 0.3), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3))
        middle = builder.add_link(xform=first, mass=1.0, inertia=inertia)
        child_pose = wp.transform_multiply(first, second)
        child = builder.add_link(xform=child_pose, mass=1.0, inertia=inertia)
        endpoint = builder.add_link(xform=wp.transform(p=wp.vec3(3.0, 0.0, 0.0)), mass=1.0, inertia=inertia)
        builder.add_joint_fixed(parent=root, child=middle, parent_xform=first)
        builder.add_joint_fixed(parent=middle, child=child, parent_xform=second)
        offset, axis = wp.vec3(0.2, 0.1, 0.0), wp.vec3(1.0, 0.0, 0.0)
        builder.add_tendon(
            [newton.TendonGuide(body=child, offset=offset, axis=axis), newton.TendonGuide(body=endpoint)]
        )
        world_point = wp.transform_point(child_pose, offset)
        world_axis = wp.transform_vector(child_pose, axis)
        builder.collapse_fixed_joints()
        self.assertEqual(builder.tendon_guide_body, [0, 1])
        retained_pose = builder.body_q[builder.tendon_guide_body[0]]
        np.testing.assert_allclose(
            wp.transform_point(retained_pose, wp.vec3(builder.tendon_guide_offset[0])), world_point, atol=2e-7
        )
        np.testing.assert_allclose(
            wp.transform_vector(retained_pose, wp.vec3(builder.tendon_guide_axis[0])), world_axis, atol=2e-7
        )
        newton.solvers.SolverXPBD(builder.finalize(device="cpu"), iterations=1)

    def test_replicated_dynamic_guides_switch_independently(self):
        """Keep routing transitions, history slots, and masked resets within one world."""
        source = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        lower = source.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)), mass=0.0)
        roller = source.add_body(xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)), mass=0.0)
        upper = source.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)), mass=0.0)
        source.add_tendon(
            [
                newton.TendonGuide(body=lower, axis=(0.0, 1.0, 0.0)),
                newton.TendonGuide(
                    body=roller,
                    guide_type=newton.TendonGuideType.ROLLER,
                    dynamic=True,
                    radius=0.1,
                    mu=0.2,
                    axis=(0.0, 1.0, 0.0),
                    compliance=1e-4,
                    rest_length=0.4,
                ),
                newton.TendonGuide(body=upper, axis=(0.0, 1.0, 0.0), compliance=2e-4, rest_length=0.6),
            ]
        )
        source.color()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.replicate(source, 3)
        for device in wp.get_devices():
            for solver_type, options in (
                (newton.solvers.SolverXPBD, {}),
                (newton.solvers.SolverVBD, {"rigid_compliant_alm": False}),
                (newton.solvers.SolverVBD, {"rigid_compliant_alm": True}),
            ):
                with self.subTest(device=device, solver=solver_type.__name__, options=options):
                    model = builder.finalize(device=device)
                    solver = solver_type(model, iterations=2, **options)
                    state_0, state_1 = model.state(), model.state()
                    control = model.control()
                    initial_rest = solver.tendon_seg_rest_length.numpy().copy()
                    np.testing.assert_array_equal(solver.tendon_guide_active.numpy()[[1, 4, 7]], [False, False, False])
                    for x, active in ((0.05, True), (0.25, False)):
                        poses = state_0.body_q.numpy()
                        poses[4, 0] = x
                        state_0.body_q.assign(poses)
                        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                        state_0, state_1 = state_1, state_0
                        np.testing.assert_array_equal(
                            solver.tendon_guide_active.numpy()[[1, 4, 7]], [False, active, False]
                        )
                        np.testing.assert_array_equal(
                            solver.tendon_seg_rest_length.numpy()[[0, 1, 4, 5]], initial_rest[[0, 1, 4, 5]]
                        )
                    changed = solver.tendon_seg_rest_length.numpy() + np.float32(0.01)
                    solver.tendon_seg_rest_length.assign(changed)
                    mask = wp.array([False, True, False, False], dtype=wp.bool, device=device)
                    solver.reset(state_0, world_mask=mask, flags=0)
                    expected = changed.copy()
                    expected[2:4] = initial_rest[2:4]
                    np.testing.assert_array_equal(solver.tendon_seg_rest_length.numpy(), expected)

    def test_xpbd_reports_wraps_only_at_accepted_pose(self):
        """Suppress unsupported-wrap warnings during unconverged iterations."""
        builder = newton.ModelBuilder()
        a = builder.add_body(mass=0.0)
        b = builder.add_body(xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)), mass=0.0)
        builder.add_tendon([newton.TendonGuide(body=a), newton.TendonGuide(body=b)])
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverXPBD(model, iterations=2)
        with patch.object(solver, "_update_tendon_cone_rows", wraps=solver._update_tendon_cone_rows) as update:
            solver.step(model.state(), model.state(), model.control(), None, 1.0 / 60.0)
        self.assertEqual([call.args[2] for call in update.call_args_list], [False, False, True])

    def test_complete_route_is_atomic(self):
        """Leave all tendon arrays unchanged when any route element is invalid."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=0.0)
        element = newton.TendonGuide
        builder.add_tendon([element(body=body), element(body=body, offset=(1.0, 0.0, 0.0))])
        before = {name: list(value) for name, value in vars(builder).items() if name.startswith("tendon_")}
        with self.assertRaisesRegex(ValueError, "body index"):
            builder.add_tendon([element(body=body), element(body=10)])
        for name, value in before.items():
            self.assertEqual(getattr(builder, name), value, name)

    def test_merge_remaps_tendon_bodies_and_starts(self):
        """Append multiple routes without changing local geometry or materials."""
        element = newton.TendonGuide
        source = newton.ModelBuilder()
        a = source.add_body(mass=0.0)
        b = source.add_body(xform=wp.transform(p=wp.vec3(2.0, 0.0, 0.0)), mass=0.0)
        source.add_tendon(
            [
                element(body=a, offset=(0.1, 0.0, 0.0), axis=(0.0, 2.0, 0.0)),
                element(body=b, compliance=1.0e-5, damping=0.2, rest_length=1.8),
            ]
        )
        source.add_tendon([element(body=b), element(body=a, compliance=2.0e-5)])
        destination = newton.ModelBuilder()
        existing = destination.add_body(mass=0.0)
        destination.add_tendon([element(body=existing), element(body=existing, offset=(1.0, 0.0, 0.0))])
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7)
        xform = wp.transform(wp.vec3(4.0, 5.0, 6.0), rotation)
        destination.add_builder(source, xform=xform)
        destination.add_builder(source)
        self.assertEqual(destination.tendon_start, [0, 2, 4, 6, 8])
        self.assertEqual(destination.tendon_guide_body, [0, 0, 1, 2, 2, 1, 3, 4, 4, 3])
        self.assertEqual(destination.tendon_guide_offset[2:6], source.tendon_guide_offset)
        self.assertEqual(destination.tendon_guide_axis[2:6], source.tendon_guide_axis)
        self.assertEqual(destination.tendon_seg_compliance, [0.0, 1.0e-5, 2.0e-5, 1.0e-5, 2.0e-5])
        np.testing.assert_allclose(destination.body_q[2], wp.transform_multiply(xform, source.body_q[b]), atol=1e-6)
        model = destination.finalize(device="cpu")
        self.assertEqual(model.tendon_count, 5)
        self.assertEqual(model.tendon_segment_count, 5)

    def test_replicate_preserves_routes_per_world(self):
        """Replicate independent routes, including the array-backed merge path."""
        element = newton.TendonGuide
        source = newton.ModelBuilder()
        a = source.add_body(mass=0.0)
        b = source.add_body(xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)), mass=0.0)
        source.add_tendon([element(body=a), element(body=b, compliance=1e-5, damping=0.1)])
        source.color()
        for replicate in (False, True):
            with self.subTest(replicate=replicate):
                builder = newton.ModelBuilder()
                if replicate:
                    builder.replicate(source, 3)
                else:
                    for _ in range(3):
                        builder.add_world(source)
                model = builder.finalize(device="cpu")
                np.testing.assert_array_equal(model.tendon_start.numpy(), [0, 2, 4, 6])
                np.testing.assert_array_equal(model.tendon_guide_body.numpy(), [0, 1, 2, 3, 4, 5])
                np.testing.assert_array_equal(model.body_world.numpy(), [0, 0, 1, 1, 2, 2])
                for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                    options = {"rigid_compliant_alm": False} if solver_type is newton.solvers.SolverVBD else {}
                    solver = solver_type(model, iterations=1, **options)
                    state_0, state_1 = model.state(), model.state()
                    solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
                    np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
