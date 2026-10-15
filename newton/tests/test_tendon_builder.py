# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test complete tendon routes and builder composition."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.model_view import ModelView


class TestTendonBuilder(unittest.TestCase):
    def test_first_guide_has_no_material(self):
        """Reject non-default incoming-span parameters on a guide with no incoming span."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=0.0)
        for kwargs in ({"compliance": 0.1}, {"damping": 1.0}, {"rest_length": 0.2}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "first.*incoming"):
                builder.add_tendon([newton.TendonGuide(body=body, **kwargs), newton.TendonGuide(body=body)])
        self.assertEqual(builder.tendon_count, 0)

    def test_tendon_attribute_domains(self):
        """Preserve labels, domain counts, and custom references through replication."""
        frequency = newton.Model.AttributeFrequency
        source = newton.ModelBuilder()
        for name, freq, ref in (
            ("route_body", frequency.TENDON, "body"),
            ("guide_route", frequency.TENDON_GUIDE, "tendon"),
            ("span_guide", frequency.TENDON_SEGMENT, "tendon_guide"),
        ):
            source.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    dtype=int,
                    frequency=freq,
                    assignment=newton.Model.AttributeAssignment.MODEL,
                    references=ref,
                    default=-1,
                )
            )
        left = source.add_body(mass=0.0)
        right = source.add_body(xform=wp.transform(p=(1.0, 0.0, 0.0)), mass=0.0)
        source.add_tendon(
            [
                newton.TendonGuide(body=left, custom_attributes={"guide_route": 0}),
                newton.TendonGuide(
                    body=right, custom_attributes={"guide_route": 0}, segment_custom_attributes={"span_guide": 1}
                ),
            ],
            label="pull",
            custom_attributes={"route_body": right},
        )
        target = newton.ModelBuilder()
        target.add_world(source, label_prefix="first")
        target.add_world(source, label_prefix="second")
        model = target.finalize(device="cpu")
        self.assertEqual(model.tendon_label, ["first/pull", "second/pull"])
        for name, freq in (
            ("tendon_start", frequency.TENDON),
            ("tendon_guide_offset", frequency.TENDON_GUIDE),
            ("tendon_segment_compliance", frequency.TENDON_SEGMENT),
        ):
            self.assertEqual(model.get_attribute_frequency(name), freq)
        self.assertEqual(model._attribute_frequency_count(frequency.TENDON_GUIDE), 4)
        self.assertEqual(model._attribute_spec("tendon_guide_body").references, frequency.BODY)
        np.testing.assert_array_equal(model.tendon_start.numpy(), [0, 2, 4])
        np.testing.assert_array_equal(model.tendon_world.numpy(), [0, 1])
        np.testing.assert_array_equal(model.route_body.numpy(), [1, 3])
        np.testing.assert_array_equal(model.guide_route.numpy(), [0, 0, 1, 1])
        np.testing.assert_array_equal(model.span_guide.numpy(), [1, 3])
        batched = newton.ModelBuilder()
        batched.replicate(source, 3)
        replicated = batched.finalize(device="cpu")
        np.testing.assert_array_equal(replicated.route_body.numpy(), [1, 3, 5])
        np.testing.assert_array_equal(replicated.span_guide.numpy(), [1, 3, 5])

    def test_tendon_view_rejects_partial_domains(self):
        """Reject compact views until complete-route remapping is supported."""
        builder = newton.ModelBuilder()
        left, right = builder.add_body(mass=0.0), builder.add_body(mass=0.0)
        builder.add_tendon([newton.TendonGuide(body=left), newton.TendonGuide(body=right)])
        view = ModelView(builder.finalize(device="cpu"), "partial")
        with self.assertRaisesRegex(ValueError, "tendon"):
            view.body_count = 1

    def test_tendon_state_and_control_attributes(self):
        """Allocate replicated custom state/control attributes with remapped span references."""
        source = newton.ModelBuilder()
        frequency = newton.Model.AttributeFrequency
        assignment = newton.Model.AttributeAssignment
        for name, freq, target, dtype, references in (
            ("history", frequency.TENDON, assignment.STATE, float, None),
            ("input", frequency.TENDON_GUIDE, assignment.CONTROL, float, None),
            ("peer", frequency.TENDON_SEGMENT, assignment.STATE, int, "tendon_segment"),
        ):
            source.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    namespace="test",
                    dtype=dtype,
                    frequency=freq,
                    assignment=target,
                    references=references,
                    default=-1,
                )
            )
        body = source.add_body(mass=0.0)
        source.add_tendon(
            [
                newton.TendonGuide(body=body, custom_attributes={"test:input": 2.0}),
                newton.TendonGuide(body=body, segment_custom_attributes={"test:peer": 0}),
            ],
            custom_attributes={"test:history": 4.0},
        )
        target = newton.ModelBuilder()
        target.replicate(source, 2)
        model = target.finalize(device="cpu")
        state, saved = model.state(), model.state()
        np.testing.assert_array_equal(state.test.history.numpy(), [4.0, 4.0])
        np.testing.assert_array_equal(state.test.peer.numpy(), [0, 1])
        np.testing.assert_array_equal(model.control().test.input.numpy(), [2.0, -1.0, 2.0, -1.0])
        state.test.history.fill_(7.0)
        saved.assign(state)
        np.testing.assert_array_equal(saved.test.history.numpy(), [7.0, 7.0])

    def test_tendon_custom_attributes_are_atomic(self):
        """An invalid custom attribute must not append a partial route or attribute row."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=0.0)
        with self.assertRaisesRegex(AttributeError, "not defined"):
            builder.add_tendon(
                [newton.TendonGuide(body=body), newton.TendonGuide(body=body)], custom_attributes={"missing": 1}
            )
        self.assertEqual(builder.tendon_count, 0)
        self.assertEqual(builder.tendon_guide_count, 0)
        self.assertEqual(builder.tendon_label, [])
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="guide_value",
                dtype=float,
                frequency=newton.Model.AttributeFrequency.TENDON_GUIDE,
                assignment=newton.Model.AttributeAssignment.MODEL,
            )
        )
        with self.assertRaisesRegex(ValueError, "frequency"):
            builder.add_tendon(
                [
                    newton.TendonGuide(body=body, custom_attributes={"guide_value": 1.0}),
                    newton.TendonGuide(body=body, segment_custom_attributes={"guide_value": 2.0}),
                ]
            )
        self.assertFalse(builder.custom_attributes["guide_value"].values)
        self.assertEqual(builder.tendon_count, 0)
        self.assertEqual(builder.tendon_guide_count, 0)
        self.assertEqual(builder.tendon_segment_count, 0)

    def test_tendon_custom_body_reference_survives_collapse(self):
        """Remap custom body references at all tendon frequencies when fixed bodies collapse."""
        builder = newton.ModelBuilder()
        frequency = newton.Model.AttributeFrequency
        for name, freq in (
            ("route_body", frequency.TENDON),
            ("guide_body", frequency.TENDON_GUIDE),
            ("span_body", frequency.TENDON_SEGMENT),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    dtype=int,
                    frequency=freq,
                    assignment=newton.Model.AttributeAssignment.MODEL,
                    references="body",
                    default=-1,
                )
            )
        root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        other = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_joint_fixed(root, child)
        builder.add_tendon(
            [
                newton.TendonGuide(body=child, custom_attributes={"guide_body": child}),
                newton.TendonGuide(
                    body=other, custom_attributes={"guide_body": other}, segment_custom_attributes={"span_body": child}
                ),
            ],
            custom_attributes={"route_body": child},
        )
        builder.collapse_fixed_joints()
        model = builder.finalize(device="cpu")
        np.testing.assert_array_equal(model.route_body.numpy(), [0])
        np.testing.assert_array_equal(model.guide_body.numpy(), [0, 1])
        np.testing.assert_array_equal(model.span_body.numpy(), [0])

    def test_collapse_preserves_tendons_and_particle_attachments(self):
        """Retain world-fixed bodies needed by either tendon guides or particle attachments."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = wp.mat33(np.eye(3))
        tendon_anchor = builder.add_link(mass=1.0, inertia=inertia)
        particle_anchor = builder.add_link(xform=wp.transform(p=(1.0, 0.0, 0.0)), mass=1.0, inertia=inertia)
        endpoint = builder.add_body(xform=wp.transform(p=(2.0, 0.0, 0.0)), mass=1.0, inertia=inertia)
        for body in (tendon_anchor, particle_anchor):
            joint = builder.add_joint_fixed(-1, body, parent_xform=builder.body_q[body])
            builder.add_articulation([joint])
        particle = builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(), mass=1.0)
        builder.add_attachment_particle_body(particle=particle, body=particle_anchor)
        builder.add_tendon(
            [newton.TendonGuide(body=tendon_anchor), newton.TendonGuide(body=endpoint, compliance=1.0e-3)]
        )

        builder.collapse_fixed_joints()
        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.tendon_guide_body, [0, 2])
        self.assertEqual(builder.attachment_particle_body_body, [1])
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
                self.assertEqual(model.attachment_particle_body_count, 1)

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
                    expected_warning = (
                        self.assertWarnsRegex(DeprecationWarning, "rigid_compliant_alm=False is deprecated")
                        if options.get("rigid_compliant_alm") is False
                        else nullcontext()
                    )
                    with expected_warning:
                        solver = solver_type(model, iterations=2, **options)
                    state_0, state_1 = model.state(), model.state()
                    control = model.control()
                    initial_rest = solver.tendon_segment_rest_length.numpy().copy()
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
                            solver.tendon_segment_rest_length.numpy()[[0, 1, 4, 5]], initial_rest[[0, 1, 4, 5]]
                        )
                    changed = state_0.tendon_segment_rest_length.numpy().copy()
                    # The second slot of each bypassed route is an inactive
                    # placeholder, not another physical free span.
                    changed[::2] += np.float32(0.01)
                    state_0.tendon_segment_rest_length.assign(changed)
                    mask = wp.array([False, True, False, False], dtype=wp.bool, device=device)
                    solver.reset(state_0, world_mask=mask, flags=0)
                    expected = changed.copy()
                    expected[2:4] = initial_rest[2:4]
                    np.testing.assert_array_equal(state_0.tendon_segment_rest_length.numpy(), expected)
                    solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                    np.testing.assert_array_equal(state_1.tendon_segment_rest_length.numpy(), expected)

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
        self.assertEqual(destination.tendon_segment_compliance, [0.0, 1.0e-5, 2.0e-5, 1.0e-5, 2.0e-5])
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
                    expected_warning = (
                        self.assertWarnsRegex(DeprecationWarning, "rigid_compliant_alm=False is deprecated")
                        if options.get("rigid_compliant_alm") is False
                        else nullcontext()
                    )
                    with expected_warning:
                        solver = solver_type(model, iterations=1, **options)
                    state_0, state_1 = model.state(), model.state()
                    solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
                    np.testing.assert_allclose(solver.tendon_segment_rest_length.numpy(), [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
