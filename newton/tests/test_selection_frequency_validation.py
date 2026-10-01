# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton.tests.unittest_utils import assert_np_equal


@wp.kernel
def sum_values(values: wp.array3d[float], loss: wp.array[float]):
    world, articulation, value = wp.tid()
    wp.atomic_add(loss, 0, values[world, articulation, value])


def add_chain(builder, label, joint_count):
    bodies = [builder.add_link(label=f"{label}/body_{i}") for i in range(joint_count)]
    joints = [builder.add_joint_fixed(-1, bodies[0], label=f"{label}/joint_0")]
    for i in range(1, joint_count):
        joints.append(builder.add_joint_revolute(bodies[i - 1], bodies[i], label=f"{label}/joint_{i}"))
    builder.add_articulation(joints, label=label)


class TestSelectionFrequencyValidation(unittest.TestCase):
    def test_non_rectangular_frequency_is_independently_unavailable(self):
        """Keep joint and body access when shape counts differ in a partial view."""
        builder = newton.ModelBuilder()
        for index, shape_count in enumerate((1, 2)):
            body = builder.add_link(label=f"robot_{index}/root")
            for shape in range(shape_count):
                builder.add_shape_sphere(body, radius=0.1, label=f"robot_{index}/shape_{shape}")
            joint = builder.add_joint_free(parent=-1, child=body)
            builder.add_articulation([joint], label=f"robot_{index}")
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "SHAPE layout is unavailable"):
            ArticulationView(model, "robot_*")

        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertEqual(view.joint_count, 1)
        self.assertEqual(view.link_count, 1)
        self.assertIsNone(view.shape_count)
        self.assertIsNone(view.shapes_contiguous)
        self.assertEqual(view.get_attribute("joint_type", model).shape, (1, 2, 1))
        with self.assertRaises(AttributeError):
            view.get_attribute("shape_margin", model)
        with self.assertRaises(AttributeError):
            view.set_attribute("shape_margin", model, wp.zeros((1, 2, 1)))
        with self.assertRaisesRegex(AttributeError, "not_an_attribute"):
            view.get_attribute("not_an_attribute", model)

    def test_unfiltered_counts_validate_every_slot(self):
        """Count every selected articulation when validating cardinality."""
        builder = newton.ModelBuilder()
        for _world in range(2):
            builder.begin_world()
            add_chain(builder, "robot_0", 2)
            add_chain(builder, "robot_1", 4)
            builder.end_world()
        model = builder.finalize()

        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertIsNone(view.joint_count)
        self.assertIsNone(view.joint_names)
        with self.assertRaises(AttributeError):
            view.get_attribute("joint_type", model)

    def test_non_affine_origins_keep_uniform_metadata(self):
        """Preserve uniform counts when irregular origins disable access."""
        builder = newton.ModelBuilder()
        add_chain(builder, "target_0", 2)
        add_chain(builder, "other_0", 1)
        add_chain(builder, "target_1", 2)
        add_chain(builder, "other_1", 3)
        add_chain(builder, "target_2", 2)
        model = builder.finalize()

        starts = model.articulation_start.numpy()
        self.assertEqual(starts[[0, 2, 4]].tolist(), [0, 3, 8])
        view = ArticulationView(model, "target_*", allow_partial_layouts=True)
        self.assertEqual(view.joint_count, 2)
        self.assertEqual(len(view.joint_names), 2)
        self.assertIsNone(view.joints_contiguous)
        with self.assertRaises(AttributeError):
            view.get_attribute("joint_type", model)

    def test_different_sparse_patterns_keep_uniform_metadata(self):
        """Preserve uniform counts when physical gaps disable access."""
        builder = newton.ModelBuilder()
        a_root = builder.add_link(label="a/root")
        builder.add_link(label="unrelated_0")
        a_tip = builder.add_link(label="a/tip")
        b_root = builder.add_link(label="b/root")
        builder.add_link(label="unrelated_1")
        builder.add_link(label="unrelated_2")
        b_tip = builder.add_link(label="b/tip")
        a_j0 = builder.add_joint_fixed(-1, a_root)
        a_j1 = builder.add_joint_revolute(a_root, a_tip)
        builder.add_articulation([a_j0, a_j1], label="robot_a")
        b_j0 = builder.add_joint_fixed(-1, b_root)
        b_j1 = builder.add_joint_revolute(b_root, b_tip)
        builder.add_articulation([b_j0, b_j1], label="robot_b")
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "BODY layout is unavailable"):
            ArticulationView(model, "robot_*")
        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertEqual(view.link_count, 2)
        self.assertEqual(len(view.link_names), 2)
        self.assertIsNone(view.links_contiguous)
        with self.assertRaises(AttributeError):
            view.get_attribute("body_mass", model)

    def test_sparse_body_and_shape_offsets_preserve_values_writes_and_gradients(self):
        """Address the correct sparse rows for reads, writes, and gradients."""
        builder = newton.ModelBuilder()
        a_root = builder.add_link(label="a/root", mass=1.0)
        b_root = builder.add_link(label="b/root", mass=2.0)
        unrelated = builder.add_link(label="unrelated", mass=0.0)
        a_tip = builder.add_link(label="a/tip", mass=3.0)
        b_tip = builder.add_link(label="b/tip", mass=4.0)
        for body, margin in ((a_root, 0.01), (b_root, 0.02), (a_tip, 0.03), (b_tip, 0.04)):
            builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(margin=margin))
        a_j0 = builder.add_joint_fixed(-1, a_root)
        a_j1 = builder.add_joint_revolute(a_root, a_tip)
        builder.add_articulation([a_j0, a_j1], label="robot_a")
        b_j0 = builder.add_joint_fixed(-1, b_root)
        b_j1 = builder.add_joint_revolute(b_root, b_tip)
        builder.add_articulation([b_j0, b_j1], label="robot_b")
        model = builder.finalize(requires_grad=True)
        view = ArticulationView(model, "robot_*")

        masses = model.body_mass.numpy()
        assert_np_equal(
            view.get_attribute("body_mass", model).numpy(),
            [[masses[[a_root, a_tip]], masses[[b_root, b_tip]]]],
        )
        assert_np_equal(
            view.get_attribute("shape_margin", model).numpy(),
            np.array([[[0.01, 0.03], [0.02, 0.04]]], dtype=np.float32),
        )
        self.assertFalse(view.links_contiguous)
        self.assertFalse(view.shapes_contiguous)

        original = model.body_mass.numpy().copy()
        values = wp.array([[[10.0, 30.0], [20.0, 40.0]]], dtype=float)
        view.set_attribute("body_mass", model, values, mask=[[True, False]])
        updated = model.body_mass.numpy()
        assert_np_equal(updated[[a_root, a_tip]], np.array([10.0, 30.0], dtype=np.float32))
        assert_np_equal(updated[[b_root, b_tip, unrelated]], original[[b_root, b_tip, unrelated]])

        loss = wp.zeros(1, dtype=float, requires_grad=True)
        with wp.Tape() as tape:
            selected = view.get_attribute("body_mass", model)
            wp.launch(sum_values, dim=selected.shape, inputs=[selected], outputs=[loss])
        tape.backward(loss)
        assert_np_equal(model.body_mass.grad.numpy(), np.array([1, 1, 0, 1, 1], dtype=np.float32))

    def test_contiguous_flags_include_inter_articulation_gaps(self):
        """Report locally contiguous blocks as non-contiguous when physical gaps separate them."""
        builder = newton.ModelBuilder()
        a_root = builder.add_link(label="a/root")
        a_tip = builder.add_link(label="a/tip")
        unrelated = builder.add_link(label="unrelated")
        b_root = builder.add_link(label="b/root")
        b_tip = builder.add_link(label="b/tip")
        for body in (a_root, a_tip, unrelated, b_root, b_tip):
            builder.add_shape_sphere(body, radius=0.1)

        a_j0 = builder.add_joint_fixed(-1, a_root)
        a_j1 = builder.add_joint_revolute(a_root, a_tip)
        builder.add_joint_revolute(-1, unrelated)
        b_j0 = builder.add_joint_fixed(-1, b_root)
        b_j1 = builder.add_joint_revolute(b_root, b_tip)
        builder.add_articulation([a_j0, a_j1], label="robot_a")
        builder.add_articulation([b_j0, b_j1], label="robot_b")
        model = builder.finalize(device="cpu")
        view = ArticulationView(model, "robot_*")

        self.assertFalse(view.joints_contiguous)
        self.assertFalse(view.joint_dofs_contiguous)
        self.assertFalse(view.joint_coords_contiguous)
        self.assertFalse(view.links_contiguous)
        self.assertFalse(view.shapes_contiguous)
        for values in (
            view.get_attribute("joint_type", model),
            view.get_dof_positions(model),
            view.get_dof_velocities(model),
            view.get_attribute("body_mass", model),
            view.get_attribute("shape_margin", model),
        ):
            self.assertFalse(values.is_contiguous)

    def test_large_body_gap_keeps_independent_shape_layout(self):
        """Keep shape access when a large body gap changes but shape ownership matches."""
        builder = newton.ModelBuilder()
        for label, filler_bodies, shapes_per_filler in (("a", 256, 1), ("b", 128, 2)):
            root = builder.add_link(label=f"{label}/root")
            builder.add_shape_sphere(root, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(margin=0.01))
            for _ in range(filler_bodies):
                filler = builder.add_link()
                for _ in range(shapes_per_filler):
                    builder.add_shape_sphere(filler, radius=0.1)
            tip = builder.add_link(label=f"{label}/tip")
            builder.add_shape_sphere(tip, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(margin=0.02))
            joints = [builder.add_joint_fixed(-1, root), builder.add_joint_revolute(root, tip)]
            builder.add_articulation(joints, label=f"robot_{label}")
        model = builder.finalize()

        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertEqual(view.link_count, 2)
        self.assertEqual(view.shape_count, 2)
        with self.assertRaises(AttributeError):
            view.get_attribute("body_mass", model)
        assert_np_equal(
            view.get_attribute("shape_margin", model).numpy(),
            np.array([[[0.01, 0.02], [0.01, 0.02]]], dtype=np.float32),
        )

        root_view = ArticulationView(model, "robot_*", include_links=[0])
        self.assertEqual(root_view.get_attribute("body_mass", model).shape, (1, 2, 1))
        self.assertEqual(root_view.get_attribute("shape_margin", model).shape, (1, 2, 1))

    def test_filtered_subset_ignores_excluded_shape_difference(self):
        """Keep a selected subset available when excluded links differ."""
        builder = newton.ModelBuilder()
        for extra_tip_shape in (False, True):
            builder.begin_world()
            root = builder.add_link(label="robot/root")
            tip = builder.add_link(label="robot/tip")
            builder.add_shape_sphere(root, radius=0.1)
            builder.add_shape_sphere(tip, radius=0.1)
            if extra_tip_shape:
                builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
            j0 = builder.add_joint_fixed(-1, root)
            j1 = builder.add_joint_revolute(root, tip)
            builder.add_articulation([j0, j1], label="robot")
            builder.end_world()
        model = builder.finalize()

        view = ArticulationView(model, "robot", include_links=[0])
        self.assertEqual(view.shape_count, 1)
        self.assertEqual(view.get_attribute("shape_margin", model).shape, (2, 1, 1))

    def test_selected_dof_origins_ignore_excluded_root_width(self):
        """Measure selected coordinate and DOF offsets from their first selected rows."""
        builder = newton.ModelBuilder()
        for floating in (True, False):
            builder.begin_world()
            root = builder.add_link(label="robot/root")
            tip = builder.add_link(label="robot/tip")
            if floating:
                root_joint = builder.add_joint_free(child=root, label="robot/root")
            else:
                root_joint = builder.add_joint_revolute(-1, root, label="robot/root")
            hinge = builder.add_joint_revolute(root, tip, label="robot/hinge")
            builder.add_articulation([root_joint, hinge], label="robot")
            builder.end_world()
        model = builder.finalize(device="cpu")

        view = ArticulationView(model, "robot", include_joints=["hinge"], allow_partial_layouts=True)
        self.assertEqual(view.get_dof_positions(model).shape, (2, 1, 1))
        self.assertEqual(view.get_dof_velocities(model).shape, (2, 1, 1))

    def test_selected_shape_origins_ignore_excluded_prefix_shapes(self):
        """Measure selected shape offsets from the first shape on a selected link."""
        builder = newton.ModelBuilder()
        for world, root_shape_count in enumerate((1, 2)):
            builder.begin_world()
            root = builder.add_link(label="robot/root")
            tip = builder.add_link(label="robot/tip")
            for _ in range(root_shape_count):
                builder.add_shape_sphere(root, radius=0.1)
            builder.add_shape_sphere(
                tip,
                radius=0.1,
                cfg=newton.ModelBuilder.ShapeConfig(margin=0.2 + 0.1 * world),
            )
            root_joint = builder.add_joint_fixed(-1, root)
            tip_joint = builder.add_joint_revolute(root, tip)
            builder.add_articulation([root_joint, tip_joint], label="robot")
            builder.end_world()
        model = builder.finalize(device="cpu")

        view = ArticulationView(model, "robot", include_links=["tip"])
        assert_np_equal(
            view.get_attribute("shape_margin", model).numpy(),
            np.array([[[0.2]], [[0.3]]], dtype=np.float32),
        )

    def test_frequency_sequences_include_joint_relationships(self):
        """Validate joint DOF and coordinate ownership separately."""
        builder = newton.ModelBuilder()
        for label, reverse in (("robot_a", False), ("robot_b", True)):
            root = builder.add_link(label=f"{label}/root")
            middle = builder.add_link(label=f"{label}/middle")
            tip = builder.add_link(label=f"{label}/tip")
            j0 = builder.add_joint_fixed(-1, root)
            if reverse:
                j1 = builder.add_joint_ball(root, middle)
                j2 = builder.add_joint_revolute(middle, tip)
            else:
                j1 = builder.add_joint_revolute(root, middle)
                j2 = builder.add_joint_ball(middle, tip)
            builder.add_articulation([j0, j1, j2], label=label)
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "JOINT_DOF layout is unavailable"):
            ArticulationView(model, "robot_*")
        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertEqual(view.joint_dof_count, 4)
        self.assertEqual(view.joint_coord_count, 5)
        with self.assertRaises(AttributeError):
            view.get_dof_velocities(model)
        with self.assertRaises(AttributeError):
            view.get_dof_positions(model)
        self.assertIsNone(view.joint_dof_counts)
        self.assertIsNone(view.joint_coord_counts)

    def test_shape_sequence_includes_link_relationship(self):
        """Check the owning link position when validating shape layouts."""
        builder = newton.ModelBuilder()
        a_root = builder.add_link(label="a/root")
        a_tip = builder.add_link(label="a/tip")
        b_root = builder.add_link(label="b/root")
        b_tip = builder.add_link(label="b/tip")
        builder.add_shape_sphere(a_root, radius=0.1)
        builder.add_shape_box(a_root, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_sphere(b_tip, radius=0.1)
        builder.add_shape_box(b_tip, hx=0.1, hy=0.1, hz=0.1)
        a_j0 = builder.add_joint_fixed(-1, a_root)
        a_j1 = builder.add_joint_revolute(a_root, a_tip)
        builder.add_articulation([a_j0, a_j1], label="robot_a")
        b_j0 = builder.add_joint_fixed(-1, b_root)
        b_j1 = builder.add_joint_revolute(b_root, b_tip)
        builder.add_articulation([b_j0, b_j1], label="robot_b")
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "SHAPE layout is unavailable"):
            ArticulationView(model, "robot_*")
        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertEqual(view.link_count, 2)
        self.assertEqual(view.shape_count, 2)
        self.assertEqual(view.get_attribute("body_mass", model).shape, (1, 2, 2))
        with self.assertRaises(AttributeError):
            view.get_attribute("shape_margin", model)
        self.assertIsNone(view.link_shapes)
        self.assertIsNone(view.body_shapes)

    def test_root_gate_uses_common_behavior(self):
        """Allow root access for common behavior when root types differ."""
        builder = newton.ModelBuilder()
        free_body = builder.add_link(label="free/root")
        free_joint = builder.add_joint_free(parent=-1, child=free_body)
        builder.add_articulation([free_joint], label="robot_free")
        distance_body = builder.add_link(label="distance/root")
        distance_joint = builder.add_joint_distance(parent=-1, child=distance_body)
        builder.add_articulation([distance_joint], label="robot_distance")
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "root metadata differs"):
            ArticulationView(model, "robot_*")
        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertIsNone(view.root_joint_type)
        self.assertTrue(view.is_floating_base)
        self.assertFalse(view.is_fixed_base)
        self.assertEqual(view.get_root_transforms(model).shape, (1, 2))
        self.assertEqual(view.get_root_velocities(model).shape, (1, 2))

    def test_mixed_root_behavior_only_disables_root_access(self):
        """Keep unrelated layouts available for mixed fixed and floating roots."""
        builder = newton.ModelBuilder()
        floating_body = builder.add_link()
        floating_joint = builder.add_joint_free(parent=-1, child=floating_body)
        builder.add_articulation([floating_joint], label="robot_floating")
        fixed_body = builder.add_link()
        fixed_joint = builder.add_joint_fixed(-1, fixed_body)
        builder.add_articulation([fixed_joint], label="robot_fixed")
        model = builder.finalize()

        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertIsNone(view.root_joint_type)
        self.assertIsNone(view.is_fixed_base)
        self.assertIsNone(view.is_floating_base)
        with self.assertRaises(AttributeError):
            view.get_root_transforms(model)
        self.assertEqual(view.get_attribute("joint_type", model).shape, (1, 2, 1))
        with self.assertRaises(AttributeError):
            view.set_root_transforms(model, wp.zeros((1, 2), dtype=wp.transform))

    def test_fixed_root_transform_setter_keeps_public_shape(self):
        """Keep the public transform tensor shape for fixed root writes."""
        builder = newton.ModelBuilder()
        for index in range(2):
            body = builder.add_link()
            joint = builder.add_joint_fixed(-1, body)
            builder.add_articulation([joint], label=f"robot_{index}")
        model = builder.finalize()
        view = ArticulationView(model, "robot_*")

        original = view.get_root_transforms(model).numpy().copy()
        values = wp.array(
            [[wp.transform((1.0, 2.0, 3.0), wp.quat_identity()), wp.transform_identity()]],
            dtype=wp.transform,
        )
        view.set_root_transforms(model, values, mask=wp.array([[True, False]], dtype=bool))

        actual = view.get_root_transforms(model).numpy()
        assert_np_equal(actual[0, 0], values.numpy()[0, 0])
        assert_np_equal(actual[0, 1], original[0, 1])

    def test_fixed_root_transform_round_trip_with_strided_values(self):
        """Accept the strided array returned by the fixed-root getter."""
        builder = newton.ModelBuilder()
        for index in range(2):
            root = builder.add_link()
            tip = builder.add_link()
            joints = [builder.add_joint_fixed(-1, root), builder.add_joint_fixed(root, tip)]
            builder.add_articulation(joints, label=f"robot_{index}")

        model = builder.finalize(device="cpu")
        view = ArticulationView(model, "robot_*")
        roots = view.get_root_transforms(model)
        self.assertEqual(roots.shape, (1, 2))
        expected = roots.numpy().copy()
        view.set_root_transforms(model, roots)
        assert_np_equal(view.get_root_transforms(model).numpy(), expected)

        if wp.get_cuda_device_count():
            source_model = builder.finalize(device="cuda:0")
            target_model = builder.finalize(device="cuda:0")
            source_view = ArticulationView(source_model, "robot_*")
            target_view = ArticulationView(target_model, "robot_*")
            target_view.set_root_transforms(
                target_model,
                wp.array(
                    [[wp.transform((1.0, 2.0, 3.0), wp.quat_identity()), wp.transform_identity()]],
                    dtype=wp.transform,
                    device="cuda:0",
                ),
            )
            source_roots = source_view.get_root_transforms(source_model)
            target_view.set_root_transforms(target_model, source_roots)
            assert_np_equal(target_view.get_root_transforms(target_model).numpy(), source_roots.numpy())

    def test_empty_exclusions_do_not_hide_extra_joints_or_links(self):
        """An exclusion that matches nothing still checks complete layouts."""
        builder = newton.ModelBuilder()
        add_chain(builder, "robot_0", 2)
        add_chain(builder, "robot_1", 3)
        model = builder.finalize(device="cpu")

        for kwargs in ({"exclude_joints": []}, {"exclude_joint_types": []}, {"exclude_links": []}):
            with self.subTest(kwargs=kwargs):
                view = ArticulationView(model, "robot_*", allow_partial_layouts=True, **kwargs)
                self.assertIsNone(view.joint_count)
                self.assertIsNone(view.link_count)

    def test_root_gate_is_atomic(self):
        """Require both root layouts for root transform and velocity access."""
        builder = newton.ModelBuilder()

        def add_free(label):
            body = builder.add_link()
            joint = builder.add_joint_free(parent=-1, child=body)
            builder.add_articulation([joint], label=label)

        add_free("target_0")
        ball_body = builder.add_link()
        ball_joint = builder.add_joint_ball(-1, ball_body)
        builder.add_articulation([ball_joint], label="filler_ball")
        add_free("target_1")
        bodies = [builder.add_link() for _ in range(4)]
        joints = [builder.add_joint_revolute(-1, bodies[0])]
        for i in range(1, 4):
            joints.append(builder.add_joint_revolute(bodies[i - 1], bodies[i]))
        builder.add_articulation(joints, label="filler_revolute")
        add_free("target_2")
        model = builder.finalize()

        view = ArticulationView(model, "target_*", allow_partial_layouts=True)
        self.assertEqual(view.get_dof_positions(model).shape, (1, 3, 7))
        with self.assertRaises(AttributeError):
            view.get_dof_velocities(model)
        with self.assertRaises(AttributeError):
            view.get_root_transforms(model)
        with self.assertRaises(AttributeError):
            view.get_root_velocities(model)

    def test_unequal_world_totals_do_not_replace_affine_validation(self):
        """Keep a valid layout when unrelated world shape totals differ."""
        builder = newton.ModelBuilder()
        for world in range(3):
            builder.begin_world()
            body = builder.add_link(label="robot/root")
            builder.add_shape_sphere(body, radius=0.1)
            joint = builder.add_joint_fixed(-1, body)
            builder.add_articulation([joint], label="robot")
            if world == 2:
                builder.add_shape_sphere(-1, radius=0.1)
            builder.end_world()
        model = builder.finalize()

        self.assertEqual(np.diff(model.shape_world_start.numpy()[:4]).tolist(), [1, 1, 2])
        view = ArticulationView(model, "robot")
        self.assertEqual(view.get_attribute("shape_margin", model).shape, (3, 1, 1))

    def test_nonowned_joint_inside_articulation_range_does_not_own_its_body(self):
        """Body ownership follows joint_articulation even for a standalone root joint."""
        builder = newton.ModelBuilder()
        first_tip_joint = None
        for index in range(2):
            root = builder.add_link(label=f"robot_{index}/root")
            tip = builder.add_link(label=f"robot_{index}/tip")
            builder.add_shape_sphere(root, radius=0.1)
            builder.add_shape_sphere(tip, radius=0.1)
            root_joint = builder.add_joint_free(child=root)
            tip_joint = builder.add_joint_revolute(root, tip)
            builder.add_articulation([root_joint, tip_joint], label=f"robot_{index}")
            if index == 0:
                first_tip_joint = tip_joint
        builder.joint_articulation[first_tip_joint] = -1
        builder.joint_parent[first_tip_joint] = -1
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "BODY layout is unavailable"):
            ArticulationView(model, "robot_*")
        view = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertEqual(view.joint_count, 2)
        self.assertIsNone(view.link_count)
        self.assertIsNone(view.shape_count)

    def test_structural_attribute_setter_refreshes_future_view_metadata(self):
        """A new view observes structural writes through the public setter."""
        builder = newton.ModelBuilder()
        for index in range(2):
            body = builder.add_link(label=f"robot_{index}/root")
            joint = builder.add_joint_free(child=body)
            builder.add_articulation([joint], label=f"robot_{index}")
        model = builder.finalize()
        view = ArticulationView(model, "robot_*")
        self.assertEqual(view.root_joint_type, newton.JointType.FREE)

        joint_types = view.get_attribute("joint_type", model).numpy().copy()
        joint_types[0, 1, 0] = newton.JointType.DISTANCE
        view.set_attribute("joint_type", model, joint_types)

        refreshed = ArticulationView(model, "robot_*", allow_partial_layouts=True)
        self.assertIsNone(refreshed.root_joint_type)

    def test_size_one_axes_keep_views_contiguous(self):
        """Views with one world or one articulation per world stay contiguous."""
        robot = newton.ModelBuilder()
        root = robot.add_link(label="robot/root")
        tip = robot.add_link(label="robot/tip")
        root_joint = robot.add_joint_free(child=root)
        tip_joint = robot.add_joint_revolute(root, tip)
        robot.add_articulation([root_joint, tip_joint], label="robot")

        replicated = newton.ModelBuilder()
        replicated.replicate(robot, world_count=3)
        single = newton.ModelBuilder()
        single.add_builder(robot)
        shared_world = newton.ModelBuilder()
        for _ in range(2):
            shared_world.add_builder(robot)

        for builder, count_per_world in ((replicated, 1), (single, 1), (shared_world, 2)):
            model = builder.finalize()
            state = model.state()
            view = ArticulationView(model, "robot")
            self.assertEqual(view.count_per_world, count_per_world)
            for values, flat in (
                (view.get_dof_positions(state), state.joint_q),
                (view.get_link_transforms(state), state.body_q),
            ):
                self.assertTrue(values.is_contiguous)
                assert_np_equal(values.flatten().numpy(), flat.numpy())

    def test_root_metadata_uses_python_types(self):
        """Root flags are Python booleans rather than NumPy scalars."""
        builder = newton.ModelBuilder()
        for index in range(2):
            add_chain(builder, f"robot_{index}", 2)
        view = ArticulationView(builder.finalize(), "robot_*")
        self.assertIs(view.is_fixed_base, True)
        self.assertIs(view.is_floating_base, False)

    def test_unsupported_frequency_is_not_reported_unavailable(self):
        """Attributes outside articulation frequencies raise a plain AttributeError."""
        builder = newton.ModelBuilder()
        for index in range(2):
            add_chain(builder, f"robot_{index}", 2)
        model = builder.finalize()
        view = ArticulationView(model, "robot_*")
        with self.assertRaisesRegex(AttributeError, "Unable to determine the layout"):
            view.get_attribute("articulation_start", model)


if __name__ == "__main__":
    unittest.main()
