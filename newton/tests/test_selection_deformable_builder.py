# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable views after builder composition and fixed-joint collapse."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import (
    ArticulationView,
    DeformableCurveView,
    DeformableSurfaceView,
    DeformableVolumeView,
)
from newton.tests._selection_deformable_test_utils import (
    _CABLE_PTS,
    _add_test_anchored_cable,
    _add_test_articulation,
    _add_test_cable,
    _add_test_cloth,
    _add_test_soft_body,
    _add_test_soft_grid,
)


class TestDeformableBuilderObjects(unittest.TestCase):
    """Select deformable objects recorded by native builder calls without requiring USD."""

    def test_builder_deformable_identities_are_public_and_mutable(self):
        """Applications can rebase deformable labels before finalization."""
        builder = newton.ModelBuilder()
        _add_test_cable(builder, label="source_curve")
        _add_test_cloth(builder, label="source_surface")
        _add_test_soft_body(builder, label="source_volume")

        self.assertEqual(builder.curve_label, ["source_curve"])
        self.assertEqual(builder.curve_world, [-1])
        self.assertEqual(builder.surface_label, ["source_surface"])
        self.assertEqual(builder.surface_world, [-1])
        self.assertEqual(builder.volume_label, ["source_volume"])
        self.assertEqual(builder.volume_world, [-1])

        builder.curve_label[:] = ["/World/envs/env_0/Cable"]
        builder.surface_label = ["/World/envs/env_0/Cloth"]
        builder.volume_label[:] = ["/World/envs/env_0/SoftBody"]

        model = builder.finalize(device="cpu")
        for label, view_type in (
            ("/World/envs/env_0/Cable", DeformableCurveView),
            ("/World/envs/env_0/Cloth", DeformableSurfaceView),
            ("/World/envs/env_0/SoftBody", DeformableVolumeView),
        ):
            with self.subTest(label=label):
                view = view_type(model, label)
                self.assertEqual((view.labels, view.worlds), ([label], [-1]))

    def test_unlabeled_curve_builders_get_default_object_labels(self):
        """Both Rod topology forms record a deformable object without an explicit label."""
        rod_builder = newton.ModelBuilder()
        rod_builder.add_rod(
            rod=newton.Rod(_CABLE_PTS, radius=0.02),
            body_frame_origin="com",
        )
        rod = DeformableCurveView(rod_builder.finalize(), "curve_0")
        self.assertEqual((rod.labels, rod.bodies_per_deformable_object), (["curve_0"], 3))

        graph_builder = newton.ModelBuilder()
        graph_builder.add_rod(
            rod=newton.Rod(_CABLE_PTS, edges=[(0, 1), (1, 2), (2, 3)], radius=0.02),
            body_frame_origin="com",
        )
        graph = DeformableCurveView(graph_builder.finalize(), "curve_0")
        self.assertEqual((graph.labels, graph.bodies_per_deformable_object), (["curve_0"], 3))

    def test_unlabeled_volume_builders_get_default_object_labels(self):
        """Mesh and grid volume constructors record a deformable object without an explicit label."""
        mesh_builder = newton.ModelBuilder()
        _add_test_soft_body(mesh_builder, label=None)
        mesh = DeformableVolumeView(mesh_builder.finalize(), "volume_0")
        self.assertEqual((mesh.labels, mesh.particles_per_deformable_object), (["volume_0"], 4))

        grid_builder = newton.ModelBuilder()
        _add_test_soft_grid(grid_builder, pos=wp.vec3(0.0, 0.0, 1.0))
        grid = DeformableVolumeView(grid_builder.finalize(), "volume_0")
        self.assertEqual((grid.labels, grid.particles_per_deformable_object), (["volume_0"], 8))

    def test_default_object_labels_survive_replication(self):
        """Generated identities remain selectable after builder replication."""
        prototype = newton.ModelBuilder()
        _add_test_cable(prototype, label=None)
        _add_test_cloth(prototype, label=None)
        _add_test_soft_body(prototype, label=None)

        scene = newton.ModelBuilder()
        scene.replicate(prototype, 2)
        model = scene.finalize(device="cpu")

        for label, view_type in (
            ("curve_0", DeformableCurveView),
            ("surface_0", DeformableSurfaceView),
            ("volume_0", DeformableVolumeView),
        ):
            with self.subTest(label=label):
                view = view_type(model, label)
                self.assertEqual((view.count, view.labels, view.worlds), (2, [label, label], [0, 1]))

    def test_rebased_builder_identities_survive_replication(self):
        """Cloning keeps application-rebased labels and assigns destination worlds."""
        prototype = newton.ModelBuilder()
        _add_test_cable(prototype, label="source_curve")
        _add_test_cloth(prototype, label="source_surface")
        _add_test_soft_body(prototype, label="source_volume")

        prototype.curve_label[0] = "/Template/Cable"
        prototype.surface_label[0] = "/Template/Cloth"
        prototype.volume_label[0] = "/Template/SoftBody"

        scene = newton.ModelBuilder()
        scene.replicate(prototype, 2)

        for labels, worlds, expected_label in (
            (scene.curve_label, scene.curve_world, "/Template/Cable"),
            (scene.surface_label, scene.surface_world, "/Template/Cloth"),
            (scene.volume_label, scene.volume_world, "/Template/SoftBody"),
        ):
            self.assertEqual(labels, [expected_label, expected_label])
            self.assertEqual(worlds, [0, 1])

        model = scene.finalize(device="cpu")
        for label, view_type in (
            ("/Template/Cable", DeformableCurveView),
            ("/Template/Cloth", DeformableSurfaceView),
            ("/Template/SoftBody", DeformableVolumeView),
        ):
            with self.subTest(label=label):
                view = view_type(model, label)
                self.assertEqual((view.count, view.labels, view.worlds), (2, [label, label], [0, 1]))

    def test_labeled_curve_builders_record_one_complete_object(self):
        """Public rod builders hide their nested construction from deformable object selection."""
        closed_builder = newton.ModelBuilder()
        closed_builder.add_rod(
            rod=newton.Rod(
                [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.0, 0.0, 1.0)],
                radius=0.02,
                closed=True,
            ),
            label="closed",
            body_frame_origin="com",
        )
        closed = DeformableCurveView(closed_builder.finalize(), "closed")
        self.assertEqual((closed.count, closed.elements_per_deformable_object("body")), (1, 3))
        # Native deformable objects include the automatic free root as well as the rod joints.
        self.assertEqual(closed.elements_per_deformable_object("joint"), 4)
        self.assertEqual(closed_builder.joint_type[0], newton.JointType.FREE)

        graph_builder = newton.ModelBuilder()
        graph_builder.add_rod(
            rod=newton.Rod(
                [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.1, 0.1, 1.0)],
                edges=[(0, 1), (1, 2), (1, 3)],
                radius=0.02,
            ),
            label="graph",
            body_frame_origin="com",
        )
        graph = DeformableCurveView(graph_builder.finalize(), "graph")
        self.assertEqual((graph.count, graph.elements_per_deformable_object("body")), (1, 3))
        self.assertEqual(graph.elements_per_deformable_object("joint"), 3)
        self.assertEqual(graph_builder.joint_type[0], newton.JointType.FREE)

    def test_deprecated_curve_inputs_still_record_objects(self):
        """Deprecated rod inputs keep their deformable-object selection behavior."""
        chain_builder = newton.ModelBuilder()
        with self.assertWarns(DeprecationWarning):
            chain_builder.add_rod(
                positions=_CABLE_PTS,
                radius=0.02,
                label="legacy_chain",
                body_frame_origin="com",
            )
        chain = DeformableCurveView(chain_builder.finalize(), "legacy_chain")
        self.assertEqual((chain.count, chain.bodies_per_deformable_object), (1, 3))

        graph_builder = newton.ModelBuilder()
        with self.assertWarns(DeprecationWarning):
            graph_builder.add_rod_graph(
                node_positions=_CABLE_PTS,
                edges=[(0, 1), (1, 2), (2, 3)],
                radius=0.02,
                label="legacy_graph",
                body_frame_origin="com",
            )
        graph = DeformableCurveView(graph_builder.finalize(), "legacy_graph")
        self.assertEqual((graph.count, graph.bodies_per_deformable_object), (1, 3))

    def test_mixed_native_deformables_replicate_with_offset_ranges(self):
        """Curve, surface, and volume deformable objects retain disjoint ranges after replication."""
        prototype = newton.ModelBuilder()

        # Curve first: three segment bodies, two rod joints, and a free root.
        prototype.add_rod(
            rod=newton.Rod(
                [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.1, 0.1, 1.0)],
                edges=[(0, 1), (1, 2), (1, 3)],
                radius=0.02,
            ),
            label="curve",
            body_frame_origin="com",
        )

        # Surface second: four particles, two triangles, and five cloth edges.
        _add_test_cloth(prototype, label="surface")

        # Volume last: eight particles and five tetrahedra.
        _add_test_soft_grid(prototype, pos=wp.vec3(0.0, 0.0, 3.0), label="volume")

        scene = newton.ModelBuilder()
        scene.replicate(prototype, 2)
        model = scene.finalize(device="cpu")
        state = model.state()

        curve = DeformableCurveView(model, "curve")
        self.assertEqual((curve.count, curve.worlds, curve.count_per_world), (2, [0, 1], 1))
        np.testing.assert_array_equal(curve.deformable_object_boundaries.numpy(), [0, 1, 2])
        self.assertEqual(curve.ranges("body"), [(0, 3), (3, 6)])
        self.assertEqual(curve.ranges("joint"), [(0, 3), (3, 6)])
        self.assertEqual(curve.get_body_transforms(state).shape, (2, 3))

        surface = DeformableSurfaceView(model, "surface")
        self.assertEqual((surface.count, surface.worlds, surface.count_per_world), (2, [0, 1], 1))
        np.testing.assert_array_equal(surface.deformable_object_boundaries.numpy(), [0, 1, 2])
        self.assertEqual(surface.ranges("particle"), [(0, 4), (12, 16)])
        self.assertEqual(surface.ranges("triangle"), [(0, 2), (14, 16)])
        self.assertEqual(surface.ranges("edge"), [(0, 5), (23, 28)])
        self.assertEqual(surface.get_particle_positions(state).shape, (2, 4))

        volume = DeformableVolumeView(model, "volume")
        self.assertEqual((volume.count, volume.worlds, volume.count_per_world), (2, [0, 1], 1))
        np.testing.assert_array_equal(volume.deformable_object_boundaries.numpy(), [0, 1, 2])
        self.assertEqual(volume.ranges("particle"), [(4, 12), (16, 24)])
        self.assertEqual(volume.ranges("tetrahedron"), [(0, 5), (5, 10)])
        self.assertEqual(volume.get_particle_positions(state).shape, (2, 8))

    def test_builder_built_prototype_clones_select_per_world(self):
        """A labeled soft body and rod built in a prototype and cloned per world with
        add_world stay selectable, with correctly offset ranges (the Isaac Lab pattern
        of building deformables in per-world builder hooks)."""
        proto = newton.ModelBuilder()
        proto.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 3],
            density=100.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            label="soft_proto",
        )
        proto.add_rod(
            rod=newton.Rod([(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)], radius=0.02),
            label="cable_proto",
            wrap_in_articulation=True,
            body_frame_origin="com",
        )

        scene = newton.ModelBuilder()
        scene.add_world(proto)
        scene.add_world(proto)
        model = scene.finalize()
        state = model.state()

        soft = DeformableVolumeView(model, "soft_proto")
        self.assertEqual((soft.count, soft.worlds, soft.particles_per_deformable_object), (2, [0, 1], 4))
        (r0, r1) = soft.ranges("particle")
        self.assertEqual(r1[0] - r0[0], 4)
        self.assertNotEqual(r0, r1)

        cable = DeformableCurveView(model, "cable_proto")
        self.assertEqual((cable.count, cable.worlds, cable.bodies_per_deformable_object), (2, [0, 1], 2))
        self.assertEqual(cable.elements_per_deformable_object("joint"), 2)

        # State access round-trips through the offset ranges.
        positions = soft.get_particle_positions(state)
        lifted = positions.numpy().copy()
        lifted[1, :, 2] += 3.0
        soft.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3))
        np.testing.assert_allclose(soft.get_particle_positions(state).numpy(), lifted, atol=1e-6)

    def test_unlabeled_cloth_gets_a_default_object_label(self):
        """A cloth deformable object exists even when the caller does not provide its label."""
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            indices=[0, 1, 2, 0, 2, 3],
            density=0.1,
        )
        model = builder.finalize()
        view = DeformableSurfaceView(model, "surface_0")
        self.assertEqual(view.labels, ["surface_0"])
        self.assertEqual(view.ranges("particle"), [(0, 4)])

    def test_fixed_joint_collapse_drops_incomplete_curve_object(self):
        """A label does not prevent collapse; an incomplete curve is not selectable."""
        builder = newton.ModelBuilder()
        _add_test_anchored_cable(builder)

        with self.assertWarnsRegex(UserWarning, "anchored_curve.*joints_to_keep"):
            builder.collapse_fixed_joints()

        self.assertEqual((builder.body_count, builder.joint_count), (1, 1))
        model = builder.finalize()

        with self.assertRaisesRegex(KeyError, "anchored_curve"):
            DeformableCurveView(model, "anchored_curve")

    def test_fixed_joint_collapse_is_label_neutral(self):
        """Otherwise identical labeled and unlabeled rods collapse identically."""

        unlabeled = newton.ModelBuilder()
        _add_test_anchored_cable(unlabeled, label=None)
        with self.assertWarnsRegex(UserWarning, "curve_0.*joints_to_keep"):
            unlabeled.collapse_fixed_joints()
        labeled = newton.ModelBuilder()
        _add_test_anchored_cable(labeled)
        with self.assertWarnsRegex(UserWarning, "anchored_curve.*joints_to_keep"):
            labeled.collapse_fixed_joints()

        self.assertEqual((labeled.body_count, labeled.joint_count), (unlabeled.body_count, unlabeled.joint_count))
        self.assertEqual(labeled.joint_type, unlabeled.joint_type)

    def test_fixed_joint_collapse_preserves_explicitly_kept_curve(self):
        """joints_to_keep retains a complete curve deformable object when requested."""
        builder = newton.ModelBuilder()
        _add_test_anchored_cable(builder)

        self.assertEqual((builder.curve_label, builder.curve_world), (["anchored_curve"], [-1]))
        builder.collapse_fixed_joints(joints_to_keep=["anchor"])
        self.assertEqual((builder.curve_label, builder.curve_world), (["anchored_curve"], [-1]))
        model = builder.finalize()
        view = DeformableCurveView(model, "anchored_curve")

        self.assertEqual((model.body_count, model.joint_count), (2, 2))
        self.assertEqual(view.ranges("body"), [(0, 2)])
        self.assertEqual(view.get_body_transforms(model.state()).shape, (1, 2))

    def test_labeled_soft_grid_is_selectable(self):
        """A labeled soft grid records one selectable volume deformable object."""
        builder = newton.ModelBuilder()
        _add_test_soft_grid(builder, pos=wp.vec3(0.0, 0.0, 0.0), label="soft_grid")

        model = builder.finalize()
        view = DeformableVolumeView(model, "soft_grid")

        self.assertEqual(view.ranges("particle"), [(0, 8)])
        self.assertEqual(view.ranges("tetrahedron"), [(0, 5)])


class TestDeformableAndArticulationViews(unittest.TestCase):
    """Rigid and deformable selections sharing finalized mixed models."""

    def test_rigid_cable_cloth_and_volume_survive_unrelated_collapse(self):
        """All selection families coexist after an unrelated fixed joint collapses."""
        builder = newton.ModelBuilder()
        collapsed_body = builder.add_link(label="collapsed_body")
        builder.add_joint_fixed(parent=-1, child=collapsed_body, label="collapse_me")
        _add_test_articulation(builder)
        cable_bodies, _cable_joints = builder.add_rod(
            rod=newton.Rod([(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)], radius=0.02),
            label="cable",
            body_frame_origin="com",
        )
        _add_test_cloth(builder)
        _add_test_soft_body(builder)

        body_count = builder.body_count
        joint_count = builder.joint_count
        builder.collapse_fixed_joints()
        self.assertEqual(builder.body_count, body_count - 1)
        self.assertEqual(builder.joint_count, joint_count - 1)

        model = builder.finalize()
        state = model.state()
        rigid = ArticulationView(model, "robot")
        cable = DeformableCurveView(model, "cable")
        cloth = DeformableSurfaceView(model, "cloth")
        soft = DeformableVolumeView(model, "soft")

        self.assertEqual(rigid.get_root_transforms(state).shape, (1, 1))
        self.assertEqual(cable.get_body_transforms(state).shape, (1, 2))
        self.assertEqual(cloth.get_particle_positions(state).shape, (1, 4))
        self.assertEqual(soft.get_particle_positions(state).shape, (1, 4))
        self.assertEqual(cable.ranges("body"), [(cable_bodies[0] - 1, cable_bodies[-1])])

        rigid_values = rigid.get_root_transforms(state).numpy().copy()
        cable_values = cable.get_body_transforms(state).numpy().copy()
        cloth_values = cloth.get_particle_positions(state).numpy().copy()
        soft_values = soft.get_particle_positions(state).numpy().copy()

        moved_rigid = rigid_values.copy()
        moved_rigid[..., 0] += 1.0
        rigid.set_root_transforms(state, wp.array(moved_rigid, dtype=wp.transform, device=model.device))
        np.testing.assert_array_equal(cable.get_body_transforms(state).numpy(), cable_values)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_cable = cable_values.copy()
        moved_cable[..., 1] += 1.0
        cable.set_body_transforms(state, wp.array(moved_cable, dtype=wp.transform, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), moved_rigid, atol=1e-6)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_cloth = cloth_values.copy()
        moved_cloth[..., 2] += 1.0
        cloth.set_particle_positions(state, wp.array(moved_cloth, dtype=wp.vec3, device=model.device))
        np.testing.assert_allclose(cable.get_body_transforms(state).numpy(), moved_cable, atol=1e-6)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_soft = soft_values.copy()
        moved_soft[..., 0] += 1.0
        soft.set_particle_positions(state, wp.array(moved_soft, dtype=wp.vec3, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), moved_rigid, atol=1e-6)
        np.testing.assert_allclose(cable.get_body_transforms(state).numpy(), moved_cable, atol=1e-6)
        np.testing.assert_allclose(cloth.get_particle_positions(state).numpy(), moved_cloth, atol=1e-6)
        np.testing.assert_allclose(soft.get_particle_positions(state).numpy(), moved_soft, atol=1e-6)

        state_out = model.state()
        newton.solvers.SolverXPBD(model, iterations=1).step(
            state,
            state_out,
            control=None,
            contacts=None,
            dt=1.0 / 60.0,
        )
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())

    def test_views_update_disjoint_state_and_simulate(self):
        """Rigid, cloth, and volume views coexist across replicated worlds."""
        prototype = newton.ModelBuilder()
        _add_test_articulation(prototype)
        _add_test_cloth(prototype)
        _add_test_soft_body(prototype)
        scene = newton.ModelBuilder()
        scene.replicate(prototype, 2)
        model = scene.finalize()
        state = model.state()

        rigid = ArticulationView(model, "robot")
        cloth = DeformableSurfaceView(model, "cloth")
        soft = DeformableVolumeView(model, "soft")
        self.assertEqual(rigid.get_root_transforms(state).shape, (2, 1))
        self.assertEqual(cloth.get_particle_positions(state).shape, (2, 4))
        self.assertEqual(soft.get_particle_positions(state).shape, (2, 4))

        rigid_before = rigid.get_root_transforms(state).numpy().copy()
        soft_before = soft.get_particle_positions(state).numpy().copy()
        cloth_values = cloth.get_particle_positions(state).numpy().copy()
        cloth_values[1, :, 2] += 2.0
        cloth.set_particle_positions(state, wp.array(cloth_values, dtype=wp.vec3, device=model.device))
        np.testing.assert_array_equal(rigid.get_root_transforms(state).numpy(), rigid_before)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_before)

        rigid_values = rigid_before.copy()
        rigid_values[0, 0, 0] += 1.0
        rigid.set_root_transforms(state, wp.array(rigid_values, dtype=wp.transform, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), rigid_values, atol=1e-6)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)

        state_out = model.state()
        newton.solvers.SolverXPBD(model, iterations=1).step(
            state,
            state_out,
            control=None,
            contacts=None,
            dt=1.0 / 60.0,
        )
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())

    def test_heterogeneous_deformables_keep_rigid_selection_uniform(self):
        """Family views retain empty worlds beside one common rigid articulation."""
        world_0 = newton.ModelBuilder()
        _add_test_articulation(world_0)
        _add_test_cloth(world_0)
        world_1 = newton.ModelBuilder()
        _add_test_articulation(world_1)
        _add_test_soft_body(world_1)
        world_2 = newton.ModelBuilder()
        _add_test_articulation(world_2)
        _add_test_cloth(world_2)
        world_2.add_rod(
            rod=newton.Rod([(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)], radius=0.02),
            label="cable",
            body_frame_origin="com",
        )
        scene = newton.ModelBuilder()
        scene.add_world(world_0)
        scene.add_world(world_1)
        scene.add_world(world_2)
        model = scene.finalize()

        rigid = ArticulationView(model, "robot")
        cloth = DeformableSurfaceView(model, "cloth")
        soft = DeformableVolumeView(model, "soft")
        cable = DeformableCurveView(model, "cable")
        self.assertEqual((rigid.count, rigid.world_count, rigid.count_per_world), (3, 3, 1))
        self.assertEqual(rigid.get_root_transforms(model).shape, (3, 1))
        np.testing.assert_array_equal(cloth.deformable_object_boundaries.numpy(), [0, 1, 1, 2])
        np.testing.assert_array_equal(soft.deformable_object_boundaries.numpy(), [0, 0, 1, 1])
        np.testing.assert_array_equal(cable.deformable_object_boundaries.numpy(), [0, 0, 0, 1])

    def test_rigid_view_survives_dropped_curve_object(self):
        """Dropping an incomplete curve leaves an unrelated rigid view valid."""
        builder = newton.ModelBuilder()
        _add_test_articulation(builder)
        _add_test_anchored_cable(builder)

        with self.assertWarnsRegex(UserWarning, "anchored_curve.*joints_to_keep"):
            builder.collapse_fixed_joints()
        model = builder.finalize()

        rigid = ArticulationView(model, "robot")
        self.assertEqual(rigid.get_root_transforms(model).shape, (1, 1))
        with self.assertRaisesRegex(KeyError, "anchored_curve"):
            DeformableCurveView(model, "anchored_curve")


if __name__ == "__main__":
    unittest.main(verbosity=2)
