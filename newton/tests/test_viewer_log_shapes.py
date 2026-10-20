# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.viewer.viewer_history import WorkspaceHistory
from newton.tests.unittest_utils import assert_np_equal
from newton.viewer import ViewerNull


class _LogShapesProbe(ViewerNull):
    """Captures args passed to ``log_instances`` so tests can inspect them."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.last_colors = None
        self.last_materials = None
        self.last_opacities = None

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False, opacities=None):
        self.last_colors = colors
        self.last_materials = materials
        self.last_opacities = opacities


class TestLogShapesBroadcast(unittest.TestCase):
    """Regression tests for broadcasting length-1 warp arrays in ``log_shapes`` (issue #1417)."""

    def test_length_one_color_and_material_broadcast(self):
        """A single-element color/material warp array should be broadcast to match num_instances."""
        viewer = _LogShapesProbe()

        num_instances = 3
        xforms = wp.array(
            [wp.transform_identity()] * num_instances,
            dtype=wp.transform,
        )
        color = wp.array([wp.vec3(0.9, 0.1, 0.1)], dtype=wp.vec3)
        material = wp.array([wp.vec4(0.0, 0.7, 0.0, 0.0)], dtype=wp.vec4)

        viewer.log_shapes(
            "/test_sphere",
            newton.GeoType.SPHERE,
            0.5,
            xforms,
            colors=color,
            materials=material,
        )

        # colors and materials should have been broadcast to num_instances
        self.assertEqual(len(viewer.last_colors), num_instances)
        self.assertEqual(len(viewer.last_materials), num_instances)

        # every row should match the original single element
        assert_np_equal(
            viewer.last_colors.numpy(),
            np.tile([0.9, 0.1, 0.1], (num_instances, 1)).astype(np.float32),
        )
        assert_np_equal(
            viewer.last_materials.numpy(),
            np.tile([0.0, 0.7, 0.0, 0.0], (num_instances, 1)).astype(np.float32),
        )

    def test_full_length_arrays_pass_through(self):
        """Arrays already matching num_instances should be passed through unchanged."""
        viewer = _LogShapesProbe()

        num_instances = 2
        xforms = wp.array(
            [wp.transform_identity()] * num_instances,
            dtype=wp.transform,
        )
        colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
            dtype=wp.vec3,
        )
        materials = wp.array(
            [wp.vec4(0.1, 0.2, 0.0, 0.0), wp.vec4(0.3, 0.4, 0.0, 0.0)],
            dtype=wp.vec4,
        )

        viewer.log_shapes(
            "/test_box",
            newton.GeoType.BOX,
            (0.5, 0.3, 0.8),
            xforms,
            colors=colors,
            materials=materials,
        )

        self.assertEqual(len(viewer.last_colors), num_instances)
        self.assertEqual(len(viewer.last_materials), num_instances)

    def test_none_colors_and_materials_use_defaults(self):
        """Passing None for colors/materials should produce default values."""
        viewer = _LogShapesProbe()

        num_instances = 2
        xforms = wp.array(
            [wp.transform_identity()] * num_instances,
            dtype=wp.transform,
        )

        viewer.log_shapes(
            "/test_capsule",
            newton.GeoType.CAPSULE,
            (0.3, 1.0),
            xforms,
        )

        self.assertEqual(len(viewer.last_colors), num_instances)
        self.assertEqual(len(viewer.last_materials), num_instances)

        # default color is (0.3, 0.8, 0.9)
        assert_np_equal(
            viewer.last_colors.numpy(),
            np.tile([0.3, 0.8, 0.9], (num_instances, 1)).astype(np.float32),
        )


class TestLogContacts(unittest.TestCase):
    def test_glyphs_scale_with_smaller_contact_shape(self):
        """Scale contact glyphs with geometry for legacy and observable inputs."""
        builder = newton.ModelBuilder()
        for radius in (0.2, 2.0, 1.0):
            builder.add_shape_sphere(body=-1, radius=radius)
        model = builder.finalize(device="cpu")

        pipeline = newton.CollisionPipeline(model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        contacts.rigid_contact_count.assign([2])
        contacts.rigid_contact_shape0.assign([0, 1])
        contacts.rigid_contact_shape1.assign([1, 2])
        contacts.rigid_contact_normal.assign([wp.vec3(0.0, 0.0, 1.0)] * 2)

        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)
        viewer.show_contacts = True
        viewer.log_contacts(contacts, model.state())
        unknown_colors = viewer._contact_disk_colors.numpy().copy()

        self.assertIsNone(viewer._get_world_extents())
        normal_lengths = np.linalg.norm(viewer._contact_points1.numpy() - viewer._contact_points0.numpy(), axis=1)
        assert_np_equal(normal_lengths, np.array([0.1, 0.5]), tol=1.0e-6)
        assert_np_equal(viewer._contact_disk_scales.numpy()[:, 0], np.array([0.02, 0.1]), tol=1.0e-6)

        solver = newton.solvers.SolverXPBD(model)
        observables = solver.observables(contacts=contacts, kinds={newton.solvers.SolverBase.ObservableKind.CONTACT_F})
        solver.validate_observables(observables, contacts)
        observables.contact_f.assign([wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] * 2)
        self.assertIsNone(contacts.force)
        viewer.show_contact_forces = True
        viewer.log_contacts(contacts, model.state(), observables=observables)
        loaded_colors = viewer._contact_disk_colors.numpy().copy()
        observables.contact_f.zero_()
        viewer.log_contacts(contacts, model.state(), observables=observables)
        unloaded_colors = viewer._contact_disk_colors.numpy().copy()
        self.assertFalse(np.allclose(unknown_colors, unloaded_colors))
        self.assertFalse(np.allclose(loaded_colors, unloaded_colors))
        observables.contact_f.assign([wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] * 2)
        viewer.log_contacts(contacts, model.state(), observables=observables)
        assert_np_equal(viewer._contact_disk_scales.numpy()[:, 0], np.array([0.02, 0.1]), tol=1.0e-6)
        force_lengths = np.linalg.norm(
            viewer._contact_force_ends.numpy() - viewer._contact_force_starts.numpy(), axis=1
        )
        expected_lengths = np.array([0.1, 0.5]) * viewer.contact_viz_scale * viewer.contact_force_scale
        assert_np_equal(force_lengths, expected_lengths, tol=1.0e-6)

        # Independent normal scaling must leave mode disks and force arrows alone.
        viewer.contact_normal_scale = 0.4
        viewer.log_contacts(contacts, model.state(), observables=observables)
        assert_np_equal(
            np.linalg.norm(viewer._contact_points1.numpy() - viewer._contact_points0.numpy(), axis=1),
            np.array([0.04, 0.2]),
            tol=1.0e-6,
        )
        assert_np_equal(viewer._contact_disk_scales.numpy()[:, 0], np.array([0.02, 0.1]), tol=1.0e-6)
        assert_np_equal(
            np.linalg.norm(viewer._contact_force_ends.numpy() - viewer._contact_force_starts.numpy(), axis=1),
            expected_lengths,
            tol=1.0e-6,
        )
        viewer.contact_normal_scale = 1.0

        # Replay must bind forces to its own Contacts storage without changing
        # the live solver's force arrays or contact identity.
        example = SimpleNamespace(
            viewer=viewer,
            solver=solver,
            state=model.state(),
            contacts=contacts,
            solver_observables=observables,
            sim_time=0.0,
        )
        example.render = lambda: viewer.log_contacts(
            example.contacts, example.state, observables=example.solver_observables
        )
        history = WorkspaceHistory(example)
        history.capture()
        observables.contact_f.zero_()
        example.sim_time = 1.0 / 60.0
        history.capture()
        history.select(0)
        history.render()
        solver.validate_observables(history.display_observables, history.display_contacts)
        assert_np_equal(viewer._contact_disk_colors.numpy(), loaded_colors)
        self.assertIs(example.solver_observables, observables)
        self.assertIs(example.contacts, contacts)
        self.assertIsNot(history.display_contacts, contacts)
        self.assertFalse(np.any(observables.contact_f.numpy()))
        history.select(1)
        history.render()
        assert_np_equal(viewer._contact_disk_colors.numpy(), unloaded_colors)

        # Paired glyphs use both body-local witnesses, including static bodies,
        # offsets, and the same world/layer placement as the scene geometry.
        builder = newton.ModelBuilder()
        builder.begin_world()
        for index, radius in enumerate((0.2, 2.0, 1.0)):
            body = -1
            if index < 2:
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(float(index), 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4)
                    )
                )
            builder.add_shape_sphere(body=body, radius=radius)
        builder.end_world()
        model = builder.finalize(device="cpu")
        viewer.set_model(model)
        viewer.show_contacts = True
        state = model.state()
        contacts.rigid_contact_point0.assign([(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)])
        contacts.rigid_contact_point1.assign([(0.2, 0.3, 0.4), (0.5, 0.6, 0.7)])
        contacts.rigid_contact_offset0.assign([(0.0, 0.01, 0.0)] * 2)
        contacts.rigid_contact_offset1.assign([(0.0, 0.0, -0.02)] * 2)
        viewer.world_offsets.assign([(4.0, 5.0, 6.0)])
        layer_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.6))
        viewer.set_layer_transform(viewer.layer.layer_id, layer_xform)
        viewer.show_contact_pairs = True
        viewer.log_contacts(contacts, state)
        expected_points = []
        for contact in range(2):
            for side in range(2):
                shape = getattr(contacts, f"rigid_contact_shape{side}").numpy()[contact]
                body = model.shape_body.numpy()[shape]
                xform = wp.transform_identity()
                if body >= 0:
                    pose = state.body_q.numpy()[body]
                    xform = wp.transform(wp.vec3(pose[:3]), wp.quat(pose[3:]))
                point = getattr(contacts, f"rigid_contact_point{side}").numpy()[contact]
                offset = getattr(contacts, f"rigid_contact_offset{side}").numpy()[contact]
                world_point = wp.transform_point(xform, wp.vec3(point + offset)) + wp.vec3(4.0, 5.0, 6.0)
                expected_points.append(wp.transform_point(layer_xform, world_point))
        assert_np_equal(viewer._contact_points0.numpy(), np.asarray(expected_points)[::2], tol=2.0e-6)
        assert_np_equal(viewer._contact_opposing_points.numpy(), np.asarray(expected_points)[1::2], tol=2.0e-6)
        assert_np_equal(viewer._contact_disk_xforms.numpy()[:, :3], np.asarray(expected_points), tol=2.0e-6)
        normal = np.asarray(wp.quat_rotate(wp.transform_get_rotation(layer_xform), wp.vec3(0.0, 0.0, 1.0)))
        assert_np_equal(
            viewer._contact_points1.numpy() - viewer._contact_points0.numpy(),
            np.array([0.1, 0.5])[:, None] * normal,
            tol=2.0e-6,
        )
        assert_np_equal(viewer._contact_disk_colors.numpy()[::2], viewer._contact_disk_colors.numpy()[1::2])
        # Modes use relative velocity at the witnesses, not body speed alone.
        contacts.force = wp.array(
            [wp.spatial_vector(0, 0, 1, 0, 0, 0), wp.spatial_vector()], dtype=wp.spatial_vector, device="cpu"
        )
        state.body_qd.assign([wp.spatial_vector(1, 0, 0, 0, 0, 0), wp.spatial_vector()])
        viewer.log_contacts(contacts, state)
        sliding_colors = viewer._contact_disk_colors.numpy().copy()
        state.body_qd.zero_()
        viewer.log_contacts(contacts, state)
        self.assertFalse(np.allclose(sliding_colors[0], viewer._contact_disk_colors.numpy()[0]))
        assert_np_equal(sliding_colors[2], viewer._contact_disk_colors.numpy()[2])
        assert_np_equal(viewer._contact_disk_colors.numpy()[::2], viewer._contact_disk_colors.numpy()[1::2])
        contacts.force = None
        # Opposing witnesses remain visible when the normal arrow is disabled.
        viewer.show_contact_normals = False
        with patch.object(viewer, "log_arrows") as arrows, patch.object(viewer, "log_lines") as lines:
            viewer.log_contacts(contacts, state)
        self.assertIsNone(arrows.call_args_list[0].args[1])
        self.assertEqual(len(lines.call_args.args[1]), 2)
        assert_np_equal(lines.call_args.args[2].numpy(), np.asarray(expected_points)[1::2], tol=2.0e-6)
        viewer.show_contact_lines = False
        with patch.object(viewer, "log_lines") as lines, patch.object(viewer, "log_instances") as instances:
            viewer.log_contacts(contacts, state)
        self.assertIsNone(lines.call_args.args[1])
        self.assertEqual(len(instances.call_args.args[2]), 4)
        viewer.show_contact_lines = True
        viewer.show_contact_normals = True

        # Hidden worlds and inactive slots suppress both sides. Switching back
        # to one side must not submit stale glyphs from the larger buffers.
        viewer._visible_worlds_mask = wp.array([0], dtype=int, device="cpu")
        viewer.log_contacts(contacts, state)
        self.assertTrue(np.isnan(viewer._contact_points0.numpy()).all())
        self.assertTrue(np.isnan(viewer._contact_opposing_points.numpy()).all())
        self.assertFalse(viewer._contact_disk_scales.numpy().any())
        viewer._visible_worlds_mask = None
        contacts.rigid_contact_count.assign([1])
        viewer.log_contacts(contacts, state)
        self.assertTrue(np.isnan(viewer._contact_points0.numpy()[1:]).all())
        self.assertFalse(viewer._contact_disk_scales.numpy()[2:].any())
        viewer.show_contact_pairs = False
        with (
            patch.object(viewer, "log_arrows") as arrows,
            patch.object(viewer, "log_instances") as instances,
            patch.object(viewer, "log_lines") as lines,
        ):
            viewer.log_contacts(contacts, state)
        self.assertIsNone(lines.call_args.args[1])
        self.assertEqual(len(arrows.call_args_list[0].args[1]), 1)
        self.assertEqual(len(instances.call_args.args[2]), 1)
        viewer.show_contact_pairs = True
        viewer.show_contacts = False
        with patch.object(viewer, "log_arrows") as arrows, patch.object(viewer, "log_instances") as instances:
            viewer.log_contacts(contacts, state)
        self.assertIsNone(arrows.call_args_list[0].args[1])
        self.assertTrue(instances.call_args.kwargs["hidden"])

        # Default force scaling must be invariant under world replication.
        world = newton.ModelBuilder()
        world.add_body(mass=2.0, inertia=wp.mat33(np.eye(3)))
        scales = []
        for world_count in (1, 8):
            replicated = newton.ModelBuilder()
            for _ in range(world_count):
                replicated.add_world(world)
            viewer.set_model(replicated.finalize(device="cpu"))
            scales.append(viewer._contact_force_scale_default)
        self.assertAlmostEqual(scales[0], scales[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
