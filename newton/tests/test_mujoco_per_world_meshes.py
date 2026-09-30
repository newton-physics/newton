# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.geometry import (
    build_shape_layout,
    compile_layout_collision_masks,
    supports_missing_meshes,
)
from newton.solvers import SolverMuJoCo


def build_world(count, *, offset=0.0):
    builder = newton.ModelBuilder()
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])),
    )
    joint = builder.add_joint_free(child=body)
    builder.add_articulation([joint])
    mesh = newton.Mesh.create_box(0.25 / count, 0.1, 0.1, compute_inertia=False)
    # Exercise MuJoCo's mesh recentering rather than only centered assets.
    mesh = newton.Mesh(mesh.vertices + np.array([offset, 0.0, 0.0]), mesh.indices, compute_inertia=False)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    for i in range(count):
        builder.add_shape_convex_hull(
            body,
            mesh=mesh,
            cfg=cfg,
            xform=wp.transform(wp.vec3(-0.25 + (i + 0.5) * 0.5 / count - offset, 0.0, 0.0), wp.quat_identity()),
        )
    builder.add_site(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
    return builder


class TestMuJoCoPerWorldMeshes(unittest.TestCase):
    def test_native_padding_requires_broadphase_fix(self):
        """Reject native padding when absent mesh slots can generate constraints."""
        builder = newton.ModelBuilder()
        builder.add_world(build_world(2))
        builder.add_world(build_world(5))
        model = builder.finalize()
        with patch("newton._src.solvers.mujoco.solver_mujoco.supports_missing_meshes", return_value=False):
            with self.assertRaisesRegex(ValueError, "mujoco_warp#1689"):
                SolverMuJoCo(model)

    def test_large_unfiltered_layout(self):
        """Allow simple collision graphs with more than 256 mesh slots."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_world(build_world(2))
        builder.add_world(build_world(257))
        model = builder.finalize()
        layout = build_shape_layout(model, skip_visual_only_geoms=True, include_sites=False, required_shapes=set())
        masks = compile_layout_collision_masks(model, layout)
        self.assertTrue(masks.exact)
        self.assertTrue(np.all((masks.collision_type[0] & masks.collision_affinity[1:]) != 0))

    def test_per_world_hull_limits_require_warp(self):
        """Treat different hull simplification limits as distinct MuJoCo assets."""
        builder = newton.ModelBuilder()
        for limit in (4, 8):
            world = build_world(1)
            world.shape_source[0].maxhullvert = limit
            builder.add_world(world)
        model = builder.finalize()
        with self.assertRaisesRegex(ValueError, "require use_mujoco_cpu=False"):
            SolverMuJoCo(model, separate_worlds=True, use_mujoco_cpu=True)

    def test_world_without_bodies(self):
        """Build heterogeneous static mesh worlds without dividing by body count."""
        builder = newton.ModelBuilder()
        mesh = newton.Mesh.create_box(0.1)
        for count in (1, 2):
            world = newton.ModelBuilder()
            for _ in range(count):
                world.add_shape_convex_hull(-1, mesh=mesh)
            builder.add_world(world)
        model = builder.finalize()
        layout = build_shape_layout(model, skip_visual_only_geoms=True, include_sites=False, required_shapes=set())
        with np.errstate(divide="raise", invalid="raise"):
            self.assertTrue(compile_layout_collision_masks(model, layout).exact)

    def test_native_collision_groups_must_match(self):
        """Require shared slot groups even when the differing pair is sometimes absent."""
        if not supports_missing_meshes():
            self.skipTest("Requires MuJoCo Warp #1689")
        builder = newton.ModelBuilder()
        for world_index in range(2):
            world = build_world(1)
            world.shape_collision_group[0] = world_index
            second_body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = world.add_joint_free(child=second_body)
            world.add_articulation([joint])
            if world_index == 1:
                world.add_shape_convex_hull(second_body, mesh=newton.Mesh.create_box(0.1))
            builder.add_world(world)
        model = builder.finalize()
        with self.assertRaisesRegex(ValueError, "collision groups must match"):
            SolverMuJoCo(model)
        SolverMuJoCo(model, use_mujoco_contacts=False)

    def test_different_hull_counts(self):
        """Simulate unequal decompositions with both contact pipelines."""
        for native in (False, True):
            for counts in ((2, 5), (5, 2), (2, 2)):
                with self.subTest(native=native, counts=counts):
                    self._simulate(native, counts)

    def test_replicated_mesh_worlds(self):
        """Preserve contacts and property updates when every world shares the same mesh assets."""
        for native in (False, True):
            with self.subTest(native=native):
                self._simulate(native, (5, 5), replicated=True)

    def test_replicated_mesh_updates_stay_on_device(self):
        """Keep homogeneous mesh property updates free of scale readbacks."""
        builder = newton.ModelBuilder()
        world = build_world(2)
        builder.add_world(world)
        builder.add_world(world)
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_contacts=False)
        model.shape_material_mu.fill_(0.42)
        with patch.object(model.shape_scale, "numpy", side_effect=AssertionError("Unexpected scale readback")):
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.geom_friction.numpy()[:, :, 0], 0.42)

    def _simulate(self, native, counts, *, mesh_floor=False, replicated=False):
        builder = newton.ModelBuilder()
        if mesh_floor:
            builder.add_shape_convex_hull(
                -1,
                mesh=newton.Mesh.create_box(1.0, 0.5, 0.05),
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
            )
        else:
            builder.add_ground_plane()
        first = build_world(counts[0], offset=0.13)
        builder.add_world(first)
        builder.add_world(first if replicated else build_world(counts[1], offset=-0.21))
        model = builder.finalize()
        if native and counts[0] != counts[1] and not supports_missing_meshes():
            with self.assertRaisesRegex(ValueError, "mujoco_warp#1689"):
                SolverMuJoCo(model, separate_worlds=True)
            return
        solver = SolverMuJoCo(model, separate_worlds=True, use_mujoco_contacts=native, nconmax=64, njmax=128)
        self.assertEqual(solver.mj_model.ngeom, max(counts) + 1)
        self.assertEqual(solver.mjw_model.nmesh, (1 if replicated else 2) + int(mesh_floor))
        mapping = solver.mjc_geom_to_newton_shape.numpy()
        self.assertEqual(np.count_nonzero(mapping[0] >= 0), counts[0] + 1)
        self.assertEqual(np.count_nonzero(mapping[1] >= 0), counts[1] + 1)
        mesh_ids = np.broadcast_to(solver.mjw_model.geom_dataid.numpy(), mapping.shape)
        np.testing.assert_array_equal(mesh_ids[mapping < 0], -1)
        np.testing.assert_allclose(solver.mjw_model.body_mass.numpy()[:, 1], 1.0)
        np.testing.assert_allclose(solver.mjw_model.site_pos.numpy()[:, 0], [[0, 0, 0.2]] * 2)
        sizes = solver.mjw_model.geom_size.numpy().copy()
        model.shape_material_mu.fill_(0.7)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_array_equal(solver.mjw_model.geom_size.numpy(), sizes)
        np.testing.assert_allclose(solver.mjw_model.geom_friction.numpy()[mapping >= 0, 0], 0.7)
        state, state_next = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        control = model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        def step():
            if not native:
                pipeline.collide(state, contacts)
            solver.step(state, state_next, control, contacts, 0.002)
            if not native:
                pipeline.collide(state_next, contacts)
            solver.step(state_next, state, control, contacts, 0.002)

        step()
        if model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                step()
            for _ in range(200):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(200):
                step()
        np.testing.assert_allclose(state.body_q.numpy()[:, 2], 0.1, atol=0.005)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertFalse(np.any(solver.mjw_data.overflow.numpy()))
        if native:
            native_contacts = newton.Contacts(
                rigid_contact_max=solver.get_max_contact_count(),
                soft_contact_max=0,
                device=model.device,
                requested_attributes={"force"},
            )
            solver.update_contacts(native_contacts, state)
            count = int(native_contacts.rigid_contact_count.numpy()[0])
            self.assertEqual(int(native_contacts.n_contacts.numpy()[0]), count)
            for field in (native_contacts.rigid_contact_shape0, native_contacts.rigid_contact_shape1):
                self.assertTrue(np.all(np.isin(field.numpy()[:count], mapping[mapping >= 0])))
            np.testing.assert_allclose(native_contacts.force.numpy()[:count, 2].sum(), -2.0 * 9.81, rtol=0.05)
        ncon = int(solver.mjw_data.nacon.numpy()[0])
        worlds = solver.mjw_data.contact.worldid.numpy()[:ncon]
        geoms = solver.mjw_data.contact.geom.numpy()[:ncon]
        for w, count in enumerate(counts):
            touched = np.unique(geoms[worlds == w])
            self.assertEqual(len(touched), count + 1)
            self.assertTrue(np.all(mapping[w, touched] >= 0))

    def test_native_mesh_contacts_with_catalog(self):
        """Use catalog collision data for native mesh-mesh contacts."""
        self._simulate(True, (5, 2), mesh_floor=True)

    def test_compiled_reference_fields(self):
        """Match independent exports for assets absent from the template and scaled meshes."""
        worlds = [build_world(5, offset=0.13), build_world(2, offset=-0.21), build_world(5, offset=0.13)]
        for shape in range(2):
            worlds[1].shape_scale[shape] = wp.vec3(1.2, 0.8, 1.1)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for world in worlds:
            builder.add_world(world)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True, use_mujoco_contacts=False, nconmax=64, njmax=128)
        self.assertEqual(solver.mjw_model.nmesh, 2)
        self.assertTrue(np.all(solver.mjw_model.mesh_graphadr.numpy() >= 0))
        self.assertTrue(np.all(solver.mjw_model.mesh_polynum.numpy() > 0))
        mapping = solver.mjc_geom_to_newton_shape.numpy()
        shape_world = model.shape_world.numpy()
        for world_index, world in enumerate(worlds):
            ref_builder = newton.ModelBuilder()
            ref_builder.add_ground_plane()
            ref_builder.add_world(world)
            ref_model = ref_builder.finalize()
            ref = SolverMuJoCo(ref_model, nconmax=64, njmax=128)
            first_shape = np.flatnonzero(shape_world == world_index)[0]
            for geom, shape in enumerate(mapping[world_index]):
                if shape < 0 or shape_world[shape] < 0:
                    continue
                ref_shape = shape - first_shape + 1
                ref_geom = np.flatnonzero(ref.mjc_geom_to_newton_shape.numpy()[0] == ref_shape)[0]
                for field in ("geom_size", "geom_aabb", "geom_rbound", "geom_pos", "geom_quat"):
                    np.testing.assert_allclose(
                        getattr(solver.mjw_model, field).numpy()[
                            world_index % getattr(solver.mjw_model, field).shape[0], geom
                        ],
                        getattr(ref.mj_model, field)[ref_geom].reshape(
                            getattr(solver.mjw_model, field).numpy().shape[2:]
                        ),
                        atol=1.0e-6,
                        err_msg=field,
                    )
            for field in ("body_mass", "body_ipos", "body_invweight0"):
                np.testing.assert_allclose(
                    getattr(solver.mjw_model, field).numpy()[world_index],
                    getattr(ref.mj_model, field),
                    rtol=1.0e-6,
                    atol=1.0e-6,
                    err_msg=field,
                )

            for body in range(solver.mj_model.nbody):
                tensors = []
                for diagonal, quat in (
                    (
                        solver.mjw_model.body_inertia.numpy()[world_index, body],
                        solver.mjw_model.body_iquat.numpy()[world_index, body],
                    ),
                    (ref.mj_model.body_inertia[body], ref.mj_model.body_iquat[body]),
                ):
                    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*quat[1:], quat[0]))).reshape(3, 3)
                    tensors.append(rotation @ np.diag(diagonal) @ rotation.T)
                np.testing.assert_allclose(tensors[0], tensors[1], atol=1.0e-6)

    def test_shared_collision_filters(self):
        """Preserve body exclusions and global contacts with unequal hull counts."""
        for global_first in (False, True):
            for group in (0, 1, -1):
                with self.subTest(global_first=global_first, group=group):
                    builder = newton.ModelBuilder()
                    if global_first:
                        builder.add_ground_plane()
                    for count in (2, 5):
                        world = build_world(count)
                        world.shape_collision_group[:count] = [group] * count
                        body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                        joint = world.add_joint_free(child=body)
                        world.add_articulation([joint])
                        sphere = world.add_shape_sphere(
                            body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(collision_group=-2)
                        )
                        for hull in range(count):
                            world.add_shape_collision_filter_pair(hull, sphere)
                        builder.add_world(world)
                    if not global_first:
                        builder.add_ground_plane()
                    model = builder.finalize()
                    layout = build_shape_layout(
                        model, skip_visual_only_geoms=True, include_sites=False, required_shapes=set()
                    )
                    masks = compile_layout_collision_masks(model, layout)
                    allowed = (masks.collision_type[:, None] & masks.collision_affinity[None, :]) != 0
                    allowed |= allowed.T.copy()
                    hulls = np.flatnonzero(layout.body_indices == 0)
                    sphere = np.flatnonzero(layout.body_indices == 1)[0]
                    ground = np.flatnonzero(layout.body_indices == -1)[0]
                    self.assertFalse(np.any(allowed[hulls, sphere]))
                    self.assertTrue(allowed[sphere, ground])
                    np.testing.assert_array_equal(allowed[hulls, ground], group != 0)

    def test_conflicting_collision_filters(self):
        """Reject per-world filters that disagree for simultaneously present slots."""
        first, second = build_world(2), build_world(5)
        for world in (first, second):
            world.add_ground_plane()
        second.add_shape_collision_filter_pair(0, 6)
        builder = newton.ModelBuilder()
        builder.add_world(first)
        builder.add_world(second)
        model = builder.finalize()
        with self.assertRaisesRegex(ValueError, "collision filters differ in world"):
            SolverMuJoCo(model, separate_worlds=True)

    def test_mixed_mesh_and_cone_asset_names(self):
        """Keep generated mesh names independent of user-authored shape labels."""
        builder = newton.ModelBuilder()
        for size in (0.1, 0.2):
            world = newton.ModelBuilder()
            body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = world.add_joint_free(child=body)
            world.add_articulation([joint])
            world.add_shape_cone(body, radius=0.1, half_height=0.1, label="newton_mesh")
            world.add_shape_convex_hull(body, mesh=newton.Mesh.create_box(size))
            builder.add_world(world)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertEqual(solver.mjw_model.nmesh, 3)

    def test_mesh_scale_change_rejected(self):
        """Require recompilation when a compiled mesh scale changes."""
        builder = newton.ModelBuilder()
        builder.add_world(build_world(2))
        builder.add_world(build_world(5))
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_contacts=False)
        scales = model.shape_scale.numpy()
        scales[0] *= 2.0
        model.shape_scale.assign(scales)
        with self.assertRaisesRegex(ValueError, "Recreate the solver after resizing"):
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    def test_pose_updates(self):
        """Apply runtime mesh and site poses to their own world after padding."""
        builder = newton.ModelBuilder()
        builder.add_world(build_world(2, offset=0.13))
        builder.add_world(build_world(5, offset=-0.21))
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_contacts=False)
        expected_geoms = solver.mjw_model.geom_pos.numpy()
        expected_sites = solver.mjw_model.site_pos.numpy()
        mapping = solver.mjc_geom_to_newton_shape.numpy()
        shapes = np.flatnonzero(model.shape_world.numpy() == 1)
        geom = int(np.flatnonzero(mapping[1] == shapes[0])[0])
        delta = np.array([0.1, -0.2, 0.3])
        transforms = model.shape_transform.numpy()
        transforms[shapes[0], :3] += delta
        transforms[shapes[-1], :3] += delta
        model.shape_transform.assign(transforms)
        expected_geoms[1, geom] += delta
        expected_sites[1, 0] += delta
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.geom_pos.numpy(), expected_geoms, atol=1.0e-7)
        np.testing.assert_allclose(solver.mjw_model.site_pos.numpy(), expected_sites, atol=1.0e-7)

    def test_per_body_padding(self):
        """Allocate per-body slots even when total world shape counts match."""
        builder = newton.ModelBuilder()
        for counts in ((2, 5), (5, 2)):
            world = build_world(counts[0])
            world.add_builder(build_world(counts[1]))
            builder.add_world(world)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True, use_mujoco_contacts=False)
        self.assertEqual(solver.mj_model.ngeom, 10)
        mapping = solver.mjc_geom_to_newton_shape.numpy()
        bodies = model.shape_body.numpy()
        mj_bodies = solver.mjc_body_to_newton.numpy()
        for world, shapes in enumerate(mapping):
            self.assertEqual(np.count_nonzero(shapes >= 0), 7)
            for geom, shape in enumerate(shapes):
                if shape >= 0:
                    self.assertEqual(bodies[shape], mj_bodies[world, solver.mj_model.geom_bodyid[geom]])

    def test_empty_mesh_slots_do_not_contact(self):
        """Prevent absent slots inside a mesh floor from generating physical constraints."""
        if not supports_missing_meshes():
            self.skipTest("Requires MuJoCo Warp #1689")
        empty = newton.ModelBuilder()
        body = empty.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = empty.add_joint_free(child=body)
        empty.add_articulation([joint])
        empty.add_site(body)
        builder = newton.ModelBuilder()
        builder.add_shape_convex_hull(
            -1,
            mesh=newton.Mesh.create_box(1.0, 0.5, 0.1),
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        )
        builder.add_world(empty)
        builder.add_world(build_world(2))
        model = builder.finalize()
        solver = SolverMuJoCo(model, nconmax=32, njmax=64)
        mapping = solver.mjc_geom_to_newton_shape.numpy()
        _, mjw = SolverMuJoCo.import_mujoco()
        for height in (-0.15, -0.1, -0.05, 0.0, 0.05):
            with self.subTest(height=height):
                poses = solver.mjw_model.geom_pos.numpy()
                poses[mapping < 0, 2] = height
                solver.mjw_model.geom_pos.assign(poses)
                mjw.forward(solver.mjw_model, solver.mjw_data)
                count = int(solver.mjw_data.nacon.numpy()[0])
                self.assertFalse(np.any(solver.mjw_data.contact.worldid.numpy()[:count] == 0))
                self.assertEqual(int(solver.mjw_data.nefc.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
