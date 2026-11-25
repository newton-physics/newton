# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for site export to MuJoCo."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton.solvers import SolverMuJoCo, mujoco
from newton.tests.unittest_utils import MUJOCO_MODULES, requires


@requires(*MUJOCO_MODULES)
class TestMuJoCoSiteExport(unittest.TestCase):
    """Test exporting sites to MuJoCo models."""

    def test_site_edit_refreshes_tendon_constants(self):
        """Keep a tendon equality at rest after changing its site and wrapping geometry."""
        for cpu, wrapping in ((False, ""), (True, ""), (False, "world"), (True, "world"), (False, "bracket")):
            with self.subTest(cpu=cpu, wrapping=wrapping):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z)
                builder.add_articulation([joint])
                moving = builder.add_site(body, xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()))
                fixed = builder.add_site(
                    -1, xform=wp.transform((-1.0, 0.0, 0.0) if wrapping else (1.0, -1.0, 0.0), wp.quat_identity())
                )
                builder.add_ground_plane()
                wraps = [mujoco.TendonWrapSite(moving)]
                if wrapping:
                    parent = -1
                    if wrapping == "bracket":
                        parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                        builder.add_joint_fixed(-1, parent)
                    sphere = builder.add_shape_sphere(
                        parent, radius=0.4, xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity())
                    )
                    wraps.append(mujoco.TendonWrapGeom(sphere))
                wraps.append(mujoco.TendonWrapSite(fixed))
                tendon = mujoco.add_tendon_spatial(builder, wraps)
                mujoco.add_equality_tendon(builder, tendon)
                model = builder.finalize()

                def make_solver(model=model, cpu=cpu):
                    with warnings.catch_warnings():
                        # The bracket is a static body, which only a joint outside articulations creates.
                        warnings.filterwarnings("ignore", message=".*standalone world roots")
                        return SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)

                solver = make_solver()
                poses = model.shape_transform.numpy().copy()
                poses[moving, 0] = 1.1 if wrapping else 2.0
                if wrapping:
                    poses[sphere, 1] = 0.15
                model.shape_transform.assign(poses)
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                reference = make_solver()
                if cpu:
                    plane = solver.mj_model.geom_type == solver._mujoco.mjtGeom.mjGEOM_PLANE
                    self.assertTrue(np.all(solver.mj_model.geom_size[plane] > 0.0))

                output, expected = model.state(), model.state()
                solver.step(model.state(), output, model.control(), None, 0.001)
                reference.step(model.state(), expected, model.control(), None, 0.001)
                np.testing.assert_allclose(expected.joint_qd.numpy(), 0.0, atol=1e-7)
                np.testing.assert_allclose(output.joint_qd.numpy(), expected.joint_qd.numpy(), atol=1e-7)
                for field in ("tendon_length0", "tendon_invweight0"):
                    actual = getattr(solver.mj_model if cpu else solver.mjw_model, field)
                    fresh = getattr(reference.mj_model if cpu else reference.mjw_model, field)
                    np.testing.assert_allclose(actual if cpu else actual.numpy(), fresh if cpu else fresh.numpy())

    def test_site_edit_preserves_live_cpu_state(self):
        """Continue from the live pose after combined edits without input synchronization."""
        import mujoco as mj

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_mjcf("""<mujoco><option gravity="0 0 0"/><worldbody>
            <geom type="plane" size="1 1 .1" pos="0 0 -.05"/>
            <site name="fixed" pos="1 -1 0"/><body><joint name="hinge" springref=".2"/>
            <geom size=".1"/><site name="moving" pos="1 0 0"/></body></worldbody>
            <tendon><spatial><site site="fixed"/><site site="moving"/></spatial></tendon><actuator>
            <motor joint="hinge"/><general site="moving" refsite="fixed" gear="1 0 0 0 0 0"/></actuator></mujoco>""")
        builder.joint_qd[0] = 0.5
        model = builder.finalize()
        with self.assertWarns(newton.NewtonDeprecationWarning):
            solver = SolverMuJoCo(model, use_mujoco_cpu=True, update_data_interval=0)
        solver.mj_data.qpos[:] = 0.4
        state, output = model.state(), model.state()
        solver.step(state, output, model.control(), None, 0.001)
        qpos, qvel = solver.mj_data.qpos.copy(), solver.mj_data.qvel.copy()
        self.assertEqual(solver.mj_data.ncon, 1)
        sites = np.flatnonzero(model.shape_flags.numpy() & int(newton.ShapeFlags.SITE))
        poses = model.shape_transform.numpy().copy()
        poses[sites, 0] += 0.1
        model.shape_transform.assign(poses)
        model.mujoco.dof_ref.fill_(0.3)
        gear = model.mujoco.actuator_gear.numpy()
        gear[solver.mj_model.actuator_trntype == mj.mjtTrn.mjTRN_JOINT, 0] = 3.0
        model.mujoco.actuator_gear.assign(gear)
        solver.notify_model_changed(
            newton.ModelFlags.SHAPE_PROPERTIES
            | newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES
            | newton.ModelFlags.ACTUATOR_PROPERTIES
        )
        np.testing.assert_array_equal(solver.mj_data.qpos, qpos)
        np.testing.assert_array_equal(solver.mj_data.qvel, qvel)
        self.assertEqual(solver.mj_data.ncon, 1)
        reference = mj.MjData(solver.mj_model)
        reference.qpos[:] = qpos
        mj.mj_forward(solver.mj_model, reference)
        for field in ("site_xpos", "ten_length", "actuator_length", "actuator_moment"):
            np.testing.assert_allclose(getattr(solver.mj_data, field), getattr(reference, field), atol=1e-7)
        self.assertIn(3.0, solver.mj_data.actuator_moment)
        solver.step(output, state, model.control(), None, 0.001)
        np.testing.assert_allclose(solver.mj_data.qpos, qpos + 0.001 * qvel, atol=1e-7)

    def test_site_edit_preserves_unnotified_connect_anchor(self):
        """Keep staged CONNECT anchor and loop-frame edits unpublished until their own notification."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(-1, body, parent_xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))
        joints, parent = [joint], body
        for _ in range(2):
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joints.append(builder.add_joint_revolute(parent, child))
            parent = child
        builder.add_articulation(joints)
        loop = builder.add_joint_ball(parent, body)
        moving = builder.add_site(body)
        fixed = builder.add_site(-1)
        mujoco.add_tendon_spatial(builder, [mujoco.TendonWrapSite(moving), mujoco.TendonWrapSite(fixed)])
        mujoco.add_equality_connect(builder, body, anchor=wp.vec3(0.1, 0.2, 0.3), enabled=False)

        for cpu in (False, True):
            with self.subTest(cpu=cpu):
                model = builder.finalize()
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                published = solver.mjw_model.eq_data.numpy()[0, :, :6].copy()
                anchor = np.array([0.4, 0.5, 0.6])
                model.mujoco.equality_constraint_anchor.assign(anchor[None, :])
                frames = model.joint_X_p.numpy().copy()
                frames[loop, 0] += 0.25
                model.joint_X_p.assign(frames)
                poses = model.shape_transform.numpy().copy()
                poses[moving, 0] += 0.1
                model.shape_transform.assign(poses)
                for flags in (newton.ModelFlags.SHAPE_PROPERTIES, newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES):
                    solver.notify_model_changed(flags)
                    actual = solver.mj_model.eq_data if cpu else solver.mjw_model.eq_data.numpy()[0]
                    np.testing.assert_allclose(actual[:, :6], published, atol=1e-7)
                solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
                actual = solver.mj_model.eq_data if cpu else solver.mjw_model.eq_data.numpy()[0]
                np.testing.assert_allclose(actual[0, :6], [0.4, 0.5, 0.6, 1.4, 2.5, 3.6], atol=1e-7)
                solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                actual = solver.mj_model.eq_data if cpu else solver.mjw_model.eq_data.numpy()[0]
                loop_eq = solver.mjc_eq_to_newton_jnt.numpy()[0] == loop
                np.testing.assert_allclose(actual[loop_eq, :6], [[0.25, 0.0, 0.0, 0.0, 0.0, 0.0]], atol=1e-7)

    def test_worldbody_site_poses_across_worlds(self):
        """Keep batched worldbody site poses synchronized with Newton."""
        world = newton.ModelBuilder()
        body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = world.add_joint_free(child=body)
        world.add_articulation([joint])
        world.add_shape_sphere(body, radius=0.1)
        world.add_site(
            -1,
            xform=wp.transform(wp.vec3(0.5, 0.25, 0.1), wp.quat_identity()),
            label="world_site",
        )

        builder = newton.ModelBuilder()
        builder.add_world(world)
        builder.add_world(
            world,
            xform=wp.transform(
                wp.vec3(4.0, -1.0, 0.2),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5),
            ),
        )
        for cpu in (False, True):
            with self.subTest(cpu=cpu):
                model = builder.finalize()
                if cpu:
                    with self.assertWarns(newton.NewtonDeprecationWarning):
                        solver = SolverMuJoCo(model, separate_worlds=True, disable_contacts=True, use_mujoco_cpu=True)
                else:
                    solver = SolverMuJoCo(model, disable_contacts=True)

                def assert_site_poses_match_model(model, solver) -> None:
                    site_shapes = np.flatnonzero(model.shape_flags.numpy() & int(newton.ShapeFlags.SITE))
                    shape_transforms = model.shape_transform.numpy()[site_shapes]
                    np.testing.assert_allclose(
                        solver.mjw_model.site_pos.numpy()[:, 0], shape_transforms[:, :3], atol=1.0e-6
                    )

                    expected_xmat = np.stack(
                        [
                            np.asarray(wp.quat_to_matrix(wp.quat(*transform[3:]))).reshape(3, 3)
                            for transform in shape_transforms
                        ]
                    )
                    if solver.use_mujoco_cpu:
                        np.testing.assert_allclose(solver.mj_model.site_pos[0], shape_transforms[0, :3], atol=1e-6)
                        np.testing.assert_allclose(
                            solver.mj_model.site_quat[0], shape_transforms[0, [6, 3, 4, 5]], atol=1e-6
                        )
                        np.testing.assert_allclose(solver.mj_data.site_xpos[0], shape_transforms[0, :3], atol=1e-6)
                        np.testing.assert_allclose(
                            solver.mj_data.site_xmat[0].reshape(3, 3), expected_xmat[0], atol=1e-6
                        )
                    else:
                        np.testing.assert_allclose(
                            solver.mjw_data.site_xpos.numpy()[:, 0], shape_transforms[:, :3], atol=1e-6
                        )
                        np.testing.assert_allclose(solver.mjw_data.site_xmat.numpy()[:, 0], expected_xmat, atol=1e-6)

                assert_site_poses_match_model(model, solver)

                transforms = [
                    wp.transform(
                        wp.vec3(-0.2 + i, 0.4 + 0.5 * i, 0.3 - 0.1 * i),
                        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2 + 0.15 * i),
                    )
                    for i in range(model.shape_count)
                ]
                model.shape_transform.assign(wp.array(transforms, dtype=wp.transform, device=model.device))
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

                assert_site_poses_match_model(model, solver)

    def test_site_size_updates_at_runtime(self):
        """Propagate runtime shape_scale changes to MuJoCo site sizes."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(body, type=GeoType.BOX, scale=(0.1, 0.2, 0.3), label="box_site")
        builder.add_site(body, type=GeoType.SPHERE, scale=(0.05, 0.05, 0.05), label="sphere_site")

        for cpu in (False, True):
            with self.subTest(cpu=cpu):
                model = builder.finalize()
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu)

                np.testing.assert_allclose(
                    solver.mjw_model.site_size.numpy(), [[0.1, 0.2, 0.3], [0.05, 0.05, 0.05]], atol=1e-6
                )

                model.shape_scale.assign(
                    wp.array([[0.4, 0.5, 0.6], [0.07, 0.0, 0.0]], dtype=wp.vec3, device=model.device)
                )
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

                # Zero components fall back to the first nonzero one, matching export.
                np.testing.assert_allclose(
                    solver.mjw_model.site_size.numpy(), [[0.4, 0.5, 0.6], [0.07, 0.07, 0.07]], atol=1e-6
                )
                if cpu:
                    np.testing.assert_allclose(solver.mj_model.site_size, solver.mjw_model.site_size.numpy(), atol=1e-6)

    def test_site_size_updates_batched_worlds(self):
        """Sync site sizes from uniform runtime scale updates across batched worlds."""
        world = newton.ModelBuilder()
        body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = world.add_joint_free(child=body)
        world.add_articulation([joint])
        world.add_site(body, type=GeoType.BOX, scale=(0.1, 0.2, 0.3), label="body_site")

        builder = newton.ModelBuilder()
        builder.add_world(world)
        builder.add_world(world, xform=wp.transform(wp.vec3(4.0, 0.0, 0.0), wp.quat_identity()))
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)

        np.testing.assert_allclose(solver.mjw_model.site_size.numpy(), [[0.1, 0.2, 0.3]], atol=1e-6)

        site_shapes = np.flatnonzero(model.shape_flags.numpy() & int(newton.ShapeFlags.SITE))
        scales = model.shape_scale.numpy()
        scales[site_shapes] = [0.4, 0.5, 0.6]
        model.shape_scale.assign(scales)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

        np.testing.assert_allclose(solver.mjw_model.site_size.numpy(), [[0.4, 0.5, 0.6]], atol=1e-6)

    def test_export_single_site(self):
        """Test that a site is exported to both MuJoCo Warp and regular MuJoCo models."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(
            body, type=GeoType.SPHERE, label="test_site", xform=wp.transform(wp.vec3(0.1, 0, 0), wp.quat_identity())
        )

        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model
        mj_model = solver.mj_model

        # Verify site exists in MuJoCo Warp model
        self.assertGreater(mjw_model.nsite, 0, "Site should be exported to MuJoCo Warp model")

        # Verify site exists in regular MuJoCo model
        self.assertGreater(mj_model.nsite, 0, "Site should be exported to regular MuJoCo model")

        # Both should have the same site count
        self.assertEqual(mjw_model.nsite, mj_model.nsite, "Both models should have the same site count")

    def test_export_multiple_sites(self):
        """Test exporting multiple sites."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

        builder.add_site(body, type=GeoType.SPHERE, label="site_1")
        builder.add_site(body, type=GeoType.SPHERE, label="site_2")
        builder.add_site(body, type=GeoType.SPHERE, label="site_3")

        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        self.assertEqual(mjw_model.nsite, 3, "Should have exactly 3 sites")

    def test_site_not_exported_as_geom(self):
        """Test that sites are NOT exported as collision geoms."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

        # Add site
        builder.add_site(body, type=GeoType.SPHERE, label="my_site")
        # Add regular collision shape
        builder.add_shape_sphere(body, radius=0.1)

        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Should have 1 site and 1 geom
        self.assertEqual(mjw_model.nsite, 1, "Should have exactly 1 site")
        self.assertEqual(mjw_model.ngeom, 1, "Should have exactly 1 geom (not counting site)")

    def test_export_site_transforms(self):
        """Test that site transforms are correctly exported."""
        builder = newton.ModelBuilder()
        body = builder.add_link(
            xform=wp.transform(
                wp.vec3(1.0, 0.2, 0.3),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4),
            ),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])

        site_xform = wp.transform(wp.vec3(0.5, 0.3, 0.1), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.57))
        builder.add_site(body, type=GeoType.SPHERE, xform=site_xform, label="positioned_site")

        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Verify site exists and check position
        self.assertGreater(mjw_model.nsite, 0, "Site should exist")
        site_pos = mjw_model.site_pos.numpy()[0, 0]  # First world, first site
        np.testing.assert_allclose(site_pos[:3], [0.5, 0.3, 0.1], atol=1e-5)
        np.testing.assert_allclose(solver.mjw_data.site_xpos.numpy()[0, 0], solver.mj_data.site_xpos[0], atol=1e-6)
        np.testing.assert_allclose(
            solver.mjw_data.site_xmat.numpy()[0, 0], solver.mj_data.site_xmat[0].reshape(3, 3), atol=1e-6
        )

    def test_export_site_types(self):
        """Test that site types are exported correctly."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

        builder.add_site(body, type=GeoType.SPHERE, scale=(0.05, 0.05, 0.05), label="sphere")
        builder.add_site(body, type=GeoType.BOX, scale=(0.1, 0.2, 0.3), label="box")

        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Verify we have 2 sites
        self.assertEqual(mjw_model.nsite, 2, "Should have 2 sites")

        # Verify types - first should be sphere (2), second should be box (6)
        site_types = mjw_model.site_type.numpy()
        if len(site_types.shape) == 2:  # [nworld, nsite]
            site_types = site_types[0]  # First world
        self.assertEqual(site_types[0], 2, "First site should be sphere (type 2)")
        self.assertEqual(site_types[1], 6, "Second site should be box (type 6)")

    def test_include_sites_parameter(self):
        """Test that the include_sites parameter controls site export."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

        builder.add_site(body, type=GeoType.SPHERE, label="my_site")

        model = builder.finalize()

        # Test 1: Sites should be exported by default (include_sites=True)
        solver_with_sites = SolverMuJoCo(model, include_sites=True)
        mjw_model_with = solver_with_sites.mjw_model
        self.assertEqual(mjw_model_with.nsite, 1, "Should have 1 site when include_sites=True")

        # Test 2: Sites should NOT be exported when include_sites=False
        solver_without_sites = SolverMuJoCo(model, include_sites=False)
        mjw_model_without = solver_without_sites.mjw_model
        self.assertEqual(mjw_model_without.nsite, 0, "Should have 0 sites when include_sites=False")

        # Test 3: Default behavior (no parameter) should include sites
        solver_default = SolverMuJoCo(model)
        mjw_model_default = solver_default.mjw_model
        self.assertEqual(mjw_model_default.nsite, 1, "Should have 1 site by default")

    def test_mjcf_roundtrip(self):
        """Test MJCF → Newton → MuJoCo round-trip preserves sites."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="link" pos="0 0 1">
                    <joint type="free"/>
                    <site name="sensor_site" pos="0.1 0.05 0" type="sphere" size="0.02"/>
                    <geom name="body_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        # Import to Newton
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Export to MuJoCo
        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Verify site exists (should have at least 1 site)
        self.assertGreater(mjw_model.nsite, 0, "Should have at least 1 site after round-trip")

        # Verify we have 1 geom (the box)
        self.assertGreater(mjw_model.ngeom, 0, "Should have at least 1 geom (the box)")


if __name__ == "__main__":
    unittest.main()
