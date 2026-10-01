# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify body ownership and supported configurations for ragged geom slots."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import get_test_devices


def _add_free_body(builder: newton.ModelBuilder) -> int:
    """Add one articulated body with explicit mass and inertia."""
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_free(body)
    builder.add_articulation([joint])
    return body


def _basic_builder() -> newton.ModelBuilder:
    """Build two independent worlds and a shared floor with MuJoCo attributes."""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_ground_plane()
    for _ in range(2):
        builder.begin_world()
        body = _add_free_body(builder)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()
    return builder


def _builder_with_optional_sites(include_sites: bool) -> newton.ModelBuilder:
    """Interleave unequal collider/site counts with a shared site and ordinary visuals."""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    if include_sites:
        builder.add_site(-1, label="global_site")
    builder.add_ground_plane()
    for world in range(2):
        builder.begin_world()
        body = _add_free_body(builder)
        for part in range(world + 1):
            if include_sites:
                builder.add_site(body, label=f"world{world}_site{part}")
            builder.add_shape_box(
                body,
                hx=0.1 / (world + 1),
                hy=0.1,
                hz=0.1,
                xform=wp.transform((0.1 * part - 0.05 * world, 0.0, 0.0), wp.quat_identity()),
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            )
        builder.add_shape_sphere(
            body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        )
        builder.end_world()
    return builder


class TestMuJoCoHeterogeneousMapping(unittest.TestCase):
    """Check mappings for the bounded Newton-contact heterogeneity mode."""

    def test_slots_preserve_body_and_contact_properties(self):
        """Map mixed collider groups to their original world, body and contact properties."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                builder.add_ground_plane()
                # Maxima occur in different worlds, and either body can have no colliders.
                for world, counts in enumerate(((1, 0), (0, 3), (2, 1))):
                    builder.begin_world()
                    for local_body, count in enumerate(counts):
                        body = _add_free_body(builder)
                        for shape in range(count):
                            custom = {
                                "mujoco:condim": 3 if shape % 2 else 4,
                                "mujoco:geom_priority": world % 2,
                            }
                            if (world + local_body) % 2:
                                builder.add_shape_sphere(body, radius=0.1, custom_attributes=custom)
                            else:
                                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, custom_attributes=custom)
                    builder.end_world()
                model = builder.finalize(device=device)
                solver = SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)
                mapping = solver.mjc_geom_to_newton_shape.numpy()
                body_mapping = solver.mjc_body_to_newton.numpy()
                shape_body = model.shape_body.numpy()
                shape_world = model.shape_world.numpy()
                condim = model.mujoco.condim.numpy()
                priority = model.mujoco.geom_priority.numpy()
                self.assertEqual(set(mapping[mapping >= 0]), set(range(model.shape_count)))
                self.assertTrue(np.any(mapping < 0))
                for world, row in enumerate(mapping):
                    self.assertEqual(len(row[row >= 0]), len(set(row[row >= 0])))
                    for geom, shape in enumerate(row):
                        if shape < 0:
                            continue
                        self.assertIn(shape_world[shape], (-1, world))
                        self.assertEqual(body_mapping[world, solver.mj_model.geom_bodyid[geom]], shape_body[shape])
                        self.assertEqual(solver.mj_model.geom_condim[geom], condim[shape])
                        self.assertEqual(solver.mj_model.geom_priority[geom], priority[shape])

    def test_legacy_single_world_keeps_dynamic_shape_attachment(self):
        """Keep dynamic shapes attached to their body when legacy world indices are negative."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane()
                body = builder.add_body()
                shape = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize(device=device)
                self.assertEqual(model.world_count, 1)
                self.assertEqual(model.shape_world.numpy()[shape], -1)
                solver = SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)
                geom = np.flatnonzero(solver.mjc_geom_to_newton_shape.numpy()[0] == shape)
                self.assertEqual(len(geom), 1)
                mj_body = solver.mj_model.geom_bodyid[geom[0]]
                self.assertGreater(mj_body, 0)
                self.assertEqual(solver.mjc_body_to_newton.numpy()[0, mj_body], body)

    def test_reject_unsupported_shape_references(self):
        """Reject sites and explicit pairs whose references need a separate mapping extension."""
        for feature in ("sites", "explicit MuJoCo contact pairs"):
            with self.subTest(feature=feature):
                builder = _basic_builder()
                if feature == "sites":
                    builder.add_site(-1)
                else:
                    builder.add_custom_values(
                        **{"mujoco:pair_world": 0, "mujoco:pair_geom1": 0, "mujoco:pair_geom2": 1}
                    )
                model = builder.finalize(device="cpu")
                message = "include_sites=False" if feature == "sites" else feature
                with self.assertRaisesRegex(ValueError, message):
                    SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)

    def test_omit_unused_sites_preserves_mapping_and_dynamics(self):
        """Omit sites from MuJoCo while retaining Newton shapes and site-free dynamics."""
        for device in get_test_devices():
            for skip_visual_only_geoms in (False, True):
                with (
                    self.subTest(device=device, skip_visual_only_geoms=skip_visual_only_geoms),
                    wp.ScopedDevice(device),
                ):
                    results = []
                    for include_sites in (False, True):
                        model = _builder_with_optional_sites(include_sites).finalize(device=device)
                        names = (
                            "shape_flags",
                            "shape_world",
                            "shape_body",
                            "shape_transform",
                            "shape_scale",
                            "shape_type",
                        )
                        original = {name: getattr(model, name).numpy().copy() for name in names}
                        labels = list(model.shape_label)
                        solver = SolverMuJoCo(
                            model,
                            allow_heterogeneous_shapes=True,
                            use_mujoco_contacts=False,
                            include_sites=False,
                            skip_visual_only_geoms=skip_visual_only_geoms,
                        )
                        self.assertEqual(solver.mj_model.nsite, 0)
                        mapping = solver.mjc_geom_to_newton_shape.numpy()
                        is_site = (original["shape_flags"] & int(newton.ShapeFlags.SITE)) != 0
                        retained = ~is_site
                        if skip_visual_only_geoms:
                            retained &= (original["shape_flags"] & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
                        self.assertEqual(set(mapping[mapping >= 0]), set(np.flatnonzero(retained)))
                        body_mapping = solver.mjc_body_to_newton.numpy()
                        for world, row in enumerate(mapping):
                            for geom, shape in enumerate(row):
                                if shape >= 0:
                                    self.assertIn(original["shape_world"][shape], (-1, world))
                                    self.assertEqual(
                                        body_mapping[world, solver.mj_model.geom_bodyid[geom]],
                                        original["shape_body"][shape],
                                    )
                        state, next_state = model.state(), model.state()
                        q = state.joint_q.numpy().reshape(2, 7)
                        q[:, 2] = 0.095
                        state.joint_q.assign(q.reshape(-1))
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
                        contacts, control = pipeline.contacts(), model.control()
                        for _ in range(12):
                            state.clear_forces()
                            pipeline.collide(state, contacts)
                            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                            solver.step(state, next_state, control, contacts, 1.0 / 240.0)
                            state, next_state = next_state, state
                        inverse = solver.newton_shape_to_mjc_geom.numpy()
                        self.assertTrue(np.all(inverse[~retained] == -1))
                        for row in mapping:
                            for geom, shape in enumerate(row):
                                if shape >= 0:
                                    self.assertEqual(inverse[shape], geom)
                        for name, expected in original.items():
                            np.testing.assert_array_equal(getattr(model, name).numpy(), expected)
                        self.assertEqual(model.shape_label, labels)
                        results.append((state.body_q.numpy(), state.body_qd.numpy()))
                    for with_sites, without_sites in zip(results[1], results[0], strict=True):
                        self.assertTrue(np.isfinite(with_sites).all())
                        np.testing.assert_allclose(with_sites, without_sites, atol=1.0e-6, rtol=1.0e-6)

    def test_omitted_sites_do_not_reject_joint_actuator_metadata(self):
        """Preserve joint targets when optional labels or ignored metadata mention sites."""
        for feature in ("resolved_joint", "deferred_joint", "joint_target"):
            with self.subTest(feature=feature):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                for world in range(2):
                    builder.begin_world()
                    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                    label = f"world{world}_target"
                    joint = builder.add_joint_revolute(-1, body, custom_attributes={"mujoco:joint_dof_label": label})
                    builder.add_articulation([joint])
                    for _ in range(world + 1):
                        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                    site = builder.add_site(body, label=label)
                    ignored = feature == "joint_target"
                    builder.add_custom_values(
                        **{
                            "mujoco:actuator_world": world,
                            "mujoco:actuator_trntype": int(
                                SolverMuJoCo.TrnType.SITE if ignored else SolverMuJoCo.TrnType.JOINT
                            ),
                            "mujoco:actuator_trnid": wp.vec2i(
                                site if ignored else -1 if feature == "deferred_joint" else world, -1
                            ),
                            "mujoco:actuator_target_label": label,
                            "mujoco:ctrl_source": int(
                                SolverMuJoCo.CtrlSource.JOINT_TARGET if ignored else SolverMuJoCo.CtrlSource.CTRL_DIRECT
                            ),
                        }
                    )
                    builder.end_world()
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(
                    model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False, include_sites=False
                )
                self.assertEqual(solver.mj_model.nsite, 0)
                self.assertEqual(solver.mj_model.nu, 0 if ignored else 1)
                if not ignored:
                    state, next_state = model.state(), model.state()
                    control = model.control()
                    control.mujoco.ctrl.assign(np.ones(2, dtype=np.float32))
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    solver.step(state, next_state, control, contacts, 1.0 / 240.0)
                    self.assertTrue(np.all(next_state.joint_qd.numpy() > 0.0))

    def test_reject_actuators_requiring_omitted_sites(self):
        """Reject site dependencies even when only a later world requires them."""
        for device in get_test_devices():
            for feature in ("site", "refsite", "slidercrank", "deferred_label"):
                with self.subTest(device=device, feature=feature), wp.ScopedDevice(device):
                    builder = _builder_with_optional_sites(True)
                    site = builder.shape_label.index("world1_site0")
                    refsite = builder.shape_label.index("world1_site1")
                    transmission = SolverMuJoCo.TrnType.SITE
                    targets = wp.vec2i(site, refsite if feature == "refsite" else -1)
                    label = ""
                    if feature == "slidercrank":
                        transmission = SolverMuJoCo.TrnType.SLIDERCRANK
                        targets = wp.vec2i(site, refsite)
                    elif feature == "deferred_label":
                        transmission = SolverMuJoCo.TrnType.JOINT
                        targets = wp.vec2i(-1, -1)
                        label = "world1_site1"
                    builder.add_custom_values(
                        **{
                            "mujoco:actuator_world": 1,
                            "mujoco:actuator_trntype": int(transmission),
                            "mujoco:actuator_trnid": targets,
                            "mujoco:actuator_target_label": label,
                        }
                    )
                    model = builder.finalize(device=device)
                    with self.assertRaisesRegex(ValueError, "actuators.*sites"):
                        SolverMuJoCo(
                            model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False, include_sites=False
                        )

    def test_omitting_sites_still_rejects_spatial_tendon_references(self):
        """Keep spatial tendon references unsupported when unused sites may be omitted."""
        for attribute in ("tendon_wrap_shape", "tendon_wrap_sidesite"):
            with self.subTest(attribute=attribute):
                builder = _builder_with_optional_sites(True)
                site = builder.shape_label.index("world1_site0")
                builder.add_custom_values(**{f"mujoco:{attribute}": site})
                model = builder.finalize(device="cpu")
                with self.assertRaisesRegex(ValueError, "spatial tendons"):
                    SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False, include_sites=False)

    def test_reject_fluid_forces_from_arguments_and_world_attributes(self):
        """Reject fluid options from constructor arguments or any world's custom attributes."""
        for option in ("density", "viscosity"):
            with self.subTest(option=option):
                model = _basic_builder().finalize(device="cpu")
                with self.assertRaisesRegex(ValueError, "does not support fluid"):
                    SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False, **{option: 1.0})
                getattr(model.mujoco, option).assign(np.array([0.0, 1.0], dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "does not support fluid"):
                    SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)

    def test_reject_different_body_joint_layouts(self):
        """Reject equal-sized articulation trees whose joint attachments differ."""
        builder = newton.ModelBuilder()
        for world in range(2):
            builder.begin_world()
            bodies = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
            root, child = bodies if world == 0 else bodies[::-1]
            root_joint = builder.add_joint_revolute(-1, root)
            child_joint = builder.add_joint_revolute(root, child)
            builder.add_articulation([root_joint, child_joint])
            for body in bodies:
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            builder.end_world()
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "identical body/joint layouts"):
            SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)

    def test_reject_cpu_mujoco_backend(self):
        """Require the Warp backend that accepts Newton-generated contacts."""
        model = _basic_builder().finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "use_mujoco_cpu=False"):
            SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False, use_mujoco_cpu=True)

    def test_reject_native_contacts(self):
        """Require Newton contacts even for homogeneous models when the opt-in is enabled."""
        model = _basic_builder().finalize(device="cpu")
        for options in ({}, {"use_mujoco_contacts": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "use_mujoco_contacts=False"):
                SolverMuJoCo(model, allow_heterogeneous_shapes=True, **options)

    def test_homogeneous_native_contacts_remain_supported(self):
        """Keep ordinary native MuJoCo Warp contact generation available without the opt-in."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _basic_builder().finalize(device=device)
                solver = SolverMuJoCo(model, use_mujoco_contacts=True)
                state = model.state()
                q = state.joint_q.numpy().reshape(2, 7).copy()
                q[:, 2] = 0.1
                state.joint_q.assign(q.reshape(-1))
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                next_state = model.state()
                control = model.control()
                for _ in range(12):
                    state.clear_forces()
                    solver.step(state, next_state, control, None, 1.0 / 240.0)
                    state, next_state = next_state, state
                np.testing.assert_allclose(state.body_q.numpy()[:, 2], [0.1, 0.1], atol=0.015)
                self.assertTrue(np.isfinite(state.body_qd.numpy()).all())
                count = int(solver.mjw_data.nacon.numpy()[0])
                self.assertGreater(count, 0)
                self.assertEqual(set(solver.mjw_data.contact.worldid.numpy()[:count]), {0, 1})


if __name__ == "__main__":
    unittest.main()
