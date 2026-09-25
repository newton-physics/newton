# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise heterogeneous convex geometry with native MuJoCo Warp collision detection."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.test_mujoco_heterogeneous_shapes import build_model
from newton.tests.unittest_utils import get_test_devices


def _create_simulation(model, *, heterogeneous=True, **solver_options):
    solver = SolverMuJoCo(
        model,
        allow_heterogeneous_shapes=heterogeneous,
        use_mujoco_contacts=True,
        iterations=50,
        ls_iterations=20,
        nconmax=solver_options.pop("nconmax", 128),
        njmax=solver_options.pop("njmax", 512),
        **solver_options,
    )
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contacts = newton.Contacts(
        rigid_contact_max=solver.get_max_contact_count(),
        soft_contact_max=0,
        requested_attributes={"force"},
        device=model.device,
    )
    return solver, state, model.state(), model.control(), contacts


def _advance(simulation, steps):
    solver, state, next_state, control, contacts = simulation
    for _ in range(steps):
        state.clear_forces()
        solver.step(state, next_state, control, None, 1.0 / 240.0)
        state, next_state = next_state, state
    return solver, state, next_state, control, contacts


def _read_contact_pairs(simulation):
    solver, state, _, _, contacts = simulation
    solver.update_contacts(contacts, state)
    count = int(contacts.rigid_contact_count.numpy()[0])
    return np.column_stack(
        (contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
    )


def _build_stack_model(variants, device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e5, kd=1.0e3, mu=0.5)
    for lower_count, upper_count, filtered in variants:
        template = newton.ModelBuilder()
        shapes_by_body = []
        for body_index, (count, height, initial_z) in enumerate(((lower_count, 0.1, 0.1), (upper_count, 0.08, 0.6))):
            body = template.add_link(
                label=f"object_{body_index}/body",
                xform=wp.transform((0.0, 0.0, initial_z), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.diag([0.02, 0.02, 0.03])),
            )
            joint = template.add_joint_free(body, label=f"object_{body_index}/free")
            template.add_articulation([joint], label=f"object_{body_index}")
            shapes = []
            for part in range(count):
                half_width = 0.24 / count
                mesh = newton.Mesh.create_box(half_width, 0.2, height, compute_inertia=False)
                shapes.append(
                    template.add_shape_convex_hull(
                        body,
                        mesh=mesh,
                        xform=wp.transform((-0.24 + (2 * part + 1) * half_width, 0.0, 0.0), wp.quat_identity()),
                        cfg=cfg,
                    )
                )
            shapes_by_body.append(shapes)
        if filtered:
            for lower in shapes_by_body[0]:
                for upper in shapes_by_body[1]:
                    template.add_shape_collision_filter_pair(lower, upper)
        builder.add_world(template)
    return builder.finalize(device=device)


class TestMuJoCoHeterogeneousNative(unittest.TestCase):
    def test_default_capacities_settle_heterogeneous_hulls(self):
        """Settle ragged hulls with automatic contact and constraint capacities."""
        from mujoco_warp import OverflowType

        capacity_overflow = int(
            OverflowType.NEFC
            | OverflowType.NJMAX_NNZ
            | OverflowType.BROADPHASE
            | OverflowType.NARROWPHASE
            | OverflowType.CCD
        )
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = build_model([(1, 0.1), (3, 0.2), (2, 0.3)], device)
                simulation = _advance(_create_simulation(model, nconmax=None, njmax=None), 240)
                np.testing.assert_allclose(simulation[1].body_q.numpy()[:, 2], [0.1, 0.2, 0.3], atol=0.015)
                self.assertEqual(set(_read_contact_pairs(simulation).flat), set(range(model.shape_count)))
                self.assertFalse(np.any(simulation[0].mjw_data.overflow.numpy() & capacity_overflow))

    def test_disable_contacts_at_construction_and_runtime(self):
        """Fall freely with contacts disabled and recover native collision after reenabling."""
        from mujoco_warp import DisableBit

        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = build_model([(1, 0.1), (3, 0.2)], device)
                disabled = _advance(_create_simulation(model, disable_contacts=True), 180)
                self.assertTrue(np.all(disabled[1].body_q.numpy()[:, 2] < -1.0))
                self.assertEqual(len(_read_contact_pairs(disabled)), 0)

                simulation = _advance(_create_simulation(model), 240)
                np.testing.assert_allclose(simulation[1].body_q.numpy()[:, 2], [0.1, 0.2], atol=0.015)
                solver = simulation[0]
                solver.mjw_model.opt.disableflags |= int(DisableBit.CONTACT)
                simulation = _advance(simulation, 120)
                self.assertTrue(np.all(simulation[1].body_q.numpy()[:, 2] < -0.8))
                self.assertEqual(len(_read_contact_pairs(simulation)), 0)

                solver.mjw_model.opt.disableflags &= ~int(DisableBit.CONTACT)
                solver.reset(simulation[1])
                simulation = _advance(simulation, 240)
                np.testing.assert_allclose(simulation[1].body_q.numpy()[:, 2], [0.1, 0.2], atol=0.015)
                self.assertEqual(set(_read_contact_pairs(simulation).flat), set(range(model.shape_count)))

    def test_large_pair_graph_keeps_real_contacts_without_phantom_slots(self):
        """Filter absent colliders when a large pair graph exceeds the exact mask compiler budget."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane()
                # Forty-six same-body shapes yield 1,035 exclusions, exceeding
                # the mask compiler's exact-cover budget. Start below ground so
                # even tiny unused slots would produce contacts without filtering.
                for count in (1, 46):
                    builder.begin_world()
                    body = builder.add_link(
                        mass=1.0,
                        inertia=wp.mat33(np.eye(3)),
                        xform=wp.transform((0.0, 0.0, -0.15), wp.quat_identity()),
                    )
                    joint = builder.add_joint_free(body)
                    builder.add_articulation([joint])
                    for slot in range(count):
                        builder.add_shape_sphere(
                            body,
                            radius=0.1,
                            xform=wp.transform((0.3 * slot, 0.0, 0.0), wp.quat_identity()),
                            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
                        )
                    builder.end_world()
                model = builder.finalize(device=device)
                solver, state, next_state, control, contacts = _create_simulation(model)
                solver.step(state, next_state, control, None, 1.0e-6)
                pairs = _read_contact_pairs((solver, next_state, state, control, contacts))
                self.assertEqual(len(pairs), 47)
                self.assertEqual({tuple(sorted(pair)) for pair in pairs}, {(0, shape) for shape in range(1, 48)})
                worlds = np.max(model.shape_world.numpy()[pairs], axis=1)
                np.testing.assert_array_equal(np.bincount(worlds, minlength=2), [1, 46])
                self.assertTrue(np.isfinite(contacts.force.numpy()[: len(pairs)]).all())

    def test_native_trajectories_and_contact_ids_match_isolated_worlds(self):
        """Match isolated native dynamics and report contacts for every original convex hull."""
        variants = [(1, 0.1), (3, 0.2), (2, 0.3)]
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = build_model(variants, device)
                shape_count = model.shape_count
                mass = model.body_mass.numpy().copy()
                inertia = model.body_inertia.numpy().copy()
                meshes = [source.vertices.copy() for source in model.shape_source if source is not None]
                simulation = _create_simulation(model)
                references = [
                    _create_simulation(build_model([variant], device), heterogeneous=False) for variant in variants
                ]
                for steps in (60, 60, 120):
                    simulation = _advance(simulation, steps)
                    references = [_advance(reference, steps) for reference in references]
                    for world, reference in enumerate(references):
                        np.testing.assert_allclose(
                            simulation[1].body_q.numpy()[world], reference[1].body_q.numpy()[0], atol=2.0e-3
                        )
                        np.testing.assert_allclose(
                            simulation[1].body_qd.numpy()[world], reference[1].body_qd.numpy()[0], atol=2.0e-2
                        )

                np.testing.assert_allclose(simulation[1].body_q.numpy()[:, 2], [0.1, 0.2, 0.3], atol=0.015)
                pairs = _read_contact_pairs(simulation)
                self.assertEqual(set(pairs.flat), set(range(shape_count)))
                shape_world = model.shape_world.numpy()
                for shape_a, shape_b in pairs:
                    self.assertTrue(
                        shape_world[shape_a] == -1
                        or shape_world[shape_b] == -1
                        or shape_world[shape_a] == shape_world[shape_b]
                    )
                self.assertTrue(np.isfinite(simulation[4].force.numpy()[: len(pairs)]).all())
                self.assertEqual(model.shape_count, shape_count)
                np.testing.assert_array_equal(model.body_mass.numpy(), mass)
                np.testing.assert_array_equal(model.body_inertia.numpy(), inertia)
                for expected, source in zip(
                    meshes, [source for source in model.shape_source if source is not None], strict=True
                ):
                    np.testing.assert_array_equal(source.vertices, expected)

                before_reset = simulation[1].body_q.numpy().copy()
                simulation[0].reset(
                    simulation[1], world_mask=wp.array([False, True, False, False], dtype=bool, device=device)
                )
                simulation = _advance(simulation, 12)
                after_reset = simulation[1].body_q.numpy()
                self.assertGreater(after_reset[1, 2], 0.9)
                np.testing.assert_allclose(after_reset[[0, 2]], before_reset[[0, 2]], atol=2.0e-3)
                reset_pairs = _read_contact_pairs(simulation)
                self.assertNotIn(1, shape_world[reset_pairs].flat)

    def test_empty_world_has_no_native_phantom_collider(self):
        """Let the shapeless world fall freely while other native collision worlds settle."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = build_model([(0, 0.1), (3, 0.2), (1, 0.3)], device)
                simulation = _advance(_create_simulation(model), 240)
                positions = simulation[1].body_q.numpy()[:, 2]
                self.assertLess(positions[0], -3.0)
                np.testing.assert_allclose(positions[1:], [0.2, 0.3], atol=0.015)
                pairs = _read_contact_pairs(simulation)
                self.assertEqual(set(pairs.flat), set(range(model.shape_count)))
                self.assertNotIn(0, model.shape_world.numpy()[pairs].flat)

    def test_body_to_body_contacts_preserve_per_world_pair_filters(self):
        """Stack convex bodies only in worlds whose body-shape pairs are enabled."""
        variants = [(1, 1, False), (2, 3, True), (3, 2, False)]
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _build_stack_model(variants, device)
                simulation = _advance(_create_simulation(model), 240)
                positions = simulation[1].body_q.numpy()[:, 2].reshape(3, 2)
                np.testing.assert_allclose(positions[:, 0], 0.1, atol=0.015)
                np.testing.assert_allclose(positions[:, 1], [0.28, 0.08, 0.28], atol=0.015)
                pairs = _read_contact_pairs(simulation)
                self.assertTrue(np.all((pairs >= 0) & (pairs < model.shape_count)))
                self.assertFalse(model.shape_collision_filter_mask(pairs).any())
                bodies = model.shape_body.numpy()[pairs]
                body_pairs = {tuple(sorted(pair)) for pair in bodies if np.all(pair >= 0)}
                self.assertEqual(body_pairs, {(0, 1), (4, 5)})

    def test_mixed_primitive_and_convex_slots_keep_world_geometry(self):
        """Preserve sphere and convex geometry when each world activates different geom slots."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for world, height in enumerate((0.1, 0.2, 0.3)):
            template = newton.ModelBuilder()
            body = template.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
            if world == 1:
                template.add_shape_convex_hull(body, mesh=newton.Mesh.create_box(0.2, 0.2, height))
            else:
                template.add_shape_sphere(body, radius=height)
            builder.add_world(template)
        model = builder.finalize(device="cpu")
        simulation = _advance(_create_simulation(model), 240)
        np.testing.assert_allclose(simulation[1].body_q.numpy()[:, 2], [0.1, 0.2, 0.3], atol=0.015)
        self.assertEqual(set(_read_contact_pairs(simulation).flat), set(range(model.shape_count)))

    def test_geometry_and_filter_updates_require_rebuilding(self):
        """Reject unsupported geometry and filter edits before changing native geom properties."""
        model = build_model([(1, 0.1), (3, 0.2)], "cpu")
        solver = _create_simulation(model)[0]
        original_sizes = solver.mjw_model.geom_size.numpy().copy()
        for name in ("shape_scale", "shape_collision_group"):
            with self.subTest(attribute=name):
                attribute = getattr(model, name)
                original = attribute.numpy().copy()
                modified = original.copy()
                if name == "shape_scale":
                    modified[1] *= 1.1
                else:
                    modified[1] = 0
                attribute.assign(modified)
                with self.assertRaisesRegex(ValueError, f"changing {name}"):
                    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                np.testing.assert_array_equal(solver.mjw_model.geom_size.numpy(), original_sizes)
                attribute.assign(original)
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    def test_shared_mesh_scales_and_transforms_match_isolated_worlds(self):
        """Match native mesh frames and bounds for a shared mesh with differing scales and poses."""
        base = newton.Mesh.create_box(0.11, 0.07, 0.09, compute_inertia=False)
        rotation = wp.quat_rpy(0.25, 0.4, 0.15)
        vertices = np.array([wp.quat_rotate(rotation, wp.vec3(vertex)) for vertex in base.vertices])
        vertices += np.array([0.02, -0.03, 0.01])
        mesh = newton.Mesh(vertices, base.indices, compute_inertia=False)
        variants = [
            (1, (0.8, 1.1, 0.9), (0.17, -0.23, 0.3)),
            (3, (1.2, 0.7, 1.4), (-0.1, 0.2, -0.35)),
            (2, (0.6, 1.4, 1.1), (0.2, 0.15, 0.1)),
        ]

        def build(selected_variants):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
            for count, scale, angles in selected_variants:
                template = newton.ModelBuilder()
                body = template.add_body(
                    mass=1.0,
                    inertia=wp.mat33(np.diag([0.02, 0.02, 0.03])),
                    xform=wp.transform((0.0, 0.0, 0.6), wp.quat_identity()),
                )
                for part in range(count):
                    template.add_shape_convex_hull(
                        body,
                        mesh=mesh,
                        scale=scale,
                        xform=wp.transform((0.22 * (part - (count - 1) / 2), 0.015, 0.025), wp.quat_rpy(*angles)),
                        cfg=cfg,
                    )
                builder.add_world(template)
            return builder.finalize(device="cpu")

        model = build(variants)
        original_scale = model.shape_scale.numpy().copy()
        original_transform = model.shape_transform.numpy().copy()
        simulation = _create_simulation(model)
        references = [_create_simulation(build([variant]), heterogeneous=False) for variant in variants]
        for steps in (60, 60):
            simulation = _advance(simulation, steps)
            references = [_advance(reference, steps) for reference in references]
            for world, reference in enumerate(references):
                np.testing.assert_allclose(
                    simulation[1].body_q.numpy()[world], reference[1].body_q.numpy()[0], atol=2.0e-3
                )
                np.testing.assert_allclose(
                    simulation[1].body_qd.numpy()[world], reference[1].body_qd.numpy()[0], atol=2.0e-2
                )
        self.assertGreater(len(_read_contact_pairs(simulation)), 0)
        np.testing.assert_array_equal(model.shape_scale.numpy(), original_scale)
        np.testing.assert_array_equal(model.shape_transform.numpy(), original_transform)
        np.testing.assert_array_equal(mesh.vertices, vertices.astype(np.float32))

    def test_sleeping_and_masked_reset_preserve_other_worlds(self):
        """Keep unselected native collision worlds asleep when resetting one ragged world."""
        with wp.ScopedDevice("cpu"):
            model = build_model([(1, 0.1), (3, 0.2)], "cpu")
            simulation = _advance(_create_simulation(model, enable_sleeping=True, sleep_tolerance=0.01), 360)
            solver = simulation[0]
            np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), [0, 0])
            np.testing.assert_allclose(simulation[1].body_q.numpy()[:, 2], [0.1, 0.2], atol=0.015)
            before_reset = simulation[1].body_q.numpy().copy()
            solver.reset(simulation[1], world_mask=wp.array([True, False, False], dtype=bool, device="cpu"))
            simulation = _advance(simulation, 1)
            np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), [1, 0])
            self.assertGreater(simulation[1].body_q.numpy()[0, 2], 0.99)
            np.testing.assert_array_equal(simulation[1].body_q.numpy()[1], before_reset[1])

    def test_fixed_body_geom_poses_and_updates_match_isolated_worlds(self):
        """Settle on each world's fixed-body colliders before and after local shape pose updates."""
        body_rotation = wp.quat_rpy(0.2, -0.1, 0.3)
        body_pose = wp.transform((0.05, -0.08, 0.1), body_rotation)
        shape_rotation = wp.quat_inverse(body_rotation)

        def build(variants, device):
            builder = newton.ModelBuilder()
            platform_shapes = []
            centers = []
            for count, local_height in variants:
                builder.begin_world()
                platform = builder.add_link(xform=body_pose, label="platform")
                # Leave this fixed root outside an articulation so it is exported as a welded body, not mocap.
                builder.add_joint_fixed(-1, platform, parent_xform=body_pose)
                local_center = wp.vec3(0.03, 0.02, local_height)
                center = wp.transform_point(body_pose, local_center)
                centers.append(np.array(center))
                platform_shapes.append(
                    builder.add_shape_box(
                        platform,
                        hx=0.3,
                        hy=0.2,
                        hz=0.1,
                        xform=wp.transform(local_center, shape_rotation),
                    )
                )
                for extra in range(count - 1):
                    builder.add_shape_box(
                        platform,
                        hx=0.1,
                        hy=0.1,
                        hz=0.1,
                        xform=wp.transform((10.0 + extra, 0.0, 0.0), shape_rotation),
                    )
                ball = builder.add_body(xform=wp.transform((center[0], center[1], 1.3), wp.quat_identity()))
                builder.add_shape_sphere(ball, radius=0.1)
                builder.end_world()
            return builder.finalize(device=device), platform_shapes, np.array(centers)

        variants = [(1, 0.1), (2, 0.5)]
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model, platform_shapes, centers = build(variants, device)
                with self.assertWarnsRegex(UserWarning, "standalone world roots"):
                    simulation = _create_simulation(model)
                reference_models = [build([variant], device) for variant in variants]
                references = []
                for reference in reference_models:
                    with self.assertWarnsRegex(UserWarning, "standalone world roots"):
                        references.append(_create_simulation(reference[0], heterogeneous=False))
                expected_heights = centers[:, 2] + 0.2

                for lower_platforms in (False, True):
                    if lower_platforms:
                        local_delta = np.array(wp.quat_rotate(shape_rotation, wp.vec3(0.0, 0.0, -0.15)))
                        for current_model, shapes, current_simulation in [
                            (model, platform_shapes, simulation),
                            *[
                                (reference_model[0], reference_model[1], reference_simulation)
                                for reference_model, reference_simulation in zip(
                                    reference_models, references, strict=True
                                )
                            ],
                        ]:
                            transforms = current_model.shape_transform.numpy().copy()
                            transforms[shapes, :3] += local_delta
                            current_model.shape_transform.assign(transforms)
                            current_simulation[0].notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                            current_simulation[0].reset(current_simulation[1])
                        expected_heights -= 0.15

                    simulation = _advance(simulation, 240)
                    references = [_advance(reference, 240) for reference in references]
                    np.testing.assert_allclose(simulation[1].body_q.numpy()[1::2, 2], expected_heights, atol=0.015)
                    for world, reference in enumerate(references):
                        np.testing.assert_allclose(
                            simulation[1].body_q.numpy()[2 * world + 1],
                            reference[1].body_q.numpy()[1],
                            atol=2.0e-3,
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph replay requires a CUDA device")
    def test_shape_property_updates_with_cuda_graph(self):
        """Match eager native dynamics when graph replay updates collider poses and friction."""
        with wp.ScopedDevice("cuda:0"):
            variants = [(1, 0.1), (3, 0.2)]
            model = build_model(variants, "cuda:0")
            reference_model = build_model(variants, "cuda:0")
            simulation = _advance(_create_simulation(model), 2)
            reference = _advance(_create_simulation(reference_model), 2)
            transforms = model.shape_transform.numpy().copy()
            updated_transforms = wp.clone(model.shape_transform)
            updated_friction = wp.clone(model.shape_material_mu)

            def update_and_step(current_simulation):
                """Synchronize device-side shape edits and advance both state buffers."""
                solver = current_simulation[0]
                wp.copy(solver.model.shape_transform, updated_transforms)
                wp.copy(solver.model.shape_material_mu, updated_friction)
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                _advance(current_simulation, 2)

            update_and_step(simulation)
            update_and_step(reference)
            with wp.ScopedCapture() as capture:
                update_and_step(simulation)

            for phase in range(2):
                with self.subTest(phase=phase):
                    poses = transforms.copy()
                    poses[0, 2] += 0.03
                    poses[1:, 2] -= 0.05 * phase
                    friction = np.linspace(0.3, 0.6, model.shape_count, dtype=np.float32) + 0.5 * phase
                    updated_transforms.assign(poses)
                    updated_friction.assign(friction)
                    for _ in range(120):
                        wp.capture_launch(capture.graph)
                        update_and_step(reference)

                    np.testing.assert_allclose(simulation[1].body_q.numpy(), reference[1].body_q.numpy(), atol=2.0e-3)
                    np.testing.assert_allclose(simulation[1].body_qd.numpy(), reference[1].body_qd.numpy(), atol=2.0e-2)
                    np.testing.assert_allclose(
                        simulation[1].body_q.numpy()[:, 2],
                        np.array([0.1, 0.2]) + 0.03 + 0.05 * phase,
                        atol=0.015,
                    )
                    mapping = simulation[0].mjc_geom_to_newton_shape.numpy()
                    valid = mapping >= 0
                    np.testing.assert_allclose(
                        simulation[0].mjw_model.geom_friction.numpy()[:, :, 0][valid], friction[mapping[valid]]
                    )
                    self.assertEqual(set(_read_contact_pairs(simulation).flat), set(range(model.shape_count)))

            scales = model.shape_scale.numpy().copy()
            scales[1] *= 1.1
            model.shape_scale.assign(scales)
            with self.assertRaisesRegex(ValueError, "changing shape_scale"):
                simulation[0].notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph replay requires a CUDA device")
    def test_sixteen_native_variants_with_cuda_graph(self):
        """Replay native GPU collision and dynamics for sixteen convex decompositions."""
        variants = [(1 + (world * 3) % 5, 0.08 + world * 0.01) for world in range(16)]
        with wp.ScopedDevice("cuda:0"):
            model = build_model(variants, "cuda:0")
            simulation = _advance(_create_simulation(model), 2)
            with wp.ScopedCapture() as capture:
                _advance(simulation, 2)
            for _ in range(180):
                wp.capture_launch(capture.graph)
            np.testing.assert_allclose(
                simulation[1].body_q.numpy()[:, 2], [height for _, height in variants], atol=0.015
            )
            self.assertTrue(np.isfinite(simulation[1].body_qd.numpy()).all())
            self.assertEqual(set(_read_contact_pairs(simulation).flat), set(range(model.shape_count)))


if __name__ == "__main__":
    unittest.main()
