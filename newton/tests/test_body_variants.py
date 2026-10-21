# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _variant(kind: str) -> newton.ModelBuilder:
    """Build a single-body variant: a 1 kg-scale cube or two small offset spheres."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    body = builder.add_body(label=kind)
    if kind == "cube":
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
    else:
        for z in (-0.05, 0.05):
            builder.add_shape_sphere(body, xform=wp.transform((0.0, 0.0, z), wp.quat_identity()), radius=0.05)
    return builder


def _scene(world_count: int, *, initial: int = 0, height: float = 0.2, fixed_mass: float = 0.0):
    """Replicate one variant body per world above a ground plane."""
    env = newton.ModelBuilder()
    body = env.add_body(xform=wp.transform((0.0, 0.0, height), wp.quat_identity()), mass=fixed_mass, label="object")
    env.add_body_variants(body, [_variant("cube"), _variant("spheres")], initial=initial)
    scene = newton.ModelBuilder()
    scene.replicate(env, world_count)
    scene.rigid_gap = 0.01
    scene.add_ground_plane()
    return scene


class TestBodyVariants(unittest.TestCase):
    def test_finalize_tables_and_initial_selection(self):
        cube, spheres = _variant("cube"), _variant("spheres")
        env = newton.ModelBuilder()
        body = env.add_body(mass=2.0, com=(0.0, 0.0, 1.0), label="object")
        env.add_shape_capsule(body, radius=0.01, half_height=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        env.add_body_variants(body, [cube, spheres], initial=1, labels=["a", "b"])
        scene = newton.ModelBuilder()
        scene.replicate(env, 2)
        model = scene.finalize(device="cpu")

        self.assertEqual(model.body_variant_count, 4)
        self.assertEqual(model.body_variant_label, ["a", "b", "a", "b"])
        np.testing.assert_array_equal(model.body_variant_start.numpy(), [0, 2, 4])
        np.testing.assert_array_equal(model.body_variant.numpy(), [1, 1])
        self.assertEqual(model.shape_label[1], "object/a/shape_0")
        # Per world: fixed capsule, cube, two spheres.
        np.testing.assert_array_equal(model._shape_active.numpy(), [1, 0, 1, 1, 1, 0, 1, 1])
        self.assertEqual(model.body_shapes[1], [4, 5, 6, 7])

        # The fixed mass at z = 1 combines with the spheres' mass centered at the origin.
        sphere_mass = spheres.body_mass[0]
        mass = 2.0 + sphere_mass
        np.testing.assert_allclose(model.body_mass.numpy(), [mass, mass], rtol=1e-6)
        np.testing.assert_allclose(model.body_inv_mass.numpy(), [1.0 / mass] * 2, rtol=1e-6)
        np.testing.assert_allclose(model.body_com.numpy()[:, 2], [2.0 / mass] * 2, rtol=1e-6)
        offset_inertia = 2.0 * (1.0 - 2.0 / mass) ** 2 + sphere_mass * (2.0 / mass) ** 2
        expected_ixx = np.asarray(spheres.body_inertia[0]).reshape(3, 3)[0, 0] + offset_inertia
        np.testing.assert_allclose(model.body_inertia.numpy()[:, 0, 0], [expected_ixx] * 2, rtol=1e-5)

    def test_select_body_variants(self):
        model = _scene(3).finalize(device="cpu")
        initial_mass = model.body_mass.numpy().copy()
        world_mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device="cpu")

        flags = model.select_body_variants(
            wp.array([1, 0, 5], dtype=wp.int32, device="cpu"),
            bodies=wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
            world_mask=world_mask,
        )

        self.assertEqual(flags, newton.ModelFlags.SHAPE_GEOMETRY | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        # Body 1 reselects its current variant and body 2's index is out of range: neither world changed.
        np.testing.assert_array_equal(world_mask.numpy(), [True, False, False, False])
        np.testing.assert_array_equal(model.body_variant.numpy(), [1, 0, 0])
        np.testing.assert_array_equal(model._shape_active.numpy()[:3], [False, True, True])
        np.testing.assert_array_equal(model._shape_active.numpy()[3:9], [True, False, False] * 2)
        self.assertLess(model.body_mass.numpy()[0], initial_mass[0])
        np.testing.assert_allclose(model.body_mass.numpy()[1:], initial_mass[1:])

        model.select_body_variants(wp.array([0, -1, 1], dtype=wp.int32, device="cpu"))
        np.testing.assert_array_equal(model.body_variant.numpy(), [0, 0, 1])

        with self.assertRaisesRegex(ValueError, "one variant per body"):
            model.select_body_variants(wp.array([0], dtype=wp.int32, device="cpu"))

    def test_add_body_variants_validation(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        two_bodies = newton.ModelBuilder()
        two_bodies.add_body()
        two_bodies.add_body()
        with self.assertRaisesRegex(ValueError, "exactly one body"):
            builder.add_body_variants(body, [two_bodies])
        static = _variant("cube")
        static.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
        with self.assertRaisesRegex(ValueError, "not attached"):
            builder.add_body_variants(body, [static])
        with self.assertRaises(IndexError):
            builder.add_body_variants(body, [_variant("cube")], initial=1)
        builder.add_body_variants(body, [_variant("cube")])
        with self.assertRaisesRegex(ValueError, "already has variants"):
            builder.add_body_variants(body, [_variant("cube")])
        with self.assertRaises(NotImplementedError):
            builder.collapse_fixed_joints()

        # Variant shapes join the body's world even outside its world context.
        scene = newton.ModelBuilder()
        scene.begin_world()
        body = scene.add_body()
        scene.end_world()
        scene.add_body_variants(body, [_variant("cube")])
        self.assertEqual(scene.shape_world, [0])

    def test_viewer_renders_selected_variant(self):
        model = _scene(2).finalize(device="cpu")
        viewer = newton.viewer.ViewerNull()
        viewer.set_model(model)

        def rendered_shapes():
            return sorted(int(s) for batch in viewer._shape_instances.values() for s in batch.model_shapes)

        # World 0 then world 1, each with cube (0) and two spheres (1, 2), then the ground plane.
        self.assertEqual(rendered_shapes(), [0, 3, 6])
        flags = model.select_body_variants(
            wp.array([1], dtype=wp.int32, device="cpu"), bodies=wp.array([1], dtype=wp.int32, device="cpu")
        )
        viewer.notify_model_changed(flags)
        self.assertEqual(rendered_shapes(), [0, 4, 5, 6])


def test_contacts_follow_selection(test, device):
    """Only the selected variant collides, in every broad phase mode."""
    # The cube's bottom face penetrates the ground; the spheres hover above it.
    model = _scene(2, height=0.24).finalize(device=device)
    for mode in ("explicit", "nxn", "sap"):
        with test.subTest(broad_phase=mode):
            model.select_body_variants(wp.array([0, 1], dtype=wp.int32, device=device))
            pipeline = newton.CollisionPipeline(model, broad_phase=mode)
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            count = contacts.rigid_contact_count.numpy()[0]
            shapes = contacts.rigid_contact_shape0.numpy()[:count].tolist()
            shapes += contacts.rigid_contact_shape1.numpy()[:count].tolist()
            test.assertGreater(count, 0)
            test.assertEqual(set(shapes), {0, 6})

            model.select_body_variants(wp.array([1, 0], dtype=wp.int32, device=device))
            pipeline.collide(state, contacts)
            count = contacts.rigid_contact_count.numpy()[0]
            shapes = contacts.rigid_contact_shape0.numpy()[:count].tolist()
            shapes += contacts.rigid_contact_shape1.numpy()[:count].tolist()
            test.assertEqual(set(shapes), {3, 6})


def test_mujoco_rest_height_follows_selection(test, device):
    """A falling body settles on the selected variant's geometry and mass."""
    model = _scene(2, height=0.5).finalize(device=device)
    flags = model.select_body_variants(wp.array([0, 1], dtype=wp.int32, device=device))
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
    solver.notify_model_changed(flags)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    for _ in range(150):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 100.0)
        state_0, state_1 = state_1, state_0
    heights = state_0.body_q.numpy()[:, 2]
    # Cube half-extent 0.25; the lower sphere's bottom sits 0.1 below the body origin.
    np.testing.assert_allclose(heights, [0.25, 0.1], atol=0.02)

    with test.assertRaises(NotImplementedError):
        newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)


devices = get_test_devices()
add_function_test(TestBodyVariants, "test_contacts_follow_selection", test_contacts_follow_selection, devices=devices)
add_function_test(
    TestBodyVariants,
    "test_mujoco_rest_height_follows_selection",
    test_mujoco_rest_height_follows_selection,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
