# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import replace

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _variant(kind: str, mu: float = 1.0, tag: int = 0) -> newton.ModelBuilder:
    """Build a single-body variant: a 125 kg cube, the same cube as a mesh, or two small offset spheres."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="tag", namespace="test", frequency=newton.Model.AttributeFrequency.SHAPE, dtype=wp.int32
        )
    )
    body = builder.add_body(label=kind)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
    attributes = {"test:tag": tag}
    if kind == "cube":
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25, cfg=cfg, custom_attributes=attributes)
    elif kind == "mesh":
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.25), cfg=cfg, custom_attributes=attributes)
    else:
        for z in (-0.05, 0.05):
            xform = wp.transform((0.0, 0.0, z), wp.quat_identity())
            builder.add_shape_sphere(body, xform=xform, radius=0.05, cfg=cfg, custom_attributes=attributes)
    return builder


def _scene(world_count: int, *, kinds=("cube", "spheres"), initial: int = 0, height: float = 0.2):
    """Replicate one body with variants per world above a ground plane."""
    env = newton.ModelBuilder()
    body = env.add_body(xform=wp.transform((0.0, 0.0, height), wp.quat_identity()), label="object")
    env.add_body_variants(body, [_variant(kind) for kind in kinds], initial=initial)
    scene = newton.ModelBuilder()
    scene.replicate(env, world_count)
    scene.rigid_gap = 0.01
    scene.add_ground_plane()
    return scene


def _select(model, variants, bodies=None):
    return model.select_body_variants(
        wp.array(variants, dtype=wp.int32, device=model.device),
        bodies=None if bodies is None else wp.array(bodies, dtype=wp.int32, device=model.device),
    )


class TestBodyVariants(unittest.TestCase):
    def test_finalize_tables_and_initial_selection(self):
        cube, spheres = _variant("cube"), _variant("spheres")
        env = newton.ModelBuilder()
        body = env.add_body(mass=2.0, com=(0.0, 0.0, 1.0), inertia=wp.diag(wp.vec3(0.1)), label="object")
        env.add_shape_capsule(body, radius=0.01, half_height=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        env.add_body_variants(body, [cube, spheres], initial=1, labels=["a", "b"])
        scene = newton.ModelBuilder()
        scene.replicate(env, 2)
        model = scene.finalize(device="cpu")

        # Per world: the fixed capsule and two variant rows. The worlds share one copy of the variants.
        self.assertEqual(model.shape_count, 6)
        self.assertEqual(model._variant_shape_count, 3)
        self.assertEqual(model.body_variant_labels(1), ["a", "b"])
        np.testing.assert_array_equal(model.body_variant.numpy(), [1, 1])
        self.assertEqual(model.shape_label[1], "object/variant_collider_0")
        self.assertEqual(model.body_shapes[1], [3, 4, 5])

        # The fixed mass at z = 1 combines with the spheres' mass centered at the origin.
        sphere_mass = spheres.body_mass[0]
        mass = 2.0 + sphere_mass
        np.testing.assert_allclose(model.body_mass.numpy(), [mass, mass], rtol=1e-6)
        np.testing.assert_allclose(model.body_inv_mass.numpy(), [1.0 / mass] * 2, rtol=1e-6)
        np.testing.assert_allclose(model.body_com.numpy()[:, 2], [2.0 / mass] * 2, rtol=1e-6)
        offset_inertia = 2.0 * (1.0 - 2.0 / mass) ** 2 + sphere_mass * (2.0 / mass) ** 2
        expected_ixx = 0.1 + np.asarray(spheres.body_inertia[0]).reshape(3, 3)[0, 0] + offset_inertia
        np.testing.assert_allclose(model.body_inertia.numpy()[:, 0, 0], [expected_ixx] * 2, rtol=1e-5)

    def test_select_body_variants(self):
        env = newton.ModelBuilder()
        body = env.add_body(xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()), label="object")
        spheres = _variant("spheres", mu=0.2, tag=2)
        spheres.shape_opacity[:] = [0.5, 0.5]
        env.add_body_variants(body, [_variant("cube", mu=0.9, tag=1), spheres])
        scene = newton.ModelBuilder()
        scene.replicate(env, 3)
        model = scene.finalize(device="cpu")
        initial_mass = model.body_mass.numpy().copy()

        flags = _select(model, [1, 0, 5], bodies=[0, 1, 2])
        self.assertEqual(
            flags,
            newton.ModelFlags.SHAPE_TYPE_AND_SOURCE
            | newton.ModelFlags.SHAPE_PROPERTIES
            | newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
        )
        # Body 2's index is out of range, so it keeps the cube.
        np.testing.assert_array_equal(model.body_variant.numpy(), [1, 0, 0])
        np.testing.assert_array_equal(model.shape_type.numpy()[:3], [newton.GeoType.SPHERE] * 2 + [newton.GeoType.BOX])
        # The cube leaves the second row empty: it neither collides nor renders.
        live = int(newton.ShapeFlags.VISIBLE | newton.ShapeFlags.COLLIDE_SHAPES)
        np.testing.assert_array_equal(model.shape_flags.numpy()[[1, 3, 5]] & live, [live, 0, 0])
        # Materials, opacity, and shape custom attributes follow the variant.
        np.testing.assert_allclose(model.shape_material_mu.numpy()[:3], [0.2, 0.2, 0.9])
        np.testing.assert_allclose(model.shape_opacity.numpy()[:3], [0.5, 0.5, 1.0])
        np.testing.assert_array_equal(model.test.tag.numpy()[:3], [2, 2, 1])
        self.assertLess(model.body_mass.numpy()[0], initial_mass[0])
        np.testing.assert_allclose(model.body_mass.numpy()[1:], initial_mass[1:])

        # Ray casts follow the BVH refit: the ray passes above world 0's spheres but hits world 1's cube.
        model.bvh_refit_shapes(model.state())
        shape_id = wp.zeros(2, dtype=wp.int32, device="cpu")
        newton.intersect_ray(
            model,
            ray_origins=wp.array([(-2.0, 0.0, 0.4)] * 2, dtype=wp.vec3, device="cpu"),
            ray_directions=wp.array([(1.0, 0.0, 0.0)] * 2, dtype=wp.vec3, device="cpu"),
            ray_worlds=wp.array([0, 1], dtype=wp.int32, device="cpu"),
            out_shape_id=shape_id,
        )
        np.testing.assert_array_equal(shape_id.numpy(), [-1, 2])

        _select(model, [0, -1, 1])
        np.testing.assert_array_equal(model.body_variant.numpy(), [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "one variant per body"):
            _select(model, [0])

    def test_variant_ids_follow_bodies(self):
        """Variant ids are positions in each body's list, whatever else the scene declares."""
        scene = newton.ModelBuilder()
        scene.add_body_variants(scene.add_body(label="first"), [_variant("spheres"), _variant("cube")])
        env = newton.ModelBuilder()
        env.add_body_variants(env.add_body(label="object"), [_variant("cube"), _variant("spheres")], initial=1)
        scene.replicate(env, 2)
        model = scene.finalize(device="cpu")

        _select(model, [0, 0], bodies=[1, 2])
        np.testing.assert_array_equal(model.shape_type.numpy()[[2, 4]], [newton.GeoType.BOX] * 2)
        self.assertEqual(model.body_variant_labels(1), ["cube", "spheres"])
        self.assertEqual(model.body_variant_labels(0), ["spheres", "cube"])
        with self.assertRaises(IndexError):
            model.body_variant_labels(-1)

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
        hydroelastic = newton.ModelBuilder()
        hydroelastic.add_shape_sphere(
            hydroelastic.add_body(),
            radius=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, sdf_max_resolution=8),
        )
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            builder.add_body_variants(body, [hydroelastic])
        with self.assertRaises(IndexError):
            builder.add_body_variants(body, [_variant("cube")], initial=1)
        builder.add_body_variants(body, [_variant("cube")])
        with self.assertRaisesRegex(ValueError, "already has variants"):
            builder.add_body_variants(body, [_variant("cube")])
        grouped = _variant("cube")
        grouped.shape_collision_group[0] = 2
        with self.assertRaisesRegex(ValueError, "collision groups"):
            builder.add_body_variants(builder.add_body(), [_variant("cube"), grouped])
        retyped = _variant("cube")
        retyped.custom_attributes["test:tag"] = replace(retyped.custom_attributes["test:tag"], dtype=wp.float32)
        with self.assertRaisesRegex(ValueError, "declared differently"):
            builder.add_body_variants(builder.add_body(), [_variant("cube"), retyped])
        stateful = _variant("cube")
        stateful.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="heat",
                frequency=newton.Model.AttributeFrequency.SHAPE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                values={0: 1.0},
            )
        )
        with self.assertRaisesRegex(ValueError, "STATE"):
            builder.add_body_variants(builder.add_body(), [stateful])
        worlds = newton.ModelBuilder()
        worlds.begin_world()
        first = worlds.add_body()
        worlds.end_world()
        worlds.begin_world()
        worlds.add_shape_sphere(worlds.add_body(), radius=0.1)
        worlds.end_world()
        with self.assertRaisesRegex(ValueError, "later worlds"):
            worlds.add_body_variants(first, [_variant("cube")])
        # The rows were checked before any was appended.
        self.assertEqual(builder.shape_count, 1)
        with self.assertRaises(NotImplementedError):
            builder.collapse_fixed_joints()

    def test_rows_empty_at_construction(self):
        """Rows that start empty still count for world starts and broad-phase pair capacity."""
        one = newton.ModelBuilder()
        one.add_shape_sphere(one.add_body(), radius=0.05)
        two = newton.ModelBuilder()
        body = two.add_body()
        for x in (-0.1, 0.1):
            two.add_shape_sphere(body, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()), radius=0.05)
        builder = newton.ModelBuilder()
        builder.add_body_variants(
            builder.add_body(xform=wp.transform((0.0, 0.0, 0.04), wp.quat_identity())), [one, two]
        )
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        np.testing.assert_array_equal(model.shape_world_start.numpy(), [3, 3, 3])

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        _select(model, [1])
        pipeline.collide(model.state(), contacts)
        count = contacts.rigid_contact_count.numpy()[0]
        self.assertEqual(
            set(contacts.rigid_contact_shape0.numpy()[:count]) | set(contacts.rigid_contact_shape1.numpy()[:count]),
            {0, 1, 2},
        )

    def test_pipeline_covers_variant_mesh_properties(self):
        """A row that starts as a box can take a surface-velocity mesh later."""
        conveyor = newton.ModelBuilder()
        mesh = newton.Mesh.create_box(0.25)
        mesh.enable_surface_velocity = True
        conveyor.add_shape_mesh(conveyor.add_body(), mesh=mesh)
        builder = newton.ModelBuilder()
        builder.add_body_variants(builder.add_body(), [_variant("cube"), conveyor])
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model)
        self.assertIsNotNone(pipeline.contacts().rigid_contact_surface_velocity)

    def test_viewer_renders_selected_variant(self):
        model = _scene(2).finalize(device="cpu")
        viewer = newton.viewer.ViewerNull()
        viewer.set_model(model)

        def rendered_shapes():
            return sorted(int(s) for batch in viewer._shape_instances.values() for s in batch.model_shapes)

        # World 0 then world 1, each with two rows holding the cube and nothing, then the ground plane.
        self.assertEqual(rendered_shapes(), [0, 2, 4])
        viewer.notify_model_changed(_select(model, [1], bodies=[1]))
        self.assertEqual(rendered_shapes(), [0, 2, 3, 4])


def test_contacts_follow_selection(test, device):
    """Only the selected variant collides, in every broad phase mode, including type changes after construction."""
    # The cube's bottom face penetrates the ground; the spheres hover above it.
    model = _scene(2, kinds=("cube", "spheres", "mesh"), height=0.24).finalize(device=device)
    for mode in ("explicit", "nxn", "sap"):
        with test.subTest(broad_phase=mode):
            _select(model, [0, 0])
            pipeline = newton.CollisionPipeline(model, broad_phase=mode)
            contacts = pipeline.contacts()
            # Two rows per world, then the ground plane (shape 4). World 0 switches to the mesh,
            # whose narrow-phase stages exist although no body held a mesh at construction.
            for variants, shapes in (([2, 1], {0, 4}), ([1, 0], {2, 4})):
                _select(model, variants)
                pipeline.collide(model.state(), contacts)
                count = contacts.rigid_contact_count.numpy()[0]
                touching = contacts.rigid_contact_shape0.numpy()[:count].tolist()
                touching += contacts.rigid_contact_shape1.numpy()[:count].tolist()
                test.assertEqual(set(touching), shapes)


def test_rows_keep_their_role(test, device):
    """A row holds shapes of one role, so a site in one variant never takes a collider's row in another."""
    with_site = newton.ModelBuilder()
    body = with_site.add_body(label="with_site")
    with_site.add_site(body, label="site")
    with_site.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    two_boxes = newton.ModelBuilder()
    body = two_boxes.add_body(label="two_boxes")
    for x in (-0.15, 0.15):
        two_boxes.add_shape_box(body, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1)
    env = newton.ModelBuilder()
    body = env.add_body(xform=wp.transform((0.0, 0.0, 0.15), wp.quat_identity()), label="object")
    env.add_body_variants(body, [with_site, two_boxes])
    scene = newton.ModelBuilder()
    scene.replicate(env, 2)
    scene.add_ground_plane()
    model = scene.finalize(device=device)
    test.assertEqual(
        model.shape_label[:3], ["object/variant_collider_0", "object/variant_collider_1", "object/variant_site_0"]
    )

    # MuJoCo compiles its geoms while world 1's second collider row is still empty.
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
    test.assertTrue(np.all(solver.newton_shape_to_mjc_geom.numpy()[[0, 1, 3, 4]] >= 0))
    solver.notify_model_changed(_select(model, [0, 1]))
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    for _ in range(100):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 100.0)
        state_0, state_1 = state_1, state_0
    count = contacts.rigid_contact_count.numpy()[0]
    touching = set(contacts.rigid_contact_shape0.numpy()[:count]) | set(contacts.rigid_contact_shape1.numpy()[:count])
    test.assertEqual(touching, {0, 3, 4, 6})
    # Both bodies rest upright on their boxes.
    body_q = state_0.body_q.numpy()
    np.testing.assert_allclose(body_q[:, 2], [0.1, 0.1], atol=0.01)
    np.testing.assert_allclose(np.abs(body_q[:, 6]), [1.0, 1.0], atol=1e-3)


def test_mujoco_rest_height_follows_selection(test, device):
    """A falling body settles on the selected variant's geometry and mass."""
    model = _scene(2, height=0.5).finalize(device=device)
    # MuJoCo compiles one geom layout for all worlds at construction; select afterwards.
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
    solver.notify_model_changed(_select(model, [0, 1]))
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
        solver.render_mujoco_viewer()
    with test.assertRaises(NotImplementedError):
        newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)


def test_selection_replays_from_cuda_graph(test, device):
    """A selection and the solver notifications it needs record into a CUDA graph and replay."""
    scene = _scene(2, height=0.5)
    scene.add_particle(pos=wp.vec3(0.0, 0.0, 2.0), vel=wp.vec3(0.0), mass=0.1, radius=0.01)
    scene.color()
    model = scene.finalize(device=device)
    mujoco = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
    vbd = newton.solvers.SolverVBD(
        model, rigid_compliant_alm=True, rigid_soft_enable_dat=True, collision_pipeline=newton.CollisionPipeline(model)
    )
    cube_radius = vbd._rigid_dat_body_bounding_radius.numpy()[0]

    variants = wp.zeros(2, dtype=wp.int32, device=device)
    with wp.ScopedCapture(device=device) as capture:
        flags = model.select_body_variants(variants)
        mujoco.notify_model_changed(flags)
        vbd.notify_model_changed(flags)
    for choice in ([1, 0], [0, 1]):
        variants.assign(choice)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(model.body_variant.numpy(), choice)
        expected = [newton.GeoType.SPHERE if c else newton.GeoType.BOX for c in choice]
        np.testing.assert_array_equal(model.shape_type.numpy()[[0, 2]], expected)
        np.testing.assert_allclose(mujoco.mjw_model.body_mass.numpy()[:, 1], model.body_mass.numpy(), rtol=1e-6)
        # The spheres reach less far from the center of mass than the cube's corners.
        radius = vbd._rigid_dat_body_bounding_radius.numpy()
        test.assertEqual([r < cube_radius for r in radius], [bool(c) for c in choice])


devices = get_test_devices()
add_function_test(TestBodyVariants, "test_contacts_follow_selection", test_contacts_follow_selection, devices=devices)
add_function_test(TestBodyVariants, "test_rows_keep_their_role", test_rows_keep_their_role, devices=devices)
add_function_test(
    TestBodyVariants,
    "test_selection_replays_from_cuda_graph",
    test_selection_replays_from_cuda_graph,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestBodyVariants,
    "test_mujoco_rest_height_follows_selection",
    test_mujoco_rest_height_follows_selection,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
