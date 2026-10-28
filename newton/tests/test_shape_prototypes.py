# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _asset(kind: str) -> newton.ModelBuilder:
    """Build a single-body asset: a 125 kg cube, the same cube as a mesh, two small spheres, or a
    two-sphere asset with an authored mass of 3 kg."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    if kind == "authored":
        body = builder.add_body(mass=3.0, com=(0.0, 0.0, 0.1), inertia=wp.diag(wp.vec3(0.2)), label=kind)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    else:
        body = builder.add_body(label=kind)
        cfg = None
    if kind == "cube":
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
    elif kind == "mesh":
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.25))
    else:
        for z in (-0.05, 0.05):
            builder.add_shape_sphere(body, xform=wp.transform((0.0, 0.0, z), wp.quat_identity()), radius=0.05, cfg=cfg)
    return builder


def _scene(world_count: int, *, kinds=("cube", "spheres"), initial: int = 0, height: float = 0.2):
    """Replicate one body with a slot holding ``kinds[initial]`` per world above a ground plane."""
    env = newton.ModelBuilder()
    prototypes = [env.add_shape_prototype(_asset(kind)) for kind in kinds]
    body = env.add_body(xform=wp.transform((0.0, 0.0, height), wp.quat_identity()), label="object")
    env.add_shape_slot(body, initial=prototypes[initial])
    scene = newton.ModelBuilder()
    scene.replicate(env, world_count)
    scene.rigid_gap = 0.01
    scene.add_ground_plane()
    return scene


def _assign(model, slots, prototypes):
    return model.assign_shape_prototypes(
        wp.array(slots, dtype=wp.int32, device=model.device), wp.array(prototypes, dtype=wp.int32, device=model.device)
    )


class TestShapePrototypes(unittest.TestCase):
    def test_finalize_tables_and_initial_assignment(self):
        env = newton.ModelBuilder()
        cube = env.add_shape_prototype(_asset("cube"))
        authored = env.add_shape_prototype(_asset("authored"), label="authored")
        body = env.add_body(mass=2.0, com=(0.0, 0.0, 1.0), inertia=wp.diag(wp.vec3(0.1)), label="object")
        env.add_shape_capsule(body, radius=0.01, half_height=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        env.add_shape_slot(body, initial=authored)
        scene = newton.ModelBuilder()
        scene.replicate(env, 2)
        model = scene.finalize(device="cpu")

        # Per world: the capsule and two slot rows. The worlds share the prototypes.
        self.assertEqual(model.shape_count, 6)
        self.assertEqual(model.shape_prototype_label, ["cube", "authored"])
        np.testing.assert_array_equal(model.shape_prototype_shape_start.numpy(), [0, 1, 3])
        np.testing.assert_allclose(model.shape_prototype_mass.numpy(), [125.0, 3.0], rtol=1e-6)
        self.assertEqual(model.shape_slot_label, ["object/slot_0"] * 2)
        np.testing.assert_array_equal(model.shape_slot_prototype.numpy(), [authored, authored])
        np.testing.assert_array_equal(model.shape_slot.numpy(), [-1, 0, 0, -1, 1, 1])
        self.assertEqual(model.shape_label[1], "object/slot_0/0")
        self.assertEqual(model.body_shapes[1], [3, 4, 5])
        self.assertEqual(cube, 0)

        # The fixed mass at z = 1 combines with the asset's authored mass properties.
        mass = 2.0 + 3.0
        com = (2.0 * 1.0 + 3.0 * 0.1) / mass
        np.testing.assert_allclose(model.body_mass.numpy(), [mass, mass], rtol=1e-6)
        np.testing.assert_allclose(model.body_inv_mass.numpy(), [1.0 / mass] * 2, rtol=1e-6)
        np.testing.assert_allclose(model.body_com.numpy()[:, 2], [com, com], rtol=1e-6)
        expected_ixx = 0.1 + 0.2 + 2.0 * (1.0 - com) ** 2 + 3.0 * (0.1 - com) ** 2
        np.testing.assert_allclose(model.body_inertia.numpy()[:, 0, 0], [expected_ixx] * 2, rtol=1e-5)

    def test_assign_shape_prototypes(self):
        scene = _scene(3)
        # Declared after the slots, which were sized for two shapes: it does not fit them.
        triplet = newton.ModelBuilder()
        triplet_body = triplet.add_body()
        for x in (-0.1, 0.0, 0.1):
            triplet.add_shape_sphere(triplet_body, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()), radius=0.05)
        too_large = scene.add_shape_prototype(triplet)
        model = scene.finalize(device="cpu")
        cube, spheres = 0, 1
        initial_mass = model.body_mass.numpy().copy()

        flags = _assign(model, [0, 1, 2], [spheres, cube, too_large])
        self.assertEqual(
            flags,
            newton.ModelFlags.SHAPE_GEOMETRY
            | newton.ModelFlags.SHAPE_PROPERTIES
            | newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
        )
        np.testing.assert_array_equal(model.shape_slot_prototype.numpy(), [spheres, cube, cube])
        # Two rows per slot; the cube leaves the second one empty.
        np.testing.assert_array_equal(model.shape_type.numpy()[:3], [newton.GeoType.SPHERE] * 2 + [newton.GeoType.BOX])
        np.testing.assert_allclose(model.body_mass.numpy()[0], model.shape_prototype_mass.numpy()[spheres], rtol=1e-6)
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

        # An empty slot keeps no mass; the body has no other shapes.
        _assign(model, [0], [-1])
        np.testing.assert_array_equal(model.shape_slot_prototype.numpy()[0], -1)
        np.testing.assert_allclose(model.body_mass.numpy()[0], 0.0, atol=1e-6)

    def test_slot_of_existing_shapes(self):
        # A static source shape adds no mass to a body, but its prototype has mass at its density.
        part = newton.ModelBuilder()
        part.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
        env = newton.ModelBuilder()
        body = env.add_body(label="object")
        box = env.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
        # After loading, mark the body's shape as replaceable; its geometry becomes a prototype.
        marked = env.add_shape_slot(shapes=[box])
        part_prototype = env.add_shape_prototype(part, label="part")
        env.add_shape_slot(body, initial=part_prototype, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        scene = newton.ModelBuilder()
        scene.replicate(env, 2)
        model = scene.finalize(device="cpu")

        self.assertEqual(model.shape_count, 4)
        self.assertEqual(model.shape_prototype_label, ["object/slot_0/initial", "part"])
        np.testing.assert_array_equal(model.shape_slot_prototype.numpy(), [0, part_prototype] * 2)
        np.testing.assert_allclose(model.shape_transform.numpy()[1, 2], 0.5)
        # The marked cube (125 kg) keeps its mass; the part adds 8 kg at z = 0.5.
        np.testing.assert_allclose(model.body_mass.numpy(), [133.0, 133.0], rtol=1e-5)
        np.testing.assert_allclose(model.body_com.numpy()[:, 2], [8.0 * 0.5 / 133.0] * 2, rtol=1e-5)

        # The marked slot can hold the part and get its own geometry back.
        _assign(model, [marked], [part_prototype])
        self.assertEqual(model.shape_type.numpy()[0], newton.GeoType.BOX)
        np.testing.assert_allclose(model.shape_scale.numpy()[0], [0.1, 0.1, 0.1])
        np.testing.assert_allclose(model.body_mass.numpy()[0], 16.0, rtol=1e-5)
        _assign(model, [marked], [0])
        np.testing.assert_allclose(model.shape_scale.numpy()[0], [0.25, 0.25, 0.25])
        np.testing.assert_allclose(model.body_mass.numpy()[0], 133.0, rtol=1e-5)

    def test_custom_attributes_follow_assignment(self):
        def tagged(tag):
            builder = newton.ModelBuilder()
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="tag", namespace="test", frequency=newton.Model.AttributeFrequency.SHAPE, dtype=wp.int32
                )
            )
            builder.add_shape_sphere(builder.add_body(), radius=0.1, custom_attributes={"test:tag": tag})
            return builder

        env = newton.ModelBuilder()
        first, second = env.add_shape_prototype(tagged(1)), env.add_shape_prototype(tagged(2))
        env.add_shape_slot(env.add_body(), initial=first)
        scene = newton.ModelBuilder()
        scene.replicate(env, 2)
        model = scene.finalize(device="cpu")

        np.testing.assert_array_equal(model.test.tag.numpy(), [1, 1])
        _assign(model, [1], [second])
        np.testing.assert_array_equal(model.test.tag.numpy(), [1, 2])

    def test_validation(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        with self.assertRaisesRegex(ValueError, "Declare a prototype"):
            builder.add_shape_slot(body)
        two_bodies = newton.ModelBuilder()
        two_bodies.add_shape_sphere(two_bodies.add_body(), radius=0.1)
        two_bodies.add_body()
        with self.assertRaisesRegex(ValueError, "at most one body"):
            builder.add_shape_prototype(two_bodies)
        hydroelastic = newton.ModelBuilder()
        hydroelastic.add_shape_sphere(
            hydroelastic.add_body(),
            radius=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, sdf_max_resolution=8),
        )
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            builder.add_shape_prototype(hydroelastic)
        spheres = builder.add_shape_prototype(_asset("spheres"))
        with self.assertRaisesRegex(ValueError, "capacity"):
            builder.add_shape_slot(body, initial=spheres, capacity=1)
        slot = builder.add_shape_slot(body, initial=spheres)
        with self.assertRaisesRegex(ValueError, "already rows"):
            builder.add_shape_slot(shapes=[builder.body_shapes[body][0]])
        self.assertEqual(slot, 0)
        with self.assertRaises(NotImplementedError):
            builder.collapse_fixed_joints()

    def test_viewer_renders_assigned_prototypes(self):
        model = _scene(2).finalize(device="cpu")
        viewer = newton.viewer.ViewerNull()
        viewer.set_model(model)

        def rendered_shapes():
            return sorted(int(s) for batch in viewer._shape_instances.values() for s in batch.model_shapes)

        # World 0 then world 1, each with two slot rows holding the cube and nothing, then the ground plane.
        self.assertEqual(rendered_shapes(), [0, 2, 4])
        viewer.notify_model_changed(_assign(model, [1], [1]))
        self.assertEqual(rendered_shapes(), [0, 2, 3, 4])


def test_contacts_follow_assignment(test, device):
    """Only the assigned prototype collides, in every broad phase mode, including type changes after construction."""
    # The cube's bottom face penetrates the ground; the spheres hover above it.
    model = _scene(2, kinds=("cube", "spheres", "mesh"), height=0.24).finalize(device=device)
    for mode in ("explicit", "nxn", "sap"):
        with test.subTest(broad_phase=mode):
            _assign(model, [0, 1], [0, 0])
            pipeline = newton.CollisionPipeline(model, broad_phase=mode)
            contacts = pipeline.contacts()
            # Two rows per world, then the ground plane (shape 4). World 0 switches to the mesh,
            # whose narrow-phase stages exist although no slot held a mesh at construction.
            for prototypes, shapes in (([2, 1], {0, 4}), ([1, 0], {2, 4})):
                _assign(model, [0, 1], prototypes)
                pipeline.collide(model.state(), contacts)
                count = contacts.rigid_contact_count.numpy()[0]
                touching = contacts.rigid_contact_shape0.numpy()[:count].tolist()
                touching += contacts.rigid_contact_shape1.numpy()[:count].tolist()
                test.assertEqual(set(touching), shapes)


def test_mujoco_rest_height_follows_assignment(test, device):
    """A falling body settles on the assigned prototype's geometry and mass."""
    model = _scene(2, height=0.5).finalize(device=device)
    # MuJoCo snapshots one geom layout for all worlds at construction; assign afterwards.
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
    solver.notify_model_changed(_assign(model, [0, 1], [0, 1]))
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
add_function_test(
    TestShapePrototypes, "test_contacts_follow_assignment", test_contacts_follow_assignment, devices=devices
)
add_function_test(
    TestShapePrototypes,
    "test_mujoco_rest_height_follows_assignment",
    test_mujoco_rest_height_follows_assignment,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
