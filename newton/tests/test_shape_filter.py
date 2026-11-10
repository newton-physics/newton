# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import re
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_cuda_test_devices, get_test_devices

ShapeConfig = newton.ModelBuilder.ShapeConfig
VISIBLE = newton.ShapeFlags.VISIBLE
COLLIDE_SHAPES = newton.ShapeFlags.COLLIDE_SHAPES

# Shapes of one world, stacked along z below a downward ray: (label, top height [m], visible, colliding, groups).
_STACK = (
    ("robot/visual", 1.6, True, False, ("robot",)),
    ("robot/collider", 1.4, False, True, ("robot",)),
    ("prop", 1.0, True, True, ("props",)),
    ("marker", 0.8, True, False, ("markers", "props")),
)


def _stack_world() -> newton.ModelBuilder:
    """One world: a body holding boxes stacked along z whose tops are at the heights of ``_STACK``."""
    builder = newton.ModelBuilder()
    body = builder.add_body(label="robot")
    for label, top, visible, colliding, groups in _STACK:
        cfg = ShapeConfig(
            is_visible=visible, has_shape_collision=colliding, has_particle_collision=colliding, groups=groups
        )
        builder.add_shape_box(
            body, xform=wp.transform((0.0, 0.0, top - 0.05), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05,
            cfg=cfg, label=label,
        )  # fmt: skip
    return builder


def _stack_scene(world_count: int = 3) -> newton.ModelBuilder:
    """``world_count`` stacks above a global ground plane in group ``terrain``."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=ShapeConfig(groups=("terrain",)), label="ground")
    builder.replicate(_stack_world(), world_count)
    return builder


def _cast_down(model: newton.Model, shape_filter: int | None, device) -> tuple[np.ndarray, np.ndarray]:
    """Cast one ray per world straight down from z = 3 and return hit distances and shape ids."""
    count = model.world_count
    dist = wp.empty(count, dtype=float, device=device)
    shape_id = wp.empty(count, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=wp.full(count, wp.vec3(0.0, 0.0, 3.0), dtype=wp.vec3, device=device),
        ray_directions=wp.full(count, wp.vec3(0.0, 0.0, -1.0), dtype=wp.vec3, device=device),
        ray_worlds=wp.array(np.arange(count), dtype=wp.int32, device=device),
        shape_filter=shape_filter,
        out_dist=dist,
        out_shape_id=shape_id,
    )
    return dist.numpy(), shape_id.numpy()


class TestShapeGroups(unittest.TestCase):
    def test_groups_from_config_patterns_and_merges(self):
        sub = newton.ModelBuilder()
        sub.add_shape_group("tools")
        body = sub.add_body()
        sub.add_shape_sphere(body, radius=0.1, label="gripper_left", cfg=ShapeConfig(groups=("robot",)))
        sub.add_shape_sphere(body, radius=0.1, label="gripper_right")
        sub.add_shape_sphere(body, radius=0.1, label="base")
        sub.add_shape_group("gripper", shapes=re.compile(r"gripper_.*"))
        np.testing.assert_array_equal(sub.shape_groups, [0b110, 0b100, 0])
        self.assertEqual(sub.shape_group_names, ["tools", "robot", "gripper"])

        # The scene registers groups in another order; merged shapes keep their groups by name.
        scene = newton.ModelBuilder()
        scene.add_ground_plane(cfg=ShapeConfig(groups=("gripper", "terrain")))
        scene.replicate(sub, 2)
        self.assertEqual(scene.shape_group_names, ["gripper", "terrain", "tools", "robot"])
        np.testing.assert_array_equal(scene.shape_groups, [0b11] + [0b1001, 0b1, 0] * 2)

        model = scene.finalize(device="cpu")
        self.assertEqual(model.shape_group_names, ["gripper", "terrain", "tools", "robot"])
        np.testing.assert_array_equal(model.shape_groups.numpy(), scene.shape_groups)
        with self.assertWarns(newton.exceptions.NewtonDeprecationWarning):
            self.assertIs(model.shape_filter, model.shape_groups)
        np.testing.assert_array_equal(newton.ShapeFilter(groups=["gripper"]).select(model), [0, 1, 2, 4, 5])

    def test_groups_follow_mesh_approximation(self):
        builder = newton.ModelBuilder()
        shape = builder.add_shape_mesh(
            builder.add_body(), mesh=newton.Mesh.create_box(0.2), cfg=ShapeConfig(groups=("robot",))
        )
        builder.approximate_meshes(method="convex_hull", shape_indices=[shape], keep_visual_shapes=True)
        # The collider is replaced in place and its visual copy is a new shape; both keep the group.
        self.assertEqual(builder.shape_count, 2)
        np.testing.assert_array_equal(builder.shape_groups, [1, 1])

    def test_groups_from_mjcf_geom_groups(self):
        mjcf = """
        <mujoco>
          <default>
            <default class="collision"><geom group="3"/></default>
          </default>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="robot">
              <freejoint/>
              <geom name="visual" type="sphere" size="0.1" group="1" contype="0" conaffinity="0"/>
              <geom name="collider" class="collision" type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, geom_groups_as_shape_groups=True)
        model = builder.finalize(device="cpu")
        for group, geom in (("mujoco_group_0", "floor"), ("mujoco_group_1", "visual"), ("mujoco_group_3", "collider")):
            selected = newton.ShapeFilter(groups=[group]).select(model)
            self.assertEqual([model.shape_label[shape].rpartition("/")[2] for shape in selected], [geom])

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        self.assertEqual(builder.shape_group_names, [])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_groups_from_usd(self):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        def author_groups(prim, groups):
            prim.CreateAttribute("newton:shapeGroups", Sdf.ValueTypeNames.TokenArray).Set(Vt.TokenArray(groups))

        robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(robot)
        author_groups(robot, ["robot"])
        collider = UsdGeom.Cube.Define(stage, "/World/Robot/collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        author_groups(collider, ["colliders", "robot"])
        UsdGeom.Sphere.Define(stage, "/World/Robot/visual")
        ground = UsdGeom.Cube.Define(stage, "/World/ground").GetPrim()
        UsdPhysics.CollisionAPI.Apply(ground)
        author_groups(ground, ["terrain"])

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        # Shapes belong to the groups authored on their prims and on all ancestors.
        self.assertEqual(builder.shape_group_names, ["robot", "colliders", "terrain"])
        groups = dict(zip(builder.shape_label, builder.shape_groups, strict=True))
        self.assertEqual(groups, {"/World/Robot/visual": 0b1, "/World/Robot/collider": 0b11, "/World/ground": 0b100})

    def test_group_limits_and_errors(self):
        builder = newton.ModelBuilder()
        for i in range(32):
            builder.add_shape_group(f"group_{i}")
        with self.assertRaisesRegex(ValueError, "at most 32 shape groups"):
            builder.add_shape_group("one_too_many")
        with self.assertRaisesRegex(TypeError, "sequence of shape group names"):
            builder.add_shape_sphere(-1, radius=0.1, cfg=ShapeConfig(groups="group_0"))
        with self.assertRaisesRegex(TypeError, "sequence of group names"):
            newton.ShapeFilter(groups="group_0")
        with self.assertRaisesRegex(ValueError, "Unknown shape group 'missing'"):
            newton.ShapeFilter(exclude_groups=["missing"]).select(builder.finalize(device="cpu"))

    def test_filter_selection_and_registration(self):
        builder = _stack_scene(world_count=2)
        terrain = builder.add_shape_filter(newton.ShapeFilter(groups=["terrain"]))
        self.assertEqual(terrain, 1)
        self.assertEqual(builder.add_shape_filter(newton.ShapeFilter(groups=("terrain",))), terrain)
        self.assertEqual(builder.add_shape_filter(newton.ShapeFilter(flags=COLLIDE_SHAPES)), 2)
        model = builder.finalize(device="cpu")

        # Shape 0 is the ground; each world holds the stack in order.
        cases = {
            newton.ShapeFilter(): range(9),
            newton.ShapeFilter(flags=VISIBLE): [0, 1, 3, 4, 5, 7, 8],
            newton.ShapeFilter(flags=COLLIDE_SHAPES, exclude_shapes="robot/*"): [0, 3, 7],
            newton.ShapeFilter(groups=["props"], exclude_groups=["markers"]): [3, 7],
            newton.ShapeFilter(flags=VISIBLE, shapes=["robot/*", "marker"]): [1, 4, 5, 8],
            newton.ShapeFilter(shapes=[0, 2]): [0, 2],
        }
        for shape_filter, expected in cases.items():
            with self.subTest(shape_filter=shape_filter):
                np.testing.assert_array_equal(shape_filter.select(model), list(expected))

        with self.assertRaisesRegex(ValueError, "Unknown shape filter id 3"):
            newton.intersect_ray(
                model,
                ray_origins=wp.zeros(1, dtype=wp.vec3, device="cpu"),
                ray_directions=wp.zeros(1, dtype=wp.vec3, device="cpu"),
                ray_worlds=wp.zeros(1, dtype=wp.int32, device="cpu"),
                shape_filter=3,
            )
        with self.assertRaisesRegex(ValueError, "registered shape filters"):
            newton.ModelBuilder().add_builder(builder)


def test_intersect_ray_shape_filters(test: TestShapeGroups, device):
    builder = _stack_scene()
    filters = [
        newton.ShapeFilter(groups=["terrain"]),
        newton.ShapeFilter(flags=COLLIDE_SHAPES),
        newton.ShapeFilter(flags=COLLIDE_SHAPES, exclude_shapes="robot/*"),
        newton.ShapeFilter(flags=VISIBLE, exclude_groups=["robot"]),
        newton.ShapeFilter(groups=["markers"]),
        newton.ShapeFilter(groups=["props"], exclude_groups=["markers"]),
        newton.ShapeFilter(shapes=["robot/visual"]),
    ]
    ids = [builder.add_shape_filter(shape_filter) for shape_filter in filters]
    model = builder.finalize(device=device)

    # Shapes selected by the same filters share a class; the BVH holds every shape some filter selects.
    test.assertGreater(model._bvh_shape_class_count, 1)
    test.assertEqual(model.bvh_shapes_group_roots.shape[0], (model.world_count + 1) * model._bvh_shape_class_count)
    # Filters selecting at most one shape per world, here the global ground plus one stack shape, skip the BVH.
    direct = [model._bvh_shape_filter(shape_filter).world_leaves is not None for shape_filter in ids]
    test.assertEqual(direct, [True, False, True, False, True, True, True])

    tops = {"ground": 0.0, **{label: top for label, top, *_ in _STACK}}
    default_filter = newton.ShapeFilter(flags=VISIBLE)
    for shape_filter, shape_filter_id in [(default_filter, None), *zip(filters, ids, strict=True)]:
        with test.subTest(shape_filter=shape_filter):
            selected = shape_filter.select(model)
            worlds = model.shape_world.numpy()[selected]
            expected = []
            for world in range(model.world_count):
                in_reach = selected[(worlds == world) | (worlds == -1)]
                expected.append(max(in_reach, key=lambda shape: tops[model.shape_label[shape]], default=-1))
            dist, shape_id = _cast_down(model, shape_filter_id, device)
            np.testing.assert_array_equal(shape_id, expected)
            expected_dist = [3.0 - tops[model.shape_label[shape]] if shape >= 0 else -1.0 for shape in expected]
            np.testing.assert_allclose(dist, expected_dist, atol=1.0e-5)


def test_intersect_ray_default_filter_keeps_world_groups(test: TestShapeGroups, device):
    model = _stack_scene().finalize(device=device)
    # Without registered filters every shape is in one class, so the group roots stay per world.
    test.assertEqual(model._bvh_shape_class_count, 1)
    test.assertEqual(model.bvh_shapes_group_roots.shape[0], model.world_count + 1)
    _dist, shape_id = _cast_down(model, None, device)
    np.testing.assert_array_equal(shape_id, [1, 5, 9])


def test_intersect_ray_shape_filter_graph_capture(test: TestShapeGroups, device):
    builder = _stack_scene(world_count=2)
    props = builder.add_shape_filter(newton.ShapeFilter(groups=["props"], exclude_groups=["markers"]))
    model = builder.finalize(device=device)
    state = model.state()
    count = model.world_count
    origins = wp.full(count, wp.vec3(0.0, 0.0, 3.0), dtype=wp.vec3, device=device)
    directions = wp.full(count, wp.vec3(0.0, 0.0, -1.0), dtype=wp.vec3, device=device)
    worlds = wp.array(np.arange(count), dtype=wp.int32, device=device)
    dist = wp.empty(count, dtype=float, device=device)

    def query():
        model.bvh_refit_shapes(state)
        newton.intersect_ray(
            model,
            ray_origins=origins,
            ray_directions=directions,
            ray_worlds=worlds,
            shape_filter=props,
            out_dist=dist,
        )

    query()  # compile outside the capture
    with wp.ScopedCapture(device) as capture:
        query()
    # Lift world 1's body by 0.5 m; the replayed graph refits the BVH and hits the lifted prop.
    body_q = state.body_q.numpy()
    body_q[1, 2] += 0.5
    state.body_q.assign(body_q)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dist.numpy(), [2.0, 1.5], atol=1.0e-5)


def test_intersect_ray_shape_filter_skips_empty_variant_rows(test: TestShapeGroups, device):
    def variant(with_marker: bool) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
        if with_marker:
            builder.add_shape_sphere(
                body, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), radius=0.1,
                cfg=ShapeConfig(has_shape_collision=False, has_particle_collision=False, groups=("markers",)),
            )  # fmt: skip
        return builder

    env = newton.ModelBuilder()
    env.add_body_variants(env.add_body(), [variant(True), variant(False)])
    builder = newton.ModelBuilder()
    builder.replicate(env, 2)
    markers = builder.add_shape_filter(newton.ShapeFilter(groups=["markers"]))
    model = builder.finalize(device=device)

    model.select_body_variants(wp.array([0, 1], dtype=wp.int32, device=device))
    model.bvh_refit_shapes(model.state())
    # World 1 selected the variant without a marker, so its marker row is empty and the filter skips it.
    _dist, shape_id = _cast_down(model, markers, device)
    np.testing.assert_array_equal(shape_id, [1, -1])


devices = get_test_devices()
add_function_test(
    TestShapeGroups, "test_intersect_ray_shape_filters", test_intersect_ray_shape_filters, devices=devices
)
add_function_test(
    TestShapeGroups,
    "test_intersect_ray_default_filter_keeps_world_groups",
    test_intersect_ray_default_filter_keeps_world_groups,
    devices=devices,
)
add_function_test(
    TestShapeGroups,
    "test_intersect_ray_shape_filter_graph_capture",
    test_intersect_ray_shape_filter_graph_capture,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestShapeGroups,
    "test_intersect_ray_shape_filter_skips_empty_variant_rows",
    test_intersect_ray_shape_filter_skips_empty_variant_rows,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
