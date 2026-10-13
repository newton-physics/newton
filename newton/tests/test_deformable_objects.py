# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native deformable recording without a dependency on selection views or USD."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices


def _add_curve(builder, label=None, topology="chain"):
    points = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.1, 0.1, 1.0)]
    if topology == "closed":
        points = [points[0], points[1], points[3], points[0]]
    edges = [(0, 1), (1, 2), (1, 3)] if topology == "graph" else None
    return builder.add_rod(
        rod=newton.Rod(points, edges=edges, closed=topology == "closed", radius=0.02),
        label=label,
        body_frame_origin="com",
    )


def _add_surface(builder, label=None, grid=False):
    common = {"pos": wp.vec3(0.0, 0.0, 2.0), "rot": wp.quat_identity(), "vel": wp.vec3(0.0), "label": label}
    if grid:
        builder.add_cloth_grid(**common, dim_x=1, dim_y=1, cell_x=1.0, cell_y=1.0, mass=1.0)
    else:
        builder.add_cloth_mesh(
            **common,
            scale=1.0,
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            indices=[0, 1, 2, 0, 2, 3],
            density=1.0,
        )


def _add_volume(builder, label=None, grid=False):
    common = {
        "pos": wp.vec3(0.0, 0.0, 3.0),
        "rot": wp.quat_identity(),
        "vel": wp.vec3(0.0),
        "density": 1.0,
        "k_mu": 1.0,
        "k_lambda": 1.0,
        "k_damp": 0.0,
        "label": label,
    }
    if grid:
        builder.add_soft_grid(**common, dim_x=1, dim_y=1, dim_z=1, cell_x=1.0, cell_y=1.0, cell_z=1.0)
    else:
        builder.add_soft_mesh(
            **common,
            scale=1.0,
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 3],
        )


def _ranges(builder, family, kind):
    """Pair the builder's start and end indices for one element kind."""
    return list(
        zip(getattr(builder, f"_{family}_{kind}_start"), getattr(builder, f"_{family}_{kind}_end"), strict=True)
    )


class TestDeformableObjects(unittest.TestCase):
    """Preserve whole deformable identities and ranges through the builder lifecycle."""

    def test_empty_builder_has_no_deformable_objects(self):
        """Keep deformable identity lists empty when no objects have been added."""
        builder = newton.ModelBuilder()
        for family in ("curve", "surface", "volume"):
            self.assertEqual(getattr(builder, f"{family}_label"), [])
            self.assertEqual(getattr(builder, f"{family}_world"), [])
        model = builder.finalize(device="cpu")
        self.assertEqual((model.body_count, model.particle_count), (0, 0))

    def test_native_constructors_record_once(self):
        """Record every native constructor with explicit or generated labels on CPU and CUDA."""
        constructors = (
            ("curve", {"body": (0, 3), "joint": (0, 3)}, _add_curve, {}),
            ("curve", {"body": (0, 3), "joint": (0, 4)}, _add_curve, {"topology": "closed"}),
            ("curve", {"body": (0, 3), "joint": (0, 3)}, _add_curve, {"topology": "graph"}),
            ("surface", {"particle": (0, 4), "tri": (0, 2), "edge": (0, 5)}, _add_surface, {}),
            ("surface", {"particle": (0, 4), "tri": (0, 2), "edge": (0, 5)}, _add_surface, {"grid": True}),
            ("volume", {"particle": (0, 4), "tet": (0, 1)}, _add_volume, {}),
            ("volume", {"particle": (0, 8), "tet": (0, 5)}, _add_volume, {"grid": True}),
        )
        for device in get_test_devices():
            for family, ranges, add, kwargs in constructors:
                for label in (None, "asset"):
                    with self.subTest(device=device, family=family, constructor=kwargs, label=label):
                        builder = newton.ModelBuilder()
                        add(builder, label=label, **kwargs)
                        expected_label = label or f"{family}_0"
                        self.assertEqual(getattr(builder, f"{family}_label"), [expected_label])
                        self.assertEqual(getattr(builder, f"{family}_world"), [-1])
                        model = builder.finalize(device=device)
                        for kind, bounds in ranges.items():
                            self.assertEqual(_ranges(builder, family, kind), [bounds])
                            self.assertEqual(getattr(model, f"{kind}_count"), bounds[1])

    def test_builder_label_edits_preserve_simulation_data(self):
        """Retain edited builder labels without changing ranges or finalized physics arrays."""
        builder = newton.ModelBuilder()
        _add_curve(builder)
        _add_surface(builder)
        _add_volume(builder)
        before = builder.finalize(device="cpu")
        kinds = (
            ("curve", "body"),
            ("curve", "joint"),
            ("surface", "particle"),
            ("surface", "tri"),
            ("surface", "edge"),
            ("volume", "particle"),
            ("volume", "tet"),
        )
        ranges_before = [_ranges(builder, family, kind) for family, kind in kinds]

        builder.curve_label[0] = "/Template/Cable"
        builder.surface_label = ["/Template/Cloth"]
        builder.volume_label[:] = ["/Template/Toy"]
        after = builder.finalize(device="cpu")
        self.assertEqual(
            (builder.curve_label, builder.surface_label, builder.volume_label),
            (["/Template/Cable"], ["/Template/Cloth"], ["/Template/Toy"]),
        )
        self.assertEqual([_ranges(builder, family, kind) for family, kind in kinds], ranges_before)
        for name in ("body_q", "body_mass", "particle_q", "particle_mass", "joint_type", "tri_indices", "tet_indices"):
            np.testing.assert_array_equal(getattr(before, name).numpy(), getattr(after, name).numpy())

    def test_generated_labels_and_repeated_labels(self):
        """Generate one name per constructor call while allowing explicit labels to repeat."""
        builder = newton.ModelBuilder()
        for add in (_add_curve, _add_surface, _add_volume):
            add(builder)
            add(builder)
            add(builder, label="same")
            add(builder, label="same")
        for family in ("curve", "surface", "volume"):
            self.assertEqual(getattr(builder, f"{family}_label"), [f"{family}_0", f"{family}_1", "same", "same"])
        for family, kind, count in (("curve", "body", 3), ("surface", "particle", 4), ("volume", "particle", 4)):
            offset = 16 if family == "volume" else 0
            self.assertEqual(
                _ranges(builder, family, kind), [(offset + i * count, offset + (i + 1) * count) for i in range(4)]
            )
        model = builder.finalize(device="cpu")
        self.assertEqual((model.body_count, model.particle_count), (12, 32))

    def test_composition_and_replication_offset_every_family(self):
        """Preserve rebased identities and disjoint ranges through both cloning paths."""
        prototype = newton.ModelBuilder()
        _add_curve(prototype, topology="graph")
        _add_surface(prototype)
        _add_volume(prototype, grid=True)
        prototype.curve_label[0] = "cable"
        prototype.surface_label[0] = "cloth"
        prototype.volume_label[0] = "toy"
        prefixes = ["env_0", "env_1"]

        for replicate in (False, True):
            with self.subTest(replicate=replicate):
                scene = newton.ModelBuilder()
                if replicate:
                    scene.replicate(prototype, 2, label_prefixes=prefixes)
                else:
                    for prefix in prefixes:
                        scene.add_world(prototype, label_prefix=prefix)
                model = scene.finalize(device="cpu")
                for family, label in (("curve", "cable"), ("surface", "cloth"), ("volume", "toy")):
                    self.assertEqual(getattr(scene, f"{family}_label"), [f"{prefix}/{label}" for prefix in prefixes])
                    self.assertEqual(getattr(scene, f"{family}_world"), [0, 1])
                for family, kind, expected in (
                    ("curve", "body", [(0, 3), (3, 6)]),
                    ("curve", "joint", [(0, 3), (3, 6)]),
                    ("surface", "particle", [(0, 4), (12, 16)]),
                    ("surface", "tri", [(0, 2), (14, 16)]),
                    ("surface", "edge", [(0, 5), (23, 28)]),
                    ("volume", "particle", [(4, 12), (16, 24)]),
                    ("volume", "tet", [(0, 5), (5, 10)]),
                ):
                    self.assertEqual(_ranges(scene, family, kind), expected)
                self.assertEqual((model.body_count, model.particle_count, model.world_count), (6, 24, 2))
        self.assertEqual((prototype.curve_label, prototype.curve_world), (["cable"], [-1]))

    def test_heterogeneous_worlds_keep_global_and_empty_worlds(self):
        """Retain identities across globals, empty worlds, and worlds with several deformables."""
        builder = newton.ModelBuilder()
        _add_volume(builder, label="global_toy")
        builder.begin_world()
        _add_curve(builder, label="cable")
        _add_surface(builder, label="cloth")
        builder.end_world()
        builder.begin_world()
        rigid = builder.add_body()  # This world has no deformable objects.
        builder.add_shape_sphere(rigid, radius=0.1)
        builder.end_world()
        builder.begin_world()
        _add_curve(builder, label="cable_0")
        _add_curve(builder, label="cable_1")
        builder.end_world()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.world_count, 3)
        self.assertEqual((builder.curve_label, builder.curve_world), (["cable", "cable_0", "cable_1"], [0, 2, 2]))
        self.assertEqual((builder.surface_label, builder.surface_world), (["cloth"], [0]))
        self.assertEqual((builder.volume_label, builder.volume_world), (["global_toy"], [-1]))

    def test_fixed_joint_collapse_drops_or_preserves_complete_curves(self):
        """Keep collapse label-neutral and retain an anchored curve only when its joint is kept."""
        for label in (None, "anchored"):
            for keep in (False, True):
                with self.subTest(label=label, keep=keep):
                    builder = newton.ModelBuilder()
                    bodies, joints = builder.add_rod(
                        rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], radius=0.02),
                        label=label,
                        wrap_in_articulation=False,
                        body_frame_origin="com",
                    )
                    anchor = builder.add_joint_fixed(-1, bodies[0], label="anchor")
                    builder.add_articulation([*joints, anchor])
                    if keep:
                        builder.collapse_fixed_joints(joints_to_keep=["anchor"])
                    else:
                        with self.assertWarnsRegex(UserWarning, "joints_to_keep"):
                            builder.collapse_fixed_joints()
                    model = builder.finalize(device="cpu")
                    if keep:
                        self.assertEqual((model.body_count, model.joint_count), (2, 2))
                        self.assertEqual(builder.curve_label, [label or "curve_0"])
                        self.assertEqual(_ranges(builder, "curve", "body"), [(0, 2)])
                        self.assertEqual(_ranges(builder, "curve", "joint"), [(0, 1)])
                    else:
                        self.assertEqual((model.body_count, model.joint_count), (1, 1))
                        self.assertEqual(builder.curve_label, [])
                        self.assertEqual(_ranges(builder, "curve", "body"), [])
                        self.assertEqual(_ranges(builder, "curve", "joint"), [])

    def test_curve_removal_warning_leaves_collapse_complete(self):
        """Finish collapse even when a removed-curve warning is treated as an exception."""
        for warnings_as_errors in (False, True):
            with self.subTest(warnings_as_errors=warnings_as_errors):
                builder = newton.ModelBuilder()
                for label in ("anchored_0", "anchored_1"):
                    bodies, joints = builder.add_rod(
                        rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], radius=0.02),
                        label=label,
                        wrap_in_articulation=False,
                        body_frame_origin="com",
                    )
                    anchor = builder.add_joint_fixed(-1, bodies[0], label=f"{label}_anchor")
                    builder.add_articulation([*joints, anchor])
                _add_curve(builder, label="free_cable")

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("error" if warnings_as_errors else "always", UserWarning)
                    if warnings_as_errors:
                        with self.assertRaisesRegex(UserWarning, "Deformable curve 'anchored_0' is unavailable"):
                            builder.collapse_fixed_joints()
                    else:
                        builder.collapse_fixed_joints()
                        self.assertEqual(len(caught), 2)
                        for warning, label in zip(caught, ("anchored_0", "anchored_1"), strict=True):
                            self.assertIn(label, str(warning.message))

                self.assertEqual((builder.body_count, builder.joint_count), (5, 5))
                self.assertEqual(builder.joint_parent, [-1, -1, -1, 2, 3])
                self.assertEqual(builder.joint_child, [0, 1, 2, 3, 4])
                self.assertEqual((builder.curve_label, builder.curve_world), (["free_cable"], [-1]))
                model = builder.finalize(device="cpu")
                self.assertEqual((model.body_count, model.joint_count, model.articulation_count), (5, 5, 3))
                np.testing.assert_array_equal(model.joint_child.numpy(), [0, 1, 2, 3, 4])

    def test_empty_curve_joint_ranges_follow_retained_joints(self):
        """Remap empty curve ranges before, between, and after retained joints in each world."""
        cases = (
            ((1, 3), (0, 0, 1, 1, 2, 2)),
            ((), (0, 0, 0, 0, 0, 0)),
            ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5)),
        )
        for retained, boundaries in cases:
            with self.subTest(retained=retained):
                prototype = newton.ModelBuilder()
                for i in range(6):
                    prototype.add_rod(
                        rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0)], radius=0.02),
                        label=f"segment_{i}",
                        wrap_in_articulation=False,
                        body_frame_origin="com",
                    )
                    if i < 5:
                        body = prototype.add_link()
                        prototype.add_shape_sphere(body, radius=0.02)
                        if i in retained:
                            joint = prototype.add_joint_free(body)
                        else:
                            joint = prototype.add_joint_fixed(-1, body)
                        prototype.add_articulation([joint])

                scene = newton.ModelBuilder()
                scene.replicate(prototype, 2, label_prefixes=["env_0", "env_1"])
                scene.collapse_fixed_joints()
                model = scene.finalize(device="cpu")
                self.assertEqual(model.body_count, 2 * (6 + len(retained)))
                self.assertEqual(model.joint_count, 2 * len(retained))
                # Each segment was inserted just before joint i, or after the final joint.
                self.assertEqual(
                    list(zip(scene.curve_label, scene.curve_world, _ranges(scene, "curve", "joint"), strict=True)),
                    [
                        (f"env_{world}/segment_{i}", world, (world * len(retained) + boundary,) * 2)
                        for world in range(2)
                        for i, boundary in enumerate(boundaries)
                    ],
                )

    def test_deprecated_curve_inputs_keep_recording(self):
        """Preserve recording through the supported deprecation period for old rod inputs."""
        points = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)]
        for graph in (False, True):
            with self.subTest(graph=graph):
                builder = newton.ModelBuilder()
                with self.assertWarns(DeprecationWarning):
                    if graph:
                        builder.add_rod_graph(node_positions=points, edges=[(0, 1), (1, 2)], radius=0.02, label="cable")
                    else:
                        builder.add_rod(positions=points, radius=0.02, label="cable")
                self.assertEqual(builder.curve_label, ["cable"])
                self.assertEqual(_ranges(builder, "curve", "body"), [(0, 2)])
                self.assertEqual(_ranges(builder, "curve", "joint"), [(0, 2)])
                model = builder.finalize(device="cpu")
                self.assertEqual((model.body_count, model.joint_count), (2, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
