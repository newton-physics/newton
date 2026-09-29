# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native deformable recording without a dependency on selection views or USD."""

import unittest

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


class TestDeformableObjects(unittest.TestCase):
    """Preserve whole deformable identities and ranges through the builder lifecycle."""

    def test_empty_model_has_no_deformable_records(self):
        """Finalize an empty builder without creating placeholder deformable objects."""
        self.assertEqual(newton.ModelBuilder().finalize(device="cpu")._deformable_objects, ())

    def test_native_constructors_record_once(self):
        """Record every native constructor with explicit or generated labels on CPU and CUDA."""
        constructors = (
            ("curve", {"body": (0, 3), "joint": (0, 3)}, _add_curve, {}),
            ("curve", {"body": (0, 3), "joint": (0, 4)}, _add_curve, {"topology": "closed"}),
            ("curve", {"body": (0, 3), "joint": (0, 3)}, _add_curve, {"topology": "graph"}),
            ("surface", {"particle": (0, 4), "triangle": (0, 2), "edge": (0, 5)}, _add_surface, {}),
            ("surface", {"particle": (0, 4), "triangle": (0, 2), "edge": (0, 5)}, _add_surface, {"grid": True}),
            ("volume", {"particle": (0, 4), "tetrahedron": (0, 1)}, _add_volume, {}),
            ("volume", {"particle": (0, 8), "tetrahedron": (0, 5)}, _add_volume, {"grid": True}),
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
                        (record,) = builder.finalize(device=device)._deformable_objects
                        self.assertEqual(
                            (record.id, record.family, record.label, record.world), (0, family, expected_label, -1)
                        )
                        self.assertEqual(record.ranges, ranges)

    def test_builder_identity_edits_survive_finalization(self):
        """Retain edited labels without changing physics arrays or the source builder."""
        builder = newton.ModelBuilder()
        _add_curve(builder)
        _add_surface(builder)
        _add_volume(builder)
        before = builder.finalize(device="cpu")

        builder.curve_label[0] = "/Template/Cable"
        builder.surface_label = ["/Template/Cloth"]
        builder.volume_label[:] = ["/Template/Toy"]
        after = builder.finalize(device="cpu")
        self.assertEqual([r.label for r in before._deformable_objects], ["curve_0", "surface_0", "volume_0"])
        self.assertEqual(
            [r.label for r in after._deformable_objects], ["/Template/Cable", "/Template/Cloth", "/Template/Toy"]
        )
        self.assertEqual([r.ranges for r in before._deformable_objects], [r.ranges for r in after._deformable_objects])
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
        records = builder.finalize(device="cpu")._deformable_objects
        self.assertEqual([r.id for r in records], list(range(12)))
        self.assertEqual(len({(r.family, tuple(r.ranges.items())) for r in records}), 12)

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
                records = model._deformable_objects
                self.assertEqual([r.id for r in records], list(range(6)))
                self.assertEqual([r.world for r in records], [0, 1, 0, 1, 0, 1])
                self.assertEqual(
                    [r.ranges for r in records],
                    [
                        {"body": (0, 3), "joint": (0, 3)},
                        {"body": (3, 6), "joint": (3, 6)},
                        {"particle": (0, 4), "triangle": (0, 2), "edge": (0, 5)},
                        {"particle": (12, 16), "triangle": (14, 16), "edge": (23, 28)},
                        {"particle": (4, 12), "tetrahedron": (0, 5)},
                        {"particle": (16, 24), "tetrahedron": (5, 10)},
                    ],
                )
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
        self.assertEqual(
            [(r.label, r.world) for r in model._deformable_objects],
            [
                ("cable", 0),
                ("cable_0", 2),
                ("cable_1", 2),
                ("cloth", 0),
                ("global_toy", -1),
            ],
        )

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
                        (record,) = model._deformable_objects
                        self.assertEqual(
                            (record.label, record.ranges), (label or "curve_0", {"body": (0, 2), "joint": (0, 1)})
                        )
                    else:
                        self.assertEqual((model.body_count, model.joint_count), (1, 1))
                        self.assertEqual(builder.curve_label, [])
                        self.assertEqual(model._deformable_objects, ())

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
                (record,) = builder.finalize(device="cpu")._deformable_objects
                self.assertEqual((record.label, record.ranges), ("cable", {"body": (0, 2), "joint": (0, 2)}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
