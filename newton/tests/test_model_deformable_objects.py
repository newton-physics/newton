# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect finalized deformable objects through public Model attributes."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSemiImplicit
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests.test_deformable_objects import _add_curve, _add_surface, _add_volume
from newton.tests.unittest_utils import get_test_devices


class TestModelDeformableObjects(unittest.TestCase):
    def test_existing_custom_curve_frequency_keeps_its_references(self):
        """Preserve an existing custom frequency named curve when adding built-in curve metadata."""
        prototype = newton.ModelBuilder()
        _add_curve(prototype, label="cable")
        prototype.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="curve"))
        prototype.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="source_row", dtype=wp.int32, frequency="curve", references="curve"
            )
        )
        prototype.add_custom_values(**{"source_row": 0})
        prototype.add_custom_values(**{"source_row": 1})
        prototype.collapse_fixed_joints()
        builder = newton.ModelBuilder()
        builder.replicate(prototype, 2)
        model = builder.finalize(device="cpu")
        self.assertEqual(model.curve_count, 2)
        self.assertEqual(model.custom_frequency_counts["curve"], 4)
        np.testing.assert_array_equal(model.source_row.numpy(), [0, 1, 2, 3])
        self.assertEqual(model.attribute_specs["source_row"].references, "curve")

    def test_family_attribute_frequencies_survive_replication(self):
        """Size model and state attributes by deformable objects rather than their elements."""
        prototype = newton.ModelBuilder()
        for family, add in (("curve", _add_curve), ("surface", _add_surface), ("volume", _add_volume)):
            add(prototype, label="asset")
            frequency = getattr(newton.Model.AttributeFrequency, family.upper())
            prototype.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"{family}_tag", dtype=wp.int32, frequency=frequency, values={0: 17}
                )
            )
            prototype.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"{family}_enabled",
                    dtype=wp.int32,
                    frequency=frequency,
                    assignment=newton.Model.AttributeAssignment.STATE,
                    default=1,
                )
            )
        builder = newton.ModelBuilder()
        builder.replicate(prototype, 2)
        model = builder.finalize(device="cpu")
        state = model.state()
        for family in ("curve", "surface", "volume"):
            np.testing.assert_array_equal(getattr(model, f"{family}_tag").numpy(), [17, 17])
            np.testing.assert_array_equal(getattr(state, f"{family}_enabled").numpy(), [1, 1])

    def test_curve_attributes_follow_fixed_joint_collapse(self):
        """Keep per-curve values and references aligned when an incomplete curve is dropped."""
        builder = newton.ModelBuilder()
        bodies, joints = builder.add_rod(
            rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], radius=0.02),
            label="anchored",
            wrap_in_articulation=False,
            body_frame_origin="com",
        )
        anchor = builder.add_joint_fixed(-1, bodies[0], label="anchor")
        builder.add_articulation([*joints, anchor])
        _add_curve(builder, label="retained")
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="asset_tag", dtype=wp.int32, frequency=newton.Model.AttributeFrequency.CURVE, values={0: 10, 1: 20}
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="selected_curve",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.ONCE,
                references="curve",
                values={0: 1},
            )
        )
        with self.assertWarnsRegex(UserWarning, "joints_to_keep"):
            builder.collapse_fixed_joints()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.curve_label, ["retained"])
        np.testing.assert_array_equal(model.curve_body_start.numpy(), [1])
        np.testing.assert_array_equal(model.curve_body_end.numpy(), [4])
        np.testing.assert_array_equal(model.asset_tag.numpy(), [20])
        np.testing.assert_array_equal(model.selected_curve.numpy(), [0])

    def test_empty_model_has_empty_public_deformable_arrays(self):
        """Inspect an empty model without placeholder objects or missing arrays."""
        for device in get_test_devices():
            model = newton.ModelBuilder().finalize(device=device)
            for family, kinds in (
                ("curve", ("body", "joint")),
                ("surface", ("particle", "tri", "edge")),
                ("volume", ("particle", "tet")),
            ):
                with self.subTest(device=device, family=family):
                    self.assertEqual(getattr(model, f"{family}_count"), 0)
                    self.assertEqual(getattr(model, f"{family}_label"), [])
                    names = [f"{family}_world"]
                    names.extend(f"{family}_{kind}_{end}" for kind in kinds for end in ("start", "end"))
                    for name in names:
                        array = getattr(model, name)
                        self.assertEqual((array.shape, array.dtype, array.device), ((0,), wp.int32, model.device))

    def test_model_identities_are_independent_builder_snapshots(self):
        """Copy edited builder labels and keep previously finalized models unchanged."""
        builder = newton.ModelBuilder()
        for add in (_add_curve, _add_surface, _add_volume):
            add(builder, label="original")
        before = builder.finalize(device="cpu")
        for family in ("curve", "surface", "volume"):
            getattr(builder, f"{family}_label")[0] = "edited"
        after = builder.finalize(device="cpu")
        for family, add in (("curve", _add_curve), ("surface", _add_surface), ("volume", _add_volume)):
            self.assertEqual(getattr(before, f"{family}_label"), ["original"])
            self.assertEqual(getattr(after, f"{family}_label"), ["edited"])
            add(builder, label="later")
            self.assertEqual(getattr(after, f"{family}_count"), 1)
            np.testing.assert_array_equal(getattr(after, f"{family}_world").numpy(), [-1])
        for name in ("body_q", "body_mass", "particle_q", "particle_mass", "joint_type", "tri_indices", "tet_indices"):
            np.testing.assert_array_equal(getattr(before, name).numpy(), getattr(after, name).numpy())

    def test_composition_and_replication_preserve_public_ranges(self):
        """Retain all families and offset each range when cloning a mixed prototype."""
        prototype = newton.ModelBuilder()
        _add_curve(prototype, label="cable", topology="graph")
        _add_surface(prototype, label="cloth")
        _add_volume(prototype, label="toy", grid=True)
        for replicate in (False, True):
            with self.subTest(replicate=replicate):
                builder = newton.ModelBuilder()
                if replicate:
                    builder.replicate(prototype, 2, label_prefixes=["a", "b"])
                else:
                    builder.add_world(prototype, label_prefix="a")
                    builder.add_world(prototype, label_prefix="b")
                model = builder.finalize(device="cpu")
                for family, label in (("curve", "cable"), ("surface", "cloth"), ("volume", "toy")):
                    self.assertEqual(getattr(model, f"{family}_label"), [f"a/{label}", f"b/{label}"])
                    np.testing.assert_array_equal(getattr(model, f"{family}_world").numpy(), [0, 1])
                for name, starts, ends in (
                    ("curve_body", [0, 3], [3, 6]),
                    ("curve_joint", [0, 3], [3, 6]),
                    ("surface_particle", [0, 12], [4, 16]),
                    ("surface_tri", [0, 14], [2, 16]),
                    ("surface_edge", [0, 23], [5, 28]),
                    ("volume_particle", [4, 16], [12, 24]),
                    ("volume_tet", [0, 5], [5, 10]),
                ):
                    np.testing.assert_array_equal(getattr(model, f"{name}_start").numpy(), starts)
                    np.testing.assert_array_equal(getattr(model, f"{name}_end").numpy(), ends)

    def test_inventory_keeps_global_and_uneven_world_objects(self):
        """Inspect global, empty, and uneven worlds without a matching view."""
        builder = newton.ModelBuilder()
        _add_volume(builder, label="global_toy")
        builder.begin_world()
        _add_curve(builder, label="cable")
        _add_surface(builder, label="cloth")
        builder.end_world()
        builder.begin_world()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.end_world()
        builder.begin_world()
        _add_curve(builder, label="cable_0")
        _add_curve(builder, label="cable_1")
        builder.end_world()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.world_count, 3)
        self.assertEqual(model.curve_label, ["cable", "cable_0", "cable_1"])
        np.testing.assert_array_equal(model.curve_world.numpy(), [0, 2, 2])
        self.assertEqual(model.surface_label, ["cloth"])
        np.testing.assert_array_equal(model.surface_world.numpy(), [0])
        self.assertEqual(model.volume_label, ["global_toy"])
        np.testing.assert_array_equal(model.volume_world.numpy(), [-1])

    def test_coupled_empty_ranges_and_partial_curves(self):
        """Keep empty joint ranges but omit a curve whose bodies are only partly selected."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        segment, _ = builder.add_rod(
            rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0)], radius=0.02),
            label="segment",
            wrap_in_articulation=False,
            body_frame_origin="com",
        )
        curve, _ = _add_curve(builder, label="partial")
        model = builder.finalize(device="cpu")
        coupled = SolverCoupled(
            model,
            entries=[SolverCoupled.Entry(name="selected", solver=SolverSemiImplicit, bodies=[*segment, curve[0]])],
        )
        view = coupled.view("selected")
        self.assertEqual(view.curve_label, ["segment"])
        np.testing.assert_array_equal(view.curve_body_start.numpy(), [0])
        np.testing.assert_array_equal(view.curve_body_end.numpy(), [1])
        np.testing.assert_array_equal(view.curve_joint_start.numpy(), [0])
        np.testing.assert_array_equal(view.curve_joint_end.numpy(), [0])
        np.testing.assert_array_equal(model.curve_joint_start.numpy(), [1, 1])

    def test_coupled_model_rebases_complete_deformable_objects(self):
        """Expose only complete curves and remap their ranges in a compact solver model."""
        builder = newton.ModelBuilder()
        _add_curve(builder, label="hidden")
        bodies, _ = _add_curve(builder, label="selected")
        _add_surface(builder, label="cloth")
        _add_volume(builder, label="toy")
        model = builder.finalize(device="cpu")
        # Include the articulation root, not just the rod's returned connecting joints.
        joints = list(range(int(model.curve_joint_start.numpy()[1]), int(model.curve_joint_end.numpy()[1])))
        coupled = SolverCoupled(
            model,
            entries=[
                SolverCoupled.Entry(name="cable", solver=SolverSemiImplicit, bodies=bodies, joints=joints),
                SolverCoupled.Entry(
                    name="particles", solver=SolverSemiImplicit, particles=list(range(model.particle_count))
                ),
            ],
        )

        cable = coupled.view("cable")
        self.assertEqual(cable.curve_count, 1)
        self.assertEqual(cable.curve_label, ["selected"])
        np.testing.assert_array_equal(cable.curve_body_start.numpy(), [0])
        np.testing.assert_array_equal(cable.curve_body_end.numpy(), [3])
        np.testing.assert_array_equal(cable.curve_joint_start.numpy(), [0])
        np.testing.assert_array_equal(cable.curve_joint_end.numpy(), [3])
        self.assertEqual((cable.surface_count, cable.volume_count), (0, 0))
        self.assertEqual(cable.surface_label, [])
        self.assertEqual(cable.volume_particle_start.shape, (0,))

        particles = coupled.view("particles")
        self.assertEqual(particles.curve_count, 0)
        self.assertEqual(particles.curve_label, [])
        self.assertEqual(particles.surface_label, ["cloth"])
        self.assertEqual(particles.volume_label, ["toy"])
        np.testing.assert_array_equal(particles.surface_particle_start.numpy(), [0])
        np.testing.assert_array_equal(particles.volume_particle_start.numpy(), [4])
        self.assertEqual(model.curve_label, ["hidden", "selected"])
        np.testing.assert_array_equal(model.curve_body_start.numpy(), [0, 3])

    def test_mixed_model_exposes_deformable_ranges(self):
        """Inspect each family without including unrelated rigid bodies or particles."""
        for device in get_test_devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
                _add_curve(builder, label="cable")
                _add_surface(builder, label="cloth")
                _add_volume(builder, label="toy")
                model = builder.finalize(device=device)

                for family, label in (("curve", "cable"), ("surface", "cloth"), ("volume", "toy")):
                    self.assertEqual(getattr(model, f"{family}_count"), 1)
                    self.assertEqual(getattr(model, f"{family}_label"), [label])
                    np.testing.assert_array_equal(getattr(model, f"{family}_world").numpy(), [-1])

                for name, expected in (
                    ("curve_body", (1, 4)),
                    ("curve_joint", (1, 4)),
                    ("surface_particle", (1, 5)),
                    ("surface_tri", (0, 2)),
                    ("surface_edge", (0, 5)),
                    ("volume_particle", (5, 9)),
                    ("volume_tet", (0, 1)),
                ):
                    for suffix, endpoint in zip(("start", "end"), expected, strict=True):
                        array = getattr(model, f"{name}_{suffix}")
                        self.assertEqual(array.device, model.device)
                        np.testing.assert_array_equal(array.numpy(), [endpoint])


if __name__ == "__main__":
    unittest.main()
