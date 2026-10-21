# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect finalized deformable objects through public Model attributes."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.exceptions import NewtonWarning
from newton.tests.test_deformable_objects import _add_curve, _add_surface, _add_volume
from newton.tests.unittest_utils import get_test_devices


class TestModelDeformableObjects(unittest.TestCase):
    def test_unknown_reference_names_suggest_builtin_enums(self):
        """Suggest enums for unregistered family names in builder and Model references."""
        for name in ("curve", "surface", "volume", "missing"):
            expected = f"Unknown custom attribute reference frequency {name!r}"
            if name != "missing":
                expected += f"; to reference the built-in {name} domain, use Model.AttributeFrequency.{name.upper()}"
            for from_builder in (False, True):
                with self.subTest(name=name, from_builder=from_builder):
                    with self.assertRaises(ValueError) as error:
                        if from_builder:
                            builder = newton.ModelBuilder()
                            builder.add_custom_attribute(
                                newton.ModelBuilder.CustomAttribute(
                                    name="reference",
                                    dtype=wp.int32,
                                    frequency=newton.Model.AttributeFrequency.ONCE,
                                    references=name,
                                    values={0: 0},
                                )
                            )
                            builder.finalize(device="cpu")
                        else:
                            model = newton.Model(device="cpu")
                            model.add_attribute(
                                "reference",
                                wp.array([0], dtype=wp.int32, device="cpu"),
                                frequency=newton.Model.AttributeFrequency.ONCE,
                                references=name,
                            )
                    self.assertEqual(str(error.exception), expected)

    def test_explicit_deformable_reference_domains(self):
        """Keep enum references to each deformable family through finalization and replication."""
        prototype = newton.ModelBuilder()
        for family, add in (("curve", _add_curve), ("surface", _add_surface), ("volume", _add_volume)):
            add(prototype, label="asset")
            frequency = getattr(newton.Model.AttributeFrequency, family.upper())
            prototype.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"{family}_reference",
                    dtype=wp.int32,
                    frequency=frequency,
                    references=frequency,
                    values={0: 0},
                )
            )
        for replicate in (False, True):
            with self.subTest(replicate=replicate):
                builder = newton.ModelBuilder() if replicate else prototype
                if replicate:
                    builder.replicate(prototype, 2)
                model = builder.finalize(device="cpu")
                for family in ("curve", "surface", "volume"):
                    name = f"{family}_reference"
                    np.testing.assert_array_equal(getattr(model, name).numpy(), [0, 1] if replicate else [0])
                    self.assertEqual(
                        model.attribute_specs[name].references,
                        getattr(newton.Model.AttributeFrequency, family.upper()),
                    )

    def test_vector_reference_sentinels_survive_composition(self):
        """Offset valid vector references without changing negative entries or source values."""
        prototype = newton.ModelBuilder()
        values = {
            "warp": wp.vec2i(0, -1),
            "numpy": np.array([0, -1], dtype=np.int32),
            "list": [0, -1],
            "tuple": (0, -1),
        }
        for family, add in (("curve", _add_curve), ("surface", _add_surface), ("volume", _add_volume)):
            add(prototype, label="asset")
            frequency = getattr(newton.Model.AttributeFrequency, family.upper())
            for kind, value in values.items():
                prototype.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name=f"{family}_{kind}",
                        dtype=wp.vec2i,
                        frequency=frequency,
                        references=frequency,
                        values={0: value},
                    )
                )
        for method in ("add_builder", "replicate"):
            builder = newton.ModelBuilder()
            if method == "add_builder":
                builder.add_builder(prototype)
                builder.add_builder(prototype)
            else:
                builder.replicate(prototype, 2)
            model = builder.finalize(device="cpu")
            for family in ("curve", "surface", "volume"):
                for kind in values:
                    with self.subTest(method=method, family=family, kind=kind):
                        array = getattr(model, f"{family}_{kind}")
                        self.assertEqual(array.dtype, wp.vec2i)
                        np.testing.assert_array_equal(array.numpy(), [[0, -1], [1, -1]])
        for value in values.values():
            np.testing.assert_array_equal(value, [0, -1])

    def test_reference_defaults_survive_composition(self):
        """Remap default and sparse references without changing source builders or ordinary defaults."""
        cases = (
            ("scalar", wp.int32, 0, {}, [0, 0, 2, 2]),
            ("implicit", wp.int32, None, {}, [0, 0, 2, 2]),
            ("sparse", wp.int32, 1, {1: -1}, [1, -1, 3, -1]),
            ("vector", wp.vec2i, wp.vec2i(0, -1), {}, [[0, -1], [0, -1], [2, -1], [2, -1]]),
            ("sparse_vector", wp.vec2i, wp.vec2i(0, -1), {1: wp.vec2i(1, -2)}, [[0, -1], [1, -2], [2, -1], [3, -2]]),
            ("negative", wp.int32, -1, {}, [-1, -1, -1, -1]),
        )
        prototype = newton.ModelBuilder()
        for family, add in (("curve", _add_curve), ("surface", _add_surface), ("volume", _add_volume)):
            add(prototype, label="first")
            add(prototype, label="second")
            frequency = getattr(newton.Model.AttributeFrequency, family.upper())
            for name, dtype, default, values, _ in cases:
                prototype.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name=f"{family}_{name}",
                        dtype=dtype,
                        frequency=frequency,
                        references=frequency,
                        default=default,
                        values=dict(values),
                    )
                )
        prototype.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="temperature",
                dtype=wp.float32,
                frequency=newton.Model.AttributeFrequency.CURVE,
                default=300.0,
            )
        )
        prototype.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="sensor"))
        prototype.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(name="sensor_tag", dtype=wp.int32, frequency="sensor")
        )
        for tag in (10, 20, 30):
            prototype.add_custom_values(sensor_tag=tag)
        for name, values in (("default", []), ("sparse", [None, 1])):
            prototype.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"sensor_curve_{name}",
                    dtype=wp.int32,
                    frequency="sensor",
                    references=newton.Model.AttributeFrequency.CURVE,
                    default=0,
                    values=values,
                )
            )
        for method in ("add_builder", "replicate", "source"):
            builder = prototype if method == "source" else newton.ModelBuilder()
            if method == "add_builder":
                builder.add_builder(prototype)
                builder.add_builder(prototype)
            elif method == "replicate":
                builder.replicate(prototype, 2)
            if method == "source":
                with self.assertWarnsRegex(
                    NewtonWarning, r"Custom attribute 'sensor_curve_sparse' has 2 values.*expects 3"
                ):
                    model = builder.finalize(device="cpu")
            else:
                model = builder.finalize(device="cpu")
            for family in ("curve", "surface", "volume"):
                for name, _, _, _, expected in cases:
                    with self.subTest(method=method, family=family, kind=name):
                        np.testing.assert_array_equal(
                            getattr(model, f"{family}_{name}").numpy(),
                            expected[:2] if method == "source" else expected,
                        )
            np.testing.assert_array_equal(model.temperature.numpy(), [300.0] * model.curve_count)
            for name, expected in (("default", [0, 0, 0, 2, 2, 2]), ("sparse", [0, 1, 0, 2, 3, 2])):
                with self.subTest(method=method, sensor=name):
                    np.testing.assert_array_equal(
                        getattr(model, f"sensor_curve_{name}").numpy(),
                        expected[:3] if method == "source" else expected,
                    )
        self.assertEqual(prototype.custom_attributes["curve_scalar"].values, {})
        self.assertEqual(prototype.custom_attributes["curve_sparse"].values, {1: -1})
        self.assertEqual(prototype.custom_attributes["sensor_curve_sparse"].values, [None, 1])

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

    def test_custom_reference_registration_order(self):
        """Keep custom reference names unchanged across registration orders and reused declarations."""
        for family in ("curve", "surface", "volume"):
            reference = newton.ModelBuilder.CustomAttribute(
                name="tag_reference",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.BODY,
                references=family,
                values={0: 1},
            )
            for custom_first in (False, True):
                with self.subTest(family=family, custom_first=custom_first):
                    prototype = newton.ModelBuilder()
                    prototype.add_body()
                    if not custom_first:
                        prototype.add_custom_attribute(reference)
                    prototype.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=family))
                    prototype.add_custom_attribute(
                        newton.ModelBuilder.CustomAttribute(name="tag", dtype=wp.int32, frequency=family)
                    )
                    prototype.add_custom_values(tag=10)
                    prototype.add_custom_values(tag=20)
                    if custom_first:
                        prototype.add_custom_attribute(reference)
                    scene = newton.ModelBuilder()
                    scene.replicate(prototype, 2)
                    model = scene.finalize(device="cpu")
                    np.testing.assert_array_equal(model.tag_reference.numpy(), [1, 3])
                    self.assertEqual(reference.references, family)
                    self.assertEqual(prototype.custom_attributes["tag_reference"].references, family)

    def test_deformable_reference_domains_survive_composition(self):
        """Keep built-in references distinct from same-named custom frequencies in either merge direction."""
        for family, add in (("curve", _add_curve), ("surface", _add_surface), ("volume", _add_volume)):
            for builtin_in_source in (False, True):
                with self.subTest(family=family, builtin_in_source=builtin_in_source):
                    builtin = newton.ModelBuilder()
                    add(builtin, label="builtin")
                    builtin.add_custom_attribute(
                        newton.ModelBuilder.CustomAttribute(
                            name="builtin_reference",
                            dtype=wp.int32,
                            frequency=newton.Model.AttributeFrequency.ONCE,
                            references=getattr(newton.Model.AttributeFrequency, family.upper()),
                            values={0: 0},
                        )
                    )
                    custom = newton.ModelBuilder()
                    for label in ("first", "second"):
                        add(custom, label=label)
                    custom.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=family))
                    custom.add_custom_attribute(
                        newton.ModelBuilder.CustomAttribute(name="tag", dtype=wp.int32, frequency=family)
                    )
                    for tag in (10, 20, 30):
                        custom.add_custom_values(tag=tag)
                    custom.add_custom_attribute(
                        newton.ModelBuilder.CustomAttribute(
                            name="custom_reference",
                            dtype=wp.int32,
                            frequency=newton.Model.AttributeFrequency.ONCE,
                            references=family,
                            values={0: 2},
                        )
                    )
                    destination, source = (custom, builtin) if builtin_in_source else (builtin, custom)
                    destination.add_builder(source)
                    model = destination.finalize(device="cpu")
                    np.testing.assert_array_equal(model.builtin_reference.numpy(), [2 if builtin_in_source else 0])
                    np.testing.assert_array_equal(model.custom_reference.numpy(), [2])
                    self.assertEqual(model.attribute_specs["custom_reference"].references, family)
                    self.assertEqual(
                        model.attribute_specs["builtin_reference"].references,
                        getattr(newton.Model.AttributeFrequency, family.upper()),
                    )

    def test_conflicting_deformable_references_reject_merge(self):
        """Reject same-name attributes with different reference meanings before modifying the destination."""
        for family, add in (("curve", _add_curve), ("surface", _add_surface), ("volume", _add_volume)):
            for builtin_in_source in (False, True):
                with self.subTest(family=family, builtin_in_source=builtin_in_source):
                    builtin, custom = newton.ModelBuilder(), newton.ModelBuilder()
                    custom.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=family))
                    for builder in (builtin, custom):
                        add(builder, label="asset")
                        builder.add_custom_attribute(
                            newton.ModelBuilder.CustomAttribute(
                                name="reference",
                                dtype=wp.int32,
                                frequency=newton.Model.AttributeFrequency.ONCE,
                                references=family
                                if builder is custom
                                else getattr(newton.Model.AttributeFrequency, family.upper()),
                                values={0: 0},
                            )
                        )
                    destination, source = (custom, builtin) if builtin_in_source else (builtin, custom)
                    counts = (destination.body_count, destination.particle_count)
                    frequencies = dict(destination.custom_frequencies)
                    with self.assertRaisesRegex(ValueError, "reference.*incompatible spec"):
                        destination.add_world(source)
                    self.assertEqual((destination.body_count, destination.particle_count), counts)
                    self.assertEqual(getattr(destination, f"{family}_label"), ["asset"])
                    self.assertEqual(destination.custom_frequencies, frequencies)
                    self.assertEqual(destination.custom_attributes["reference"].values, {0: 0})
                    self.assertEqual((destination.world_count, destination.current_world), (0, -1))

    def test_curve_attributes_follow_fixed_joint_collapse(self):
        """Keep per-curve values and references when a rod loses its anchored segment."""
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
                references=newton.Model.AttributeFrequency.CURVE,
                values={0: 1},
            )
        )
        builder.collapse_fixed_joints()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.curve_label, ["anchored", "retained"])
        np.testing.assert_array_equal(model.curve_body_start.numpy(), [0, 1])
        np.testing.assert_array_equal(model.curve_body_end.numpy(), [1, 4])
        np.testing.assert_array_equal(model.asset_tag.numpy(), [10, 20])
        np.testing.assert_array_equal(model.selected_curve.numpy(), [1])

    def test_curve_and_articulation_vector_references_follow_collapse(self):
        """Remap vector references to removed and surviving objects, preserving sentinels."""
        builder = newton.ModelBuilder()
        points = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)]
        body_owners, joint_owners = {}, {}
        for owner, label in enumerate(("removed", "reduced", "unaffected")):
            bodies, joints = builder.add_rod(
                rod=newton.Rod(points[:2] if label == "removed" else points, radius=0.02),
                label=label,
                wrap_in_articulation=False,
                body_frame_origin="com",
            )
            root = (
                builder.add_joint_free(bodies[0]) if label == "unaffected" else builder.add_joint_fixed(-1, bodies[0])
            )
            builder.add_articulation([*joints, root], label=label)
            if owner:
                body_owners.update(dict.fromkeys(bodies, owner))
                joint_owners.update(dict.fromkeys([*joints, root], owner))

        vector_cases = (
            ("pair", wp.vec2i, wp.vec2i(1, 2), [[0, 1]]),
            ("sentinels", wp.vec3i, wp.vec3i(0, -1, 1), [[-1, -1, 0]]),
            ("numpy_pair", wp.vec2i, np.array([0, 1], dtype=np.int32), [[-1, 0]]),
            ("numpy_sentinels", wp.vec3i, np.array([0, -2, 1], dtype=np.int32), [[-1, -2, 0]]),
            ("list_pair", wp.vec2i, [0, 1], [[-1, 0]]),
            ("tuple_pair", wp.vec2i, (0, 1), [[-1, 0]]),
        )
        default_cases = (
            ("scalar", wp.int32, 1, {}, [0, 0]),
            ("implicit", wp.int32, None, {}, [-1, -1]),
            ("sparse", wp.int32, 1, {2: 2}, [0, 1]),
            ("vector", wp.vec2i, wp.vec2i(1, -1), {}, [[0, -1], [0, -1]]),
            ("sparse_vector", wp.vec2i, wp.vec2i(1, 0), {2: wp.vec2i(2, -2)}, [[0, -1], [1, -2]]),
            ("negative", wp.int32, -2, {}, [-2, -2]),
        )
        for family in ("curve", "articulation"):
            for kind, owners in (("body", body_owners), ("joint", joint_owners)):
                builder.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name=f"{family}_{kind}_owner",
                        dtype=wp.int32,
                        frequency=getattr(newton.Model.AttributeFrequency, kind.upper()),
                        references=getattr(newton.Model.AttributeFrequency, family.upper()),
                        default=0,
                        values=owners.copy(),
                    )
                )
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"{family}_tag",
                    dtype=wp.int32,
                    frequency=getattr(newton.Model.AttributeFrequency, family.upper()),
                    values={0: 10, 1: 20, 2: 30},
                )
            )
            for name, dtype, value, _ in vector_cases:
                builder.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name=f"{family}_{name}",
                        dtype=dtype,
                        frequency=newton.Model.AttributeFrequency.ONCE,
                        references=family
                        if family == "articulation" and name == "pair"
                        else getattr(newton.Model.AttributeFrequency, family.upper()),
                        values={0: value},
                    )
                )
            for name, dtype, default, values, _ in default_cases:
                frequency = getattr(newton.Model.AttributeFrequency, family.upper())
                builder.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name=f"{family}_default_{name}",
                        dtype=dtype,
                        frequency=frequency,
                        references=frequency,
                        default=default,
                        values=dict(values),
                    )
                )
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"{family}_default_once",
                    dtype=wp.vec2i,
                    frequency=newton.Model.AttributeFrequency.ONCE,
                    references=getattr(newton.Model.AttributeFrequency, family.upper()),
                    default=wp.vec2i(0, 1),
                )
            )

        builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="curve"))
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="custom_curve_pair",
                dtype=wp.vec2i,
                frequency="curve",
                references="curve",
            )
        )
        builder.add_custom_values(custom_curve_pair=wp.vec2i(0, 1))
        builder.add_custom_values(custom_curve_pair=wp.vec2i(1, 0))
        for name, values in (("default", []), ("sparse", [None, 2])):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"custom_row_curve_{name}",
                    dtype=wp.int32,
                    frequency="curve",
                    references=newton.Model.AttributeFrequency.CURVE,
                    default=1,
                    values=values,
                )
            )
        with self.assertWarnsRegex(NewtonWarning, "removed.*all.*segment bodies"):
            builder.collapse_fixed_joints()
        model = builder.finalize(device="cpu")
        for family in ("curve", "articulation"):
            self.assertEqual(getattr(model, f"{family}_label"), ["reduced", "unaffected"])
            np.testing.assert_array_equal(getattr(model, f"{family}_tag").numpy(), [20, 30])
            for kind in ("body", "joint"):
                with self.subTest(family=family, kind=kind):
                    np.testing.assert_array_equal(getattr(model, f"{family}_{kind}_owner").numpy(), [0, 1, 1])
            for name, dtype, _, expected in vector_cases:
                with self.subTest(family=family, kind=name):
                    array = getattr(model, f"{family}_{name}")
                    self.assertEqual(array.dtype, dtype)
                    np.testing.assert_array_equal(array.numpy(), expected)
            for name, _, _, _, expected in default_cases:
                with self.subTest(family=family, default=name):
                    np.testing.assert_array_equal(getattr(model, f"{family}_default_{name}").numpy(), expected)
            np.testing.assert_array_equal(getattr(model, f"{family}_default_once").numpy(), [[-1, 0]])
        np.testing.assert_array_equal(model.custom_curve_pair.numpy(), [[0, 1], [1, 0]])
        np.testing.assert_array_equal(model.custom_row_curve_default.numpy(), [0, 0])
        np.testing.assert_array_equal(model.custom_row_curve_sparse.numpy(), [0, 1])
        self.assertEqual(model.attribute_specs["curve_pair"].references, newton.Model.AttributeFrequency.CURVE)
        self.assertEqual(model.attribute_specs["custom_curve_pair"].references, "curve")

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
