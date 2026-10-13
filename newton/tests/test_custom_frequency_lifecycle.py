# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise solver-independent custom-frequency lifecycle behavior."""

import unittest
from dataclasses import replace

import numpy as np
import warp as wp

import newton


def _register_rows(builder, **kwargs):
    """Register labeled rows with world ownership and optional lifecycle callbacks."""
    builder.add_custom_frequency(
        newton.ModelBuilder.CustomFrequency(
            name="row",
            namespace="test",
            label_attribute="test:row_label",
            world_attribute="test:row_world",
            **kwargs,
        )
    )
    for name, dtype, default, references in (
        ("row_label", str, "", None),
        ("row_world", wp.int32, -1, "world"),
        ("row_body", wp.int32, -1, "body"),
        ("row_point", wp.vec3, wp.vec3(), None),
    ):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=name,
                namespace="test",
                frequency="test:row",
                dtype=dtype,
                default=default,
                references=references,
            )
        )


class TestCustomFrequencyLifecycle(unittest.TestCase):
    def test_plain_builder_has_no_solver_attributes(self):
        """Build and finalize an empty model without registering a solver schema."""
        builder = newton.ModelBuilder()
        self.assertEqual(builder.custom_attributes, {})
        self.assertEqual(builder.custom_frequencies, {})
        self.assertFalse(hasattr(builder.finalize(device="cpu"), "mujoco"))

    def test_world_metadata_and_labels_after_composition(self):
        """Preserve global rows, world ownership, and labels through repeated merges."""
        source = newton.ModelBuilder()
        _register_rows(source)
        source.add_custom_values(**{"test:row_label": "first"})
        source.add_custom_values(**{"test:row_label": ""})
        destination = newton.ModelBuilder()
        _register_rows(destination)
        destination.add_custom_values(**{"test:row_label": "front"})
        destination.add_world(source, label_prefix="left")
        destination.add_world(source, label_prefix="right")
        destination.add_custom_values(**{"test:row_label": "tail"})

        for _ in range(2):
            model = destination.finalize(device="cpu")
            self.assertEqual(model.test.row_count, 6)
            np.testing.assert_array_equal(model.test.row_world.numpy(), [-1, 0, 0, 1, 1, -1])
            np.testing.assert_array_equal(model.test.row_world_start.numpy(), [1, 3, 5, 6])
            self.assertEqual(model.test.row_label, ["front", "left/first", "", "right/first", "", "tail"])
        self.assertEqual(source.custom_attributes["test:row_label"].values, ["first", ""])

    def test_empty_world_metadata(self):
        """Publish zero counts and starts for a registered frequency with no rows."""
        builder = newton.ModelBuilder()
        _register_rows(builder)
        model = builder.finalize(device="cpu")
        self.assertEqual(model.test.row_count, 0)
        np.testing.assert_array_equal(model.test.row_world_start.numpy(), [0, 0, 0])

    def test_world_validation_can_be_skipped(self):
        """Validate custom world ownership with the same switches as built-in entities."""
        builder = newton.ModelBuilder()
        _register_rows(builder)
        builder.add_custom_values(**{"test:row_world": 0})
        builder.add_custom_values(**{"test:row_world": -1})
        builder.add_custom_values(**{"test:row_world": 0})
        with self.assertRaisesRegex(ValueError, "Invalid world ordering in test:row_world"):
            builder.finalize(device="cpu")
        builder.finalize(device="cpu", skip_validation_worlds=True)

    def test_validator_survives_replication(self):
        """Copy a frequency validator and honor the structural-validation switch."""
        calls = []

        def validate(builder):
            calls.append(builder._custom_frequency_counts["test:row"])
            raise ValueError("invalid custom row")

        source = newton.ModelBuilder()
        _register_rows(source, builder_validator=validate)
        source.add_custom_values(**{"test:row_label": "row"})
        destination = newton.ModelBuilder()
        destination.replicate(source, 2)
        with self.assertRaisesRegex(ValueError, "invalid custom row"):
            destination.finalize(device="cpu")
        self.assertEqual(calls, [2])
        destination.finalize(device="cpu", skip_validation_structure=True)
        destination.finalize(device="cpu", skip_all_validations=True)
        self.assertEqual(calls, [2])

    def test_collapse_transformer_preserves_local_frame(self):
        """Apply a registered frame fixup before generic body-reference remapping."""

        def collapse(builder, context):
            bodies = builder.custom_attributes["test:row_body"].values
            points = builder.custom_attributes["test:row_point"].values
            for row, body in enumerate(bodies):
                if body in context["body_merged_transform"]:
                    points[row] = wp.transform_point(context["body_merged_transform"][body], points[row])

        source = newton.ModelBuilder()
        _register_rows(source, collapse_transformer=collapse)
        parent = source.add_link(mass=1.0)
        child = source.add_link(mass=1.0)
        root = source.add_joint_free(parent)
        fixed = source.add_joint_fixed(parent, child, parent_xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))
        source.add_articulation([root, fixed])
        source.add_custom_values(**{"test:row_body": child, "test:row_point": wp.vec3(0.5, 0.0, 0.0)})
        destination = newton.ModelBuilder()
        destination.add_builder(source)
        destination.collapse_fixed_joints()
        self.assertEqual(destination.body_count, 1)
        self.assertEqual(destination.custom_attributes["test:row_body"].values, [0])
        np.testing.assert_allclose(destination.custom_attributes["test:row_point"].values[0], [1.5, 2.0, 3.0])
        np.testing.assert_allclose(source.custom_attributes["test:row_point"].values[0], [0.5, 0.0, 0.0])

    def test_lifecycle_schema_conflicts_are_rejected_before_merge(self):
        """Reject different lifecycle metadata without appending source entities."""
        source = newton.ModelBuilder()
        _register_rows(source)
        source.add_body()
        frequency = source.custom_frequencies["test:row"]
        for field, value in (
            ("world_attribute", None),
            ("builder_validator", lambda _builder: None),
            ("collapse_transformer", lambda _builder, _context: None),
        ):
            with self.subTest(field=field):
                destination = newton.ModelBuilder()
                destination.add_custom_frequency(replace(frequency, **{field: value}))
                with self.assertRaisesRegex(ValueError, "different callbacks or metadata"):
                    destination.add_builder(source)
                self.assertEqual(destination.body_count, 0)

    def test_preserve_referenced_bodies_is_opt_in(self):
        """Keep a world-fixed body only when its reference requires a surviving body."""
        for preserve in (False, True):
            with self.subTest(preserve=preserve):
                builder = newton.ModelBuilder()
                builder.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name="body_reference",
                        dtype=wp.int32,
                        frequency=newton.Model.AttributeFrequency.ONCE,
                        references="body",
                        default=-1,
                        values={0: 0},
                        preserve_referenced_bodies=preserve,
                    )
                )
                body = builder.add_link(mass=1.0)
                joint = builder.add_joint_fixed(-1, body)
                builder.add_articulation([joint])
                builder.collapse_fixed_joints()
                self.assertEqual(builder.body_count, int(preserve))
                self.assertEqual(builder.custom_attributes["body_reference"].values, {0: 0 if preserve else -1})

    def test_preserve_referenced_bodies_schema_is_checked(self):
        """Reject invalid preservation declarations and conflicting merge schemas."""
        with self.assertRaisesRegex(ValueError, "preserve_referenced_bodies requires"):
            newton.ModelBuilder.CustomAttribute(
                name="ref",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.ONCE,
                references="joint",
                preserve_referenced_bodies=True,
            )
        attribute = newton.ModelBuilder.CustomAttribute(
            name="ref",
            dtype=wp.int32,
            frequency=newton.Model.AttributeFrequency.ONCE,
            references="body",
            preserve_referenced_bodies=True,
        )
        source = newton.ModelBuilder()
        source.add_custom_attribute(attribute)
        destination = newton.ModelBuilder()
        destination.add_custom_attribute(replace(attribute, preserve_referenced_bodies=False))
        with self.assertRaisesRegex(ValueError, "incompatible spec"):
            destination.add_builder(source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
