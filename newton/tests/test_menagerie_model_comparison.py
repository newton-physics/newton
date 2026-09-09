# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for the Menagerie model comparison entry point."""

import unittest

import numpy as np
import warp as wp

from newton.tests import test_menagerie_usd_mujoco
from newton.tests.test_menagerie_mujoco import (
    DEFAULT_MODEL_SKIP_FIELDS,
    MUJOCO_AVAILABLE,
    _mujoco,
    _mujoco_warp,
    compare_models,
)


@unittest.skipIf(not MUJOCO_AVAILABLE, "mujoco/mujoco_warp not installed")
class TestMenagerieModelComparison(unittest.TestCase):
    def setUp(self):
        """Keep all model allocations on CPU."""
        self.device = wp.ScopedDevice("cpu")
        self.device.__enter__()
        self.addCleanup(self.device.__exit__, None, None, None)

    @staticmethod
    def _model(*, reverse=False, empty=False):
        geoms = [
            '<geom type="sphere" size="0.1" group="1" friction="0.7 0.01 0.001"/>',
            '<geom type="box" size="0.1 0.2 0.3" group="2" pos="1 0 0"/>',
            '<geom type="capsule" size="0.1 0.2" group="3" pos="2 0 0"/>',
            '<geom type="cylinder" size="0.2 0.3" group="4" pos="3 0 0"/>',
        ]
        if reverse:
            geoms.reverse()
        if empty:
            geoms = []
        xml = "<mujoco><worldbody><body><freejoint/>"
        xml += '<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
        xml += "".join(geoms) + "</body></worldbody></mujoco>"
        return _mujoco_warp.put_model(_mujoco.MjModel.from_xml_string(xml))

    def test_equal_and_reordered_models(self):
        """Accept equal models and reordering of distinct body/type groups."""
        expected = self._model()
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                compare_models(self._model(reverse=reverse), expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)

    def test_default_geom_mismatches(self):
        """Reject friction and group differences through the default entry point."""
        for field in ("geom_friction", "geom_group"):
            with self.subTest(field=field):
                actual, expected = self._model(), self._model()
                arr = getattr(actual, field)
                arr.assign(arr.numpy() + 1)
                with self.assertRaisesRegex(AssertionError, field):
                    compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)

    def test_field_exclusion(self):
        """Skip only the selected field while still detecting other differences."""
        actual, expected = self._model(), self._model()
        actual.geom_friction.assign(actual.geom_friction.numpy() * 2)
        skips = DEFAULT_MODEL_SKIP_FIELDS | {"geom_friction"}
        compare_models(actual, expected, skip_fields=skips)
        actual.geom_group.assign(actual.geom_group.numpy() + 1)
        with self.assertRaisesRegex(AssertionError, "geom_group"):
            compare_models(actual, expected, skip_fields=skips)

    def test_geom_opt_out(self):
        """Preserve explicit broad geometry exclusions."""
        actual, expected = self._model(), self._model()
        actual.geom_friction.assign(actual.geom_friction.numpy() * 2)
        actual.geom_group.assign(actual.geom_group.numpy() + 1)
        for skips in (
            DEFAULT_MODEL_SKIP_FIELDS | {"geom_"},
            test_menagerie_usd_mujoco.TestMenagerieUSD.model_skip_fields,
        ):
            with self.subTest(skips=skips):
                compare_models(actual, expected, skip_fields=skips)

    def test_geom_counts(self):
        """Accept empty models and reject unequal counts unless explicitly skipped."""
        actual, expected = self._model(empty=True), self._model(empty=True)
        compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
        actual = self._model()
        with self.assertRaisesRegex(AssertionError, "ngeom"):
            compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
        compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS | {"ngeom", "nmaxcondim", "nmaxpyramid"})

    def test_batched_geom_fields(self):
        """Compare every world of supported scalar and vector geometry arrays."""
        actual, expected = self._model(), self._model()
        for field in ("geom_friction", "geom_margin", "geom_size"):
            for model in (actual, expected):
                arr = getattr(model, field)
                values = np.repeat(arr.numpy(), 2, axis=0)
                setattr(model, field, wp.array(values, dtype=arr.dtype))
        compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
        for field in ("geom_friction", "geom_margin", "geom_size"):
            with self.subTest(field=field):
                arr = getattr(actual, field)
                original = arr.numpy().copy()
                changed = original.copy()
                changed[1, 0] += 1
                arr.assign(changed)
                with self.assertRaisesRegex(AssertionError, field):
                    compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
                arr.assign(original)

    def test_single_geom_and_plane_exceptions(self):
        """Compare single geometries and preserve cosmetic plane exceptions."""
        for geom in (
            '<geom type="sphere" size="0.1"/>',
            '<geom type="plane" size="1 1 0.1"/>',
        ):
            with self.subTest(geom=geom):
                source = _mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{geom}</worldbody></mujoco>")
                actual, expected = _mujoco_warp.put_model(source), _mujoco_warp.put_model(source)
                compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
                if "plane" in geom:
                    actual.geom_pos.assign(actual.geom_pos.numpy() + 1)
                    actual.geom_quat.assign(actual.geom_quat.numpy() + 1)
                    actual.geom_size.assign(actual.geom_size.numpy() + 1)
                    compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
                actual.geom_friction.assign(actual.geom_friction.numpy() * 2)
                with self.assertRaisesRegex(AssertionError, "geom_friction"):
                    compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)

    def test_geom_array_shapes(self):
        """Reject unmatched batch sizes rather than ignoring extra worlds."""
        for field in ("geom_friction", "geom_size"):
            with self.subTest(field=field):
                actual, expected = self._model(), self._model()
                arr = getattr(expected, field)
                setattr(expected, field, wp.array(np.repeat(arr.numpy(), 2, axis=0), dtype=arr.dtype))
                with self.assertRaisesRegex(AssertionError, f"{field}: shape"):
                    compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)

    def test_geom_size_semantics(self):
        """Ignore unused size components but compare physical dimensions."""
        actual, expected = self._model(), self._model()
        size = actual.geom_size.numpy()
        size[0, 0, 1:] = 9  # sphere: radius only
        size[0, 2:, 2] = 9  # capsule/cylinder: radius and half-length
        actual.geom_size.assign(size)
        compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
        size[0, 1, 2] += 1  # box: all three dimensions matter
        actual.geom_size.assign(size)
        with self.assertRaisesRegex(AssertionError, "geom_size"):
            compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS)
        compare_models(actual, expected, skip_fields=DEFAULT_MODEL_SKIP_FIELDS | {"geom_size"})


if __name__ == "__main__":
    unittest.main()
