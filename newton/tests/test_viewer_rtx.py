# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib.util
import unittest
import warnings
from unittest import mock

import numpy as np

from newton._src.viewer.viewer_rtx import ViewerRTX

OVRTX_AVAILABLE = importlib.util.find_spec("ovrtx") is not None
OVSTAGE_AVAILABLE = importlib.util.find_spec("ovstage") is not None


@unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
class TestViewerRTXVersionCompatibility(unittest.TestCase):
    def test_legacy_ovrtx_does_not_require_ovstage(self):
        """Construct ViewerRTX with legacy OVRTX without importing OVStage."""
        import ovrtx

        original_import = builtins.__import__

        def import_without_ovstage(name, *args, **kwargs):
            if name == "ovstage":
                raise AssertionError("legacy OVRTX must not import ovstage")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(ovrtx, "__version__", "0.3.0"),
            mock.patch("builtins.__import__", side_effect=import_without_ovstage),
        ):
            viewer = ViewerRTX(headless=True)

        viewer.close()

    def test_modern_ovrtx_requires_ovstage(self):
        """Require OVStage when constructing ViewerRTX with OVRTX 0.4 or newer."""
        import ovrtx

        original_import = builtins.__import__

        def import_without_ovstage(name, *args, **kwargs):
            if name == "ovstage":
                raise ImportError("ovstage unavailable")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(ovrtx, "__version__", "0.4.0"),
            mock.patch("builtins.__import__", side_effect=import_without_ovstage),
            self.assertRaisesRegex(ImportError, "OVRTX 0.4 or newer"),
        ):
            ViewerRTX(headless=True)


@unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
class TestViewerRTXOvstage(unittest.TestCase):
    def setUp(self):
        """Create an ovstage-backed ViewerRTX runtime stub."""
        import ovstage

        self.ovstage = ovstage
        self.viewer = ViewerRTX.__new__(ViewerRTX)
        self.viewer._rtx = None
        self.viewer._use_ovstage = True
        self.viewer._ovstage = self.ovstage.Stage("newton.test.ViewerRTX")
        self.viewer._ovstage_attached = False
        self.viewer._ovstage_paths = self.ovstage.PathDictionary(self.viewer._ovstage)
        self.viewer._ovstage_queries = {}
        self.viewer._ovstage_ordinal = 1
        self.ovstage.population.open_usd_from_string(
            self.viewer._ovstage,
            """#usda 1.0
def Xform "World"
{
    def Xform "A"
    {
    }
    def Xform "B"
    {
    }
}
""",
            ordinal=self.viewer._ovstage_ordinal,
            time_code=0.0,
        )
        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()

    def tearDown(self):
        """Release the ovstage runtime stub."""
        self.viewer._release_ovstage()

    def test_write_visibility_uses_reusable_query_without_deprecation(self):
        """Write token attributes through one reusable ordered query."""
        prim_paths = ["/World/A", "/World/B"]
        self.viewer._ovstage_ordinal = 2

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.viewer._write_runtime_attribute(
                prim_paths,
                "visibility",
                ["inherited", "invisible"],
            )

        self.assertFalse([warning for warning in caught if warning.category is DeprecationWarning])
        query = self.viewer._get_ovstage_query(prim_paths)
        self.assertIs(query, self.viewer._get_ovstage_query(prim_paths))
        self.assertEqual(len(self.viewer._ovstage_queries), 1)

        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()
        attribute = self.viewer._ovstage_paths.intern_token("visibility")
        with self.viewer._ovstage.read_attributes(
            query,
            [attribute],
            self.ovstage.OrdinalRange.latest(self.viewer._ovstage_ordinal),
        ) as read:
            group = read.fetch_next()
            data_rows = np.from_dlpack(group.dlpack(0)).copy()
            values = [data_rows[group.data_row_index(index)] for index in range(group.data_count)]
            self.viewer._ovstage.release_group(group)

        self.assertEqual(
            [self.viewer._ovstage_paths.token_to_string(int(value)) for value in values],
            ["inherited", "invisible"],
        )

    def test_write_array_attribute_preserves_vector_lanes(self):
        """Write a runtime vector array with its element layout intact."""
        points = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        self.viewer._ovstage_ordinal = 2
        self.viewer._write_runtime_array_attribute("/World/A", "points", points)
        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()

        query = self.viewer._get_ovstage_query(["/World/A"])
        attribute = self.viewer._ovstage_paths.intern_token("points")
        with self.viewer._ovstage.read_attributes(
            query,
            [attribute],
            self.ovstage.OrdinalRange.latest(self.viewer._ovstage_ordinal),
        ) as read:
            group = read.fetch_next()
            self.assertEqual(group.tensor(0).dtype.lanes, 3)
            values = np.from_dlpack(group.dlpack(0)).copy()
            self.viewer._ovstage.release_group(group)

        np.testing.assert_allclose(values, points)


if __name__ == "__main__":
    unittest.main(verbosity=2)
