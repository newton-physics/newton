# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from projects.digital_instron_v2.geometry import BakedMidsoleGeometry, MeshFrame
from projects.digital_instron_v2.workflow import (
    _soften_binary_coverage,
    compute_baked_compression,
)


def _test_baked_geometry() -> BakedMidsoleGeometry:
    bottom = np.asarray([[0.0, 0.002], [0.0, 0.002]], dtype=np.float64)
    thickness = np.full((2, 2), 0.010, dtype=np.float64)
    top = bottom + thickness
    return BakedMidsoleGeometry(
        thickness_map=thickness,
        top_map=top,
        bottom_map=bottom,
        mins_uv=np.asarray([0.0, 0.0], dtype=np.float64),
        maxs_uv=np.asarray([1.0, 1.0], dtype=np.float64),
        frame=MeshFrame(
            plane_axes=(0, 1),
            thickness_axis=2,
            center_m=np.zeros(3, dtype=np.float64),
            extents_m=np.ones(3, dtype=np.float64),
        ),
        valid_map=np.ones((2, 2), dtype=np.float64),
        grid_uv_m=np.asarray([[1.0, 0.0]], dtype=np.float64),
        xy_m=np.asarray([[1.0, 0.0]], dtype=np.float64),
        cell_area_m2=1.0,
        spacing_m=1.0,
    )


class TestDigitalInstronV2(unittest.TestCase):
    def test_baked_compression_uses_local_bottom_support_in_equilibrium(self):
        geometry = _test_baked_geometry()
        xy_m = np.asarray([[1.0, 0.0]], dtype=np.float64)
        indenter_map = geometry.top_map + 0.002
        indenter_valid = np.asarray([[1.0, 0.25], [1.0, 0.25]], dtype=np.float64)

        comp = compute_baked_compression(
            xy_m,
            geometry,
            indenter_map,
            indenter_valid,
            0.003,
            use_equilibrium=True,
            use_subcell_coverage=True,
        )
        np.testing.assert_allclose(comp, [0.001])

        comp = compute_baked_compression(
            xy_m,
            geometry,
            indenter_map,
            indenter_valid,
            0.005,
            use_equilibrium=True,
            use_subcell_coverage=True,
        )
        np.testing.assert_allclose(comp, [0.003])

        gated_comp = compute_baked_compression(
            xy_m,
            geometry,
            indenter_map,
            indenter_valid,
            0.005,
            use_equilibrium=True,
            use_subcell_coverage=False,
        )
        np.testing.assert_allclose(gated_comp, [0.0])

    def test_soften_binary_coverage_keeps_interior_and_fractions_edges(self):
        valid = np.asarray(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ]
        )

        coverage = _soften_binary_coverage(valid)

        self.assertEqual(float(coverage[0, 0]), 0.0)
        self.assertGreater(float(coverage[1, 1]), 0.0)
        self.assertLess(float(coverage[1, 1]), 1.0)

        full_valid = np.ones((3, 3), dtype=bool)
        full_coverage = _soften_binary_coverage(full_valid)
        self.assertEqual(float(full_coverage[1, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
