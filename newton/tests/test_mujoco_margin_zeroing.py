# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for issue #2106: NATIVECCD margin NotImplementedError.

Upstream ``mujoco_warp`` rejects non-zero geom margins at ``put_model()``
time when NATIVECCD is enabled.  Newton must zero margins in the MJCF spec,
and keep ``mjw_model.geom_margin`` zero when MuJoCo handles collisions.
"""

import unittest
import warnings

import numpy as np

import newton
from newton import ModelFlags
from newton.solvers import SolverMuJoCo


class TestMuJoCoMarginZeroing(unittest.TestCase):
    """Verify SolverMuJoCo margin handling for NATIVECCD compatibility."""

    @staticmethod
    def _build_model_with_margin(margin: float = 1e-5, *, nativeccd: bool = True) -> newton.Model:
        """Build a minimal model with two boxes that have non-zero margin."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.custom_attributes["mujoco:flag_nativeccd"].values[0] = nativeccd
        builder.add_shape_box(
            body=-1,
            hx=1.0,
            hy=1.0,
            hz=0.01,
            cfg=newton.ModelBuilder.ShapeConfig(margin=margin),
        )
        b = builder.add_body(label="box")
        builder.add_shape_box(
            body=b,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(margin=margin),
        )
        return builder.finalize()

    def test_geom_margin_zeroed_throughout_lifecycle(self):
        """Preserve margins unless enabled NATIVECCD or MULTICCD requires zeroing."""
        for nativeccd, multiccd in ((True, False), (False, False), (False, True)):
            with self.subTest(nativeccd=nativeccd, multiccd=multiccd):
                model = self._build_model_with_margin(margin=0.01, nativeccd=nativeccd)
                expected_margin = 0.0 if nativeccd or multiccd else 0.01
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver = SolverMuJoCo(model, enable_multiccd=multiccd)
                self.assertEqual(
                    any("zeroed for NATIVECCD/MULTICCD" in str(w.message) for w in caught),
                    expected_margin == 0.0,
                )
                np.testing.assert_allclose(solver.mj_model.geom_margin, expected_margin)
                np.testing.assert_allclose(solver.mjw_model.geom_margin.numpy(), expected_margin)
                solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
                np.testing.assert_allclose(solver.mjw_model.geom_margin.numpy(), expected_margin)


if __name__ == "__main__":
    unittest.main()
