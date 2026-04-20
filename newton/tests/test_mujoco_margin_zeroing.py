# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for issue #2106: NATIVECCD margin NotImplementedError.

Upstream ``mujoco_warp`` rejects non-zero geom margins at ``put_model()``
time when NATIVECCD is enabled.  Newton must zero margins in the MJCF spec,
and keep ``mjw_model.geom_margin`` zero when MuJoCo handles collisions.
"""

import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo, SolverNotifyFlags


class TestMuJoCoMarginZeroing(unittest.TestCase):
    """Verify SolverMuJoCo margin handling for NATIVECCD compatibility."""

    @staticmethod
    def _build_model_with_margin(margin: float = 1e-5) -> newton.Model:
        """Build a minimal model with two boxes that have non-zero margin."""
        builder = newton.ModelBuilder()
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

    def test_mj_model_geom_margin_zero(self):
        """The compiled MjModel must have zero geom_margin (zeroed in MjSpec before compilation)."""
        model = self._build_model_with_margin(margin=1e-5)
        with self.assertWarnsRegex(UserWarning, r"zeroed for NATIVECCD/MULTICCD"):
            solver = SolverMuJoCo(model, use_mujoco_contacts=True)
        np.testing.assert_array_equal(
            solver.mj_model.geom_margin,
            np.zeros_like(solver.mj_model.geom_margin),
            err_msg="MjModel geom_margin should be zero in the spec",
        )

    def test_geom_margin_zero_with_mujoco_contacts(self):
        """When MuJoCo handles collisions, mjw_model.geom_margin must stay zero."""
        model = self._build_model_with_margin(margin=1e-5)
        with self.assertWarnsRegex(UserWarning, r"zeroed for NATIVECCD/MULTICCD"):
            solver = SolverMuJoCo(model, use_mujoco_contacts=True)
        geom_margin = solver.mjw_model.geom_margin.numpy()
        np.testing.assert_array_equal(
            geom_margin,
            np.zeros_like(geom_margin),
            err_msg="geom_margin should be zero when use_mujoco_contacts=True",
        )

    def test_geom_margin_stays_zero_after_notify(self):
        """Margins must stay zero after notify_model_changed with MuJoCo contacts."""
        model = self._build_model_with_margin(margin=1e-5)
        with self.assertWarnsRegex(UserWarning, r"zeroed for NATIVECCD/MULTICCD"):
            solver = SolverMuJoCo(model, use_mujoco_contacts=True)
        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)
        geom_margin = solver.mjw_model.geom_margin.numpy()
        np.testing.assert_array_equal(
            geom_margin,
            np.zeros_like(geom_margin),
            err_msg="geom_margin should remain zero after notify_model_changed",
        )


if __name__ == "__main__":
    unittest.main()
