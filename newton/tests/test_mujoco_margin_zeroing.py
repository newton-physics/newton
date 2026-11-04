# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for SolverMuJoCo margin handling (issue #2106).

Earlier ``mujoco_warp`` releases rejected non-zero geom margins at ``put_model()``
when NATIVECCD or MULTICCD was enabled. The MuJoCo Warp backend still zeroes
margins in the MJCF spec and keeps ``mjw_model.geom_margin`` zero when MuJoCo
handles collisions; the CPU backend keeps the authored margins.
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

    def test_cpu_backend_keeps_margins_like_native_mujoco(self):
        """The CPU backend keeps authored geom and pair margins and their contacts."""
        mujoco, _ = SolverMuJoCo.import_mujoco()
        mesh = '<mesh name="tetra" vertex="-0.1 -0.1 0  0.1 -0.1 0  0 0.1 0  0 0 0.1"/>'
        for multiccd in ("enable", "disable"):
            with self.subTest(multiccd=multiccd):
                mjcf = f"""<mujoco><option><flag multiccd="{multiccd}"/></option><asset>{mesh}</asset>
                  <worldbody><geom name="floor" type="box" size="1 1 0.01" margin="0.004"/>
                    <body name="box" pos="0 0 0.062"><freejoint/>
                      <geom name="box" type="box" size="0.05 0.05 0.05" margin="0.003"/></body>
                    <body name="mesh" pos="0.5 0 0.012"><freejoint/>
                      <geom name="mesh" type="mesh" mesh="tetra" margin="0.002"/></body>
                  </worldbody>
                  <contact><pair geom1="floor" geom2="mesh" margin="0.005"/></contact></mujoco>"""
                native = mujoco.MjModel.from_xml_string(mjcf)
                builder = newton.ModelBuilder()
                builder.add_mjcf(mjcf)
                model = builder.finalize(device="cpu")
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True)
                self.assertEqual(solver.mj_model.opt.disableflags, native.opt.disableflags)
                # Match each compiled geom to the native geom it was imported from.
                geom_names = [
                    model.shape_label[shape].rsplit("/", 1)[-1] for shape in solver.mjc_geom_to_newton_shape.numpy()[0]
                ]
                native_margins = [native.geom(name).margin[0] for name in geom_names]
                np.testing.assert_allclose(solver.mj_model.geom_margin, native_margins, atol=1e-7)
                np.testing.assert_allclose(solver.mj_model.pair_margin, native.pair_margin, atol=1e-7)
                expected, actual = mujoco.MjData(native), mujoco.MjData(solver.mj_model)
                mujoco.mj_forward(native, expected)
                mujoco.mj_forward(solver.mj_model, actual)
                self.assertGreater(expected.ncon, 0)
                self.assertEqual(actual.ncon, expected.ncon)
                np.testing.assert_allclose(
                    np.sort(actual.contact.dist[: actual.ncon]),
                    np.sort(expected.contact.dist[: expected.ncon]),
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
