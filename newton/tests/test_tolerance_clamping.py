# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import newton
from newton.solvers import SolverMuJoCo


class TestToleranceClamping(unittest.TestCase):
    def test_tolerance_clamped_to_1e6(self):
        """Test that tolerance is clamped to 1e-6 to match mujoco_warp behavior.

        MuJoCo Warp clamps tolerance to 1e-6 for float32 precision (see mujoco_warp/_src/io.py).
        The update_solver_options_kernel must apply the same clamping when updating tolerance
        from Newton model custom attributes.
        """
        # Create a simple MJCF with tolerance set to 1e-8 (lower than the 1e-6 clamp)
        mjcf = """<?xml version="1.0" ?>
<mujoco model="tolerance_test">
  <option timestep="0.01" tolerance="1e-8" gravity="0 0 -9.81"/>

  <worldbody>
    <body name="box" pos="0 0 1">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

        # Build model with 2 worlds to test per-world clamping
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)

        scene_builder = newton.ModelBuilder()
        scene_builder.replicate(builder, 2)
        model = scene_builder.finalize()

        # Verify that the Newton model has the unclamped value (1e-8)
        self.assertTrue(hasattr(model, "mujoco"), "MuJoCo custom attributes not registered")
        self.assertTrue(hasattr(model.mujoco, "tolerance"), "Tolerance attribute not found")

        tolerance_values = model.mujoco.tolerance.numpy()
        self.assertEqual(len(tolerance_values), 2, "Expected 2 worlds")

        # Newton model should have the parsed value (1e-8 from MJCF)
        for i in range(2):
            self.assertAlmostEqual(
                tolerance_values[i],
                1e-8,
                places=12,
                msg=f"Newton model tolerance[{i}] should be 1e-8 (unclamped)",
            )

        # Create solver - this will call update_solver_options_kernel
        solver = SolverMuJoCo(model)

        # Verify that mjw_model.opt.tolerance is clamped to 1e-6
        mjw_tolerance_values = solver.mjw_model.opt.tolerance.numpy()

        for i in range(len(mjw_tolerance_values)):
            self.assertAlmostEqual(
                mjw_tolerance_values[i],
                1e-6,
                places=9,
                msg=f"MuJoCo Warp tolerance[{i}] should be clamped to 1e-6",
            )

    def test_tolerance_not_clamped_when_above_minimum(self):
        """Test that tolerance values above 1e-6 are not modified."""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="tolerance_test">
  <option timestep="0.01" tolerance="1e-5" gravity="0 0 -9.81"/>

  <worldbody>
    <body name="box" pos="0 0 1">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Verify Newton model has 1e-5
        tolerance_values = model.mujoco.tolerance.numpy()
        self.assertAlmostEqual(tolerance_values[0], 1e-5, places=9, msg="Newton model tolerance should be 1e-5")

        # Create solver
        solver = SolverMuJoCo(model)

        # Verify mjw_model.opt.tolerance is still 1e-5 (not clamped)
        mjw_tolerance = solver.mjw_model.opt.tolerance.numpy()[0]
        self.assertAlmostEqual(
            mjw_tolerance, 1e-5, places=9, msg="MuJoCo Warp tolerance should remain 1e-5 (not clamped)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
