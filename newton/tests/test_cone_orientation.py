# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""Unit tests for cone shape orientation and properties."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core import quat_between_axes
from newton._src.geometry import kernels


class TestConeOrientation(unittest.TestCase):
    """Test cone shape implementation with apex pointing positive."""

    def setUp(self):
        """Set up test parameters."""
        self.radius = 1.0
        self.half_height = 2.0  # Use different height to avoid special case where Ia = Ib
        self.density = 1000.0

    def test_cone_com_position(self):
        """Test that cone COM is at -h/4 from center (1/4 from base toward apex)."""
        builder = newton.ModelBuilder()
        body_id = builder.add_body()
        builder.add_shape_cone(
            body=body_id,
            radius=self.radius,
            half_height=self.half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=self.density),
        )

        model = builder.finalize()
        com = model.body_com.numpy()[0]

        # COM should be at -half_height/2 (1/4 of total height from base at -half_height)
        expected_com_z = -self.half_height / 2.0

        self.assertAlmostEqual(com[0], 0.0, places=6, msg="COM X should be 0")
        self.assertAlmostEqual(com[1], 0.0, places=6, msg="COM Y should be 0")
        self.assertAlmostEqual(
            com[2], expected_com_z, places=6, msg=f"COM Z should be {expected_com_z} (1/4 from base toward apex)"
        )

    def test_cone_sdf_values(self):
        """Test cone SDF values at key points."""

        @wp.kernel
        def compute_sdf_kernel(
            points: wp.array(dtype=wp.vec3), sdf_values: wp.array(dtype=float), radius: float, half_height: float
        ):
            tid = wp.tid()
            p = points[tid]
            sdf_values[tid] = kernels.sdf_cone(p, radius, half_height, int(newton.Axis.Z))

        # Test points with expected SDF values
        test_cases = [
            # (point, expected_sdf, description)
            ((0, 0, self.half_height), 0.0, "Apex"),
            ((0, 0, -self.half_height), 0.0, "Base center"),
            ((self.radius, 0, -self.half_height), 0.0, "Base edge"),
            ((0, 0, 0), -0.48507124, "Origin (inside)"),
            ((self.radius / 2, 0, 0), 0.0, "Mid-height edge"),
        ]

        points = [tc[0] for tc in test_cases]
        wp_points = wp.array(points, dtype=wp.vec3)
        wp_sdf_values = wp.zeros(len(points), dtype=float)

        wp.launch(compute_sdf_kernel, dim=len(points), inputs=[wp_points, wp_sdf_values, self.radius, self.half_height])

        sdf_values = wp_sdf_values.numpy()

        for i, (point, expected, desc) in enumerate(test_cases):
            with self.subTest(description=desc, point=point):
                self.assertAlmostEqual(
                    sdf_values[i], expected, places=5, msg=f"{desc}: SDF at {point} should be {expected}"
                )

    def test_cone_orientation_consistency(self):
        """Test cone orientation is consistent for different axes."""
        com_offset = -self.half_height / 2.0
        for axis_name, axis_enum, expected_com in [
            ("X", newton.Axis.X, (com_offset, 0, 0)),
            ("Y", newton.Axis.Y, (0, com_offset, 0)),
            ("Z", newton.Axis.Z, (0, 0, com_offset)),
        ]:
            with self.subTest(axis=axis_name):
                builder = newton.ModelBuilder()
                body_id = builder.add_body()
                # Apply axis rotation to transform
                xform = wp.transform(wp.vec3(), quat_between_axes(newton.Axis.Z, axis_enum))
                builder.add_shape_cone(
                    body=body_id,
                    xform=xform,
                    radius=self.radius,
                    half_height=self.half_height,
                    cfg=newton.ModelBuilder.ShapeConfig(density=self.density),
                )

                model = builder.finalize()
                com = model.body_com.numpy()[0]

                # COM should be at -half_height/2 along the specified axis
                np.testing.assert_array_almost_equal(
                    com, expected_com, decimal=5, err_msg=f"COM for {axis_name}-axis cone should be {expected_com}"
                )

    def test_cone_mass_calculation(self):
        """Test that cone mass calculation is correct."""
        builder = newton.ModelBuilder()
        body_id = builder.add_body()
        builder.add_shape_cone(
            body=body_id,
            radius=self.radius,
            half_height=self.half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=self.density),
        )

        model = builder.finalize()
        mass = model.body_mass.numpy()[0]

        # Expected mass: density * pi * r^2 * h / 3
        # where h = 2 * half_height
        expected_mass = self.density * np.pi * self.radius**2 * (2 * self.half_height) / 3.0

        self.assertAlmostEqual(mass, expected_mass, places=3, msg=f"Mass should be {expected_mass:.3f}")

    def test_cone_inertia_symmetry(self):
        """Test that cone inertia tensor has correct symmetry."""
        builder = newton.ModelBuilder()
        body_id = builder.add_body()
        builder.add_shape_cone(
            body=body_id,
            radius=self.radius,
            half_height=self.half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=self.density),
        )

        model = builder.finalize()
        inertia = model.body_inertia.numpy()[0]

        # For Z-axis cone, I_xx should equal I_yy
        self.assertAlmostEqual(inertia[0, 0], inertia[1, 1], places=5, msg="I_xx should equal I_yy for Z-axis cone")

        # I_zz should be different from I_xx
        self.assertNotAlmostEqual(
            inertia[0, 0], inertia[2, 2], places=2, msg="I_xx should not equal I_zz for Z-axis cone"
        )

        # Off-diagonal elements should be zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertAlmostEqual(
                        inertia[i, j], 0.0, places=6, msg=f"Off-diagonal element I[{i},{j}] should be zero"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
