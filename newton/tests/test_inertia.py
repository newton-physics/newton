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

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core import quat_between_axes
from newton._src.geometry.inertia import (
    compute_inertia_box,
    compute_inertia_capsule,
    compute_inertia_cone,
    compute_inertia_cylinder,
    compute_inertia_mesh,
    compute_inertia_shape,
    compute_inertia_sphere,
)
from newton._src.geometry.types import GeoType
from newton.tests.unittest_utils import assert_np_equal


class TestInertia(unittest.TestCase):
    def test_cube_mesh_inertia(self):
        # Unit cube
        vertices = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
        indices = [
            [1, 2, 3],
            [7, 6, 5],
            [4, 5, 1],
            [5, 6, 2],
            [2, 6, 7],
            [0, 3, 7],
            [0, 1, 3],
            [4, 7, 5],
            [0, 4, 1],
            [1, 5, 2],
            [3, 2, 7],
            [4, 0, 7],
        ]

        mass_0, com_0, I_0, volume_0 = compute_inertia_mesh(
            density=1000, vertices=vertices, indices=indices, is_solid=True
        )

        self.assertAlmostEqual(mass_0, 1000.0, delta=1e-6)
        self.assertAlmostEqual(volume_0, 1.0, delta=1e-6)
        assert_np_equal(np.array(com_0), np.array([0.5, 0.5, 0.5]), tol=1e-6)

        # Check against analytical inertia (unit cube has half-extents 0.5)
        mass_box, com_box, I_box = compute_inertia_box(1000.0, 0.5, 0.5, 0.5)
        self.assertAlmostEqual(mass_box, mass_0, delta=1e-6)
        assert_np_equal(np.array(com_box), np.zeros(3), tol=1e-6)
        assert_np_equal(np.array(I_0), np.array(I_box), tol=1e-4)

        # Compute hollow box inertia
        mass_0_hollow, com_0_hollow, I_0_hollow, volume_0_hollow = compute_inertia_mesh(
            density=1000,
            vertices=vertices,
            indices=indices,
            is_solid=False,
            thickness=0.1,
        )
        assert_np_equal(np.array(com_0_hollow), np.array([0.5, 0.5, 0.5]), tol=1e-6)

        # Add vertex between [0.0, 0.0, 0.0] and [1.0, 0.0, 0.0]
        vertices.append([0.5, 0.0, 0.0])
        indices[5] = [0, 8, 7]
        indices.append([8, 3, 7])
        indices[6] = [0, 1, 8]
        indices.append([8, 1, 3])

        mass_1, com_1, I_1, volume_1 = compute_inertia_mesh(
            density=1000, vertices=vertices, indices=indices, is_solid=True
        )

        # Inertia values should be the same as before
        self.assertAlmostEqual(mass_1, mass_0, delta=1e-6)
        self.assertAlmostEqual(volume_1, volume_0, delta=1e-6)
        assert_np_equal(np.array(com_1), np.array([0.5, 0.5, 0.5]), tol=1e-6)
        assert_np_equal(np.array(I_1), np.array(I_0), tol=1e-4)

        # Compute hollow box inertia
        mass_1_hollow, com_1_hollow, I_1_hollow, volume_1_hollow = compute_inertia_mesh(
            density=1000,
            vertices=vertices,
            indices=indices,
            is_solid=False,
            thickness=0.1,
        )

        # Inertia values should be the same as before
        self.assertAlmostEqual(mass_1_hollow, mass_0_hollow, delta=2e-3)
        self.assertAlmostEqual(volume_1_hollow, volume_0_hollow, delta=1e-6)
        assert_np_equal(np.array(com_1_hollow), np.array([0.5, 0.5, 0.5]), tol=1e-6)
        assert_np_equal(np.array(I_1_hollow), np.array(I_0_hollow), tol=1e-4)

    def test_sphere_mesh_inertia(self):
        mesh = newton.Mesh.create_sphere(
            radius=2.5,
            num_latitudes=500,
            num_longitudes=500,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

        offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vertices = mesh.vertices + offset

        mass_mesh, com_mesh, I_mesh, vol_mesh = compute_inertia_mesh(
            density=1000,
            vertices=vertices,
            indices=mesh.indices,
            is_solid=True,
        )

        # Check against analytical inertia
        mass_sphere, _, I_sphere = compute_inertia_sphere(1000.0, 2.5)
        self.assertAlmostEqual(mass_mesh, mass_sphere, delta=1e2)
        assert_np_equal(np.array(com_mesh), np.array(offset), tol=2e-3)
        assert_np_equal(np.array(I_mesh), np.array(I_sphere), tol=4e2)
        # Check volume
        self.assertAlmostEqual(vol_mesh, 4.0 / 3.0 * np.pi * 2.5**3, delta=3e-2)

    def test_body_inertia(self):
        mesh = newton.Mesh.create_sphere(
            radius=2.5,
            num_latitudes=500,
            num_longitudes=500,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

        offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vertices = mesh.vertices + offset

        builder = newton.ModelBuilder()
        b = builder.add_body()
        tf = wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_rpy(0.5, -0.8, 1.3))
        builder.add_shape_mesh(
            b,
            xform=tf,
            mesh=newton.Mesh(vertices=vertices, indices=mesh.indices),
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
        )
        transformed_com = wp.transform_point(tf, wp.vec3(*offset))
        assert_np_equal(np.array(builder.body_com[0]), np.array(transformed_com), tol=3e-3)
        mass_sphere, _, I_sphere = compute_inertia_sphere(1000.0, 2.5)
        assert_np_equal(np.array(builder.body_inertia[0]), np.array(I_sphere), tol=4e2)
        self.assertAlmostEqual(builder.body_mass[0], mass_sphere, delta=1e2)

    def test_capsule_cylinder_cone_axis_inertia(self):
        """Test that capsules, cylinders, and cones have correct inertia for different axis orientations."""
        # Test parameters
        radius = 0.5
        half_height = 1.0
        density = 1000.0

        # Test capsule inertia for different axes
        # Z-axis capsule (default)
        builder_z = newton.ModelBuilder()
        body_z = builder_z.add_body()
        builder_z.add_shape_capsule(
            body=body_z,
            radius=radius,
            half_height=half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=density),
        )
        model_z = builder_z.finalize()
        I_z = model_z.body_inertia.numpy()[0]

        # For Z-axis aligned capsule, I_xx should equal I_yy, and I_zz should be different
        self.assertAlmostEqual(I_z[0, 0], I_z[1, 1], delta=1e-6, msg="I_xx should equal I_yy for Z-axis capsule")
        self.assertNotAlmostEqual(I_z[0, 0], I_z[2, 2], delta=1e-3, msg="I_xx should not equal I_zz for Z-axis capsule")

        # Y-axis capsule
        builder_y = newton.ModelBuilder()
        body_y = builder_y.add_body()
        # Apply Y-axis rotation
        xform = wp.transform(wp.vec3(), quat_between_axes(newton.Axis.Z, newton.Axis.Y))
        builder_y.add_shape_capsule(
            body=body_y,
            xform=xform,
            radius=radius,
            half_height=half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=density),
        )
        model_y = builder_y.finalize()
        I_y = model_y.body_inertia.numpy()[0]

        # For Y-axis aligned capsule, I_xx should equal I_zz, and I_yy should be different
        self.assertAlmostEqual(I_y[0, 0], I_y[2, 2], delta=1e-6, msg="I_xx should equal I_zz for Y-axis capsule")
        self.assertNotAlmostEqual(I_y[0, 0], I_y[1, 1], delta=1e-3, msg="I_xx should not equal I_yy for Y-axis capsule")

        # X-axis capsule
        builder_x = newton.ModelBuilder()
        body_x = builder_x.add_body()
        # Apply X-axis rotation
        xform = wp.transform(wp.vec3(), quat_between_axes(newton.Axis.Z, newton.Axis.X))
        builder_x.add_shape_capsule(
            body=body_x,
            xform=xform,
            radius=radius,
            half_height=half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=density),
        )
        model_x = builder_x.finalize()
        I_x = model_x.body_inertia.numpy()[0]

        # For X-axis aligned capsule, I_yy should equal I_zz, and I_xx should be different
        self.assertAlmostEqual(I_x[1, 1], I_x[2, 2], delta=1e-6, msg="I_yy should equal I_zz for X-axis capsule")
        self.assertNotAlmostEqual(I_x[0, 0], I_x[1, 1], delta=1e-3, msg="I_xx should not equal I_yy for X-axis capsule")

        # Test cylinder inertia for Z-axis
        builder_cyl = newton.ModelBuilder()
        body_cyl = builder_cyl.add_body()
        builder_cyl.add_shape_cylinder(
            body=body_cyl,
            radius=radius,
            half_height=half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=density),
        )
        model_cyl = builder_cyl.finalize()
        I_cyl = model_cyl.body_inertia.numpy()[0]

        self.assertAlmostEqual(I_cyl[0, 0], I_cyl[1, 1], delta=1e-6, msg="I_xx should equal I_yy for Z-axis cylinder")
        self.assertNotAlmostEqual(
            I_cyl[0, 0], I_cyl[2, 2], delta=1e-3, msg="I_xx should not equal I_zz for Z-axis cylinder"
        )

        # Test cone inertia for Z-axis
        builder_cone = newton.ModelBuilder()
        body_cone = builder_cone.add_body()
        builder_cone.add_shape_cone(
            body=body_cone,
            radius=radius,
            half_height=half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=density),
        )
        model_cone = builder_cone.finalize()
        I_cone = model_cone.body_inertia.numpy()[0]

        self.assertAlmostEqual(I_cone[0, 0], I_cone[1, 1], delta=1e-6, msg="I_xx should equal I_yy for Z-axis cone")
        self.assertNotAlmostEqual(
            I_cone[0, 0], I_cone[2, 2], delta=1e-3, msg="I_xx should not equal I_zz for Z-axis cone"
        )

    @staticmethod
    def _create_cone_mesh(radius, half_height, num_segments=500):
        """Create a cone mesh with apex at +half_height and base at -half_height."""
        vertices = [[0, 0, half_height], [0, 0, -half_height]]
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), -half_height])
        indices = []
        for i in range(num_segments):
            ni = (i + 1) % num_segments
            indices.append([0, i + 2, ni + 2])
            indices.append([1, ni + 2, i + 2])
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    def test_cone_mesh_inertia(self):
        """Test cone inertia by comparing analytical formula with mesh computation."""

        # Test parameters
        radius = 2.5
        half_height = 3.0
        density = 1000.0

        # Create high-resolution cone mesh
        vertices, indices = self._create_cone_mesh(radius, half_height, num_segments=500)

        # Compute mesh inertia
        mass_mesh, com_mesh, I_mesh, vol_mesh = compute_inertia_mesh(
            density=density,
            vertices=vertices,
            indices=indices,
            is_solid=True,
        )

        # Compute analytical inertia
        mass_cone, com_cone, I_cone = compute_inertia_cone(density, radius, half_height)

        # Check mass (within 0.1%)
        self.assertAlmostEqual(mass_mesh, mass_cone, delta=mass_cone * 0.001)

        # Check COM (cone COM is at -half_height/2 from center)
        assert_np_equal(np.array(com_mesh), np.array(com_cone), tol=1e-3)

        # Check inertia (within 0.1%)
        assert_np_equal(np.array(I_mesh), np.array(I_cone), tol=I_cone[0, 0] * 0.001)

        # Check volume
        vol_cone = np.pi * radius**2 * (2 * half_height) / 3
        self.assertAlmostEqual(vol_mesh, vol_cone, delta=vol_cone * 0.001)

    def test_compute_inertia_shape_dispatcher(self):
        """Test compute_inertia_shape for primitive shapes against analytical formulas.

        Validates that the scale/half-extent conventions are consistently threaded
        through the dispatcher (e.g. no erroneous factor-of-2 doubling).
        """
        density = 1000.0

        # BOX: unit cube, half-extents = 0.5 → mass = 8 * 0.5^3 * 1000 = 1000 kg
        m, com, I = compute_inertia_shape(GeoType.BOX, wp.vec3(0.5, 0.5, 0.5), None, density)
        self.assertAlmostEqual(m, 1000.0, delta=1e-6)
        assert_np_equal(np.array(com), np.zeros(3), tol=1e-6)
        expected_I_box = 1.0 / 3.0 * 1000.0 * (0.25 + 0.25)  # 1/3 * m * (hy² + hz²) = 166.667
        self.assertAlmostEqual(float(I[0, 0]), expected_I_box, delta=1e-3)
        self.assertAlmostEqual(float(I[1, 1]), expected_I_box, delta=1e-3)
        self.assertAlmostEqual(float(I[2, 2]), expected_I_box, delta=1e-3)

        # SPHERE: radius = 1.0 → mass = 4/3 * pi * 1000 ≈ 4188.79 kg
        radius = 1.0
        m, com, I = compute_inertia_shape(GeoType.SPHERE, wp.vec3(radius, 0.0, 0.0), None, density)
        mass_ref, _, I_ref = compute_inertia_sphere(density, radius)
        self.assertAlmostEqual(m, mass_ref, delta=1e-6)
        assert_np_equal(np.array(I), np.array(I_ref), tol=1e-6)

        # CAPSULE: radius=0.5, half_height=1.0 → check axis symmetry and exact match
        m, com, I = compute_inertia_shape(GeoType.CAPSULE, wp.vec3(0.5, 1.0, 0.0), None, density)
        mass_ref, _, I_ref = compute_inertia_capsule(density, 0.5, 1.0)
        self.assertAlmostEqual(m, mass_ref, delta=1e-6)
        assert_np_equal(np.array(I), np.array(I_ref), tol=1e-6)

        # CYLINDER: radius=0.5, half_height=1.0
        m, com, I = compute_inertia_shape(GeoType.CYLINDER, wp.vec3(0.5, 1.0, 0.0), None, density)
        mass_ref, _, I_ref = compute_inertia_cylinder(density, 0.5, 1.0)
        self.assertAlmostEqual(m, mass_ref, delta=1e-6)
        assert_np_equal(np.array(I), np.array(I_ref), tol=1e-6)

        # CONE: radius=0.5, half_height=1.0
        m, com, I = compute_inertia_shape(GeoType.CONE, wp.vec3(0.5, 1.0, 0.0), None, density)
        mass_ref, com_ref, I_ref = compute_inertia_cone(density, 0.5, 1.0)
        self.assertAlmostEqual(m, mass_ref, delta=1e-6)
        assert_np_equal(np.array(com), np.array(com_ref), tol=1e-6)
        assert_np_equal(np.array(I), np.array(I_ref), tol=1e-6)

    def test_hollow_cone_inertia(self):
        """Test hollow cone inertia via compute_inertia_shape against mesh subtraction.

        The hollow cone has a non-zero COM, so outer and inner cones have
        different COMs and the inertia tensors must be shifted (parallel-axis
        theorem) before subtraction.
        """

        density = 1000.0
        outer_radius = 1.0
        outer_half_height = 2.0
        thickness = 0.1

        # Analytical hollow cone via compute_inertia_shape
        scale = wp.vec3(outer_radius, outer_half_height, 0.0)
        m_an, com_an, I_an = compute_inertia_shape(
            GeoType.CONE, scale, None, density, is_solid=False, thickness=thickness
        )

        # Reference: mesh subtraction with proper parallel-axis shifts
        inner_radius = outer_radius - thickness
        inner_half_height = outer_half_height - thickness
        v_out, i_out = self._create_cone_mesh(outer_radius, outer_half_height)
        v_in, i_in = self._create_cone_mesh(inner_radius, inner_half_height)
        m_out, com_out, I_out, _ = compute_inertia_mesh(density, v_out, i_out, is_solid=True)
        m_in, com_in, I_in, _ = compute_inertia_mesh(density, v_in, i_in, is_solid=True)
        m_ref = m_out - m_in
        com_ref = (m_out * np.array(com_out) - m_in * np.array(com_in)) / m_ref

        def _shift(mass, I_mat, com_f, com_t):
            d = np.array(com_t) - np.array(com_f)
            return np.array(I_mat).reshape(3, 3) + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        I_ref = _shift(m_out, I_out, com_out, com_ref) - _shift(m_in, I_in, com_in, com_ref)

        tol = 0.01  # 1% relative tolerance
        self.assertAlmostEqual(m_an, m_ref, delta=tol * abs(m_ref))
        assert_np_equal(np.array(com_an), com_ref, tol=1e-3)
        I_an_np = np.array(I_an).reshape(3, 3)
        # Check each diagonal with its own element-specific scale
        for i in range(3):
            self.assertAlmostEqual(I_an_np[i, i], I_ref[i, i], delta=tol * abs(I_ref[i, i]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
