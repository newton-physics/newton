# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Minkowski Portal Refinement (MPR)."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.geometry.mpr import MPR_BOX_SUPPORT_TIE_EPSILON, create_solve_mpr
from newton._src.geometry.support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    support_map,
)


@wp.kernel
def _triangle_mpr_kernel(
    triangle_b: wp.array[wp.vec3],
    triangle_c: wp.array[wp.vec3],
    shape_b_type: int,
    shape_b_scale: wp.vec3,
    shape_b_position: wp.array[wp.vec3],
    shape_b_orientation: wp.array[wp.quat],
    collision_out: wp.array[int],
    point_a_out: wp.array[wp.vec3],
    point_b_out: wp.array[wp.vec3],
    normal_out: wp.array[wp.vec3],
    penetration_out: wp.array[float],
):
    """Run MPR directly, with each triangle's first vertex at the origin."""
    i = wp.tid()

    shape_a = GenericShapeData()
    shape_a.shape_type = int(GeoTypeEx.TRIANGLE)
    shape_a.scale = triangle_b[i]
    shape_a.auxiliary = triangle_c[i]

    shape_b = GenericShapeData()
    shape_b.shape_type = shape_b_type
    shape_b.scale = shape_b_scale
    shape_b.auxiliary = wp.vec3(0.0)

    data_provider = SupportMapDataProvider()
    collision, point_a, point_b, normal, penetration = wp.static(create_solve_mpr(support_map).core)(
        shape_a,
        shape_b,
        shape_b_orientation[i],
        shape_b_position[i],
        0.0,
        data_provider,
    )

    collision_out[i] = int(collision)
    point_a_out[i] = point_a
    point_b_out[i] = point_b
    normal_out[i] = normal
    penetration_out[i] = penetration


@wp.kernel
def _box_mpr_kernel(
    shape_b_position: wp.array[wp.vec3],
    shape_b_orientation: wp.array[wp.quat],
    collision_out: wp.array[int],
    point_a_out: wp.array[wp.vec3],
    point_b_out: wp.array[wp.vec3],
    normal_out: wp.array[wp.vec3],
    penetration_out: wp.array[float],
):
    """Run MPR directly for two unit boxes."""
    i = wp.tid()

    shape_a = GenericShapeData()
    shape_a.shape_type = int(GeoType.BOX)
    shape_a.scale = wp.vec3(0.5)
    shape_a.auxiliary = wp.vec3(0.0)

    shape_b = GenericShapeData()
    shape_b.shape_type = int(GeoType.BOX)
    shape_b.scale = wp.vec3(0.5)
    shape_b.auxiliary = wp.vec3(0.0)

    data_provider = SupportMapDataProvider()
    collision, point_a, point_b, normal, penetration = wp.static(create_solve_mpr(support_map).core)(
        shape_a,
        shape_b,
        shape_b_orientation[i],
        shape_b_position[i],
        0.0,
        data_provider,
    )

    collision_out[i] = int(collision)
    point_a_out[i] = point_a
    point_b_out[i] = point_b
    normal_out[i] = normal
    penetration_out[i] = penetration


def _run_triangle_mpr(triangle_b, triangle_c, shape_type, shape_scale, shape_positions, shape_orientations):
    """Run direct triangle-vs-convex MPR cases on CPU and return NumPy outputs."""
    device = "cpu"
    count = len(triangle_b)

    triangle_b_wp = wp.array(np.asarray(triangle_b, dtype=np.float32), dtype=wp.vec3, device=device)
    triangle_c_wp = wp.array(np.asarray(triangle_c, dtype=np.float32), dtype=wp.vec3, device=device)
    shape_positions_wp = wp.array(np.asarray(shape_positions, dtype=np.float32), dtype=wp.vec3, device=device)
    shape_orientations_wp = wp.array(np.asarray(shape_orientations, dtype=np.float32), dtype=wp.quat, device=device)

    collision = wp.zeros(count, dtype=int, device=device)
    point_a = wp.zeros(count, dtype=wp.vec3, device=device)
    point_b = wp.zeros(count, dtype=wp.vec3, device=device)
    normal = wp.zeros(count, dtype=wp.vec3, device=device)
    penetration = wp.zeros(count, dtype=float, device=device)

    wp.launch(
        _triangle_mpr_kernel,
        dim=count,
        inputs=[
            triangle_b_wp,
            triangle_c_wp,
            int(shape_type),
            wp.vec3(*shape_scale),
            shape_positions_wp,
            shape_orientations_wp,
        ],
        outputs=[collision, point_a, point_b, normal, penetration],
        device=device,
    )

    return (
        collision.numpy(),
        point_a.numpy(),
        point_b.numpy(),
        normal.numpy(),
        penetration.numpy(),
    )


def _run_box_mpr(shape_positions, shape_orientations):
    """Run direct box-vs-box MPR cases on CPU and return NumPy outputs."""
    device = "cpu"
    count = len(shape_positions)

    shape_positions_wp = wp.array(np.asarray(shape_positions, dtype=np.float32), dtype=wp.vec3, device=device)
    shape_orientations_wp = wp.array(np.asarray(shape_orientations, dtype=np.float32), dtype=wp.quat, device=device)
    collision = wp.zeros(count, dtype=int, device=device)
    point_a = wp.zeros(count, dtype=wp.vec3, device=device)
    point_b = wp.zeros(count, dtype=wp.vec3, device=device)
    normal = wp.zeros(count, dtype=wp.vec3, device=device)
    penetration = wp.zeros(count, dtype=float, device=device)

    wp.launch(
        _box_mpr_kernel,
        dim=count,
        inputs=[shape_positions_wp, shape_orientations_wp],
        outputs=[collision, point_a, point_b, normal, penetration],
        device=device,
    )

    return collision.numpy(), point_a.numpy(), point_b.numpy(), normal.numpy(), penetration.numpy()


class TestMPRBoxSupportTie(unittest.TestCase):
    """Cover box support directions around the MPR tie threshold."""

    def test_support_tie_boundary_preserves_contact_witnesses(self):
        """Keep valid witnesses immediately below and above the box support tie threshold."""
        angles = MPR_BOX_SUPPORT_TIE_EPSILON * np.array([0.5, 2.0], dtype=np.float32)
        orientations = np.zeros((len(angles), 4), dtype=np.float32)
        orientations[:, 0] = np.sin(0.5 * angles)
        orientations[:, 3] = np.cos(0.5 * angles)
        positions = np.tile([0.0, 0.999, 0.0], (len(angles), 1))

        collision, points_a, points_b, normals, penetrations = _run_box_mpr(positions, orientations)

        expected_penetrations = 0.5 + 0.5 * (np.cos(angles) + np.sin(angles)) - positions[:, 1]
        witness_separations = np.sum((points_b - points_a) * normals, axis=1)

        np.testing.assert_array_equal(collision, np.ones(len(angles), dtype=np.int32))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-6)
        np.testing.assert_allclose(normals, np.tile([0.0, 1.0, 0.0], (len(angles), 1)), atol=1.0e-5)
        np.testing.assert_allclose(penetrations, expected_penetrations, atol=2.0e-5)
        np.testing.assert_allclose(witness_separations, -penetrations, atol=1.0e-6)


class TestMPRTriangleInitialization(unittest.TestCase):
    """Cover triangle MPR initialization across disparate geometry scales."""

    def test_small_cylinder_on_large_triangle(self):
        """Resolve a small cylinder on a large triangle along the face normal.

        A fixed 1% triangle-centroid blend moves the initial point by nearly
        half a meter in this case.  With the cylinder center only 0.36 mm above
        the face, that makes the MPR ray almost tangential and used to produce
        a 58.8 m penetration instead of the geometric 61.7 mm penetration.
        """
        cylinder_position = np.array([28.87991333, 260.53430176, 0.00036136055], dtype=np.float32)
        cylinder_orientation = np.array(
            [-0.16498479, 0.36682746, -0.48718625, 0.77515626],
            dtype=np.float32,
        )
        radius = 0.02
        half_height = 0.07

        collision, points_a, points_b, normals, penetrations = _run_triangle_mpr(
            triangle_b=[[100.0, 320.0, 0.0]],
            triangle_c=[[0.0, 320.0, 0.0]],
            shape_type=GeoType.CYLINDER,
            shape_scale=(radius, half_height, 0.0),
            shape_positions=[cylinder_position],
            shape_orientations=[cylinder_orientation],
        )

        # The cylinder axis is its rotated local Z axis.  Its support extent
        # along world Z is h*|axis_z| + r*sqrt(1-axis_z^2).
        qx, qy, _qz, _qw = cylinder_orientation
        axis_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        vertical_extent = half_height * abs(axis_z) + radius * np.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        expected_penetration = vertical_extent - cylinder_position[2]

        self.assertEqual(collision[0], 1)
        self.assertAlmostEqual(float(penetrations[0]), float(expected_penetration), delta=2.0e-5)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=2.0e-4)
        self.assertLess(float(np.linalg.norm(points_a[0, :2] - cylinder_position[:2])), 0.1)
        self.assertLess(float(np.linalg.norm(points_b[0] - cylinder_position)), 0.08)

    def test_coplanar_cylinder_center_on_large_triangle(self):
        """Handle an exactly coplanar convex center without a tangential roundoff ray."""
        cylinder_orientation = np.array(
            [-0.16498479, 0.36682746, -0.48718625, 0.77515626],
            dtype=np.float32,
        )
        radius = 0.02
        half_height = 0.07

        cylinder_position = np.array([28.87991333, 260.53430176, 0.0], dtype=np.float32)
        collision, points_a, points_b, normals, penetrations = _run_triangle_mpr(
            triangle_b=[[100.0, 320.0, 0.0]],
            triangle_c=[[0.0, 320.0, 0.0]],
            shape_type=GeoType.CYLINDER,
            shape_scale=(radius, half_height, 0.0),
            shape_positions=[cylinder_position],
            shape_orientations=[cylinder_orientation],
        )

        qx, qy, _qz, _qw = cylinder_orientation
        axis_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        expected_penetration = half_height * abs(axis_z) + radius * np.sqrt(max(0.0, 1.0 - axis_z * axis_z))

        self.assertEqual(collision[0], 1)
        self.assertAlmostEqual(float(penetrations[0]), float(expected_penetration), delta=2.0e-5)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=2.0e-4)
        self.assertLess(float(np.linalg.norm(points_a[0, :2] - cylinder_position[:2])), 0.1)
        self.assertLess(float(np.linalg.norm(points_b[0] - cylinder_position)), 0.08)

    def test_shared_edge_witnesses_remain_triangle_specific(self):
        """Retain distinct witnesses across a mesh seam with the bounded nudge.

        The two triangles form a square split along x=y, and the box center is
        directly above that shared edge.  The witness on each triangle should
        lie on its own side of the seam rather than both being edge-biased.
        """
        collision, points_a, _points_b, normals, penetrations = _run_triangle_mpr(
            triangle_b=[[2.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
            triangle_c=[[2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            shape_type=GeoType.BOX,
            shape_scale=(0.15, 0.12, 0.1),
            shape_positions=[[1.0, 1.0, 0.095], [1.0, 1.0, 0.095]],
            shape_orientations=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )

        np.testing.assert_array_equal(collision, [1, 1])
        self.assertGreater(float(points_a[0, 0] - points_a[0, 1]), 1.0e-4)
        self.assertGreater(float(points_a[1, 1] - points_a[1, 0]), 1.0e-4)
        np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1.0e-5)
        np.testing.assert_allclose(penetrations, [0.005, 0.005], atol=1.0e-5)

    def test_coplanar_shared_edge_witnesses_remain_triangle_specific(self):
        """Retain the triangle-specific fallback when both centers coincide."""
        collision, points_a, _points_b, normals, penetrations = _run_triangle_mpr(
            triangle_b=[[2.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
            triangle_c=[[2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            shape_type=GeoType.BOX,
            shape_scale=(0.15, 0.12, 0.1),
            shape_positions=[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            shape_orientations=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )

        np.testing.assert_array_equal(collision, [1, 1])
        self.assertGreater(float(points_a[0, 0] - points_a[0, 1]), 1.0e-4)
        self.assertGreater(float(points_a[1, 1] - points_a[1, 0]), 1.0e-4)
        np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1.0e-5)
        np.testing.assert_allclose(penetrations, [0.1, 0.1], atol=1.0e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
