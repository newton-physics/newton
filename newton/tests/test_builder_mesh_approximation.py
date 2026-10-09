# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test mesh-wide limits during convex decomposition."""

import types
import unittest

import numpy as np

import newton
from newton._src.geometry.utils import remesh_convex_hull
from newton.tests.unittest_utils import patch_sys_module


class TestMeshApproximationLimits(unittest.TestCase):
    def test_merged_hull_vertex_limits(self):
        """Preserve default, explicit, and disabled decimation limits after merging."""
        # Two interleaved sets of points on one sphere yield disconnected hulls
        # whose union has more vertices than either component's decimation cap.
        indices = np.arange(400)
        z = 1.0 - 2.0 * (indices + 0.5) / len(indices)
        angle = indices * (np.pi * (3.0 - np.sqrt(5.0)))
        radius = np.sqrt(1.0 - z * z)
        points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
        first, first_faces = remesh_convex_hull(points[::2])
        second, second_faces = remesh_convex_hull(points[1::2])
        vertices = np.concatenate((first, second))
        faces = np.concatenate((first_faces, second_faces + len(first)))
        calls = []
        fake_coacd = types.ModuleType("coacd")
        fake_coacd.Mesh = lambda vertices, indices: (vertices, indices)

        def run_coacd(mesh, **kwargs):
            calls.append(kwargs)
            return [mesh]

        fake_coacd.run_coacd = run_coacd
        for settings, expected_limit in (
            ({"decimate": True}, 256),
            ({"decimate": True, "max_ch_vertex": 64}, 64),
            ({"decimate": False}, None),
        ):
            with self.subTest(settings=settings):
                calls.clear()
                builder = newton.ModelBuilder()
                shape = builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, faces.flatten(), compute_inertia=False))
                with (
                    patch_sys_module("coacd", fake_coacd),
                    self.assertWarnsRegex(UserWarning, "merging the nearest hulls"),
                ):
                    builder.approximate_meshes(
                        method="coacd",
                        shape_indices=[shape],
                        raise_on_failure=True,
                        merge=True,
                        max_convex_hull=1,
                        **settings,
                    )
                self.assertEqual(len(calls), 2)
                self.assertEqual(builder.shape_count, 1)
                vertex_count = len(builder.shape_source[shape].vertices)
                if expected_limit is None:
                    self.assertGreater(vertex_count, 256)
                else:
                    self.assertLessEqual(vertex_count, expected_limit)


if __name__ == "__main__":
    unittest.main()
