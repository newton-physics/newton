# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh quality validation utilities."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.utils.mesh import validate_tet_mesh, validate_triangle_mesh
from newton.utils import validate_tet_mesh as public_validate_tet_mesh
from newton.utils import validate_triangle_mesh as public_validate_triangle_mesh


def _equilateral_triangle(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a single equilateral triangle (vertices, indices)."""
    h = scale * np.sqrt(3) / 2
    verts = np.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [scale / 2, h, 0.0]])
    indices = np.array([[0, 1, 2]])
    return verts, indices


def _regular_tet(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a single regular tetrahedron (vertices, indices)."""
    s = scale
    verts = np.array(
        [
            [1, 1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, 1],
        ],
        dtype=float,
    ) * (s / 2)
    indices = np.array([[0, 1, 2, 3]])
    return verts, indices


class TestValidateTriangleMesh(unittest.TestCase):
    def test_clean_mesh_no_warning(self):
        verts, inds = _equilateral_triangle(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 0)

    def test_empty_mesh_warns(self):
        verts = np.zeros((0, 3))
        inds = np.zeros((0, 3), dtype=int)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("no triangles", str(w[0].message))

    def test_small_area(self):
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-4, 0.0, 0.0],
                [0.5e-4, 1e-4, 0.0],
            ]
        )
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("area <", str(w[0].message))

    def test_sliver_triangle(self):
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1e-3, 0.0],
            ]
        )
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("sliver", str(w[0].message))
        self.assertIn("aspect ratio >", str(w[0].message))

    def test_extreme_angle(self):
        angle_rad = np.radians(2.0)
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.1, np.tan(angle_rad) * 0.1, 0.0],
            ]
        )
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("minimum angle < 5", msgs)

    def test_non_manifold_edge(self):
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ]
        )
        inds = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("non-manifold", msgs)

    def test_borderline_pass(self):
        area_target = 2e-6
        h = 2 * area_target / 0.01
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.005, h, 0.0],
            ]
        )
        inds = np.array([[0, 1, 2]])
        actual_area = 0.5 * 0.01 * h
        self.assertGreater(actual_area, 1e-6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        quality_warnings = [wi for wi in w if "area <" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 0)

    def test_mixed_issues(self):
        good_verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.05, 0.08, 0.0],
            ]
        )
        tiny_verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-4, 0.0, 0.0],
                [0.5e-4, 1e-4, 0.0],
            ]
        )
        sliver_verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1e-3, 0.0],
            ]
        )
        offset1 = np.array([2, 0, 0])
        offset2 = np.array([4, 0, 0])
        verts = np.vstack([good_verts, tiny_verts + offset1, sliver_verts + offset2])
        inds = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        msg = str(w[0].message)
        self.assertIn("area <", msg)
        self.assertIn("sliver", msg)

    def test_custom_thresholds(self):
        verts, inds = _equilateral_triangle(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds, min_area=1.0)
        self.assertEqual(len(w), 1)
        self.assertIn("area <", str(w[0].message))

    def test_remesh_suggestion(self):
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-4, 0.0, 0.0],
                [0.5e-4, 1e-4, 0.0],
            ]
        )
        inds = np.array([[0, 1, 2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertIn("Consider remeshing", str(w[0].message))


class TestValidateTetMesh(unittest.TestCase):
    def test_clean_mesh_no_warning(self):
        verts, inds = _regular_tet(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 0)

    def test_empty_mesh_warns(self):
        verts = np.zeros((0, 3))
        inds = np.zeros((0, 4), dtype=int)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("no tetrahedra", str(w[0].message))

    def test_inverted_tet(self):
        verts, inds = _regular_tet(scale=0.1)
        inds[0, 2], inds[0, 3] = inds[0, 3], inds[0, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("inverted", str(w[0].message))

    def test_small_volume(self):
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-3, 0.0, 0.0],
                [0.0, 1e-3, 0.0],
                [0.0, 0.0, 1e-3],
            ]
        )
        inds = np.array([[0, 1, 2, 3]])
        vol = abs(np.dot(verts[1] - verts[0], np.cross(verts[2] - verts[0], verts[3] - verts[0]))) / 6
        self.assertLess(vol, 1e-9)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("volume <", msgs)

    def test_flat_sliver(self):
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1e-6],
            ]
        )
        inds = np.array([[0, 1, 2, 3]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("sliver", msgs)

    def test_needle_sliver(self):
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.01, 0.01, 0.0],
                [0.005, 0.005, 0.01],
            ]
        )
        inds = np.array([[0, 1, 2, 3]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("sliver", msgs)

    def test_non_manifold_face(self):
        base_verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.3, 1.0],
                [0.5, 0.3, -1.0],
                [0.5, 0.3, 2.0],
            ]
        )
        inds = np.array(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 4],
                [0, 1, 2, 5],
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(base_verts, inds)
        msgs = " ".join(str(wi.message) for wi in w)
        self.assertIn("non-manifold", msgs)

    def test_custom_thresholds(self):
        verts, inds = _regular_tet(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds, min_eta=1.1)
        self.assertEqual(len(w), 1)
        self.assertIn("sliver", str(w[0].message))

    def test_no_remesh_suggestion_in_tet_warning(self):
        verts, inds = _regular_tet(scale=0.1)
        inds[0, 2], inds[0, 3] = inds[0, 3], inds[0, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertNotIn("remeshing", str(w[0].message))


class TestIndexValidation(unittest.TestCase):
    def test_triangle_bad_index_count(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        inds = np.array([0, 1], dtype=int)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("multiple of 3", str(w[0].message))

    def test_triangle_out_of_range(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        inds = np.array([[0, 1, 5]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("out of range", str(w[0].message))

    def test_triangle_negative_index(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        inds = np.array([[0, 1, -1]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("out of range", str(w[0].message))

    def test_tet_bad_index_count(self):
        verts, _ = _regular_tet(scale=0.1)
        inds = np.array([0, 1, 2], dtype=int)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("multiple of 4", str(w[0].message))

    def test_tet_out_of_range(self):
        verts, _ = _regular_tet(scale=0.1)
        inds = np.array([[0, 1, 2, 99]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("out of range", str(w[0].message))

    def test_tet_negative_index(self):
        verts, _ = _regular_tet(scale=0.1)
        inds = np.array([[0, 1, 2, -1]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tet_mesh(verts, inds)
        self.assertEqual(len(w), 1)
        self.assertIn("out of range", str(w[0].message))


class TestPublicExport(unittest.TestCase):
    def test_importable_from_newton_utils(self):
        self.assertIs(public_validate_triangle_mesh, validate_triangle_mesh)
        self.assertIs(public_validate_tet_mesh, validate_tet_mesh)


class TestBuilderIntegration(unittest.TestCase):
    def test_add_cloth_mesh_default_no_warning(self):
        builder = newton.ModelBuilder()
        verts, inds = _equilateral_triangle(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=inds.flatten().tolist(),
                density=100.0,
            )
        quality_warnings = [wi for wi in w if "Mesh quality" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 0)

    def test_add_cloth_mesh_validate_bad_mesh(self):
        builder = newton.ModelBuilder()
        verts = [[0, 0, 0], [1e-4, 0, 0], [0.5e-4, 1e-4, 0]]
        inds = [0, 1, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts,
                indices=inds,
                density=100.0,
                validate_mesh=True,
            )
        quality_warnings = [wi for wi in w if "Mesh quality" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 1)

    def test_add_cloth_mesh_validate_good_mesh(self):
        builder = newton.ModelBuilder()
        verts, inds = _equilateral_triangle(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=inds.flatten().tolist(),
                density=100.0,
                validate_mesh=True,
            )
        quality_warnings = [wi for wi in w if "Mesh quality" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 0)

    def test_add_soft_mesh_default_no_warning(self):
        builder = newton.ModelBuilder()
        verts, inds = _regular_tet(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_soft_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=inds.flatten().tolist(),
                density=1000.0,
            )
        quality_warnings = [wi for wi in w if "Tet mesh quality" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 0)

    def test_add_soft_mesh_validate_bad_mesh(self):
        builder = newton.ModelBuilder()
        verts, inds = _regular_tet(scale=0.1)
        inds[0, 2], inds[0, 3] = inds[0, 3], inds[0, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_soft_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=inds.flatten().tolist(),
                density=1000.0,
                validate_mesh=True,
            )
        quality_warnings = [wi for wi in w if "Tet mesh quality" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 1)

    def test_add_soft_mesh_validate_good_mesh(self):
        builder = newton.ModelBuilder()
        verts, inds = _regular_tet(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_soft_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=inds.flatten().tolist(),
                density=1000.0,
                validate_mesh=True,
            )
        quality_warnings = [wi for wi in w if "Tet mesh quality" in str(wi.message)]
        self.assertEqual(len(quality_warnings), 0)

    def test_add_cloth_mesh_validate_bad_index_count_warns_without_raising(self):
        builder = newton.ModelBuilder()
        verts, _ = _equilateral_triangle(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=[0, 1],
                density=100.0,
                validate_mesh=True,
            )
        self.assertTrue(any("multiple of 3" in str(wi.message) for wi in w))

    def test_add_soft_mesh_validate_bad_index_count_warns_without_raising(self):
        builder = newton.ModelBuilder()
        verts, _ = _regular_tet(scale=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_soft_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=[0, 1, 2],
                density=1000.0,
                validate_mesh=True,
            )
        self.assertTrue(any("multiple of 4" in str(wi.message) for wi in w))


class TestDeformableRegistry(unittest.TestCase):
    def _add_cloth(self, builder, scale=0.1, label=None):
        verts, inds = _equilateral_triangle(scale=scale)
        builder.add_cloth_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0, 0, 0),
            vertices=verts.tolist(),
            indices=inds.flatten().tolist(),
            density=100.0,
            label=label,
        )

    def _add_soft(self, builder, scale=0.1, label=None):
        verts, inds = _regular_tet(scale=scale)
        builder.add_soft_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0, 0, 0),
            vertices=verts.tolist(),
            indices=inds.flatten().tolist(),
            density=1000.0,
            label=label,
        )

    def test_initial_state(self):
        builder = newton.ModelBuilder()
        self.assertEqual(builder.deformable_count, 0)
        self.assertEqual(builder.deformable_label, [])
        self.assertEqual(builder.deformable_offset, [0])

    def test_default_labels_global_id(self):
        builder = newton.ModelBuilder()
        self._add_cloth(builder)
        self._add_soft(builder)
        self._add_cloth(builder)
        # Single global counter, kind suffix from the source function.
        self.assertEqual(
            builder.deformable_label,
            ["cloth_mesh_0", "soft_mesh_1", "cloth_mesh_2"],
        )
        self.assertEqual(builder.deformable_count, 3)

    def test_custom_label_overrides_default(self):
        builder = newton.ModelBuilder()
        self._add_cloth(builder, label="left_panel")
        self._add_cloth(builder)
        self.assertEqual(builder.deformable_label, ["left_panel", "cloth_mesh_1"])

    def test_offsets_partition_particle_range(self):
        builder = newton.ModelBuilder()
        self._add_cloth(builder)
        cloth_end = builder.particle_count
        self._add_soft(builder)
        soft_end = builder.particle_count
        self.assertEqual(len(builder.deformable_offset), builder.deformable_count + 1)
        self.assertEqual(builder.deformable_offset[0], 0)
        self.assertEqual(builder.deformable_offset[1], cloth_end)
        self.assertEqual(builder.deformable_offset[2], soft_end)

    def test_offsets_initialized_to_first_deformable_start(self):
        builder = newton.ModelBuilder()
        # Add free particles before any deformable; first deformable should start past them.
        builder.add_particle(wp.vec3(0, 0, 0), wp.vec3(0, 0, 0), 1.0)
        builder.add_particle(wp.vec3(0, 0, 0), wp.vec3(0, 0, 0), 1.0)
        self._add_cloth(builder)
        self.assertEqual(builder.deformable_offset[0], 2)
        self.assertEqual(builder.deformable_offset[1], builder.particle_count)

    def test_grid_label_uses_grid_suffix(self):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
        )
        self.assertEqual(builder.deformable_label, ["cloth_grid_0"])

    def test_soft_grid_label_uses_grid_suffix(self):
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1000.0,
            k_mu=1e4,
            k_lambda=1e4,
            k_damp=0.0,
        )
        self.assertEqual(builder.deformable_label, ["soft_grid_0"])

    def test_validate_mesh_warning_includes_label(self):
        builder = newton.ModelBuilder()
        bad = [[0, 0, 0], [1e-4, 0, 0], [0.5e-4, 1e-4, 0]]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=bad,
                indices=[0, 1, 2],
                density=100.0,
                validate_mesh=True,
                label="my_panel",
            )
        quality = [wi for wi in w if "Mesh quality" in str(wi.message)]
        self.assertEqual(len(quality), 1)
        self.assertIn("[my_panel]", str(quality[0].message))

    def test_validator_default_no_label_no_brackets(self):
        # Direct validator call with label=None must not produce stray "[None]".
        verts = np.array([[0, 0, 0], [1e-4, 0, 0], [0.5e-4, 1e-4, 0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_triangle_mesh(verts, np.array([[0, 1, 2]]))
        self.assertEqual(len(w), 1)
        self.assertNotIn("[", str(w[0].message))

    def test_finalize_propagates_label_and_offset_to_model(self):
        builder = newton.ModelBuilder()
        self._add_cloth(builder)
        self._add_soft(builder)
        model = builder.finalize()
        self.assertEqual(model.deformable_label, ["cloth_mesh_0", "soft_mesh_1"])
        self.assertIsNotNone(model.deformable_offset)
        self.assertEqual(model.deformable_offset.dtype, wp.int32)
        self.assertEqual(list(model.deformable_offset.numpy()), builder.deformable_offset)

    def test_failed_validation_does_not_register_deformable(self):
        # Malformed indices early-return; no deformable should be recorded.
        builder = newton.ModelBuilder()
        verts, _ = _equilateral_triangle(scale=0.1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0, 0, 0),
                vertices=verts.tolist(),
                indices=[0, 1],
                density=100.0,
                validate_mesh=True,
            )
        self.assertEqual(builder.deformable_count, 0)
        self.assertEqual(builder.deformable_offset, [0])


if __name__ == "__main__":
    unittest.main()
