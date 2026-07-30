# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Mesh finalized-mesh cache and its invalidation."""

import unittest

import numpy as np

from newton import Mesh
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_tet_mesh() -> Mesh:
    """Return a small watertight tetrahedron mesh."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    indices = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
    return Mesh(vertices, indices, compute_inertia=False)


class TestMeshCache(unittest.TestCase):
    def test_invalidate_cache_refreshes_edges(self):
        """Verify invalidate_cache() drops cached edge data after in-place index edits."""
        mesh = _make_tet_mesh()
        self.assertEqual(len(mesh.edges), 6)

        # in-place edit bypasses the indices setter, so the edge cache goes stale
        mesh.indices[:] = np.tile([0, 1, 2], 4)
        self.assertEqual(len(mesh.edges), 6)

        mesh.invalidate_cache()
        self.assertEqual(len(mesh.edges), 3)

    def test_invalidate_cache_refreshes_hash(self):
        """Verify invalidate_cache() drops the cached mesh hash after in-place vertex edits."""
        mesh = _make_tet_mesh()
        old_hash = hash(mesh)

        mesh.vertices[0] = (5.0, 0.0, 0.0)
        self.assertEqual(hash(mesh), old_hash)

        mesh.invalidate_cache()
        self.assertNotEqual(hash(mesh), old_hash)


def test_finalize_reuses_cached_mesh(test: TestMeshCache, device):
    """Verify finalize() reuses the cached Warp mesh only for identical arguments.

    Repeated calls with the same arguments must return the same Warp mesh,
    while a differing ``requires_grad`` or ``bvh_constructor`` is a distinct
    cache entry.
    """
    mesh = _make_tet_mesh()
    mesh_id_1 = mesh.finalize(device=device)
    mesh_id_2 = mesh.finalize(device=device)
    test.assertEqual(mesh_id_1, mesh_id_2)

    grad_mesh_id = mesh.finalize(device=device, requires_grad=True)
    test.assertNotEqual(grad_mesh_id, mesh_id_1)
    test.assertEqual(mesh.finalize(device=device, requires_grad=True), grad_mesh_id)

    sah_mesh_id = mesh.finalize(device=device, bvh_constructor="sah")
    test.assertNotEqual(sah_mesh_id, mesh_id_1)
    test.assertEqual(mesh.finalize(device=device, bvh_constructor="sah"), sah_mesh_id)
    test.assertNotEqual(mesh.finalize(device=device, bvh_constructor="median"), sah_mesh_id)


def test_vertex_reassignment_invalidates_finalized_mesh(test: TestMeshCache, device):
    """Verify assigning a new vertex array through the setter invalidates the finalized-mesh cache."""
    mesh = _make_tet_mesh()
    mesh.finalize(device=device)

    new_vertices = mesh.vertices.copy()
    new_vertices[0] = (5.0, 0.0, 0.0)
    mesh.vertices = new_vertices

    mesh.finalize(device=device)
    np.testing.assert_allclose(mesh.mesh.points.numpy(), new_vertices)


def test_index_reassignment_invalidates_finalized_mesh(test: TestMeshCache, device):
    """Verify assigning a new index array through the setter invalidates the finalized-mesh cache."""
    mesh = _make_tet_mesh()
    mesh.finalize(device=device)
    test.assertEqual(len(mesh.edges), 6)

    new_indices = np.array([0, 1, 2], dtype=np.int32)
    mesh.indices = new_indices

    mesh.finalize(device=device)
    np.testing.assert_array_equal(mesh.mesh.indices.numpy(), new_indices)
    test.assertEqual(len(mesh.edges), 3)


def test_copy_does_not_share_finalized_meshes(test: TestMeshCache, device):
    """Verify Mesh.copy() starts with an empty finalized-mesh cache.

    A copy is an independent geometry object, so finalizing it must build its
    own wp.Mesh rather than aliasing the original's cached buffers, even when
    copy() carries other topology-derived caches forward.
    """
    mesh = _make_tet_mesh()
    mesh_id = mesh.finalize(device=device)

    copied = mesh.copy()
    copy_id = copied.finalize(device=device)
    test.assertNotEqual(copy_id, mesh_id)

    # the original's cache entry is untouched by finalizing the copy
    test.assertEqual(mesh.finalize(device=device), mesh_id)


def test_in_place_edit_requires_invalidate_cache(test: TestMeshCache, device):
    """Verify invalidate_cache() refreshes the finalized mesh after in-place vertex edits.

    In-place assignment (``mesh.vertices[0] = ...``) bypasses the vertices
    property setter, so finalize() keeps returning the cached Warp mesh until
    the caller invalidates explicitly (review scenario from PR #3637).
    """
    mesh = _make_tet_mesh()
    mesh_id = mesh.finalize(device=device)

    mesh.vertices[0] = (5.0, 0.0, 0.0)
    test.assertEqual(mesh.finalize(device=device), mesh_id)

    mesh.invalidate_cache()
    mesh.finalize(device=device)
    np.testing.assert_allclose(mesh.mesh.points.numpy()[0], (5.0, 0.0, 0.0))


devices = get_test_devices()
add_function_test(TestMeshCache, "test_finalize_reuses_cached_mesh", test_finalize_reuses_cached_mesh, devices=devices)
add_function_test(
    TestMeshCache,
    "test_vertex_reassignment_invalidates_finalized_mesh",
    test_vertex_reassignment_invalidates_finalized_mesh,
    devices=devices,
)
add_function_test(
    TestMeshCache,
    "test_index_reassignment_invalidates_finalized_mesh",
    test_index_reassignment_invalidates_finalized_mesh,
    devices=devices,
)
add_function_test(
    TestMeshCache,
    "test_copy_does_not_share_finalized_meshes",
    test_copy_does_not_share_finalized_meshes,
    devices=devices,
)
add_function_test(
    TestMeshCache,
    "test_in_place_edit_requires_invalidate_cache",
    test_in_place_edit_requires_invalidate_cache,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
