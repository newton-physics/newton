# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resource inspection follows geometry currently stored in body-variant rows."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.viewer.viewer_model_tree import ModelTree
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_variant_resources(test: unittest.TestCase, device: wp.Device) -> None:
    """Variant switches update lazy resource references without altering Model storage."""
    sphere = newton.ModelBuilder()
    sphere.add_shape_sphere(sphere.add_body(), radius=0.25)
    meshes = [newton.Mesh.create_box(0.25), newton.Mesh.create_box(0.5)]
    meshes[0].texture = np.full((2, 2, 3), 255, dtype=np.uint8)
    if device.is_cuda:
        with wp.ScopedDevice(device):
            meshes[0].build_sdf(max_resolution=8)
    variants = [sphere]
    for mesh in meshes:
        variant = newton.ModelBuilder()
        body = variant.add_body()
        variant.add_shape_mesh(body, mesh=mesh)
        if mesh is meshes[0]:
            variant.add_shape_mesh(body, mesh=mesh, xform=wp.transform((1, 0, 0), wp.quat_identity()))
        variants.append(variant)
    builder = newton.ModelBuilder()
    builder.add_body_variants(builder.add_body(), variants, initial=1)
    model = builder.finalize(device=device)
    tree = ModelTree(model)
    resources, bvhs = tree.resources, tree.resources.bvhs
    shape = ("shape", 0)
    tree.expanded.add(shape)
    tree.resources.add_shape(tree, 0)

    test.assertIs(resources.source(0), meshes[0])
    test.assertIs(resources.source(1), meshes[0])
    test.assertIn(("shape_texture", 0), tree.children[shape])
    test.assertIn(("shape_bvh", 0), tree.children[shape])
    test.assertEqual(resources.shape_sdf(0) >= 0, device.is_cuda)
    test.assertEqual(resources.shape_sdf(1) >= 0, device.is_cuda)
    first_resource = bvhs.entries[("shape_bvh", 0)]
    first_mesh = first_resource.get()
    ws = SimpleNamespace(tree=tree, world=0)
    mesh_key = ("bvh_mesh", first_resource.index)
    test.assertEqual(bvhs._shape(ws, mesh_key, first_resource, first_mesh), 0)

    # A retained resource tree must invalidate both source pages and inverse
    # mesh owners; switching to another mesh preserves the stable shape key.
    for choice in (2, 0, 1):
        model.select_body_variants(wp.array([choice], dtype=int, device=device))
        pointers = model.shape_source_ptr.numpy().copy()
        sdf_indices = model._shape_sdf_index.numpy().copy() if model._shape_sdf_index is not None else None
        removed = tree.refresh_shapes()
        test.assertIn(shape, tree.labels)
        test.assertIn(shape, tree.expanded)
        expected = None if choice == 0 else meshes[choice - 1]
        test.assertIs(resources.source(0), expected)
        test.assertIs(resources.source(1), meshes[0] if choice == 1 else None)
        test.assertEqual(tree.meta[shape]["type"], int(model.shape_type.numpy()[0]))
        if choice:
            test.assertIn(("shape_bvh", 0), tree.children[shape])
            resource = bvhs.entries[("shape_bvh", 0)].get()
            test.assertEqual(resource.id, int(model.shape_source_ptr.numpy()[0]))
        else:
            test.assertNotIn(("shape_bvh", 0), tree.children[shape])
            test.assertIn(("shape_bvh", 0), removed)
        test.assertEqual(("shape_texture", 0) in tree.children[shape], choice == 1)
        for index in (0, 1):
            test.assertEqual(resources.shape_sdf(index) >= 0, choice == 1 and device.is_cuda)
        test.assertEqual(("mesh_sdf", 0) in tree.children[shape], choice == 1 and device.is_cuda)
        test.assertEqual(("shape_sdf", 0) in tree.children[shape], choice == 1 and device.is_cuda)
        test.assertEqual(bvhs._shape(ws, mesh_key, first_resource, first_mesh), 0 if choice == 1 else -1)
        np.testing.assert_array_equal(model.shape_source_ptr.numpy(), pointers)
        if sdf_indices is not None:
            np.testing.assert_array_equal(model._shape_sdf_index.numpy(), sdf_indices)


class TestViewerResources(unittest.TestCase):
    pass


add_function_test(TestViewerResources, "test_variant_resources", test_variant_resources, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
