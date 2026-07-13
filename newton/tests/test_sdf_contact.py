# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_contact import (
    mesh_sdf_contact_inner_spatial_depth,
    mesh_sdf_contact_search_precision,
)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_search_precision_kernel(out: wp.array[wp.float32]):
    out[0] = mesh_sdf_contact_search_precision(0.0, 1.0, 0.001, True)
    out[1] = mesh_sdf_contact_search_precision(0.01, 1.0, 0.001, True)
    out[2] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.1, True)
    out[3] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.001, False)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_inner_spatial_depth_kernel(out: wp.array[wp.float32]):
    # Texture SDF: expand the inner tier by one scaled voxel radius.
    out[0] = mesh_sdf_contact_inner_spatial_depth(0.0, 0.01, 2.0, 0.0001, True)
    # The tolerance scales with the SDF shape.
    out[1] = mesh_sdf_contact_inner_spatial_depth(0.0, 0.01, 0.5, 0.0001, True)
    # Never expand the inner tier past the outer gap boundary.
    out[2] = mesh_sdf_contact_inner_spatial_depth(0.001, 0.00005, 2.0, 0.0001, True)
    # BVH and heightfield paths have no texture-SDF resolution uncertainty.
    out[3] = mesh_sdf_contact_inner_spatial_depth(0.001, 0.01, 2.0, 0.0001, False)
    # A zero gap must leave the exact margin boundary unchanged.
    out[4] = mesh_sdf_contact_inner_spatial_depth(0.001, 0.0, 2.0, 0.0001, True)


class TestSDFContact(unittest.TestCase):
    def test_mesh_sdf_contact_search_precision_uses_inner_envelope(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(4, dtype=wp.float32, device=device)

        wp.launch(_mesh_sdf_contact_search_precision_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_allclose(values.numpy(), np.array([0.0, 0.001, 0.005, 0.005], dtype=np.float32))

    def test_mesh_sdf_contact_inner_spatial_depth_uses_voxel_tolerance(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(5, dtype=wp.float32, device=device)

        wp.launch(_mesh_sdf_contact_inner_spatial_depth_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_allclose(
            values.numpy(),
            np.array([0.0002, 0.00005, 0.00105, 0.001, 0.001], dtype=np.float32),
            rtol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
