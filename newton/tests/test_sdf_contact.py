# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_contact import (
    _sdf_rsqrt_rn,
    compute_block_counts_from_weights,
    mesh_sdf_contact_search_precision,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_search_precision_kernel(out: wp.array[wp.float32]):
    out[0] = mesh_sdf_contact_search_precision(0.0, 1.0, 0.001, True)
    out[1] = mesh_sdf_contact_search_precision(0.01, 1.0, 0.001, True)
    out[2] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.1, True)
    out[3] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.001, False)


@wp.kernel(enable_backward=False)
def _sdf_rsqrt_rn_kernel(values: wp.array[wp.float32], out: wp.array[wp.float32]):
    tid = wp.tid()
    out[tid] = _sdf_rsqrt_rn(values[tid])


def test_mesh_mesh_sdf_surface_witnesses(test, device, box_sdf, *, probe_sdf=False):
    """Keep SDF-based mesh-mesh witnesses on both surfaces with reduction and speculation."""
    # Parallel faces give exact witness planes. A common shift of both witnesses
    # preserves their separation, so checking only contact distance misses the bug.
    cases = (
        ("direct", False, -0.1),
        ("predictive", False, 0.1),
        ("reduced", True, -0.1),
        ("reduced_predictive", True, 0.1),
    )
    with wp.ScopedDevice(device):
        if probe_sdf:
            # A closed probe supplies the second SDF and exercises texture-only kernels.
            probe = newton.Mesh.create_box(0.2, 0.2, 0.2)
            probe.build_sdf(max_resolution=32, device=device)
        else:
            probe = newton.Mesh(
                np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
                np.array([0, 2, 1], dtype=np.int32),
                compute_inertia=False,
            )
        probe_half_height = 0.2 if probe_sdf else 0.0
        box = newton.Mesh.create_box(1.0, 1.0, 0.5)
        if box_sdf:
            box.build_sdf(max_resolution=32, device=device)
        # Allow texture quantization error, well below the original 0.05 m shift.
        sdf_tolerance = 1.0e-3 if box_sdf else 2.0e-5
        for probe_first in (True, False):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.0, gap=0.0)
            for is_probe in (probe_first, not probe_first):
                builder.add_shape_mesh(
                    body if is_probe else -1,
                    mesh=probe if is_probe else box,
                    cfg=cfg,
                )
            # Each shape order shares one model across all collision-only cases.
            model = builder.finalize(device=device)
            state = model.state()
            state.body_qd.assign([[0.0, 0.0, -2.0, 0.0, 0.0, 0.0]])
            for case, reduce_contacts, distance in cases:
                with test.subTest(probe_first=probe_first, case=case):
                    state.body_q.assign([wp.transform(wp.vec3(0.0, 0.0, 0.5 + distance + probe_half_height))])
                    pipeline = newton.CollisionPipeline(
                        model,
                        broad_phase="nxn",
                        reduce_contacts=reduce_contacts,
                        speculative_contact_gap_max=0.2 if distance > 0.0 else None,
                    )
                    if probe_sdf:
                        test.assertTrue(pipeline.narrow_phase.mesh_sdf_texture_only)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts, dt=0.1)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    test.assertGreater(count, 0)
                    points0 = contacts.rigid_contact_point0.numpy()[:count]
                    points1 = contacts.rigid_contact_point1.numpy()[:count]
                    probe_points, box_points = (points0, points1) if probe_first else (points1, points0)
                    # Dynamic-body witnesses are body-local; static ones are world-space.
                    if probe_sdf:
                        # Side edges can also produce contacts; accept any probe-box face.
                        np.testing.assert_allclose(np.max(np.abs(probe_points), axis=1), 0.2, atol=sdf_tolerance)
                    else:
                        np.testing.assert_allclose(probe_points[:, 2], 0.0, atol=2.0e-5)
                    np.testing.assert_allclose(box_points[:, 2], 0.5, atol=sdf_tolerance)


class TestSDFContact(unittest.TestCase):
    def test_block_count_scan_ignores_inactive_tail(self) -> None:
        """Keep active block offsets independent of stale inactive slots."""
        for device in get_test_devices():
            with self.subTest(device=device):
                weights = wp.array([1024, 2048, 1024, 99, 99, 99, 99, 99], dtype=wp.int32, device=device)
                pair_count = wp.array([3], dtype=wp.int32, device=device)
                total_weight = wp.array([4096], dtype=wp.int32, device=device)
                block_counts = wp.full(8, 77, dtype=wp.int32, device=device)
                offsets = wp.empty_like(block_counts)

                wp.launch(
                    compute_block_counts_from_weights,
                    dim=2,
                    inputs=[total_weight, weights, pair_count, len(weights), 4, block_counts, 2],
                    device=device,
                )
                wp.utils.array_scan(block_counts, offsets, inclusive=False)
                np.testing.assert_array_equal(offsets.numpy()[:4], [0, 1, 3, 4])

                pair_count.assign([1])
                total_weight.assign([1024])
                wp.launch(
                    compute_block_counts_from_weights,
                    dim=2,
                    inputs=[total_weight, weights, pair_count, len(weights), 4, block_counts, 2],
                    device=device,
                )
                wp.utils.array_scan(block_counts, offsets, inclusive=False)
                np.testing.assert_array_equal(offsets.numpy()[:2], [0, 4])

    def test_mesh_sdf_contact_search_precision_uses_inner_envelope(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(4, dtype=wp.float32, device=device)

        wp.launch(_mesh_sdf_contact_search_precision_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_allclose(values.numpy(), np.array([0.0, 0.001, 0.005, 0.005], dtype=np.float32))

    def test_sdf_rsqrt_rn_on_all_devices(self) -> None:
        """Verify the native reciprocal square root on CPU and CUDA."""
        values_np = np.array([0.25, 1.0, 2.0, 4.0, 100.0], dtype=np.float32)
        expected = np.float32(1.0) / np.sqrt(values_np)

        for device in get_test_devices():
            with self.subTest(device=device):
                values = wp.array(values_np, device=device)
                result = wp.empty_like(values)
                wp.launch(_sdf_rsqrt_rn_kernel, dim=len(values_np), inputs=[values, result], device=device)

                np.testing.assert_allclose(result.numpy(), expected, rtol=np.finfo(np.float32).eps, atol=0.0)


add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_bvh_bvh",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_test_devices(),
    box_sdf=False,
)
add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_bvh_texture",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_cuda_test_devices(),
    box_sdf=True,
)
add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_texture_texture",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_cuda_test_devices(),
    box_sdf=True,
    probe_sdf=True,
)


if __name__ == "__main__":
    unittest.main()
