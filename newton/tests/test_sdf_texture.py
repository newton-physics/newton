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

"""Tests for texture-based SDF construction and sampling.

Validates TextureSDFData construction, sampling accuracy against NanoVDB,
gradient quality, extrapolation, array indexing, and multi-resolution behavior.

Note: These tests require GPU (CUDA) since wp.Texture3D only supports CUDA devices.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Mesh
from newton._src.geometry.sdf_texture import (
    QuantizationMode,
    TextureSDFData,
    block_coords_from_subgrid_required,
    compute_isomesh_from_texture_sdf,
    create_empty_texture_sdf_data,
    create_texture_sdf_from_mesh,
    create_texture_sdf_from_volume,
    texture_sample_sdf,
    texture_sample_sdf_grad,
)
from newton._src.geometry.sdf_utils import SDFData, sample_sdf_extrapolated, sample_sdf_grad_extrapolated
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_cuda_available = wp.is_cuda_available()


def _create_box_mesh(half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> Mesh:
    """Create a simple box mesh for testing."""
    hx, hy, hz = half_extents
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0,
            2,
            1,
            0,
            3,
            2,  # Bottom
            4,
            5,
            6,
            4,
            6,
            7,  # Top
            0,
            1,
            5,
            0,
            5,
            4,  # Front
            2,
            3,
            7,
            2,
            7,
            6,  # Back
            0,
            4,
            7,
            0,
            7,
            3,  # Left
            1,
            2,
            6,
            1,
            6,
            5,  # Right
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, indices)


@wp.kernel
def _sample_texture_sdf_kernel(
    sdf: TextureSDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    results[tid] = texture_sample_sdf(sdf, query_points[tid])


@wp.kernel
def _sample_texture_sdf_grad_kernel(
    sdf: TextureSDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, grad = texture_sample_sdf_grad(sdf, query_points[tid])
    results[tid] = dist
    gradients[tid] = grad


@wp.kernel
def _sample_nanovdb_value_kernel(
    sdf_data: SDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    results[tid] = sample_sdf_extrapolated(sdf_data, query_points[tid])


@wp.kernel
def _sample_nanovdb_grad_kernel(
    sdf_data: SDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, grad = sample_sdf_grad_extrapolated(sdf_data, query_points[tid])
    results[tid] = dist
    gradients[tid] = grad


@wp.kernel
def _sample_texture_sdf_from_array_kernel(
    sdf_table: wp.array(dtype=TextureSDFData),
    sdf_idx: int,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    results[tid] = texture_sample_sdf(sdf_table[sdf_idx], query_points[tid])


def _build_texture_and_nanovdb(mesh, resolution=64, margin=0.05, narrow_band_range=(-0.1, 0.1), device="cuda:0"):
    """Build both texture SDF and NanoVDB SDF for comparison."""
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    # Build texture SDF
    tex_sdf, coarse_tex, subgrid_tex, _block_coords = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=margin,
        narrow_band_range=narrow_band_range,
        max_resolution=resolution,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )

    # Build NanoVDB SDF on the same device so volume pointers are valid
    mesh.build_sdf(
        device=device,
        max_resolution=resolution,
        narrow_band_range=narrow_band_range,
        margin=margin,
    )
    nanovdb_data = mesh.sdf.to_kernel_data()

    return tex_sdf, coarse_tex, subgrid_tex, nanovdb_data, wp_mesh


def _generate_query_points(mesh, num_points=1000, seed=42):
    """Generate random query points near the mesh."""
    rng = np.random.default_rng(seed)
    verts = mesh.vertices
    min_ext = verts.min(axis=0) - 0.05
    max_ext = verts.max(axis=0) + 0.05

    # Mix of near-surface and random points
    num_near = num_points * 7 // 10
    num_random = num_points - num_near

    vert_indices = rng.integers(0, len(verts), size=num_near)
    offsets = rng.normal(0, 0.02, size=(num_near, 3)).astype(np.float32)
    near_points = verts[vert_indices] + offsets

    random_points = rng.uniform(min_ext, max_ext, size=(num_random, 3)).astype(np.float32)

    points = np.concatenate([near_points, random_points], axis=0)
    rng.shuffle(points)
    return points


class TestTextureSDF(unittest.TestCase):
    pass


def test_texture_sdf_construction(test, device):
    """Build TextureSDFData and verify fields are populated."""
    mesh = _create_box_mesh()
    tex_sdf, _coarse_tex, _subgrid_tex, _, _wp_mesh = _build_texture_and_nanovdb(mesh, device=device)

    test.assertGreater(tex_sdf.inv_sdf_dx[0], 0.0)
    test.assertGreater(tex_sdf.inv_sdf_dx[1], 0.0)
    test.assertGreater(tex_sdf.inv_sdf_dx[2], 0.0)
    test.assertGreater(tex_sdf.subgrid_size, 0)
    test.assertEqual(tex_sdf.subgrid_size_f, float(tex_sdf.subgrid_size))
    test.assertEqual(tex_sdf.subgrid_samples_f, float(tex_sdf.subgrid_size + 1))

    # Verify box bounds contain the mesh
    box_lower = np.array([tex_sdf.sdf_box_lower[0], tex_sdf.sdf_box_lower[1], tex_sdf.sdf_box_lower[2]])
    box_upper = np.array([tex_sdf.sdf_box_upper[0], tex_sdf.sdf_box_upper[1], tex_sdf.sdf_box_upper[2]])
    mesh_min = mesh.vertices.min(axis=0)
    mesh_max = mesh.vertices.max(axis=0)
    test.assertTrue(np.all(box_lower <= mesh_min))
    test.assertTrue(np.all(box_upper >= mesh_max))


def test_texture_sdf_values_match_nanovdb(test, device):
    """Compare texture SDF vs NanoVDB at random points."""
    mesh = _create_box_mesh()
    tex_sdf, _coarse_tex, _subgrid_tex, nanovdb_data, _wp_mesh = _build_texture_and_nanovdb(mesh, device=device)

    query_np = _generate_query_points(mesh, num_points=1000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    tex_results = wp.zeros(1000, dtype=float, device=device)
    nano_results = wp.zeros(1000, dtype=float, device=device)

    wp.launch(_sample_texture_sdf_kernel, dim=1000, inputs=[tex_sdf, query_points, tex_results], device=device)
    wp.launch(_sample_nanovdb_value_kernel, dim=1000, inputs=[nanovdb_data, query_points, nano_results], device=device)
    wp.synchronize()

    tex_np = tex_results.numpy()
    nano_np = nano_results.numpy()

    # Filter to valid points (both give reasonable values)
    valid = (np.abs(tex_np) < 1e5) & (np.abs(nano_np) < 1e5)
    test.assertTrue(np.sum(valid) > 500, f"Too few valid points: {np.sum(valid)}")

    diff = np.abs(tex_np[valid] - nano_np[valid])
    mean_err = diff.mean()
    test.assertLess(mean_err, 0.02, f"Mean SDF error too large: {mean_err:.6f}")


def test_texture_sdf_gradient_accuracy(test, device):
    """Compare texture analytical gradient vs NanoVDB gradient."""
    mesh = _create_box_mesh()
    tex_sdf, _coarse_tex, _subgrid_tex, nanovdb_data, _wp_mesh = _build_texture_and_nanovdb(mesh, device=device)

    query_np = _generate_query_points(mesh, num_points=1000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    tex_vals = wp.zeros(1000, dtype=float, device=device)
    tex_grads = wp.zeros(1000, dtype=wp.vec3, device=device)
    nano_vals = wp.zeros(1000, dtype=float, device=device)
    nano_grads = wp.zeros(1000, dtype=wp.vec3, device=device)

    wp.launch(
        _sample_texture_sdf_grad_kernel,
        dim=1000,
        inputs=[tex_sdf, query_points, tex_vals, tex_grads],
        device=device,
    )
    wp.launch(
        _sample_nanovdb_grad_kernel,
        dim=1000,
        inputs=[nanovdb_data, query_points, nano_vals, nano_grads],
        device=device,
    )
    wp.synchronize()

    tg = tex_grads.numpy()
    ng = nano_grads.numpy()
    tv = tex_vals.numpy()
    nv = nano_vals.numpy()

    # Compute gradient angles for valid points
    valid_mask = (np.abs(tv) < 1e5) & (np.abs(nv) < 1e5)
    n1 = np.linalg.norm(tg, axis=1)
    n2 = np.linalg.norm(ng, axis=1)
    grad_valid = valid_mask & (n1 > 1e-8) & (n2 > 1e-8)

    test.assertTrue(np.sum(grad_valid) > 300, f"Too few valid gradient points: {np.sum(grad_valid)}")

    tg_n = tg[grad_valid] / n1[grad_valid, None]
    ng_n = ng[grad_valid] / n2[grad_valid, None]
    dots = np.sum(tg_n * ng_n, axis=1)
    angles = np.arccos(np.clip(dots, -1, 1)) * 180.0 / np.pi

    mean_angle = float(angles.mean())
    test.assertLess(mean_angle, 10.0, f"Mean gradient angle too large: {mean_angle:.2f} deg")


def test_texture_sdf_extrapolation(test, device):
    """Points outside box have correct extrapolated distance."""
    mesh = _create_box_mesh(half_extents=(0.5, 0.5, 0.5))
    tex_sdf, _coarse_tex, _subgrid_tex, _, _wp_mesh = _build_texture_and_nanovdb(mesh, device=device)

    # Points well outside the box along +X axis
    outside_points = np.array(
        [
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    query_points = wp.array(outside_points, dtype=wp.vec3, device=device)
    results = wp.zeros(4, dtype=float, device=device)

    wp.launch(_sample_texture_sdf_kernel, dim=4, inputs=[tex_sdf, query_points, results], device=device)
    wp.synchronize()

    vals = results.numpy()
    # Points far outside should have positive distance
    for i in range(4):
        test.assertGreater(vals[i], 0.5, f"Point {i} should be far outside, got dist={vals[i]:.4f}")


def test_texture_sdf_array_indexing(test, device):
    """Create wp.array(dtype=TextureSDFData) with 2 entries, sample from kernel via index."""
    mesh1 = _create_box_mesh(half_extents=(0.5, 0.5, 0.5))
    mesh2 = _create_box_mesh(half_extents=(0.3, 0.3, 0.3))

    wp_mesh1 = wp.Mesh(
        points=wp.array(mesh1.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh1.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    wp_mesh2 = wp.Mesh(
        points=wp.array(mesh2.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh2.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    tex_sdf1, _coarse1, _sub1, _ = create_texture_sdf_from_mesh(
        wp_mesh1,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=32,
        device=device,
    )
    tex_sdf2, _coarse2, _sub2, _ = create_texture_sdf_from_mesh(
        wp_mesh2,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=32,
        device=device,
    )

    sdf_array = wp.array([tex_sdf1, tex_sdf2], dtype=TextureSDFData, device=device)

    # Query point at origin (inside both boxes)
    query = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    results0 = wp.zeros(1, dtype=float, device=device)
    results1 = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        _sample_texture_sdf_from_array_kernel,
        dim=1,
        inputs=[sdf_array, 0, query, results0],
        device=device,
    )
    wp.launch(
        _sample_texture_sdf_from_array_kernel,
        dim=1,
        inputs=[sdf_array, 1, query, results1],
        device=device,
    )
    wp.synchronize()

    val0 = float(results0.numpy()[0])
    val1 = float(results1.numpy()[0])

    # Origin is inside both boxes, so both should be negative
    test.assertLess(val0, 0.0, f"Origin should be inside box1, got {val0:.4f}")
    test.assertLess(val1, 0.0, f"Origin should be inside box2, got {val1:.4f}")
    # Box2 is smaller, so origin should be closer to its surface (less negative)
    test.assertGreater(
        val1, val0, f"Origin should be closer to surface in smaller box: val0={val0:.4f}, val1={val1:.4f}"
    )


def test_texture_sdf_multi_resolution(test, device):
    """Test at resolutions 32, 64, 128, 256 - higher res should be more accurate."""
    mesh = _create_box_mesh()
    query_np = _generate_query_points(mesh, num_points=500)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    # Build NanoVDB reference at high resolution
    mesh_copy = _create_box_mesh()
    mesh_copy.build_sdf(device=device, max_resolution=256, narrow_band_range=(-0.1, 0.1), margin=0.05)
    ref_data = mesh_copy.sdf.to_kernel_data()
    ref_results = wp.zeros(500, dtype=float, device=device)
    wp.launch(_sample_nanovdb_value_kernel, dim=500, inputs=[ref_data, query_points, ref_results], device=device)
    wp.synchronize()
    ref_np = ref_results.numpy()

    prev_mean_err = float("inf")
    for resolution in [32, 64, 128]:
        wp_mesh = wp.Mesh(
            points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
            support_winding_number=True,
        )
        tex_sdf, _coarse_tex, _subgrid_tex, _ = create_texture_sdf_from_mesh(
            wp_mesh,
            margin=0.05,
            narrow_band_range=(-0.1, 0.1),
            max_resolution=resolution,
            device=device,
        )
        tex_results = wp.zeros(500, dtype=float, device=device)
        wp.launch(_sample_texture_sdf_kernel, dim=500, inputs=[tex_sdf, query_points, tex_results], device=device)
        wp.synchronize()

        tex_np = tex_results.numpy()
        valid = (np.abs(tex_np) < 1e5) & (np.abs(ref_np) < 1e5)
        if np.sum(valid) > 100:
            mean_err = float(np.abs(tex_np[valid] - ref_np[valid]).mean())
            # Error should decrease (or at least not increase much) with resolution
            test.assertLess(
                mean_err,
                prev_mean_err * 2.0,
                f"Error increased too much at res={resolution}: {mean_err:.6f} vs prev {prev_mean_err:.6f}",
            )
            prev_mean_err = mean_err


def test_texture_sdf_in_model(test, device):
    """Build a scene with 2 mesh shapes with SDFs and verify model.texture_sdf_data."""
    builder = newton.ModelBuilder(gravity=0.0)

    for i in range(2):
        body = builder.add_body(xform=wp.transform(wp.vec3(float(i) * 2.0, 0.0, 0.0)))
        mesh = _create_box_mesh(half_extents=(0.5, 0.5, 0.5))
        mesh.build_sdf(device=device, max_resolution=8)
        builder.add_shape_mesh(body, mesh=mesh)

    model = builder.finalize(device=device)

    # Both shapes should have SDF indices
    sdf_indices = model.shape_sdf_index.numpy()
    test.assertEqual(sdf_indices[0], 0)
    test.assertEqual(sdf_indices[1], 1)

    # texture_sdf_data should have 2 entries
    test.assertIsNotNone(model.texture_sdf_data)
    test.assertEqual(len(model.texture_sdf_data), 2)

    # Both entries should have valid coarse textures (not empty)
    for idx in range(2):
        test.assertGreater(model.texture_sdf_coarse_textures[idx].width, 0, f"texture_sdf_data[{idx}] is empty")

    # Texture references should be kept alive
    test.assertEqual(len(model.texture_sdf_coarse_textures), 2)
    test.assertEqual(len(model.texture_sdf_subgrid_textures), 2)


def test_empty_texture_sdf_data(test, device):
    """Verify create_empty_texture_sdf_data returns a valid empty struct."""
    empty = create_empty_texture_sdf_data()
    test.assertEqual(empty.subgrid_size, 0)
    test.assertFalse(empty.scale_baked)


def test_texture_sdf_quantization_uint16(test, device):
    """Build texture SDF with UINT16 quantization and verify sampling accuracy."""
    mesh = _create_box_mesh()
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    tex_sdf_f32, _, _, _ = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=32,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )
    tex_sdf_u16, _, _, _ = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=32,
        quantization_mode=QuantizationMode.UINT16,
        device=device,
    )

    query_np = _generate_query_points(mesh, num_points=500)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    results_f32 = wp.zeros(500, dtype=float, device=device)
    results_u16 = wp.zeros(500, dtype=float, device=device)

    wp.launch(_sample_texture_sdf_kernel, dim=500, inputs=[tex_sdf_f32, query_points, results_f32], device=device)
    wp.launch(_sample_texture_sdf_kernel, dim=500, inputs=[tex_sdf_u16, query_points, results_u16], device=device)
    wp.synchronize()

    f32_np = results_f32.numpy()
    u16_np = results_u16.numpy()

    valid = (np.abs(f32_np) < 1e5) & (np.abs(u16_np) < 1e5)
    test.assertGreater(np.sum(valid), 200, f"Too few valid points: {np.sum(valid)}")

    diff = np.abs(f32_np[valid] - u16_np[valid])
    mean_err = diff.mean()
    test.assertLess(mean_err, 0.05, f"UINT16 vs FLOAT32 mean error too large: {mean_err:.6f}")


def test_texture_sdf_quantization_uint8(test, device):
    """Build texture SDF with UINT8 quantization and verify sampling accuracy."""
    mesh = _create_box_mesh()
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    tex_sdf_f32, _, _, _ = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=32,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )
    tex_sdf_u8, _, _, _ = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=32,
        quantization_mode=QuantizationMode.UINT8,
        device=device,
    )

    query_np = _generate_query_points(mesh, num_points=500)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    results_f32 = wp.zeros(500, dtype=float, device=device)
    results_u8 = wp.zeros(500, dtype=float, device=device)

    wp.launch(_sample_texture_sdf_kernel, dim=500, inputs=[tex_sdf_f32, query_points, results_f32], device=device)
    wp.launch(_sample_texture_sdf_kernel, dim=500, inputs=[tex_sdf_u8, query_points, results_u8], device=device)
    wp.synchronize()

    f32_np = results_f32.numpy()
    u8_np = results_u8.numpy()

    valid = (np.abs(f32_np) < 1e5) & (np.abs(u8_np) < 1e5)
    test.assertGreater(np.sum(valid), 200, f"Too few valid points: {np.sum(valid)}")

    diff = np.abs(f32_np[valid] - u8_np[valid])
    mean_err = diff.mean()
    # UINT8 is coarser than UINT16, allow larger tolerance
    test.assertLess(mean_err, 0.1, f"UINT8 vs FLOAT32 mean error too large: {mean_err:.6f}")


def test_texture_sdf_isomesh_extraction(test, device):
    """Extract isosurface mesh from texture SDF and verify it has geometry."""
    mesh = _create_box_mesh()
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    tex_sdf, _coarse_tex, _subgrid_tex, _block_coords = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=32,
        device=device,
    )

    tex_array = wp.array([tex_sdf], dtype=TextureSDFData, device=device)

    coarse_w = _coarse_tex.width - 1
    coarse_h = _coarse_tex.height - 1
    coarse_d = _coarse_tex.depth - 1
    coarse_dims = (coarse_w, coarse_h, coarse_d)

    iso_mesh = compute_isomesh_from_texture_sdf(
        tex_array,
        0,
        tex_sdf.subgrid_start_slots,
        coarse_dims,
        device=device,
    )

    test.assertIsNotNone(iso_mesh, "Isomesh should not be None for a box mesh")
    test.assertGreater(len(iso_mesh.vertices), 0, "Isomesh should have vertices")
    test.assertGreater(len(iso_mesh.indices), 0, "Isomesh should have faces")


def test_block_coords_from_subgrid_required(test, device):
    """Verify block_coords_from_subgrid_required produces correct coordinates."""
    coarse_dims = (3, 2, 2)
    subgrid_size = 4
    w, h, d = coarse_dims
    total = w * h * d

    subgrid_required = np.zeros(total, dtype=np.int32)
    subgrid_required[0] = 1  # (0,0,0)
    subgrid_required[5] = 1  # bx=2, by=1, bz=0

    coords = block_coords_from_subgrid_required(subgrid_required, coarse_dims, subgrid_size)
    test.assertEqual(len(coords), 2)
    test.assertEqual(coords[0], wp.vec3us(0 * subgrid_size, 0 * subgrid_size, 0 * subgrid_size))
    test.assertEqual(coords[1], wp.vec3us(2 * subgrid_size, 1 * subgrid_size, 0 * subgrid_size))

    # With subgrid_occupied, all occupied subgrids are included
    subgrid_occupied = np.ones(total, dtype=np.int32)
    coords_all = block_coords_from_subgrid_required(
        subgrid_required, coarse_dims, subgrid_size, subgrid_occupied=subgrid_occupied
    )
    test.assertEqual(len(coords_all), total)


def test_texture_sdf_scale_baked(test, device):
    """Verify scale_baked flag propagates through construction."""
    mesh = _create_box_mesh()
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    tex_sdf_unbaked, _, _, _ = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=16,
        scale_baked=False,
        device=device,
    )
    tex_sdf_baked, _, _, _ = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=16,
        scale_baked=True,
        device=device,
    )

    test.assertFalse(tex_sdf_unbaked.scale_baked)
    test.assertTrue(tex_sdf_baked.scale_baked)


def test_texture_sdf_from_volume(test, device):
    """Build texture SDF from NanoVDB volumes and verify sampling."""
    mesh = _create_box_mesh()
    mesh.build_sdf(device=device, max_resolution=32, narrow_band_range=(-0.1, 0.1), margin=0.05)

    sdf = mesh.sdf
    sdf_data = sdf.to_kernel_data()

    min_ext = np.array(
        [
            sdf_data.center[0] - sdf_data.half_extents[0],
            sdf_data.center[1] - sdf_data.half_extents[1],
            sdf_data.center[2] - sdf_data.half_extents[2],
        ]
    )
    max_ext = np.array(
        [
            sdf_data.center[0] + sdf_data.half_extents[0],
            sdf_data.center[1] + sdf_data.half_extents[1],
            sdf_data.center[2] + sdf_data.half_extents[2],
        ]
    )
    voxel_size = np.array(
        [
            sdf_data.sparse_voxel_size[0],
            sdf_data.sparse_voxel_size[1],
            sdf_data.sparse_voxel_size[2],
        ]
    )

    tex_sdf, coarse_tex, _subgrid_tex = create_texture_sdf_from_volume(
        sdf.sparse_volume,
        sdf.coarse_volume,
        min_ext=min_ext,
        max_ext=max_ext,
        voxel_size=voxel_size,
        narrow_band_range=(-0.1, 0.1),
        device=device,
    )

    test.assertGreater(tex_sdf.subgrid_size, 0)
    test.assertGreater(coarse_tex.width, 0)

    # Sample at origin (inside box) — should be negative
    query = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    result = wp.zeros(1, dtype=float, device=device)
    wp.launch(_sample_texture_sdf_kernel, dim=1, inputs=[tex_sdf, query, result], device=device)
    wp.synchronize()
    val = float(result.numpy()[0])
    test.assertLess(val, 0.0, f"Origin should be inside box, got {val:.4f}")

    # Sample well outside — should be positive
    query_out = wp.array([wp.vec3(2.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    result_out = wp.zeros(1, dtype=float, device=device)
    wp.launch(_sample_texture_sdf_kernel, dim=1, inputs=[tex_sdf, query_out, result_out], device=device)
    wp.synchronize()
    val_out = float(result_out.numpy()[0])
    test.assertGreater(val_out, 0.0, f"Far point should be outside box, got {val_out:.4f}")


# Register tests for CUDA devices
devices = get_cuda_test_devices()
add_function_test(TestTextureSDF, "test_texture_sdf_construction", test_texture_sdf_construction, devices=devices)
add_function_test(
    TestTextureSDF, "test_texture_sdf_values_match_nanovdb", test_texture_sdf_values_match_nanovdb, devices=devices
)
add_function_test(
    TestTextureSDF, "test_texture_sdf_gradient_accuracy", test_texture_sdf_gradient_accuracy, devices=devices
)
add_function_test(TestTextureSDF, "test_texture_sdf_extrapolation", test_texture_sdf_extrapolation, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_array_indexing", test_texture_sdf_array_indexing, devices=devices)
add_function_test(
    TestTextureSDF, "test_texture_sdf_multi_resolution", test_texture_sdf_multi_resolution, devices=devices
)
add_function_test(TestTextureSDF, "test_texture_sdf_in_model", test_texture_sdf_in_model, devices=devices)
add_function_test(TestTextureSDF, "test_empty_texture_sdf_data", test_empty_texture_sdf_data, devices=devices)
add_function_test(
    TestTextureSDF, "test_texture_sdf_quantization_uint16", test_texture_sdf_quantization_uint16, devices=devices
)
add_function_test(
    TestTextureSDF, "test_texture_sdf_quantization_uint8", test_texture_sdf_quantization_uint8, devices=devices
)
add_function_test(
    TestTextureSDF, "test_texture_sdf_isomesh_extraction", test_texture_sdf_isomesh_extraction, devices=devices
)
add_function_test(
    TestTextureSDF,
    "test_block_coords_from_subgrid_required",
    test_block_coords_from_subgrid_required,
    devices=devices,
)
add_function_test(TestTextureSDF, "test_texture_sdf_scale_baked", test_texture_sdf_scale_baked, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_from_volume", test_texture_sdf_from_volume, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
