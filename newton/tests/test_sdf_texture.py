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
from newton._src.geometry.sdf_utils import (
    SDFData,
    get_distance_to_mesh,
    sample_sdf_extrapolated,
    sample_sdf_grad_extrapolated,
)
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


def _create_sphere_mesh(radius: float = 0.5, subdivisions: int = 3) -> Mesh:
    """Create an icosphere mesh for smooth-SDF testing."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_list = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    norm_factor = np.linalg.norm(verts_list[0])
    verts_list = [[v[i] / norm_factor * radius for i in range(3)] for v in verts_list]

    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}

        def get_midpoint(i0, i1, _ep=edge_midpoints):
            key = (min(i0, i1), max(i0, i1))
            if key not in _ep:
                v0, v1 = verts_list[i0], verts_list[i1]
                mid = [(v0[j] + v1[j]) / 2 for j in range(3)]
                length = np.sqrt(sum(m * m for m in mid))
                _ep[key] = len(verts_list)
                verts_list.append([m / length * radius for m in mid])
            return _ep[key]

        for f in faces:
            a = get_midpoint(f[0], f[1])
            b = get_midpoint(f[1], f[2])
            c = get_midpoint(f[2], f[0])
            new_faces.extend([[f[0], a, c], [f[1], b, a], [f[2], c, b], [a, b, c]])
        faces = new_faces

    verts = np.array(verts_list, dtype=np.float32)
    indices = np.array(faces, dtype=np.int32).flatten()
    return Mesh(verts, indices)


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


@wp.kernel
def _bvh_ground_truth_kernel(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    results[tid] = get_distance_to_mesh(mesh, query_points[tid], 10000.0, 0.5)


@wp.kernel
def _bvh_ground_truth_grad_kernel(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
):
    """Compute BVH ground truth distance and finite-difference gradient."""
    tid = wp.tid()
    p = query_points[tid]
    d = get_distance_to_mesh(mesh, p, 10000.0, 0.5)
    results[tid] = d
    eps = 1.0e-4
    dx = get_distance_to_mesh(mesh, p + wp.vec3(eps, 0.0, 0.0), 10000.0, 0.5) - get_distance_to_mesh(
        mesh, p - wp.vec3(eps, 0.0, 0.0), 10000.0, 0.5
    )
    dy = get_distance_to_mesh(mesh, p + wp.vec3(0.0, eps, 0.0), 10000.0, 0.5) - get_distance_to_mesh(
        mesh, p - wp.vec3(0.0, eps, 0.0), 10000.0, 0.5
    )
    dz = get_distance_to_mesh(mesh, p + wp.vec3(0.0, 0.0, eps), 10000.0, 0.5) - get_distance_to_mesh(
        mesh, p - wp.vec3(0.0, 0.0, eps), 10000.0, 0.5
    )
    inv_2eps = 0.5 / eps
    gradients[tid] = wp.vec3(dx * inv_2eps, dy * inv_2eps, dz * inv_2eps)


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


def _compare_texture_vs_nanovdb(test, tex_sdf, nanovdb_data, query_points, narrow_band, device):
    """Shared helper: sample both SDFs and compute contact-zone error statistics.

    Only considers points where ``|nanovdb_distance| <= 0.5 * narrow_band``
    to avoid the subgrid-to-coarse transition fringe at the narrow-band edge
    where errors are expected.  This keeps the comparison inside the region
    that actually matters for contacts.

    Returns a dict with ``nb_dist_*``, ``nb_angle_*`` keys for distance and
    gradient-angle stats (mean, median, p95, max).
    """
    n = query_points.shape[0]
    tex_vals = wp.zeros(n, dtype=float, device=device)
    tex_grads = wp.zeros(n, dtype=wp.vec3, device=device)
    nano_vals = wp.zeros(n, dtype=float, device=device)
    nano_grads = wp.zeros(n, dtype=wp.vec3, device=device)

    wp.launch(
        _sample_texture_sdf_grad_kernel, dim=n, inputs=[tex_sdf, query_points, tex_vals, tex_grads], device=device
    )
    wp.launch(
        _sample_nanovdb_grad_kernel, dim=n, inputs=[nanovdb_data, query_points, nano_vals, nano_grads], device=device
    )
    wp.synchronize()

    tv = tex_vals.numpy()
    nv = nano_vals.numpy()
    tg = tex_grads.numpy()
    ng = nano_grads.numpy()

    valid = (np.abs(tv) < 1e5) & (np.abs(nv) < 1e5)
    inner_band = 0.5 * narrow_band
    nb = valid & (np.abs(nv) <= inner_band)

    stats = {"nb_count": int(nb.sum()), "all_count": int(valid.sum())}

    for tag, mask in [("nb", nb), ("all", valid)]:
        if mask.sum() == 0:
            continue
        diff = np.abs(tv[mask] - nv[mask])
        stats[f"{tag}_dist_mean"] = float(diff.mean())
        stats[f"{tag}_dist_median"] = float(np.median(diff))
        stats[f"{tag}_dist_p95"] = float(np.percentile(diff, 95))
        stats[f"{tag}_dist_max"] = float(diff.max())

        n1 = np.linalg.norm(tg[mask], axis=1)
        n2 = np.linalg.norm(ng[mask], axis=1)
        gv = (n1 > 1e-6) & (n2 > 1e-6)
        if gv.sum() > 0:
            tg_n = tg[mask][gv] / n1[gv, None]
            ng_n = ng[mask][gv] / n2[gv, None]
            dots = np.clip(np.sum(tg_n * ng_n, axis=1), -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            stats[f"{tag}_angle_mean"] = float(angles.mean())
            stats[f"{tag}_angle_median"] = float(np.median(angles))
            stats[f"{tag}_angle_p95"] = float(np.percentile(angles, 95))
            stats[f"{tag}_angle_max"] = float(angles.max())
            stats[f"{tag}_grad_valid"] = int(gv.sum())

    return stats


def test_texture_sdf_values_match_nanovdb(test, device):
    """Compare float32 texture SDF vs NanoVDB distance in the contact zone.

    Uses the inner half of the narrow band (``|d| <= 0.05``) to avoid the
    subgrid-to-coarse transition fringe and demand sub-millimeter accuracy
    where contacts actually happen.
    """
    mesh = _create_box_mesh()
    tex_sdf, _coarse_tex, _subgrid_tex, nanovdb_data, _wp_mesh = _build_texture_and_nanovdb(mesh, device=device)

    query_np = _generate_query_points(mesh, num_points=2000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)
    narrow_band = 0.1

    s = _compare_texture_vs_nanovdb(test, tex_sdf, nanovdb_data, query_points, narrow_band, device)

    test.assertGreater(s["nb_count"], 300, f"Too few contact-zone points: {s['nb_count']}")

    test.assertLess(s["nb_dist_mean"], 5e-4, f"Contact-zone mean dist error: {s['nb_dist_mean']:.4e}")
    test.assertLess(s["nb_dist_median"], 1e-4, f"Contact-zone median dist error: {s['nb_dist_median']:.4e}")
    test.assertLess(s["nb_dist_p95"], 2e-3, f"Contact-zone p95 dist error: {s['nb_dist_p95']:.4e}")
    test.assertLess(s["nb_dist_max"], 0.01, f"Contact-zone max dist error: {s['nb_dist_max']:.4e}")


def test_texture_sdf_gradient_accuracy(test, device):
    """Compare float32 texture gradient vs NanoVDB gradient in the contact zone.

    Uses the inner half of the narrow band to avoid the transition fringe.
    Max angle is not asserted because box corners produce inherent gradient
    discontinuities even in the contact zone. The p95 tolerance is generous
    because ``_generate_query_points`` concentrates 70% of samples near box
    vertices, placing ~5% of inner-band points near corners where the SDF
    gradient is multi-valued.
    """
    mesh = _create_box_mesh()
    tex_sdf, _coarse_tex, _subgrid_tex, nanovdb_data, _wp_mesh = _build_texture_and_nanovdb(mesh, device=device)

    query_np = _generate_query_points(mesh, num_points=2000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)
    narrow_band = 0.1

    s = _compare_texture_vs_nanovdb(test, tex_sdf, nanovdb_data, query_points, narrow_band, device)

    test.assertGreater(
        s.get("nb_grad_valid", 0), 200, f"Too few contact-zone gradient points: {s.get('nb_grad_valid', 0)}"
    )

    test.assertLess(s["nb_angle_mean"], 3.0, f"Contact-zone mean gradient angle: {s['nb_angle_mean']:.2f} deg")
    test.assertLess(s["nb_angle_median"], 0.5, f"Contact-zone median gradient angle: {s['nb_angle_median']:.2f} deg")
    test.assertLess(s["nb_angle_p95"], 15.0, f"Contact-zone p95 gradient angle: {s['nb_angle_p95']:.2f} deg")


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


def _build_texture_sdf_with_mode(
    mesh, quantization_mode, resolution=64, margin=0.05, narrow_band_range=(-0.1, 0.1), device="cuda:0"
):
    """Build a texture SDF with a specific quantization mode."""
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    tex_sdf, coarse_tex, subgrid_tex, _block_coords = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=margin,
        narrow_band_range=narrow_band_range,
        max_resolution=resolution,
        quantization_mode=quantization_mode,
        device=device,
    )
    return tex_sdf, coarse_tex, subgrid_tex, wp_mesh


def test_uint16_native_texture_dtype(test, device):
    """Verify uint16 mode produces native uint16 subgrid textures."""
    mesh = _create_box_mesh()
    _tex_sdf, _coarse_tex, subgrid_tex, _wp_mesh = _build_texture_sdf_with_mode(
        mesh,
        QuantizationMode.UINT16,
        resolution=32,
        device=device,
    )
    import warp  # noqa: PLC0415

    test.assertEqual(subgrid_tex.dtype, warp.uint16, "Subgrid texture should be uint16")
    test.assertEqual(_coarse_tex.dtype, warp.float32, "Coarse texture should remain float32")


def test_uint16_vs_nanovdb_distance(test, device):
    """Compare uint16 texture SDF vs NanoVDB distance in the contact zone.

    Uses the inner half of the narrow band to avoid the subgrid-to-coarse
    transition fringe and demand sub-millimeter accuracy.
    """
    mesh = _create_box_mesh()
    tex_sdf, _ct, _st, _wm = _build_texture_sdf_with_mode(
        mesh,
        QuantizationMode.UINT16,
        resolution=64,
        device=device,
    )

    mesh_copy = _create_box_mesh()
    mesh_copy.build_sdf(max_resolution=64, narrow_band_range=(-0.1, 0.1), margin=0.05)
    nanovdb_data = mesh_copy.sdf.to_kernel_data()

    query_np = _generate_query_points(mesh, num_points=2000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)
    narrow_band = 0.1

    s = _compare_texture_vs_nanovdb(test, tex_sdf, nanovdb_data, query_points, narrow_band, device)

    test.assertGreater(s["nb_count"], 300, f"Too few contact-zone points: {s['nb_count']}")

    test.assertLess(s["nb_dist_mean"], 5e-4, f"Contact-zone mean dist error: {s['nb_dist_mean']:.4e}")
    test.assertLess(s["nb_dist_median"], 1e-4, f"Contact-zone median dist error: {s['nb_dist_median']:.4e}")
    test.assertLess(s["nb_dist_p95"], 2e-3, f"Contact-zone p95 dist error: {s['nb_dist_p95']:.4e}")
    test.assertLess(s["nb_dist_max"], 0.01, f"Contact-zone max dist error: {s['nb_dist_max']:.4e}")


def test_uint16_vs_nanovdb_gradient(test, device):
    """Compare uint16 texture SDF gradient vs NanoVDB gradient in the contact zone.

    Uses the inner half of the narrow band. Max angle is not asserted because
    box corners produce inherent gradient discontinuities. The p95 tolerance
    is generous because vertex-concentrated sampling places ~5% of inner-band
    points near corners where the SDF gradient is multi-valued.
    """
    mesh = _create_box_mesh()
    tex_sdf, _ct, _st, _wm = _build_texture_sdf_with_mode(
        mesh,
        QuantizationMode.UINT16,
        resolution=64,
        device=device,
    )

    mesh_copy = _create_box_mesh()
    mesh_copy.build_sdf(max_resolution=64, narrow_band_range=(-0.1, 0.1), margin=0.05)
    nanovdb_data = mesh_copy.sdf.to_kernel_data()

    query_np = _generate_query_points(mesh, num_points=2000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)
    narrow_band = 0.1

    s = _compare_texture_vs_nanovdb(test, tex_sdf, nanovdb_data, query_points, narrow_band, device)

    test.assertGreater(
        s.get("nb_grad_valid", 0), 200, f"Too few contact-zone gradient points: {s.get('nb_grad_valid', 0)}"
    )

    test.assertLess(s["nb_angle_mean"], 3.0, f"Contact-zone mean gradient angle: {s['nb_angle_mean']:.2f} deg")
    test.assertLess(s["nb_angle_median"], 0.5, f"Contact-zone median gradient angle: {s['nb_angle_median']:.2f} deg")
    test.assertLess(s["nb_angle_p95"], 15.0, f"Contact-zone p95 gradient angle: {s['nb_angle_p95']:.2f} deg")


def test_uint16_vs_float32_texture_accuracy(test, device):
    """Verify uint16 native textures match float32 textures within quantization precision.

    Tests that switching from float32 to uint16 subgrid textures introduces
    only minimal error from the 16-bit quantization, confirming the native
    uint16 texture path works correctly.
    """
    mesh = _create_box_mesh()

    tex_f32, _cf, _sf, _wf = _build_texture_sdf_with_mode(
        mesh,
        QuantizationMode.FLOAT32,
        resolution=64,
        device=device,
    )
    tex_u16, _cu, _su, _wu = _build_texture_sdf_with_mode(
        mesh,
        QuantizationMode.UINT16,
        resolution=64,
        device=device,
    )

    query_np = _generate_query_points(mesh, num_points=1000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)
    n = len(query_np)

    results_f32 = wp.zeros(n, dtype=float, device=device)
    results_u16 = wp.zeros(n, dtype=float, device=device)
    grads_f32 = wp.zeros(n, dtype=wp.vec3, device=device)
    grads_u16 = wp.zeros(n, dtype=wp.vec3, device=device)

    wp.launch(
        _sample_texture_sdf_grad_kernel, dim=n, inputs=[tex_f32, query_points, results_f32, grads_f32], device=device
    )
    wp.launch(
        _sample_texture_sdf_grad_kernel, dim=n, inputs=[tex_u16, query_points, results_u16, grads_u16], device=device
    )
    wp.synchronize()

    f32_np = results_f32.numpy()
    u16_np = results_u16.numpy()

    valid = (np.abs(f32_np) < 1e5) & (np.abs(u16_np) < 1e5)
    test.assertGreater(np.sum(valid), 500)

    dist_diff = np.abs(f32_np[valid] - u16_np[valid])
    mean_dist_err = float(dist_diff.mean())
    max_dist_err = float(dist_diff.max())
    test.assertLess(mean_dist_err, 1e-4, f"UINT16 vs FLOAT32 mean distance error: {mean_dist_err:.2e}")
    test.assertLess(max_dist_err, 1e-3, f"UINT16 vs FLOAT32 max distance error: {max_dist_err:.2e}")

    gf = grads_f32.numpy()
    gu = grads_u16.numpy()
    n1 = np.linalg.norm(gf, axis=1)
    n2 = np.linalg.norm(gu, axis=1)
    grad_valid = valid & (n1 > 1e-8) & (n2 > 1e-8)

    if np.sum(grad_valid) > 100:
        gf_n = gf[grad_valid] / n1[grad_valid, None]
        gu_n = gu[grad_valid] / n2[grad_valid, None]
        dots = np.sum(gf_n * gu_n, axis=1)
        angles = np.arccos(np.clip(dots, -1, 1)) * 180.0 / np.pi
        mean_angle = float(angles.mean())
        test.assertLess(mean_angle, 1.0, f"UINT16 vs FLOAT32 mean gradient angle: {mean_angle:.4f} deg")


def _generate_sphere_query_points(radius: float = 0.5, num_points: int = 3000, seed: int = 42) -> np.ndarray:
    """Generate random query points distributed around a sphere surface."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_points, 3)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms
    radial_offsets = rng.normal(0, 0.03, size=(num_points, 1)).astype(np.float32)
    return directions * (radius + radial_offsets)


def test_texture_sdf_vs_ground_truth_distance(test, device):
    """Compare texture SDF distance against BVH ground truth in the contact zone.

    Uses a sphere mesh (no edges/corners) so the only error source is
    trilinear interpolation between grid vertices.  Tests the inner
    contact zone (``|d| < 0.05``) where contacts happen.
    """
    mesh = _create_sphere_mesh(radius=0.5, subdivisions=3)
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    tex_sdf, _ct, _st, _bc = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=64,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )

    query_np = _generate_sphere_query_points(radius=0.5, num_points=3000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)
    n = len(query_np)

    tex_results = wp.zeros(n, dtype=float, device=device)
    bvh_results = wp.zeros(n, dtype=float, device=device)
    wp.launch(_sample_texture_sdf_kernel, dim=n, inputs=[tex_sdf, query_points, tex_results], device=device)
    wp.launch(_bvh_ground_truth_kernel, dim=n, inputs=[wp_mesh.id, query_points, bvh_results], device=device)
    wp.synchronize()

    tex_np = tex_results.numpy()
    bvh_np = bvh_results.numpy()

    inner_band = 0.05
    valid = (np.abs(tex_np) < 1e5) & (np.abs(bvh_np) < inner_band)
    test.assertGreater(valid.sum(), 500, f"Too few inner-band points: {valid.sum()}")

    diff = np.abs(tex_np[valid] - bvh_np[valid])
    test.assertLess(float(diff.mean()), 2e-4, f"GT mean dist error: {diff.mean():.4e}")
    test.assertLess(float(np.median(diff)), 1.5e-4, f"GT median dist error: {np.median(diff):.4e}")
    test.assertLess(float(np.percentile(diff, 95)), 7e-4, f"GT p95 dist error: {np.percentile(diff, 95):.4e}")
    test.assertLess(float(diff.max()), 2e-3, f"GT max dist error: {diff.max():.4e}")


def test_texture_sdf_vs_ground_truth_gradient(test, device):
    """Compare texture SDF gradient against BVH ground truth gradient.

    Uses a sphere mesh (smooth SDF everywhere) so gradient errors are
    purely from trilinear interpolation, not geometry discontinuities.
    BVH ground truth gradient is computed via central finite differences
    of ``get_distance_to_mesh``.
    """
    mesh = _create_sphere_mesh(radius=0.5, subdivisions=3)
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    tex_sdf, _ct, _st, _bc = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=0.05,
        narrow_band_range=(-0.1, 0.1),
        max_resolution=64,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )

    query_np = _generate_sphere_query_points(radius=0.5, num_points=3000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)
    n = len(query_np)

    tex_vals = wp.zeros(n, dtype=float, device=device)
    tex_grads = wp.zeros(n, dtype=wp.vec3, device=device)
    bvh_vals = wp.zeros(n, dtype=float, device=device)
    bvh_grads = wp.zeros(n, dtype=wp.vec3, device=device)

    wp.launch(
        _sample_texture_sdf_grad_kernel, dim=n, inputs=[tex_sdf, query_points, tex_vals, tex_grads], device=device
    )
    wp.launch(
        _bvh_ground_truth_grad_kernel, dim=n, inputs=[wp_mesh.id, query_points, bvh_vals, bvh_grads], device=device
    )
    wp.synchronize()

    bv = bvh_vals.numpy()
    tg = tex_grads.numpy()
    bg = bvh_grads.numpy()

    inner_band = 0.05
    valid = np.abs(bv) < inner_band
    n1 = np.linalg.norm(tg[valid], axis=1)
    n2 = np.linalg.norm(bg[valid], axis=1)
    gv = (n1 > 1e-6) & (n2 > 1e-6)
    test.assertGreater(gv.sum(), 500, f"Too few valid gradient points: {gv.sum()}")

    tg_n = tg[valid][gv] / n1[gv, None]
    bg_n = bg[valid][gv] / n2[gv, None]
    dots = np.clip(np.sum(tg_n * bg_n, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))

    test.assertLess(float(angles.mean()), 2.0, f"GT mean gradient angle: {angles.mean():.2f} deg")
    test.assertLess(float(np.median(angles)), 1.5, f"GT median gradient angle: {np.median(angles):.2f} deg")
    test.assertLess(
        float(np.percentile(angles, 95)), 5.0, f"GT p95 gradient angle: {np.percentile(angles, 95):.2f} deg"
    )


def test_build_sdf_texture_format_parameter(test, device):
    """Verify Mesh.build_sdf() respects the texture_format parameter."""
    mesh_u16 = _create_box_mesh()
    sdf_u16 = mesh_u16.build_sdf(max_resolution=32, texture_format="uint16", device=device)
    test.assertIsNotNone(sdf_u16)
    test.assertIsNotNone(sdf_u16._subgrid_texture)
    test.assertEqual(sdf_u16._subgrid_texture.dtype, wp.uint16)

    mesh_f32 = _create_box_mesh()
    sdf_f32 = mesh_f32.build_sdf(max_resolution=32, texture_format="float32", device=device)
    test.assertIsNotNone(sdf_f32)
    test.assertIsNotNone(sdf_f32._subgrid_texture)
    test.assertEqual(sdf_f32._subgrid_texture.dtype, wp.float32)


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
add_function_test(TestTextureSDF, "test_uint16_native_texture_dtype", test_uint16_native_texture_dtype, devices=devices)
add_function_test(TestTextureSDF, "test_uint16_vs_nanovdb_distance", test_uint16_vs_nanovdb_distance, devices=devices)
add_function_test(TestTextureSDF, "test_uint16_vs_nanovdb_gradient", test_uint16_vs_nanovdb_gradient, devices=devices)
add_function_test(
    TestTextureSDF, "test_uint16_vs_float32_texture_accuracy", test_uint16_vs_float32_texture_accuracy, devices=devices
)
add_function_test(
    TestTextureSDF, "test_build_sdf_texture_format_parameter", test_build_sdf_texture_format_parameter, devices=devices
)
add_function_test(
    TestTextureSDF,
    "test_texture_sdf_vs_ground_truth_distance",
    test_texture_sdf_vs_ground_truth_distance,
    devices=devices,
)
add_function_test(
    TestTextureSDF,
    "test_texture_sdf_vs_ground_truth_gradient",
    test_texture_sdf_vs_ground_truth_gradient,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
