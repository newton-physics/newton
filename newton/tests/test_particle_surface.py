# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.particle_surface_colliders import (
    _sample_field_trilinear,
    extrapolate_surface_sdf_into_colliders,
)
from newton.geometry import ParticleSurface, extract_particle_surface
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _sample_trilinear_kernel(
    field: wp.array3d[wp.float32],
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    positions: wp.array[wp.vec3],
    values: wp.array[float],
):
    tid = wp.tid()
    values[tid] = _sample_field_trilinear(field, grid_origin, inv_voxel_size, positions[tid])


def _make_sphere_particles(n=3000, seed=42, device=None):
    rng = np.random.default_rng(seed)
    pts = []
    count = 0
    while count < n:
        p = rng.uniform(-1, 1, size=(n * 2, 3))
        accepted = p[np.linalg.norm(p, axis=1) < 1.0]
        pts.append(accepted)
        count += accepted.shape[0]
    pts = np.concatenate(pts)[:n].astype(np.float32)
    positions = wp.array(pts, dtype=wp.vec3, device=device)
    radii = wp.full(n, value=0.05, dtype=float, device=device)
    return positions, radii


def _make_ellipsoid_particles(n=3000, seed=42, device=None):
    positions, _ = _make_sphere_particles(n=n, seed=seed, device="cpu")
    pts = positions.numpy()
    pts *= np.array([1.0, 0.35, 0.2], dtype=np.float32)
    positions = wp.array(pts, dtype=wp.vec3, device=device)
    radii = wp.full(n, value=0.04, dtype=float, device=device)
    return positions, radii


def _make_disk_particles(n=3000, seed=123, z_extent=0.02, device=None):
    rng = np.random.default_rng(seed)
    xy = []
    count = 0
    while count < n:
        candidates = rng.uniform(-1.0, 1.0, size=(n, 2))
        accepted = candidates[np.linalg.norm(candidates, axis=1) < 1.0]
        xy.append(accepted)
        count += accepted.shape[0]
    xy = np.concatenate(xy)[:n]
    z = rng.uniform(-z_extent, z_extent, size=(n, 1))
    positions_np = np.concatenate((xy, z), axis=1).astype(np.float32)
    positions = wp.array(positions_np, dtype=wp.vec3, device=device)
    radii = wp.full(n, value=0.04, dtype=float, device=device)
    return positions, radii


def test_one_shot(test, device):
    positions, radii = _make_sphere_particles(device=device)
    verts, indices, normals = extract_particle_surface(
        positions,
        radii,
        voxel_size=0.05,
        kernel_radius=0.15,
    )
    test.assertIsNotNone(verts)
    test.assertGreater(verts.shape[0], 0)
    test.assertGreater(indices.shape[0], 0)
    test.assertEqual(indices.shape[0] % 3, 0)
    test.assertEqual(normals.shape[0], verts.shape[0])


def test_reusable_context(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(voxel_size=0.05, kernel_radius=0.15, device=device)
    verts, _, _ = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts)

    verts2, _, _ = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts2)


def test_field_only_extraction(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(voxel_size=0.05, kernel_radius=0.15, device=device)
    verts, indices, normals = ctx.extract(positions, radii=radii, compute_mesh=False)

    test.assertIsNone(verts)
    test.assertIsNone(indices)
    test.assertIsNone(normals)
    test.assertIsNotNone(ctx.field)

    verts, indices, normals = ctx.resurface()
    test.assertIsNotNone(verts)
    test.assertGreater(verts.shape[0], 0)
    test.assertEqual(indices.shape[0] % 3, 0)
    test.assertEqual(normals.shape[0], verts.shape[0])


def test_dynamic_grid_uses_realized_support(test, device):
    positions = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    radii = wp.array([0.05], dtype=wp.float32, device=device)
    ctx = ParticleSurface(
        voxel_size=0.02,
        kernel_radius=0.2,
        smooth_lambda=0.0,
        anisotropic=True,
        anisotropy_ratio=16.0,
        anisotropy_scale=2.0,
        anisotropy_min_neighbors=4,
        field_smooth_iterations=0,
        mesh_smooth_iterations=0,
        device=device,
    )
    ctx.extract(positions, radii=radii, compute_mesh=False)

    G = ctx.anisotropy_matrices.numpy()[0]
    reach = 2.0 * np.linalg.norm(np.linalg.inv(G), axis=0)
    grid_min = np.array([ctx.grid_origin[i] for i in range(3)])
    grid_max = grid_min + (np.array(ctx.grid_dims) - 1) * ctx.voxel_size
    test.assertTrue(np.all(grid_min <= -reach))
    test.assertTrue(np.all(grid_max >= reach))

    actual_extent = grid_max - grid_min
    conservative_extent = 2.0 * ctx._grid_padding(radii, device)
    test.assertTrue(np.all(actual_extent < conservative_extent))


def test_mesh_smoothing(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(voxel_size=0.05, kernel_radius=0.15, mesh_smooth_iterations=3, device=device)
    verts, _, _ = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts)


def test_empty_particles(test, device):
    positions = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    radii = wp.array(np.zeros(0, dtype=np.float32), dtype=float, device=device)
    ctx = ParticleSurface(voxel_size=0.05, device=device)
    verts, indices, normals = ctx.extract(positions, radii=radii)
    test.assertIsNone(verts)
    test.assertIsNone(indices)
    test.assertIsNone(normals)


def test_radii_length_mismatch(test, device):
    positions = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    radii = wp.full(3, value=0.05, dtype=float, device=device)
    ctx = ParticleSurface(voxel_size=0.1, device=device)
    with test.assertRaisesRegex(ValueError, "radii length"):
        ctx.extract(positions, radii=radii, compute_normals=False)


def test_invalid_radii_values(test, device):
    positions = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    ctx = ParticleSurface(voxel_size=0.1, device=device)

    for invalid_value in (0.0, -0.05, np.nan, np.inf):
        radii_np = np.full(4, 0.05, dtype=np.float32)
        radii_np[2] = invalid_value
        radii = wp.array(radii_np, dtype=float, device=device)
        with test.assertRaisesRegex(ValueError, "finite and positive"):
            ctx.extract(positions, radii=radii, compute_normals=False)


def test_radii_device_mismatch(test, device):
    if not wp.is_cuda_available():
        test.skipTest("requires CUDA for cross-device validation")

    positions_device = wp.get_device(device)
    radii_device = wp.get_device("cuda:0") if positions_device.is_cpu else wp.get_device("cpu")
    positions = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=positions_device)
    radii = wp.full(4, value=0.05, dtype=float, device=radii_device)
    ctx = ParticleSurface(voxel_size=0.1, device=positions_device)

    with test.assertRaisesRegex(ValueError, "radii device"):
        ctx.extract(positions, radii=radii, compute_normals=False)


def test_array_layout_validation(test, device):
    positions = wp.zeros(4, dtype=wp.vec3, device=device)
    radii = wp.full(4, value=0.05, dtype=wp.float32, device=device)
    ctx = ParticleSurface(voxel_size=0.1, device=device)

    with test.assertRaisesRegex(ValueError, "positions must be a 1-D array"):
        ctx.extract(wp.zeros((4, 3), dtype=wp.float32, device=device), radii=radii)
    with test.assertRaisesRegex(TypeError, "positions must have dtype wp.vec3"):
        ctx.extract(wp.zeros(4, dtype=wp.float32, device=device), radii=radii)
    with test.assertRaisesRegex(TypeError, "radii must have dtype wp.float32"):
        ctx.extract(positions, radii=wp.full(4, value=0.05, dtype=wp.float64, device=device))
    with test.assertRaisesRegex(TypeError, "particle_flags must have dtype wp.int32"):
        ctx.extract(positions, radii=radii, particle_flags=wp.ones(4, dtype=wp.float32, device=device))


def test_fem_field(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(voxel_size=0.1, kernel_radius=0.3, mesh_smooth_iterations=0, device=device)
    ctx.extract(positions, radii=radii)

    with wp.ScopedDevice(device):
        sdf = ctx.fem_field()
        field_np = ctx.field.numpy()
        origin = np.array([ctx.grid_origin[i] for i in range(3)])
        dx = ctx.voxel_size
        dims = ctx.grid_dims

        # Query at exact grid node positions — Q1 interpolation must reproduce DOF values.
        node_pts = []
        node_vals = []
        for ix in range(1, dims[0] - 1, 3):
            for iy in range(1, dims[1] - 1, 3):
                for iz in range(1, dims[2] - 1, 3):
                    node_pts.append(origin + np.array([ix, iy, iz]) * dx)
                    node_vals.append(field_np[ix, iy, iz])
        node_pts = np.array(node_pts, dtype=np.float32)
        node_vals = np.array(node_vals, dtype=np.float32)

        query_wp = wp.array(node_pts, dtype=wp.vec3, device=device)
        domain = fem.Cells(sdf.space.geometry)
        pic = fem.PicQuadrature(domain, positions=query_wp)
        fem_values = wp.zeros(len(node_pts), dtype=float, device=device)
        fem.interpolate(sdf, dest=fem_values, at=pic)

    diff = np.abs(fem_values.numpy() - node_vals)
    test.assertLess(diff.max(), 1e-4, f"FEM interpolation at grid nodes should be exact, got max_diff={diff.max():.6f}")


def test_anisotropic(test, device):
    positions, radii = _make_ellipsoid_particles(device=device)
    ctx = ParticleSurface(
        voxel_size=0.05,
        kernel_radius=0.15,
        anisotropic=True,
        mesh_smooth_iterations=0,
        device=device,
    )
    verts, _, _ = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts, "Anisotropic extraction produced no surface")
    test.assertGreater(verts.shape[0], 0)

    # Field should be close to 1.0 in the interior
    field_np = ctx.field.numpy()
    test.assertGreater(field_np.max(), 0.5, "Anisotropic field max should be significant")

    # G matrices should not all be isotropic (anisotropy should be active)
    G_np = ctx.anisotropy_matrices.numpy()
    diag_spread = np.max(np.abs(G_np[:, 0, 0] - G_np[:, 2, 2]))
    test.assertGreater(diag_spread, 1e-5, "G matrices should have direction-dependent scales")


def test_anisotropy_strength(test, device):
    positions, radii = _make_ellipsoid_particles(n=1500, device=device)
    ctx = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        anisotropic=True,
        anisotropy_strength=0.0,
        field_smooth_iterations=0,
        mesh_smooth_iterations=0,
        device=device,
    )
    verts, indices, _ = ctx.extract(positions, radii=radii, compute_normals=False)
    test.assertIsNotNone(verts)
    test.assertGreater(indices.shape[0], 0)

    G_np = ctx.anisotropy_matrices.numpy()
    iso_scale = 1.0 / (ctx.kernel_scale * ctx.kernel_radius)
    np.testing.assert_allclose(G_np[:, 0, 0], iso_scale, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(G_np[:, 1, 1], iso_scale, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(G_np[:, 2, 2], iso_scale, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(G_np[:, 0, 1], 0.0, atol=1.0e-5)
    np.testing.assert_allclose(G_np[:, 0, 2], 0.0, atol=1.0e-5)
    np.testing.assert_allclose(G_np[:, 1, 2], 0.0, atol=1.0e-5)


def test_sdf_field_mode(test, device):
    positions, radii = _make_sphere_particles(n=2000, device=device)
    ctx = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        field_mode="sdf",
        redistance_iterations=2,
        mesh_smooth_iterations=0,
        device=device,
    )
    verts, indices, _ = ctx.extract(positions, radii=radii, compute_normals=False)
    test.assertIsNotNone(verts)
    test.assertGreater(indices.shape[0], 0)

    field_np = ctx.field.numpy()
    test.assertLess(field_np.min(), 0.0)
    test.assertGreater(field_np.max(), 0.0)

    verts2, indices2, _ = ctx.resurface(compute_normals=False)
    test.assertIsNotNone(verts2)
    test.assertGreater(indices2.shape[0], 0)

    smooth_ctx = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        field_mode="sdf",
        redistance_iterations=2,
        mesh_smooth_iterations=2,
        device=device,
    )
    test.assertGreater(smooth_ctx._marching_threshold(), 0.0)


def test_update_field_matches_extract_field(test, device):
    positions, radii = _make_sphere_particles(n=900, device=device)
    ref = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        field_mode="sdf",
        field_smooth_iterations=1,
        field_smooth_radius=1,
        redistance_iterations=1,
        mesh_smooth_iterations=0,
        device=device,
    )
    ref.extract(positions, radii=radii, compute_normals=False)

    ctx = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        field_mode="sdf",
        field_smooth_iterations=1,
        field_smooth_radius=1,
        redistance_iterations=1,
        mesh_smooth_iterations=0,
        device=device,
    )
    ctx.configure_field_grid(ref.grid_origin, ref.grid_dims, max_particles=positions.shape[0], device=device)
    ctx.update_field(positions, radii)

    np.testing.assert_allclose(ctx.field.numpy(), ref.field.numpy(), rtol=1.0e-6, atol=1.0e-6)


def test_update_field_skips_inactive_particles(test, device):
    active_positions, active_radii = _make_sphere_particles(n=600, seed=7, device="cpu")
    active_np = active_positions.numpy()
    inactive_np = np.full_like(active_np, np.nan)
    all_np = np.concatenate((active_np, inactive_np), axis=0)
    positions = wp.array(all_np, dtype=wp.vec3, device=device)

    radii_np = np.concatenate((active_radii.numpy(), np.full(active_radii.shape[0], np.nan, dtype=np.float32)))
    radii = wp.array(radii_np, dtype=float, device=device)

    flags_np = np.concatenate(
        (
            np.full(active_np.shape[0], newton.ParticleFlags.ACTIVE, dtype=np.int32),
            np.zeros(active_np.shape[0], dtype=np.int32),
        )
    )
    flags = wp.array(flags_np, dtype=wp.int32, device=device)

    ref = ParticleSurface(
        voxel_size=0.08, kernel_radius=0.24, anisotropic=True, anisotropy_min_neighbors=4, device=device
    )
    ref.configure_field_grid((-1.5, -1.5, -1.5), (40, 40, 40), max_particles=active_np.shape[0], device=device)
    ref.update_field(
        wp.array(active_np, dtype=wp.vec3, device=device), wp.array(active_radii.numpy(), dtype=float, device=device)
    )

    ctx = ParticleSurface(
        voxel_size=0.08, kernel_radius=0.24, anisotropic=True, anisotropy_min_neighbors=4, device=device
    )
    ctx.configure_field_grid((-1.5, -1.5, -1.5), (40, 40, 40), max_particles=all_np.shape[0], device=device)
    ctx.update_field(positions, radii, particle_flags=flags)

    np.testing.assert_allclose(ctx.field.numpy(), ref.field.numpy(), rtol=1.0e-6, atol=1.0e-6)


def test_update_field_cuda_graph(test, device):
    device_obj = wp.get_device(device)
    if not device_obj.is_cuda:
        test.skipTest("requires CUDA graph capture")

    positions_cpu, radii_cpu = _make_sphere_particles(n=512, seed=9, device="cpu")
    positions_np = positions_cpu.numpy()
    radii_np = radii_cpu.numpy()
    flags_np = np.full(positions_np.shape[0], newton.ParticleFlags.ACTIVE, dtype=np.int32)
    flags_np[::7] = 0
    radii_np[::7] = np.nan
    positions_np[::7] = np.nan

    positions = wp.array(positions_np, dtype=wp.vec3, device=device)
    radii = wp.array(radii_np, dtype=float, device=device)
    flags = wp.array(flags_np, dtype=wp.int32, device=device)

    ctx = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        anisotropic=True,
        anisotropy_min_neighbors=4,
        field_mode="sdf",
        field_smooth_iterations=1,
        field_smooth_radius=1,
        redistance_iterations=1,
        device=device,
    )
    ctx.configure_field_grid((-1.5, -1.5, -1.5), (40, 40, 40), max_particles=positions.shape[0], device=device)
    ctx.update_field(positions, radii, particle_flags=flags)

    with wp.ScopedCapture(device=device_obj) as capture:
        ctx.update_field(positions, radii, particle_flags=flags)
    wp.capture_launch(capture.graph)

    field_np = ctx.field.numpy()
    test.assertLess(field_np.min(), 0.0)
    test.assertGreater(field_np.max(), 0.0)


def test_particle_sdf_surface_method(test, device):
    positions, radii = _make_sphere_particles(n=1200, device=device)
    ctx = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        surface_method="particle_sdf",
        particle_sdf_radius_scale=1.8,
        anisotropic=True,
        anisotropy_min_neighbors=4,
        mesh_smooth_iterations=0,
        device=device,
    )
    verts, indices, _ = ctx.extract(positions, radii=radii, compute_normals=False)
    test.assertEqual(ctx.field_mode, "sdf")
    test.assertIsNotNone(verts)
    test.assertGreater(indices.shape[0], 0)

    field_np = ctx.field.numpy()
    test.assertLess(field_np.min(), 0.0)
    test.assertGreater(field_np.max(), 0.0)
    test.assertEqual(ctx._marching_threshold(), 0.0)


def test_isotropic_particle_sdf_values(test, device):
    positions = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    radii = wp.array([0.5], dtype=wp.float32, device=device)
    ctx = ParticleSurface(
        voxel_size=1.0,
        kernel_radius=3.0,
        smooth_lambda=0.0,
        surface_method="particle_sdf",
        particle_sdf_band=2.0,
        device=device,
    )
    ctx.configure_field_grid((-1.0, -1.0, -1.0), (3, 3, 3), max_particles=1, device=device)
    ctx.update_field(positions, radii)

    field = ctx.field.numpy()
    test.assertAlmostEqual(float(field[1, 1, 1]), -0.5)
    test.assertAlmostEqual(float(field[0, 1, 1]), 0.5)
    test.assertAlmostEqual(float(field[1, 0, 1]), 0.5)
    test.assertAlmostEqual(float(field[1, 1, 0]), 0.5)
    test.assertAlmostEqual(float(field[0, 0, 0]), 6.0)


def test_anisotropy_scale(test, device):
    positions, radii = _make_disk_particles(device=device)
    ctx_tight = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        anisotropic=True,
        anisotropy_scale=0.75,
        anisotropy_min_neighbors=4,
        mesh_smooth_iterations=0,
        field_smooth_iterations=0,
        device=device,
    )
    ctx_wide = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        anisotropic=True,
        anisotropy_scale=1.5,
        anisotropy_min_neighbors=4,
        mesh_smooth_iterations=0,
        field_smooth_iterations=0,
        device=device,
    )

    ctx_tight.extract(positions, radii=radii, compute_normals=False)
    ctx_wide.extract(positions, radii=radii, compute_normals=False)

    G_tight = ctx_tight.anisotropy_matrices.numpy()
    G_wide = ctx_wide.anisotropy_matrices.numpy()
    test.assertGreater(np.max(np.abs(G_tight - G_wide)), 1e-5)


def test_anisotropic_kernel_scale_matches_isotropic_scale(test, device):
    positions, radii = _make_disk_particles(device=device)
    kernel_radius = 0.24
    kernel_scale = 0.6
    anisotropy_scale = 1.25
    ctx = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=kernel_radius,
        anisotropic=True,
        kernel_scale=kernel_scale,
        anisotropy_scale=anisotropy_scale,
        anisotropy_min_neighbors=4,
        mesh_smooth_iterations=0,
        field_smooth_iterations=0,
        device=device,
    )

    ctx.extract(positions, radii=radii, compute_normals=False)

    expected_det = 1.0 / (kernel_radius * kernel_scale * anisotropy_scale) ** 3
    actual_det = np.median(ctx.anisotropy_det.numpy())
    test.assertLess(abs(actual_det - expected_det) / expected_det, 0.05)


def test_anisotropy_ratio(test, device):
    positions, radii = _make_disk_particles(device=device)
    ctx_low = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        anisotropic=True,
        anisotropy_ratio=2.0,
        anisotropy_min_neighbors=4,
        mesh_smooth_iterations=0,
        field_smooth_iterations=0,
        device=device,
    )
    ctx_high = ParticleSurface(
        voxel_size=0.08,
        kernel_radius=0.24,
        anisotropic=True,
        anisotropy_ratio=8.0,
        anisotropy_min_neighbors=4,
        mesh_smooth_iterations=0,
        field_smooth_iterations=0,
        device=device,
    )

    ctx_low.extract(positions, radii=radii, compute_normals=False)
    ctx_high.extract(positions, radii=radii, compute_normals=False)

    low_eigs = np.linalg.eigvalsh(ctx_low.anisotropy_matrices.numpy())
    high_eigs = np.linalg.eigvalsh(ctx_high.anisotropy_matrices.numpy())
    low_ratios = low_eigs[:, -1] / np.maximum(low_eigs[:, 0], 1.0e-12)
    high_ratios = high_eigs[:, -1] / np.maximum(high_eigs[:, 0], 1.0e-12)

    test.assertLessEqual(np.percentile(low_ratios, 95), 2.05)
    test.assertGreater(np.percentile(high_ratios, 90), np.percentile(low_ratios, 90) + 1.0)


def test_particle_flags_filter_inactive(test, device):
    active_positions, active_radii = _make_sphere_particles(n=1000, seed=11, device=device)
    active_np = active_positions.numpy()
    all_np = np.concatenate((active_np, active_np + np.array([8.0, 0.0, 0.0], dtype=np.float32)), axis=0)

    positions = wp.array(all_np, dtype=wp.vec3, device=device)
    radii_np = np.full(all_np.shape[0], 0.05, dtype=np.float32)
    radii_np[active_np.shape[0] :] = np.nan
    radii = wp.array(radii_np, dtype=float, device=device)
    flags_np = np.concatenate(
        (
            np.full(active_np.shape[0], int(newton.ParticleFlags.ACTIVE), dtype=np.int32),
            np.zeros(active_np.shape[0], dtype=np.int32),
        )
    )
    flags = wp.array(flags_np, dtype=wp.int32, device=device)

    active_ctx = ParticleSurface(voxel_size=0.08, kernel_radius=0.24, field_smooth_iterations=0, device=device)
    flagged_ctx = ParticleSurface(voxel_size=0.08, kernel_radius=0.24, field_smooth_iterations=0, device=device)

    active_ctx.extract(active_positions, radii=active_radii, compute_normals=False)
    flagged_ctx.extract(positions, radii=radii, compute_normals=False, particle_flags=flags)

    test.assertEqual(flagged_ctx.grid_dims, active_ctx.grid_dims)
    np.testing.assert_allclose(np.array(flagged_ctx.grid_origin), np.array(active_ctx.grid_origin), atol=1e-6)

    inactive_flags = wp.zeros(all_np.shape[0], dtype=wp.int32, device=device)
    verts, indices, normals = flagged_ctx.extract(
        positions, radii=radii, compute_normals=False, particle_flags=inactive_flags
    )
    test.assertIsNone(verts)
    test.assertIsNone(indices)
    test.assertIsNone(normals)
    test.assertIsNone(flagged_ctx.field)
    test.assertIsNone(flagged_ctx.grid_dims)
    test.assertIsNone(flagged_ctx.grid_origin)
    verts, indices, normals = flagged_ctx.resurface(compute_normals=False)
    test.assertIsNone(verts)
    test.assertIsNone(indices)
    test.assertIsNone(normals)


def test_collider_field_sampler_clamps_to_grid(test, device):
    field_np = np.zeros((3, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                field_np[i, j, k] = float(i + 10 * j + 100 * k)

    field = wp.array(field_np, dtype=wp.float32, device=device)
    positions = wp.array(
        [
            [-0.25, 0.0, 0.0],
            [2.25, 0.0, 0.0],
            [1.5, 1.5, 1.5],
        ],
        dtype=wp.vec3,
        device=device,
    )
    values = wp.empty(positions.shape[0], dtype=float, device=device)

    wp.launch(
        _sample_trilinear_kernel,
        dim=positions.shape[0],
        inputs=[field, wp.vec3(0.0), 1.0, positions, values],
        device=device,
    )

    np.testing.assert_allclose(values.numpy(), np.array([0.0, 2.0, 166.5], dtype=np.float32), rtol=1e-6)


def test_solver_extract_particle_surface(test, device):
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(-0.15, -0.15, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=4,
        dim_y=4,
        dim_z=4,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        mass=1.0,
        jitter=0.0,
        radius_mean=0.05,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    state = model.state()

    options = SolverImplicitMPM.Config()
    options.grid_type = "dense"
    options.voxel_size = 0.1
    solver = SolverImplicitMPM(model, options)
    default_surface = solver.create_particle_surface()
    test.assertAlmostEqual(default_surface.voxel_size, 0.045)
    surface = solver.create_particle_surface(voxel_size=0.08, kernel_radius=0.24, field_smooth_iterations=0)

    verts, indices, normals = solver.extract_particle_surface(state, surface, compute_normals=False)

    test.assertIsNotNone(verts)
    test.assertGreater(verts.shape[0], 0)
    test.assertGreater(indices.shape[0], 0)
    test.assertIsNone(normals)

    inactive_flags = wp.zeros(model.particle_count, dtype=wp.int32, device=device)
    verts, indices, normals = solver.extract_particle_surface(
        state, surface, compute_normals=False, particle_flags=inactive_flags
    )
    test.assertIsNone(verts)
    test.assertIsNone(indices)
    test.assertIsNone(normals)

    surface_sdf = solver.create_particle_surface(
        voxel_size=0.08,
        kernel_radius=0.24,
        field_smooth_iterations=0,
        field_mode="sdf",
        redistance_iterations=1,
    )
    verts, indices, normals = solver.extract_particle_surface(
        state,
        surface_sdf,
        compute_normals=False,
        extrapolate_into_colliders=True,
        collider_extrapolation_depth=0.08,
    )

    test.assertIsNotNone(verts)
    test.assertGreater(verts.shape[0], 0)
    test.assertGreater(indices.shape[0], 0)
    test.assertIsNone(normals)


class TestParticleSurface(unittest.TestCase):
    def test_fem_field_requires_populated_field(self):
        ctx = ParticleSurface(voxel_size=0.1, device="cpu")
        with self.assertRaisesRegex(RuntimeError, r"extract\(\) or update_field\(\)"):
            ctx.fem_field()

    def test_defaults_are_inclusive(self):
        ctx = ParticleSurface(voxel_size=0.1, device="cpu")
        self.assertEqual(ctx.threshold, 0.25)
        self.assertEqual(ctx.smooth_lambda, 0.5)
        self.assertEqual(ctx.anisotropy_ratio, 4.0)
        self.assertEqual(ctx.kernel_scale, 0.5)
        self.assertEqual(ctx.anisotropy_scale, 1.0)
        self.assertEqual(ctx.anisotropy_min_neighbors, 25)
        self.assertEqual(ctx.anisotropy_strength, 1.0)
        self.assertEqual(ctx.surface_method, "density")
        self.assertEqual(ctx.particle_sdf_radius_scale, 1.0)
        self.assertEqual(ctx.particle_sdf_band, 2.0)
        self.assertEqual(ctx.field_smooth_iterations, 0)
        self.assertEqual(ctx.field_smooth_radius, 1)

    def test_one_shot_helper_defaults_are_inclusive(self):
        params = inspect.signature(extract_particle_surface).parameters
        self.assertEqual(params["threshold"].default, 0.25)
        self.assertEqual(params["smooth_lambda"].default, 0.5)
        self.assertEqual(params["anisotropy_ratio"].default, 4.0)
        self.assertEqual(params["kernel_scale"].default, 0.5)
        self.assertEqual(params["anisotropy_scale"].default, 1.0)
        self.assertEqual(params["field_mode"].default, None)
        self.assertEqual(params["anisotropy_min_neighbors"].default, 25)
        self.assertEqual(params["anisotropy_strength"].default, 1.0)
        self.assertEqual(params["surface_method"].default, "density")
        self.assertEqual(params["particle_sdf_radius_scale"].default, 1.0)
        self.assertEqual(params["particle_sdf_band"].default, 2.0)
        self.assertEqual(params["field_smooth_iterations"].default, 0)
        self.assertEqual(params["field_smooth_radius"].default, 1)

    def test_collider_extrapolation_onset_defaults_to_surface(self):
        solver_default = (
            inspect.signature(SolverImplicitMPM.extract_particle_surface)
            .parameters["collider_extrapolation_onset"]
            .default
        )
        adapter_default = inspect.signature(extrapolate_surface_sdf_into_colliders).parameters["onset"].default
        self.assertEqual(solver_default, 0.0)
        self.assertEqual(adapter_default, 0.0)

    def test_constructor_rejects_invalid_parameters(self):
        invalid_cases = [
            ({"voxel_size": 0.0}, "voxel_size"),
            ({"voxel_size": np.nan}, "voxel_size"),
            ({"voxel_size": 0.1, "kernel_radius": 0.0}, "kernel_radius"),
            ({"voxel_size": 0.1, "threshold": np.nan}, "threshold"),
            ({"voxel_size": 0.1, "smooth_lambda": -0.1}, "smooth_lambda"),
            ({"voxel_size": 0.1, "anisotropy_ratio": 0.5}, "anisotropy_ratio"),
            ({"voxel_size": 0.1, "kernel_scale": 0.0}, "kernel_scale"),
            ({"voxel_size": 0.1, "anisotropy_scale": 0.0}, "anisotropy_scale"),
            ({"voxel_size": 0.1, "anisotropy_strength": 1.5}, "anisotropy_strength"),
            ({"voxel_size": 0.1, "field_smooth_iterations": -1}, "field_smooth_iterations"),
            ({"voxel_size": 0.1, "mesh_smooth_lambda": 1.5}, "mesh_smooth_lambda"),
            ({"voxel_size": 0.1, "surface_method": "invalid"}, "surface_method"),
            (
                {"voxel_size": 0.1, "surface_method": "particle_sdf", "field_mode": "density"},
                "surface_method='particle_sdf'",
            ),
        ]
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    ParticleSurface(device="cpu", **kwargs)

    def test_grid_padding_covers_anisotropic_support(self):
        radii = wp.array(np.array([0.04, 0.05], dtype=np.float32), dtype=float, device="cpu")
        ctx = ParticleSurface(
            voxel_size=0.025,
            kernel_radius=0.075,
            anisotropic=True,
            anisotropy_ratio=8.0,
            kernel_scale=0.5,
            anisotropy_scale=1.2,
            padding=2,
            device="cpu",
        )
        expected_density_support = (
            2.0 * ctx.kernel_scale * ctx.anisotropy_scale * ctx.anisotropy_ratio ** (2.0 / 3.0) * ctx.kernel_radius
        )
        self.assertAlmostEqual(ctx._grid_padding(radii, "cpu"), expected_density_support + 2 * ctx.voxel_size)

        sdf_ctx = ParticleSurface(
            voxel_size=0.025,
            kernel_radius=0.075,
            anisotropic=True,
            anisotropy_ratio=8.0,
            surface_method="particle_sdf",
            particle_sdf_radius_scale=3.0,
            particle_sdf_band=2.0,
            padding=2,
            device="cpu",
        )
        expected_sdf_support = (
            0.05
            * sdf_ctx.particle_sdf_radius_scale
            * sdf_ctx.particle_sdf_band
            * sdf_ctx.anisotropy_ratio ** (2.0 / 3.0)
        )
        self.assertAlmostEqual(sdf_ctx._grid_padding(radii, "cpu"), expected_sdf_support + 2 * sdf_ctx.voxel_size)

    def test_sphere_particle_helper_returns_requested_count(self):
        positions, radii = _make_sphere_particles(n=37, device="cpu")
        self.assertEqual(positions.shape[0], 37)
        self.assertEqual(radii.shape[0], 37)

    def test_context_reallocates_resources_on_device_change(self):
        if not wp.is_cuda_available():
            self.skipTest("requires CUDA for cross-device reuse validation")

        positions_cpu, radii_cpu = _make_sphere_particles(n=512, device="cpu")
        positions_cuda = wp.array(positions_cpu.numpy(), dtype=wp.vec3, device="cuda:0")
        radii_cuda = wp.array(radii_cpu.numpy(), dtype=float, device="cuda:0")

        ctx = ParticleSurface(
            voxel_size=0.1,
            kernel_radius=0.3,
            anisotropic=True,
            field_smooth_iterations=1,
            device="cpu",
        )
        ctx.extract(positions_cpu, radii=radii_cpu, compute_normals=False)
        self.assertEqual(ctx.field.device, wp.get_device("cpu"))
        self.assertEqual(ctx.smoothed_positions.device, wp.get_device("cpu"))
        self.assertEqual(ctx.anisotropy_matrices.device, wp.get_device("cpu"))

        ctx.extract(positions_cuda, radii=radii_cuda, compute_normals=False)
        self.assertEqual(ctx.field.device, wp.get_device("cuda:0"))
        self.assertEqual(ctx.smoothed_positions.device, wp.get_device("cuda:0"))
        self.assertEqual(ctx.anisotropy_matrices.device, wp.get_device("cuda:0"))
        self.assertEqual(ctx._blur_weights.device, wp.get_device("cuda:0"))


devices = get_test_devices(mode="basic")

add_function_test(TestParticleSurface, "test_one_shot", test_one_shot, devices=devices)
add_function_test(TestParticleSurface, "test_reusable_context", test_reusable_context, devices=devices)
add_function_test(TestParticleSurface, "test_field_only_extraction", test_field_only_extraction, devices=devices)
add_function_test(
    TestParticleSurface,
    "test_dynamic_grid_uses_realized_support",
    test_dynamic_grid_uses_realized_support,
    devices=devices,
)
add_function_test(TestParticleSurface, "test_mesh_smoothing", test_mesh_smoothing, devices=devices)
add_function_test(TestParticleSurface, "test_empty_particles", test_empty_particles, devices=devices)
add_function_test(TestParticleSurface, "test_radii_length_mismatch", test_radii_length_mismatch, devices=devices)
add_function_test(TestParticleSurface, "test_invalid_radii_values", test_invalid_radii_values, devices=devices)
add_function_test(TestParticleSurface, "test_radii_device_mismatch", test_radii_device_mismatch, devices=devices)
add_function_test(TestParticleSurface, "test_array_layout_validation", test_array_layout_validation, devices=devices)
# fem_field test uses Grid3D which doesn't support multi-GPU partitioning;
# run only on the first selected test device.
add_function_test(TestParticleSurface, "test_fem_field", test_fem_field, devices=devices[:1])
add_function_test(TestParticleSurface, "test_anisotropic", test_anisotropic, devices=devices)
add_function_test(TestParticleSurface, "test_anisotropy_strength", test_anisotropy_strength, devices=devices)
add_function_test(TestParticleSurface, "test_sdf_field_mode", test_sdf_field_mode, devices=devices)
add_function_test(
    TestParticleSurface,
    "test_update_field_matches_extract_field",
    test_update_field_matches_extract_field,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_update_field_skips_inactive_particles",
    test_update_field_skips_inactive_particles,
    devices=devices,
)
add_function_test(TestParticleSurface, "test_update_field_cuda_graph", test_update_field_cuda_graph, devices=devices)
add_function_test(
    TestParticleSurface, "test_particle_sdf_surface_method", test_particle_sdf_surface_method, devices=devices
)
add_function_test(
    TestParticleSurface, "test_isotropic_particle_sdf_values", test_isotropic_particle_sdf_values, devices=devices
)
add_function_test(TestParticleSurface, "test_anisotropy_scale", test_anisotropy_scale, devices=devices)
add_function_test(
    TestParticleSurface,
    "test_anisotropic_kernel_scale_matches_isotropic_scale",
    test_anisotropic_kernel_scale_matches_isotropic_scale,
    devices=devices,
)
add_function_test(TestParticleSurface, "test_anisotropy_ratio", test_anisotropy_ratio, devices=devices)
add_function_test(
    TestParticleSurface, "test_particle_flags_filter_inactive", test_particle_flags_filter_inactive, devices=devices
)
add_function_test(
    TestParticleSurface,
    "test_collider_field_sampler_clamps_to_grid",
    test_collider_field_sampler_clamps_to_grid,
    devices=devices,
)
add_function_test(
    TestParticleSurface, "test_solver_extract_particle_surface", test_solver_extract_particle_surface, devices=devices
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
