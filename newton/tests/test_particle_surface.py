# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import ParticleSurface, extract_particle_surface
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_test_devices

_TEST_MAX_GRID_CELLS = 32_768
_TEST_MULTI_WORLD_MAX_GRID_CELLS = 49_152


@wp.kernel
def _sample_sparse_field_kernel(
    volume: wp.uint64,
    voxel_data: wp.array[wp.float32],
    coordinates: wp.array[wp.vec3],
    background: float,
    samples: wp.array[wp.float32],
):
    index = wp.tid()
    samples[index] = wp.volume_sample_index(
        volume,
        coordinates[index],
        wp.Volume.LINEAR,
        voxel_data,
        background,
    )


def _make_sphere_particles(n=512, seed=42, device=None):
    """Create a deterministic particle sampling of a unit sphere."""
    rng = np.random.default_rng(seed)
    pts = []
    count = 0
    while count < n:
        candidates = rng.uniform(-1.0, 1.0, size=(2 * n, 3))
        accepted = candidates[np.linalg.norm(candidates, axis=1) < 1.0]
        pts.append(accepted)
        count += accepted.shape[0]
    positions = wp.array(np.concatenate(pts)[:n].astype(np.float32), dtype=wp.vec3, device=device)
    radii = wp.full(n, value=0.05, dtype=wp.float32, device=device)
    return positions, radii


def _assert_nonempty_mesh(test, mesh):
    """Return the arrays of a nonempty triangle mesh."""
    vertices, indices, normals = mesh.to_arrays()
    test.assertIsNotNone(vertices)
    test.assertIsNotNone(indices)
    test.assertGreater(vertices.shape[0], 0)
    test.assertGreater(indices.shape[0], 0)
    test.assertEqual(indices.shape[0] % 3, 0)
    return vertices, indices, normals


def _sample_sparse_field(surface, positions, *, world=0):
    """Sample a public sparse field at world-local positions."""
    sparse_field = surface.sparse_field
    if sparse_field is None:
        raise ValueError("Particle surface field has not been extracted")
    positions = np.asarray(positions, dtype=np.float32)
    offset = sparse_field.world_index_offsets.numpy()[world]
    coordinates = positions / np.float32(surface.voxel_size) + offset
    coordinates = wp.array(coordinates, dtype=wp.vec3, device=sparse_field.voxel_data.device)
    samples = wp.empty(coordinates.shape[0], dtype=wp.float32, device=coordinates.device)
    wp.launch(
        _sample_sparse_field_kernel,
        dim=coordinates.shape[0],
        inputs=[
            sparse_field.volume.id,
            sparse_field.voxel_data,
            coordinates,
            sparse_field.background,
            samples,
        ],
        device=coordinates.device,
    )
    return samples.numpy()


def _sorted_vertices(vertices):
    """Return mesh vertices in deterministic lexicographic order."""
    values = vertices.numpy()
    return values[np.lexsort(values.T[::-1])]


def test_one_shot(test, device):
    """Extract a valid anisotropic mesh with the one-shot public API."""
    positions, radii = _make_sphere_particles(device=device)
    mesh = extract_particle_surface(
        positions,
        radii,
        voxel_size=0.1,
        kernel_radius=0.3,
        anisotropic=True,
        anisotropy_min_neighbors=4,
        anisotropy_binning=True,
    )

    vertices, _indices, normals = _assert_nonempty_mesh(test, mesh)
    test.assertEqual(normals.shape[0], vertices.shape[0])


def test_reusable_context(test, device):
    """Reuse a particle-surface context for consecutive extractions."""
    positions, radii = _make_sphere_particles(n=256, device=device)
    surface = ParticleSurface(
        voxel_size=0.12,
        kernel_radius=0.36,
        mesh_smooth_iterations=1,
        device=device,
    )

    first = surface.extract(positions, radii)
    second = surface.extract(positions, radii)

    first_vertices, first_indices, _first_normals = _assert_nonempty_mesh(test, first)
    second_vertices, second_indices, _second_normals = _assert_nonempty_mesh(test, second)
    test.assertEqual(first_vertices.shape, second_vertices.shape)
    test.assertEqual(first_indices.shape, second_indices.shape)


def test_multi_world_mesh(test, device):
    """Pack independent world meshes into disjoint public offset ranges."""
    positions, radii = _make_sphere_particles(n=256, device=device)
    positions_np = positions.numpy()
    offset = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    combined_positions = wp.array(
        np.concatenate((positions_np, positions_np + offset)),
        dtype=wp.vec3,
        device=device,
    )
    combined_radii = wp.array(
        np.concatenate((radii.numpy(), radii.numpy())),
        dtype=wp.float32,
        device=device,
    )
    particle_world = wp.array(
        np.concatenate((np.zeros(256, dtype=np.int32), np.ones(256, dtype=np.int32))),
        dtype=wp.int32,
        device=device,
    )

    surface = ParticleSurface(
        voxel_size=0.1,
        kernel_radius=0.3,
        anisotropic=True,
        anisotropy_min_neighbors=4,
        anisotropy_binning=True,
        world_count=3,
        device=device,
    )
    mesh = surface.extract(combined_positions, combined_radii, particle_world=particle_world)
    vertices, indices, normals = _assert_nonempty_mesh(test, mesh)

    vertex_offsets = mesh.vertex_world_offsets.numpy()
    index_offsets = mesh.index_world_offsets.numpy()
    test.assertEqual(int(vertex_offsets[-1]), vertices.shape[0])
    test.assertEqual(int(index_offsets[-1]), indices.shape[0])
    test.assertEqual(int(vertex_offsets[1]), int(vertex_offsets[2] - vertex_offsets[1]))
    test.assertEqual(int(index_offsets[1]), int(index_offsets[2] - index_offsets[1]))
    test.assertEqual(int(vertex_offsets[2]), int(vertex_offsets[3]))
    test.assertEqual(int(index_offsets[2]), int(index_offsets[3]))

    vertices_np = vertices.numpy()
    indices_np = indices.numpy()
    for world in range(2):
        world_indices = indices_np[index_offsets[world] : index_offsets[world + 1]]
        test.assertGreaterEqual(int(np.min(world_indices)), int(vertex_offsets[world]))
        test.assertLess(int(np.max(world_indices)), int(vertex_offsets[world + 1]))
    mean_offset = np.mean(vertices_np[vertex_offsets[1] : vertex_offsets[2]], axis=0) - np.mean(
        vertices_np[vertex_offsets[0] : vertex_offsets[1]], axis=0
    )
    np.testing.assert_allclose(mean_offset, offset, atol=2.0e-5)
    test.assertEqual(normals.shape[0], vertices.shape[0])

    sparse_field = surface.sparse_field
    test.assertIsInstance(sparse_field, ParticleSurface.SparseField)
    test.assertEqual(sparse_field.world_index_offsets.shape, (3,))
    np.testing.assert_array_equal(sparse_field.world_index_offsets.numpy()[2], [0, 0, 0])
    np.testing.assert_array_equal(sparse_field.per_world_status.numpy(), [0, 0, 0])


def test_multi_world_fixed_capacity(test, device):
    """Preserve per-world mesh ranges with fixed-capacity extraction."""
    positions, radii = _make_sphere_particles(n=256, device=device)
    positions_np = positions.numpy()
    offset = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    combined_positions = wp.array(
        np.concatenate((positions_np, positions_np + offset)),
        dtype=wp.vec3,
        device=device,
    )
    combined_radii = wp.array(
        np.concatenate((radii.numpy(), radii.numpy())),
        dtype=wp.float32,
        device=device,
    )
    particle_world = wp.array(
        np.concatenate((np.zeros(256, dtype=np.int32), np.ones(256, dtype=np.int32))),
        dtype=wp.int32,
        device=device,
    )
    surface = ParticleSurface(
        voxel_size=0.1,
        kernel_radius=0.3,
        max_grid_cells=_TEST_MULTI_WORLD_MAX_GRID_CELLS,
        world_count=2,
        device=device,
    )

    mesh = surface.extract(combined_positions, combined_radii, particle_world=particle_world)
    vertices, indices, _normals = _assert_nonempty_mesh(test, mesh)

    vertex_offsets = mesh.vertex_world_offsets.numpy()
    index_offsets = mesh.index_world_offsets.numpy()
    test.assertEqual(int(vertex_offsets[-1]), vertices.shape[0])
    test.assertEqual(int(index_offsets[-1]), indices.shape[0])
    test.assertEqual(int(vertex_offsets[1]), int(vertex_offsets[2] - vertex_offsets[1]))
    test.assertEqual(int(index_offsets[1]), int(index_offsets[2] - index_offsets[1]))
    np.testing.assert_array_equal(surface.sparse_field.per_world_status.numpy(), [0, 0])


def test_multi_world_large_topology_halo(test, device):
    """Keep large field-processing halos isolated between packed worlds."""
    positions = wp.zeros(2, dtype=wp.vec3, device=device)
    radii = wp.array([0.18, 0.32], dtype=wp.float32, device=device)
    particle_world = wp.array([0, 1], dtype=wp.int32, device=device)
    options = {
        "voxel_size": 0.1,
        "kernel_radius": 0.3,
        "surface_method": "particle_sdf",
        "particle_sdf_band": 2.0,
        "field_smooth_iterations": 3,
        "field_smooth_radius": 2,
        "redistance_iterations": 4,
        "mesh_smooth_iterations": 0,
        "device": device,
    }
    multi_world_surface = ParticleSurface(world_count=2, **options)
    multi_world_surface.update_field(positions, radii, particle_world=particle_world)

    axis = np.arange(-16, 17, dtype=np.float32) * options["voxel_size"]
    samples = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    for world in range(2):
        with test.subTest(world=world):
            single_world_surface = ParticleSurface(**options)
            single_world_surface.update_field(positions[world : world + 1], radii[world : world + 1])
            actual = _sample_sparse_field(multi_world_surface, samples, world=world)
            expected = _sample_sparse_field(single_world_surface, samples)
            np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_update_field_and_resurface(test, device):
    """Update the public sparse field without a mesh and resurface it later."""
    positions, radii = _make_sphere_particles(n=256, device=device)
    surface = ParticleSurface(
        voxel_size=0.12,
        kernel_radius=0.36,
        field_mode="sdf",
        redistance_iterations=1,
        device=device,
    )

    sparse_field = surface.update_field(positions, radii)

    test.assertIsInstance(sparse_field, ParticleSurface.SparseField)
    test.assertIsInstance(sparse_field.volume, wp.Volume)
    test.assertEqual(sparse_field.voxel_data.dtype, wp.float32)
    test.assertLess(float(sparse_field.voxel_data.numpy().min()), 0.0)
    mesh = surface.resurface(compute_normals=False)
    _vertices, _indices, normals = _assert_nonempty_mesh(test, mesh)
    test.assertIsNone(normals)


def test_capacity_status_and_mesh(test, device):
    """Match exact extraction and report fixed-capacity overflow."""
    positions, radii = _make_sphere_particles(n=256, seed=17, device=device)
    reference = ParticleSurface(
        voxel_size=0.12,
        kernel_radius=0.36,
        device=device,
    )
    reference_mesh = reference.extract(positions, radii, compute_normals=False)
    reference_vertices, reference_indices, _reference_normals = _assert_nonempty_mesh(test, reference_mesh)
    surface = ParticleSurface(
        voxel_size=0.12,
        kernel_radius=0.36,
        max_grid_cells=_TEST_MAX_GRID_CELLS,
        device=device,
    )

    mesh = surface.extract(positions, radii, compute_normals=False)

    vertices, indices, normals = _assert_nonempty_mesh(test, mesh)
    test.assertIsNone(normals)
    test.assertEqual(vertices.shape, reference_vertices.shape)
    test.assertEqual(indices.shape, reference_indices.shape)
    np.testing.assert_allclose(
        _sorted_vertices(vertices), _sorted_vertices(reference_vertices), rtol=1.0e-6, atol=1.0e-6
    )
    axis = np.arange(-10, 11, 2, dtype=np.float32) * surface.voxel_size
    samples = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(
        _sample_sparse_field(surface, samples),
        _sample_sparse_field(reference, samples),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(surface.sparse_field.per_world_status.numpy(), [0])

    overflow_surface = ParticleSurface(
        voxel_size=0.12,
        kernel_radius=0.36,
        max_grid_cells=512,
        device=device,
    )
    overflow_mesh = overflow_surface.extract(positions[:1], radii[:1], compute_normals=False)
    test.assertNotEqual(int(overflow_surface.sparse_field.per_world_status.numpy()[0]), 0)
    with test.assertRaisesRegex(ValueError, "exceeds configured max_grid_cells"):
        overflow_mesh.to_arrays()


def test_isotropic_support_is_not_clipped(test, device):
    """Preserve the complete isotropic surface in exact and capacity modes."""
    positions = wp.zeros(1, dtype=wp.vec3, device=device)
    radii = wp.array([1.0], dtype=wp.float32, device=device)
    meshes = []
    for max_grid_cells in (None, 10_000):
        with test.subTest(max_grid_cells=max_grid_cells):
            surface = ParticleSurface(
                voxel_size=1.0,
                max_grid_cells=max_grid_cells,
                kernel_radius=3.0,
                kernel_scale=1.0,
                threshold=0.01,
                anisotropic=False,
                padding=0,
                field_smooth_iterations=0,
                mesh_smooth_iterations=0,
                device=device,
            )
            vertices, indices, _normals = _assert_nonempty_mesh(
                test,
                surface.extract(positions, radii, compute_normals=False),
            )
            test.assertEqual(vertices.shape[0], 270)
            test.assertEqual(indices.shape[0], 1608)
            test.assertGreater(float(np.min(np.max(np.abs(vertices.numpy()), axis=0))), 3.8)
            np.testing.assert_array_equal(surface.sparse_field.per_world_status.numpy(), [0])
            meshes.append(_sorted_vertices(vertices))
    np.testing.assert_allclose(meshes[0], meshes[1], rtol=1.0e-6, atol=1.0e-6)


def test_sparse_grid_handles_distant_particles(test, device):
    """Extract distant particles without reserving their empty intervening span."""
    positions = wp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    radii = wp.full(2, value=0.05, dtype=wp.float32, device=device)
    surface = ParticleSurface(
        voxel_size=0.1,
        kernel_radius=0.3,
        smooth_lambda=0.0,
        max_grid_cells=4_096,
        device=device,
    )

    sparse_field = surface.update_field(positions, radii)

    test.assertIsInstance(sparse_field, ParticleSurface.SparseField)
    np.testing.assert_array_equal(sparse_field.per_world_status.numpy(), [0])
    test.assertTrue(np.all(_sample_sparse_field(surface, [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]) > 0.0))


def test_empty_particles(test, device):
    """Return an empty result for an empty particle set."""
    positions = wp.empty(0, dtype=wp.vec3, device=device)
    radii = wp.empty(0, dtype=wp.float32, device=device)
    surface = ParticleSurface(voxel_size=0.1, device=device)

    vertices, indices, normals = surface.extract(positions, radii)

    test.assertIsNone(vertices)
    test.assertIsNone(indices)
    test.assertIsNone(normals)


def test_nonfinite_positions_are_skipped(test, device):
    """Ignore nonfinite particles without changing the valid sparse field."""
    reference_positions = wp.zeros(1, dtype=wp.vec3, device=device)
    reference_radii = wp.array([0.05], dtype=wp.float32, device=device)
    positions = wp.array(
        [[0.0, 0.0, 0.0], [np.inf, np.nan, -np.inf]],
        dtype=wp.vec3,
        device=device,
    )
    radii = wp.array([0.05, 0.05], dtype=wp.float32, device=device)
    reference = ParticleSurface(voxel_size=0.05, kernel_radius=0.15, device=device)
    surface = ParticleSurface(voxel_size=0.05, kernel_radius=0.15, device=device)
    reference.update_field(reference_positions, reference_radii)
    surface.update_field(positions, radii)

    axis = np.arange(-5, 6, dtype=np.float32) * surface.voxel_size
    samples = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(
        _sample_sparse_field(surface, samples),
        _sample_sparse_field(reference, samples),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_radii_length_mismatch(test, device):
    """Reject radii whose length does not match the particle positions."""
    positions = wp.zeros(4, dtype=wp.vec3, device=device)
    radii = wp.full(3, value=0.05, dtype=wp.float32, device=device)
    surface = ParticleSurface(voxel_size=0.1, device=device)

    with test.assertRaisesRegex(ValueError, "radii length"):
        surface.extract(positions, radii, compute_normals=False)


def test_radii_device_mismatch(test, device):
    """Reject radii stored on a different device than the positions."""
    if not wp.is_cuda_available():
        test.skipTest("requires CUDA for cross-device validation")

    positions_device = wp.get_device(device)
    radii_device = wp.get_device("cuda:0") if positions_device.is_cpu else wp.get_device("cpu")
    positions = wp.zeros(4, dtype=wp.vec3, device=positions_device)
    radii = wp.full(4, value=0.05, dtype=wp.float32, device=radii_device)
    surface = ParticleSurface(voxel_size=0.1, device=positions_device)

    with test.assertRaisesRegex(ValueError, "radii device"):
        surface.extract(positions, radii, compute_normals=False)


def test_array_layout_validation(test, device):
    """Reject arrays with unsupported dimensions or element types."""
    positions = wp.zeros(4, dtype=wp.vec3, device=device)
    radii = wp.full(4, value=0.05, dtype=wp.float32, device=device)
    surface = ParticleSurface(voxel_size=0.1, device=device)

    with test.assertRaisesRegex(ValueError, "positions must be a 1-D array"):
        surface.extract(wp.zeros((4, 3), dtype=wp.float32, device=device), radii)
    with test.assertRaisesRegex(TypeError, "positions must have dtype wp.vec3"):
        surface.extract(wp.zeros(4, dtype=wp.float32, device=device), radii)
    with test.assertRaisesRegex(TypeError, "radii must have dtype wp.float32"):
        surface.extract(positions, wp.full(4, value=0.05, dtype=wp.float64, device=device))
    with test.assertRaisesRegex(TypeError, "particle_flags must have dtype wp.int32"):
        surface.extract(positions, radii, particle_flags=wp.ones(4, dtype=wp.float32, device=device))
    with test.assertRaisesRegex(TypeError, "particle_world must have dtype wp.int32"):
        surface.extract(positions, radii, particle_world=wp.zeros(4, dtype=wp.float32, device=device))


def test_cuda_graph_extraction(test, device):
    """Capture and replay fixed-capacity extraction in a CUDA graph."""
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("requires CUDA graph capture")

    positions, radii = _make_sphere_particles(n=256, seed=9, device=device)
    surface = ParticleSurface(
        voxel_size=0.12,
        kernel_radius=0.36,
        anisotropic=True,
        anisotropy_min_neighbors=4,
        anisotropy_binning=True,
        field_mode="sdf",
        redistance_iterations=1,
        max_grid_cells=_TEST_MAX_GRID_CELLS,
        device=device,
    )
    initial_mesh = surface.extract(positions, radii, compute_normals=False)
    initial_vertices, _initial_indices, _initial_normals = _assert_nonempty_mesh(test, initial_mesh)
    initial_mean = np.mean(initial_vertices.numpy(), axis=0)

    with wp.ScopedCapture(device=device) as capture:
        captured_mesh = surface.extract(positions, radii, compute_normals=False)

    translation = np.array([4.0 * surface.voxel_size, 0.0, 0.0], dtype=np.float32)
    moved_positions = wp.array(positions.numpy() + translation, dtype=wp.vec3, device=device)
    wp.copy(positions, moved_positions)
    wp.capture_launch(capture.graph)

    captured_vertices, _captured_indices, _captured_normals = _assert_nonempty_mesh(test, captured_mesh)
    np.testing.assert_allclose(np.mean(captured_vertices.numpy(), axis=0) - initial_mean, translation, atol=2.0e-5)


def test_particle_sdf_surface_method(test, device):
    """Extract a signed-distance surface with the particle-SDF method."""
    positions, radii = _make_sphere_particles(n=256, device=device)
    surface = ParticleSurface(
        voxel_size=0.12,
        kernel_radius=0.36,
        surface_method="particle_sdf",
        particle_sdf_radius_scale=1.8,
        anisotropic=True,
        anisotropy_min_neighbors=4,
        device=device,
    )

    mesh = surface.extract(positions, radii, compute_normals=False)

    _vertices, _indices, normals = _assert_nonempty_mesh(test, mesh)
    test.assertIsNone(normals)
    test.assertEqual(surface.field_mode, "sdf")
    field = surface.sparse_field.voxel_data.numpy()
    test.assertLess(float(field.min()), 0.0)
    test.assertGreater(float(field.max()), 0.0)


def test_isotropic_particle_sdf_samples(test, device):
    """Return analytic signed distances for one isotropic particle."""
    positions = wp.zeros(1, dtype=wp.vec3, device=device)
    radii = wp.array([0.5], dtype=wp.float32, device=device)
    surface = ParticleSurface(
        voxel_size=1.0,
        kernel_radius=3.0,
        smooth_lambda=0.0,
        surface_method="particle_sdf",
        particle_sdf_band=2.0,
        device=device,
    )
    surface.update_field(positions, radii)

    samples = _sample_sparse_field(
        surface,
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [10.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(samples, [-0.5, 0.5, 0.5, 0.5, 6.0], rtol=1.0e-6, atol=1.0e-6)


def test_particle_flags_filter_inactive(test, device):
    """Exclude particles without the public ACTIVE flag from extraction."""
    active_positions, active_radii = _make_sphere_particles(n=256, seed=11, device=device)
    active_np = active_positions.numpy()
    positions_np = np.concatenate((active_np, np.full_like(active_np, np.nan)))
    radii_np = np.concatenate((active_radii.numpy(), np.full(active_radii.shape[0], np.nan, dtype=np.float32)))
    flags_np = np.concatenate(
        (
            np.full(active_np.shape[0], int(newton.ParticleFlags.ACTIVE), dtype=np.int32),
            np.zeros(active_np.shape[0], dtype=np.int32),
        )
    )
    positions = wp.array(positions_np, dtype=wp.vec3, device=device)
    radii = wp.array(radii_np, dtype=wp.float32, device=device)
    flags = wp.array(flags_np, dtype=wp.int32, device=device)
    surface = ParticleSurface(voxel_size=0.12, kernel_radius=0.36, device=device)

    mesh = surface.extract(positions, radii, compute_normals=False, particle_flags=flags)

    vertices, indices, _normals = _assert_nonempty_mesh(test, mesh)
    reference_surface = ParticleSurface(voxel_size=0.12, kernel_radius=0.36, device=device)
    reference_vertices, reference_indices, _reference_normals = _assert_nonempty_mesh(
        test,
        reference_surface.extract(active_positions, active_radii, compute_normals=False),
    )
    test.assertEqual(vertices.shape, reference_vertices.shape)
    test.assertEqual(indices.shape, reference_indices.shape)

    inactive_flags = wp.zeros(positions.shape[0], dtype=wp.int32, device=device)
    empty_mesh = surface.extract(positions, radii, compute_normals=False, particle_flags=inactive_flags)
    test.assertEqual(empty_mesh.to_arrays(), (None, None, None))


def _build_mpm_solver(device, *, world_count=1):
    """Build a small MPM particle block for solver integration tests."""
    blueprint = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(blueprint)
    blueprint.add_particle_grid(
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
    blueprint.add_ground_plane()

    if world_count == 1:
        builder = blueprint
    else:
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        for world in range(world_count):
            builder.add_world(
                blueprint,
                xform=wp.transform(wp.vec3(2.0 * world, 0.0, 0.0), wp.quat_identity()),
            )
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config()
    config.grid_type = "dense"
    config.voxel_size = 0.1
    config.separate_worlds = world_count > 1
    return model, SolverImplicitMPM(model, config)


def test_solver_extract_particle_surface(test, device):
    """Extract particles and collider-extended SDFs through the MPM solver API."""
    model, solver = _build_mpm_solver(device)
    state = model.state()
    default_surface = solver.create_particle_surface()
    test.assertAlmostEqual(default_surface.voxel_size, 0.045)

    surface = solver.create_particle_surface(
        voxel_size=0.08,
        kernel_radius=0.24,
        field_smooth_iterations=0,
    )
    mesh = solver.extract_particle_surface(state, surface, compute_normals=False)
    _vertices, _indices, normals = _assert_nonempty_mesh(test, mesh)
    test.assertIsNone(normals)

    inactive_flags = wp.zeros(model.particle_count, dtype=wp.int32, device=device)
    empty_mesh = solver.extract_particle_surface(
        state,
        surface,
        compute_normals=False,
        particle_flags=inactive_flags,
    )
    test.assertEqual(empty_mesh.to_arrays(), (None, None, None))

    collider_surface = solver.create_particle_surface(
        voxel_size=0.08,
        max_grid_cells=_TEST_MAX_GRID_CELLS,
        kernel_radius=0.24,
        field_smooth_iterations=0,
        field_mode="sdf",
        redistance_iterations=1,
    )
    collider_mesh = solver.extract_particle_surface(
        state,
        collider_surface,
        compute_normals=False,
        extrapolate_into_colliders=True,
        collider_extrapolation_depth=0.08,
    )
    _collider_vertices, _collider_indices, collider_normals = _assert_nonempty_mesh(test, collider_mesh)
    test.assertIsNone(collider_normals)

    with test.assertRaisesRegex(ValueError, "max_depth|topology halo"):
        solver.extract_particle_surface(
            state,
            collider_surface,
            compute_normals=False,
            extrapolate_into_colliders=True,
            collider_extrapolation_depth=float("inf"),
        )


def test_solver_extract_particle_surface_multi_world(test, device):
    """Preserve per-world mesh ranges through MPM surface extraction."""
    model, solver = _build_mpm_solver(device, world_count=2)
    surface = solver.create_particle_surface(
        voxel_size=0.08,
        kernel_radius=0.24,
        field_smooth_iterations=0,
    )

    mesh = solver.extract_particle_surface(model.state(), surface, compute_normals=False)
    vertices, indices, normals = _assert_nonempty_mesh(test, mesh)

    vertex_offsets = mesh.vertex_world_offsets.numpy()
    index_offsets = mesh.index_world_offsets.numpy()
    test.assertEqual(surface.world_count, 2)
    test.assertEqual(int(vertex_offsets[-1]), vertices.shape[0])
    test.assertEqual(int(index_offsets[-1]), indices.shape[0])
    test.assertEqual(int(vertex_offsets[1]), int(vertex_offsets[2] - vertex_offsets[1]))
    test.assertEqual(int(index_offsets[1]), int(index_offsets[2] - index_offsets[1]))
    test.assertIsNone(normals)


class TestParticleSurface(unittest.TestCase):
    def test_constructor_rejects_invalid_parameters(self):
        """Reject invalid particle-surface configuration values."""
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
            ({"voxel_size": 0.1, "particle_sdf_band": 0.5}, "particle_sdf_band"),
            ({"voxel_size": 0.1, "world_count": 0}, "world_count"),
            ({"voxel_size": 0.1, "field_smooth_iterations": -1}, "field_smooth_iterations"),
            ({"voxel_size": 0.1, "mesh_smooth_lambda": 1.5}, "mesh_smooth_lambda"),
            ({"voxel_size": 0.1, "surface_method": "invalid"}, "surface_method"),
            (
                {"voxel_size": 0.1, "surface_method": "particle_sdf", "field_mode": "density"},
                "surface_method='particle_sdf'",
            ),
        ]
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                ParticleSurface(device="cpu", **kwargs)


devices = get_test_devices(mode="basic")

add_function_test(TestParticleSurface, "test_one_shot", test_one_shot, devices=devices)
add_function_test(TestParticleSurface, "test_reusable_context", test_reusable_context, devices=devices)
add_function_test(TestParticleSurface, "test_multi_world_mesh", test_multi_world_mesh, devices=devices)
add_function_test(
    TestParticleSurface,
    "test_multi_world_fixed_capacity",
    test_multi_world_fixed_capacity,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_multi_world_large_topology_halo",
    test_multi_world_large_topology_halo,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_update_field_and_resurface",
    test_update_field_and_resurface,
    devices=devices,
)
add_function_test(TestParticleSurface, "test_capacity_status_and_mesh", test_capacity_status_and_mesh, devices=devices)
add_function_test(
    TestParticleSurface,
    "test_isotropic_support_is_not_clipped",
    test_isotropic_support_is_not_clipped,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_sparse_grid_handles_distant_particles",
    test_sparse_grid_handles_distant_particles,
    devices=devices,
)
add_function_test(TestParticleSurface, "test_empty_particles", test_empty_particles, devices=devices)
add_function_test(
    TestParticleSurface,
    "test_nonfinite_positions_are_skipped",
    test_nonfinite_positions_are_skipped,
    devices=devices,
)
add_function_test(TestParticleSurface, "test_radii_length_mismatch", test_radii_length_mismatch, devices=devices)
add_function_test(TestParticleSurface, "test_radii_device_mismatch", test_radii_device_mismatch, devices=devices)
add_function_test(TestParticleSurface, "test_array_layout_validation", test_array_layout_validation, devices=devices)
add_function_test(TestParticleSurface, "test_cuda_graph_extraction", test_cuda_graph_extraction, devices=devices)
add_function_test(
    TestParticleSurface,
    "test_particle_sdf_surface_method",
    test_particle_sdf_surface_method,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_isotropic_particle_sdf_samples",
    test_isotropic_particle_sdf_samples,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_particle_flags_filter_inactive",
    test_particle_flags_filter_inactive,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_solver_extract_particle_surface",
    test_solver_extract_particle_surface,
    devices=devices,
)
add_function_test(
    TestParticleSurface,
    "test_solver_extract_particle_surface_multi_world",
    test_solver_extract_particle_surface_multi_world,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
