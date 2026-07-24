# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton import Heightfield
from newton._src.utils import is_graph_capture_allocation_enabled
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import assert_np_equal

_cuda_available = wp.is_cuda_available()


class TestHeightfield(unittest.TestCase):
    """Test suite for heightfield support."""

    def test_heightfield_creation(self):
        """Test creating a Heightfield with auto-normalization."""
        nrow, ncol = 10, 10
        raw_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32) * 5.0  # 0-5 meters

        hfield = Heightfield(data=raw_data, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0)

        self.assertEqual(hfield.nrow, nrow)
        self.assertEqual(hfield.ncol, ncol)
        self.assertEqual(hfield.hx, 5.0)
        self.assertEqual(hfield.hy, 5.0)
        self.assertEqual(hfield.data.dtype, np.float32)
        self.assertEqual(hfield.data.shape, (nrow, ncol))

        # Data should be normalized to [0, 1]
        self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=5)
        self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=5)

        # min_z/max_z should be auto-derived from raw data
        self.assertAlmostEqual(hfield.min_z, float(raw_data.min()), places=5)
        self.assertAlmostEqual(hfield.max_z, float(raw_data.max()), places=5)

    def test_heightfield_explicit_z_range(self):
        """Test creating a Heightfield with explicit min_z/max_z."""
        nrow, ncol = 5, 5
        data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)

        hfield = Heightfield(data=data, nrow=nrow, ncol=ncol, hx=3.0, hy=3.0, min_z=-1.0, max_z=4.0)

        self.assertEqual(hfield.min_z, -1.0)
        self.assertEqual(hfield.max_z, 4.0)
        # Data still normalized
        self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=5)
        self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=5)

    def test_heightfield_flat(self):
        """Test that flat (constant) data produces zeros."""
        nrow, ncol = 5, 5
        flat_data = np.full((nrow, ncol), 3.0, dtype=np.float32)

        hfield = Heightfield(data=flat_data, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)

        assert_np_equal(hfield.data, np.zeros((nrow, ncol)), tol=1e-6)
        self.assertAlmostEqual(hfield.min_z, 3.0, places=5)
        self.assertAlmostEqual(hfield.max_z, 3.0, places=5)

    def test_heightfield_hash(self):
        """Test that heightfield hashing works for deduplication."""
        nrow, ncol = 5, 5
        data_a = np.array([[i + j for j in range(ncol)] for i in range(nrow)], dtype=np.float32)
        data_b = np.array([[i + j for j in range(ncol)] for i in range(nrow)], dtype=np.float32)
        data_c = np.array([[i * j for j in range(ncol)] for i in range(nrow)], dtype=np.float32)

        hfield1 = Heightfield(data=data_a, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)
        hfield2 = Heightfield(data=data_b, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)
        hfield3 = Heightfield(data=data_c, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)

        # Same data should produce same hash
        self.assertEqual(hash(hfield1), hash(hfield2))

        # Different data should produce different hash
        self.assertNotEqual(hash(hfield1), hash(hfield3))

    def test_add_shape_heightfield(self):
        """Test adding a heightfield shape via ModelBuilder."""
        builder = newton.ModelBuilder()

        nrow, ncol = 8, 8
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)
        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

        shape_id = builder.add_shape_heightfield(heightfield=hfield)

        self.assertGreaterEqual(shape_id, 0)
        self.assertEqual(builder.shape_count, 1)
        self.assertEqual(builder.shape_type[shape_id], newton.GeoType.HFIELD)
        self.assertIs(builder.shape_source[shape_id], hfield)

    def test_model_heightfield_count_and_deprecated_alias(self):
        """Model exposes heightfield_count and warns for the legacy boolean."""
        builder = newton.ModelBuilder()
        hfield = Heightfield(data=np.zeros((3, 3), dtype=np.float32), nrow=3, ncol=3, hx=1.0, hy=1.0)
        builder.add_shape_heightfield(heightfield=hfield)

        model = builder.finalize(device="cpu")

        self.assertEqual(model.heightfield_count, 1)
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(model.has_heightfields)

        empty_model = newton.Model(device="cpu")
        self.assertEqual(empty_model.heightfield_count, 0)
        with self.assertWarns(DeprecationWarning):
            self.assertFalse(empty_model.has_heightfields)

        with self.assertWarns(DeprecationWarning):
            empty_model.has_heightfields = True
        self.assertEqual(empty_model.heightfield_count, 1)

        with self.assertWarns(DeprecationWarning):
            empty_model.has_heightfields = False
        self.assertEqual(empty_model.heightfield_count, 0)

    def test_mjcf_hfield_parsing(self):
        """Test parsing MJCF file with hfield asset."""
        mjcf = """
        <mujoco model="test_heightfield">
          <compiler autolimits="true"/>
          <asset>
            <hfield name="terrain" nrow="10" ncol="10"
                    size="5 5 1 0"/>
          </asset>
          <worldbody>
            <geom type="hfield" hfield="terrain"/>
            <body pos="0 0 2">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_meshes=True)

        hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]
        self.assertEqual(len(hfield_shapes), 1)

        hfield = builder.shape_source[hfield_shapes[0]]
        self.assertIsInstance(hfield, Heightfield)
        self.assertEqual(hfield.nrow, 10)
        self.assertEqual(hfield.ncol, 10)
        # MuJoCo size (5, 5, 1, 0) → hx=5, hy=5, min_z=0, max_z=1
        self.assertAlmostEqual(hfield.hx, 5.0)
        self.assertAlmostEqual(hfield.hy, 5.0)
        self.assertAlmostEqual(hfield.min_z, 0.0)
        self.assertAlmostEqual(hfield.max_z, 1.0)

        # Data should be all zeros (no file, no elevation → flat)
        assert_np_equal(hfield.data, np.zeros((10, 10)), tol=1e-6)

    def test_mjcf_hfield_binary_file(self):
        """Test parsing MJCF with binary heightfield file."""
        nrow, ncol = 4, 6
        rng = np.random.default_rng(42)
        elevation = rng.random((nrow, ncol)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tmp_path = f.name
            np.array([nrow, ncol], dtype=np.int32).tofile(f)
            elevation.tofile(f)

        def resolver(_base_dir, _file_path):
            return tmp_path

        mjcf = """
        <mujoco>
          <asset>
            <hfield name="terrain" nrow="4" ncol="6"
                    size="3 2 1 0" file="terrain.bin"/>
          </asset>
          <worldbody>
            <geom type="hfield" hfield="terrain"/>
          </worldbody>
        </mujoco>
        """

        try:
            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf, parse_meshes=True, path_resolver=resolver)

            hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]
            self.assertEqual(len(hfield_shapes), 1)

            hfield = builder.shape_source[hfield_shapes[0]]
            self.assertEqual(hfield.nrow, nrow)
            self.assertEqual(hfield.ncol, ncol)
            self.assertAlmostEqual(hfield.hx, 3.0)
            self.assertAlmostEqual(hfield.hy, 2.0)
            # Data is normalized — check shape and range
            self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=4)
            self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=4)
        finally:
            os.unlink(tmp_path)

    def test_mjcf_hfield_inline_elevation(self):
        """Test parsing MJCF with inline elevation attribute."""
        mjcf = """
        <mujoco>
          <asset>
            <hfield name="terrain" nrow="3" ncol="3"
                    size="2 2 1 0"
                    elevation="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"/>
          </asset>
          <worldbody>
            <geom type="hfield" hfield="terrain"/>
          </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_meshes=True)

        hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]
        self.assertEqual(len(hfield_shapes), 1)

        hfield = builder.shape_source[hfield_shapes[0]]
        self.assertEqual(hfield.nrow, 3)
        self.assertEqual(hfield.ncol, 3)
        # Data is normalized from [0.1, 0.9] to [0, 1]
        self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=5)
        self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=5)
        self.assertAlmostEqual(hfield.min_z, -0.0)  # size_base=0 → min_z=0
        self.assertAlmostEqual(hfield.max_z, 1.0)  # size_z=1 → max_z=1

    def test_solver_mujoco_hfield(self):
        """Test converting Newton model with heightfield to MuJoCo."""
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo not installed")

        builder = newton.ModelBuilder()

        nrow, ncol = 5, 5
        elevation_data = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=2.0, hy=2.0, min_z=0.0, max_z=0.5)

        builder.add_shape_heightfield(heightfield=hfield)

        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)

        model = builder.finalize()

        try:
            newton.solvers.SolverMuJoCo(model)
        except Exception as e:
            self.fail(f"Failed to create MuJoCo solver with heightfield: {e}")

    def test_heightfield_collision(self):
        """Test that a sphere doesn't fall through a heightfield."""
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo not installed")

        builder = newton.ModelBuilder()

        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield)

        sphere_radius = 0.1
        start_z = 0.5
        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, start_z), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)

        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(model)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        sim_dt = 1.0 / 240.0

        device = model.device
        use_graph = is_graph_capture_allocation_enabled(device)
        if use_graph:
            # warmup (2 steps for full ping-pong cycle)
            solver.step(state_in, state_out, control, None, sim_dt)
            solver.step(state_out, state_in, control, None, sim_dt)
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, sim_dt)
                solver.step(state_out, state_in, control, None, sim_dt)
            graph = capture.graph

        remaining = 500 - (4 if use_graph else 0)
        for _ in range(remaining // 2 if use_graph else remaining):
            if use_graph:
                wp.capture_launch(graph)
            else:
                solver.step(state_in, state_out, control, None, sim_dt)
                state_in, state_out = state_out, state_in
        if use_graph and remaining % 2 == 1:
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        final_z = float(state_in.body_q.numpy()[sphere_body, 2])

        self.assertGreater(
            final_z,
            -sphere_radius,
            f"Sphere fell through heightfield: z={final_z:.4f}",
        )

    def test_heightfield_always_static(self):
        """Test that heightfields are always static (zero mass, zero inertia)."""
        nrow, ncol = 10, 10
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)

        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0)

        self.assertEqual(hfield.mass, 0.0)
        self.assertFalse(hfield.has_inertia)

    def test_heightfield_radius_computation(self):
        """Test bounding sphere radius computation for heightfield."""
        from newton._src.geometry.utils import compute_shape_radius  # noqa: PLC0415

        nrow, ncol = 10, 10
        elevation_data = np.zeros((nrow, ncol), dtype=np.float32)

        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=4.0, hy=3.0, min_z=0.0, max_z=2.0)

        scale = (1.0, 1.0, 1.0)
        radius = compute_shape_radius(newton.GeoType.HFIELD, scale, hfield)

        # Expected: sqrt(hx^2 + hy^2 + max(|min_z|, |max_z|)^2)
        expected_radius = np.sqrt(4.0**2 + 3.0**2 + max(abs(0.0), abs(2.0)) ** 2)
        self.assertAlmostEqual(radius, expected_radius, places=5)

    def test_heightfield_native_collision_flat(self):
        """Test native CollisionPipeline detects contact between sphere and flat heightfield."""
        builder = newton.ModelBuilder()

        # Flat heightfield at z=0
        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield)

        # Sphere slightly above the heightfield surface
        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)

        model = builder.finalize()
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        # Should detect at least one contact (sphere is within contact margin of heightfield)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0, "No contacts detected between sphere and heightfield")

    def test_heightfield_native_collision_scaled(self):
        """Per-instance ``scale`` on ``add_shape_heightfield`` is honored by narrow-phase.

        The sphere sits at XY=(1.5, 0) -- inside the scaled extent ``[-2, 2]`` but
        outside the unscaled asset extent ``[-1, 1]``. A pre-fix build (narrow-phase
        ignoring ``scale``) would treat the sphere as outside the heightfield
        footprint and generate no contacts.
        """
        builder = newton.ModelBuilder()

        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        # Small heightfield (hx=hy=1) scaled 2x in XY; baked extent becomes [-2, 2].
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield, scale=(2.0, 2.0, 1.0))

        # Sphere straddling the scaled surface at XY=(1.5, 0).
        sphere_body = builder.add_body(xform=wp.transform((1.5, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)

        model = builder.finalize()
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0, "No contacts detected between sphere and scaled heightfield")

    def test_heightfield_native_collision_no_contact(self):
        """Test that no contacts are generated when sphere is far above heightfield."""
        builder = newton.ModelBuilder()

        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield)

        # Sphere far above the heightfield
        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, 5.0), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)

        model = builder.finalize()
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(contact_count, 0, f"Unexpected contacts detected: {contact_count}")

    @staticmethod
    def _create_non_convex_mesh() -> newton.Mesh:
        """Create a non-convex spike mesh from a tetrahedron base (no SDF)."""
        base_vertices = np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
            dtype=np.float32,
        )
        base_vertices /= np.linalg.norm(base_vertices, axis=1, keepdims=True)
        base_vertices *= 0.3

        faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
        vertices: list[np.ndarray] = []
        indices: list[int] = []
        for face in faces:
            a, b, c = (base_vertices[i] for i in face)
            normal = np.cross(b - a, c - a)
            normal /= np.linalg.norm(normal)
            centroid = (a + b + c) / 3.0
            if np.dot(normal, centroid) < 0.0:
                b, c = c, b
                normal = -normal
            apex = (a + b + c) / 3.0 + normal * 0.4
            idx = len(vertices)
            vertices.extend([a, b, c, apex])
            indices.extend(
                [idx, idx + 1, idx + 2, idx, idx + 1, idx + 3, idx + 1, idx + 2, idx + 3, idx + 2, idx, idx + 3]
            )
        return newton.Mesh(
            vertices=np.asarray(vertices, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.int32),
        )

    def _build_mesh_vs_heightfield(self, mesh: newton.Mesh, mesh_z: float = 0.15):
        """Build a model with a non-convex mesh above a flat heightfield."""
        builder = newton.ModelBuilder()
        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield)
        mesh_body = builder.add_body(xform=wp.transform((0.0, 0.0, mesh_z), wp.quat_identity()))
        builder.add_shape_mesh(body=mesh_body, mesh=mesh)
        return builder.finalize(), mesh_body

    @unittest.skipUnless(_cuda_available, "mesh-heightfield collision requires CUDA")
    def test_non_convex_mesh_vs_heightfield(self):
        """Test non-convex mesh (no SDF) generates contacts against a flat heightfield."""
        mesh = self._create_non_convex_mesh()
        model, _mesh_body = self._build_mesh_vs_heightfield(mesh)
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0, "No contacts between non-convex mesh and heightfield")

    @unittest.skipUnless(_cuda_available, "build_sdf requires CUDA")
    def test_non_convex_mesh_with_sdf_vs_heightfield(self):
        """Test non-convex mesh (with SDF) generates contacts against a flat heightfield."""
        mesh = self._create_non_convex_mesh()
        mesh.build_sdf(max_resolution=16)
        model, _mesh_body = self._build_mesh_vs_heightfield(mesh)
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0, "No contacts between SDF mesh and heightfield")

    @unittest.skipUnless(_cuda_available, "mesh-heightfield collision requires CUDA")
    def test_non_convex_mesh_vs_heightfield_no_contact(self):
        """Test no contacts when non-convex mesh is far above heightfield."""
        mesh = self._create_non_convex_mesh()
        model, _mesh_body = self._build_mesh_vs_heightfield(mesh, mesh_z=5.0)
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(contact_count, 0, f"Unexpected contacts: {contact_count}")

    def test_particle_heightfield_soft_contacts(self):
        """Test that particles generate soft contacts against heightfield via on-the-fly SDF."""
        builder = newton.ModelBuilder()
        hfield = Heightfield(
            data=np.zeros((8, 8), dtype=np.float32),
            nrow=8,
            ncol=8,
            hx=2.0,
            hy=2.0,
            min_z=0.0,
            max_z=1.0,
        )
        hfield_shape = builder.add_shape_heightfield(heightfield=hfield)
        builder.add_particle(pos=(0.0, 0.0, 0.02), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05)

        model = builder.finalize()
        state = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        soft_count = int(contacts.soft_contact_count.numpy()[0])
        self.assertGreater(soft_count, 0)
        self.assertEqual(int(contacts.soft_contact_shape.numpy()[0]), hfield_shape)

    def test_create_from_mesh_sloped_plane(self):
        """Rasterize a sloped plane mesh and verify sampled heights and placement."""
        # A single-valued surface z = 0.5*x + 0.25*y over [0, 4] x [0, 8].
        xs = np.linspace(0.0, 4.0, 9, dtype=np.float32)
        ys = np.linspace(0.0, 8.0, 17, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        gz = 0.5 * gx + 0.25 * gy
        verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)

        rows, cols = gx.shape
        faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                v00 = r * cols + c
                v10 = v00 + 1
                v01 = v00 + cols
                v11 = v01 + 1
                faces += [v00, v10, v11, v00, v11, v01]
        mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3),
            indices=wp.array(np.array(faces, dtype=np.int32), dtype=wp.int32),
        )

        hfield, xform = newton.Heightfield.create_from_mesh(mesh, resolution=0.5)

        # Grid dimensions: col -> x (extent 4), row -> y (extent 8) at 0.5 m spacing.
        self.assertEqual(hfield.ncol, 9)
        self.assertEqual(hfield.nrow, 17)
        self.assertAlmostEqual(hfield.hx, 2.0, places=5)
        self.assertAlmostEqual(hfield.hy, 4.0, places=5)

        # Placement centers the origin-centered grid on the mesh XY center.
        origin = wp.transform_get_translation(xform)
        self.assertAlmostEqual(origin[0], 2.0, places=4)
        self.assertAlmostEqual(origin[1], 4.0, places=4)

        # World heights (denormalized) must match the analytic plane at every sample.
        world = hfield.min_z + hfield.data * (hfield.max_z - hfield.min_z)
        expected = 0.5 * gx + 0.25 * gy
        assert_np_equal(world, expected.astype(np.float32), tol=1e-3)

    def test_rasterize_mesh_rejects_small_max_cells_per_axis(self):
        """Reject a maximum grid dimension smaller than two."""
        mesh = wp.Mesh(
            points=wp.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=wp.vec3,
            ),
            indices=wp.array([0, 1, 2], dtype=wp.int32),
        )

        with self.assertRaisesRegex(ValueError, "max_cells_per_axis must be at least 2"):
            newton.utils.rasterize_mesh_to_heightfield(mesh, resolution=0.5, max_cells_per_axis=1)

    def test_rasterize_mesh_missed_rays_use_floor(self):
        """Rasterize a mesh with a hole and verify missed rays fall back to min Z."""
        # A flat quad at z = 1 covering only the +x half (x in [1, 2]) of the bounds,
        # plus a lone low vertex at the origin so the mesh minimum Z is 0.
        verts = np.array(
            [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [1.0, 2.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3), indices=wp.array(faces, dtype=wp.int32))

        heights, bounds = newton.utils.rasterize_mesh_to_heightfield(mesh, resolution=0.5)
        self.assertEqual(bounds, (0.0, 0.0, 2.0, 2.0))

        # Columns over the covered half (x >= 1) hit the quad at z = 1; columns over
        # the uncovered half (x < 1) miss and fall back to the mesh minimum Z (0).
        # Grid columns sample x = [0, 0.5, 1.0, 1.5, 2.0].
        expected = np.tile([0.0, 0.0, 1.0, 1.0, 1.0], (heights.shape[0], 1)).astype(np.float32)
        assert_np_equal(heights, expected, tol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
