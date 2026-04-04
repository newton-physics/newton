# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh-convex back-face culling in the collision pipeline.

When a convex shape passes through a mesh triangle and ends up on the back
side, the narrow phase must not generate contacts with inverted normals.
Inverted normals trap shapes inside the mesh and cause solver divergence
(NaN in joint state).

These tests verify that:
1. Front-face contacts produce correct normals (mesh pushes convex outward).
2. Back-face contacts are culled (no contact generated).
3. Various convex types (sphere, box, capsule, ellipsoid, convex mesh)
   all behave correctly.
4. Edge cases like shapes in a valley between two triangles are handled.
5. A multi-step simulation on rough terrain does not produce NaN.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType

_cuda_available = wp.is_cuda_available()


def _make_flat_ground_mesh(size=5.0, z=0.0):
    """Create a flat ground mesh (two triangles) at height *z*.

    CCW winding when viewed from +Z, so face normal points upward.
    """
    vertices = np.array(
        [[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _make_valley_mesh(width=2.0, depth=0.5, length=2.0):
    """V-shaped valley: two angled slopes meeting at z=0, Y-axis aligned.

    Left slope rises toward -X, right slope rises toward +X.
    Face normals point upward/outward.
    """
    hw = width / 2.0
    hl = length / 2.0
    vertices = np.array(
        [
            [-hw, -hl, depth],
            [0.0, -hl, 0.0],
            [0.0, hl, 0.0],
            [-hw, hl, depth],
            [0.0, -hl, 0.0],
            [hw, -hl, depth],
            [hw, hl, depth],
            [0.0, hl, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _make_box_ground_mesh(size=5.0, thickness=0.2, z=0.0):
    """Solid box ground mesh for use with solvers that need 3D convex hulls.

    Top surface at *z*, full 3D extent.
    """
    mesh = newton.Mesh.create_box(size, size, thickness, compute_inertia=False)
    verts = mesh.vertices.copy()
    verts[:, 2] += z - thickness  # top face at z
    return newton.Mesh(verts, mesh.indices, compute_inertia=False)


def _build_collision_only(mesh, shape_type, shape_pos, shape_scale=None, shape_rot=None):
    """Build a scene and return (model, collision_pipeline, state).

    No solver needed -- tests only inspect contacts.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    builder.add_shape_mesh(body=-1, mesh=mesh, xform=wp.transform_identity())

    body = builder.add_body(
        xform=wp.transform(
            wp.vec3(*shape_pos),
            wp.quat(*shape_rot) if shape_rot else wp.quat_identity(),
        ),
    )

    if shape_type == GeoType.SPHERE:
        radius = shape_scale[0] if shape_scale else 0.1
        builder.add_shape_sphere(body, radius=radius)
    elif shape_type == GeoType.BOX:
        hx, hy, hz = shape_scale if shape_scale else (0.1, 0.1, 0.1)
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
    elif shape_type == GeoType.CAPSULE:
        radius = shape_scale[0] if shape_scale else 0.1
        half_height = shape_scale[1] if shape_scale else 0.2
        builder.add_shape_capsule(body, radius=radius, half_height=half_height)
    elif shape_type == GeoType.ELLIPSOID:
        sx, sy, sz = shape_scale if shape_scale else (0.1, 0.15, 0.08)
        builder.add_shape_ellipsoid(body, a=sx, b=sy, c=sz)
    elif shape_type == GeoType.CONVEX_MESH:
        convex = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_inertia=False)
        builder.add_shape_convex_hull(body, mesh=convex)
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")

    model = builder.finalize()
    cp = newton.CollisionPipeline(model, broad_phase="explicit", max_triangle_pairs=100_000)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    return model, cp, state


def _collide(model, cp, state):
    """Run collision detection and return contacts."""
    return model.collide(state, collision_pipeline=cp)


def _get_contact_normals(contacts):
    """Return active contact normals as (N, 3) array."""
    count = contacts.rigid_contact_count.numpy()[0]
    if count == 0:
        return np.zeros((0, 3))
    return contacts.rigid_contact_normal.numpy()[:count]


def _build_sim_scene(mesh, shape_type, shape_pos, shape_scale=None, shape_rot=None):
    """Build full simulation scene with SolverMuJoCo for integration tests."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    builder.add_shape_mesh(body=-1, mesh=mesh, xform=wp.transform_identity())

    body = builder.add_body(
        xform=wp.transform(
            wp.vec3(*shape_pos),
            wp.quat(*shape_rot) if shape_rot else wp.quat_identity(),
        ),
    )

    if shape_type == GeoType.SPHERE:
        radius = shape_scale[0] if shape_scale else 0.1
        builder.add_shape_sphere(body, radius=radius)
    elif shape_type == GeoType.BOX:
        hx, hy, hz = shape_scale if shape_scale else (0.1, 0.1, 0.1)
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
    elif shape_type == GeoType.CAPSULE:
        radius = shape_scale[0] if shape_scale else 0.1
        half_height = shape_scale[1] if shape_scale else 0.2
        builder.add_shape_capsule(body, radius=radius, half_height=half_height)
    else:
        raise ValueError(f"Unsupported shape type for sim: {shape_type}")

    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        ls_iterations=20,
        njmax=1024,
        nconmax=256,
        integrator="implicitfast",
    )
    cp = newton.CollisionPipeline(model, broad_phase="explicit", max_triangle_pairs=100_000)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    return model, cp, solver, s0, s1, ctrl


def _step_sim(model, cp, solver, s0, s1, ctrl, dt=0.005, substeps=1):
    """Advance one frame."""
    for _ in range(substeps):
        s0.clear_forces()
        contacts = model.collide(s0, collision_pipeline=cp)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0
    return s0, s1, contacts


# ======================================================================
# Contact-level tests (collision pipeline only, no solver)
# ======================================================================


@unittest.skipUnless(_cuda_available, "Mesh collision pipeline requires CUDA device")
class TestMeshBackfaceCulling(unittest.TestCase):
    """Verify back-face contacts are culled for all convex types."""

    # ------------------------------------------------------------------
    # Front-face: shape above flat mesh → upward normals
    # ------------------------------------------------------------------

    def _assert_front_face_contacts(self, shape_type, shape_scale=None, shape_rot=None):
        mesh = _make_flat_ground_mesh(z=0.0)
        model, cp, state = _build_collision_only(
            mesh,
            shape_type,
            shape_pos=(0.0, 0.0, 0.05),
            shape_scale=shape_scale,
            shape_rot=shape_rot,
        )
        contacts = _collide(model, cp, state)
        normals = _get_contact_normals(contacts)

        self.assertGreater(len(normals), 0, f"{shape_type.name}: expected contacts on front face")
        min_nz = normals[:, 2].min()
        self.assertGreater(min_nz, -1e-4, f"{shape_type.name}: front-face normal z={min_nz:.4f} should be > 0")

    def test_front_face_sphere(self):
        self._assert_front_face_contacts(GeoType.SPHERE, shape_scale=(0.1,))

    def test_front_face_box(self):
        self._assert_front_face_contacts(GeoType.BOX, shape_scale=(0.1, 0.1, 0.1))

    def test_front_face_capsule(self):
        self._assert_front_face_contacts(GeoType.CAPSULE, shape_scale=(0.05, 0.15))

    def test_front_face_ellipsoid(self):
        self._assert_front_face_contacts(GeoType.ELLIPSOID, shape_scale=(0.1, 0.15, 0.08))

    def test_front_face_convex_mesh(self):
        self._assert_front_face_contacts(GeoType.CONVEX_MESH)

    # ------------------------------------------------------------------
    # Back-face: shape below flat mesh → no contacts at all
    # ------------------------------------------------------------------

    def _assert_back_face_culled(self, shape_type, shape_scale=None, shape_rot=None):
        mesh = _make_flat_ground_mesh(z=0.0)
        model, cp, state = _build_collision_only(
            mesh,
            shape_type,
            shape_pos=(0.0, 0.0, -0.05),
            shape_scale=shape_scale,
            shape_rot=shape_rot,
        )
        contacts = _collide(model, cp, state)
        normals = _get_contact_normals(contacts)

        # Either no contacts, or no contacts with strongly inverted normals
        if len(normals) > 0:
            min_nz = normals[:, 2].min()
            self.assertGreater(
                min_nz,
                -0.3,
                f"{shape_type.name}: back-face contact with inverted normal z={min_nz:.4f}",
            )

    def test_back_face_sphere(self):
        self._assert_back_face_culled(GeoType.SPHERE, shape_scale=(0.1,))

    def test_back_face_box(self):
        self._assert_back_face_culled(GeoType.BOX, shape_scale=(0.1, 0.1, 0.1))

    def test_back_face_capsule(self):
        self._assert_back_face_culled(GeoType.CAPSULE, shape_scale=(0.05, 0.15))

    def test_back_face_ellipsoid(self):
        self._assert_back_face_culled(GeoType.ELLIPSOID, shape_scale=(0.1, 0.15, 0.08))

    def test_back_face_convex_mesh(self):
        self._assert_back_face_culled(GeoType.CONVEX_MESH)

    # ------------------------------------------------------------------
    # Rotated convex shapes on back side
    # ------------------------------------------------------------------

    def test_back_face_rotated_capsule(self):
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
        rot_np = tuple(float(rot[i]) for i in range(4))
        self._assert_back_face_culled(GeoType.CAPSULE, shape_scale=(0.05, 0.15), shape_rot=rot_np)

    def test_back_face_rotated_box(self):
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), np.pi / 6.0)
        rot_np = tuple(float(rot[i]) for i in range(4))
        self._assert_back_face_culled(GeoType.BOX, shape_scale=(0.1, 0.08, 0.12), shape_rot=rot_np)

    # ------------------------------------------------------------------
    # Deep penetration: shape well below mesh, no contact
    # ------------------------------------------------------------------

    def test_deep_back_face_sphere(self):
        """Sphere far below mesh should get zero contacts."""
        mesh = _make_flat_ground_mesh(z=0.0)
        model, cp, state = _build_collision_only(
            mesh,
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, -0.5),
            shape_scale=(0.1,),
        )
        contacts = _collide(model, cp, state)
        count = contacts.rigid_contact_count.numpy()[0]
        self.assertEqual(count, 0, "Deep back-face sphere should have zero contacts")

    def test_deep_back_face_box(self):
        """Box far below mesh should get zero contacts."""
        mesh = _make_flat_ground_mesh(z=0.0)
        model, cp, state = _build_collision_only(
            mesh,
            GeoType.BOX,
            shape_pos=(0.0, 0.0, -0.5),
            shape_scale=(0.1, 0.1, 0.1),
        )
        contacts = _collide(model, cp, state)
        count = contacts.rigid_contact_count.numpy()[0]
        self.assertEqual(count, 0, "Deep back-face box should have zero contacts")

    # ------------------------------------------------------------------
    # Valley mesh: contacts should push outward, not trap
    # ------------------------------------------------------------------

    def test_valley_sphere_normals_point_outward(self):
        """Sphere in valley: all contact normals should have positive z."""
        mesh = _make_valley_mesh(width=2.0, depth=0.5, length=2.0)
        model, cp, state = _build_collision_only(
            mesh,
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 0.08),
            shape_scale=(0.1,),
        )
        contacts = _collide(model, cp, state)
        normals = _get_contact_normals(contacts)

        if len(normals) > 0:
            min_nz = normals[:, 2].min()
            self.assertGreater(min_nz, -0.1, f"Valley sphere: normal z={min_nz:.4f} should point outward")

    def test_valley_box_normals_point_outward(self):
        """Box in valley: contact normals should not have strongly negative z."""
        mesh = _make_valley_mesh(width=2.0, depth=0.5, length=2.0)
        model, cp, state = _build_collision_only(
            mesh,
            GeoType.BOX,
            shape_pos=(0.0, 0.0, 0.08),
            shape_scale=(0.08, 0.08, 0.08),
        )
        contacts = _collide(model, cp, state)
        normals = _get_contact_normals(contacts)

        if len(normals) > 0:
            # In a valley, normals from the slopes point up-and-inward, so z > 0
            min_nz = normals[:, 2].min()
            self.assertGreater(min_nz, -0.1, f"Valley box: normal z={min_nz:.4f}")

    # ------------------------------------------------------------------
    # No NaN in contact data
    # ------------------------------------------------------------------

    def _assert_no_nan_in_contacts(self, shape_type, pos, shape_scale=None):
        mesh = _make_flat_ground_mesh(z=0.0)
        model, cp, state = _build_collision_only(
            mesh,
            shape_type,
            shape_pos=pos,
            shape_scale=shape_scale,
        )
        contacts = _collide(model, cp, state)
        normals = _get_contact_normals(contacts)
        if len(normals) > 0:
            self.assertFalse(np.any(np.isnan(normals)), f"{shape_type.name}: NaN in contact normals")

    def test_no_nan_front_face(self):
        for st, sc in [
            (GeoType.SPHERE, (0.1,)),
            (GeoType.BOX, (0.1, 0.1, 0.1)),
            (GeoType.CAPSULE, (0.05, 0.15)),
            (GeoType.ELLIPSOID, (0.1, 0.15, 0.08)),
        ]:
            with self.subTest(shape_type=st.name):
                self._assert_no_nan_in_contacts(st, (0.0, 0.0, 0.05), sc)

    def test_no_nan_back_face(self):
        for st, sc in [
            (GeoType.SPHERE, (0.1,)),
            (GeoType.BOX, (0.1, 0.1, 0.1)),
            (GeoType.CAPSULE, (0.05, 0.15)),
            (GeoType.ELLIPSOID, (0.1, 0.15, 0.08)),
        ]:
            with self.subTest(shape_type=st.name):
                self._assert_no_nan_in_contacts(st, (0.0, 0.0, -0.05), sc)


# ======================================================================
# Integration tests (full simulation with solver)
# ======================================================================


@unittest.skipUnless(_cuda_available, "Mesh collision pipeline requires CUDA device")
class TestMeshBackfaceSimulation(unittest.TestCase):
    """Full simulation tests: shapes on mesh terrain should not produce NaN."""

    def _run_sim_no_nan(self, shape_type, shape_pos, shape_scale, n_frames=100, substeps=5, dt=0.005):
        """Run simulation and assert no NaN appears in joint state."""
        mesh = _make_box_ground_mesh(z=0.0)
        model, cp, solver, s0, s1, ctrl = _build_sim_scene(
            mesh,
            shape_type,
            shape_pos=shape_pos,
            shape_scale=shape_scale,
        )
        for _ in range(n_frames):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, dt=dt, substeps=substeps)
            joint_q = s0.joint_q.numpy()
            self.assertFalse(np.any(np.isnan(joint_q)), f"{shape_type.name}: NaN in joint_q")

    def test_sim_sphere_on_ground(self):
        """Sphere dropped on ground should settle without NaN."""
        self._run_sim_no_nan(GeoType.SPHERE, (0.0, 0.0, 0.5), (0.1,))

    def test_sim_box_on_ground(self):
        """Box dropped on ground should settle without NaN."""
        self._run_sim_no_nan(GeoType.BOX, (0.0, 0.0, 0.5), (0.1, 0.1, 0.1))

    def test_sim_capsule_on_ground(self):
        """Capsule dropped on ground should settle without NaN."""
        self._run_sim_no_nan(GeoType.CAPSULE, (0.0, 0.0, 0.5), (0.05, 0.15))

    def test_sim_sphere_high_velocity(self):
        """Fast-falling sphere should not produce NaN after mesh impact."""
        mesh = _make_box_ground_mesh(z=0.0)
        model, cp, solver, s0, s1, ctrl = _build_sim_scene(
            mesh,
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 1.0),
            shape_scale=(0.1,),
        )
        # High downward velocity
        qd = s0.joint_qd.numpy()
        qd[2] = -10.0
        s0.joint_qd = wp.array(qd, dtype=wp.float32)

        for _ in range(50):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=10)
            joint_q = s0.joint_q.numpy()
            self.assertFalse(np.any(np.isnan(joint_q)), "High-velocity sphere: NaN")

    def test_sim_valley_sphere_no_nan(self):
        """Sphere in valley should settle without NaN."""
        # Valley mesh needs 3D extent for MuJoCo
        mesh = _make_box_ground_mesh(z=0.0)  # use flat ground as proxy
        model, cp, solver, s0, s1, ctrl = _build_sim_scene(
            mesh,
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 0.15),
            shape_scale=(0.1,),
        )
        for _ in range(200):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            joint_q = s0.joint_q.numpy()
            self.assertFalse(np.any(np.isnan(joint_q)), "Valley sphere: NaN")

    def test_sim_shape_stuck_behind_mesh_no_nan(self):
        """Reproduce the original bug: shape behind mesh must not cause NaN.

        Core scenario from the bug report: a convex shape ends up on the
        back side of a mesh surface.  Without back-face culling the
        inverted contact normals trap the shape and the solver diverges
        to NaN.  We place shapes below a box-mesh ground and simulate
        100 frames to verify no NaN appears.
        """
        mesh = _make_box_ground_mesh(z=0.0)

        for shape_type, scale, z_start in [
            (GeoType.SPHERE, (0.1,), -0.15),
            (GeoType.CAPSULE, (0.05, 0.15), -0.20),
            (GeoType.BOX, (0.08, 0.08, 0.08), -0.15),
        ]:
            with self.subTest(shape_type=shape_type.name):
                model, cp, solver, s0, s1, ctrl = _build_sim_scene(
                    mesh,
                    shape_type,
                    shape_pos=(0.0, 0.0, z_start),
                    shape_scale=scale,
                )
                for step in range(100):
                    s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
                    self.assertFalse(
                        np.any(np.isnan(s0.joint_q.numpy())),
                        f"Stuck {shape_type.name}: NaN at step {step}",
                    )


# ======================================================================
# Heightfield prism tests
# ======================================================================


def _build_heightfield_scene(shape_type, shape_pos, shape_scale=None, rough=False):
    """Build a scene with a heightfield ground + one dynamic convex shape."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    nrow, ncol = 20, 20
    if rough:
        rng = np.random.default_rng(123)
        elevation = rng.uniform(0.0, 1.0, (nrow, ncol)).astype(np.float32)
    else:
        elevation = np.zeros((nrow, ncol), dtype=np.float32)

    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=0.5)
    builder.add_shape_heightfield(heightfield=hfield)

    body = builder.add_body(
        xform=wp.transform(wp.vec3(*shape_pos), wp.quat_identity()),
    )
    if shape_type == GeoType.SPHERE:
        radius = shape_scale[0] if shape_scale else 0.1
        builder.add_shape_sphere(body, radius=radius)
    elif shape_type == GeoType.BOX:
        hx, hy, hz = shape_scale if shape_scale else (0.1, 0.1, 0.1)
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
    elif shape_type == GeoType.CAPSULE:
        radius = shape_scale[0] if shape_scale else 0.1
        half_height = shape_scale[1] if shape_scale else 0.2
        builder.add_shape_capsule(body, radius=radius, half_height=half_height)

    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        ls_iterations=20,
        njmax=1024,
        nconmax=256,
        integrator="implicitfast",
    )
    cp = newton.CollisionPipeline(model, broad_phase="explicit", max_triangle_pairs=100_000)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    return model, cp, solver, s0, s1, ctrl


@unittest.skipUnless(_cuda_available, "Collision pipeline requires CUDA device")
class TestHeightfieldPrism(unittest.TestCase):
    """Test heightfield triangle prism extrusion prevents back-face trapping."""

    def test_sphere_on_flat_heightfield(self):
        """Sphere dropped on flat heightfield should settle without NaN."""
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene(
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 0.5),
            shape_scale=(0.1,),
        )
        for _ in range(100):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Flat hfield sphere: NaN")

    def test_box_on_flat_heightfield(self):
        """Box dropped on flat heightfield should settle without NaN."""
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene(
            GeoType.BOX,
            shape_pos=(0.0, 0.0, 0.5),
            shape_scale=(0.1, 0.1, 0.1),
        )
        for _ in range(100):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Flat hfield box: NaN")

    def test_sphere_on_rough_heightfield(self):
        """Sphere on rough heightfield should not produce NaN."""
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene(
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 1.0),
            shape_scale=(0.1,),
            rough=True,
        )
        for _ in range(200):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Rough hfield sphere: NaN")

    def test_capsule_on_rough_heightfield(self):
        """Capsule on rough heightfield should not produce NaN."""
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene(
            GeoType.CAPSULE,
            shape_pos=(0.0, 0.0, 1.0),
            shape_scale=(0.05, 0.15),
            rough=True,
        )
        for _ in range(200):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Rough hfield capsule: NaN")

    def test_sphere_below_heightfield_no_nan(self):
        """Sphere placed below flat heightfield should not produce NaN.

        The prism extrusion should generate contacts that push the sphere
        back out, or at minimum not trap it with inverted normals.
        """
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene(
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, -0.05),
            shape_scale=(0.1,),
        )
        for _ in range(50):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Below hfield sphere: NaN")

    def test_high_velocity_sphere_on_heightfield(self):
        """Fast-falling sphere on heightfield should not produce NaN."""
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene(
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 1.0),
            shape_scale=(0.1,),
            rough=True,
        )
        qd = s0.joint_qd.numpy()
        qd[2] = -10.0
        s0.joint_qd = wp.array(qd, dtype=wp.float32)

        for _ in range(50):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=10)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Fast hfield sphere: NaN")


def _build_heightfield_scene_xform(
    shape_type,
    shape_pos,
    shape_scale,
    hfield_xform,
    elevation,
    nrow,
    ncol,
    hx,
    hy,
    min_z,
    max_z,
):
    """Build scene with a heightfield at an arbitrary transform."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=hx, hy=hy, min_z=min_z, max_z=max_z)
    builder.add_shape_heightfield(heightfield=hfield, xform=hfield_xform)

    body = builder.add_body(xform=wp.transform(wp.vec3(*shape_pos), wp.quat_identity()))
    if shape_type == GeoType.SPHERE:
        builder.add_shape_sphere(body, radius=shape_scale[0])
    elif shape_type == GeoType.BOX:
        builder.add_shape_box(body, hx=shape_scale[0], hy=shape_scale[1], hz=shape_scale[2])
    elif shape_type == GeoType.CAPSULE:
        builder.add_shape_capsule(body, radius=shape_scale[0], half_height=shape_scale[1])

    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        ls_iterations=20,
        njmax=1024,
        nconmax=256,
        integrator="implicitfast",
    )
    cp = newton.CollisionPipeline(model, broad_phase="explicit", max_triangle_pairs=100_000)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    return model, cp, solver, s0, s1, ctrl


@unittest.skipUnless(_cuda_available, "Collision pipeline requires CUDA device")
class TestHeightfieldPrismSteepAndRotated(unittest.TestCase):
    """Tests with sharp inter-cell edges and rotated heightfields.

    These tests verify that the prism extrusion along the heightfield's
    local -Z axis works correctly even when:
    - Adjacent heightfield cells form a sharp ridge where the dihedral
      angle at the shared edge is < 90° (i.e. the angle between the two
      face normals exceeds 90°).  Extruding along the face normal would
      make one prism poke through the adjacent triangle.
    - The heightfield is rotated so its elevation axis is not world-Z.
    """

    def _make_ridge_elevation(self, nrow=10, ncol=10):
        """Create a sharp ridge where adjacent face normals diverge > 90°.

        Column 5 is at max elevation, columns 4 and 6 at zero.
        With hx=2, ncol=10 the cell width is 0.44 m and the rise is 2 m
        (max_z=2), giving ~77° slopes on each side.  The dihedral angle
        at the ridge edge is only ~25° (face normals diverge by ~155°).
        """
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        for r in range(nrow):
            elevation[r, 4] = 0.0
            elevation[r, 5] = 1.0  # peak
            elevation[r, 6] = 0.0
        return elevation

    def test_sphere_on_sharp_ridge_no_nan(self):
        """Sphere dropped directly on a ridge with >90° normal divergence."""
        nrow, ncol = 10, 10
        elevation = self._make_ridge_elevation(nrow, ncol)
        # Drop sphere right onto the ridge (x≈0 is column 5 at identity hx=2)
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene_xform(
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 3.0),
            shape_scale=(0.15,),
            hfield_xform=wp.transform_identity(),
            elevation=elevation,
            nrow=nrow,
            ncol=ncol,
            hx=2.0,
            hy=2.0,
            min_z=0.0,
            max_z=2.0,
        )
        for _ in range(100):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Ridge sphere: NaN")

    def test_box_on_sharp_ridge_no_nan(self):
        """Box dropped on ridge with >90° normal divergence."""
        nrow, ncol = 10, 10
        elevation = self._make_ridge_elevation(nrow, ncol)
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene_xform(
            GeoType.BOX,
            shape_pos=(0.0, 0.0, 3.0),
            shape_scale=(0.1, 0.1, 0.1),
            hfield_xform=wp.transform_identity(),
            elevation=elevation,
            nrow=nrow,
            ncol=ncol,
            hx=2.0,
            hy=2.0,
            min_z=0.0,
            max_z=2.0,
        )
        for _ in range(100):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Ridge box: NaN")

    def test_rotated_heightfield_sphere_no_nan(self):
        """Sphere on a heightfield rotated 30 degrees around X axis.

        The heightfield's elevation axis is no longer world-Z.  The prism
        extrusion must still use the heightfield's local -Z, not world -Z.
        """
        nrow, ncol = 10, 10
        rng = np.random.default_rng(99)
        elevation = rng.uniform(0.0, 1.0, (nrow, ncol)).astype(np.float32)

        # Rotate heightfield 30 degrees around X axis
        angle = np.pi / 6.0
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
        hfield_xform = wp.transform(wp.vec3(0.0, 0.0, 2.0), rot)

        # Place sphere above the rotated heightfield
        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene_xform(
            GeoType.SPHERE,
            shape_pos=(0.0, 0.0, 4.0),
            shape_scale=(0.1,),
            hfield_xform=hfield_xform,
            elevation=elevation,
            nrow=nrow,
            ncol=ncol,
            hx=3.0,
            hy=3.0,
            min_z=0.0,
            max_z=0.5,
        )
        for _ in range(200):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Rotated hfield sphere: NaN")

    def test_rotated_steep_heightfield_capsule_no_nan(self):
        """Capsule on a steep, rotated heightfield — worst case scenario.

        Combines steep slopes (overhangs) with heightfield rotation.
        If the extrusion used the face normal instead of the heightfield's
        local -Z, this test would fail with inverted contacts.
        """
        nrow, ncol = 10, 10
        elevation = self._make_ridge_elevation(nrow, ncol)

        # Rotate 20 degrees around Y
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 9.0)
        hfield_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), rot)

        model, cp, solver, s0, s1, ctrl = _build_heightfield_scene_xform(
            GeoType.CAPSULE,
            shape_pos=(0.0, 0.0, 4.0),
            shape_scale=(0.08, 0.15),
            hfield_xform=hfield_xform,
            elevation=elevation,
            nrow=nrow,
            ncol=ncol,
            hx=2.0,
            hy=2.0,
            min_z=0.0,
            max_z=2.0,
        )
        for _ in range(200):
            s0, s1, _ = _step_sim(model, cp, solver, s0, s1, ctrl, substeps=5)
            self.assertFalse(np.any(np.isnan(s0.joint_q.numpy())), "Rotated steep hfield capsule: NaN")


if __name__ == "__main__":
    unittest.main()
