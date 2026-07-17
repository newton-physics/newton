# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import Counter
from enum import IntFlag, auto

import numpy as np
import warp as wp
import warp.examples

import newton
from newton import GeoType
from newton._src.geometry import create_mesh_terrain
from newton._src.geometry.flags import ParticleFlags, ShapeFlags
from newton._src.geometry.kernels import create_soft_contacts, mesh_sdf
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import (
    SDF_EDGE_ITERS,
    SDF_FACE_ITERS,
    SDF_LS_ITERS,
    _is_analytic,
    _shape_frames,
    eval_shape_sdf,
    launch_soft_ef_contacts,
    optimize_edge_sdf,
    optimize_face_sdf,
)
from newton._src.sim.collide import (
    _build_soft_edge_rigid_contact_pairs,
    _build_soft_face_rigid_contact_pairs,
    _compute_per_world_shape_pairs_max,
    _estimate_rigid_contact_max,
)
from newton._src.utils.heightfield import HeightfieldData
from newton.examples import test_body_state
from newton.tests.unittest_utils import (
    add_function_test,
    configure_sdf_for_collision_shapes,
    get_cuda_test_devices,
    get_test_devices,
)


class TestLevel(IntFlag):
    VELOCITY_X = auto()
    VELOCITY_YZ = auto()
    VELOCITY_LINEAR = VELOCITY_X | VELOCITY_YZ
    VELOCITY_ANGULAR = auto()
    STRICT = VELOCITY_LINEAR | VELOCITY_ANGULAR


def type_to_str(shape_type: GeoType):
    if shape_type == GeoType.SPHERE:
        return "sphere"
    elif shape_type == GeoType.BOX:
        return "box"
    elif shape_type == GeoType.CAPSULE:
        return "capsule"
    elif shape_type == GeoType.CYLINDER:
        return "cylinder"
    elif shape_type == GeoType.CONE:
        return "cone"
    elif shape_type == GeoType.MESH:
        return "mesh"
    elif shape_type == GeoType.CONVEX_MESH:
        return "convex_hull"
    elif shape_type == GeoType.PLANE:
        return "plane"
    else:
        return "unknown"


class CollisionSetup:
    def __init__(
        self,
        viewer,
        device,
        shape_type_a,
        shape_type_b,
        solver_fn,
        sim_substeps,
        broad_phase="explicit",
        sdf_max_resolution_a=None,
        sdf_max_resolution_b=None,
    ):
        self.sim_substeps = sim_substeps
        self.frame_dt = 1 / 60
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.shape_type_a = shape_type_a
        self.shape_type_b = shape_type_b
        self.sdf_max_resolution_a = sdf_max_resolution_a
        self.sdf_max_resolution_b = sdf_max_resolution_b
        self._device = device

        self.builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # Set contact margin to match previous test expectations
        # Note: margins are now summed (margin_a + margin_b), so we use half the previous value
        self.builder.rigid_gap = 0.005

        body_a = self.builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0)))
        self.add_shape(shape_type_a, body_a, sdf_max_resolution=sdf_max_resolution_a)

        self.init_velocity = 5.0
        self.builder.joint_qd[0] = self.builder.body_qd[-1][0] = self.init_velocity

        body_b = self.builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)))
        self.add_shape(shape_type_b, body_b, sdf_max_resolution=sdf_max_resolution_b)

        self.model = self.builder.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=broad_phase,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.solver = solver_fn(self.model)

        self.viewer = viewer
        self.viewer.set_model(self.model)

        self.graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                self.simulate()
            self.graph = capture.graph

    def add_shape(self, shape_type: GeoType, body: int, sdf_max_resolution: int | None = None):
        if shape_type == GeoType.BOX:
            self.builder.add_shape_box(body, label=type_to_str(shape_type))
        elif shape_type == GeoType.SPHERE:
            self.builder.add_shape_sphere(body, radius=0.5, label=type_to_str(shape_type))
        elif shape_type == GeoType.CAPSULE:
            self.builder.add_shape_capsule(body, radius=0.25, half_height=0.3, label=type_to_str(shape_type))
        elif shape_type == GeoType.CYLINDER:
            self.builder.add_shape_cylinder(body, radius=0.25, half_height=0.4, label=type_to_str(shape_type))
        elif shape_type == GeoType.CONE:
            # Rotate cone so flat base faces -X (toward the incoming object)
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 2.0)
            xform = wp.transform(wp.vec3(), rot)
            self.builder.add_shape_cone(body, xform=xform, radius=0.25, half_height=0.4, label=type_to_str(shape_type))
        elif shape_type == GeoType.MESH:
            # Use box mesh (works correctly with collision pipeline)
            mesh = newton.Mesh.create_box(
                0.5,
                0.5,
                0.5,
                duplicate_vertices=False,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            if sdf_max_resolution is not None:
                mesh.build_sdf(max_resolution=sdf_max_resolution, device=self._device)
            self.builder.add_shape_mesh(body, mesh=mesh, label=type_to_str(shape_type))
        elif shape_type == GeoType.CONVEX_MESH:
            # Use a sphere mesh as it's already convex
            mesh = newton.Mesh.create_sphere(0.5, compute_normals=False, compute_uvs=False, compute_inertia=False)
            self.builder.add_shape_convex_hull(body, mesh=mesh, label=type_to_str(shape_type))
        else:
            raise NotImplementedError(f"Shape type {shape_type} not implemented")

    def capture(self):
        if wp.get_device(self._device).is_cuda:
            with wp.ScopedCapture(device=self._device) as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self, test_level: TestLevel, body: int, tolerance: float = 3e-3):
        body_name = f"body {body} ({self.model.shape_label[body]})"
        if test_level & TestLevel.VELOCITY_X:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} is moving forward",
                lambda _q, qd: qd[0] > 0.03 and qd[0] <= wp.static(self.init_velocity),
                indices=[body],
                show_body_qd=True,
            )
        if test_level & TestLevel.VELOCITY_YZ:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} has correct linear velocity",
                lambda _q, qd: abs(qd[1]) < tolerance and abs(qd[2]) < tolerance,
                indices=[body],
                show_body_qd=True,
            )
        if test_level & TestLevel.VELOCITY_ANGULAR:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} has correct angular velocity",
                lambda _q, qd: abs(qd[3]) < tolerance and abs(qd[4]) < tolerance and abs(qd[5]) < tolerance,
                indices=[body],
                show_body_qd=True,
            )


# The exhaustive shape/broad-phase matrix is one of the heaviest GPU suites.
# Keep it on one CUDA device; targeted deterministic tests below use all selected CUDA devices.
devices = get_cuda_test_devices(mode="basic")


class TestCollisionPipeline(unittest.TestCase):
    def test_soft_contact_max_zero_disables_soft_contact_generation(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()
        builder.add_particle(pos=(0.0, 0.0, 0.025), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05)
        model = builder.finalize(device="cpu")
        state = model.state()

        enabled_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1)
        enabled_contacts = enabled_pipeline.contacts()
        enabled_pipeline.collide(state, enabled_contacts)
        self.assertGreater(int(enabled_contacts.soft_contact_count.numpy()[0]), 0)

        disabled_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_max=0,
            soft_contact_margin=0.1,
        )
        disabled_contacts = disabled_pipeline.contacts()
        disabled_pipeline.collide(state, disabled_contacts)

        self.assertEqual(disabled_contacts.soft_contact_max, 0)
        self.assertEqual(int(disabled_contacts.soft_contact_count.numpy()[0]), 0)


# Collision pipeline tests - now supports both MESH and CONVEX_MESH
# Format: (shape_a, shape_b, test_level_a, test_level_b, tolerance)
# tolerance defaults to 3e-3 if not specified
collision_pipeline_contact_tests = [
    (GeoType.SPHERE, GeoType.SPHERE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CYLINDER, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CONE, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_YZ),
    (GeoType.SPHERE, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.BOX, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    # Box-vs-triangle-mesh contact can accumulate a small lateral drift on CUDA
    # due to triangulation/discretization details; keep this tolerance slightly looser.
    (GeoType.BOX, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR, 0.03),
    (GeoType.BOX, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.CAPSULE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (
        GeoType.MESH,
        GeoType.MESH,
        TestLevel.VELOCITY_YZ,
        TestLevel.VELOCITY_LINEAR,
    ),
    # Mesh-vs-convex-hull likewise accumulates small lateral drift from
    # triangulated mesh faces (same root cause as box-vs-mesh above).
    (GeoType.MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR, 0.02),
    (GeoType.CONVEX_MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
]


def test_collision_pipeline(
    _test,
    device,
    shape_type_a: GeoType,
    shape_type_b: GeoType,
    test_level_a: TestLevel,
    test_level_b: TestLevel,
    broad_phase: str,
    tolerance: float = 3e-3,
):
    viewer = newton.viewer.ViewerNull()
    setup = CollisionSetup(
        viewer=viewer,
        device=device,
        solver_fn=newton.solvers.SolverXPBD,
        sim_substeps=10,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        broad_phase=broad_phase,
    )
    for _ in range(100):
        setup.step()
        setup.render()
    setup.test(test_level_a, 0, tolerance=tolerance)
    setup.test(test_level_b, 1, tolerance=tolerance)


# Wrapper functions for each broad phase mode
def test_collision_pipeline_explicit(
    _test,
    device,
    shape_type_a: GeoType,
    shape_type_b: GeoType,
    test_level_a: TestLevel,
    test_level_b: TestLevel,
    tolerance: float = 3e-3,
):
    test_collision_pipeline(
        _test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, "explicit", tolerance
    )


def test_collision_pipeline_nxn(
    _test,
    device,
    shape_type_a: GeoType,
    shape_type_b: GeoType,
    test_level_a: TestLevel,
    test_level_b: TestLevel,
    tolerance: float = 3e-3,
):
    test_collision_pipeline(_test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, "nxn", tolerance)


def test_collision_pipeline_sap(
    _test,
    device,
    shape_type_a: GeoType,
    shape_type_b: GeoType,
    test_level_a: TestLevel,
    test_level_b: TestLevel,
    tolerance: float = 3e-3,
):
    test_collision_pipeline(_test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, "sap", tolerance)


for test_config in collision_pipeline_contact_tests:
    shape_type_a, shape_type_b, test_level_a, test_level_b = test_config[:4]
    tolerance = test_config[4] if len(test_config) > 4 else 3e-3
    # EXPLICIT broad phase tests
    add_function_test(
        TestCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}_explicit",
        test_collision_pipeline_explicit,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
        tolerance=tolerance,
    )
    # NXN broad phase tests
    add_function_test(
        TestCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}_nxn",
        test_collision_pipeline_nxn,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
        tolerance=tolerance,
    )
    # SAP broad phase tests
    add_function_test(
        TestCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}_sap",
        test_collision_pipeline_sap,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
        tolerance=tolerance,
    )


# Mesh-mesh collision with different SDF configurations
# Test all four modes: SDF vs SDF, SDF vs BVH, BVH vs SDF, and BVH vs BVH
def test_mesh_mesh_sdf_modes(
    _test,
    device,
    sdf_max_resolution_a: int | None,
    sdf_max_resolution_b: int | None,
    broad_phase: str,
    tolerance: float = 3e-3,
):
    """Test mesh-mesh collision with specific SDF configurations."""
    viewer = newton.viewer.ViewerNull()
    setup = CollisionSetup(
        viewer=viewer,
        device=device,
        solver_fn=newton.solvers.SolverXPBD,
        sim_substeps=10,
        shape_type_a=GeoType.MESH,
        shape_type_b=GeoType.MESH,
        broad_phase=broad_phase,
        sdf_max_resolution_a=sdf_max_resolution_a,
        sdf_max_resolution_b=sdf_max_resolution_b,
    )
    for _ in range(100):
        setup.step()
        setup.render()
    setup.test(TestLevel.VELOCITY_YZ, 0, tolerance=tolerance)
    setup.test(TestLevel.VELOCITY_LINEAR, 1, tolerance=tolerance)


# Wrapper functions for different SDF modes
def test_mesh_mesh_sdf_vs_sdf(_test, device, broad_phase: str):
    """Test mesh-mesh collision where both meshes have SDFs."""
    # SDF-SDF hydroelastic contacts can have some variability in contact normal direction
    test_mesh_mesh_sdf_modes(
        _test, device, sdf_max_resolution_a=64, sdf_max_resolution_b=64, broad_phase=broad_phase, tolerance=0.1
    )


def test_mesh_mesh_sdf_vs_bvh(_test, device, broad_phase: str):
    """Test mesh-mesh collision where first mesh has SDF, second uses BVH."""
    # Mixed SDF/BVH mode has slightly more asymmetric contact behavior, use higher tolerance
    test_mesh_mesh_sdf_modes(
        _test,
        device,
        sdf_max_resolution_a=64,
        sdf_max_resolution_b=None,
        broad_phase=broad_phase,
        tolerance=0.2,
    )


def test_mesh_mesh_bvh_vs_sdf(_test, device, broad_phase: str):
    """Test mesh-mesh collision where first mesh uses BVH, second has SDF."""
    # Mixed SDF/BVH mode has slightly more asymmetric contact behavior, use higher tolerance
    test_mesh_mesh_sdf_modes(
        _test,
        device,
        sdf_max_resolution_a=None,
        sdf_max_resolution_b=64,
        broad_phase=broad_phase,
        tolerance=0.5,
    )


def test_mesh_mesh_bvh_vs_bvh(_test, device, broad_phase: str):
    """Test mesh-mesh collision where both meshes use BVH (no SDF)."""
    test_mesh_mesh_sdf_modes(
        _test, device, sdf_max_resolution_a=None, sdf_max_resolution_b=None, broad_phase=broad_phase
    )


# Add mesh-mesh SDF mode tests for all broad phase modes
mesh_mesh_sdf_tests = [
    ("sdf_vs_sdf", test_mesh_mesh_sdf_vs_sdf),
    ("sdf_vs_bvh", test_mesh_mesh_sdf_vs_bvh),
    ("bvh_vs_sdf", test_mesh_mesh_bvh_vs_sdf),
    ("bvh_vs_bvh", test_mesh_mesh_bvh_vs_bvh),
]

for mode_name, test_func in mesh_mesh_sdf_tests:
    for broad_phase_name, broad_phase in [
        ("explicit", "explicit"),
        ("nxn", "nxn"),
        ("sap", "sap"),
    ]:
        add_function_test(
            TestCollisionPipeline,
            f"test_mesh_mesh_{mode_name}_{broad_phase_name}",
            test_func,
            devices=devices,
            broad_phase=broad_phase,
            check_output=False,  # Disable output checking due to Warp module loading messages
        )


def test_mesh_sdf_voxel_tolerance_preserves_inner_contact_coverage(test, device):
    """Texture-SDF uncertainty keeps near-margin contacts in the preferred reduction tier."""
    vertices = np.array(
        [
            [-0.10, -0.10, 0.50005],
            [0.10, -0.10, 0.50005],
            [0.00, 0.10, 0.50005],
            [0.15, -0.10, 0.50500],
            [0.35, -0.10, 0.50500],
            [0.25, 0.10, 0.50500],
        ],
        dtype=np.float32,
    )
    probe_mesh = newton.Mesh(vertices, np.arange(6, dtype=np.int32), compute_inertia=False)
    sdf_mesh = newton.Mesh.create_box(
        0.5,
        0.5,
        0.5,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
        compute_inertia=False,
    )
    sdf_mesh.build_sdf(max_resolution=64, device=device)

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.01, margin=0.0)
    builder.add_shape_mesh(body, mesh=probe_mesh, cfg=cfg)
    builder.add_shape_mesh(-1, mesh=sdf_mesh, cfg=cfg)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        deterministic=True,
        rigid_contact_max=128,
        max_triangle_pairs=128,
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    points = contacts.rigid_contact_point0.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    top_face_points = points[normals[:, 2] < -0.99]

    # All three directional representatives from the near-margin triangle must
    # survive. With an exact zero-width inner tier, only two survive and the
    # positive-X representative flickers into the gap-only outer tier.
    near_margin_points = top_face_points[top_face_points[:, 2] < 0.501]
    test.assertEqual(len(near_margin_points), 3)
    test.assertTrue(np.any(near_margin_points[:, 0] > 0.09))


add_function_test(
    TestCollisionPipeline,
    "test_mesh_sdf_voxel_tolerance_preserves_inner_contact_coverage",
    test_mesh_sdf_voxel_tolerance_preserves_inner_contact_coverage,
    devices=devices,
)


# ============================================================================
# Mesh sign query regressions
# ============================================================================


class TestMeshSignQueries(unittest.TestCase):
    pass


@wp.kernel
def _query_mesh_signs(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: float,
    parity_sign: wp.array[float],
    normal_sign: wp.array[float],
):
    i = wp.tid()
    p = points[i]

    parity = wp.mesh_query_point_sign_parity(mesh, p, max_dist)
    parity_sign[i] = parity.sign if parity.result else 0.0

    sign = float(0.0)
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    normal_hit = wp.mesh_query_point_sign_normal(mesh, p, max_dist, sign, face, u, v)
    normal_sign[i] = sign if normal_hit else 0.0


@wp.kernel
def _query_mesh_sdf(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: float,
    distances: wp.array[float],
):
    i = wp.tid()
    distances[i] = mesh_sdf(mesh, points[i], max_dist)


@wp.func
def _solid_angle(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    pa = a - point
    pb = b - point
    pc = c - point
    la = wp.length(pa)
    lb = wp.length(pb)
    lc = wp.length(pc)
    numerator = wp.dot(pa, wp.cross(pb, pc))
    denominator = la * lb * lc + wp.dot(pa, pb) * lc + wp.dot(pb, pc) * la + wp.dot(pc, pa) * lb
    return 2.0 * wp.atan2(numerator, denominator)


@wp.kernel
def _query_brute_force_winding_signs(
    vertices: wp.array[wp.vec3],
    indices: wp.array[int],
    face_count: int,
    points: wp.array[wp.vec3],
    signs: wp.array[float],
):
    i = wp.tid()
    point = points[i]
    angle_sum = float(0.0)

    for face_index in range(face_count):
        offset = face_index * 3
        a = vertices[indices[offset + 0]]
        b = vertices[indices[offset + 1]]
        c = vertices[indices[offset + 2]]
        angle_sum += _solid_angle(point, a, b, c)

    winding_number = angle_sum / 12.566370614359172
    if wp.abs(winding_number) > 0.5:
        signs[i] = -1.0
    else:
        signs[i] = 1.0


def _make_warp_mesh(vertices: np.ndarray, faces: np.ndarray, device) -> wp.Mesh:
    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(faces.astype(np.int32).reshape(-1), dtype=wp.int32, device=device),
    )


def _make_mixed_winding_convex_pile_proxy() -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = 0.12, 0.07, 0.05
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

    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            # Intentionally mixed winding on the +X face. This reproduces the
            # failure mode from compacted convex hulls whose triangle winding
            # is not consistently outward.
            [1, 6, 2],
            [1, 5, 6],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _make_watertight_box(
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _make_thin_gap_box_pair() -> tuple[np.ndarray, np.ndarray]:
    gap = 2.0e-4
    hx, hy, hz = 0.75, 0.55, 0.45
    left_vertices, left_faces = _make_watertight_box((-hx - 0.5 * gap, 0.0, 0.0), (hx, hy, hz))
    right_vertices, right_faces = _make_watertight_box((hx + 0.5 * gap, 0.0, 0.0), (hx, hy, hz))
    vertices = np.vstack([left_vertices, right_vertices]).astype(np.float32)
    faces = np.vstack([left_faces, right_faces + left_vertices.shape[0]]).astype(np.int32)
    return vertices, faces


def _sample_thin_gap_points(sample_count: int = 8192) -> np.ndarray:
    rng = np.random.default_rng(23)
    gap = 2.0e-4
    hy, hz = 0.55, 0.45
    points = np.empty((sample_count, 3), dtype=np.float32)
    points[:, 0] = rng.uniform(-0.45 * gap, 0.45 * gap, sample_count)
    points[:, 1] = rng.uniform(-0.8 * hy, 0.8 * hy, sample_count)
    points[:, 2] = rng.uniform(-0.8 * hz, 0.8 * hz, sample_count)
    return points


def test_mixed_winding_convex_pile_contact_normal(test, device):
    vertices, faces = _make_mixed_winding_convex_pile_proxy()
    mesh = _make_warp_mesh(vertices, faces, device)

    query_point = np.array([[0.13, 0.018, 0.012]], dtype=np.float32)
    points = wp.array(query_point, dtype=wp.vec3, device=device)
    parity_sign = wp.zeros(1, dtype=wp.float32, device=device)
    normal_sign = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(_query_mesh_signs, dim=1, inputs=[mesh.id, points, 0.1, parity_sign, normal_sign], device=device)

    test.assertGreater(float(parity_sign.numpy()[0]), 0.0)
    test.assertLess(float(normal_sign.numpy()[0]), 0.0)

    soft_contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    soft_contact_particle = wp.empty(1, dtype=wp.int32, device=device)
    soft_contact_indices = wp.empty(1, dtype=wp.vec3i, device=device)
    soft_contact_barycentric = wp.empty(1, dtype=wp.vec3, device=device)
    soft_contact_shape = wp.empty(1, dtype=wp.int32, device=device)
    soft_contact_body_pos = wp.empty(1, dtype=wp.vec3, device=device)
    soft_contact_body_vel = wp.empty(1, dtype=wp.vec3, device=device)
    soft_contact_normal = wp.empty(1, dtype=wp.vec3, device=device)
    soft_contact_tids = wp.empty(1, dtype=wp.int32, device=device)
    soft_rigid_contact_pairs = wp.array([wp.vec2i(0, 0)], dtype=wp.vec2i, device=device)

    wp.launch(
        create_soft_contacts,
        dim=1,
        inputs=[
            soft_rigid_contact_pairs,
            points,
            wp.array([0.05], dtype=wp.float32, device=device),
            wp.array([int(ParticleFlags.ACTIVE)], dtype=wp.int32, device=device),
            wp.array([-1], dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.transform, device=device),
            wp.array([wp.transform()], dtype=wp.transform, device=device),
            wp.array([-1], dtype=wp.int32, device=device),
            wp.array([int(GeoType.CONVEX_MESH)], dtype=wp.int32, device=device),
            wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device),
            wp.array([mesh.id], dtype=wp.uint64, device=device),
            wp.array([-1], dtype=wp.int32, device=device),
            0.0,
            wp.array([0.0], dtype=wp.float32, device=device),
            1,
            wp.array([int(ShapeFlags.COLLIDE_PARTICLES)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.empty(0, dtype=HeightfieldData, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
        ],
        outputs=[
            soft_contact_count,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            soft_contact_tids,
        ],
        device=device,
    )

    test.assertEqual(int(soft_contact_count.numpy()[0]), 1)
    normal = np.asarray(soft_contact_normal.numpy()[0], dtype=np.float32)
    test.assertGreater(float(np.dot(normal, np.array([1.0, 0.0, 0.0], dtype=np.float32))), 0.99)


def test_parity_sign_accuracy_exceeds_normal_query(test, device):
    vertices, faces = _make_thin_gap_box_pair()
    points_np = _sample_thin_gap_points()
    vertices_wp = wp.array(vertices, dtype=wp.vec3, device=device)
    indices_wp = wp.array(faces.reshape(-1), dtype=wp.int32, device=device)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    mesh = wp.Mesh(points=vertices_wp, indices=indices_wp)

    expected_signs_wp = wp.zeros(points_np.shape[0], dtype=wp.float32, device=device)
    wp.launch(
        _query_brute_force_winding_signs,
        dim=points_np.shape[0],
        inputs=[vertices_wp, indices_wp, faces.shape[0], points_wp, expected_signs_wp],
        device=device,
    )

    parity_signs = wp.zeros(points_np.shape[0], dtype=wp.float32, device=device)
    normal_signs = wp.zeros(points_np.shape[0], dtype=wp.float32, device=device)
    wp.launch(
        _query_mesh_signs,
        dim=points_np.shape[0],
        inputs=[mesh.id, points_wp, 10.0, parity_signs, normal_signs],
        device=device,
    )

    distances = wp.zeros(points_np.shape[0], dtype=wp.float32, device=device)
    wp.launch(
        _query_mesh_sdf,
        dim=points_np.shape[0],
        inputs=[mesh.id, points_wp, 10.0, distances],
        device=device,
    )

    expected_signs = expected_signs_wp.numpy()
    production_signs = np.where(distances.numpy() < 0.0, -1.0, 1.0).astype(np.float32)
    parity_accuracy = float(np.mean(parity_signs.numpy() == expected_signs))
    production_accuracy = float(np.mean(production_signs == expected_signs))
    normal_accuracy = float(np.mean(normal_signs.numpy() == expected_signs))

    test.assertTrue(np.all(expected_signs > 0.0))
    test.assertGreaterEqual(
        parity_accuracy,
        0.99,
        f"Parity query accuracy was {parity_accuracy:.3f} against brute-force winding",
    )
    test.assertGreaterEqual(
        production_accuracy,
        0.99,
        f"mesh_sdf accuracy was {production_accuracy:.3f} against brute-force winding",
    )
    test.assertLessEqual(
        normal_accuracy,
        0.05,
        f"Expected the old normal query to fail in the thin gap, got accuracy {normal_accuracy:.3f}",
    )
    test.assertGreater(
        production_accuracy,
        normal_accuracy + 0.9,
        f"Expected parity-backed mesh_sdf accuracy ({production_accuracy:.3f}) to exceed "
        f"normal-query accuracy ({normal_accuracy:.3f})",
    )


add_function_test(
    TestMeshSignQueries,
    "test_mixed_winding_convex_pile_contact_normal",
    test_mixed_winding_convex_pile_contact_normal,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestMeshSignQueries,
    "test_parity_sign_accuracy_exceeds_normal_query",
    test_parity_sign_accuracy_exceeds_normal_query,
    devices=devices,
    check_output=False,
)


# ============================================================================
# Shape collision filter pairs (excluded pairs) with NxN/SAP
# ============================================================================


class TestCollisionPipelineFilterPairs(unittest.TestCase):
    pass


def test_shape_collision_filter_pairs(test, device, broad_phase: str):
    """Verify that excluded shape pairs produce no contacts under NxN or SAP broad phase.

    Args:
        test: The test case instance.
        device: Warp device to run on.
        broad_phase: Broad phase algorithm to test (NXN or SAP).
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.rigid_gap = 0.01
        # Two overlapping spheres (same position so they definitely overlap)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        shape_a = builder.add_shape_sphere(body=body_a, radius=0.5)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        shape_b = builder.add_shape_sphere(body=body_b, radius=0.5)
        # Exclude this pair so they must not generate contacts
        builder.shape_collision_filter_pairs.append((min(shape_a, shape_b), max(shape_a, shape_b)))
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, broad_phase=broad_phase)
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n = contacts.rigid_contact_count.numpy()[0]
        excluded = (min(shape_a, shape_b), max(shape_a, shape_b))
        for i in range(n):
            s0 = int(contacts.rigid_contact_shape0.numpy()[i])
            s1 = int(contacts.rigid_contact_shape1.numpy()[i])
            pair = (min(s0, s1), max(s0, s1))
            test.assertNotEqual(
                pair,
                excluded,
                f"Excluded pair {excluded} must not appear in contacts (broad_phase={broad_phase})",
            )
        # With the only pair excluded, we must have zero rigid contacts
        test.assertEqual(n, 0, f"Expected 0 rigid contacts when only pair is excluded (got {n})")


add_function_test(
    TestCollisionPipelineFilterPairs,
    "test_shape_collision_filter_pairs_nxn",
    test_shape_collision_filter_pairs,
    devices=devices,
    broad_phase="nxn",
)
add_function_test(
    TestCollisionPipelineFilterPairs,
    "test_shape_collision_filter_pairs_sap",
    test_shape_collision_filter_pairs,
    devices=devices,
    broad_phase="sap",
)


def test_collision_filter_consistent_across_broadphases(test, device):
    """Verify that all broad phase modes produce the same contact pairs when collision filtering is applied.

    Creates three overlapping spheres and excludes one pair, then checks that
    EXPLICIT, NXN, and SAP all report exactly the same set of contacting shape pairs.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.rigid_gap = 0.01

        # Three overlapping spheres at the same position
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        shape_a = builder.add_shape_sphere(body=body_a, radius=0.5)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        shape_b = builder.add_shape_sphere(body=body_b, radius=0.5)
        body_c = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_shape_sphere(body=body_c, radius=0.5)

        # Exclude one pair so only two pairs should generate contacts
        excluded = (min(shape_a, shape_b), max(shape_a, shape_b))
        builder.shape_collision_filter_pairs.append(excluded)

        model = builder.finalize(device=device)

        def _contact_pairs(broad_phase):
            pipeline = newton.CollisionPipeline(model, broad_phase=broad_phase)
            state = model.state()
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            n = contacts.rigid_contact_count.numpy()[0]
            shape0_np = contacts.rigid_contact_shape0.numpy()
            shape1_np = contacts.rigid_contact_shape1.numpy()
            pairs = set()
            for i in range(n):
                s0 = int(shape0_np[i])
                s1 = int(shape1_np[i])
                pairs.add((min(s0, s1), max(s0, s1)))
            return pairs

        pairs_explicit = _contact_pairs("explicit")
        pairs_nxn = _contact_pairs("nxn")
        pairs_sap = _contact_pairs("sap")

        # The excluded pair must not appear in any broad phase result
        for name, pairs in [("EXPLICIT", pairs_explicit), ("NXN", pairs_nxn), ("SAP", pairs_sap)]:
            test.assertNotIn(excluded, pairs, f"Excluded pair {excluded} must not appear in {name} contacts")

        # All three broad phases must report the same set of contacting pairs
        test.assertEqual(pairs_explicit, pairs_nxn, "EXPLICIT and NXN should produce the same contact pairs")
        test.assertEqual(pairs_explicit, pairs_sap, "EXPLICIT and SAP should produce the same contact pairs")

        # With 3 shapes and 1 excluded pair, we expect exactly 2 contacting pairs
        test.assertEqual(
            len(pairs_explicit), 2, f"Expected 2 contact pairs, got {len(pairs_explicit)}: {pairs_explicit}"
        )


add_function_test(
    TestCollisionPipelineFilterPairs,
    "test_collision_filter_consistent_across_broadphases",
    test_collision_filter_consistent_across_broadphases,
    devices=devices,
)


# ============================================================================
# Rigid Contact Normal Direction Tests
# ============================================================================
# These tests verify that Contacts.rigid_contact_normal points from shape 0
# toward shape 1 (A-to-B convention) after running the full collision pipeline.


class TestRigidContactNormal(unittest.TestCase):
    pass


def test_rigid_contact_normal_sphere_sphere(test, device, broad_phase: str):
    """Verify rigid_contact_normal on four sphere-pair scenarios.

    All spheres have radius 0.5 and a per-shape gap of 0.05 (summed gap = 0.1).
    The four pairs are spaced along the Y axis so they don't interact:

    * Pair 0 - **overlap**: centers 0.6 apart  (penetration = -0.4)
    * Pair 1 - **exact touch**: centers 1.0 apart  (penetration = 0.0)
    * Pair 2 - **within gap**: centers 1.08 apart  (separation 0.08 < summed gap 0.1)
    * Pair 3 - **separated**: centers 1.5 apart  (well outside gap, no contact)

    For every contact produced the test checks:
    1. Normal is unit length.
    2. Normal points from shape 0 toward shape 1 (A-to-B convention).
    3. Contact midpoint lies between the two sphere centers.

    Pair 3 must produce zero contacts.
    """
    with wp.ScopedDevice(device):
        radius = 0.5
        gap = 0.05

        pair_half_dists = [0.3, 0.5, 0.54, 0.75]
        y_offsets = [0.0, 3.0, 6.0, 9.0]
        expect_contact = [True, True, True, False]

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.rigid_gap = gap

        positions = []
        for half_dist, y in zip(pair_half_dists, y_offsets, strict=True):
            pa = wp.vec3(-half_dist, y, 0.0)
            pb = wp.vec3(half_dist, y, 0.0)
            positions.append(pa)
            positions.append(pb)

            ba = builder.add_body(xform=wp.transform(pa))
            builder.add_shape_sphere(body=ba, radius=radius)
            bb = builder.add_body(xform=wp.transform(pb))
            builder.add_shape_sphere(body=bb, radius=radius)

        model = builder.finalize(device=device)
        state = model.state()

        pipeline = newton.CollisionPipeline(model, broad_phase=broad_phase)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        normals = contacts.rigid_contact_normal.numpy()[:count]
        shape0s = contacts.rigid_contact_shape0.numpy()[:count]
        shape1s = contacts.rigid_contact_shape1.numpy()[:count]
        point0s = contacts.rigid_contact_point0.numpy()[:count]
        point1s = contacts.rigid_contact_point1.numpy()[:count]

        positions_np = np.array(positions, dtype=np.float32)

        expected_contacting_pairs = sum(expect_contact)
        contacts_per_pair: dict[int, list[int]] = {p: [] for p in range(4)}
        for i in range(count):
            s0 = int(shape0s[i])
            pair_idx = s0 // 2
            contacts_per_pair[pair_idx].append(i)

        pairs_with_contacts = sum(1 for c in contacts_per_pair.values() if c)
        test.assertEqual(
            pairs_with_contacts,
            expected_contacting_pairs,
            f"Expected exactly {expected_contacting_pairs} pairs with contacts, got {pairs_with_contacts}",
        )

        for pair_idx in range(4):
            pair_contacts = contacts_per_pair[pair_idx]
            label = f"pair {pair_idx} (half_dist={pair_half_dists[pair_idx]})"

            if not expect_contact[pair_idx]:
                test.assertEqual(len(pair_contacts), 0, f"{label}: expected no contacts but got {len(pair_contacts)}")
                continue

            test.assertGreater(len(pair_contacts), 0, f"{label}: expected at least one contact")

            for i in pair_contacts:
                normal = normals[i]
                s0 = int(shape0s[i])
                s1 = int(shape1s[i])

                normal_len = np.linalg.norm(normal)
                test.assertAlmostEqual(
                    normal_len,
                    1.0,
                    places=3,
                    msg=f"{label} contact {i}: normal must be unit length (got {normal_len})",
                )

                center_a = positions_np[s0]
                center_b = positions_np[s1]
                expected_dir = center_b - center_a
                expected_dir = expected_dir / np.linalg.norm(expected_dir)

                dot = np.dot(normal, expected_dir)
                test.assertGreater(
                    dot,
                    0.95,
                    f"{label} contact {i}: normal must point from shape {s0} toward shape {s1} "
                    f"(dot={dot:.4f}, normal={normal}, expected_dir={expected_dir})",
                )

                # point0/point1 are in body-local frames; transform to world
                p0_world = point0s[i] + center_a
                p1_world = point1s[i] + center_b
                midpoint = (p0_world + p1_world) / 2.0
                lo = min(center_a[0], center_b[0])
                hi = max(center_a[0], center_b[0])
                test.assertTrue(
                    lo - 1e-3 <= midpoint[0] <= hi + 1e-3,
                    f"{label} contact {i}: midpoint x={midpoint[0]:.4f} should lie between "
                    f"center x=[{lo:.4f}, {hi:.4f}]",
                )


for bp_name in ("explicit", "nxn", "sap"):
    add_function_test(
        TestRigidContactNormal,
        f"test_rigid_contact_normal_sphere_sphere_{bp_name}",
        test_rigid_contact_normal_sphere_sphere,
        devices=devices,
        broad_phase=bp_name,
    )


def test_box_box_quaternion_perturbation(test, device, broad_phase: str):
    """Verify box-box contacts are correct under tiny quaternion perturbation.

    Two identical cubes are placed face-to-face with a non-trivial base
    rotation (30 deg around X) and a tiny quaternion perturbation (~1e-14)
    in the second box only. Without the support-map deadband fix this
    produces 1 invalid contact instead of 4 face-corner contacts, with an
    out-of-bounds body-frame point and a wrong world-frame normal.

    Regression test for issue #2024 / #2430.
    """
    with wp.ScopedDevice(device):
        half = 0.495
        q_clean = wp.quat(0.2588233343021173, 0.0, 0.0, 0.9659246769912934)
        q_noisy = wp.quat(0.2588233343021173, -2.27e-14, 9.25e-15, 0.9659246769912934)

        y, z = 6.680443286895752, 4.4285125732421875

        builder = newton.ModelBuilder()
        b0 = builder.add_body(xform=wp.transform(p=wp.vec3(half, y, z), q=q_clean))
        builder.add_shape_box(body=b0, hx=half, hy=half, hz=half)

        b1 = builder.add_body(xform=wp.transform(p=wp.vec3(-half, y, z), q=q_noisy))
        builder.add_shape_box(body=b1, hx=half, hy=half, hz=half)

        model = builder.finalize(device=device)
        state = model.state()

        pipeline = newton.CollisionPipeline(model, broad_phase=broad_phase)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        cc = int(contacts.rigid_contact_count.numpy()[0])
        points0 = contacts.rigid_contact_point0.numpy()[:cc]
        normals = contacts.rigid_contact_normal.numpy()[:cc]

        test.assertEqual(cc, 4, f"Expected 4 face-corner contacts, got {cc}")

        for i in range(cc):
            pt = points0[i]
            n = normals[i]
            for j in range(3):
                test.assertLessEqual(
                    abs(pt[j]),
                    half * 1.01,
                    f"Contact {i} body-frame point[{j}] = {pt[j]:.4f} outside half-extent {half}",
                )
            test.assertAlmostEqual(
                abs(n[0]),
                1.0,
                places=2,
                msg=f"Contact {i} normal = [{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}], expected [+-1, 0, 0]",
            )


for bp_name in ("explicit", "nxn", "sap"):
    add_function_test(
        TestRigidContactNormal,
        f"test_box_box_quaternion_perturbation_{bp_name}",
        test_box_box_quaternion_perturbation,
        devices=devices,
        broad_phase=bp_name,
    )


# ============================================================================
# Particle-Shape (Soft) Contact Tests
# ============================================================================
# These tests verify that particle-shape contacts are correctly generated
# by both collision pipelines.


class TestParticleShapeContacts(unittest.TestCase):
    def _assert_pairs_valid(self, model, pipeline):
        # Pairs are a world-compatible superset over all particles/shapes; ACTIVE / COLLIDE_PARTICLES
        # are filtered dynamically in create_soft_contacts, so only world compatibility is asserted.
        pw = model.particle_world.numpy()
        sw = model.shape_world.numpy()
        for p, s in pipeline.soft_rigid_contact_pairs.numpy():
            self.assertTrue(pw[p] == sw[s] or pw[p] < 0 or sw[s] < 0, f"cross-world pair ({p}, {s})")

    def test_soft_rigid_pairs_multi_world_isolated(self):
        sub = newton.ModelBuilder()
        sub.add_shape_sphere(body=-1, radius=1.0)
        sub.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        builder = newton.ModelBuilder()
        builder.add_world(sub)
        builder.add_world(sub)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

        # Two worlds, each one active particle x one particle-colliding shape; no cross-world pairs.
        self.assertEqual(pipeline.soft_rigid_contact_pair_count, 2)
        self._assert_pairs_valid(model, pipeline)

    def test_soft_contacts_respect_active_and_collide_flags(self):
        # Pairs are a world-compatible superset (flags are not baked in); create_soft_contacts applies
        # ACTIVE / COLLIDE_PARTICLES dynamically, so only the active particle x the particle-colliding
        # shape actually produces a contact.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()  # collides with particles (default)
        builder.add_shape_sphere(body=-1, radius=1.0, cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False))
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.05), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)  # active
        builder.add_particle(pos=wp.vec3(0.1, 0.0, 0.05), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0, flags=0)  # inactive
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()

        # 2 particles x 2 shapes, all in the global world -> 4 candidate pairs regardless of flags.
        self.assertEqual(pipeline.soft_rigid_contact_pair_count, 4)
        self._assert_pairs_valid(model, pipeline)

        pipeline.collide(model.state(), contacts)
        # Only (active particle, particle-colliding ground) survives the dynamic flag checks.
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 1)

    def test_soft_contacts_track_runtime_flag_changes(self):
        # Regression: pairs are precomputed once, so a particle activated *after* the pipeline is built
        # must still produce a contact (flags are filtered dynamically, not baked into the pair list).
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.05), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0, flags=0)  # inactive
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()

        # The candidate pair is cached even though the particle is inactive at construction.
        self.assertEqual(pipeline.soft_rigid_contact_pair_count, 1)
        pipeline.collide(model.state(), contacts)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 0)

        # Activate the particle at runtime -> the contact appears without rebuilding the pipeline.
        flags = model.particle_flags.numpy()
        flags[0] = int(ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
        pipeline.collide(model.state(), contacts)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 1)

    def test_soft_contact_capacity_defaults_to_pair_count(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 0.05),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.1,
        )
        model = builder.finalize(device="cpu")

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()

        self.assertEqual(pipeline.soft_contact_max, pipeline.soft_rigid_contact_pair_count)
        self.assertEqual(contacts.soft_contact_max, pipeline.soft_rigid_contact_pair_count)

    def test_soft_contact_explicit_capacity_is_respected(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.05), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_max=1)

        self.assertEqual(pipeline.soft_rigid_contact_pair_count, 1)
        self.assertEqual(pipeline.soft_contact_max, 1)

    def test_soft_contact_explicit_capacity_overflow_still_counts_candidates(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.05), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=wp.vec3(0.1, 0.0, 0.05), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_max=1)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)

        self.assertEqual(pipeline.soft_rigid_contact_pair_count, 2)
        self.assertEqual(contacts.soft_contact_max, 1)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 2)

    def test_soft_contacts_skip_cross_world_shape_particle_pairs(self):
        particle_builder = newton.ModelBuilder()
        particle_builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)

        shape_builder = newton.ModelBuilder()
        shape_builder.add_shape_sphere(body=-1, radius=1.0)

        builder = newton.ModelBuilder()
        builder.add_world(particle_builder)
        builder.add_world(shape_builder)
        model = builder.finalize(device="cpu")

        contacts = model.collide(model.state())

        self.assertEqual(model._collision_pipeline.soft_rigid_contact_pair_count, 0)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 0)

    def test_global_shape_contacts_particles_in_all_worlds(self):
        particle_builder = newton.ModelBuilder()
        particle_builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.05), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_world(particle_builder)
        builder.add_world(particle_builder)
        model = builder.finalize(device="cpu")

        contacts = model.contacts()
        model.collide(model.state(), contacts)

        self.assertEqual(model._collision_pipeline.soft_rigid_contact_pair_count, 2)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 2)


class TestContactEstimator(unittest.TestCase):
    def test_visual_only_meshes_do_not_inflate_estimate(self):
        """Visual meshes should not affect rigid contact capacity estimates."""
        model = newton.Model()
        model.world_count = 1
        model.shape_contact_pair_count = 0

        shape_type = np.array(
            [int(GeoType.BOX)] * 4 + [int(GeoType.MESH)] * 100,
            dtype=np.int32,
        )
        shape_world = np.zeros(len(shape_type), dtype=np.int32)
        shape_flags = np.array(
            [int(ShapeFlags.COLLIDE_SHAPES)] * 4 + [int(ShapeFlags.VISIBLE)] * 100,
            dtype=np.int32,
        )

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)
        model.shape_flags = wp.array(shape_flags, dtype=wp.int32)

        estimate = _estimate_rigid_contact_max(model)
        self.assertEqual(estimate, 1000)

    def test_heuristic_caps_large_pair_count(self):
        """When pair count is huge, the heuristic provides a tighter bound."""
        model = newton.Model()
        model.world_count = 1
        model.shape_contact_pair_count = 999999

        # 4 primitives (CPP=5), 3 meshes (CPP=40), 2 planes, all in world 0.
        # non-plane: (4*20*5 + 3*20*40) // 2 = (400 + 2400) // 2 = 1400
        # weighted_plane_cpp: (4*5 + 3*40) // 7 = 140 // 7 = 20
        # plane (per-world): 2*7 pairs * 20 = 280
        # heuristic = 1680, pair = huge => min = 1680
        shape_type = np.array(
            [int(GeoType.BOX)] * 4 + [int(GeoType.MESH)] * 3 + [int(GeoType.PLANE)] * 2,
            dtype=np.int32,
        )
        shape_world = np.zeros(len(shape_type), dtype=np.int32)

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)

        estimate = _estimate_rigid_contact_max(model)
        self.assertEqual(estimate, 1680)

    def test_world_aware_plane_estimate(self):
        """Per-world plane computation avoids quadratic cross-world overcount."""
        model = newton.Model()
        model.world_count = 4
        model.shape_contact_pair_count = 0

        # 4 worlds, each with 10 boxes (CPP=5) and 10 planes.
        # non-plane: (40*20*5) // 2 = 2000
        # weighted_plane_cpp: (40*5) // 40 = 5
        # plane (per-world): 4*(10*10) pairs * 5 = 2000
        # total = 4000
        shape_type = np.array(
            ([int(GeoType.BOX)] * 10 + [int(GeoType.PLANE)] * 10) * 4,
            dtype=np.int32,
        )
        shape_world = np.repeat(np.arange(4, dtype=np.int32), 20)

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)

        estimate = _estimate_rigid_contact_max(model)
        self.assertEqual(estimate, 4000)

    def test_pair_count_tighter_than_heuristic(self):
        """When precomputed pair count is tighter than the heuristic, it is used."""
        model = newton.Model()
        model.world_count = 4
        model.shape_contact_pair_count = 300

        # 40 boxes (CPP=5) across 4 worlds, no planes.
        # heuristic: (40*20*5) // 2 = 2000
        # weighted_cpp: max(5, 5) = 5
        # pair-based: 300 * 5 = 1500
        # min(2000, 1500) = 1500
        shape_type = np.array(
            [int(GeoType.BOX)] * 40,
            dtype=np.int32,
        )
        shape_world = np.repeat(np.arange(4, dtype=np.int32), 10)

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)

        estimate = _estimate_rigid_contact_max(model)
        self.assertEqual(estimate, 1500)


class TestShapePairsMaxScaling(unittest.TestCase):
    """Verify that shape_pairs_max scales linearly with world count, not quadratically."""

    @staticmethod
    def _make_model(num_worlds, shapes_per_world, num_global=0, shape_flags_value=None):
        """Build a minimal Model with the given world/shape layout."""
        total = num_worlds * shapes_per_world + num_global
        world_ids = np.repeat(np.arange(num_worlds, dtype=np.int32), shapes_per_world)
        if num_global > 0:
            world_ids = np.concatenate([world_ids, np.full(num_global, -1, dtype=np.int32)])

        model = newton.Model()
        model.shape_count = total
        model.shape_world = wp.array(world_ids, dtype=wp.int32)

        if shape_flags_value is not None:
            flags = np.full(total, shape_flags_value, dtype=np.int32)
        else:
            flags = np.full(total, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        model.shape_flags = wp.array(flags, dtype=wp.int32)
        return model

    def test_single_world_matches_global_formula(self):
        """Single world should give the same result as the naive N*(N-1)/2."""
        model = self._make_model(num_worlds=1, shapes_per_world=20)
        result = _compute_per_world_shape_pairs_max(model)
        self.assertEqual(result, 20 * 19 // 2)

    def test_multi_world_scales_linearly(self):
        """Doubling worlds should roughly double shape_pairs_max, not quadruple it."""
        model_w1 = self._make_model(num_worlds=1, shapes_per_world=20)
        model_w2 = self._make_model(num_worlds=2, shapes_per_world=20)
        model_w4 = self._make_model(num_worlds=4, shapes_per_world=20)

        pairs_w1 = _compute_per_world_shape_pairs_max(model_w1)
        pairs_w2 = _compute_per_world_shape_pairs_max(model_w2)
        pairs_w4 = _compute_per_world_shape_pairs_max(model_w4)

        self.assertEqual(pairs_w1, 190)
        self.assertEqual(pairs_w2, 2 * 190)
        self.assertEqual(pairs_w4, 4 * 190)

    def test_many_worlds_no_quadratic_blowup(self):
        """At 256 worlds the per-world sum must be far below the global N^2 formula."""
        num_worlds = 256
        spw = 10
        model = self._make_model(num_worlds=num_worlds, shapes_per_world=spw)
        result = _compute_per_world_shape_pairs_max(model)

        per_world_expected = num_worlds * (spw * (spw - 1) // 2)
        global_n = num_worlds * spw
        global_quadratic = global_n * (global_n - 1) // 2

        self.assertEqual(result, per_world_expected)
        self.assertLess(result, global_quadratic / 100, "shape_pairs_max must not scale quadratically with world count")

    def test_global_shapes_included_per_world(self):
        """Global shapes (world=-1) are added to each world's segment."""
        model = self._make_model(num_worlds=2, shapes_per_world=3, num_global=1)
        result = _compute_per_world_shape_pairs_max(model)
        # Each world: 3 local + 1 global = 4 shapes -> 6 pairs
        # Dedicated -1 segment: 1 shape -> 0 pairs
        self.assertEqual(result, 2 * 6)

    def test_global_shapes_dedicated_segment(self):
        """Multiple global shapes get their own dedicated segment."""
        model = self._make_model(num_worlds=2, shapes_per_world=3, num_global=2)
        result = _compute_per_world_shape_pairs_max(model)
        # Each world: 3 + 2 = 5 -> 10 pairs. Dedicated: 2 -> 1 pair.
        self.assertEqual(result, 2 * 10 + 1)

    def test_no_shapes(self):
        model = newton.Model()
        model.shape_count = 0
        model.shape_world = None
        self.assertEqual(_compute_per_world_shape_pairs_max(model), 0)

    def test_pipeline_buffer_size_scales_linearly(self):
        """End-to-end: CollisionPipeline buffers must not explode with many worlds."""
        num_worlds = 64
        spw = 5

        robot_builder = newton.ModelBuilder()
        for _ in range(spw):
            b = robot_builder.add_body()
            robot_builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1)

        builder = newton.ModelBuilder()
        for _ in range(num_worlds):
            builder.add_world(robot_builder)

        model = builder.finalize()

        for bp_mode in ("nxn", "sap"):
            pipeline = newton.CollisionPipeline(model, broad_phase=bp_mode)

            global_n = model.shape_count
            global_quadratic = global_n * (global_n - 1) // 2
            per_world_linear = num_worlds * (spw * (spw - 1) // 2)

            self.assertEqual(
                pipeline.shape_pairs_max,
                per_world_linear,
                f"broad_phase={bp_mode}: shape_pairs_max should scale linearly",
            )
            self.assertLess(
                pipeline.shape_pairs_max,
                global_quadratic / 10,
                f"broad_phase={bp_mode}: shape_pairs_max must not be quadratic",
            )

    def test_visual_only_mesh_does_not_enable_mesh_narrow_phase(self):
        """Visual meshes should not opt the pipeline into mesh contact kernels."""
        builder = newton.ModelBuilder()
        body_a = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity()))
        body_b = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body_a, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=body_b, hx=0.1, hy=0.1, hz=0.1)

        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
            is_visible=True,
        )
        visual_mesh = newton.Mesh.create_box(
            0.2,
            0.2,
            0.2,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        builder.add_shape_mesh(body=body_a, mesh=visual_mesh, cfg=visual_cfg)

        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, broad_phase="sap")

        self.assertFalse(pipeline.narrow_phase.has_meshes)


def test_particle_shape_contacts(test, device, shape_type: GeoType):
    """
    Test that particle-shape contacts are correctly generated.

    Creates a cloth grid (particles) above a shape and verifies that
    soft contacts are generated when the particles are within contact margin.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()

        # Add a shape for particles to collide with
        if shape_type == GeoType.PLANE:
            builder.add_ground_plane()
        elif shape_type == GeoType.BOX:
            builder.add_shape_box(
                body=-1,  # static shape
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
                hx=2.0,
                hy=2.0,
                hz=0.5,
            )
        elif shape_type == GeoType.SPHERE:
            builder.add_shape_sphere(
                body=-1,
                xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()),
                radius=1.0,
            )

        # Add cloth grid (particles) slightly above the shape
        # Position them within the soft contact margin
        particle_z = 0.05  # Just above ground plane at z=0
        soft_contact_margin = 0.1
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, particle_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5,
            dim_y=5,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.1,
        )

        model = builder.finalize(device=device)

        # Create collision pipeline
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_margin=soft_contact_margin,
        )

        state = model.state()

        # Run collision detection
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state, contacts)

        # Verify soft contacts were generated
        soft_count = contacts.soft_contact_count.numpy()[0]

        # All particles should be within contact margin of the shape
        # For a 6x6 grid (dim+1), that's 36 particles
        expected_particle_count = 36
        test.assertEqual(model.particle_count, expected_particle_count, f"Expected {expected_particle_count} particles")

        # Each particle should generate a contact with the shape
        test.assertGreater(
            soft_count,
            0,
            f"Expected soft contacts to be generated (got {soft_count})",
        )

        # Verify contact data is valid
        if soft_count > 0:
            contact_particles = contacts.soft_contact_particle.numpy()[:soft_count]
            contact_shapes = contacts.soft_contact_shape.numpy()[:soft_count]
            contact_normals = contacts.soft_contact_normal.numpy()[:soft_count]

            # All particle indices should be valid
            test.assertTrue(
                (contact_particles >= 0).all() and (contact_particles < model.particle_count).all(),
                "Contact particle indices should be valid",
            )

            # All shape indices should be valid
            test.assertTrue(
                (contact_shapes >= 0).all() and (contact_shapes < model.shape_count).all(),
                "Contact shape indices should be valid",
            )

            # Contact normals should be normalized (or close to it)
            normal_lengths = np.linalg.norm(contact_normals, axis=1)
            test.assertTrue(
                np.allclose(normal_lengths, 1.0, atol=0.01),
                f"Contact normals should be normalized, got lengths: {normal_lengths}",
            )


# Shape types to test for particle-shape contacts
particle_shape_tests = [
    GeoType.PLANE,
    GeoType.BOX,
    GeoType.SPHERE,
]


# Add tests for collision pipeline
for shape_type in particle_shape_tests:
    add_function_test(
        TestParticleShapeContacts,
        f"test_particle_{type_to_str(shape_type)}",
        test_particle_shape_contacts,
        devices=devices,
        shape_type=shape_type,
    )


# ============================================================================
# Full-scene deterministic contact pipeline test
# ============================================================================


class TestDeterministicPipeline(unittest.TestCase):
    """Test that deterministic=True yields bit-identical contacts across collide calls."""

    pass


class TestMeshConvexMidphase(unittest.TestCase):
    """Test mesh-vs-convex triangle candidate generation."""

    pass


class TestHeightfieldConvexMidphase(unittest.TestCase):
    """Test heightfield-vs-convex triangle candidate generation."""

    pass


class TestPlanarSDFRouting(unittest.TestCase):
    """Test normal SDF contact routing for planar-faced non-mesh shapes."""

    pass


class TestCurvedPrimitiveSDFExclusion(unittest.TestCase):
    def test_non_hydro_sphere_sdf_config_does_not_build_general_sdf(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        cfg = newton.ModelBuilder.ShapeConfig(sdf_max_resolution=32)
        sphere = builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg)

        model = builder.finalize()

        self.assertEqual(int(model._shape_sdf_index.numpy()[sphere]), -1)


class TestPlanarSDFOptIn(unittest.TestCase):
    def test_default_box_does_not_build_general_sdf(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        box = builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)

        model = builder.finalize()

        self.assertEqual(int(model._shape_sdf_index.numpy()[box]), -1)
        self.assertEqual(int(model.shape_edge_range.numpy()[box][1]), 0)

    def test_default_convex_hull_does_not_build_general_sdf(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        mesh = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
        convex = builder.add_shape_convex_hull(body=body, mesh=mesh)

        model = builder.finalize()

        self.assertEqual(int(model._shape_sdf_index.numpy()[convex]), -1)


def test_mesh_convex_midphase_queries_margin_shell(test, device):
    margin = 0.02
    gap = 0.005
    radius = 0.1
    surface_separation = 0.03

    cfg = newton.ModelBuilder.ShapeConfig(margin=margin, gap=gap)
    builder = newton.ModelBuilder()

    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh(vertices, indices), cfg=cfg)

    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius + surface_separation), wp.quat_identity()))
    builder.add_joint_free(child=body)
    builder.add_shape_sphere(body=body, radius=radius, cfg=cfg)

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(contact_count, 0)


def test_mesh_convex_with_sdf_routes_to_sdf_contact(test, device):
    """A convex mesh with SDF should use the SDF pair route against a triangle mesh."""
    mesh = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
    convex = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
    mesh.build_sdf(max_resolution=32, device=device)
    convex.build_sdf(max_resolution=32, device=device)

    builder = newton.ModelBuilder()
    body_mesh = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_convex = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()))
    builder.add_shape_mesh(body=body_mesh, mesh=mesh)
    builder.add_shape_convex_hull(body=body_convex, mesh=convex)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=256)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    sdf_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
    mesh_convex_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_count.numpy()[0])
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(sdf_pair_count, 0)
    test.assertEqual(mesh_convex_pair_count, 0)
    test.assertGreater(contact_count, 0)


def test_mesh_convex_one_sdf_keeps_existing_route(test, device):
    """Avoid SDF routing when it would require expensive BVH fallback on one side."""
    mesh = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
    convex = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
    convex.build_sdf(max_resolution=32, device=device)

    builder = newton.ModelBuilder()
    body_mesh = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_convex = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()))
    builder.add_shape_mesh(body=body_mesh, mesh=mesh)
    builder.add_shape_convex_hull(body=body_convex, mesh=convex)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=256)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    sdf_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
    mesh_convex_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_count.numpy()[0])
    test.assertEqual(sdf_pair_count, 0)
    test.assertGreater(mesh_convex_pair_count, 0)


def test_mesh_box_with_sdf_routes_to_sdf_contact(test, device):
    """An explicitly SDF-configured box should use the SDF pair route against a mesh."""
    mesh = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
    mesh.build_sdf(max_resolution=32, device=device)

    builder = newton.ModelBuilder()
    body_mesh = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()))
    builder.add_shape_mesh(body=body_mesh, mesh=mesh)
    box_cfg = newton.ModelBuilder.ShapeConfig(sdf_max_resolution=32)
    box = builder.add_shape_box(body=body_box, hx=0.5, hy=0.5, hz=0.5, cfg=box_cfg)

    model = builder.finalize(device=device)
    test.assertGreaterEqual(int(model._shape_sdf_index.numpy()[box]), 0)
    test.assertEqual(int(model.shape_edge_range.numpy()[box][1]), 12)

    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=256)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    sdf_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
    mesh_convex_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_count.numpy()[0])
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(sdf_pair_count, 0)
    test.assertEqual(mesh_convex_pair_count, 0)
    test.assertGreater(contact_count, 0)


def test_box_box_with_sdf_keeps_primitive_route(test, device):
    """Box-box contacts should keep the primitive path even when both boxes have SDFs."""
    cfg = newton.ModelBuilder.ShapeConfig(sdf_max_resolution=32)
    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()))
    box_a = builder.add_shape_box(body=body_a, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
    box_b = builder.add_shape_box(body=body_b, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)

    model = builder.finalize(device=device)
    shape_sdf_index = model._shape_sdf_index.numpy()
    test.assertGreaterEqual(int(shape_sdf_index[box_a]), 0)
    test.assertGreaterEqual(int(shape_sdf_index[box_b]), 0)

    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=256)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    sdf_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
    gjk_pair_count = int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0])
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(sdf_pair_count, 0)
    test.assertGreater(gjk_pair_count, 0)
    test.assertGreater(contact_count, 0)


def test_convex_convex_with_sdf_routes_to_sdf_contact(test, device):
    """Two SDF-backed convex meshes should use the SDF path and produce contacts."""
    mesh_a = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
    mesh_b = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
    mesh_a.build_sdf(max_resolution=32, device=device)
    mesh_b.build_sdf(max_resolution=32, device=device)

    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()))
    builder.add_shape_convex_hull(body=body_a, mesh=mesh_a)
    builder.add_shape_convex_hull(body=body_b, mesh=mesh_b)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=256)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    sdf_pair_count = int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
    gjk_pair_count = int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0])
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(sdf_pair_count, 0)
    test.assertEqual(gjk_pair_count, 0)
    test.assertGreater(contact_count, 0)


def test_heightfield_convex_midphase_queries_margin_shell_at_lateral_edge(test, device):
    margin = 0.02
    gap = 0.005
    radius = 0.1
    surface_separation = 0.03

    cfg = newton.ModelBuilder.ShapeConfig(margin=margin, gap=gap)
    builder = newton.ModelBuilder()

    heightfield = newton.Heightfield(
        data=np.zeros((3, 3), dtype=np.float32),
        nrow=3,
        ncol=3,
        hx=1.0,
        hy=1.0,
        min_z=0.0,
        max_z=0.0,
    )
    builder.add_shape_heightfield(heightfield=heightfield, cfg=cfg)

    body = builder.add_body(
        xform=wp.transform(wp.vec3(1.0 + radius + surface_separation, 0.0, 0.0), wp.quat_identity())
    )
    builder.add_joint_free(child=body)
    builder.add_shape_sphere(body=body, radius=radius, cfg=cfg)

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(contact_count, 0)


def _build_deterministic_scene(device):
    """Build the mixed-shape scene from example_basic_shapes6_determinism."""
    builder = newton.ModelBuilder()

    # Procedural mesh terrain ground
    terrain_vertices, terrain_indices = create_mesh_terrain(
        grid_size=(6, 6),
        block_size=(5.0, 5.0),
        terrain_types=["pyramid_stairs"],
        terrain_params={
            "pyramid_stairs": {"step_width": 0.4, "step_height": 0.05, "platform_width": 0.8},
        },
        seed=42,
    )
    terrain_mesh = newton.Mesh(terrain_vertices, terrain_indices)
    terrain_mesh.build_sdf(max_resolution=512)
    builder.add_shape_mesh(body=-1, mesh=terrain_mesh, xform=wp.transform(p=wp.vec3(-15.0, -15.0, -0.5)))

    # Icosahedron mesh
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_radius = 0.35
    ico_base_vertices = np.array(
        [
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
        ],
        dtype=np.float32,
    )
    for i in range(len(ico_base_vertices)):
        ico_base_vertices[i] = ico_base_vertices[i] / np.linalg.norm(ico_base_vertices[i]) * ico_radius
    ico_face_indices = [
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
    ico_vertices, ico_normals, ico_indices = [], [], []
    for face_idx, face in enumerate(ico_face_indices):
        v0, v1, v2 = ico_base_vertices[face[0]], ico_base_vertices[face[1]], ico_base_vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        ico_vertices.extend([v0, v1, v2])
        ico_normals.extend([normal, normal, normal])
        base = face_idx * 3
        ico_indices.extend([base, base + 1, base + 2])
    ico_mesh = newton.Mesh(
        np.array(ico_vertices, dtype=np.float32),
        np.array(ico_indices, dtype=np.int32),
        normals=np.array(ico_normals, dtype=np.float32),
    )

    # Cube mesh with SDF
    hs = 0.3
    cube_verts = np.array(
        [
            [-hs, -hs, -hs],
            [hs, -hs, -hs],
            [hs, hs, -hs],
            [-hs, hs, -hs],
            [-hs, -hs, hs],
            [hs, -hs, hs],
            [hs, hs, hs],
            [-hs, hs, hs],
        ],
        dtype=np.float32,
    )
    cube_tris = np.array(
        [0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
        dtype=np.int32,
    )
    cube_mesh = newton.Mesh(cube_verts, cube_tris)
    cube_mesh.build_sdf(max_resolution=64)

    # 4x4x4 grid of mixed shapes
    shape_types = ["sphere", "box", "capsule", "mesh_cube", "cylinder", "cone", "icosahedron"]
    grid_offset = wp.vec3(-5.0, -5.0, 0.5)
    rng = np.random.default_rng(42)
    shape_index = 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                pos = wp.vec3(
                    float(grid_offset[0]) + ix * 1.5 + (rng.random() - 0.5) * 0.4,
                    float(grid_offset[1]) + iy * 1.5 + (rng.random() - 0.5) * 0.4,
                    float(grid_offset[2]) + iz * 1.5 + (rng.random() - 0.5) * 0.4,
                )
                shape_type = shape_types[shape_index % len(shape_types)]
                shape_index += 1
                body = builder.add_body(xform=wp.transform(p=pos, q=wp.quat_identity()))
                if shape_type == "sphere":
                    builder.add_shape_sphere(body, radius=0.3)
                elif shape_type == "box":
                    builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.3)
                elif shape_type == "capsule":
                    builder.add_shape_capsule(body, radius=0.2, half_height=0.4)
                elif shape_type == "cylinder":
                    builder.add_shape_cylinder(body, radius=0.25, half_height=0.35)
                elif shape_type == "cone":
                    builder.add_shape_cone(body, radius=0.3, half_height=0.4)
                elif shape_type == "mesh_cube":
                    builder.add_shape_mesh(body, mesh=cube_mesh)
                elif shape_type == "icosahedron":
                    builder.add_shape_convex_hull(body, mesh=ico_mesh)
                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])

    model = builder.finalize(device=device)
    return model


def test_deterministic_pipeline_500_steps(test, device):
    """Run 500 frames of the mixed-shape scene and assert bit-identical contacts on every frame.

    GPU-only by construction (registered with ``get_cuda_test_devices``).
    All per-frame GPU work -- ``sim_substeps`` iterations of
    ``clear_forces`` / ``collide`` (both pipelines) / ``solver.step`` /
    Python state swap -- is captured into a single CUDA graph and
    replayed via ``wp.capture_launch``.  ``sim_substeps`` is even, so
    after one full frame the Python ``state_0``/``state_1`` references
    end up in their original orientation and the captured kernels
    reference the correct buffers on every replay.

    The contact arrays are checked at the frame boundary (rather than
    every substep) because graph capture serialises the substep loop
    into one launch; reading contacts mid-graph would require splitting
    or breaking out of capture.
    """
    test.assertTrue(wp.get_device(device).is_cuda, "Deterministic pipeline test requires a CUDA device")
    with wp.ScopedDevice(device):
        model = _build_deterministic_scene(device)

        pipeline_a = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            reduce_contacts=True,
            rigid_contact_max=50000,
        )
        pipeline_b = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            reduce_contacts=True,
            rigid_contact_max=50000,
        )
        contacts_a = pipeline_a.contacts()
        contacts_b = pipeline_b.contacts()

        solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_contact_relaxation=0.8)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        fps = 100
        sim_substeps = 10
        sim_dt = 1.0 / fps / sim_substeps
        assert sim_substeps % 2 == 0, (
            "Even sim_substeps required so state ref parity is preserved across the captured graph"
        )

        checked_arrays = [
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_point0",
            "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_offset0",
            "rigid_contact_offset1",
            "rigid_contact_margin0",
            "rigid_contact_margin1",
        ]

        # Capture the per-frame substep loop.  ``state_0``/``state_1``
        # are local Python names that get rebound by the swap inside the
        # loop; because ``sim_substeps`` is even, the names end up in
        # their original orientation by the time the graph closes, so
        # replays operate on the same buffers in the same order.
        def _frame():
            nonlocal state_0, state_1
            for _ in range(sim_substeps):
                state_0.clear_forces()
                pipeline_a.collide(state_0, contacts_a)
                pipeline_b.collide(state_0, contacts_b)
                solver.step(state_0, state_1, control, contacts_a, sim_dt)
                state_0, state_1 = state_1, state_0

        # Warm-up frame outside capture so lazy module loads / JIT
        # finish before recording.  This advances the simulation by one
        # frame; that's harmless for a determinism check (both pipelines
        # see the same warm-up state) and is the standard Newton
        # graph-capture pattern (see ``test_rigid_contact``,
        # ``example_basic_pendulum``).
        _frame()

        with wp.ScopedCapture(device=device) as capture:
            _frame()
        graph = capture.graph

        for _frame_idx in range(500):
            wp.capture_launch(graph)

            count_a = int(contacts_a.rigid_contact_count.numpy()[0])
            count_b = int(contacts_b.rigid_contact_count.numpy()[0])
            test.assertEqual(
                count_a,
                count_b,
                f"Contact count mismatch at frame {_frame_idx}: {count_a} vs {count_b}",
            )
            if count_a > 0:
                # Also compare sort keys to distinguish ordering vs value issues
                keys_a = pipeline_a._sort_key_array.numpy()[:count_a]
                keys_b = pipeline_b._sort_key_array.numpy()[:count_a]
                keys_match = np.array_equal(keys_a, keys_b)

                for name in checked_arrays:
                    a = getattr(contacts_a, name).numpy()[:count_a]
                    b = getattr(contacts_b, name).numpy()[:count_a]
                    if not np.array_equal(a, b):
                        diff_mask = a != b
                        diff_indices = np.argwhere(diff_mask)
                        msg = (
                            f"Determinism failure in {name} at frame {_frame_idx} "
                            f"({int(np.count_nonzero(diff_mask))} elements differ, {count_a} contacts)\n"
                            f"  sort_keys_match={keys_match}\n"
                        )
                        for raw_idx in diff_indices[:5]:
                            tidx = tuple(raw_idx)
                            msg += f"  [{tidx}]: a={a[tidx]!r}  b={b[tidx]!r}  diff={float(a[tidx]) - float(b[tidx]):.18e}\n"
                        if not keys_match:
                            key_diff = np.argwhere(keys_a != keys_b)
                            msg += f"  sort_key diffs at indices: {key_diff[:10].flatten().tolist()}\n"
                            for ki in key_diff[:5].flatten():
                                msg += f"    key[{ki}]: a=0x{keys_a[ki]:016x}  b=0x{keys_b[ki]:016x}\n"
                        # Show shape pairs for differing contacts
                        s0_a = contacts_a.rigid_contact_shape0.numpy()[:count_a]
                        s1_a = contacts_a.rigid_contact_shape1.numpy()[:count_a]
                        for idx in diff_indices[:5]:
                            ci = idx[0] if len(idx) > 1 else int(idx)
                            msg += f"  contact[{ci}]: shapes=({s0_a[ci]}, {s1_a[ci]}), key_a=0x{keys_a[ci]:016x}\n"
                        test.assertTrue(False, msg)


add_function_test(
    TestDeterministicPipeline,
    "test_deterministic_pipeline_500_steps",
    test_deterministic_pipeline_500_steps,
    devices=get_cuda_test_devices(),
    check_output=False,
)


def test_deterministic_pipeline_sticky_500_steps(test, device):
    """Same scene as ``test_deterministic_pipeline_500_steps`` but with sticky
    contact matching enabled.

    Sticky mode runs the matcher (which carries cross-frame state) and then
    overwrites matched rows with the previous frame's body-frame contact
    geometry via ``replay_matched``.  Two parallel pipelines starting from
    the same state and stepping the same input must therefore evolve
    identical match indices and identical replayed contact geometry every
    frame -- this is the regression test for the sticky-mode tie-break
    determinism fix in the contact matcher.

    GPU-only and graph-captured (see
    ``test_deterministic_pipeline_500_steps`` for the rationale).
    """
    test.assertTrue(wp.get_device(device).is_cuda, "Sticky deterministic pipeline test requires a CUDA device")
    with wp.ScopedDevice(device):
        model = _build_deterministic_scene(device)

        common_kwargs = {
            "broad_phase": "nxn",
            "reduce_contacts": True,
            "rigid_contact_max": 50000,
            # contact_matching="sticky" implies deterministic=True.
            "contact_matching": "sticky",
        }
        pipeline_a = newton.CollisionPipeline(model, **common_kwargs)
        pipeline_b = newton.CollisionPipeline(model, **common_kwargs)
        contacts_a = pipeline_a.contacts()
        contacts_b = pipeline_b.contacts()

        solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_contact_relaxation=0.8)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        fps = 100
        sim_substeps = 10
        sim_dt = 1.0 / fps / sim_substeps
        assert sim_substeps % 2 == 0, (
            "Even sim_substeps required so state ref parity is preserved across the captured graph"
        )

        checked_arrays = [
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_point0",
            "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_offset0",
            "rigid_contact_offset1",
            "rigid_contact_margin0",
            "rigid_contact_margin1",
            "rigid_contact_match_index",
        ]

        def _frame():
            nonlocal state_0, state_1
            for _ in range(sim_substeps):
                state_0.clear_forces()
                pipeline_a.collide(state_0, contacts_a)
                pipeline_b.collide(state_0, contacts_b)
                solver.step(state_0, state_1, control, contacts_a, sim_dt)
                state_0, state_1 = state_1, state_0

        # Warm-up frame outside capture so lazy module loads / JIT
        # finish before recording.  This advances the simulation by one
        # frame; harmless for a determinism check (both pipelines see
        # the same warm-up state).
        _frame()

        with wp.ScopedCapture(device=device) as capture:
            _frame()
        graph = capture.graph

        # Sticky adds per-step work; 100 frames * 10 substeps = 1000
        # collide calls is enough to let cross-frame state accumulate
        # and exercise the resolve/replay paths thoroughly.
        num_frames = 100
        for _frame_idx in range(num_frames):
            wp.capture_launch(graph)

            count_a = int(contacts_a.rigid_contact_count.numpy()[0])
            count_b = int(contacts_b.rigid_contact_count.numpy()[0])
            test.assertEqual(
                count_a,
                count_b,
                f"Sticky contact count mismatch at frame {_frame_idx}: {count_a} vs {count_b}",
            )
            if count_a > 0:
                keys_a = pipeline_a._sort_key_array.numpy()[:count_a]
                keys_b = pipeline_b._sort_key_array.numpy()[:count_a]
                keys_match = np.array_equal(keys_a, keys_b)

                for name in checked_arrays:
                    a = getattr(contacts_a, name).numpy()[:count_a]
                    b = getattr(contacts_b, name).numpy()[:count_a]
                    if not np.array_equal(a, b):
                        diff_mask = a != b
                        diff_indices = np.argwhere(diff_mask)
                        msg = (
                            f"Sticky determinism failure in {name} at frame {_frame_idx} "
                            f"({int(np.count_nonzero(diff_mask))} elements differ, {count_a} contacts)\n"
                            f"  sort_keys_match={keys_match}\n"
                        )
                        for raw_idx in diff_indices[:5]:
                            tidx = tuple(raw_idx)
                            msg += f"  [{tidx}]: a={a[tidx]!r}  b={b[tidx]!r}\n"
                        test.assertTrue(False, msg)


add_function_test(
    TestDeterministicPipeline,
    "test_deterministic_pipeline_sticky_500_steps",
    test_deterministic_pipeline_sticky_500_steps,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestMeshConvexMidphase,
    "test_mesh_convex_midphase_queries_margin_shell",
    test_mesh_convex_midphase_queries_margin_shell,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestPlanarSDFRouting,
    "test_mesh_convex_with_sdf_routes_to_sdf_contact",
    test_mesh_convex_with_sdf_routes_to_sdf_contact,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestPlanarSDFRouting,
    "test_mesh_convex_one_sdf_keeps_existing_route",
    test_mesh_convex_one_sdf_keeps_existing_route,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestPlanarSDFRouting,
    "test_mesh_box_with_sdf_routes_to_sdf_contact",
    test_mesh_box_with_sdf_routes_to_sdf_contact,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestPlanarSDFRouting,
    "test_box_box_with_sdf_keeps_primitive_route",
    test_box_box_with_sdf_keeps_primitive_route,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestPlanarSDFRouting,
    "test_convex_convex_with_sdf_routes_to_sdf_contact",
    test_convex_convex_with_sdf_routes_to_sdf_contact,
    devices=get_cuda_test_devices(),
    check_output=False,
)

add_function_test(
    TestHeightfieldConvexMidphase,
    "test_heightfield_convex_midphase_queries_margin_shell_at_lateral_edge",
    test_heightfield_convex_midphase_queries_margin_shell_at_lateral_edge,
    devices=get_cuda_test_devices(),
    check_output=False,
)


def _build_cloth_over_plane(device, particle_z: float = 0.05):
    """A 5x5 cloth grid hovering just above a ground plane (particles overlap within margin)."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, particle_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=5,
        dim_y=5,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    return builder.finalize(device=device)


def test_soft_contact_schema(test, device):
    """soft_contact_count is a 1-element total; unified soft_contact_indices + barycentric added."""
    model = _build_cloth_over_plane(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1)
    contacts = pipeline.contacts()

    # Single total soft counter (bit-identical in shape to a build without the feature).
    test.assertEqual(tuple(contacts.soft_contact_count.shape), (1,))
    test.assertFalse(contacts._enable_rigid_soft_full_surface_contact)  # flag off by default

    # Unified record fields + the particle-only view, all sized to soft_contact_max.
    test.assertEqual(contacts.soft_contact_indices.shape[0], contacts.soft_contact_max)
    test.assertEqual(contacts.soft_contact_indices.dtype, wp.vec3i)
    test.assertEqual(contacts.soft_contact_particle.shape[0], contacts.soft_contact_max)
    test.assertEqual(contacts.soft_contact_barycentric.shape[0], contacts.soft_contact_max)

    # Rigid counter untouched and independent of the soft counter.
    test.assertEqual(tuple(contacts.rigid_contact_count.shape), (1,))

    # Flag-off collide: only the particle pass runs, so every record is a particle: (p, -1, -1).
    state = model.state()
    pipeline.collide(state, contacts)
    total = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(total, 0)
    idx = contacts.soft_contact_indices.numpy()[:total]
    test.assertTrue(np.all(idx[:, 1] < 0))  # no edge/face records
    test.assertTrue(np.all(idx[:, 0] >= 0))
    # soft_contact_particle mirrors index slot 0 for particle contacts.
    test.assertTrue(np.array_equal(contacts.soft_contact_particle.numpy()[:total], idx[:, 0]))


soft_devices = get_test_devices()


class TestFullSurfaceSoftContact(unittest.TestCase):
    pass


add_function_test(
    TestFullSurfaceSoftContact,
    "test_soft_contact_schema",
    test_soft_contact_schema,
    devices=soft_devices,
)


# ---------------------------------------------------------------------------
# SDF optimizers (Macklin 2020) validated against a brute-force grid min.
# The brute-force reference samples phi on a fine grid and takes the argmin,
# so these isolate "does the optimizer find the minimum of phi".
# ---------------------------------------------------------------------------


def _box_sdf_np(point, half):
    """Reference box SDF (matches geometry.kernels.sdf_box) for brute-force comparison."""
    q = np.abs(point) - half
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))


@wp.kernel
def _edge_opt_kernel(
    geo: wp.int32,
    scale: wp.vec3,
    p: wp.vec3,
    q: wp.vec3,
    shape_sdf_index: wp.int32,
    table: wp.array[TextureSDFData],
    n_iter: wp.int32,
    out_u: wp.array[float],
    out_phi: wp.array[float],
    out_x: wp.array[wp.vec3],
):
    u, x, phi, _grad = optimize_edge_sdf(geo, scale, p, q, shape_sdf_index, table, n_iter)
    out_u[0] = u
    out_phi[0] = phi
    out_x[0] = x


@wp.kernel
def _face_opt_kernel(
    geo: wp.int32,
    scale: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_sdf_index: wp.int32,
    table: wp.array[TextureSDFData],
    n_iter: wp.int32,
    ls_iter: wp.int32,
    out_bary: wp.array[wp.vec3],
    out_phi: wp.array[float],
    out_x: wp.array[wp.vec3],
):
    bary, x, phi, _grad = optimize_face_sdf(geo, scale, a, b, c, shape_sdf_index, table, n_iter, ls_iter)
    out_bary[0] = bary
    out_phi[0] = phi
    out_x[0] = x


def _empty_sdf_table(device):
    return wp.zeros(0, dtype=TextureSDFData, device=device)


def test_optimize_edge_sdf_box(test, device):
    """Golden-section edge optimizer finds the deepest point of phi along the segment."""
    half = (0.5, 0.5, 0.5)
    p = (0.8, 0.0, 0.0)
    q = (0.0, 0.8, 0.0)
    out_u = wp.zeros(1, dtype=float, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _edge_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.BOX),
            wp.vec3(*half),
            wp.vec3(*p),
            wp.vec3(*q),
            -1,
            _empty_sdf_table(device),
            SDF_EDGE_ITERS,
        ],
        outputs=[out_u, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    pa, qa, ha = np.array(p), np.array(q), np.array(half)
    phi_brute = min(_box_sdf_np((1.0 - u) * pa + u * qa, ha) for u in np.linspace(0.0, 1.0, 20001))
    test.assertLess(abs(phi_opt - phi_brute), 1.0e-4)


def test_optimize_face_sdf_box(test, device):
    """Frank-Wolfe face optimizer finds the deepest point of phi over the triangle."""
    half = (0.5, 0.5, 0.5)
    a = (0.9, 0.0, 0.0)
    b = (0.0, 0.9, 0.0)
    c = (0.0, 0.0, 0.9)
    out_bary = wp.zeros(1, dtype=wp.vec3, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _face_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.BOX),
            wp.vec3(*half),
            wp.vec3(*a),
            wp.vec3(*b),
            wp.vec3(*c),
            -1,
            _empty_sdf_table(device),
            SDF_FACE_ITERS,
            SDF_LS_ITERS,
        ],
        outputs=[out_bary, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    aa, ba, ca, ha = np.array(a), np.array(b), np.array(c), np.array(half)
    n = 200
    best = min(
        _box_sdf_np((i / n) * aa + (j / n) * ba + (1.0 - i / n - j / n) * ca, ha)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    )
    test.assertLess(abs(phi_opt - best), 2.0e-3)


def _sphere_sdf_np(point, radius):
    """Reference sphere SDF (matches geometry.kernels.sdf_sphere)."""
    return float(np.linalg.norm(point) - radius)


def test_optimize_edge_sdf_sphere(test, device):
    """Golden-section on a smooth field finds the segment's closest approach to the sphere."""
    r = 0.5
    p = (1.0, 0.5, 0.0)
    q = (0.5, 1.0, 0.3)
    out_u = wp.zeros(1, dtype=float, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _edge_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.SPHERE),
            wp.vec3(r, r, r),
            wp.vec3(*p),
            wp.vec3(*q),
            -1,
            _empty_sdf_table(device),
            SDF_EDGE_ITERS,
        ],
        outputs=[out_u, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    pa, qa = np.array(p), np.array(q)
    phi_brute = min(_sphere_sdf_np((1.0 - u) * pa + u * qa, r) for u in np.linspace(0.0, 1.0, 20001))
    test.assertLess(abs(phi_opt - phi_brute), 1.0e-4)


def test_optimize_face_sdf_sphere(test, device):
    """Frank-Wolfe on a smooth field moves to a non-centroid optimum (asymmetric triangle)."""
    r = 0.5
    a = (1.0, 0.0, 0.2)
    b = (0.0, 1.0, 0.2)
    c = (0.3, 0.3, 1.2)
    out_bary = wp.zeros(1, dtype=wp.vec3, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _face_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.SPHERE),
            wp.vec3(r, r, r),
            wp.vec3(*a),
            wp.vec3(*b),
            wp.vec3(*c),
            -1,
            _empty_sdf_table(device),
            SDF_FACE_ITERS,
            SDF_LS_ITERS,
        ],
        outputs=[out_bary, out_phi, out_x],
        device=device,
    )
    phi_opt = float(out_phi.numpy()[0])
    aa, ba, ca = np.array(a), np.array(b), np.array(c)
    n = 200
    best = min(
        _sphere_sdf_np((i / n) * aa + (j / n) * ba + (1.0 - i / n - j / n) * ca, r)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    )
    # Face Frank-Wolfe tail at SDF_FACE_ITERS on a smooth field (~3e-3); ample for contact-within-margin.
    test.assertLess(abs(phi_opt - best), 6.0e-3)


# ---------------------------------------------------------------------------
# Edge + face pass kernels (record emission).
# ---------------------------------------------------------------------------


def test_edge_face_passes_box(test, device):
    """A cloth sheet inside a box: every unique soft edge and triangle emits exactly one record."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    # Large fixed buffer to isolate the kernels; the flag-aware default sizing is covered separately.
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1, soft_contact_max=4096)
    contacts = pipeline.contacts()
    state = model.state()
    contacts.soft_contact_count.zero_()
    edge_pairs = _build_soft_edge_rigid_contact_pairs(model)
    face_pairs = _build_soft_face_rigid_contact_pairs(model)
    # Isolated launch (no particle pass), so this pass's tids start at 0.
    launch_soft_ef_contacts(
        model=model,
        state=state,
        contacts=contacts,
        margin=0.1,
        device=device,
        edge_pairs=edge_pairs,
        face_pairs=face_pairs,
        n_particle_pairs=0,
    )

    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    # Records self-describe by -1 padding: edge (v0, v1, -1), face (v0, v1, v2).
    n_edge = int(np.sum((idx[:, 1] >= 0) & (idx[:, 2] < 0)))
    n_face = int(np.sum(idx[:, 2] >= 0))
    n_edges = model.edge_count
    # Structural dedup: the sheet is entirely inside the box, so every unique edge / triangle
    # emits exactly once (one thread per unique edge / triangle).
    test.assertEqual(n_edge, n_edges)
    test.assertEqual(n_face, model.tri_count)
    test.assertEqual(n_edge + n_face, total)  # no particle records in this isolated launch

    barys = contacts.soft_contact_barycentric.numpy()[:total]
    normals = contacts.soft_contact_normal.numpy()[:total]
    body_pos = contacts.soft_contact_body_pos.numpy()[:total]
    half = np.array([0.5, 0.5, 0.5])
    n_particles = model.particle_count

    for i in range(total):
        c = idx[i]
        n_valid = int(np.sum(c >= 0))
        test.assertIn(n_valid, (2, 3))  # edge or face
        for k in range(n_valid):
            test.assertTrue(0 <= int(c[k]) < n_particles)  # corners are soft particle ids
        test.assertAlmostEqual(float(barys[i].sum()), 1.0, places=4)
        test.assertGreater(float(normals[i][2]), 0.99)  # +z face of the box
        test.assertLess(abs(_box_sdf_np(body_pos[i], half)), 1.0e-2)  # closest point on the box surface


def test_edge_face_respect_shape_margin(test, device):
    """EDGE/FACE culls must include the per-shape margin (#2994) like the legacy particle pass:
    a sheet beyond ``soft_contact_margin`` but within ``soft_contact_margin + shape margin``
    must still emit every edge/face record."""
    margin = 0.05
    shape_margin = 0.2
    builder = newton.ModelBuilder()
    builder.default_particle_radius = 0.01
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(margin=shape_margin),
    )
    # Sheet 0.15 above the box top face: outside margin + radius (0.06), inside
    # margin + shape_margin + radius (0.26).
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.65),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=margin, soft_contact_max=4096)
    contacts = pipeline.contacts()
    state = model.state()
    contacts.soft_contact_count.zero_()
    edge_pairs = _build_soft_edge_rigid_contact_pairs(model)
    face_pairs = _build_soft_face_rigid_contact_pairs(model)
    launch_soft_ef_contacts(
        model=model,
        state=state,
        contacts=contacts,
        margin=margin,
        device=device,
        edge_pairs=edge_pairs,
        face_pairs=face_pairs,
        n_particle_pairs=0,
    )

    # Sanity: the gap really is beyond the threshold without the shape margin, so any record
    # emitted below can only come from the per-shape margin term.
    max_radius = float(model.particle_radius.numpy().max())
    test.assertGreater(0.15, margin + max_radius)

    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    n_edge = int(np.sum((idx[:, 1] >= 0) & (idx[:, 2] < 0)))
    n_face = int(np.sum(idx[:, 2] >= 0))
    test.assertEqual(n_edge, model.edge_count)
    test.assertEqual(n_face, model.tri_count)


# ---------------------------------------------------------------------------
# Dispatch flag — backward-compat (bit-for-bit) and full-surface regression.
# ---------------------------------------------------------------------------


def _sorted_particle_records(contacts, c0):
    """Particle-range records sorted by particle id (emission order is non-deterministic on GPU).

    The particle pass runs first, so the first ``c0`` records are the particle contacts.
    """
    prim = contacts.soft_contact_particle.numpy()[:c0]
    order = np.argsort(prim, kind="stable")
    return (
        prim[order],
        contacts.soft_contact_shape.numpy()[:c0][order],
        contacts.soft_contact_body_pos.numpy()[:c0][order],
        contacts.soft_contact_normal.numpy()[:c0][order],
    )


def test_backward_compat_bit_for_bit(test, device):
    """Flag on vs off (same buffer): the particle range is bit-identical; on only adds E/F records."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()

    # The flag is fixed at construction, so off vs on are two separately-sized pipelines.
    pipeline_off = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=False
    )
    contacts_off = pipeline_off.contacts()
    pipeline_off.collide(state, contacts_off)
    c0 = int(contacts_off.soft_contact_count.numpy()[0])
    test.assertGreater(c0, 0)
    # Flag off: every record is a particle contact (p, -1, -1).
    test.assertTrue(np.all(contacts_off.soft_contact_indices.numpy()[:c0][:, 1] < 0))
    prim_off, shape_off, pos_off, nrm_off = _sorted_particle_records(contacts_off, c0)

    pipeline_on = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts_on = pipeline_on.contacts()
    pipeline_on.collide(state, contacts_on)
    total_on = int(contacts_on.soft_contact_count.numpy()[0])
    idx_on = contacts_on.soft_contact_indices.numpy()[:total_on]
    n_particle_on = int(np.sum(idx_on[:, 1] < 0))
    test.assertEqual(n_particle_on, c0)  # particle-contact count unchanged; E/F only added
    prim_on, shape_on, pos_on, nrm_on = _sorted_particle_records(contacts_on, c0)

    # Bit-identical particle range (same legacy kernel, same inputs; particle records come first).
    test.assertTrue(np.array_equal(prim_on, prim_off))
    test.assertTrue(np.array_equal(shape_on, shape_off))
    test.assertTrue(np.array_equal(pos_on, pos_off))
    test.assertTrue(np.array_equal(nrm_on, nrm_off))
    # Flag on only ADDS edge/face records.
    test.assertGreater(total_on - n_particle_on, 0)


def test_full_surface_catches_what_particles_miss(test, device):
    """A soft quad spanning a box with all corners outside margin: per-particle misses, E/F catches."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    # 1x1 cloth = one quad (2 tris); corners at (+-1, +-1, 0.45) are far outside the box margin,
    # but the quad's interior/diagonal cross the box's +z face within margin.
    builder.add_cloth_grid(
        pos=wp.vec3(-1.0, -1.0, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=2.0,
        cell_y=2.0,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()

    # Per-particle path alone (flag off at construction): every corner is outside margin -> no contact.
    pipeline_off = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=False
    )
    contacts_off = pipeline_off.contacts()
    pipeline_off.collide(state, contacts_off)
    test.assertEqual(int(contacts_off.soft_contact_count.numpy()[0]), 0)

    # Full-surface path (flag on): the edge/face passes detect the crossing the particles miss.
    pipeline_on = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts_on = pipeline_on.contacts()
    pipeline_on.collide(state, contacts_on)
    total = int(contacts_on.soft_contact_count.numpy()[0])
    idx = contacts_on.soft_contact_indices.numpy()[:total]
    test.assertEqual(int(np.sum(idx[:, 1] < 0)), 0)  # still no per-particle contact
    test.assertGreater(total, 0)  # caught by edge/face


for _name, _fn in (
    ("test_optimize_edge_sdf_box", test_optimize_edge_sdf_box),
    ("test_optimize_face_sdf_box", test_optimize_face_sdf_box),
    ("test_optimize_edge_sdf_sphere", test_optimize_edge_sdf_sphere),
    ("test_optimize_face_sdf_sphere", test_optimize_face_sdf_sphere),
    ("test_edge_face_passes_box", test_edge_face_passes_box),
    ("test_edge_face_respect_shape_margin", test_edge_face_respect_shape_margin),
    ("test_backward_compat_bit_for_bit", test_backward_compat_bit_for_bit),
    ("test_full_surface_catches_what_particles_miss", test_full_surface_catches_what_particles_miss),
):
    add_function_test(TestFullSurfaceSoftContact, _name, _fn, devices=soft_devices)


# ---------------------------------------------------------------------------
# Mesh volume-SDF provisioning at finalize. Texture SDFs are CUDA-only.
# ---------------------------------------------------------------------------


def test_mesh_sdf_provisioned_and_emits(test, device):
    """A participating MESH shape gets a volume SDF baked at finalize and emits EDGE/FACE records."""
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    builder = newton.ModelBuilder()
    mesh_shape = builder.add_shape_mesh(body=-1, mesh=box_mesh)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    # The participating mesh now carries a provisioned volume SDF.
    test.assertGreaterEqual(int(model._shape_sdf_index.numpy()[mesh_shape]), 0)

    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    # The mesh's volume SDF feeds the edge/face passes -> edge/face records emitted.
    test.assertGreater(int(np.sum(idx[:, 1] >= 0)), 0)


def test_force_sdf_provisions_collision_meshes(test, device):
    """ShapeConfig.configure_sdf(force_sdf=True) marks a mesh/convex COLLIDE_PARTICLES shape for SDF
    construction; analytic primitives are skipped (builder-level; no SDF is built here)."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.configure_sdf(force_sdf=True)
    m0 = builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.5), cfg=cfg)
    m1 = builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.5))  # default cfg, no force_sdf
    box = builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)  # analytic: never provisioned

    test.assertTrue(builder.shape_force_sdf[m0])
    test.assertFalse(builder.shape_force_sdf[m1])
    # force_sdf on an analytic shape is captured but harmless: finalize only builds mesh/convex SDFs.
    test.assertTrue(builder.shape_force_sdf[box])

    # configure_sdf still rejects both resolution knobs at once.
    with test.assertRaises(ValueError):
        newton.ModelBuilder.ShapeConfig().configure_sdf(max_resolution=64, target_voxel_size=0.01)


add_function_test(
    TestFullSurfaceSoftContact,
    "test_force_sdf_provisions_collision_meshes",
    test_force_sdf_provisions_collision_meshes,
    devices=soft_devices,
)


def test_optimize_against_mesh_texture_sdf(test, device):
    """optimize_edge/face_sdf against a MESH's provisioned texture SDF match the box it represents.

    Validates the volume-SDF branch of eval_shape_sdf (texture sampling + query-time scaling) end to
    end through the optimizers, to within the texture grid's resolution.
    """
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=box_mesh)
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    sdf_idx = int(model._shape_sdf_index.numpy()[0])
    test.assertGreaterEqual(sdf_idx, 0)
    table = model._texture_sdf_data
    scale = wp.vec3(*(float(s) for s in model.shape_scale.numpy()[0]))
    half = np.array([0.5, 0.5, 0.5])
    tol = 3.0e-2  # texture SDF grid resolution (default 64^3 over a unit box) + optimizer tail

    # Edge: from just inside the +z face to outside; the minimum is the inside endpoint.
    p, q = (0.0, 0.0, 0.45), (0.0, 0.0, 0.65)
    out_u = wp.zeros(1, dtype=float, device=device)
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_x = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _edge_opt_kernel,
        dim=1,
        inputs=[int(GeoType.MESH), scale, wp.vec3(*p), wp.vec3(*q), sdf_idx, table, SDF_EDGE_ITERS],
        outputs=[out_u, out_phi, out_x],
        device=device,
    )
    pa, qa = np.array(p), np.array(q)
    phi_ref_edge = min(_box_sdf_np((1.0 - u) * pa + u * qa, half) for u in np.linspace(0.0, 1.0, 4001))
    test.assertLess(abs(float(out_phi.numpy()[0]) - phi_ref_edge), tol)

    # Face: a small triangle grazing the +z face.
    a, b, c = (0.2, 0.0, 0.45), (-0.2, 0.1, 0.45), (0.0, -0.2, 0.45)
    out_bary = wp.zeros(1, dtype=wp.vec3, device=device)
    out_phi2 = wp.zeros(1, dtype=float, device=device)
    out_x2 = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _face_opt_kernel,
        dim=1,
        inputs=[
            int(GeoType.MESH),
            scale,
            wp.vec3(*a),
            wp.vec3(*b),
            wp.vec3(*c),
            sdf_idx,
            table,
            SDF_FACE_ITERS,
            SDF_LS_ITERS,
        ],
        outputs=[out_bary, out_phi2, out_x2],
        device=device,
    )
    aa, ba, ca = np.array(a), np.array(b), np.array(c)
    n = 80
    phi_ref_face = min(
        _box_sdf_np((i / n) * aa + (j / n) * ba + (1.0 - i / n - j / n) * ca, half)
        for i in range(n + 1)
        for j in range(n + 1 - i)
    )
    test.assertLess(abs(float(out_phi2.numpy()[0]) - phi_ref_face), tol)


for _name, _fn in (
    ("test_mesh_sdf_provisioned_and_emits", test_mesh_sdf_provisioned_and_emits),
    ("test_optimize_against_mesh_texture_sdf", test_optimize_against_mesh_texture_sdf),
):
    add_function_test(TestFullSurfaceSoftContact, _name, _fn, devices=get_cuda_test_devices())


@wp.kernel
def _eval_shape_sdf_kernel(
    geo: wp.int32,
    scale: wp.vec3,
    x: wp.vec3,
    sdf_idx: wp.int32,
    table: wp.array[TextureSDFData],
    out_phi: wp.array[float],
    out_grad: wp.array[wp.vec3],
):
    _phi_l, phi, grad = eval_shape_sdf(geo, scale, x, sdf_idx, table)
    out_phi[0] = phi
    out_grad[0] = grad


def _make_box_mesh_sdf_model(device):
    """A single box MESH with a provisioned (unscaled) volume SDF, for eval_shape_sdf tests."""
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.5))
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    return model, int(model._shape_sdf_index.numpy()[0])


def test_eval_shape_sdf_mirrored_mesh_scale_preserves_sign(test, device):
    """A mirrored (negative) mesh scale must not flip the SDF sign (E3). wp.min(scale) would go
    negative and invert an outside distance; wp.min(wp.abs(scale)) keeps the magnitude positive."""
    model, sdf_idx = _make_box_mesh_sdf_model(device)
    test.assertGreaterEqual(sdf_idx, 0)
    table = model._texture_sdf_data
    x_out = wp.vec3(1.0, 0.0, 0.0)  # clearly outside the |x| <= 0.5 box
    out_phi = wp.zeros(1, dtype=float, device=device)
    out_grad = wp.zeros(1, dtype=wp.vec3, device=device)

    def _sample(scl):
        wp.launch(
            _eval_shape_sdf_kernel,
            dim=1,
            inputs=[int(GeoType.MESH), scl, x_out, sdf_idx, table],
            outputs=[out_phi, out_grad],
            device=device,
        )
        return float(out_phi.numpy()[0]), out_grad.numpy()[0].copy()

    phi_id, grad_id = _sample(wp.vec3(1.0, 1.0, 1.0))
    phi_mir, grad_mir = _sample(wp.vec3(-1.0, 1.0, 1.0))

    test.assertGreater(phi_id, 0.0, "identity-scale SDF must be positive outside the box")
    test.assertGreater(phi_mir, 0.0, "mirrored mesh scale must not flip the SDF sign")
    test.assertLess(abs(phi_id - phi_mir), 3.0e-2, "mirror of a symmetric box must not change |phi|")
    test.assertGreater(float(grad_id[0]), 0.0, "gradient must point outward (+x)")
    test.assertGreater(float(grad_mir[0]), 0.0, "mirrored gradient must still point outward (+x)")


def test_full_surface_empty_sdf_descriptor_rejected(test, device):
    """A participating mesh whose shape_sdf_index points at an empty placeholder descriptor (coarse
    texture None, e.g. a mesh-mesh BVH fallback) is rejected by the full-surface guard rather than
    sampled -- sampling one reproduced CUDA error 700 (E1)."""
    model, sdf_idx = _make_box_mesh_sdf_model(device)
    test.assertGreaterEqual(sdf_idx, 0)
    # Simulate an empty placeholder descriptor at that slot: a nonnegative index whose descriptor
    # carries no texture (coarse texture None), exactly what a BVH fallback appends.
    model._texture_sdf_coarse_textures[sdf_idx] = None
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(model, broad_phase="nxn", enable_rigid_soft_full_surface_contact=True)


def _add_soft_triangle(builder, z=1.0):
    p0 = builder.add_particle(wp.vec3(-0.2, -0.2, z), wp.vec3(0.0), 0.1, radius=0.0)
    p1 = builder.add_particle(wp.vec3(0.2, -0.2, z), wp.vec3(0.0), 0.1, radius=0.0)
    p2 = builder.add_particle(wp.vec3(0.0, 0.2, z), wp.vec3(0.0), 0.1, radius=0.0)
    builder.add_triangle(p0, p1, p2)


def test_soft_contact_tids_decoupled_from_capacity(test, device):
    """soft_contact_tids is sized independently of soft_contact_max so a small custom capacity cannot
    drop a launch thread's replay slot (E2, unit)."""
    from newton._src.sim.contacts import Contacts  # noqa: PLC0415

    c = Contacts(rigid_contact_max=4, soft_contact_max=1, soft_contact_tids_size=37, device=device)
    test.assertEqual(c.soft_contact_tids.shape[0], 37)
    c_default = Contacts(rigid_contact_max=4, soft_contact_max=9, device=device)
    test.assertEqual(c_default.soft_contact_tids.shape[0], 9)


def test_full_surface_replay_spans_candidate_space(test, device):
    """The pipeline sizes soft_contact_tids to the full particle+edge+face candidate space even when
    soft_contact_max is overridden smaller, so differentiable backward never loses a thread (E2)."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    _add_soft_triangle(builder)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", enable_rigid_soft_full_surface_contact=True, soft_contact_max=1
    )
    contacts = pipeline.contacts()
    candidate = (
        pipeline.soft_rigid_contact_pair_count
        + len(pipeline.soft_edge_rigid_pairs)
        + len(pipeline.soft_face_rigid_pairs)
    )
    test.assertGreater(candidate, 1, "test needs a candidate space larger than the capacity override")
    test.assertEqual(contacts.soft_contact_max, 1, "explicit soft_contact_max capacity must be honored")
    test.assertEqual(contacts.soft_contact_tids.shape[0], candidate, "replay array must span the full candidate space")


def test_collide_syncs_full_surface_marker(test, device):
    """collide() sets the buffer's full-surface capability marker on every call so a buffer created
    elsewhere (or by a flag-off pipeline) cannot silently misroute edge/face records (E6)."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    _add_soft_triangle(builder, z=0.5)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    # Simulate a buffer whose marker was left False (e.g. constructed by a flag-off pipeline).
    contacts._enable_rigid_soft_full_surface_contact = False
    pipeline.collide(model.state(), contacts)
    test.assertTrue(
        contacts._enable_rigid_soft_full_surface_contact,
        "collide() must re-sync the full-surface marker so particle-only solvers can refuse the buffer",
    )


def test_full_surface_finite_plane_falls_back(test, device):
    """A finite plane can't do edge/face (its +Z normal is wrong off the quad), so it warns and falls
    back to per-particle soft contact instead of failing the pipeline: construction succeeds, the
    plane is excluded from the edge/face candidate pairs, and a capable box still keeps them (E4)."""
    builder = newton.ModelBuilder()
    box = builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    plane = builder.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=5.0, length=5.0)  # finite
    _add_soft_triangle(builder)
    model = builder.finalize(device=device)
    with test.assertWarns(UserWarning):
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", enable_rigid_soft_full_surface_contact=True)
    face_shapes = (
        {int(s) for s in pipeline.soft_face_rigid_pairs.numpy()[:, 1]} if len(pipeline.soft_face_rigid_pairs) else set()
    )
    test.assertIn(box, face_shapes, "the capable box keeps its full-surface face pairs")
    test.assertNotIn(plane, face_shapes, "the finite plane is excluded from full-surface (fell back)")


def test_full_surface_heightfield_falls_back(test, device):
    """A heightfield exposes only a per-cell local-plane distance (discontinuous across cells),
    unsuitable for the edge/face SDF optimizers, so it warns and falls back to per-particle soft
    contact rather than failing the pipeline; a capable box keeps full-surface (E4)."""
    builder = newton.ModelBuilder()
    box = builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    hf = builder.add_shape_heightfield(
        heightfield=newton.Heightfield(
            data=np.zeros((3, 3), dtype=np.float32), nrow=3, ncol=3, hx=1.0, hy=1.0, min_z=0.0, max_z=0.0
        )
    )
    _add_soft_triangle(builder)
    model = builder.finalize(device=device)
    with test.assertWarns(UserWarning):
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", enable_rigid_soft_full_surface_contact=True)
    face_shapes = (
        {int(s) for s in pipeline.soft_face_rigid_pairs.numpy()[:, 1]} if len(pipeline.soft_face_rigid_pairs) else set()
    )
    test.assertIn(box, face_shapes, "the capable box keeps its full-surface face pairs")
    test.assertNotIn(hf, face_shapes, "the heightfield is excluded from full-surface (fell back)")


def test_full_surface_allows_infinite_plane(test, device):
    """An infinite plane (width=length=0) is supported by full-surface (+Z normal is correct
    everywhere), so the common ground-plane case keeps working (E4 regression guard)."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()  # infinite
    _add_soft_triangle(builder)
    model = builder.finalize(device=device)
    # Must not raise.
    newton.CollisionPipeline(model, broad_phase="nxn", enable_rigid_soft_full_surface_contact=True)


def _nonuniform_box_mesh_gap_model(device, tri_x):
    """Box MESH scaled (2, 1, 1) at the origin -> its +x face sits at body x = 0.5 * 2 = 1.0. A soft
    triangle parallel to that face at x = ``tri_x`` (within the face's y/z extent) has a uniform gap of
    ``tri_x - 1.0``. Used to probe the nonuniform-scale distance (E8)."""
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=newton.Mesh.create_box(0.5, 0.5, 0.5), scale=(2.0, 1.0, 1.0))
    configure_sdf_for_collision_shapes(builder)
    p0 = builder.add_particle(wp.vec3(tri_x, -0.2, -0.2), wp.vec3(0.0), 0.1, radius=0.0)
    p1 = builder.add_particle(wp.vec3(tri_x, 0.2, -0.2), wp.vec3(0.0), 0.1, radius=0.0)
    p2 = builder.add_particle(wp.vec3(tri_x, 0.0, 0.2), wp.vec3(0.0), 0.1, radius=0.0)
    builder.add_triangle(p0, p1, p2)
    return builder.finalize(device=device)


def test_full_surface_nonuniform_mesh_accurate_distance(test, device):
    """Under nonuniform mesh scale the volume-SDF distance stretches along the surface normal, not by
    the smallest scale factor -- so full-surface keeps working (no fallback) and the distance is right:
    a soft triangle 0.08 m outside a (2,1,1) box's +x face (0.06 m margin) yields NO ghost contact, and
    one 0.03 m inside is caught and projected onto the true surface x=1.0, not past it. min_scale would
    report 0.04 (ghost) and project to ~1.015 (E8)."""
    # 0.08 m gap, 0.06 m margin -> outside -> no contact. min_scale would under-report 0.04 < 0.06.
    model_out = _nonuniform_box_mesh_gap_model(device, tri_x=1.08)
    pipe_out = newton.CollisionPipeline(
        model_out, broad_phase="nxn", soft_contact_margin=0.06, enable_rigid_soft_full_surface_contact=True
    )
    contacts_out = pipe_out.contacts()
    pipe_out.collide(model_out.state(), contacts_out)
    test.assertEqual(
        int(contacts_out.soft_contact_count.numpy()[0]), 0, "no ghost contact 0.08 m outside a 0.06 m margin"
    )

    # 0.03 m gap -> inside the margin -> contact, projected onto the true +x surface at x = 1.0.
    model_in = _nonuniform_box_mesh_gap_model(device, tri_x=1.03)
    pipe_in = newton.CollisionPipeline(
        model_in, broad_phase="nxn", soft_contact_margin=0.06, enable_rigid_soft_full_surface_contact=True
    )
    contacts_in = pipe_in.contacts()
    pipe_in.collide(model_in.state(), contacts_in)
    n_in = int(contacts_in.soft_contact_count.numpy()[0])
    test.assertGreater(
        n_in, 0, "a 0.03 m gap is within the 0.06 m margin -> full-surface still active for nonuniform scale"
    )
    body_pos_x = contacts_in.soft_contact_body_pos.numpy()[:n_in, 0]
    test.assertTrue(
        bool(np.all(np.abs(body_pos_x - 1.0) < 5e-3)),
        f"contact projects onto the true surface x=1.0 (min_scale would land ~1.015), got {body_pos_x}",
    )


for _name, _fn in (
    ("test_soft_contact_tids_decoupled_from_capacity", test_soft_contact_tids_decoupled_from_capacity),
    ("test_full_surface_replay_spans_candidate_space", test_full_surface_replay_spans_candidate_space),
    ("test_collide_syncs_full_surface_marker", test_collide_syncs_full_surface_marker),
    ("test_full_surface_finite_plane_falls_back", test_full_surface_finite_plane_falls_back),
    ("test_full_surface_heightfield_falls_back", test_full_surface_heightfield_falls_back),
    ("test_full_surface_allows_infinite_plane", test_full_surface_allows_infinite_plane),
):
    add_function_test(TestFullSurfaceSoftContact, _name, _fn, devices=soft_devices)

for _name, _fn in (
    ("test_eval_shape_sdf_mirrored_mesh_scale_preserves_sign", test_eval_shape_sdf_mirrored_mesh_scale_preserves_sign),
    ("test_full_surface_empty_sdf_descriptor_rejected", test_full_surface_empty_sdf_descriptor_rejected),
    ("test_full_surface_nonuniform_mesh_accurate_distance", test_full_surface_nonuniform_mesh_accurate_distance),
):
    add_function_test(TestFullSurfaceSoftContact, _name, _fn, devices=get_cuda_test_devices())


def test_unprovisioned_mesh_raises(test, device):
    """A participating mesh with no SDF makes CollisionPipeline raise when the flag is enabled.

    Mirrors SolverVBD raising on an uncolored model: provisioning an SDF (e.g. via
    ShapeConfig.configure_sdf(force_sdf=True)) is a required build step, and skipping it is an error
    rather than a silent degrade to the per-particle path.
    """
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=box_mesh)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    # SDF provisioning intentionally skipped -> the mesh carries no SDF.
    model = builder.finalize(device=device)
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(
            model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=True
        )


add_function_test(
    TestFullSurfaceSoftContact,
    "test_unprovisioned_mesh_raises",
    test_unprovisioned_mesh_raises,
    devices=soft_devices,
)


# ---------------------------------------------------------------------------
# End-to-end: all shape types + random soft triangles, full-surface on, no
# false positives / false negatives vs a brute-force grid min of the same
# eval_shape_sdf. (For analytic shapes the optimizer evaluates phi on the
# feature, so phi* >= true min => false positives are structurally impossible;
# this guards false negatives and the dispatch/record matching too.)
# ---------------------------------------------------------------------------


@wp.kernel
def _brute_face_min_kernel(
    n_tris: wp.int32,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    n_grid: wp.int32,
    out_min: wp.array[float],
):
    tid = wp.tid()
    shape_index = tid // n_tris
    t = tid % n_tris
    out_min[tid] = 1.0e10
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape_index]
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        return
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    a = wp.transform_point(X_sw, particle_q[tri_indices[t, 0]])
    b = wp.transform_point(X_sw, particle_q[tri_indices[t, 1]])
    c = wp.transform_point(X_sw, particle_q[tri_indices[t, 2]])
    scale = shape_scale[shape_index]
    m = float(1.0e10)
    for k in range((n_grid + 1) * (n_grid + 1)):
        i = k // (n_grid + 1)
        j = k % (n_grid + 1)
        if i + j <= n_grid:
            u = float(i) / float(n_grid)
            v = float(j) / float(n_grid)
            _phi_l, phi, _g = eval_shape_sdf(geo, scale, u * a + v * b + (1.0 - u - v) * c, sdf_idx, texture_sdf_table)
            m = wp.min(m, phi)
    out_min[tid] = m


@wp.kernel
def _brute_edge_min_kernel(
    n_edges: wp.int32,
    particle_q: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    n_grid: wp.int32,
    out_min: wp.array[float],
):
    tid = wp.tid()
    shape_index = tid // n_edges
    e = tid % n_edges
    out_min[tid] = 1.0e10
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape_index]
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        return
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    p = wp.transform_point(X_sw, particle_q[edge_indices[e, 2]])
    q = wp.transform_point(X_sw, particle_q[edge_indices[e, 3]])
    scale = shape_scale[shape_index]
    m = float(1.0e10)
    for i in range(n_grid + 1):
        u = float(i) / float(n_grid)
        _phi_l, phi, _g = eval_shape_sdf(geo, scale, (1.0 - u) * p + u * q, sdf_idx, texture_sdf_table)
        m = wp.min(m, phi)
    out_min[tid] = m


def _build_all_shapes_scene(device, rng):
    """Ground plane + the six analytic primitives, with random soft triangles seeded near each."""
    z = 1.0
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    box_mesh = newton.Mesh.create_box(0.5, 0.5, 0.5)
    primitives = [
        (
            lambda: builder.add_shape_sphere(
                body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), radius=0.5
            ),
            (0.0, 0.0, z),
            0.5,
        ),
        (
            lambda: builder.add_shape_box(
                body=-1, xform=wp.transform(wp.vec3(2.0, 0.0, z), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
            ),
            (2.0, 0.0, z),
            0.6,
        ),
        (
            lambda: builder.add_shape_capsule(
                body=-1, xform=wp.transform(wp.vec3(4.0, 0.0, z), wp.quat_identity()), radius=0.4, half_height=0.4
            ),
            (4.0, 0.0, z),
            0.55,
        ),
        (
            lambda: builder.add_shape_cylinder(
                body=-1, xform=wp.transform(wp.vec3(6.0, 0.0, z), wp.quat_identity()), radius=0.5, half_height=0.4
            ),
            (6.0, 0.0, z),
            0.6,
        ),
        (
            lambda: builder.add_shape_cone(
                body=-1, xform=wp.transform(wp.vec3(8.0, 0.0, z), wp.quat_identity()), radius=0.5, half_height=0.5
            ),
            (8.0, 0.0, z),
            0.6,
        ),
        (
            lambda: builder.add_shape_ellipsoid(
                body=-1, xform=wp.transform(wp.vec3(10.0, 0.0, z), wp.quat_identity()), rx=0.5, ry=0.4, rz=0.6
            ),
            (10.0, 0.0, z),
            0.6,
        ),
        # MESH (a box-shaped triangle mesh): on CUDA its texture SDF is provisioned at finalize and
        # validated; on CPU texture SDFs are unavailable so the passes and the brute-force reference
        # both gate it out identically.
        (
            lambda: builder.add_shape_mesh(
                body=-1, xform=wp.transform(wp.vec3(12.0, 0.0, z), wp.quat_identity()), mesh=box_mesh
            ),
            (12.0, 0.0, z),
            0.6,
        ),
    ]
    centers, sizes = [], []
    for add, center, size in primitives:
        add()
        centers.append(np.array(center))
        sizes.append(size)

    verts, indices = [], []

    def add_tri(centroid):
        base = len(verts)
        for _ in range(3):
            off = rng.normal(0.0, 0.12, 3)
            verts.append(wp.vec3(float(centroid[0] + off[0]), float(centroid[1] + off[1]), float(centroid[2] + off[2])))
        indices.extend([base, base + 1, base + 2])

    for center, size in zip(centers, sizes, strict=True):
        for _ in range(5):
            d = rng.normal(0.0, 1.0, 3)
            d = d / np.linalg.norm(d)
            add_tri(center + d * (size + rng.uniform(-0.12, 0.18)))
    for _ in range(6):  # near the ground plane (z ~ 0)
        add_tri(np.array([rng.uniform(0.0, 10.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.05, 0.22)]))

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=verts,
        indices=indices,
        density=0.1,
        particle_radius=0.0,  # so the pass threshold is exactly `margin` (matches the brute-force check)
    )
    configure_sdf_for_collision_shapes(builder)
    return builder.finalize(device=device)


def test_end_to_end_no_false_pos_neg(test, device):
    """All shapes + random triangles: full-surface emissions match a brute-force grid min (no FP/FN)."""
    margin = 0.1
    model = _build_all_shapes_scene(device, np.random.default_rng(0))
    n_tris = model.tri_count
    n_edges = model.soft_mesh_adjacency.edge_indices.shape[0]
    n_shapes = model.shape_count

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=margin,
        soft_contact_max=n_shapes * (n_tris + n_edges) + 16,
        enable_rigid_soft_full_surface_contact=True,
    )
    contacts = pipeline.contacts()
    state = model.state()
    contacts.soft_contact_count.zero_()
    edge_pairs = _build_soft_edge_rigid_contact_pairs(model)
    face_pairs = _build_soft_face_rigid_contact_pairs(model)
    launch_soft_ef_contacts(
        model=model,
        state=state,
        contacts=contacts,
        margin=margin,
        device=device,
        edge_pairs=edge_pairs,
        face_pairs=face_pairs,
        n_particle_pairs=0,
    )

    total = int(contacts.soft_contact_count.numpy()[0])
    rec_idx = contacts.soft_contact_indices.numpy()[:total]
    n_edge_rec = int(np.sum((rec_idx[:, 1] >= 0) & (rec_idx[:, 2] < 0)))
    n_face_rec = int(np.sum(rec_idx[:, 2] >= 0))
    test.assertGreater(n_edge_rec + n_face_rec, 0)  # the scene actually generates contacts

    # Brute-force ground truth: min phi per (shape, feature) using the same eval_shape_sdf.
    face_min = wp.empty(n_shapes * n_tris, dtype=float, device=device)
    edge_min = wp.empty(n_shapes * n_edges, dtype=float, device=device)
    shape_args = [
        model.shape_body,
        model.shape_type,
        model.shape_flags,
        model.shape_transform,
        model.shape_scale,
        state.body_q,
        model._shape_sdf_index,
        model._texture_sdf_data,
    ]
    # MeshAdjacency.edge_indices is host numpy; upload for the brute-force kernel.
    edge_indices_dev = wp.array(
        np.ascontiguousarray(model.soft_mesh_adjacency.edge_indices, dtype=np.int32), dtype=wp.int32, device=device
    )
    wp.launch(
        _brute_face_min_kernel,
        dim=n_shapes * n_tris,
        inputs=[n_tris, state.particle_q, model.tri_indices, *shape_args, 40],
        outputs=[face_min],
        device=device,
    )
    wp.launch(
        _brute_edge_min_kernel,
        dim=n_shapes * n_edges,
        inputs=[n_edges, state.particle_q, edge_indices_dev, *shape_args, 200],
        outputs=[edge_min],
        device=device,
    )
    face_min = face_min.numpy().reshape(n_shapes, n_tris)
    edge_min = edge_min.numpy().reshape(n_shapes, n_edges)

    # Emitted records self-describe via -1 padding; map each back to its (shape, feature) by matching
    # its corner set to the mesh's triangles / edges (records store particle ids, not a feature id).
    rec_shape = contacts.soft_contact_shape.numpy()[:total]
    tri_np = model.tri_indices.numpy()
    tri_by_corners = {frozenset(int(x) for x in tri_np[t]): t for t in range(n_tris)}
    adj_edges = np.asarray(model.soft_mesh_adjacency.edge_indices)  # matches edge_min's indexing
    edge_by_endpoints = {frozenset((int(adj_edges[e, 2]), int(adj_edges[e, 3]))): e for e in range(n_edges)}
    edge_owner = np.asarray(model.soft_mesh_adjacency.edge_tri_indices)[:, 0]

    emitted_faces = set()
    emitted_edge_owner = Counter()
    for i in range(total):
        c = rec_idx[i]
        if int(c[2]) >= 0:  # face record (v0, v1, v2)
            t = tri_by_corners.get(frozenset(int(x) for x in c))
            if t is not None:
                emitted_faces.add((int(rec_shape[i]), t))
        elif int(c[1]) >= 0:  # edge record (v0, v1, -1)
            e = edge_by_endpoints.get(frozenset((int(c[0]), int(c[1]))))
            if e is not None:
                emitted_edge_owner[(int(rec_shape[i]), int(edge_owner[e]))] += 1

    delta = 0.03  # margin band: optimizer tail + brute grid step; borderline cases are not asserted

    # Faces match exactly on (shape, tri).
    for s in range(n_shapes):
        for t in range(n_tris):
            if face_min[s, t] < margin - delta:
                test.assertIn(
                    (s, t), emitted_faces, f"false negative: face (shape {s}, tri {t}) phi={face_min[s, t]:.4f}"
                )
    for s, t in emitted_faces:
        test.assertLess(face_min[s, t], margin + delta, f"false positive: face (shape {s}, tri {t})")

    # Edges: one record per near owned edge, but the bary degenerates to a vertex when the contact is
    # at an endpoint, so match by owner triangle + count. For each (shape, owner-tri) the number of
    # emitted edge records must lie within [#edges clearly inside, #edges possibly inside].
    for s in range(n_shapes):
        for t in range(n_tris):
            near_lo = sum(1 for e in range(n_edges) if edge_owner[e] == t and edge_min[s, e] < margin - delta)
            near_hi = sum(1 for e in range(n_edges) if edge_owner[e] == t and edge_min[s, e] < margin + delta)
            got = emitted_edge_owner[(s, t)]
            test.assertGreaterEqual(got, near_lo, f"false negative: edges of (shape {s}, tri {t}): {got} < {near_lo}")
            test.assertLessEqual(got, near_hi, f"false positive: edges of (shape {s}, tri {t}): {got} > {near_hi}")


add_function_test(
    TestFullSurfaceSoftContact,
    "test_end_to_end_no_false_pos_neg",
    test_end_to_end_no_false_pos_neg,
    devices=soft_devices,
    check_output=False,  # CPU emits a benign warning when the mesh's texture SDF cannot be provisioned
)


def test_graph_capture_stable(test, device):
    """A flag-on collide is CUDA-graph-capturable and replays to identical soft-contact counts."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.4, -0.4, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.2,
        cell_y=0.2,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()

    # Warm up so all kernels are compiled before capture.
    pipeline.collide(state, contacts)
    counts0 = contacts.soft_contact_count.numpy().copy()
    total0 = int(counts0[0])
    test.assertGreater(int(np.sum(contacts.soft_contact_indices.numpy()[:total0][:, 1] >= 0)), 0)

    # Capture the flag-on collide and replay it; counts must be stable across replays.
    with wp.ScopedCapture(device) as capture:
        pipeline.collide(state, contacts)
    for _ in range(3):
        wp.capture_launch(capture.graph)
        test.assertTrue(np.array_equal(contacts.soft_contact_count.numpy(), counts0))


add_function_test(
    TestFullSurfaceSoftContact,
    "test_graph_capture_stable",
    test_graph_capture_stable,
    devices=get_cuda_test_devices(),
)


def test_face_cull_uses_max_vertex_reach(test, device):
    """Regression: the FACE cull reach must be the max centroid-to-vertex distance, not circumradius.

    A deliberately non-equilateral triangle whose near vertex is also the one *farthest* from the
    centroid, so circumradius (~0.124) is smaller than the true reach (~0.163). The near vertex sits
    inside the sphere's contact margin (phi ~= 0.005 < 0.01), so a real FACE contact exists -- but the
    centroid SDF (~0.168) exceeds ``margin + circumradius`` (~0.134), so the old circumradius cull
    dropped the whole triangle. The correct reach keeps it (``margin + reach`` ~= 0.173 > 0.168).
    A sphere gives an unambiguous radial SDF, so the culled point is genuinely within margin.
    """
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.1)
    # Near vertex b0 (just outside the sphere along +x) is farthest from the centroid; the a0/c0
    # cluster sits far out along +x, making the triangle strongly non-equilateral.
    b0 = builder.add_particle(wp.vec3(0.105, 0.0, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    a0 = builder.add_particle(wp.vec3(0.35, 0.03, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    c0 = builder.add_particle(wp.vec3(0.35, -0.03, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    builder.add_triangle(b0, a0, c0)

    builder.color()
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.01, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state = model.state()

    pipeline.collide(state, contacts)
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    n_face = int(np.sum(idx[:, 2] >= 0))  # face records are (v0, v1, v2)
    test.assertGreater(
        n_face, 0, "FACE contact wrongly culled: the cull reach must be the max centroid-to-vertex distance"
    )


add_function_test(
    TestFullSurfaceSoftContact,
    "test_face_cull_uses_max_vertex_reach",
    test_face_cull_uses_max_vertex_reach,
    devices=soft_devices,
)


def test_edge_face_pairs_respect_worlds(test, device):
    """Multi-world: the full-surface edge/face candidate pairs never cross worlds.

    Two worlds, each a box + a triangle. The edge/face pair builders must emit exactly the
    world-compatible (feature, shape) pairs (same world, or either global -1) -- matching a
    brute-force reference -- and must strictly exclude the cross-world combinations, mirroring the
    particle path's ``_build_soft_particle_rigid_contact_pairs``.
    """

    def _sub():
        # A cloth grid (not a bare triangle) so finalize builds soft-mesh edge adjacency.
        b = newton.ModelBuilder()
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5)
        b.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.1,
        )
        return b

    builder = newton.ModelBuilder()
    builder.add_world(_sub())
    builder.add_world(_sub())
    model = builder.finalize(device=device)

    edge_pairs = _build_soft_edge_rigid_contact_pairs(model)
    face_pairs = _build_soft_face_rigid_contact_pairs(model)
    pw = model.particle_world.numpy()
    sw = model.shape_world.numpy()
    tri = model.tri_indices.numpy()
    owner = np.asarray(model.soft_mesh_adjacency.edge_tri_indices)[:, 0]
    n_shapes = int(model.shape_count)
    n_tris = int(model.tri_count)
    n_edges = int(model.soft_mesh_adjacency.edge_indices.shape[0])

    # The setup must actually span multiple worlds, else there is nothing to isolate.
    test.assertGreaterEqual(len(set(pw.tolist())), 2)

    def _compat(feature_world, s):
        return feature_world == sw[s] or feature_world < 0 or sw[s] < 0

    face_world = pw[tri[:, 0]]
    expected_face = {(t, s) for t in range(n_tris) for s in range(n_shapes) if _compat(face_world[t], s)}
    test.assertEqual({tuple(int(v) for v in p) for p in face_pairs.numpy()}, expected_face)

    edge_world = pw[tri[owner, 0]]
    expected_edge = {(e, s) for e in range(n_edges) for s in range(n_shapes) if _compat(edge_world[e], s)}
    test.assertEqual({tuple(int(v) for v in p) for p in edge_pairs.numpy()}, expected_edge)

    # Filtering must drop the cross-world combinations (fewer than the naive full cross product).
    test.assertLess(len(face_pairs), n_tris * n_shapes)
    test.assertLess(len(edge_pairs), n_edges * n_shapes)


add_function_test(
    TestFullSurfaceSoftContact,
    "test_edge_face_pairs_respect_worlds",
    test_edge_face_pairs_respect_worlds,
    devices=soft_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
