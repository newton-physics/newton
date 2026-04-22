# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from enum import IntFlag, auto

import numpy as np
import warp as wp
import warp.examples

import newton
from newton import GeoType
from newton._src.sim.collide import _compute_per_world_shape_pairs_max, _estimate_rigid_contact_max
from newton.examples import test_body_state
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


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

        self.builder = newton.ModelBuilder(gravity=0.0)
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
            with wp.ScopedCapture() as capture:
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
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
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


devices = get_cuda_test_devices(mode="basic")


class TestCollisionPipeline(unittest.TestCase):
    pass


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
        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
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

        builder = newton.ModelBuilder(gravity=0.0)
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
    pass


class TestContactEstimator(unittest.TestCase):
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
        from newton._src.geometry.flags import ShapeFlags  # noqa: PLC0415

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


def _build_deterministic_scene(device):
    """Build the mixed-shape scene from example_basic_shapes6_determinism."""
    from newton._src.geometry import create_mesh_terrain  # noqa: PLC0415

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
    """Run 500 steps of the mixed-shape scene and assert bit-identical contacts on every collide."""
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

        total_checks = 0
        for _frame in range(500):
            for _sub in range(sim_substeps):
                state_0.clear_forces()
                pipeline_a.collide(state_0, contacts_a)
                pipeline_b.collide(state_0, contacts_b)

                count_a = int(contacts_a.rigid_contact_count.numpy()[0])
                count_b = int(contacts_b.rigid_contact_count.numpy()[0])
                test.assertEqual(
                    count_a,
                    count_b,
                    f"Contact count mismatch at frame {_frame} substep {_sub}: {count_a} vs {count_b}",
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
                                f"Determinism failure in {name} at frame {_frame} substep {_sub} "
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
                total_checks += 1

                solver.step(state_0, state_1, control, contacts_a, sim_dt)
                state_0, state_1 = state_1, state_0


add_function_test(
    TestDeterministicPipeline,
    "test_deterministic_pipeline_500_steps",
    test_deterministic_pipeline_500_steps,
    devices=get_cuda_test_devices(),
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
