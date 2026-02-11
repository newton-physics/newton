# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

import unittest
from enum import IntFlag, auto

import numpy as np
import warp as wp
import warp.examples

import newton
from newton import GeoType
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
        broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
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

        self.builder = newton.ModelBuilder(gravity=0.0)
        # Set contact margin to match previous test expectations
        # Note: margins are now summed (margin_a + margin_b), so we use half the previous value
        self.builder.rigid_contact_margin = 0.005

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

        # Create collision pipeline with the requested broad phase mode
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase_mode=broad_phase_mode,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.solver = solver_fn(self.model)

        self.viewer = viewer
        self.viewer.set_model(self.model)

        self.graph = None
        if wp.get_device(device).is_cuda:
            # Warm-up: run once outside capture
            self.simulate()
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def add_shape(self, shape_type: GeoType, body: int, sdf_max_resolution: int | None = None):
        if shape_type == GeoType.BOX:
            self.builder.add_shape_box(body, key=type_to_str(shape_type))
        elif shape_type == GeoType.SPHERE:
            self.builder.add_shape_sphere(body, radius=0.5, key=type_to_str(shape_type))
        elif shape_type == GeoType.CAPSULE:
            self.builder.add_shape_capsule(body, radius=0.25, half_height=0.3, key=type_to_str(shape_type))
        elif shape_type == GeoType.CYLINDER:
            self.builder.add_shape_cylinder(body, radius=0.25, half_height=0.4, key=type_to_str(shape_type))
        elif shape_type == GeoType.CONE:
            # Rotate cone so flat base faces -X (toward the incoming object)
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 2.0)
            xform = wp.transform(wp.vec3(), rot)
            self.builder.add_shape_cone(body, xform=xform, radius=0.25, half_height=0.4, key=type_to_str(shape_type))
        elif shape_type == GeoType.MESH:
            # Use box mesh (works correctly with collision pipeline)
            vertices, indices = newton.utils.create_box_mesh(extents=(0.5, 0.5, 0.5))
            # Configure SDF settings if specified
            cfg = newton.ModelBuilder.ShapeConfig(sdf_max_resolution=sdf_max_resolution)
            self.builder.add_shape_mesh(
                body, mesh=newton.Mesh(vertices[:, :3], indices), cfg=cfg, key=type_to_str(shape_type)
            )
        elif shape_type == GeoType.CONVEX_MESH:
            # Use a sphere mesh as it's already convex
            vertices, indices = newton.utils.create_sphere_mesh(radius=0.5)
            mesh = newton.Mesh(vertices[:, :3], indices)
            self.builder.add_shape_convex_hull(body, mesh=mesh, key=type_to_str(shape_type))
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
        body_name = f"body {body} ({self.model.shape_key[body]})"
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
    (GeoType.BOX, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR, 0.02),
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
    (GeoType.MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CONVEX_MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
]


def test_collision_pipeline(
    _test,
    device,
    shape_type_a: GeoType,
    shape_type_b: GeoType,
    test_level_a: TestLevel,
    test_level_b: TestLevel,
    broad_phase_mode: newton.BroadPhaseMode,
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
        broad_phase_mode=broad_phase_mode,
    )
    for _ in range(200):
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
        _test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, newton.BroadPhaseMode.EXPLICIT, tolerance
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
    test_collision_pipeline(
        _test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, newton.BroadPhaseMode.NXN, tolerance
    )


def test_collision_pipeline_sap(
    _test,
    device,
    shape_type_a: GeoType,
    shape_type_b: GeoType,
    test_level_a: TestLevel,
    test_level_b: TestLevel,
    tolerance: float = 3e-3,
):
    test_collision_pipeline(
        _test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, newton.BroadPhaseMode.SAP, tolerance
    )


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
    broad_phase_mode: newton.BroadPhaseMode,
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
        broad_phase_mode=broad_phase_mode,
        sdf_max_resolution_a=sdf_max_resolution_a,
        sdf_max_resolution_b=sdf_max_resolution_b,
    )
    for _ in range(200):
        setup.step()
        setup.render()
    setup.test(TestLevel.VELOCITY_YZ, 0, tolerance=tolerance)
    setup.test(TestLevel.VELOCITY_LINEAR, 1, tolerance=tolerance)


# Wrapper functions for different SDF modes
def test_mesh_mesh_sdf_vs_sdf(_test, device, broad_phase_mode: newton.BroadPhaseMode):
    """Test mesh-mesh collision where both meshes have SDFs."""
    # SDF-SDF hydroelastic contacts can have some variability in contact normal direction
    test_mesh_mesh_sdf_modes(
        _test, device, sdf_max_resolution_a=8, sdf_max_resolution_b=8, broad_phase_mode=broad_phase_mode, tolerance=0.1
    )


def test_mesh_mesh_sdf_vs_bvh(_test, device, broad_phase_mode: newton.BroadPhaseMode):
    """Test mesh-mesh collision where first mesh has SDF, second uses BVH."""
    # Mixed SDF/BVH mode has slightly more asymmetric contact behavior, use higher tolerance
    test_mesh_mesh_sdf_modes(
        _test,
        device,
        sdf_max_resolution_a=8,
        sdf_max_resolution_b=None,
        broad_phase_mode=broad_phase_mode,
        tolerance=0.2,
    )


def test_mesh_mesh_bvh_vs_sdf(_test, device, broad_phase_mode: newton.BroadPhaseMode):
    """Test mesh-mesh collision where first mesh uses BVH, second has SDF."""
    # Mixed SDF/BVH mode has slightly more asymmetric contact behavior, use higher tolerance
    test_mesh_mesh_sdf_modes(
        _test,
        device,
        sdf_max_resolution_a=None,
        sdf_max_resolution_b=8,
        broad_phase_mode=broad_phase_mode,
        tolerance=0.5,
    )


def test_mesh_mesh_bvh_vs_bvh(_test, device, broad_phase_mode: newton.BroadPhaseMode):
    """Test mesh-mesh collision where both meshes use BVH (no SDF)."""
    test_mesh_mesh_sdf_modes(
        _test, device, sdf_max_resolution_a=None, sdf_max_resolution_b=None, broad_phase_mode=broad_phase_mode
    )


# Add mesh-mesh SDF mode tests for all broad phase modes
mesh_mesh_sdf_tests = [
    ("sdf_vs_sdf", test_mesh_mesh_sdf_vs_sdf),
    ("sdf_vs_bvh", test_mesh_mesh_sdf_vs_bvh),
    ("bvh_vs_sdf", test_mesh_mesh_bvh_vs_sdf),
    ("bvh_vs_bvh", test_mesh_mesh_bvh_vs_bvh),
]

for mode_name, test_func in mesh_mesh_sdf_tests:
    for broad_phase_name, broad_phase_mode in [
        ("explicit", newton.BroadPhaseMode.EXPLICIT),
        ("nxn", newton.BroadPhaseMode.NXN),
        ("sap", newton.BroadPhaseMode.SAP),
    ]:
        add_function_test(
            TestCollisionPipeline,
            f"test_mesh_mesh_{mode_name}_{broad_phase_name}",
            test_func,
            devices=devices,
            broad_phase_mode=broad_phase_mode,
            check_output=False,  # Disable output checking due to Warp module loading messages
        )


# ============================================================================
# Shape collision filter pairs (excluded pairs) with NxN/SAP
# ============================================================================


class TestCollisionPipelineFilterPairs(unittest.TestCase):
    pass


def test_shape_collision_filter_pairs(test, device, broad_phase_mode: newton.BroadPhaseMode):
    """Verify that excluded shape pairs produce no contacts under NxN or SAP broad phase.

    Args:
        test: The test case instance.
        device: Warp device to run on.
        broad_phase_mode: Broad phase algorithm to test (NXN or SAP).
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.rigid_contact_margin = 0.01
        # Two overlapping spheres (same position so they definitely overlap)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        shape_a = builder.add_shape_sphere(body=body_a, radius=0.5)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        shape_b = builder.add_shape_sphere(body=body_b, radius=0.5)
        # Exclude this pair so they must not generate contacts
        builder.shape_collision_filter_pairs.append((min(shape_a, shape_b), max(shape_a, shape_b)))
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, broad_phase_mode=broad_phase_mode)
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
                f"Excluded pair {excluded} must not appear in contacts (broad_phase={broad_phase_mode.name})",
            )
        # With the only pair excluded, we must have zero rigid contacts
        test.assertEqual(n, 0, f"Expected 0 rigid contacts when only pair is excluded (got {n})")


add_function_test(
    TestCollisionPipelineFilterPairs,
    "test_shape_collision_filter_pairs_nxn",
    test_shape_collision_filter_pairs,
    devices=devices,
    broad_phase_mode=newton.BroadPhaseMode.NXN,
)
add_function_test(
    TestCollisionPipelineFilterPairs,
    "test_shape_collision_filter_pairs_sap",
    test_shape_collision_filter_pairs,
    devices=devices,
    broad_phase_mode=newton.BroadPhaseMode.SAP,
)


def test_collision_filter_consistent_across_broadphases(test, device):
    """Verify that all broad phase modes produce the same contact pairs when collision filtering is applied.

    Creates three overlapping spheres and excludes one pair, then checks that
    EXPLICIT, NXN, and SAP all report exactly the same set of contacting shape pairs.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.rigid_contact_margin = 0.01

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

        def _contact_pairs(broad_phase_mode):
            pipeline = newton.CollisionPipeline(model, broad_phase_mode=broad_phase_mode)
            state = model.state()
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            n = contacts.rigid_contact_count.numpy()[0]
            pairs = set()
            for i in range(n):
                s0 = int(contacts.rigid_contact_shape0.numpy()[i])
                s1 = int(contacts.rigid_contact_shape1.numpy()[i])
                pairs.add((min(s0, s1), max(s0, s1)))
            return pairs

        pairs_explicit = _contact_pairs(newton.BroadPhaseMode.EXPLICIT)
        pairs_nxn = _contact_pairs(newton.BroadPhaseMode.NXN)
        pairs_sap = _contact_pairs(newton.BroadPhaseMode.SAP)

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
# Particle-Shape (Soft) Contact Tests
# ============================================================================
# These tests verify that particle-shape contacts are correctly generated
# by both collision pipelines.


class TestParticleShapeContacts(unittest.TestCase):
    pass


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
            broad_phase_mode=newton.BroadPhaseMode.NXN,
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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
