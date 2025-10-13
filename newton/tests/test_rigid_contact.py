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

import numpy as np
import warp as wp

import newton
from newton._src.core import quat_between_axes
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices

wp.config.quiet = True


class TestRigidContact(unittest.TestCase):
    pass


def simulate(solver, model, state_0, state_1, control, sim_dt, substeps):
    if not isinstance(solver, newton.solvers.SolverMuJoCo):
        contacts = model.collide(state_0, rigid_contact_margin=100.0)
    else:
        contacts = None
    for _ in range(substeps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
        state_0, state_1 = state_1, state_0


def test_shapes_on_plane(test: TestRigidContact, device, solver_fn):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1e4
    builder.default_shape_cfg.kd = 500.0
    builder.add_ground_plane()
    size = 0.3
    # fmt: off
    vertices = np.array([
        [-size, -size, -size],
        [-size, -size, size],
        [-size, size, size],
        [-size, size, -size],
        [size, -size, -size],
        [size, -size, size],
        [size, size, size],
        [size, size, -size],
        [-size, -size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, -size, -size],
        [-size, size, -size],
        [-size, size, size],
        [size, size, size],
        [size, size, -size],
        [-size, -size, -size,],
        [-size, size, -size,],
        [size, size, -size,],
        [size, -size, -size,],
        [-size, -size, size],
        [-size, size, size],
        [size, size, size],
        [size, -size, size],
    ], dtype=np.float32)
    # Add some offset to the vertices to test proper handling of non-zero origin
    # e.g. MuJoCo transforms the mesh to the origin
    mesh_offset = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vertices += mesh_offset
    cube_mesh = newton.Mesh(
        vertices=vertices,
        indices = [
            0, 1, 2,
            0, 2, 3,
            4, 6, 5,
            4, 7, 6,
            8, 10, 9,
            8, 11, 10,
            12, 13, 14,
            12, 14, 15,
            16, 17, 18,
            16, 18, 19,
            20, 22, 21,
            20, 23, 22,
        ],
    )
    # fmt: on

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 2e4
    builder.default_shape_cfg.kd = 500.0
    # !!! disable friction for SemiImplicit integrators
    builder.default_shape_cfg.kf = 0.0

    expected_end_positions = []

    for i, scale in enumerate([0.5, 1.0]):
        y_pos = i * 1.5

        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_sphere(
            body=b,
            radius=0.1 * scale,
        )
        expected_end_positions.append(wp.vec3(0.0, y_pos, 0.1 * scale))

        b = builder.add_body(xform=wp.transform(wp.vec3(2.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        # Apply Y-axis rotation to capsule
        xform = wp.transform(wp.vec3(), quat_between_axes(newton.Axis.Z, newton.Axis.Y))
        builder.add_shape_capsule(
            body=b,
            xform=xform,
            radius=0.1 * scale,
            half_height=0.3 * scale,
        )
        expected_end_positions.append(wp.vec3(2.0, y_pos, 0.1 * scale))

        b = builder.add_body(xform=wp.transform(wp.vec3(4.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_box(
            body=b,
            hx=0.2 * scale,
            hy=0.25 * scale,
            hz=0.3 * scale,
        )
        expected_end_positions.append(wp.vec3(4.0, y_pos, 0.3 * scale))

        b = builder.add_body(xform=wp.transform(wp.vec3(5.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_cylinder(
            body=b,
            radius=0.1 * scale,
            half_height=0.3 * scale,
        )
        expected_end_positions.append(wp.vec3(5.0, y_pos, 0.3 * scale))

        b = builder.add_body(xform=wp.transform(wp.vec3(7.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_mesh(
            body=b,
            mesh=cube_mesh,
            scale=wp.vec3(scale, scale, scale),
        )
        expected_end_positions.append(wp.vec3(7.0, y_pos, 0.3 * scale))

    builder.add_ground_plane()

    model = builder.finalize(device=device)

    solver = solver_fn(model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    substeps = 10
    sim_dt = 1.0 / 60.0
    if use_cuda_graph:
        # ensure data is allocated and modules are loaded before graph capture
        # in case of an earlier CUDA version
        simulate(solver, model, state_0, state_1, control, sim_dt, substeps)
        with wp.ScopedCapture(device) as capture:
            simulate(solver, model, state_0, state_1, control, sim_dt, substeps)
        graph = capture.graph

    for _ in range(250):
        if use_cuda_graph:
            wp.capture_launch(graph)
        else:
            simulate(solver, model, state_0, state_1, control, sim_dt, substeps)

    body_q = state_0.body_q.numpy()
    expected_end_positions = np.array(expected_end_positions)
    assert_np_equal(body_q[:, :3], expected_end_positions, tol=1e-1)
    expected_quats = np.tile(wp.quat_identity(), (model.body_count, 1))
    assert_np_equal(body_q[:, 3:], expected_quats, tol=1e-1)


def test_shape_collisions_gjk_mpr_multicontact(test: TestRigidContact, device, verbose=False):
    """Test that objects on a ramp with end wall remain stable (don't move or rotate significantly)"""

    # Scene Configuration (from example_basic_shapes2.py)
    RAMP_LENGTH = 10.0
    RAMP_THICKNESS = 0.5
    RAMP_ANGLE = np.radians(30.0)
    WALL_HEIGHT = 2.0
    CUBE_SIZE = 1.0 * 0.99
    RAMP_WIDTH = CUBE_SIZE * 2.01

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 2e4
    builder.default_shape_cfg.kd = 500.0
    builder.default_shape_cfg.kf = 0.5  # Add some friction

    # Calculate ramp geometry
    ramp_center_y = RAMP_LENGTH / 2 * np.cos(RAMP_ANGLE)
    ramp_center_z = RAMP_LENGTH / 2 * np.sin(RAMP_ANGLE)
    ramp_center = wp.vec3(0.0, ramp_center_y, ramp_center_z)

    # Create tilted ramp using a plane (static)
    ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(RAMP_ANGLE))

    builder.add_shape_plane(
        body=-1,
        xform=wp.transform(p=ramp_center, q=ramp_quat),
        width=0,
        length=0,
    )

    # Compute coordinate system vectors for the tilted ramp
    ramp_forward = wp.quat_rotate(ramp_quat, wp.vec3(0.0, -1.0, 0.0))
    ramp_up = wp.quat_rotate(ramp_quat, wp.vec3(0.0, 0.0, 1.0))
    ramp_right = wp.quat_rotate(ramp_quat, wp.vec3(1.0, 0.0, 0.0))

    ramp_center_surface = ramp_center

    # Add side guide walls along the ramp
    guide_height = 0.3
    guide_thickness = 0.1

    # Left side guide wall
    left_guide_offset = (RAMP_WIDTH / 2 + guide_thickness / 2) * ramp_right
    left_guide_center = ramp_center + left_guide_offset + (guide_height / 2) * ramp_up
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=left_guide_center, q=ramp_quat),
        hx=guide_thickness / 2,
        hy=RAMP_LENGTH / 2,
        hz=guide_height / 2,
    )

    # Right side guide wall
    right_guide_offset = -(RAMP_WIDTH / 2 + guide_thickness / 2) * ramp_right
    right_guide_center = ramp_center + right_guide_offset + (guide_height / 2) * ramp_up
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=right_guide_center, q=ramp_quat),
        hx=guide_thickness / 2,
        hy=RAMP_LENGTH / 2,
        hz=guide_height / 2,
    )

    start_shift = 0.6 * RAMP_LENGTH

    # Create end wall at the bottom of the ramp
    tmp = ramp_center_surface + 0.5 * CUBE_SIZE * (ramp_up + start_shift * ramp_forward)
    wall_y = tmp.y - CUBE_SIZE / 2 * 1.4 - RAMP_THICKNESS / 2
    wall_z = tmp.z

    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, wall_y, wall_z), q=wp.quat_identity()),
        hx=RAMP_WIDTH / 2,
        hy=RAMP_THICKNESS / 2,
        hz=WALL_HEIGHT / 2,
    )

    # Rotate shapes to match ramp orientation
    cube_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(RAMP_ANGLE))

    offset_a = 0.5 * CUBE_SIZE * (ramp_up + ramp_right + start_shift * ramp_forward)
    offset_b = 0.5 * CUBE_SIZE * (ramp_up - ramp_right + start_shift * ramp_forward)

    # Cube 1 (left side)
    body_cube1 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat))
    builder.add_joint_free(body_cube1)
    builder.add_shape_box(body=body_cube1, hx=CUBE_SIZE / 2, hy=CUBE_SIZE / 2, hz=CUBE_SIZE / 2)

    # Cube 2 (right side)
    body_cube2 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat))
    builder.add_joint_free(body_cube2)
    builder.add_shape_box(body=body_cube2, hx=CUBE_SIZE / 2, hy=CUBE_SIZE / 2, hz=CUBE_SIZE / 2)

    # Spheres
    offset_a = 0.5 * CUBE_SIZE * (ramp_up + ramp_right + (start_shift - 2.01) * ramp_forward)
    offset_b = 0.5 * CUBE_SIZE * (ramp_up - ramp_right + (start_shift - 2.01) * ramp_forward)

    sphere_radius = CUBE_SIZE / 2
    body_sphere1 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat))
    builder.add_joint_free(body_sphere1)
    builder.add_shape_sphere(body=body_sphere1, radius=sphere_radius)

    body_sphere2 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat))
    builder.add_joint_free(body_sphere2)
    builder.add_shape_sphere(body=body_sphere2, radius=sphere_radius)

    # Capsule
    capsule_radius = CUBE_SIZE / 2
    capsule_height = 2 * capsule_radius
    offset_capsule = 0.5 * CUBE_SIZE * (ramp_up + (start_shift - 4.02) * ramp_forward)

    capsule_local_quat = quat_between_axes(newton.Axis.Z, newton.Axis.X)
    capsule_quat = cube_quat * capsule_local_quat

    body_capsule = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_capsule, q=capsule_quat))
    builder.add_joint_free(body_capsule)
    builder.add_shape_capsule(body=body_capsule, radius=capsule_radius, half_height=capsule_height / 2)

    # Cylinder
    cylinder_radius = CUBE_SIZE / 2
    cylinder_height = 4 * cylinder_radius
    offset_cylinder = 0.5 * CUBE_SIZE * (ramp_up + (start_shift - 6.03) * ramp_forward)

    cylinder_local_quat = quat_between_axes(newton.Axis.Z, newton.Axis.X)
    cylinder_quat = cube_quat * cylinder_local_quat

    body_cylinder = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_cylinder, q=cylinder_quat))
    builder.add_joint_free(body_cylinder)
    builder.add_shape_cylinder(body=body_cylinder, radius=cylinder_radius, half_height=cylinder_height / 2)

    # Two more cubes after the cylinder
    offset_a = 0.5 * CUBE_SIZE * (ramp_up + ramp_right + (start_shift - 8.04) * ramp_forward)
    offset_b = 0.5 * CUBE_SIZE * (ramp_up - ramp_right + (start_shift - 8.04) * ramp_forward)

    # Cube 3 (left side)
    body_cube3 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat))
    builder.add_joint_free(body_cube3)
    builder.add_shape_box(body=body_cube3, hx=CUBE_SIZE / 2, hy=CUBE_SIZE / 2, hz=CUBE_SIZE / 2)

    # Cube 4 (right side)
    body_cube4 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat))
    builder.add_joint_free(body_cube4)
    builder.add_shape_box(body=body_cube4, hx=CUBE_SIZE / 2, hy=CUBE_SIZE / 2, hz=CUBE_SIZE / 2)

    # Two cones after the cubes (z-axis aligned with ramp_up)
    cone_radius = CUBE_SIZE / 2
    cone_height = 2 * cone_radius
    offset_a = 0.5 * CUBE_SIZE * (ramp_up + ramp_right + (start_shift - 10.05) * ramp_forward)
    offset_b = 0.5 * CUBE_SIZE * (ramp_up - ramp_right + (start_shift - 10.05) * ramp_forward)

    cone_quat = cube_quat

    # Cone 1 (left side)
    body_cone1 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_a, q=cone_quat))
    builder.add_joint_free(body_cone1)
    builder.add_shape_cone(body=body_cone1, radius=cone_radius, half_height=cone_height / 2)

    # Cone 2 (right side)
    body_cone2 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_b, q=cone_quat))
    builder.add_joint_free(body_cone2)
    builder.add_shape_cone(body=body_cone2, radius=cone_radius, half_height=cone_height / 2)

    # Two more cubes after the cones
    offset_a = 0.5 * CUBE_SIZE * (ramp_up + ramp_right + (start_shift - 12.06) * ramp_forward)
    offset_b = 0.5 * CUBE_SIZE * (ramp_up - ramp_right + (start_shift - 12.06) * ramp_forward)

    # Cube 5 (left side)
    body_cube5 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat))
    builder.add_joint_free(body_cube5)
    builder.add_shape_box(body=body_cube5, hx=CUBE_SIZE / 2, hy=CUBE_SIZE / 2, hz=CUBE_SIZE / 2)

    # Cube 6 (right side)
    body_cube6 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat))
    builder.add_joint_free(body_cube6)
    builder.add_shape_box(body=body_cube6, hx=CUBE_SIZE / 2, hy=CUBE_SIZE / 2, hz=CUBE_SIZE / 2)

    # Two cubes using convex hull representation (8 corner points)
    cube_half = CUBE_SIZE / 2
    cube_vertices = np.array(
        [
            # Bottom face (z = -cube_half)
            [-cube_half, -cube_half, -cube_half],
            [cube_half, -cube_half, -cube_half],
            [cube_half, cube_half, -cube_half],
            [-cube_half, cube_half, -cube_half],
            # Top face (z = cube_half)
            [-cube_half, -cube_half, cube_half],
            [cube_half, -cube_half, cube_half],
            [cube_half, cube_half, cube_half],
            [-cube_half, cube_half, cube_half],
        ],
        dtype=np.float32,
    )

    cube_indices = np.array(
        [
            0,
            2,
            1,
            0,
            3,
            2,  # Bottom face
            4,
            5,
            6,
            4,
            6,
            7,  # Top face
            0,
            1,
            5,
            0,
            5,
            4,  # Front face
            1,
            2,
            6,
            1,
            6,
            5,  # Right face
            2,
            3,
            7,
            2,
            7,
            6,  # Back face
            3,
            0,
            4,
            3,
            4,
            7,  # Left face
        ],
        dtype=np.int32,
    )

    cube_mesh = newton.Mesh(cube_vertices, cube_indices)

    offset_a = 0.5 * CUBE_SIZE * (ramp_up + ramp_right + (start_shift - 14.07) * ramp_forward)
    offset_b = 0.5 * CUBE_SIZE * (ramp_up - ramp_right + (start_shift - 14.07) * ramp_forward)

    convex_cube_quat = cube_quat

    # Convex Hull Cube 1 (left side)
    body_convex_cube1 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_a, q=convex_cube_quat))
    builder.add_joint_free(body_convex_cube1)
    builder.add_shape_convex_hull(body=body_convex_cube1, mesh=cube_mesh, scale=(1.0, 1.0, 1.0))

    # Convex Hull Cube 2 (right side)
    body_convex_cube2 = builder.add_body(xform=wp.transform(p=ramp_center_surface + offset_b, q=convex_cube_quat))
    builder.add_joint_free(body_convex_cube2)
    builder.add_shape_convex_hull(body=body_convex_cube2, mesh=cube_mesh, scale=(1.0, 1.0, 1.0))

    # Add ground plane
    builder.add_ground_plane()

    # Finalize model (shape pairs are built automatically)
    model = builder.finalize(device=device)

    # Create CollisionPipelineUnified with EXPLICIT broad phase mode
    collision_pipeline = newton.CollisionPipelineUnified.from_model(
        model,
        rigid_contact_max_per_pair=10,
        rigid_contact_margin=0.01,
        broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
    )

    # Use XPBD solver
    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Store initial positions and rotations
    initial_body_q = state_0.body_q.numpy().copy()

    # Simulate for 100 frames (same as example_basic_shapes2.py)
    substeps = 10
    sim_dt = 1.0 / 60.0
    max_frames = 100

    for _frame in range(max_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            # Use unified collision pipeline (CollisionPipelineUnified)
            contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
            solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
            state_0, state_1 = state_1, state_0

    # Get final positions and rotations
    final_body_q = state_0.body_q.numpy()

    # Print results for each body (same as example_basic_shapes2.py)
    if verbose:
        print("\n" + "=" * 80)
        print(f"TEST RESULTS AFTER {max_frames} FRAMES ({max_frames * sim_dt:.2f} seconds)")
        print("=" * 80)

        for i in range(model.body_count):
            # Calculate position displacement
            initial_pos = initial_body_q[i, :3]
            final_pos = final_body_q[i, :3]
            displacement = np.linalg.norm(final_pos - initial_pos)

            # Calculate rotation angle using quaternion math
            initial_quat = initial_body_q[i, 3:]
            final_quat = final_body_q[i, 3:]

            dot_product = np.abs(np.dot(initial_quat, final_quat))
            dot_product = np.clip(dot_product, 0.0, 1.0)
            rotation_angle_rad = 2.0 * np.arccos(dot_product)
            rotation_angle_deg = np.degrees(rotation_angle_rad)

            print(f"Body {i}: displacement = {displacement:.6f} units, rotation = {rotation_angle_deg:.2f} degrees")

        print("=" * 80 + "\n")

    # Now check thresholds (more relaxed than before)
    position_threshold = 0.15 * CUBE_SIZE  # Allow up to 0.15 * CUBE_SIZE movement
    max_rotation_deg = 10.0  # Allow up to 10 degrees rotation

    for i in range(model.body_count):
        initial_pos = initial_body_q[i, :3]
        final_pos = final_body_q[i, :3]
        displacement = np.linalg.norm(final_pos - initial_pos)

        test.assertLess(
            displacement,
            position_threshold,
            f"Body {i} moved {displacement:.6f}, exceeding threshold {position_threshold:.6f}",
        )

        initial_quat = initial_body_q[i, 3:]
        final_quat = final_body_q[i, 3:]

        dot_product = np.abs(np.dot(initial_quat, final_quat))
        dot_product = np.clip(dot_product, 0.0, 1.0)
        rotation_angle = 2.0 * np.arccos(dot_product)

        test.assertLess(
            rotation_angle,
            np.radians(max_rotation_deg),
            f"Body {i} rotated {np.degrees(rotation_angle):.2f} degrees, exceeding threshold {max_rotation_deg} degrees",
        )


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.SolverFeatherstone(model),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, njmax=150),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, iterations=2),
    "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cpu and solver_name == "mujoco_warp":
            continue
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        add_function_test(
            TestRigidContact,
            f"test_shapes_on_plane_{solver_name}",
            test_shapes_on_plane,
            devices=[device],
            solver_fn=solver_fn,
        )

# Add test for ramp scene stability with XPBD solver
for device in devices:
    add_function_test(
        TestRigidContact,
        "test_shape_collisions_gjk_mpr_multicontact",
        test_shape_collisions_gjk_mpr_multicontact,
        devices=[device],
    )


def test_mujoco_warp_newton_contacts(test: TestRigidContact, device):
    """Test that MuJoCo Warp solver correctly handles contact transfer from Newton's unified collision pipeline.

    This test creates 4 environments, each with a single cube on the ground, and verifies that the cubes
    remain stable (don't fall through the ground) when using Newton's collision detection with MuJoCo Warp solver.
    """
    # Create a simple cube model
    cube_builder = newton.ModelBuilder()
    cube_builder.default_shape_cfg.ke = 5.0e4
    cube_builder.default_shape_cfg.kd = 5.0e2
    cube_builder.default_shape_cfg.kf = 1.0e3
    cube_builder.default_shape_cfg.mu = 0.75

    # Add a single cube body
    cube_size = 0.5
    body = cube_builder.add_body(xform=wp.transform(wp.vec3(0, 0, cube_size), wp.quat_identity()))
    cube_builder.add_joint_free(body)
    cube_builder.add_shape_box(body=body, hx=cube_size / 2, hy=cube_size / 2, hz=cube_size / 2)

    # Replicate the cube across 4 environments
    builder = newton.ModelBuilder()
    num_envs = 4
    builder.replicate(cube_builder, num_envs, spacing=(3, 3, 0))

    # Add ground plane
    builder.add_ground_plane()

    # Finalize model (shape pairs are built automatically)
    model = builder.finalize(device=device)

    # Create unified collision pipeline (critical for this test)
    collision_pipeline = newton.CollisionPipelineUnified.from_model(
        model,
        rigid_contact_max_per_pair=10,
        rigid_contact_margin=0.01,
        broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
    )

    # Create MuJoCo Warp solver with Newton contacts
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        use_mujoco_contacts=False,  # Use Newton's collision pipeline instead of MuJoCo's
        solver="newton",
        integrator="euler",
        njmax=100,
        ncon_per_env=50,
        cone="elliptic",
        impratio=100,
        iterations=100,
        ls_iterations=50,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Store initial positions (cubes should be at z = cube_size)
    initial_body_q = state_0.body_q.numpy().copy()

    # Simulate for enough frames to ensure cubes settle
    substeps = 6
    sim_dt = 1.0 / 60.0
    max_frames = 100

    for _ in range(max_frames):
        for _ in range(substeps):
            state_0.clear_forces()

            # Use unified collision pipeline - this is the key part being tested
            contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

            solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
            state_0, state_1 = state_1, state_0

    # Get final positions
    final_body_q = state_0.body_q.numpy()

    # Test that cubes are resting on the ground (not fallen through)
    # Each cube should be at approximately z = cube_size/2 (half the cube size)
    for i in range(num_envs):
        initial_z = initial_body_q[i, 2]
        final_z = final_body_q[i, 2]

        # The cube should have settled down from z=cube_size to approximately z=cube_size/2
        test.assertGreater(
            final_z,
            cube_size * 0.3,  # Should be well above ground (at least 30% of cube size)
            f"Cube {i} fell through the ground (z={final_z:.6f}, expected > {cube_size * 0.3:.6f})",
        )

        test.assertLess(
            final_z,
            initial_z + 0.1,  # Should not have jumped up significantly
            f"Cube {i} jumped up unexpectedly (z={final_z:.6f}, initial={initial_z:.6f})",
        )

        # Check that the cube is approximately at rest (small velocity)
        final_vel_z = state_0.body_qd.numpy()[i, 2]
        test.assertLess(
            abs(final_vel_z),
            0.01,
            f"Cube {i} has too much vertical velocity ({final_vel_z:.6f}), not at rest",
        )


# Add test for MuJoCo Warp with Newton contacts (only for CUDA devices)
for device in devices:
    if device.is_cuda:
        add_function_test(
            TestRigidContact,
            "test_mujoco_warp_newton_contacts",
            test_mujoco_warp_newton_contacts,
            devices=[device],
        )

if __name__ == "__main__":
    # wp.clear_kernel_cache()
    unittest.main(verbosity=2)
