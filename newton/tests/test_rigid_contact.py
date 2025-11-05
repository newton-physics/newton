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

        builder.add_articulation()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_sphere(
            body=b,
            radius=0.1 * scale,
        )
        expected_end_positions.append(wp.vec3(0.0, y_pos, 0.1 * scale))

        builder.add_articulation()
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

        builder.add_articulation()
        b = builder.add_body(xform=wp.transform(wp.vec3(4.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_box(
            body=b,
            hx=0.2 * scale,
            hy=0.25 * scale,
            hz=0.3 * scale,
        )
        expected_end_positions.append(wp.vec3(4.0, y_pos, 0.3 * scale))

        builder.add_articulation()
        b = builder.add_body(xform=wp.transform(wp.vec3(5.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_cylinder(
            body=b,
            radius=0.1 * scale,
            half_height=0.3 * scale,
        )
        expected_end_positions.append(wp.vec3(5.0, y_pos, 0.3 * scale))

        builder.add_articulation()
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


def test_shapes_on_plane_with_up_axis(test: TestRigidContact, device, solver_fn, up_axis: newton.Axis):
    # Create a simple scenario to verify static shape handling with different up-axis
    builder = newton.ModelBuilder(up_axis=up_axis, gravity=-9.81)
    builder.default_shape_cfg.ke = 2e4
    builder.default_shape_cfg.kd = 500.0
    # !!! disable friction for Euler integrators
    builder.default_shape_cfg.kf = 0.0

    # Add a single sphere falling onto the ground to verify the physics still work with up-axis
    # Position the sphere initially above the ground in the up-axis direction
    initial_pos = wp.vec3(0.0, 0.0, 0.0)
    if up_axis == newton.Axis.X:
        initial_pos = wp.vec3(1.0, 0.0, 0.0)  # Position above ground in X direction
    elif up_axis == newton.Axis.Y:
        initial_pos = wp.vec3(0.0, 1.0, 0.0)  # Position above ground in Y direction
    else:  # Z-axis (default)
        initial_pos = wp.vec3(0.0, 0.0, 1.0)  # Position above ground in Z direction

    builder.add_articulation()
    b = builder.add_body(xform=wp.transform(initial_pos, wp.quat_identity()))
    builder.add_joint_free(b)
    builder.add_shape_sphere(
        body=b,
        radius=0.1,
    )

    builder.add_ground_plane()

    model = builder.finalize(device=device)

    # Verify that static shapes (ground plane) have the correct orientation for the up-axis
    shape_bodies = model.shape_body.numpy()
    shape_transforms = model.shape_transform.numpy()

    # Find the ground plane shape(s) which should be static (shape_body == -1)
    for i, (shape_body, shape_transform_arr) in enumerate(zip(shape_bodies, shape_transforms, strict=False)):
        if shape_body == -1:  # static shape (ground plane)
            # Convert the numpy array to a proper transform object
            shape_transform = wp.transform(*shape_transform_arr)

            # The ground plane should be oriented such that its normal aligns with the up-axis
            # For up_axis X, Y, Z the normals should be (1,0,0), (0,1,0), (0,0,1) respectively
            expected_normal = wp.vec3(0.0, 0.0, 0.0)
            if up_axis == newton.Axis.X:
                expected_normal = wp.vec3(1.0, 0.0, 0.0)
            elif up_axis == newton.Axis.Y:
                expected_normal = wp.vec3(0.0, 1.0, 0.0)
            else:  # Z-axis
                expected_normal = wp.vec3(0.0, 0.0, 1.0)

            # Apply the rotation to the default local Z-axis (0,0,1) to get the actual normal
            local_z_axis = wp.vec3(0.0, 0.0, 1.0)
            actual_normal = wp.quat_rotate(shape_transform.q, local_z_axis)

            # Check that the normal aligns with the expected up-axis
            for j in range(3):  # x, y, z components
                test.assertAlmostEqual(
                    actual_normal[j],
                    expected_normal[j],
                    delta=1e-5,
                    msg=f"Ground plane normal mismatch for up_axis {up_axis} at component {j}, shape {i}",
                )

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

    # Simulate for a few steps to make sure physics is stable
    for _ in range(250):
        if use_cuda_graph:
            wp.capture_launch(graph)
        else:
            simulate(solver, model, state_0, state_1, control, sim_dt, substeps)

    # For up-axis, the sphere should end up at the appropriate coordinate near the ground
    body_q = state_0.body_q.numpy()

    # Check final position based on up-axis
    if up_axis == newton.Axis.X:
        # In X-up, gravity pulls in -X direction, so sphere should end up at around X=0.1 (radius above ground at X=0)
        expected_pos = wp.vec3(0.1, 0.0, 0.0)
    elif up_axis == newton.Axis.Y:
        # In Y-up, gravity pulls in -Y direction, so sphere should end up at around Y=0.1 (radius above ground at Y=0)
        expected_pos = wp.vec3(0.0, 0.1, 0.0)
    else:  # Z-axis
        # In Z-up, gravity pulls in -Z direction, so sphere should end up at around Z=0.1 (radius above ground at Z=0)
        expected_pos = wp.vec3(0.0, 0.0, 0.1)

    assert_np_equal(body_q[0, :3], np.array(expected_pos), tol=2e-1)
    expected_quat = wp.quat_identity()
    assert_np_equal(body_q[0, 3:], np.array(expected_quat), tol=1e-1)


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

# Add tests for static shape handling with different up axes
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cpu and solver_name == "mujoco_warp":
            continue
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        add_function_test(
            TestRigidContact,
            f"test_shapes_on_plane_x_up_{solver_name}",
            test_shapes_on_plane_with_up_axis,
            devices=[device],
            solver_fn=solver_fn,
            up_axis=newton.Axis.X,
        )
        add_function_test(
            TestRigidContact,
            f"test_shapes_on_plane_y_up_{solver_name}",
            test_shapes_on_plane_with_up_axis,
            devices=[device],
            solver_fn=solver_fn,
            up_axis=newton.Axis.Y,
        )

if __name__ == "__main__":
    # wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
