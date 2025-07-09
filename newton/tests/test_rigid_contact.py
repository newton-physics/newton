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
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices

wp.config.quiet = True


class TestRigidContact(unittest.TestCase):
    pass


def simulate(solver, model, state_0, state_1, control, sim_dt, substeps):
    if not isinstance(solver, newton.solvers.MuJoCoSolver):
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
    cube_mesh = newton.geometry.Mesh(
        vertices = [
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
        ],
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
    # !!! disable friction for Euler integrators
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
        builder.add_shape_capsule(
            body=b,
            radius=0.1 * scale,
            half_height=0.3 * scale,
            up_axis=newton.Axis.Y,
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

        b = builder.add_body(xform=wp.transform(wp.vec3(6.0, y_pos, 1.0), wp.quat_identity()))
        builder.add_joint_free(b)
        builder.add_shape_mesh(
            body=b,
            mesh=cube_mesh,
            scale=wp.vec3(scale, scale, scale),
        )
        expected_end_positions.append(wp.vec3(6.0, y_pos, 0.3 * scale))

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


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.FeatherstoneSolver(model),
    "mujoco_c": lambda model: newton.solvers.MuJoCoSolver(model, use_mujoco=True),
    "mujoco_warp": lambda model: newton.solvers.MuJoCoSolver(model, use_mujoco=False),
    "xpbd": lambda model: newton.solvers.XPBDSolver(model, iterations=2),
    "semi_implicit": lambda model: newton.solvers.SemiImplicitSolver(model),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cpu and solver_name == "mujoco_warp":
            continue
        if device.is_cuda and solver_name == "mujoco_c":
            continue
        add_function_test(
            TestRigidContact,
            f"test_shapes_on_plane_{solver_name}",
            test_shapes_on_plane,
            devices=[device],
            solver_fn=solver_fn,
        )

if __name__ == "__main__":
    # wp.clear_kernel_cache()
    unittest.main(verbosity=2)
