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

import warp as wp

import newton
from newton.solvers import SolverSemiImplicit, SolverXPBD
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestRuntimeGravity(unittest.TestCase):
    pass


def test_runtime_gravity_particles(test, device, solver_fn):
    """Test that particles respond correctly to runtime gravity changes"""
    builder = newton.ModelBuilder(gravity=-9.81)

    # Add a particle
    builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()

    dt = 0.01

    # Step 1: Simulate with default gravity
    for _ in range(10):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    z_vel_default = state_0.particle_qd.numpy()[0, 2]
    test.assertLess(z_vel_default, -0.5)  # Should be falling

    # Step 2: Change gravity to zero at runtime
    model.set_gravity((0.0, 0.0, 0.0))
    solver.notify_model_changed(newton.solvers.SolverNotifyFlags.MODEL_PROPERTIES)

    # Simulate with zero gravity
    for _ in range(10):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    z_vel_zero_g = state_0.particle_qd.numpy()[0, 2]
    # Velocity should remain constant with zero gravity
    test.assertAlmostEqual(z_vel_zero_g, z_vel_default, places=4)

    # Step 3: Change gravity to positive (upward)
    model.set_gravity((0.0, 0.0, 9.81))
    solver.notify_model_changed(newton.solvers.SolverNotifyFlags.MODEL_PROPERTIES)

    # Simulate with upward gravity
    for _ in range(20):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    z_vel_upward = state_0.particle_qd.numpy()[0, 2]
    test.assertGreater(z_vel_upward, z_vel_zero_g)  # Should be accelerating upward


def test_runtime_gravity_bodies(test, device, solver_fn):
    """Test that rigid bodies respond correctly to runtime gravity changes"""
    builder = newton.ModelBuilder(gravity=-9.81)

    # Set default shape density
    builder.default_shape_cfg.density = 1000.0

    # Add a free-floating rigid body
    b = builder.add_body()
    builder.add_shape_box(b, hx=0.5, hy=0.5, hz=0.5)
    builder.add_joint_free(b)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()

    dt = 0.01

    # Step 1: Simulate with default gravity
    for _ in range(10):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    body_vel_default = state_0.body_qd.numpy()[0, :3]
    test.assertLess(body_vel_default[2], -0.5)  # Should be falling

    # Step 2: Change gravity to horizontal
    model.set_gravity((9.81, 0.0, 0.0))
    solver.notify_model_changed(newton.solvers.SolverNotifyFlags.MODEL_PROPERTIES)

    # Simulate with horizontal gravity
    for _ in range(20):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    body_vel_horizontal = state_0.body_qd.numpy()[0, :3]
    test.assertGreater(body_vel_horizontal[0], 0.5)  # Should be accelerating in X direction


def test_gravity_fallback(test, device):
    """Test that solvers fall back to model gravity when state gravity is not set"""
    builder = newton.ModelBuilder(gravity=-9.81)

    # Add a particle
    builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0)

    model = builder.finalize(device=device)
    solver = SolverXPBD(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()

    # Verify model gravity is set correctly
    gravity_vec = model.gravity.numpy()[0]
    test.assertAlmostEqual(gravity_vec[2], -9.81, places=4)

    dt = 0.01

    # Simulate with model gravity
    for _ in range(10):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    z_vel = state_0.particle_qd.numpy()[0, 2]
    test.assertLess(z_vel, -0.5)  # Should be falling with model gravity


def test_runtime_gravity_with_cuda_graph(test, device):
    """Test that runtime gravity changes work with CUDA graph capture"""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture only available on CUDA devices")

    builder = newton.ModelBuilder(gravity=-9.81)

    # Add a few particles
    for i in range(5):
        builder.add_particle(pos=(i * 0.5, 0.0, 2.0), vel=(0.0, 0.0, 0.0), mass=1.0)

    model = builder.finalize(device=device)
    solver = SolverXPBD(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    dt = 0.01

    # Step once to initialize
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, dt)

    # Start graph capture
    wp.capture_begin()

    try:
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)

        # End capture and get graph
        graph = wp.capture_end()

        # Now test that we can change gravity and it affects the simulation
        # even when using the captured graph

        # Test 1: Default gravity
        for _ in range(10):
            wp.capture_launch(graph)
            state_0, state_1 = state_1, state_0

        z_vel_default = state_0.particle_qd.numpy()[0, 2]
        test.assertLess(z_vel_default, -0.5)  # Should be falling

        # Test 2: Change to zero gravity
        model.set_gravity((0.0, 0.0, 0.0))
        # Note: We don't need to notify solver for graph replay

        vel_before = state_0.particle_qd.numpy()[0, 2]
        for _ in range(10):
            wp.capture_launch(graph)
            state_0, state_1 = state_1, state_0

        vel_after = state_0.particle_qd.numpy()[0, 2]
        test.assertAlmostEqual(vel_before, vel_after, places=4)  # Velocity should stay constant

        # Test 3: Change to upward gravity
        model.set_gravity((0.0, 0.0, 9.81))

        for _ in range(20):
            wp.capture_launch(graph)
            state_0, state_1 = state_1, state_0

        z_vel_upward = state_0.particle_qd.numpy()[0, 2]
        test.assertGreater(z_vel_upward, 0.5)  # Should be moving upward

    except Exception as e:
        # Make sure to end capture if something goes wrong
        wp.capture_end()
        raise e


devices = get_test_devices()

# Test with different solvers
solvers_particles = {
    "xpbd": lambda model: SolverXPBD(model),
    "semi_implicit": lambda model: SolverSemiImplicit(model),
}

solvers_bodies = {
    "xpbd": lambda model: SolverXPBD(model),
    "semi_implicit": lambda model: SolverSemiImplicit(model),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, update_data_interval=0),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, update_data_interval=0),
}

# Add tests for each device and solver combination
for device in devices:
    # Particle tests (MuJoCo doesn't support pure particle simulation)
    for solver_name, solver_fn in solvers_particles.items():
        add_function_test(
            TestRuntimeGravity,
            f"test_runtime_gravity_particles_{solver_name}",
            test_runtime_gravity_particles,
            devices=[device],
            solver_fn=solver_fn,
        )

    # Body tests (all solvers including MuJoCo)
    for solver_name, solver_fn in solvers_bodies.items():
        # Skip CPU MuJoCo on CUDA devices
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        add_function_test(
            TestRuntimeGravity,
            f"test_runtime_gravity_bodies_{solver_name}",
            test_runtime_gravity_bodies,
            devices=[device],
            solver_fn=solver_fn,
        )

    # Test gravity fallback once per device
    add_function_test(
        TestRuntimeGravity,
        "test_gravity_fallback",
        test_gravity_fallback,
        devices=[device],
    )

    # Test CUDA graph capture (only on CUDA devices)
    if device.is_cuda:
        add_function_test(
            TestRuntimeGravity,
            "test_runtime_gravity_with_cuda_graph",
            test_runtime_gravity_with_cuda_graph,
            devices=[device],
        )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
