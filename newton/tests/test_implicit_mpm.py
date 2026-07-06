# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_sand_cube_on_plane(test, device):
    # Emits a cube of particles on the ground

    N = 4
    particles_per_cell = 3
    voxel_size = 0.5
    particle_spacing = voxel_size / particles_per_cell
    friction = 0.6
    dt = 0.04

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)

    # Register MPM custom attributes before adding particles
    SolverImplicitMPM.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.5 * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=N * particles_per_cell,
        dim_y=N * particles_per_cell,
        dim_z=N * particles_per_cell,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=1.0,
        jitter=0.0,
        custom_attributes={"mpm:friction": friction},
    )
    builder.add_ground_plane()

    model: newton.Model = builder.finalize(device=device)

    state_0: newton.State = model.state()
    state_1: newton.State = model.state()

    options = SolverImplicitMPM.Config()
    options.grid_type = "dense"  # use dense grid as sparse grid is GPU-only
    options.voxel_size = voxel_size

    solver = SolverImplicitMPM(model, config=options)

    init_pos = state_0.particle_q.numpy()

    # Run a few steps
    for _k in range(25):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    # Checks the final bounding box corresponds to the expected collapse
    end_pos = state_0.particle_q.numpy()
    bb_min, bb_max = np.min(end_pos, axis=0), np.max(end_pos, axis=0)
    assert bb_min[model.up_axis] > -voxel_size
    assert voxel_size < bb_max[model.up_axis] < N * voxel_size

    assert np.all(bb_min > -N * voxel_size)
    assert np.all(bb_min < np.min(init_pos, axis=0))
    assert np.all(bb_max < 2 * N * voxel_size)

    # Checks that contact impulses are consistent
    impulses, impulse_positions, _collider_ids = solver.collect_collider_impulses(state_0)

    impulses = impulses.numpy()
    impulse_positions = impulse_positions.numpy()

    active_contacts = np.flatnonzero(np.linalg.norm(impulses, axis=1) > 0.01)
    contact_points = impulse_positions[active_contacts]
    contact_impulses = impulses[active_contacts]

    assert np.all(contact_points[:, model.up_axis] == 0.0)
    assert np.all(contact_impulses[:, model.up_axis] < 0.0)


def test_finite_difference_collider_velocity(test, device):
    """Test that finite-difference velocity mode correctly computes collider velocity.

    This test compares the two velocity modes with body_qd=0:
    - instantaneous mode: sees zero velocity (from body_qd), particles don't move with platform
    - finite_difference mode: computes velocity from position change, particles move with platform

    This directly validates that finite-difference mode correctly handles the case where
    body transforms are updated externally but body_qd doesn't reflect the actual motion.
    """
    voxel_size = 0.1
    particles_per_cell = 2
    particle_spacing = voxel_size / particles_per_cell
    dt = 0.02
    n_steps = 15

    # Platform moves in +X direction
    platform_vel_x = 0.5  # m/s

    def run_simulation(velocity_mode):
        """Run simulation with given velocity mode and return particle displacement."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        # Add particles resting on the platform
        builder.add_particle_grid(
            pos=wp.vec3(-0.05, 0.12, -0.05),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2 * particles_per_cell,
            dim_y=2 * particles_per_cell,
            dim_z=2 * particles_per_cell,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=1.0,
            jitter=0.0,
            custom_attributes={"mpm:friction": 1.0},  # high friction
        )

        # Add a platform that particles rest on
        platform_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        platform_mesh = newton.Mesh.create_box(
            0.5,
            0.1,
            0.5,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)  # kinematic
        shape_cfg.margin = 0.02
        builder.add_shape_mesh(
            body=platform_body,
            mesh=platform_mesh,
            cfg=shape_cfg,
        )

        model = builder.finalize(device=device)

        state_0 = model.state()
        state_1 = model.state()

        options = SolverImplicitMPM.Config()
        options.voxel_size = voxel_size
        options.grid_type = "dense"
        options.collider_velocity_mode = velocity_mode

        solver = SolverImplicitMPM(model, config=options)

        init_mean_x = np.mean(state_0.particle_q.numpy()[:, 0])

        # Move platform with body_qd = 0
        for k in range(n_steps):
            t = (k + 1) * dt
            new_platform_x = platform_vel_x * t

            body_q_np = state_0.body_q.numpy()
            body_q_np[0] = (new_platform_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            state_0.body_q.assign(body_q_np)

            # KEY: body_qd is ZERO - doesn't reflect actual motion
            body_qd_np = state_0.body_qd.numpy()
            body_qd_np[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            state_0.body_qd.assign(body_qd_np)

            solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
            state_0, state_1 = state_1, state_0

        end_mean_x = np.mean(state_0.particle_q.numpy()[:, 0])
        return end_mean_x - init_mean_x

    # 'forward' sees the current collider velocity; 'backward' derives it from
    # the previous-step collider position.
    displacement_instantaneous = run_simulation("forward")
    displacement_finite_diff = run_simulation("backward")

    # With instantaneous mode and body_qd=0, particles should barely move
    # (they see zero collider velocity, so no friction drag)
    test.assertLess(
        abs(displacement_instantaneous),
        0.02,
        f"instantaneous mode with body_qd=0 should show minimal particle movement, "
        f"but got {displacement_instantaneous:.3f}",
    )

    # With finite_difference mode, particles should move significantly
    # (velocity computed from position change)
    test.assertGreater(
        displacement_finite_diff,
        0.05,
        f"finite_difference mode should move particles with platform, "
        f"but displacement was only {displacement_finite_diff:.3f}",
    )

    # finite_difference should show significantly more movement than instantaneous
    test.assertGreater(
        displacement_finite_diff,
        displacement_instantaneous + 0.03,
        f"finite_difference ({displacement_finite_diff:.3f}) should show significantly more "
        f"movement than instantaneous ({displacement_instantaneous:.3f})",
    )


def test_cg_rheology_whole_step_graph_capture(test, device):
    """Capture a whole step with an iterative linear rheology solver.

    Regression for newton-physics/newton#3155: the iterative linear solver synced
    its device-side results to the host inside the capture, raising CUDA error 906.
    Both verbose settings are covered, since the verbose report is what reads those
    device-side results back. The scene has no colliders so ``solver="cg"`` is
    admissible.
    """
    if not device.is_cuda:
        test.skipTest("whole-step graph capture requires a CUDA device")
    if not wp.is_conditional_graph_supported():
        test.skipTest("whole-step graph capture requires conditional CUDA graph support")

    voxel_size = 0.1
    emit_lo = np.array([-0.15, -0.15, 0.1])
    emit_hi = np.array([0.15, 0.15, 0.4])
    dt = 1.0 / 120.0

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    res = np.ceil(3 * (emit_hi - emit_lo) / voxel_size).astype(int)
    cell = (emit_hi - emit_lo) / res
    builder.add_particle_grid(
        pos=wp.vec3(*emit_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1,
        dim_y=int(res[1]) + 1,
        dim_z=int(res[2]) + 1,
        cell_x=float(cell[0]),
        cell_y=float(cell[1]),
        cell_z=float(cell[2]),
        mass=float(np.prod(cell) * 1000.0),
        jitter=0.0,
        radius_mean=float(np.max(cell) * 0.5),
    )

    model = builder.finalize(device=device)

    # verbose=True is the path that reads the solver's device-side results back
    # for its report; both settings must capture without forcing a host sync.
    for verbose in (False, True):
        with test.subTest(verbose=verbose):
            options = SolverImplicitMPM.Config()
            options.solver = "cg"
            options.voxel_size = voxel_size
            options.grid_type = "fixed"  # whole-step capture precondition
            options.grid_padding = 8
            options.max_active_cell_count = 1 << 15
            options.max_iterations = 50
            options.tolerance = 1.0e-4

            solver = SolverImplicitMPM(model, options, verbose=verbose)
            state_0, state_1 = model.state(), model.state()

            with wp.ScopedCapture(device=device) as capture:
                solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

            for _ in range(5):
                wp.capture_launch(capture.graph)

            # .numpy() performs the synchronous device-to-host copy that drains the replays.
            test.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))


def test_proxy_particle_gravity_is_not_coupling_feedback(test, device):
    gravity = -9.81
    dt = 1.0 / 60.0

    builder = newton.ModelBuilder(gravity=gravity)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.03)
    builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.03)
    model = builder.finalize(device=device)
    model.mpm.yield_pressure.fill_(1.0e5)

    config = SolverImplicitMPM.Config()
    config.voxel_size = 0.2
    config.grid_type = "fixed"
    config.grid_padding = 2
    config.warmstart_mode = "none"
    config.transfer_scheme = "pic"
    config.max_iterations = 1

    solver = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="xpbd",
                solver=lambda view: SolverXPBD(model=view, iterations=1),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="mpm",
                solver=lambda view: SolverImplicitMPM(model=view, config=config),
                particles=[1],
                in_place=True,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="xpbd", destination="mpm", particles=[0])]
        ),
    )

    state_0 = model.state()
    state_1 = model.state()
    for step in range(1, 3):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        expected_velocity = np.array([0.0, 0.0, step * gravity * dt])
        np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_velocity, atol=1.0e-4)
        state_0, state_1 = state_1, state_0


devices = get_test_devices()


class TestImplicitMPM(unittest.TestCase):
    pass


add_function_test(
    TestImplicitMPM, "test_sand_cube_on_plane", test_sand_cube_on_plane, devices=devices, check_output=False
)

add_function_test(
    TestImplicitMPM,
    "test_finite_difference_collider_velocity",
    test_finite_difference_collider_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestImplicitMPM,
    "test_cg_rheology_whole_step_graph_capture",
    test_cg_rheology_whole_step_graph_capture,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestImplicitMPM,
    "test_proxy_particle_gravity_is_not_coupling_feedback",
    test_proxy_particle_gravity_is_not_coupling_feedback,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
