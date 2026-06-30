# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.sph._coupling import SPHRigidBodyCoupling
from newton.solvers import SolverWCSPH, sph
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_fluid_block(device, *, dim=3, gravity=-9.81, height=0.5, ground_plane=False):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=gravity)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=20.0,
        viscosity=0.001,
        smoothing_length=0.16,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0, height, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        jitter=0.0,
        radius_mean=0.04,
    )
    if ground_plane:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _step_solver(solver, state_0, state_1, *, steps: int, dt: float):
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def _fluid_indices(model):
    roles = model.sph.role.numpy()
    return np.flatnonzero(roles == int(sph.SPHRole.FLUID))


def _fluid_mass(model):
    indices = _fluid_indices(model)
    return float(np.sum(model.particle_mass.numpy()[indices]))


def _max_particle_speed(state, indices):
    return float(np.max(np.linalg.norm(state.particle_qd.numpy()[indices], axis=1)))


def _build_two_particle_fluid(device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=10.0,
        viscosity=0.001,
        smoothing_length=0.18,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0, 0.5, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        jitter=0.0,
        radius_mean=0.04,
    )
    return builder.finalize(device=device)


def _build_tiny_dam_break(device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=16.0,
        viscosity=0.001,
        smoothing_length=0.16,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(-0.08, 0.08, -0.04),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.2, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))
    return model


def _run_tiny_dam_break(device):
    model = _build_tiny_dam_break(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model)
    fluid = _fluid_indices(model)
    initial_q = state_0.particle_q.numpy()[fluid]

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=5, dt=1.0e-4)

    final_q = state_0.particle_q.numpy()[fluid]
    return {
        "initial_x_max": float(np.max(initial_q[:, 0])),
        "final_x_max": float(np.max(final_q[:, 0])),
        "initial_x_extent": float(np.ptp(initial_q[:, 0])),
        "final_x_extent": float(np.ptp(final_q[:, 0])),
        "initial_min_height": float(np.min(initial_q[:, model.up_axis])),
        "final_min_height": float(np.min(final_q[:, model.up_axis])),
        "min_height": float(np.min(final_q[:, model.up_axis])),
        "max_speed": _max_particle_speed(state_0, fluid),
        "density_min": float(np.min(state_0.sph.density.numpy()[fluid])),
        "density_max": float(np.max(state_0.sph.density.numpy()[fluid])),
        "finite": bool(
            np.isfinite(final_q).all()
            and np.isfinite(state_0.particle_qd.numpy()[fluid]).all()
            and np.isfinite(state_0.sph.density.numpy()[fluid]).all()
        ),
    }


def _build_wave_propagation_block(device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=50.0,
        viscosity=0.5,
        smoothing_length=0.24,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(-0.12, 0.2, -0.03),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=4,
        dim_y=2,
        dim_z=2,
        cell_x=0.06,
        cell_y=0.06,
        cell_z=0.06,
        material=material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.03,
    )
    return builder.finalize(device=device)


def _run_wave_propagation(device):
    model = _build_wave_propagation_block(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model)
    fluid = _fluid_indices(model)
    initial_q = state_0.particle_q.numpy()
    left = fluid[initial_q[fluid, 0] <= np.min(initial_q[fluid, 0]) + 1.0e-6]
    right = fluid[initial_q[fluid, 0] >= np.max(initial_q[fluid, 0]) - 1.0e-6]
    initial_qd = state_0.particle_qd.numpy()
    initial_qd[left, 0] = 2.0
    state_0.particle_qd.assign(initial_qd)

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=100, dt=1.0e-4)

    q = state_0.particle_q.numpy()[fluid]
    qd = state_0.particle_qd.numpy()
    density = state_0.sph.density.numpy()[fluid]
    return {
        "left_mean_vx": float(np.mean(qd[left, 0])),
        "right_mean_vx": float(np.mean(qd[right, 0])),
        "right_max_vx": float(np.max(qd[right, 0])),
        "x_extent": float(np.ptp(q[:, 0])),
        "max_speed": _max_particle_speed(state_0, fluid),
        "density_min": float(np.min(density)),
        "density_max": float(np.max(density)),
        "finite": bool(np.isfinite(q).all() and np.isfinite(qd[fluid]).all() and np.isfinite(density).all()),
    }


def test_sph_wcsph_step_smoke(test, device):
    model = _build_fluid_block(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model)

    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    test.assertTrue(np.isfinite(state_1.particle_q.numpy()).all())
    test.assertTrue(np.isfinite(state_1.particle_qd.numpy()).all())
    test.assertTrue(np.isfinite(state_1.sph.density.numpy()).all())
    test.assertTrue(np.isfinite(state_1.sph.pressure.numpy()).all())
    test.assertTrue(np.all(state_1.sph.density.numpy() > 0.0))


def test_sph_step_rejects_invalid_timestep(test, device):
    model = _build_two_particle_fluid(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model)

    for dt in (0.0, -1.0e-4, float("nan")):
        with test.assertRaisesRegex(ValueError, "dt must be finite and positive"):
            solver.step(state_0, state_1, control=None, contacts=None, dt=dt)


def test_sph_rejects_unknown_config_option(test, device):
    model = _build_two_particle_fluid(device)

    with test.assertRaisesRegex(TypeError, "Unknown SolverWCSPH config option"):
        SolverWCSPH(model, method="delta_sph")


def test_sph_rejects_invalid_config_values(test, device):
    model = _build_two_particle_fluid(device)

    invalid_options = (
        {"smoothing_length": -0.1},
        {"smoothing_length": 0.0},
        {"rest_density": 0.0},
        {"sound_speed": -1.0},
        {"stiffness": -1.0},
        {"viscosity": -0.001},
        {"boundary_margin": -0.01},
        {"rest_density": np.bool_(True)},
        {"enable_shape_boundaries": np.bool_(True)},
    )
    for options in invalid_options:
        with test.assertRaises(ValueError):
            SolverWCSPH(model, **options)

    with test.assertRaises(ValueError):
        sph.SPHMaterial(rest_density=np.bool_(True)).validate()


def test_sph_fluid_block_on_plane_stays_bounded(test, device):
    model = _build_fluid_block(device, dim=3, height=0.02, ground_plane=True)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model)
    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=12, dt=1.0e-4)

    q = state_0.particle_q.numpy()
    qd = state_0.particle_qd.numpy()
    density = state_0.sph.density.numpy()[_fluid_indices(model)]
    bb_min = np.min(q, axis=0)
    bb_max = np.max(q, axis=0)

    test.assertTrue(np.isfinite(q).all())
    test.assertTrue(np.isfinite(qd).all())
    test.assertTrue(np.isfinite(density).all())
    test.assertTrue(np.all(density > 0.0))
    test.assertLess(float(np.max(np.linalg.norm(qd, axis=1))), 25.0)
    test.assertGreaterEqual(bb_min[model.up_axis], 0.04 - 1.0e-6)
    test.assertLess(bb_max[model.up_axis], 1.0)


def test_sph_fluid_block_under_gravity_moves_downward(test, device):
    model = _build_fluid_block(device, dim=2, gravity=-9.81)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model)
    fluid = _fluid_indices(model)
    initial_mean_height = float(np.mean(state_0.particle_q.numpy()[fluid, model.up_axis]))

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=8, dt=2.0e-4)

    final_q = state_0.particle_q.numpy()
    final_qd = state_0.particle_qd.numpy()
    final_mean_height = float(np.mean(final_q[fluid, model.up_axis]))
    test.assertLess(final_mean_height, initial_mean_height)
    test.assertGreater(float(np.max(np.linalg.norm(final_qd[fluid], axis=1))), 0.0)


def test_sph_fluid_particles_project_outside_ground_plane(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.14,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(-0.04, 0.02, -0.04),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, -1.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        jitter=0.0,
        radius_mean=0.04,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model, boundary_friction=0.05)

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=8, dt=2.0e-4)

    q = state_0.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertGreaterEqual(float(np.min(q[:, model.up_axis])), -1.0e-4)


def test_sph_project_outside_matches_mpm_style_projection(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0, 0.02, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, -1.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        jitter=0.0,
        radius_mean=0.04,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = SolverWCSPH(model, boundary_friction=0.0)

    solver.project_outside(state_in, state_out, dt=1.0e-4)

    particle_q = state_out.particle_q.numpy()
    particle_qd = state_out.particle_qd.numpy()
    test.assertTrue(np.isfinite(particle_q).all())
    test.assertGreaterEqual(float(particle_q[0, model.up_axis]), 0.04 - 1.0e-6)
    test.assertGreaterEqual(float(particle_qd[0, model.up_axis]), -1.0e-6)


def test_sph_density_volume_reconstructs_fluid_mass(test, device):
    model = _build_fluid_block(device, dim=3)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model)
    expected_mass = _fluid_mass(model)

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=10, dt=1.0e-4)

    fluid = _fluid_indices(model)
    density = state_0.sph.density.numpy()[fluid]
    volume = state_0.sph.volume.numpy()[fluid]
    reconstructed_mass = float(np.sum(density * volume))
    test.assertTrue(np.isfinite(density).all())
    test.assertTrue(np.isfinite(volume).all())
    test.assertAlmostEqual(reconstructed_mass, expected_mass, delta=max(1.0e-6, expected_mass * 1.0e-5))


def test_sph_surface_tension_feature_accelerates_free_surface(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.18,
        surface_tension=25.0,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(
        model,
        enable_surface_tension=True,
        surface_tension_normal_threshold=0.0,
    )

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=1, dt=1.0e-4)
    surface_acceleration = state_0.sph.surface_acceleration.numpy()
    normal = state_0.sph.normal.numpy()

    test.assertTrue(np.isfinite(surface_acceleration).all())
    test.assertGreater(float(np.max(np.linalg.norm(normal, axis=1))), 0.0)
    test.assertGreater(float(np.max(np.linalg.norm(surface_acceleration, axis=1))), 0.0)


def test_sph_sampled_boundary_wetting_and_adhesion_accelerate_fluid(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    fluid_material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.16,
        adhesion=8.0,
        wetting=8.0,
        contact_angle=0.0,
    )
    boundary_material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.16,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0, 0.04, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=fluid_material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    sph.add_sph_boundary_points(
        builder,
        points=(wp.vec3(0.0, 0.0, 0.0),),
        normals=(wp.vec3(0.0, 1.0, 0.0),),
        material=boundary_material,
        spacing=0.08,
        radius=0.04,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = SolverWCSPH(model, enable_boundary_adhesion=True, enable_boundary_wetting=True)

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=1, dt=1.0e-4)
    adhesion_acceleration = state_0.sph.adhesion_acceleration.numpy()
    wetting_acceleration = state_0.sph.wetting_acceleration.numpy()

    test.assertTrue(np.isfinite(adhesion_acceleration).all())
    test.assertTrue(np.isfinite(wetting_acceleration).all())
    test.assertGreater(float(np.max(np.linalg.norm(adhesion_acceleration, axis=1))), 0.0)
    test.assertGreater(float(np.max(np.linalg.norm(wetting_acceleration, axis=1))), 0.0)


def test_sph_external_body_collider_impulse_collection(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.16,
    )
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.05, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(-1.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    collider_state = collider_model.state()
    state_0.body_q = collider_state.body_q
    state_0.body_qd = collider_state.body_qd
    state_0.body_f = collider_state.body_f
    state_1.body_q = wp.empty_like(collider_state.body_q)
    state_1.body_qd = wp.empty_like(collider_state.body_qd)
    state_1.body_f = wp.empty_like(collider_state.body_f)

    solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
    solver.setup_collider(model=collider_model)
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    collider_body_index = solver.collider_body_index.numpy()
    impulses, impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)
    impulses_np = impulses.numpy()
    impulse_positions_np = impulse_positions.numpy()
    collider_ids_np = collider_ids.numpy()

    test.assertEqual(collider_body_index.tolist(), [body])
    test.assertEqual(collider_ids_np.tolist(), [0])
    test.assertTrue(np.isfinite(impulses_np).all())
    test.assertTrue(np.isfinite(impulse_positions_np).all())
    test.assertLess(float(impulses_np[0, 0]), 0.0)
    test.assertGreater(float(np.linalg.norm(impulses_np[0])), 0.0)


def test_sph_rigid_body_coupling_helper_applies_impulses(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.16,
    )
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.05, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(-1.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)

    fluid_state_0 = fluid_model.state()
    fluid_state_1 = fluid_model.state()
    collider_state = collider_model.state()
    solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
    coupling = SPHRigidBodyCoupling(collider_model, solver, collider_state, (fluid_state_0, fluid_state_1), 1.0e-4)

    solver.step(fluid_state_0, fluid_state_1, control=None, contacts=None, dt=1.0e-4)
    coupling.collect_impulses(fluid_state_1)
    body_impulse = coupling.collider_impulses.numpy()[0].copy()
    impulse_position = coupling.collider_impulse_positions.numpy()[0].copy()
    body_angular_impulse = np.cross(impulse_position, body_impulse)

    collider_state.clear_forces()
    coupling.apply_forces(collider_state)
    body_f = collider_state.body_f.numpy()

    test.assertGreater(coupling.max_collider_impulse_norm, 0.0)
    test.assertTrue(np.isfinite(body_f).all())
    test.assertLess(float(body_f[body, 0]), 0.0)
    test.assertGreater(float(np.linalg.norm(body_f[body, 0:3])), 0.0)
    np.testing.assert_allclose(body_f[body, 0:3] * 1.0e-4, body_impulse, rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(body_f[body, 3:6] * 1.0e-4, body_angular_impulse, rtol=1.0e-5, atol=1.0e-7)

    coupling.save_applied_forces(collider_state)
    coupling.update_fluid_state(collider_state, fluid_state_0)
    collider_qd = fluid_state_0.body_qd.numpy()
    test.assertGreater(float(collider_qd[body, 0]), 0.0)


def test_sph_offcenter_collider_impulse_preserves_torque(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.16,
    )
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.09, 0.05, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(-1.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    collider_state = collider_model.state()
    state_0.body_q = collider_state.body_q
    state_0.body_qd = collider_state.body_qd
    state_0.body_f = collider_state.body_f
    state_1.body_q = wp.empty_like(collider_state.body_q)
    state_1.body_qd = wp.empty_like(collider_state.body_qd)
    state_1.body_f = wp.empty_like(collider_state.body_f)

    solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
    solver.setup_collider(model=collider_model)
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    impulses, impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)
    impulse_np = impulses.numpy()[0]
    position_np = impulse_positions.numpy()[0]
    angular_impulse_np = solver._sph_model.boundary_handler.analytic_body_angular_impulse_wp.numpy()[body]
    com_np = collider_model.body_com.numpy()[body]
    reconstructed_angular = np.cross(position_np - com_np, impulse_np)

    test.assertEqual(collider_ids.numpy().tolist(), [0])
    test.assertGreater(float(np.linalg.norm(angular_impulse_np)), 0.0)
    test.assertGreater(float(np.linalg.norm(position_np - com_np)), 0.0)
    np.testing.assert_allclose(reconstructed_angular, angular_impulse_np, rtol=1.0e-5, atol=1.0e-7)


def test_sph_model_collider_material_overrides(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=12.0,
        viscosity=0.001,
        smoothing_length=0.16,
    )
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.13, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(-1.0, 1.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    collider_state = collider_model.state()
    state_0.body_q = collider_state.body_q
    state_0.body_qd = collider_state.body_qd
    state_0.body_f = collider_state.body_f
    state_1.body_q = wp.empty_like(collider_state.body_q)
    state_1.body_qd = wp.empty_like(collider_state.body_qd)
    state_1.body_f = wp.empty_like(collider_state.body_f)

    solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
    solver.setup_collider(
        model=collider_model,
        collider_body_ids=[body],
        collider_margins=[0.03],
        collider_friction=[1.0],
        collider_adhesion=[2.0],
        collider_projection_threshold=[0.02],
    )
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    particle_qd = state_1.particle_qd.numpy()
    shape_adhesion = solver._sph_model.boundary_handler.model_collider_shape_adhesion_wp.numpy()
    shape_projection_threshold = solver._sph_model.boundary_handler.model_collider_shape_projection_threshold_wp.numpy()

    test.assertGreaterEqual(float(particle_q[0, 0]), 0.19 - 1.0e-6)
    test.assertGreaterEqual(float(particle_qd[0, 0]), -1.0e-6)
    test.assertAlmostEqual(float(particle_qd[0, 1]), 0.0, delta=1.0e-6)
    test.assertTrue(np.any(shape_adhesion == 2.0))
    test.assertTrue(np.any(shape_projection_threshold == 0.02))


def test_sph_explicit_mesh_collider_material_options(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.0, 0.02, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, -1.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)
    state_0 = fluid_model.state()
    state_1 = fluid_model.state()

    vertices = np.array(
        (
            (-0.5, 0.0, -0.5),
            (0.5, 0.0, -0.5),
            (-0.5, 0.0, 0.5),
            (0.5, 0.0, 0.5),
        ),
        dtype=np.float32,
    )
    indices = np.array((0, 2, 1, 1, 2, 3), dtype=np.int32)
    mesh = newton.Mesh(vertices, indices, compute_inertia=False)

    solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
    solver.setup_collider(
        collider_meshes=[mesh],
        collider_margins=[0.01],
        collider_friction=[0.5],
        collider_adhesion=[1.5],
        collider_projection_threshold=[0.02],
    )
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    test.assertGreaterEqual(float(particle_q[0, 1]), 0.07 - 1.0e-6)
    test.assertEqual(solver._sph_model.boundary_handler.explicit_collider_adhesion, (1.5,))
    test.assertEqual(solver._sph_model.boundary_handler.explicit_collider_projection_threshold, (0.02,))


def test_sph_analytic_collider_adhesion_pulls_fluid_toward_shape(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.13, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    collider_state = collider_model.state()
    state_0.body_q = collider_state.body_q
    state_0.body_qd = collider_state.body_qd
    state_0.body_f = collider_state.body_f
    state_1.body_q = wp.empty_like(collider_state.body_q)
    state_1.body_qd = wp.empty_like(collider_state.body_qd)
    state_1.body_f = wp.empty_like(collider_state.body_f)

    solver = SolverWCSPH(fluid_model, boundary_friction=0.0, enable_boundary_adhesion=True)
    solver.setup_collider(model=collider_model, collider_body_ids=[body], collider_adhesion=[50.0])
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-3)

    particle_qd = state_1.particle_qd.numpy()

    test.assertTrue(np.isfinite(particle_qd).all())
    test.assertLess(float(particle_qd[0, 0]), -0.01)
    test.assertAlmostEqual(
        float(np.max(solver._sph_model.boundary_handler.model_collider_shape_adhesion_wp.numpy())), 50.0
    )


def test_sph_primitive_body_collider_impulse_collection(test, device):
    primitive_cases = (
        ("capsule", lambda builder, body: builder.add_shape_capsule(body=body, radius=0.1, half_height=0.12)),
        ("cylinder", lambda builder, body: builder.add_shape_cylinder(body=body, radius=0.1, half_height=0.12)),
        ("ellipsoid", lambda builder, body: builder.add_shape_ellipsoid(body=body, rx=0.1, ry=0.08, rz=0.12)),
        ("cone", lambda builder, body: builder.add_shape_cone(body=body, radius=0.14, half_height=0.12)),
    )

    for primitive_name, add_shape in primitive_cases:
        with test.subTest(primitive=primitive_name, device=device):
            fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
            material = sph.SPHMaterial(
                rest_density=1000.0,
                sound_speed=12.0,
                viscosity=0.001,
                smoothing_length=0.16,
            )
            sph.add_sph_particle_grid(
                fluid_builder,
                pos=wp.vec3(0.03, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(-1.0, 0.0, 0.0),
                dim_x=1,
                dim_y=1,
                dim_z=1,
                cell_x=0.08,
                cell_y=0.08,
                cell_z=0.08,
                material=material,
                role=sph.SPHRole.FLUID,
                jitter=0.0,
                radius_mean=0.04,
            )
            fluid_model = fluid_builder.finalize(device=device)

            collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
            body = collider_builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.diag(wp.vec3(0.01)),
            )
            add_shape(collider_builder, body)
            collider_model = collider_builder.finalize(device=device)

            state_0 = fluid_model.state()
            state_1 = fluid_model.state()
            collider_state = collider_model.state()
            state_0.body_q = collider_state.body_q
            state_0.body_qd = collider_state.body_qd
            state_0.body_f = collider_state.body_f
            state_1.body_q = wp.empty_like(collider_state.body_q)
            state_1.body_qd = wp.empty_like(collider_state.body_qd)
            state_1.body_f = wp.empty_like(collider_state.body_f)

            solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
            solver.setup_collider(model=collider_model)
            solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

            impulses, impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)
            impulses_np = impulses.numpy()
            impulse_positions_np = impulse_positions.numpy()
            collider_ids_np = collider_ids.numpy()

            test.assertEqual(collider_ids_np.tolist(), [0])
            test.assertTrue(np.isfinite(impulses_np).all())
            test.assertTrue(np.isfinite(impulse_positions_np).all())
            test.assertLess(float(impulses_np[0, 0]), 0.0)
            test.assertGreater(float(np.linalg.norm(impulses_np[0])), 0.0)


def test_sph_moving_external_collider_velocity_moves_fluid(test, device):
    def run(collider_velocity: float) -> np.ndarray:
        fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        sph.add_sph_particle_grid(
            fluid_builder,
            pos=wp.vec3(0.13, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.08,
            cell_y=0.08,
            cell_z=0.08,
            material=sph.SPHMaterial(smoothing_length=0.16),
            role=sph.SPHRole.FLUID,
            jitter=0.0,
            radius_mean=0.04,
        )
        fluid_model = fluid_builder.finalize(device=device)

        collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        body = collider_builder.add_body(
            xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.diag(wp.vec3(0.01)),
        )
        collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        collider_model = collider_builder.finalize(device=device)

        state_0 = fluid_model.state()
        state_1 = fluid_model.state()
        collider_state = collider_model.state()
        body_qd = collider_state.body_qd.numpy()
        body_qd[body] = np.array((collider_velocity, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
        collider_state.body_qd.assign(body_qd)
        state_0.body_q = collider_state.body_q
        state_0.body_qd = collider_state.body_qd
        state_0.body_f = collider_state.body_f
        state_1.body_q = wp.empty_like(collider_state.body_q)
        state_1.body_qd = wp.empty_like(collider_state.body_qd)
        state_1.body_f = wp.empty_like(collider_state.body_f)

        solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
        solver.setup_collider(model=collider_model)
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)
        return state_1.particle_qd.numpy()[0]

    stationary_velocity = run(0.0)
    moving_velocity = run(1.0)
    test.assertAlmostEqual(float(stationary_velocity[0]), 0.0, delta=1.0e-6)
    test.assertGreater(float(moving_velocity[0] - stationary_velocity[0]), 0.9)
    np.testing.assert_allclose(moving_velocity[1:], stationary_velocity[1:], atol=1.0e-6)


def test_sph_backward_collider_velocity_uses_body_pose_delta(test, device):
    def run_simulation(collider_velocity_mode: str) -> float:
        fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        material = sph.SPHMaterial(
            rest_density=1000.0,
            sound_speed=12.0,
            viscosity=0.001,
            smoothing_length=0.16,
        )
        sph.add_sph_particle_grid(
            fluid_builder,
            pos=wp.vec3(0.05, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.08,
            cell_y=0.08,
            cell_z=0.08,
            material=material,
            role=sph.SPHRole.FLUID,
            jitter=0.0,
            radius_mean=0.04,
        )
        fluid_model = fluid_builder.finalize(device=device)

        collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        body = collider_builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.diag(wp.vec3(0.01)),
        )
        collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        collider_model = collider_builder.finalize(device=device)

        state_0 = fluid_model.state()
        state_1 = fluid_model.state()
        collider_state = collider_model.state()
        state_0.body_q = collider_state.body_q
        state_0.body_qd = collider_state.body_qd
        state_0.body_f = collider_state.body_f
        state_1.body_q = wp.empty_like(collider_state.body_q)
        state_1.body_qd = wp.empty_like(collider_state.body_qd)
        state_1.body_f = wp.empty_like(collider_state.body_f)

        solver = SolverWCSPH(
            fluid_model,
            boundary_friction=0.0,
            collider_velocity_mode=collider_velocity_mode,
        )
        solver.setup_collider(model=collider_model)

        dt = 1.0e-4
        body_q = collider_state.body_q.numpy()
        body_q[body, 0] = dt
        collider_state.body_q.assign(body_q)
        body_qd = collider_state.body_qd.numpy()
        body_qd[body] = 0.0
        collider_state.body_qd.assign(body_qd)
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        return float(state_1.particle_qd.numpy()[0, 0])

    forward_velocity = run_simulation("forward")
    backward_velocity = run_simulation("backward")

    test.assertAlmostEqual(forward_velocity, 0.0, delta=1.0e-6)
    test.assertGreater(backward_velocity, 0.9)


def test_sph_setup_collider_accepts_kinematic_body_arrays(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.05, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)

    solver = SolverWCSPH(fluid_model)
    solver.setup_collider(
        model=collider_model,
        body_mass=wp.zeros_like(collider_model.body_mass),
        body_inv_inertia=wp.zeros_like(collider_model.body_inv_inertia),
        body_q=collider_model.body_q,
    )

    test.assertEqual(solver.collider_body_index.numpy().tolist(), [body])
    test.assertIs(solver._sph_model.collider_body_q, collider_model.body_q)


def test_sph_setup_collider_rejects_mismatched_array_device(test, device):
    if not getattr(device, "is_cuda", False):
        test.skipTest("A CUDA solver device is required for a device-mismatch check")

    fluid_model = _build_fluid_block(device, dim=1, gravity=0.0)
    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)
    cpu_overrides = {
        "body_com": wp.zeros(collider_model.body_count, dtype=wp.vec3, device="cpu"),
        "body_mass": wp.zeros(collider_model.body_count, dtype=float, device="cpu"),
        "body_inv_inertia": wp.zeros(collider_model.body_count, dtype=wp.mat33, device="cpu"),
        "body_q": wp.zeros(collider_model.body_count, dtype=wp.transform, device="cpu"),
    }

    solver = SolverWCSPH(fluid_model)
    for name, value in cpu_overrides.items():
        with test.subTest(name=name), test.assertRaisesRegex(ValueError, f"{name} must be allocated"):
            solver.setup_collider(model=collider_model, **{name: value})


def test_sph_mesh_body_collider_impulse_collection(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.0, 0.02, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, -1.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    vertices = np.array(
        (
            (-0.5, 0.0, -0.5),
            (0.5, 0.0, -0.5),
            (-0.5, 0.0, 0.5),
            (0.5, 0.0, 0.5),
        ),
        dtype=np.float32,
    )
    indices = np.array((0, 2, 1, 1, 2, 3), dtype=np.int32)
    collider_builder.add_shape_mesh(body=body, mesh=newton.Mesh(vertices, indices, compute_inertia=False))
    collider_model = collider_builder.finalize(device=device)

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    collider_state = collider_model.state()
    state_0.body_q = collider_state.body_q
    state_0.body_qd = collider_state.body_qd
    state_0.body_f = collider_state.body_f
    state_1.body_q = wp.empty_like(collider_state.body_q)
    state_1.body_qd = wp.empty_like(collider_state.body_qd)
    state_1.body_f = wp.empty_like(collider_state.body_f)

    solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
    solver.setup_collider(model=collider_model)
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    particle_qd = state_1.particle_qd.numpy()
    impulses, _impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)

    test.assertGreaterEqual(float(particle_q[0, 1]), 0.04 - 1.0e-6)
    test.assertGreaterEqual(float(particle_qd[0, 1]), -1.0e-6)
    test.assertEqual(solver.collider_body_index.numpy().tolist(), [body])
    test.assertEqual(collider_ids.numpy().tolist(), [0])
    test.assertLess(float(impulses.numpy()[0, 1]), 0.0)


def test_sph_explicit_mesh_collider_projects_fluid(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.0, 0.02, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, -1.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)
    state_0 = fluid_model.state()
    state_1 = fluid_model.state()

    vertices = np.array(
        (
            (-0.5, 0.0, -0.5),
            (0.5, 0.0, -0.5),
            (-0.5, 0.0, 0.5),
            (0.5, 0.0, 0.5),
        ),
        dtype=np.float32,
    )
    indices = np.array((0, 2, 1, 1, 2, 3), dtype=np.int32)
    mesh = newton.Mesh(vertices, indices, compute_inertia=False)

    solver = SolverWCSPH(fluid_model, boundary_friction=0.0)
    solver.setup_collider(collider_meshes=[mesh])
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    particle_qd = state_1.particle_qd.numpy()
    impulses, _impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)

    test.assertEqual(solver.collider_body_index.numpy().tolist(), [-1])
    test.assertEqual(solver._sph_model.boundary_handler.analytic_shape_count(), 0)
    test.assertEqual(solver._sph_model.boundary_handler.explicit_collider_mesh_count(), 1)
    test.assertGreaterEqual(float(particle_q[0, 1]), 0.04 - 1.0e-6)
    test.assertGreaterEqual(float(particle_qd[0, 1]), -1.0e-6)
    test.assertEqual(impulses.shape[0], 0)
    test.assertEqual(collider_ids.shape[0], 0)


def test_sph_dynamic_explicit_mesh_collider_moves_with_body(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.0, 0.02, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        role=sph.SPHRole.FLUID,
        jitter=0.0,
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)
    mesh = newton.Mesh(
        np.array(((-0.5, 0.0, -0.5), (0.5, 0.0, -0.5), (-0.5, 0.0, 0.5)), dtype=np.float32),
        np.array((0, 2, 1), dtype=np.int32),
        compute_inertia=False,
    )

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_model = collider_builder.finalize(device=device)

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    collider_state = collider_model.state()
    body_qd = collider_state.body_qd.numpy()
    body_qd[body] = np.array((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    collider_state.body_qd.assign(body_qd)
    state_0.body_q = collider_state.body_q
    state_0.body_qd = collider_state.body_qd
    state_0.body_f = collider_state.body_f
    state_1.body_q = wp.empty_like(collider_state.body_q)
    state_1.body_qd = wp.empty_like(collider_state.body_qd)
    state_1.body_f = wp.empty_like(collider_state.body_f)

    solver = SolverWCSPH(fluid_model, boundary_friction=1.0)
    solver.setup_collider(model=collider_model, collider_meshes=[mesh, mesh], collider_body_ids=[body, body])
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    particle_qd = state_1.particle_qd.numpy()
    impulses, _impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)

    test.assertEqual(solver.collider_body_index.numpy().tolist(), [body, body])
    test.assertEqual(solver._sph_model.boundary_handler.explicit_collider_mesh_count(), 2)
    test.assertEqual(
        np.count_nonzero(np.asarray(solver._sph_model.boundary_handler.explicit_collider_body_ids) >= 0), 2
    )
    test.assertGreaterEqual(float(particle_q[0, 1]), 0.04 - 1.0e-6)
    test.assertGreater(float(particle_qd[0, 0]), 0.9)
    test.assertEqual(collider_ids.numpy().tolist(), [0])
    test.assertEqual(impulses.shape[0], 1)
    test.assertLess(float(impulses.numpy()[0, 0]), 0.0)


def test_sph_dam_break_tiny_cpu_case(test, device):
    metrics = _run_tiny_dam_break(device)

    test.assertTrue(metrics["finite"])
    test.assertGreater(metrics["max_speed"], 0.0)
    test.assertGreater(metrics["final_x_max"], metrics["initial_x_max"] + 5.0e-5)
    test.assertGreaterEqual(metrics["final_x_extent"], metrics["initial_x_extent"] - 1.0e-6)
    test.assertLess(metrics["final_min_height"], metrics["initial_min_height"])
    test.assertGreaterEqual(metrics["min_height"], -1.0e-4)
    test.assertGreater(metrics["density_min"], 0.0)


def test_sph_wave_disturbance_propagates_across_block(test, device):
    metrics = _run_wave_propagation(device)

    test.assertTrue(metrics["finite"])
    test.assertGreater(metrics["right_mean_vx"], 0.05)
    test.assertGreaterEqual(metrics["right_max_vx"], metrics["right_mean_vx"])
    test.assertLess(metrics["left_mean_vx"], 2.0)
    test.assertGreater(metrics["max_speed"], 0.1)
    test.assertGreater(metrics["density_min"], 0.0)


def test_sph_tier1_cpu_cuda_parity(test, device):
    if not wp.is_cuda_available():
        test.skipTest("CUDA device is not available")

    for run_scenario in (_run_tiny_dam_break, _run_wave_propagation):
        cpu_metrics = run_scenario("cpu")
        cuda_metrics = run_scenario("cuda:0")

        test.assertEqual(cpu_metrics["finite"], cuda_metrics["finite"])
        test.assertTrue(cuda_metrics["finite"])
        for key in ("max_speed", "density_min", "density_max"):
            test.assertAlmostEqual(cuda_metrics[key], cpu_metrics[key], delta=max(1.0e-3, abs(cpu_metrics[key]) * 0.2))

        if run_scenario is _run_tiny_dam_break:
            test.assertGreater(cuda_metrics["final_x_max"], cuda_metrics["initial_x_max"] + 5.0e-5)
            test.assertLess(cuda_metrics["final_min_height"], cuda_metrics["initial_min_height"])
        else:
            test.assertGreater(cuda_metrics["right_mean_vx"], 0.05)


devices = get_test_devices(mode="basic")


class TestSolverWCSPH(unittest.TestCase):
    """Focused WCSPH solver tests."""


for test_func in (
    test_sph_wcsph_step_smoke,
    test_sph_step_rejects_invalid_timestep,
    test_sph_rejects_unknown_config_option,
    test_sph_rejects_invalid_config_values,
    test_sph_fluid_block_on_plane_stays_bounded,
    test_sph_fluid_block_under_gravity_moves_downward,
    test_sph_fluid_particles_project_outside_ground_plane,
    test_sph_project_outside_matches_mpm_style_projection,
    test_sph_density_volume_reconstructs_fluid_mass,
):
    add_function_test(TestSolverWCSPH, test_func.__name__, test_func, devices=devices)

add_function_test(
    TestSolverWCSPH,
    test_sph_surface_tension_feature_accelerates_free_surface.__name__,
    test_sph_surface_tension_feature_accelerates_free_surface,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_sampled_boundary_wetting_and_adhesion_accelerate_fluid.__name__,
    test_sph_sampled_boundary_wetting_and_adhesion_accelerate_fluid,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_external_body_collider_impulse_collection.__name__,
    test_sph_external_body_collider_impulse_collection,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_rigid_body_coupling_helper_applies_impulses.__name__,
    test_sph_rigid_body_coupling_helper_applies_impulses,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_offcenter_collider_impulse_preserves_torque.__name__,
    test_sph_offcenter_collider_impulse_preserves_torque,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_model_collider_material_overrides.__name__,
    test_sph_model_collider_material_overrides,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_primitive_body_collider_impulse_collection.__name__,
    test_sph_primitive_body_collider_impulse_collection,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_moving_external_collider_velocity_moves_fluid.__name__,
    test_sph_moving_external_collider_velocity_moves_fluid,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_backward_collider_velocity_uses_body_pose_delta.__name__,
    test_sph_backward_collider_velocity_uses_body_pose_delta,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_setup_collider_accepts_kinematic_body_arrays.__name__,
    test_sph_setup_collider_accepts_kinematic_body_arrays,
    devices=wp.get_device("cpu"),
)

add_function_test(
    TestSolverWCSPH,
    test_sph_setup_collider_rejects_mismatched_array_device.__name__,
    test_sph_setup_collider_rejects_mismatched_array_device,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_mesh_body_collider_impulse_collection.__name__,
    test_sph_mesh_body_collider_impulse_collection,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_explicit_mesh_collider_projects_fluid.__name__,
    test_sph_explicit_mesh_collider_projects_fluid,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_explicit_mesh_collider_material_options.__name__,
    test_sph_explicit_mesh_collider_material_options,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_analytic_collider_adhesion_pulls_fluid_toward_shape.__name__,
    test_sph_analytic_collider_adhesion_pulls_fluid_toward_shape,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_dynamic_explicit_mesh_collider_moves_with_body.__name__,
    test_sph_dynamic_explicit_mesh_collider_moves_with_body,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_dam_break_tiny_cpu_case.__name__,
    test_sph_dam_break_tiny_cpu_case,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_wave_disturbance_propagates_across_block.__name__,
    test_sph_wave_disturbance_propagates_across_block,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_tier1_cpu_cuda_parity.__name__,
    test_sph_tier1_cpu_cuda_parity,
    devices=wp.get_device("cpu"),
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
