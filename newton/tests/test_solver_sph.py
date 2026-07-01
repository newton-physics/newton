# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.sph._coupling import SPHRigidBodyCoupling
from newton.solvers import SolverWCSPH, sph
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_solver(model, **config):
    return SolverWCSPH(model, SolverWCSPH.Config(**config))


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
    return np.arange(model.particle_count, dtype=np.int32)


def _max_particle_speed(state, indices):
    return float(np.max(np.linalg.norm(state.particle_qd.numpy()[indices], axis=1)))


def _collider_wrench(impulses, positions, center):
    impulses = np.asarray(impulses)
    positions = np.asarray(positions)
    linear_impulse = np.sum(impulses, axis=0)
    angular_impulse = np.sum(np.cross(positions - center, impulses), axis=0)
    return linear_impulse, angular_impulse


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
        sound_speed=20.0,
        viscosity=0.001,
        smoothing_length=0.12,
    )
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(-0.06, 0.03, -0.03),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=3,
        dim_y=6,
        dim_z=2,
        cell_x=0.06,
        cell_y=0.06,
        cell_z=0.06,
        material=material,
        jitter=0.0,
        radius_mean=0.03,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))
    return model


def _run_tiny_dam_break(device):
    model = _build_tiny_dam_break(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = _make_solver(model)
    fluid = _fluid_indices(model)
    initial_q = state_0.particle_q.numpy()[fluid]
    initial_qd = state_0.particle_qd.numpy()[fluid]

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=1000, dt=2.0e-4)

    final_q = state_0.particle_q.numpy()[fluid]
    return {
        "initial_x_max": float(np.max(initial_q[:, 0])),
        "final_x_max": float(np.max(final_q[:, 0])),
        "initial_x_extent": float(np.ptp(initial_q[:, 0])),
        "final_x_extent": float(np.ptp(final_q[:, 0])),
        "initial_min_height": float(np.min(initial_q[:, model.up_axis])),
        "final_min_height": float(np.min(final_q[:, model.up_axis])),
        "initial_mean_height": float(np.mean(initial_q[:, model.up_axis])),
        "final_mean_height": float(np.mean(final_q[:, model.up_axis])),
        "initial_max_speed": float(np.max(np.linalg.norm(initial_qd, axis=1))),
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
        jitter=0.0,
        radius_mean=0.03,
    )
    return builder.finalize(device=device)


def _run_wave_propagation(device):
    model = _build_wave_propagation_block(device)
    state_0 = model.state()
    state_1 = model.state()
    solver = _make_solver(model)
    fluid = _fluid_indices(model)
    initial_q = state_0.particle_q.numpy()
    mass = model.particle_mass.numpy()
    support = 0.24
    coefficient = 315.0 / (64.0 * np.pi * support**9)
    equilibrium_density = np.zeros(model.particle_count, dtype=np.float32)
    for i in fluid:
        for j in fluid:
            distance = float(np.linalg.norm(initial_q[i] - initial_q[j]))
            if distance < support:
                equilibrium_density[i] += mass[j] * coefficient * (support * support - distance * distance) ** 3
    model.sph.rest_density.assign(equilibrium_density)

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
    solver = _make_solver(model)

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
    solver = _make_solver(model)

    for dt in (0.0, -1.0e-4, float("nan")):
        with test.assertRaisesRegex(ValueError, "dt must be finite and positive"):
            solver.step(state_0, state_1, control=None, contacts=None, dt=dt)


def test_sph_rejects_unknown_config_option(test, device):
    model = _build_two_particle_fluid(device)

    with test.assertRaisesRegex(TypeError, "unexpected keyword argument 'method'"):
        _make_solver(model, method="delta_sph")


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
            _make_solver(model, **options)

    with test.assertRaises(ValueError):
        sph.SPHMaterial(rest_density=np.bool_(True)).validate()


def test_sph_default_grid_builds_interacting_neighborhood(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    indices = sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
    )
    model = builder.finalize(device=device)
    solver = _make_solver(model)
    state_0 = model.state()
    state_1 = model.state()

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    support = solver._sph_model.default_support_radius
    mass = model.particle_mass.numpy()[indices.start]
    self_density = mass * 315.0 / (64.0 * np.pi * support**3)
    density = state_0.sph.density.numpy()[list(indices)]
    test.assertGreater(support, 0.1)
    test.assertTrue(np.all(density > self_density))


def test_sph_config_material_overrides_are_applied(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        material=sph.SPHMaterial(
            rest_density=1000.0,
            sound_speed=20.0,
            pressure_min=-1.0e9,
            smoothing_length=0.2,
        ),
    )
    model = builder.finalize(device=device)
    solver = _make_solver(model, rest_density=750.0, sound_speed=7.0, pressure_exponent=1.0, viscosity=0.25)
    state_0 = model.state()
    state_1 = model.state()

    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    density = state_0.sph.density.numpy()
    expected_pressure = 7.0**2 * (density - 750.0)
    np.testing.assert_allclose(state_0.sph.pressure.numpy(), expected_pressure, rtol=1.0e-5, atol=1.0e-3)
    test.assertEqual(solver.config.viscosity, 0.25)


def test_sph_per_particle_support_uses_solver_fallback(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
    )
    model = builder.finalize(device=device)
    model.sph.smoothing_length.assign(np.array((0.0, 0.2), dtype=np.float32))
    solver = _make_solver(model, smoothing_length=0.1)

    support = model.sph.smoothing_length.numpy()
    support = np.where(support > 0.0, support, solver._sph_model.default_support_radius)
    np.testing.assert_allclose(support, np.array((0.1, 0.2), dtype=np.float32))
    test.assertAlmostEqual(solver._sph_model.max_support_radius, 0.2)

    model.sph.smoothing_length.assign(np.array((0.0, 0.05), dtype=np.float32))
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    test.assertAlmostEqual(solver._sph_model.max_support_radius, 0.1)


def test_sph_coulomb_friction_requires_normal_impulse(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.14, 0.0, 0.0),
        vel=wp.vec3(0.0, 1.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(smoothing_length=0.16),
        radius_mean=0.04,
    )
    builder.add_shape_box(
        body=-1,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(mu=1.0, margin=0.0),
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()

    _make_solver(model, boundary_friction=1.0).step(state_0, state_1, None, None, 1.0e-4)

    np.testing.assert_allclose(state_1.particle_qd.numpy()[0], np.array((0.0, 1.0, 0.0)), atol=1.0e-6)


def test_sph_finite_plane_does_not_project_outside_extent(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    material = sph.SPHMaterial(smoothing_length=0.08)
    for position in (wp.vec3(0.0, 0.0, -0.01), wp.vec3(1.0, 0.0, -0.01)):
        sph.add_sph_particle_grid(
            builder,
            pos=position,
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.04,
            cell_y=0.04,
            cell_z=0.04,
            material=material,
            radius_mean=0.02,
        )
    builder.add_shape_plane(
        body=-1,
        width=0.2,
        length=0.2,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0),
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()

    _make_solver(model).step(state_0, state_1, None, None, 1.0e-4)

    positions = state_1.particle_q.numpy()
    test.assertGreaterEqual(float(positions[0, 2]), 0.02 - 1.0e-6)
    test.assertAlmostEqual(float(positions[1, 2]), -0.01, delta=1.0e-6)


def test_sph_closed_mesh_projects_toward_exterior(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.09, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.04,
        cell_y=0.04,
        cell_z=0.04,
        material=sph.SPHMaterial(smoothing_length=0.08),
        radius_mean=0.02,
    )
    builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh.create_box(0.1, 0.1, 0.1, compute_inertia=False),
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0),
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()

    _make_solver(model).step(state_0, state_1, None, None, 1.0e-4)

    test.assertGreaterEqual(float(state_1.particle_q.numpy()[0, 0]), 0.12 - 1.0e-6)


def test_sph_fluid_block_on_plane_stays_bounded(test, device):
    model = _build_fluid_block(device, dim=3, height=0.02, ground_plane=True)
    state_0 = model.state()
    state_1 = model.state()
    solver = _make_solver(model)
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


def test_sph_hydrostatic_pressure_increases_with_depth(test, device):
    spacing = 0.04
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(-0.06, 0.04, -0.06),
        dim_x=4,
        dim_y=8,
        dim_z=4,
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        material=sph.SPHMaterial(
            sound_speed=20.0,
            viscosity=0.01,
            smoothing_length=2.0 * spacing,
        ),
        radius_mean=0.5 * spacing,
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.05, is_visible=False)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        hx=0.12,
        hy=0.02,
        hz=0.12,
        cfg=shape_cfg,
    )
    for x in (-0.1, 0.1):
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(x, 0.18, 0.0), wp.quat_identity()),
            hx=0.02,
            hy=0.18,
            hz=0.12,
            cfg=shape_cfg,
        )
    for z in (-0.1, 0.1):
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.18, z), wp.quat_identity()),
            hx=0.12,
            hy=0.18,
            hz=0.02,
            cfg=shape_cfg,
        )

    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = _make_solver(model, kernel="wendland", xsph=0.03)
    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=800, dt=5.0e-4)

    position = state_0.particle_q.numpy()
    pressure = state_0.sph.pressure.numpy()
    lower_height = np.percentile(position[:, 1], 30.0)
    upper_height = np.percentile(position[:, 1], 70.0)
    lower_pressure = float(np.mean(pressure[position[:, 1] <= lower_height]))
    upper_pressure = float(np.mean(pressure[position[:, 1] >= upper_height]))

    test.assertGreater(lower_pressure, 2.0 * upper_pressure)
    test.assertLess(_max_particle_speed(state_0, _fluid_indices(model)), 0.2)
    test.assertGreaterEqual(float(np.min(position[:, 1])), -1.0e-4)


def test_sph_fluid_block_under_gravity_moves_downward(test, device):
    model = _build_fluid_block(device, dim=2, gravity=-9.81)
    state_0 = model.state()
    state_1 = model.state()
    solver = _make_solver(model)
    fluid = _fluid_indices(model)
    initial_mean_height = float(np.mean(state_0.particle_q.numpy()[fluid, model.up_axis]))

    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=8, dt=2.0e-4)

    final_q = state_0.particle_q.numpy()
    final_qd = state_0.particle_qd.numpy()
    final_mean_height = float(np.mean(final_q[fluid, model.up_axis]))
    test.assertLess(final_mean_height, initial_mean_height)
    test.assertGreater(float(np.max(np.linalg.norm(final_qd[fluid], axis=1))), 0.0)
    expected_mean_velocity = -9.81 * 8 * 2.0e-4
    test.assertAlmostEqual(float(np.mean(final_qd[fluid, model.up_axis])), expected_mean_velocity, delta=2.0e-5)


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
    solver = _make_solver(model, boundary_friction=0.05)

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
    solver = _make_solver(model, boundary_friction=0.0)

    solver.project_outside(state_in, state_out, dt=1.0e-4)

    particle_q = state_out.particle_q.numpy()
    particle_qd = state_out.particle_qd.numpy()
    test.assertTrue(np.isfinite(particle_q).all())
    test.assertGreaterEqual(float(particle_q[0, model.up_axis]), 0.04 - 1.0e-6)
    test.assertGreaterEqual(float(particle_qd[0, model.up_axis]), -1.0e-6)


def test_sph_internal_forces_preserve_linear_and_angular_momentum(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(-0.08, -0.08, -0.08),
        dim_x=3,
        dim_y=3,
        dim_z=3,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(
            rest_density=500.0,
            sound_speed=12.0,
            viscosity=0.0,
            smoothing_length=0.18,
        ),
        radius_mean=0.04,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = _make_solver(model)
    mass = model.particle_mass.numpy()
    rng = np.random.default_rng(1234)
    velocity = rng.normal(0.0, 0.1, size=(model.particle_count, 3)).astype(np.float32)
    velocity -= np.sum(mass[:, None] * velocity, axis=0) / np.sum(mass)
    state_0.particle_qd.assign(velocity)

    def momenta(state):
        position = state.particle_q.numpy()
        particle_momentum = mass[:, None] * state.particle_qd.numpy()
        return np.sum(particle_momentum, axis=0), np.sum(np.cross(position, particle_momentum), axis=0)

    initial_linear, initial_angular = momenta(state_0)
    initial_center = np.sum(mass[:, None] * state_0.particle_q.numpy(), axis=0) / np.sum(mass)
    steps = 25
    dt = 1.0e-5
    state_0, _state_1 = _step_solver(solver, state_0, state_1, steps=steps, dt=dt)
    final_linear, final_angular = momenta(state_0)
    final_center = np.sum(mass[:, None] * state_0.particle_q.numpy(), axis=0) / np.sum(mass)

    np.testing.assert_allclose(final_linear, initial_linear, rtol=2.0e-4, atol=2.0e-6)
    np.testing.assert_allclose(final_angular, initial_angular, rtol=2.0e-4, atol=2.0e-6)
    expected_center = initial_center + steps * dt * initial_linear / np.sum(mass)
    np.testing.assert_allclose(final_center, expected_center, rtol=2.0e-5, atol=2.0e-6)


def test_sph_pressure_separates_compressed_pair(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(-0.04, 0.0, 0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(
            rest_density=100.0,
            sound_speed=10.0,
            viscosity=0.0,
            smoothing_length=0.16,
        ),
        mass=0.5,
        radius_mean=0.04,
    )
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()

    _make_solver(model).step(state_0, state_1, None, None, 1.0e-4)

    velocity = state_1.particle_qd.numpy()
    momentum = np.sum(model.particle_mass.numpy()[:, None] * velocity, axis=0)
    test.assertLess(float(velocity[0, 0]), 0.0)
    test.assertGreater(float(velocity[1, 0]), 0.0)
    np.testing.assert_allclose(momentum, np.zeros(3), atol=1.0e-6)


def test_sph_viscosity_dissipates_energy_without_changing_momentum(test, device):
    def run(viscosity: float):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        sph.add_sph_particle_grid(
            builder,
            pos=wp.vec3(-0.04, 0.0, 0.0),
            dim_x=2,
            dim_y=1,
            dim_z=1,
            cell_x=0.08,
            cell_y=0.08,
            cell_z=0.08,
            material=sph.SPHMaterial(
                sound_speed=0.0,
                stiffness=0.0,
                viscosity=0.0,
                smoothing_length=0.16,
            ),
            radius_mean=0.04,
        )
        model = builder.finalize(device=device)
        state_0 = model.state()
        state_1 = model.state()
        mass = np.array((0.5, 1.0), dtype=np.float32)
        model.particle_mass.assign(mass)
        model.particle_inv_mass.assign(1.0 / mass)
        state_0.particle_qd.assign(np.array(((0.0, 1.0, 0.0), (0.0, -0.5, 0.0)), dtype=np.float32))
        _make_solver(model, viscosity=viscosity).step(state_0, state_1, None, None, 1.0e-4)
        velocity = state_1.particle_qd.numpy()
        energy = 0.5 * np.sum(mass * np.sum(velocity**2, axis=1))
        momentum = np.sum(mass[:, None] * velocity, axis=0)
        return float(energy), momentum

    inviscid_energy, inviscid_momentum = run(0.0)
    viscous_energy, viscous_momentum = run(100.0)

    test.assertLess(viscous_energy, inviscid_energy)
    np.testing.assert_allclose(inviscid_momentum, np.zeros(3), atol=1.0e-6)
    np.testing.assert_allclose(viscous_momentum, inviscid_momentum, atol=1.0e-6)


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

    solver = _make_solver(fluid_model, boundary_friction=0.0)
    solver.setup_collider(model=collider_model)
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    collider_body_index = solver.collider_body_index.numpy()
    impulses, impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)
    impulses_np = impulses.numpy()
    impulse_positions_np = impulse_positions.numpy()
    collider_ids_np = collider_ids.numpy()
    total_impulse = np.sum(impulses_np, axis=0)

    test.assertEqual(collider_body_index.tolist(), [body])
    test.assertEqual(collider_ids_np.tolist(), [0, 0])
    test.assertTrue(np.isfinite(impulses_np).all())
    test.assertTrue(np.isfinite(impulse_positions_np).all())
    test.assertLess(float(total_impulse[0]), 0.0)
    test.assertGreater(float(np.linalg.norm(total_impulse)), 0.0)


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
    solver = _make_solver(fluid_model, boundary_friction=0.0)
    coupling = SPHRigidBodyCoupling(collider_model, solver, collider_state, (fluid_state_0, fluid_state_1), 1.0e-4)

    solver.step(fluid_state_0, fluid_state_1, control=None, contacts=None, dt=1.0e-4)
    coupling.collect_impulses(fluid_state_1)
    body_impulse, body_angular_impulse = _collider_wrench(
        coupling.collider_impulses.numpy(),
        coupling.collider_impulse_positions.numpy(),
        collider_model.body_com.numpy()[body],
    )

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

    solver = _make_solver(fluid_model, boundary_friction=0.0)
    solver.setup_collider(model=collider_model)
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    impulses, impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)
    impulse_np = impulses.numpy()
    position_np = impulse_positions.numpy()
    angular_impulse_np = solver._sph_model.boundary_handler.analytic_body_angular_impulse_wp.numpy()[body]
    linear_impulse_np = solver._sph_model.boundary_handler.analytic_body_impulse_wp.numpy()[body]
    com_np = collider_model.body_com.numpy()[body]
    reconstructed_linear, reconstructed_angular = _collider_wrench(impulse_np, position_np, com_np)

    test.assertEqual(collider_ids.numpy().tolist(), [0, 0])
    test.assertGreater(float(np.linalg.norm(angular_impulse_np)), 0.0)
    test.assertGreater(float(np.linalg.norm(position_np - com_np)), 0.0)
    np.testing.assert_allclose(reconstructed_linear, linear_impulse_np, rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(reconstructed_angular, angular_impulse_np, rtol=1.0e-5, atol=1.0e-7)


def test_sph_collider_impulse_collection_preserves_pure_torque(test, device):
    fluid_model = _build_fluid_block(device, dim=1, gravity=0.0)
    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)
    collider_state = collider_model.state()
    fluid_state = fluid_model.state()
    fluid_state.body_q = collider_state.body_q
    fluid_state.body_qd = collider_state.body_qd
    fluid_state.body_f = collider_state.body_f

    solver = _make_solver(fluid_model)
    solver.setup_collider(model=collider_model)
    handler = solver._sph_model.boundary_handler
    handler.analytic_body_impulse_wp.assign(np.zeros((1, 3), dtype=np.float32))
    expected_angular = np.array((0.0, 0.0, 0.8), dtype=np.float32)
    handler.analytic_body_angular_impulse_wp.assign(expected_angular.reshape(1, 3))

    impulses, positions, collider_ids = solver.collect_collider_impulses(fluid_state)
    linear, angular = _collider_wrench(
        impulses.numpy(),
        positions.numpy(),
        collider_model.body_com.numpy()[body],
    )

    test.assertEqual(collider_ids.numpy().tolist(), [0, 0])
    np.testing.assert_allclose(linear, np.zeros(3), atol=1.0e-7)
    np.testing.assert_allclose(angular, expected_angular, rtol=1.0e-5, atol=1.0e-7)


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

    solver = _make_solver(fluid_model, boundary_friction=0.0)
    solver.setup_collider(
        model=collider_model,
        collider_body_ids=[body],
        collider_margins=[0.03],
        collider_friction=[1.0],
        collider_adhesion=[2500.0],
        collider_projection_threshold=[0.02],
    )
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    particle_qd = state_1.particle_qd.numpy()
    shape_adhesion = solver._sph_model.boundary_handler.model_collider_shape_adhesion_wp.numpy()
    shape_projection_threshold = solver._sph_model.boundary_handler.model_collider_shape_projection_threshold_wp.numpy()

    test.assertGreaterEqual(float(particle_q[0, 0]), 0.17 - 1.0e-6)
    test.assertGreater(float(particle_qd[0, 0]), -1.0)
    test.assertLessEqual(float(particle_qd[0, 0]), solver._sph_model.max_depenetration_velocity)
    test.assertGreaterEqual(float(particle_qd[0, 1]), 0.0)
    test.assertLess(float(particle_qd[0, 1]), 1.0)
    test.assertTrue(np.any(shape_adhesion == 2500.0))
    test.assertTrue(np.any(shape_projection_threshold == 0.02))


def test_sph_collider_adhesion_preserves_linear_momentum(test, device):
    def run(adhesion: float):
        fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        sph.add_sph_particle_grid(
            fluid_builder,
            pos=wp.vec3(0.14, 0.0, 0.0),
            vel=wp.vec3(1.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.08,
            cell_y=0.08,
            cell_z=0.08,
            material=sph.SPHMaterial(sound_speed=0.0, stiffness=0.0, smoothing_length=0.16),
            jitter=0.0,
            radius_mean=0.04,
        )
        fluid_model = fluid_builder.finalize(device=device)

        collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        body = collider_builder.add_body(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
        collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        collider_model = collider_builder.finalize(device=device)
        collider_state = collider_model.state()

        state_0 = fluid_model.state()
        state_1 = fluid_model.state()
        for state in (state_0, state_1):
            state.body_q = wp.clone(collider_state.body_q)
            state.body_qd = wp.clone(collider_state.body_qd)
            state.body_f = wp.zeros_like(collider_state.body_f)

        solver = _make_solver(fluid_model, boundary_friction=0.0)
        solver.setup_collider(
            model=collider_model,
            collider_body_ids=[body],
            collider_adhesion=[adhesion],
        )
        initial_velocity = state_0.particle_qd.numpy().copy()
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)
        impulses, _positions, _ids = solver.collect_collider_impulses(state_1)
        return fluid_model, collider_model, state_1, initial_velocity, np.sum(impulses.numpy(), axis=0)

    baseline_model, baseline_collider_model, baseline_state, _initial_velocity, _baseline_body_impulse = run(0.0)
    fluid_model, _collider_model, state, initial_velocity, body_impulse = run(1.0e6)
    final_velocity = state.particle_qd.numpy()
    particle_impulse = fluid_model.particle_mass.numpy()[:, None] * (final_velocity - initial_velocity)

    test.assertAlmostEqual(float(baseline_state.particle_qd.numpy()[0, 0]), 1.0, delta=1.0e-6)
    test.assertGreater(float(final_velocity[0, 0]), 0.0)
    test.assertLess(float(final_velocity[0, 0]), float(baseline_state.particle_qd.numpy()[0, 0]))
    test.assertGreater(float(body_impulse[0]), 0.0)
    np.testing.assert_allclose(np.sum(particle_impulse, axis=0) + body_impulse, np.zeros(3), atol=2.0e-6)

    solver = _make_solver(baseline_model)
    with test.assertRaisesRegex(ValueError, "collider_adhesion.*non-negative"):
        solver.setup_collider(model=baseline_collider_model, collider_adhesion=[-1.0])


def test_sph_model_collider_selection_excludes_unselected_bodies(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(1.09, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(sound_speed=0.0, stiffness=0.0, smoothing_length=0.16),
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    selected_body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    unselected_body = collider_builder.add_body(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=selected_body, hx=0.1, hy=0.1, hz=0.1)
    collider_builder.add_shape_box(body=unselected_body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)

    collider_state = collider_model.state()
    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    for state in (state_0, state_1):
        state.body_q = wp.clone(collider_state.body_q)
        state.body_qd = wp.clone(collider_state.body_qd)
        state.body_f = wp.zeros_like(collider_state.body_f)

    solver = _make_solver(fluid_model)
    solver.setup_collider(model=collider_model, collider_body_ids=[selected_body])
    position_before = state_0.particle_q.numpy()
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    np.testing.assert_allclose(state_1.particle_q.numpy(), position_before, rtol=0.0, atol=1.0e-7)
    test.assertEqual(solver.collider_body_index.numpy().tolist(), [selected_body])
    test.assertEqual(solver._sph_model.boundary_handler.model_collider_shape_indices_wp.shape[0], 1)


def test_sph_depenetration_impulse_preserves_linear_momentum(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.09, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(sound_speed=10.0, viscosity=0.0, smoothing_length=0.16),
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)
    collider_state = collider_model.state()

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    for state in (state_0, state_1):
        state.body_q = wp.clone(collider_state.body_q)
        state.body_qd = wp.clone(collider_state.body_qd)
        state.body_f = wp.zeros_like(collider_state.body_f)

    solver = _make_solver(fluid_model)
    solver.setup_collider(model=collider_model, collider_body_ids=[body])
    position_before = state_0.particle_q.numpy().copy()
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)
    impulses, _positions, _ids = solver.collect_collider_impulses(state_1)

    particle_impulse = fluid_model.particle_mass.numpy()[:, None] * state_1.particle_qd.numpy()
    body_impulse = np.sum(impulses.numpy(), axis=0)
    test.assertGreater(float(state_1.particle_q.numpy()[0, 0]), float(position_before[0, 0]))
    test.assertGreater(float(np.linalg.norm(body_impulse)), 0.0)
    np.testing.assert_allclose(np.sum(particle_impulse, axis=0) + body_impulse, np.zeros(3), atol=2.0e-6)


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

    solver = _make_solver(fluid_model, boundary_friction=0.0)
    solver.setup_collider(
        collider_meshes=[mesh],
        collider_margins=[0.01],
        collider_friction=[0.5],
        collider_adhesion=[1000.0],
        collider_projection_threshold=[0.02],
    )
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    test.assertGreaterEqual(float(particle_q[0, 1]), 0.05 - 1.0e-6)
    test.assertEqual(solver._sph_model.boundary_handler.explicit_collider_adhesion, (1000.0,))
    test.assertEqual(solver._sph_model.boundary_handler.explicit_collider_projection_threshold, (0.02,))


def test_sph_explicit_meshes_replace_model_shape_colliders(test, device):
    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    sph.add_sph_particle_grid(
        fluid_builder,
        pos=wp.vec3(0.09, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=sph.SPHMaterial(sound_speed=0.0, stiffness=0.0, smoothing_length=0.16),
        radius_mean=0.04,
    )
    fluid_model = fluid_builder.finalize(device=device)

    collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    body = collider_builder.add_body(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.diag(wp.vec3(0.01)),
    )
    collider_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    collider_model = collider_builder.finalize(device=device)
    collider_state = collider_model.state()

    state_0 = fluid_model.state()
    state_1 = fluid_model.state()
    for state in (state_0, state_1):
        state.body_q = wp.clone(collider_state.body_q)
        state.body_qd = wp.clone(collider_state.body_qd)
        state.body_f = wp.zeros_like(collider_state.body_f)

    mesh = newton.Mesh.create_plane(1.0, 1.0, compute_inertia=False)
    mesh.vertices[:, 1] -= 1.0
    solver = _make_solver(fluid_model)
    solver.setup_collider(model=collider_model, collider_meshes=[mesh])
    position_before = state_0.particle_q.numpy()
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    np.testing.assert_allclose(state_1.particle_q.numpy(), position_before, rtol=0.0, atol=1.0e-7)
    test.assertEqual(solver._sph_model.boundary_handler.model_collider_shape_indices_wp.shape[0], 0)
    test.assertEqual(solver._sph_model.boundary_handler.explicit_collider_mesh_count(), 1)


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

            solver = _make_solver(fluid_model, boundary_friction=0.0)
            solver.setup_collider(model=collider_model)
            solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

            impulses, impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)
            impulses_np = impulses.numpy()
            impulse_positions_np = impulse_positions.numpy()
            collider_ids_np = collider_ids.numpy()
            total_impulse = np.sum(impulses_np, axis=0)

            test.assertEqual(collider_ids_np.tolist(), [0, 0])
            test.assertTrue(np.isfinite(impulses_np).all())
            test.assertTrue(np.isfinite(impulse_positions_np).all())
            test.assertLess(float(total_impulse[0]), 0.0)
            test.assertGreater(float(np.linalg.norm(total_impulse)), 0.0)


def test_sph_moving_external_collider_velocity_moves_fluid(test, device):
    def run(collider_velocity: float, *, kinematic: bool = False) -> np.ndarray:
        fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        sph.add_sph_particle_grid(
            fluid_builder,
            pos=wp.vec3(0.1399, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
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
        fluid_model = fluid_builder.finalize(device=device)

        collider_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        body = collider_builder.add_body(
            xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.diag(wp.vec3(0.01)),
            is_kinematic=kinematic,
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

        solver = _make_solver(fluid_model, boundary_friction=0.0)
        solver.setup_collider(model=collider_model)
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)
        return state_1.particle_qd.numpy()[0]

    stationary_velocity = run(0.0)
    moving_velocity = run(1.0)
    kinematic_velocity = run(1.0, kinematic=True)
    test.assertGreaterEqual(float(stationary_velocity[0]), 0.0)
    test.assertGreater(float(moving_velocity[0] - stationary_velocity[0]), 0.0)
    test.assertLess(float(moving_velocity[0]), float(kinematic_velocity[0]))
    test.assertGreater(float(kinematic_velocity[0]), 0.9)
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
            pos=wp.vec3(0.1399, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.08,
            cell_y=0.08,
            cell_z=0.08,
            material=material,
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

        solver = _make_solver(
            fluid_model,
            boundary_friction=0.0,
            collider_velocity_mode=collider_velocity_mode,
        )
        solver.setup_collider(
            model=collider_model,
            body_mass=wp.zeros_like(collider_model.body_mass),
            body_inv_inertia=wp.zeros_like(collider_model.body_inv_inertia),
        )

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

    test.assertGreaterEqual(forward_velocity, 0.0)
    test.assertGreater(backward_velocity, forward_velocity + 0.9)


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

    solver = _make_solver(fluid_model)
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

    solver = _make_solver(fluid_model)
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

    solver = _make_solver(fluid_model, boundary_friction=0.0)
    solver.setup_collider(model=collider_model)
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    particle_qd = state_1.particle_qd.numpy()
    impulses, _impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)

    test.assertGreaterEqual(float(particle_q[0, 1]), 0.04 - 1.0e-6)
    test.assertGreater(float(particle_qd[0, 1]), -1.0)
    test.assertEqual(solver.collider_body_index.numpy().tolist(), [body])
    test.assertEqual(collider_ids.numpy().tolist(), [0, 0])
    test.assertLess(float(np.sum(impulses.numpy(), axis=0)[1]), 0.0)


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

    solver = _make_solver(fluid_model, boundary_friction=0.0)
    solver.setup_collider(collider_meshes=[mesh])
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    particle_q = state_1.particle_q.numpy()
    particle_qd = state_1.particle_qd.numpy()
    impulses, _impulse_positions, collider_ids = solver.collect_collider_impulses(state_1)

    test.assertEqual(solver.collider_body_index.numpy().tolist(), [-1])
    test.assertEqual(fluid_model.shape_count, 0)
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

    solver = _make_solver(fluid_model, boundary_friction=1.0)
    solver.setup_collider(
        model=collider_model,
        collider_meshes=[mesh, mesh],
        collider_body_ids=[body, body],
        body_mass=wp.zeros_like(collider_model.body_mass),
        body_inv_inertia=wp.zeros_like(collider_model.body_inv_inertia),
    )
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
    test.assertEqual(collider_ids.numpy().tolist(), [0, 0])
    test.assertEqual(impulses.shape[0], 2)
    test.assertLess(float(np.sum(impulses.numpy(), axis=0)[0]), 0.0)


def test_sph_dam_break_tiny_cpu_case(test, device):
    metrics = _run_tiny_dam_break(device)

    test.assertTrue(metrics["finite"])
    test.assertEqual(metrics["initial_max_speed"], 0.0)
    test.assertGreater(metrics["max_speed"], 0.0)
    test.assertGreater(metrics["final_x_extent"], metrics["initial_x_extent"] + 5.0e-3)
    test.assertLess(metrics["final_mean_height"], metrics["initial_mean_height"] - 1.0e-2)
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
            test.assertAlmostEqual(cuda_metrics[key], cpu_metrics[key], delta=max(1.0e-3, abs(cpu_metrics[key]) * 0.02))

        if run_scenario is _run_tiny_dam_break:
            test.assertGreater(cuda_metrics["final_x_extent"], cuda_metrics["initial_x_extent"] + 5.0e-3)
            test.assertLess(cuda_metrics["final_mean_height"], cuda_metrics["initial_mean_height"] - 1.0e-2)
        else:
            test.assertGreater(cuda_metrics["right_mean_vx"], 0.05)


def test_sph_multiworld_isolates_neighbors_and_colliders(test, device):
    material = sph.SPHMaterial(
        rest_density=1000.0,
        sound_speed=10.0,
        viscosity=0.0,
        smoothing_length=0.16,
    )

    def build_neighbor_model(*, isolated: bool):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        SolverWCSPH.register_custom_attributes(builder)
        if isolated:
            positions = (wp.vec3(0.0), wp.vec3(0.08, 0.0, 0.0))
            for position in positions:
                builder.begin_world()
                sph.add_sph_particle_grid(
                    builder,
                    pos=position,
                    dim_x=1,
                    dim_y=1,
                    dim_z=1,
                    cell_x=0.08,
                    cell_y=0.08,
                    cell_z=0.08,
                    material=material,
                    jitter=0.0,
                    radius_mean=0.04,
                )
                builder.end_world()
        else:
            builder.begin_world()
            sph.add_sph_particle_grid(
                builder,
                pos=wp.vec3(0.0),
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
            builder.end_world()
        return builder.finalize(device=device)

    isolated_model = build_neighbor_model(isolated=True)
    coupled_model = build_neighbor_model(isolated=False)
    isolated_state = isolated_model.state()
    coupled_state = coupled_model.state()
    _make_solver(isolated_model)._compute_density_pressure(isolated_state)
    _make_solver(coupled_model)._compute_density_pressure(coupled_state)

    test.assertEqual(isolated_model.particle_world.numpy().tolist(), [0, 1])
    test.assertEqual(coupled_model.particle_world.numpy().tolist(), [0, 0])
    test.assertTrue(np.all(coupled_state.sph.density.numpy() > isolated_state.sph.density.numpy() + 1.0e-5))

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverWCSPH.register_custom_attributes(builder)
    builder.begin_world()
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.09, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        jitter=0.0,
        radius_mean=0.04,
    )
    builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1)
    builder.end_world()
    builder.begin_world()
    sph.add_sph_particle_grid(
        builder,
        pos=wp.vec3(0.09, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.08,
        cell_y=0.08,
        cell_z=0.08,
        material=material,
        jitter=0.0,
        radius_mean=0.04,
    )
    builder.end_world()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    _make_solver(model).step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)

    test.assertEqual(model.shape_world.numpy().tolist(), [0])
    test.assertGreater(float(state_1.particle_q.numpy()[0, 0]), 0.13)
    test.assertAlmostEqual(float(state_1.particle_q.numpy()[1, 0]), 0.09, delta=1.0e-6)


def test_sph_whole_step_cuda_graph_capture(test, device):
    if not device.is_cuda:
        test.skipTest("whole-step graph capture requires a CUDA device")

    model = _build_fluid_block(device, dim=2, gravity=-9.81, height=0.25, ground_plane=True)
    solver = _make_solver(model)
    warm_state_0 = model.state()
    warm_state_1 = model.state()
    solver.step(warm_state_0, warm_state_1, control=None, contacts=None, dt=1.0e-4)

    state_0 = model.state()
    state_1 = model.state()
    initial_height = float(np.mean(state_0.particle_q.numpy()[:, model.up_axis]))
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0e-4)
        solver.step(state_1, state_0, control=None, contacts=None, dt=1.0e-4)

    for _ in range(5):
        wp.capture_launch(capture.graph)

    particle_q = state_0.particle_q.numpy()
    test.assertTrue(np.isfinite(particle_q).all())
    test.assertLess(float(np.mean(particle_q[:, model.up_axis])), initial_height)


devices = get_test_devices(mode="basic")


class TestSolverWCSPH(unittest.TestCase):
    """Focused WCSPH solver tests."""


for test_func in (
    test_sph_wcsph_step_smoke,
    test_sph_step_rejects_invalid_timestep,
    test_sph_rejects_unknown_config_option,
    test_sph_rejects_invalid_config_values,
    test_sph_default_grid_builds_interacting_neighborhood,
    test_sph_config_material_overrides_are_applied,
    test_sph_per_particle_support_uses_solver_fallback,
    test_sph_coulomb_friction_requires_normal_impulse,
    test_sph_finite_plane_does_not_project_outside_extent,
    test_sph_closed_mesh_projects_toward_exterior,
    test_sph_fluid_block_on_plane_stays_bounded,
    test_sph_hydrostatic_pressure_increases_with_depth,
    test_sph_fluid_block_under_gravity_moves_downward,
    test_sph_fluid_particles_project_outside_ground_plane,
    test_sph_project_outside_matches_mpm_style_projection,
    test_sph_internal_forces_preserve_linear_and_angular_momentum,
    test_sph_pressure_separates_compressed_pair,
    test_sph_viscosity_dissipates_energy_without_changing_momentum,
):
    add_function_test(TestSolverWCSPH, test_func.__name__, test_func, devices=devices)

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
    test_sph_collider_impulse_collection_preserves_pure_torque.__name__,
    test_sph_collider_impulse_collection_preserves_pure_torque,
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
    test_sph_collider_adhesion_preserves_linear_momentum.__name__,
    test_sph_collider_adhesion_preserves_linear_momentum,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_model_collider_selection_excludes_unselected_bodies.__name__,
    test_sph_model_collider_selection_excludes_unselected_bodies,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_depenetration_impulse_preserves_linear_momentum.__name__,
    test_sph_depenetration_impulse_preserves_linear_momentum,
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
    test_sph_explicit_meshes_replace_model_shape_colliders.__name__,
    test_sph_explicit_meshes_replace_model_shape_colliders,
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

add_function_test(
    TestSolverWCSPH,
    test_sph_multiworld_isolates_neighbors_and_colliders.__name__,
    test_sph_multiworld_isolates_neighbors_and_colliders,
    devices=devices,
)

add_function_test(
    TestSolverWCSPH,
    test_sph_whole_step_cuda_graph_capture.__name__,
    test_sph_whole_step_cuda_graph_capture,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
