# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import add_critical_strain_offset
from newton._src.solvers.implicit_mpm.rasterized_collisions import stabilize_collider_velocity
from newton._src.solvers.implicit_mpm.solver_implicit_mpm import _WARP_AT_LEAST_1_19
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_test_devices

wp.set_module_options({"enable_backward": False})

_PARTICLE_RADIUS = 0.1


def _config(voxel_size, **options):
    """Short P0 solve with particle contacts, shared by the tests to compile one solver configuration."""
    defaults = {
        "strain_basis": "P0",
        "collider_basis": "pic8",
        "solver": "jacobi",
        "max_iterations": 1,
        "tolerance": 0.0,
        "warmstart_mode": "none",
        "air_drag": 1.0e-6,
    }
    return SolverImplicitMPM.Config(voxel_size=voxel_size, **{**defaults, **options})


def test_contact_target(test, device):
    """Correct penetration and preserve separated contacts and tangential collider motion."""
    with wp.ScopedDevice(device):
        sdf = wp.array([-0.02, 0.05, 0.0, -0.02], dtype=float, device=device)
        normals = wp.array([(0, 0, 1)] * 3 + [(0, 0, 0)], dtype=wp.vec3, device=device)
        velocity = wp.array([(0.2, -0.3, 0.1)] * 4, dtype=wp.vec3, device=device)
        wp.launch(stabilize_collider_velocity, dim=4, inputs=[0.1, 0.25, sdf, normals, velocity], device=device)
        np.testing.assert_allclose(
            velocity.numpy(),
            [(0.2, -0.3, 0.15), (0.2, -0.3, 0.1), (0.2, -0.3, 0.1), (0.2, -0.3, 0.1)],
            atol=1.0e-7,
        )


def _make_contact_solver(device, height, velocity, recovery_time):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle(pos=(0.5, 0.5, height), vel=velocity, mass=1.0, radius=_PARTICLE_RADIUS)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    model = builder.finalize(device=device)
    return model, SolverImplicitMPM(model, config=_config(1.0, collider_recovery_time=recovery_time))


def test_contact_recovery_truncated(test, device):
    """Push a penetrating particle back to its radius from the surface over one-iteration P0 steps."""
    with wp.ScopedDevice(device):
        model, solver = _make_contact_solver(device, _PARTICLE_RADIUS - 0.02, (0.0, 0.0, 0.0), 0.35)
        state = model.state()
        for _ in range(8):
            solver.step(state, state, None, None, 0.1)
        height = state.particle_q.numpy()[0, 2]
        test.assertGreater(height, _PARTICLE_RADIUS - 0.003)
        test.assertLess(height, _PARTICLE_RADIUS + 0.03)


def test_predictive_contact_truncated(test, device):
    """Stop an approaching particle with its center the larger of one radius and 0.25 voxel out, in one P0 iteration."""
    with wp.ScopedDevice(device):
        model, solver = _make_contact_solver(device, _PARTICLE_RADIUS + 0.6, (0.2, 0.0, -10.0), 0.35)
        state = model.state()
        solver.step(state, state, None, None, 0.1)
        test.assertAlmostEqual(float(state.particle_q.numpy()[0, 2]), max(_PARTICLE_RADIUS, 0.25), delta=2.0e-5)
        test.assertAlmostEqual(float(state.particle_qd.numpy()[0, 0]), 0.2, delta=2.0e-3)


def test_project_outside_keeps_particle_radius(test, device):
    """Project penetrating particles to one radius from collider surfaces with particle contacts only."""
    with wp.ScopedDevice(device):
        model = _make_contact_solver(device, -0.05, (0.0, 0.0, 0.0), None)[0]
        # Projection lets particles sink up to the default threshold of 0.01 voxel
        threshold = 0.01
        for collider_basis, height in (("pic8", _PARTICLE_RADIUS - threshold), ("S2", -threshold)):
            with test.subTest(collider_basis=collider_basis):
                solver = SolverImplicitMPM(model, config=_config(1.0, collider_basis=collider_basis))
                state = model.state()
                solver.project_outside(state, state, 0.1)
                test.assertAlmostEqual(float(state.particle_q.numpy()[0, 2]), height, delta=1.0e-5)


def test_sdf_gradient_normals_particle_contacts(test, device):
    """Give particles beside a box wall normals from the collider distance gradient, pointing out of the wall."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverImplicitMPM.register_custom_attributes(builder)
        for y in (0.3, 0.5, 0.7):
            builder.add_particle(pos=(0.45, y, 0.5), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.02)
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(1.0, 0.5, 0.5), wp.quat_identity()), hx=0.5, hy=2.0, hz=2.0
        )
        model = builder.finalize(device=device)
        config = _config(0.1, collider_normal_from_sdf_gradient=True)
        if not _WARP_AT_LEAST_1_19:
            with test.assertRaisesRegex(ValueError, "Warp 1.19"):
                SolverImplicitMPM(model, config=config)
            return
        solver = SolverImplicitMPM(model, config=config)
        state = model.state()
        apply_transform = solver._apply_strain_basis_transform
        normals = []

        def capture_normals(scratch, basis_transform, stress_transform):
            normals.append(scratch.collider_normal_field.dof_values.numpy().copy())
            apply_transform(scratch, basis_transform, stress_transform)

        solver._apply_strain_basis_transform = capture_normals
        solver.step(state, state, None, None, 0.01)
        np.testing.assert_allclose(normals[0], np.tile((-1.0, 0.0, 0.0), (3, 1)), atol=1.0e-3)


def _make_volume_solver(device, recovery_time, *, target=1.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    # A block of four voxels per axis, packed at 1.2 times its volume, wider than the 3-voxel density kernel
    spacing = 0.5
    volume = 1.2 * spacing**3
    coordinates = np.arange(8) * spacing + 0.25
    positions = np.array(np.meshgrid(coordinates, coordinates, coordinates)).reshape(3, -1).T
    for position in positions:
        builder.add_particle(
            pos=position, vel=(0.0, 0.0, 0.0), mass=1000.0 * volume, radius=0.5 * volume ** (1.0 / 3.0)
        )
    model = builder.finalize(device=device)
    config = _config(1.0, max_iterations=3, volume_recovery_time=recovery_time, volume_recovery_target=target)
    return model, SolverImplicitMPM(model, config=config)


def test_volume_recovery_truncated(test, device):
    """Expand overpacked material with a short P0 solve."""
    with wp.ScopedDevice(device):
        # 0.19 s at dt = 0.02 s corrects about a tenth of the overpacking per step
        model, solver = _make_volume_solver(device, 0.19)
        state = model.state()
        initial_extent = np.prod(np.ptp(state.particle_q.numpy(), axis=0))
        for _ in range(10):
            solver.step(state, state, None, None, 0.02)
        positions = state.particle_q.numpy()
        test.assertTrue(np.isfinite(positions).all())
        test.assertGreater(np.prod(np.ptp(positions, axis=0)), 1.03 * initial_extent)


def _first_density_offset(solver, state):
    """Most compressive density strain offset of one step."""
    build_offset = solver._build_density_strain_offset
    captured = []

    def capture_offset(*args):
        build_offset(*args)
        captured.append(float(args[1].unilateral_strain_offset.numpy().min()))

    solver._build_density_strain_offset = capture_offset
    solver.step(state, state, None, None, 0.02)
    return captured[0]


def test_volume_recovery_offsets(test, device):
    """Push material packed above the target fraction apart, measuring volumes scaled by det(F_e)."""
    with wp.ScopedDevice(device):
        model, solver = _make_volume_solver(device, 0.19)
        overpacked = _first_density_offset(solver, model.state())
        test.assertLess(overpacked, 0.0)

        # Elastic compression offsets the reference overpacking
        state = model.state()
        state.mpm.particle_elastic_strain.fill_(wp.mat33(np.eye(3) * (1.0 / 1.2) ** (1.0 / 3.0)))
        test.assertGreater(_first_density_offset(solver, state), 0.1 * overpacked)

        # Material packed below the target fraction is left alone
        model, solver = _make_volume_solver(device, 0.19, target=1.3)
        test.assertEqual(_first_density_offset(solver, model.state()), 0.0)


def test_density_offset_bounded_in_p1d_modes(test, device):
    """Sample density offsets at the strain quadrature, so they stay bounded in nearly singular P1d modes.

    A close particle pair sunk below the ground fills a solid cell that reads as overpacked. Its
    second P1d mode has a tiny mass, which an offset integrated over the whole cell would overwhelm.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_ground_plane()
        spacing = 0.25
        volume = 1.2 * spacing**3
        xz = np.arange(16) * spacing + 0.5 * spacing
        y = np.arange(10) * spacing + 0.5 * spacing
        positions = np.array(np.meshgrid(xz, y, xz, indexing="ij")).reshape(3, -1).T
        positions = np.vstack((positions, [(2.0, -0.1, 2.0), (2.02, -0.1, 2.0)]))
        for position in positions:
            builder.add_particle(
                pos=position, vel=(0.0, 0.0, 0.0), mass=1000.0 * volume, radius=0.5 * volume ** (1.0 / 3.0)
            )
        model = builder.finalize(device=device)
        solver = SolverImplicitMPM(model, config=_config(1.0, strain_basis="P1d", volume_recovery_time=0.07))
        apply_transform = solver._apply_strain_basis_transform
        ratios = []

        def capture_ratio(scratch, basis_transform, stress_transform):
            apply_transform(scratch, basis_transform, stress_transform)
            offset = scratch.unilateral_strain_offset.numpy()
            mass = scratch.strain_node_particle_volume.numpy()
            active = mass > 0.0
            ratios.append(float(np.max(np.abs(offset[active]) / mass[active])))

        solver._apply_strain_basis_transform = capture_ratio
        solver.step(model.state(), model.state(), None, None, 0.02)
        test.assertGreater(ratios[0], 0.0)
        test.assertLess(ratios[0], 0.5)


def test_critical_offset_modes(test, device):
    """Rotate critical packing bounds into the strain modes, then keep the nonnegative part of each mode."""
    with wp.ScopedDevice(device):
        critical = wp.array([0.1, 0.3, -0.1, 0.2], dtype=float)
        # One element of two nodes, with modes mixing them into their sum and difference
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        eigenvectors = wp.array([[[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]]] * 2, dtype=float)
        offsets = wp.full(4, value=-0.05, dtype=float)
        wp.launch(add_critical_strain_offset, dim=4, inputs=[eigenvectors, critical, offsets])
        np.testing.assert_allclose(
            offsets.numpy(), [0.4 * inv_sqrt2 - 0.05, -0.05, 0.1 * inv_sqrt2 - 0.05, -0.05], atol=1.0e-6
        )

        # An untransformed basis clamps the bounds node by node
        offsets = wp.zeros(4, dtype=float)
        wp.launch(add_critical_strain_offset, dim=4, inputs=[wp.empty((0, 0, 0), dtype=float), critical, offsets])
        np.testing.assert_allclose(offsets.numpy(), [0.1, 0.3, 0.0, 0.2], atol=1.0e-7)


def test_recovery_config_validation(test, device):
    """Reject invalid recovery fractions, gaps, and unsupported combinations."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        model = builder.finalize(device=device)
        for options in (
            {"volume_recovery_time": float("nan")},
            {"volume_recovery_time": 0.0},
            {"volume_recovery_time": -0.1},
            {"volume_recovery_target": 0.0},
            {"volume_recovery_target": float("inf")},
            {"critical_fraction_measurement": "box"},
            {"collider_recovery_time": float("inf")},
            {"collider_recovery_time": 0.0},
        ):
            with test.subTest(options=options), test.assertRaises(ValueError):
                SolverImplicitMPM(model, config=SolverImplicitMPM.Config(**options))


class TestImplicitMPMRecovery(unittest.TestCase):
    pass


for function in (
    test_contact_target,
    test_contact_recovery_truncated,
    test_predictive_contact_truncated,
    test_project_outside_keeps_particle_radius,
    test_sdf_gradient_normals_particle_contacts,
    test_volume_recovery_truncated,
    test_volume_recovery_offsets,
    test_density_offset_bounded_in_p1d_modes,
    test_critical_offset_modes,
    test_recovery_config_validation,
):
    add_function_test(TestImplicitMPMRecovery, function.__name__, function, devices=get_test_devices(mode="basic"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
