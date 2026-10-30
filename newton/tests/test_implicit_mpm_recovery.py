# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_test_devices

wp.set_module_options({"enable_backward": False})


def _make_volume_solver(device, fraction, *, scheme="pic"):
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
    config = SolverImplicitMPM.Config(
        voxel_size=1.0,
        grid_type="fixed",
        grid_padding=2,
        strain_basis="P0",
        warmstart_mode="none",
        integration_scheme=scheme,
        solver="gs",
        max_iterations=3,
        tolerance=0.0,
        density_strain_fraction=fraction,
    )
    return model, SolverImplicitMPM(model, config=config)


def test_volume_recovery_truncated(test, device):
    """Expand overpacked material with a short P0 solve."""
    with wp.ScopedDevice(device):
        for scheme in ("pic", "gimp"):
            extents = []
            for fraction in (0.0, 0.1):
                with test.subTest(scheme=scheme, fraction=fraction):
                    model, solver = _make_volume_solver(device, fraction, scheme=scheme)
                    state = model.state()
                    for _ in range(10):
                        solver.step(state, state, None, None, 0.02)
                    positions = state.particle_q.numpy()
                    test.assertTrue(np.isfinite(positions).all())
                    extents.append(np.prod(np.ptp(positions, axis=0)))
            test.assertGreater(extents[1], 1.03 * extents[0])


def test_density_measures_elastic_volume(test, device):
    """Measure particle volumes scaled by det(F_e), so elastic compression offsets reference overpacking."""
    with wp.ScopedDevice(device):
        offsets = []
        for elastic_ratio in (1.0, 1.0 / 1.2):
            with test.subTest(elastic_ratio=elastic_ratio):
                model, solver = _make_volume_solver(device, 0.1)
                state = model.state()
                state.mpm.particle_elastic_strain.fill_(wp.mat33(np.eye(3) * elastic_ratio ** (1.0 / 3.0)))
                build_offset = solver._build_density_strain_offset
                captured = []

                def capture_offset(*args, build_offset=build_offset, captured=captured):
                    build_offset(*args)
                    captured.append(float(args[1].unilateral_strain_offset.numpy().min()))

                solver._build_density_strain_offset = capture_offset
                solver.step(state, model.state(), None, None, 0.02)
                offsets.append(captured[0])
        test.assertLess(offsets[0], 0.0)
        test.assertGreater(offsets[1], 0.1 * offsets[0])


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
        config = SolverImplicitMPM.Config(
            voxel_size=1.0,
            grid_type="fixed",
            grid_padding=2,
            strain_basis="P1d",
            warmstart_mode="none",
            solver="gs",
            max_iterations=3,
            tolerance=0.0,
            density_strain_fraction=0.25,
        )
        solver = SolverImplicitMPM(model, config=config)
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


def test_critical_fraction_gimp_p1d(test, device):
    """Keep critical packing bounds nonnegative in the diagonalized P1d modes of GIMP integration."""
    with wp.ScopedDevice(device):
        spacing = 0.05
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle_grid(
            pos=wp.vec3(0.5 * spacing),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=4,
            dim_z=6,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=1000.0 * spacing**3,
            jitter=0.0,
            radius_mean=0.5 * spacing,
            custom_attributes={"mpm:young_modulus": 1.0e5, "mpm:poisson_ratio": 0.3, "mpm:friction": 0.6},
        )
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        config = SolverImplicitMPM.Config(
            voxel_size=0.1,
            grid_type="dense",
            strain_basis="P1d",
            integration_scheme="gimp",
            solver="gs",
            max_iterations=50,
            tolerance=1.0e-6,
            critical_fraction=0.6,
        )
        solver = SolverImplicitMPM(model, config=config)
        state_0, state_1 = model.state(), model.state()
        for _ in range(10):
            solver.step(state_0, state_1, None, None, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        positions = state_0.particle_q.numpy()
        test.assertTrue(np.all(np.isfinite(positions)))
        # The 0.3 x 0.2 x 0.3 m block stays together on the ground
        test.assertLess(float(np.ptp(positions[:, 0])), 0.35)
        test.assertLess(float(np.ptp(positions[:, 2])), 0.35)
        test.assertLess(float(positions[:, 1].max()), 0.25)


def _critical_bound_flags(device, spacing_scale):
    """Fraction of occupied P0 cells with an active critical bound, and the largest cell fill, in a closed box."""
    voxel = 0.1
    spacing = voxel / 3.0
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_ground_plane()
    # Walls and a lid close the 0.6 x 0.6 x 0.8 m box on the grid's cell faces
    for center, half in (
        ((-0.05, 0.3, 0.4), (0.05, 0.4, 0.5)),
        ((0.65, 0.3, 0.4), (0.05, 0.4, 0.5)),
        ((0.3, -0.05, 0.4), (0.4, 0.05, 0.5)),
        ((0.3, 0.65, 0.4), (0.4, 0.05, 0.5)),
        ((0.3, 0.3, 0.85), (0.4, 0.4, 0.05)),
    ):
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()), hx=half[0], hy=half[1], hz=half[2]
        )
    step = spacing * spacing_scale
    counts = (np.array([0.6, 0.6, 0.8]) / step).astype(int)
    builder.add_particle_grid(
        pos=wp.vec3(0.5 * step),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(counts[0]),
        dim_y=int(counts[1]),
        dim_z=int(counts[2]),
        cell_x=step,
        cell_y=step,
        cell_z=step,
        mass=1000.0 * spacing**3,
        jitter=0.0,
        radius_mean=0.5 * spacing,
    )
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config(
        voxel_size=voxel,
        strain_basis="P0",
        warmstart_mode="none",
        solver="gs",
        max_iterations=3,
        tolerance=0.0,
        critical_fraction=0.6,
    )
    solver = SolverImplicitMPM(model, config=config)
    apply_transform = solver._apply_strain_basis_transform
    flags = []

    def capture_flags(scratch, basis_transform, stress_transform):
        bound = scratch.critical_strain_offset.numpy()
        fill = scratch.strain_node_particle_volume.numpy()
        flags.append((float(np.mean(bound[fill > 0.0] > 0.0)), fill.max()))
        apply_transform(scratch, basis_transform, stress_transform)

    solver._apply_strain_basis_transform = capture_flags
    solver.step(model.state(), model.state(), None, None, 0.01)
    return flags[0]


def test_critical_fraction_flags_loose_material(test, device):
    """Flag loose packings, and no cells of packed material against collider walls on every side."""
    with wp.ScopedDevice(device):
        packed, packed_fill = _critical_bound_flags(device, 1.0)
        loose, loose_fill = _critical_bound_flags(device, 2.0)
        test.assertGreater(packed_fill, 0.9)
        test.assertLess(loose_fill, 0.5)
        test.assertLess(packed, 0.05)
        test.assertGreater(loose, 0.9)


def test_recovery_config_validation(test, device):
    """Reject invalid recovery fractions and unsupported combinations."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        model = builder.finalize(device=device)
        for options in (
            {"density_strain_fraction": float("nan")},
            {"density_strain_fraction": -0.1},
            {"critical_fraction_measurement": "box"},
        ):
            with test.subTest(options=options), test.assertRaises(ValueError):
                SolverImplicitMPM(model, config=SolverImplicitMPM.Config(**options))


class TestImplicitMPMRecovery(unittest.TestCase):
    pass


for function in (
    test_volume_recovery_truncated,
    test_density_measures_elastic_volume,
    test_density_offset_bounded_in_p1d_modes,
    test_critical_fraction_gimp_p1d,
    test_critical_fraction_flags_loose_material,
    test_recovery_config_validation,
):
    add_function_test(TestImplicitMPMRecovery, function.__name__, function, devices=get_test_devices(mode="basic"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
