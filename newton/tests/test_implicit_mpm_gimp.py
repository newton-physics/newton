# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp

import newton
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


class _TransferOnlyMPM(SolverImplicitMPM):
    def _solve_rheology(self, scratch, rigidity_operator):
        """Isolate quadrature and stress transfer from constitutive convergence."""
        self._apply_strain_basis_transform(scratch, scratch.strain_basis_transform, scratch.strain_stress_transform)
        self._unapply_strain_basis_transform(scratch, scratch.strain_basis_transform)


def _make_solver(
    device,
    *,
    grid_type="fixed",
    positions=((0.9, 0.4, 0.4),),
    contact=False,
    collider_basis="pic",
    strain_basis="P0",
    warmstart_mode="particles",
    solver_class=_TransferOnlyMPM,
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for position in positions:
        builder.add_particle(pos=position, vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)
    if contact:
        builder.add_ground_plane(height=0.45)
    model = builder.finalize(device=device)
    solver = solver_class(
        model,
        SolverImplicitMPM.Config(
            voxel_size=1.0,
            grid_type=grid_type,
            grid_padding=0,
            max_active_cell_count=64,
            integration_scheme="gimp",
            strain_basis=strain_basis,
            collider_basis=collider_basis if contact else "S2",
            warmstart_mode=warmstart_mode,
            max_iterations=5,
            tolerance=0.0,
            solver="gs",
            air_drag=0.0,
        ),
    )
    return solver, model.state()


@fem.integrand
def _integrate_field(s: fem.Sample, u: fem.Field):
    return u(s)


def test_gimp_overlap_quadrature(test, device):
    """Integrate a grid hat function exactly across a particle domain straddling a cell face."""
    for grid_type in ("dense", "fixed", "sparse"):
        if grid_type == "sparse" and not wp.get_device(device).is_cuda:
            continue
        with test.subTest(grid_type=grid_type), wp.ScopedDevice(device):
            shift = 0.5 if grid_type == "sparse" else 0.0
            solver, state = _make_solver(device, grid_type=grid_type, positions=(np.array((0.9, 0.4, 0.4)) - shift,))
            pic = solver._particles_to_cells(state.particle_q)
            test.assertEqual(pic.active_cell_count(), 2)
            space = fem.make_polynomial_space(pic.domain.geometry, degree=1)
            field = space.make_field()
            x = space.basis.node_positions().numpy()[:, 0]
            field.dof_values.assign(np.maximum(1.0 - np.abs(x - (1.0 - shift)), 0.0))
            integral = fem.integrate(_integrate_field, quadrature=pic, fields={"u": field})
            volume = solver._mpm_model.particle_volume.numpy()[0]
            test.assertAlmostEqual(integral / volume, 0.875, delta=2.0e-6)


def test_gimp_particle_contact_weights(test, device):
    """Use the particle's GIMP-averaged grid velocity for its contact constraint."""
    for basis in ("pic", "pic1"):
        with test.subTest(collider_basis=basis):
            solver, state = _make_solver(
                device,
                contact=True,
                collider_basis=basis,
                positions=((0.9, 0.4, 0.4), (0.6, 0.4, 0.4), (1.2, 0.4, 0.4)),
            )
            solver.step(state, state, None, None, 0.001)
            scratch = solver._scratchpad
            ids = scratch.velocity_field.space_partition.space_node_indices().numpy()
            nodes = scratch.velocity_field.space.basis.node_positions().numpy()[ids]
            velocity = np.maximum(1.0 - np.abs(nodes[:, 0] - 1.0), 0.0)
            result = wp.empty(scratch.collider_matrix.nrow, dtype=float, device=device)
            sp.bsr_mv(scratch.collider_matrix, wp.array(velocity, dtype=float, device=device), result)
            indices = scratch.impulse_field.space_partition.space_node_indices().numpy()
            valid = (indices >= 0) & (np.linalg.norm(scratch.collider_normal_field.dof_values.numpy(), axis=1) > 0.0)
            particles = solver._scratchpad.contact_pic.cell_particle_indices.numpy()[indices[valid]]
            expected = np.zeros(len(indices))
            expected[valid] = np.array((0.875, 0.6, 0.8))[particles]
            np.testing.assert_allclose(result.numpy(), expected, atol=2.0e-6)
            impulses = np.arange(9, dtype=np.float32).reshape(3, 3)
            solver._last_step_data.ws_impulse_field.dof_values.assign(impulses)
            solver.step(state, state, None, None, 0.001)
            expected_impulses = np.zeros_like(impulses)
            expected_impulses[particles] = impulses[particles]
            np.testing.assert_array_equal(solver._last_step_data.ws_impulse_field.dof_values.numpy(), expected_impulses)


def test_gimp_stress_roundtrip(test, device):
    """Retain distinct cell pressures through warmstarts, face crossings, and cache invalidation."""
    solver, state = _make_solver(device, grid_type="dense", positions=((0.9, 0.4, 0.4), (0.6, 0.4, 0.4)))
    state.mpm.particle_stress.assign(np.array((1.0, 3.0))[:, None, None] * np.eye(3))
    for x, pressure in ((0.9, 13.0 / 7.0), (0.9, 13.0 / 7.0), (1.1, 9.0 / 7.0), (2.9, 9.0 / 7.0)):
        positions = state.particle_q.numpy()
        positions[0, 0] = x
        state.particle_q.assign(positions)
        solver.step(state, state, None, None, 0.001)
        expected = np.array((pressure, 15.0 / 7.0))[:, None, None] * np.eye(3)
        np.testing.assert_allclose(state.mpm.particle_stress.numpy(), expected, atol=5.0e-6)
    solver.reset(state)
    np.testing.assert_array_equal(solver._last_step_data.gimp_stress[1].numpy(), 0.0)


def test_gimp_uniform_stress(test, device):
    """Preserve constant pressure through particle-domain transfers for each strain basis."""
    for basis, mode in (
        ("P0", "particles"),
        ("P1d", "particles"),
        ("Q1d", "particles"),
        ("pic", "particles"),
        ("pic", "grid"),
    ):
        with test.subTest(strain_basis=basis, warmstart_mode=mode):
            solver, state = _make_solver(device, strain_basis=basis, warmstart_mode=mode)
            if basis == "pic":
                test.assertEqual(solver._scratchpad._strain_basis.topology.MAX_NODES_PER_ELEMENT, 1)
            state.mpm.particle_stress.fill_(wp.mat33(2.0 * np.eye(3)))
            for _ in range(2):
                target = solver.model.state()
                solver.step(state, target, None, None, 0.001)
                state = target
                stress, valid, _ = solver._last_step_data.gimp_stress
                active = valid.numpy() > 0.0
                test.assertTrue(np.any(active))
                expected = np.broadcast_to(2.0 * np.eye(3), stress.numpy()[active].shape)
                np.testing.assert_allclose(stress.numpy()[active], expected, atol=5.0e-6)
                if mode == "particles":
                    np.testing.assert_allclose(state.mpm.particle_stress.numpy()[0], 2.0 * np.eye(3), atol=5.0e-6)


def test_gimp_sparse_graph_rebuild(test, device):
    """Match eager GIMP contact transfers while a captured sparse grid follows moving particle domains."""
    solver, state = _make_solver(
        device,
        grid_type="sparse",
        positions=((0.4, -0.1, 0.4),),
        contact=True,
        collider_basis="pic27",
        solver_class=SolverImplicitMPM,
    )
    reference_solver, reference = _make_solver(
        device,
        grid_type="sparse",
        positions=((0.4, -0.1, 0.4),),
        contact=True,
        collider_basis="pic27",
        solver_class=SolverImplicitMPM,
    )
    state.particle_qd.fill_(wp.vec3(1.0, 0.0, 0.0))
    reference.particle_qd.fill_(wp.vec3(1.0, 0.0, 0.0))
    solver.step(state, state, None, None, 0.3)
    reference_solver.step(reference, reference, None, None, 0.3)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, state, None, None, 0.3)
        solver.step(state, state, None, None, 0.3)
    for _ in range(3):
        wp.capture_launch(capture.graph)
        for _ in range(2):
            reference_solver.step(reference, reference, None, None, 0.3)
    solver.check_sparse_grid_rebuild_status()
    np.testing.assert_allclose(state.particle_q.numpy(), reference.particle_q.numpy(), atol=2.0e-6)
    np.testing.assert_allclose(state.particle_qd.numpy(), reference.particle_qd.numpy(), atol=2.0e-6)


def test_gimp_masked_reset(test, device):
    """Reset one world's subpoint history without discarding another world's pressure."""
    local = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(local)
    local.add_particle(pos=(0.9, 0.4, 0.4), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)
    solver = _TransferOnlyMPM(
        model,
        SolverImplicitMPM.Config(
            voxel_size=1.0,
            grid_type="fixed",
            grid_padding=0,
            max_active_cell_count=64,
            integration_scheme="gimp",
            strain_basis="P0",
            collider_basis="S2",
            warmstart_mode="particles",
            separate_worlds=True,
            air_drag=0.0,
        ),
    )
    state = model.state()
    state.mpm.particle_stress.assign(np.array((2.0, 4.0))[:, None, None] * np.eye(3))
    solver.step(state, state, None, None, 0.001)
    valid = solver._last_step_data.gimp_stress[1]
    before = valid.numpy().copy()
    test.assertTrue(np.any(before[0]) and np.any(before[1]))
    solver.reset(state, world_mask=wp.array((True, False, False), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(valid.numpy()[0], 0.0)
    np.testing.assert_array_equal(valid.numpy()[1], before[1])
    solver.step(state, state, None, None, 0.001)
    np.testing.assert_allclose(
        state.mpm.particle_stress.numpy(), np.array((0.0, 4.0))[:, None, None] * np.eye(3), atol=5.0e-6
    )


def test_gimp_uncapped_capture(test, device):
    """Reject uncapped point bases during graph capture rather than using a global point bound."""
    solver, state = _make_solver(device, contact=True)
    solver.step(state, state, None, None, 0.001)
    with test.assertRaisesRegex(RuntimeError, "explicit 'picN' limit"):
        with wp.ScopedCapture(device=device):
            solver.step(state, state, None, None, 0.001)


class TestImplicitMPMGimp(unittest.TestCase):
    pass


for name, func in (
    ("test_gimp_overlap_quadrature", test_gimp_overlap_quadrature),
    ("test_gimp_particle_contact_weights", test_gimp_particle_contact_weights),
    ("test_gimp_stress_roundtrip", test_gimp_stress_roundtrip),
    ("test_gimp_uniform_stress", test_gimp_uniform_stress),
    ("test_gimp_masked_reset", test_gimp_masked_reset),
):
    add_function_test(TestImplicitMPMGimp, name, func, devices=get_test_devices(mode="basic"))
for name, func in (
    ("test_gimp_sparse_graph_rebuild", test_gimp_sparse_graph_rebuild),
    ("test_gimp_uncapped_capture", test_gimp_uncapped_capture),
):
    add_function_test(TestImplicitMPMGimp, name, func, devices=get_cuda_test_devices(mode="basic"))

if __name__ == "__main__":
    unittest.main()
