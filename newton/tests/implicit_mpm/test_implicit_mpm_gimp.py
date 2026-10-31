# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from itertools import product

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
    gs_colored_contacts=False,
    velocity_basis="Q1",
    integration_scheme="gimp",
    young_modulus=None,
    max_iterations=5,
    air_drag=0.0,
    rheology_solver="gs",
    collider_mass=None,
    collider_contact_gap=0.0,
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for position in positions:
        builder.add_particle(pos=position, vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)
    if contact:
        if collider_mass is None:
            builder.add_ground_plane(height=0.45)
        else:
            body = builder.add_body(mass=collider_mass, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
            builder.add_shape_box(body, hx=3.0, hy=2.0, hz=0.45, cfg=builder.ShapeConfig(density=0.0, mu=0.0))
    model = builder.finalize(device=device)
    if young_modulus is not None:
        model.mpm.young_modulus.fill_(young_modulus)
    solver = solver_class(
        model,
        SolverImplicitMPM.Config(
            voxel_size=1.0,
            grid_type=grid_type,
            grid_padding=0,
            max_active_cell_count=64,
            integration_scheme=integration_scheme,
            velocity_basis=velocity_basis,
            strain_basis=strain_basis,
            collider_basis=collider_basis if contact else "S2",
            warmstart_mode=warmstart_mode,
            max_iterations=max_iterations,
            tolerance=0.0,
            solver=rheology_solver,
            gs_colored_contacts=gs_colored_contacts,
            air_drag=air_drag,
            collider_contact_gap=collider_contact_gap,
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
    """Use cell-local Q1 contact weights at particle centers and retain contact history."""
    for basis, colored in (("pic", False), ("pic1", False), ("pic", True)):
        with test.subTest(collider_basis=basis, colored=colored):
            solver, state = _make_solver(
                device,
                contact=True,
                collider_basis=basis,
                positions=((0.9, 0.4, 0.4), (0.6, 0.4, 0.4), (1.2, 0.4, 0.4)),
                gs_colored_contacts=colored,
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
            expected[valid] = np.array((0.9, 0.6, 0.8))[particles]
            np.testing.assert_allclose(result.numpy(), expected, atol=2.0e-6)
            np.testing.assert_allclose(
                scratch.collider_position_field.dof_values.numpy()[valid],
                state.particle_q.numpy()[particles],
                atol=2.0e-6,
            )
            np.testing.assert_array_equal(np.diff(scratch.collider_matrix.offsets.numpy())[valid], 8)
            impulses = np.arange(9, dtype=np.float32).reshape(3, 3)
            solver._last_step_data.ws_impulse_field.dof_values.assign(impulses)
            solver.step(state, state, None, None, 0.001)
            expected_impulses = np.zeros_like(impulses)
            expected_impulses[particles] = impulses[particles]
            np.testing.assert_array_equal(solver._last_step_data.ws_impulse_field.dof_values.numpy(), expected_impulses)


def test_gimp_stress_roundtrip(test, device):
    """Retain distinct cell pressures through warmstarts, face crossings, and cache invalidation."""
    for swap in (False, True):
        with test.subTest(swap=swap):
            solver, state = _make_solver(device, grid_type="dense", positions=((0.9, 0.4, 0.4), (0.6, 0.4, 0.4)))
            states = [state, solver.model.state() if swap else state]
            state.mpm.particle_stress.assign(np.array((1.0, 3.0))[:, None, None] * np.eye(3))
            for x, pressure in ((0.9, 13.0 / 7.0), (0.9, 13.0 / 7.0), (1.1, 9.0 / 7.0), (2.9, 9.0 / 7.0)):
                positions = states[0].particle_q.numpy()
                positions[0, 0] = x
                states[0].particle_q.assign(positions)
                solver.step(states[0], states[1], None, None, 0.001)
                states.reverse()
                expected = np.array((pressure, 15.0 / 7.0))[:, None, None] * np.eye(3)
                np.testing.assert_allclose(states[0].mpm.particle_stress.numpy(), expected, atol=5.0e-6)
            solver.reset(states[0])
            np.testing.assert_array_equal(solver._last_step_data.gimp_stress[1].numpy(), 0.0)


def test_gimp_capture_history(test, device):
    """Keep overlap history across replays of a graph captured before the first step."""
    for reset, swap in ((False, False), (True, False), (True, True)):
        with test.subTest(reset=reset, swap=swap):
            solver, state = _make_solver(device, grid_type="fixed", positions=((0.9, 0.4, 0.4), (0.6, 0.4, 0.4)))
            if reset:
                solver.reset(state)
            state.mpm.particle_stress.assign(np.array((1.0, 3.0))[:, None, None] * np.eye(3))
            with wp.ScopedCapture(device=device) as capture:
                if swap:
                    # Alternate between two states inside the graph, ending back on the first.
                    other = solver.model.state()
                    solver.step(state, other, None, None, 0.001)
                    solver.step(other, state, None, None, 0.001)
                else:
                    solver.step(state, state, None, None, 0.001)
            for x, pressure in ((0.9, 13.0 / 7.0), (0.9, 13.0 / 7.0), (1.1, 9.0 / 7.0)):
                positions = state.particle_q.numpy()
                positions[0, 0] = x
                state.particle_q.assign(positions)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(state.mpm.particle_stress.numpy()[0], pressure * np.eye(3), atol=5.0e-6)


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


def test_gimp_sparse_graph_rebuild(test, device, colored=False, velocity_basis="Q1"):
    """Match eager GIMP contact transfers while a captured sparse grid follows moving particle domains."""
    solver, state = _make_solver(
        device,
        grid_type="sparse",
        positions=((0.4, -0.1, 0.4),),
        contact=True,
        collider_basis="pic27",
        solver_class=SolverImplicitMPM,
        gs_colored_contacts=colored,
        velocity_basis=velocity_basis,
        air_drag=0.0 if velocity_basis == "Q1" else 1.0,
    )
    reference_solver, reference = _make_solver(
        device,
        grid_type="sparse",
        positions=((0.4, -0.1, 0.4),),
        contact=True,
        collider_basis="pic27",
        solver_class=SolverImplicitMPM,
        gs_colored_contacts=colored,
        velocity_basis=velocity_basis,
        air_drag=0.0 if velocity_basis == "Q1" else 1.0,
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


def test_gimp_masked_reset(test, device, colored=False):
    """Reset one world's subpoint history without discarding another world's pressure."""
    local = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(local)
    local.add_particle(pos=(0.9, 0.4, 0.4), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.2)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(local)
    builder.add_world(local)
    if colored:
        builder.add_ground_plane(height=0.45)
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
            collider_basis="pic27" if colored else "S2",
            gs_colored_contacts=colored,
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
    if colored:
        impulses = solver._last_step_data.ws_impulse_field.dof_values
        impulses.assign(np.array((1.0, 2.0))[:, None] * np.ones((2, 3)))
    solver.reset(state, world_mask=wp.array((True, False, False), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(valid.numpy()[0], 0.0)
    np.testing.assert_array_equal(valid.numpy()[1], before[1])
    if colored:
        np.testing.assert_array_equal(impulses.numpy()[:1], 0.0)
        np.testing.assert_array_equal(impulses.numpy()[1:], 2.0)
        impulses.zero_()
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


def test_colored_contact_velocity_bases(test, device):
    """Stop particles at a plane with the velocity basis's existing DG cell coloring."""
    cases = [(velocity, integration, "P0") for velocity in ("Q1", "B2", "B3") for integration in ("pic", "gimp")]
    cases.append(("B2", "pic", "Q1d"))
    for velocity, integration, strain in cases:
        with test.subTest(velocity=velocity, integration=integration, strain=strain), wp.ScopedDevice(device):
            solver, state = _make_solver(
                device,
                positions=((0.2, 0.4, 0.4), (0.6, 0.4, 0.4), (3.2, 0.4, 0.4), (3.6, 0.4, 0.4)),
                contact=True,
                collider_basis="pic8",
                strain_basis=strain,
                velocity_basis=velocity,
                integration_scheme=integration,
                gs_colored_contacts=True,
                solver_class=SolverImplicitMPM,
                young_modulus=12000.0,
                max_iterations=50,
                air_drag=1.0,
            )
            state.particle_qd.fill_(wp.vec3(0.0, 0.0, -1.0))
            solver.step(state, state, None, None, 0.001)
            velocity_after = state.particle_qd.numpy()
            test.assertTrue(np.isfinite(velocity_after).all())
            test.assertGreater(np.min(velocity_after[:, 2]), -1.0e-3)
            test.assertLess(np.max(np.linalg.norm(velocity_after, axis=1)), 1.0)


def test_colored_contact_default(test, device):
    """Select colored contacts automatically for compatible solvers and allow an opt-out."""
    cases = (
        ("gs", "P0", "pic27", None, True),
        ("gs-soa", "P1d", "pic27", None, True),
        ("gauss-seidel", "Q1d", "pic27", None, True),
        ("gauss-seidel-soa", "P0", "pic27", None, True),
        ("gs-batched", "P0", "pic27", None, False),
        ("jacobi", "P0", "pic27", None, False),
        ("gs", "Q1", "pic27", None, False),
        ("gs", "P0", "S2", None, False),
        ("gs", "P0", "pic27", False, False),
    )
    for mode, strain, collider, option, expected in cases:
        with test.subTest(solver=mode, strain=strain, collider=collider, option=option):
            solver, state = _make_solver(
                device,
                contact=True,
                collider_basis=collider,
                strain_basis=strain,
                rheology_solver=mode,
                gs_colored_contacts=option,
                solver_class=SolverImplicitMPM,
                young_modulus=12000.0,
                air_drag=1.0,
                max_iterations=50,
            )
            test.assertEqual(solver.gs_colored_contacts, expected)
            # Start outside contact, then enter it using the same solver.
            for height in (0.6, 0.4):
                state.particle_q.assign(((0.9, 0.4, height),))
                state.particle_qd.fill_(wp.vec3(0.0, 0.0, -1.0))
                solver.step(state, state, None, None, 0.001)
                test.assertTrue(np.isfinite(state.particle_qd.numpy()).all())
            # Every contact path stops the penetrating particle, which approached at 1 m/s.
            test.assertGreater(state.particle_qd.numpy()[0, 2], -0.05)


def test_pic_speculative_contact(test, device, capture=False):
    """Stop approach at the contact band edge, hold particles inside the band with friction, and allow separation."""
    # Particle contacts measure gaps from the particle surface, 0.2 below the center, to the ground at 0.45.
    # The band is 0.25 voxel deep; a contact gap of 0.5 activates contacts beyond it.
    cases = (
        (0.1, 0.1, (0.3, 0.0, -0.4), (0.0, 0.0, 0.0), 0.0),
        (0.1, 0.05, (0.0, 0.0, -3.0), (0.0, 0.0, 0.0), 0.0),
        (0.4, 0.1, (0.0, 0.0, -3.0), (0.0, 0.0, -1.5), 0.5),
        (0.4, 0.1, (0.0, 0.0, -1.0), (0.0, 0.0, -1.0), 0.5),
        (-0.05, 0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        (-0.05, 0.1, (0.0, 0.0, 0.4), (0.0, 0.0, 0.4), 0.0),
        (0.3, 0.1, (0.0, 0.0, -0.4), (0.0, 0.0, -0.4), 0.0),
    )
    for mode, (gap, dt, velocity, expected, contact_gap) in product(("gs", "gs-soa", "jacobi"), cases):
        with test.subTest(solver=mode, gap=gap, dt=dt, velocity=velocity), wp.ScopedDevice(device):
            solver, state = _make_solver(
                device,
                positions=((0.4, 0.4, 0.65 + gap),),
                contact=True,
                collider_basis="pic27",
                integration_scheme="pic",
                rheology_solver=mode,
                gs_colored_contacts=None,
                solver_class=SolverImplicitMPM,
                young_modulus=12000.0,
                air_drag=1.0e-4,
                max_iterations=50,
                collider_contact_gap=contact_gap,
            )
            state.particle_qd.assign((velocity,))
            if capture:
                solver.step(state, state, None, None, dt)
                state.particle_q.assign(((0.4, 0.4, 0.65 + gap),))
                state.particle_qd.assign((velocity,))
                solver.reset(state)
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state, state, None, None, dt)
                wp.capture_launch(graph.graph)
            else:
                solver.step(state, state, None, None, dt)
            active = np.linalg.norm(solver._scratchpad.collider_normal_field.dof_values.numpy(), axis=1) > 0.0
            test.assertEqual(bool(active.any()), gap < max(0.25, contact_gap))
            np.testing.assert_allclose(state.particle_qd.numpy()[0], expected, atol=2.0e-4)


def test_pic_speculative_contact_adhesion(test, device):
    """Hold particles within the contact band with adhesion and let particles beyond it separate freely."""
    # Gaps are measured from the particle surface, 0.2 below the center, to the ground at 0.45
    cases = (
        (0.1, (0.0, 0.0, 0.4), (0.0, 0.0, 0.0)),
        (-0.05, (0.0, 0.0, 0.4), (0.0, 0.0, 0.0)),
        (0.3, (0.0, 0.0, 0.4), (0.0, 0.0, 0.4)),
    )
    for mode, (gap, velocity, expected) in product(("gs", "jacobi"), cases):
        with test.subTest(solver=mode, gap=gap), wp.ScopedDevice(device):
            solver, state = _make_solver(
                device,
                positions=((0.4, 0.4, 0.65 + gap),),
                contact=True,
                collider_basis="pic27",
                integration_scheme="pic",
                rheology_solver=mode,
                gs_colored_contacts=None,
                solver_class=SolverImplicitMPM,
                young_modulus=12000.0,
                air_drag=1.0e-4,
                max_iterations=50,
            )
            solver._mpm_model.collider.material_adhesion.fill_(1.0e4)
            state.particle_qd.assign((velocity,))
            solver.step(state, state, None, None, 0.1)
            np.testing.assert_allclose(state.particle_qd.numpy()[0], expected, atol=2.0e-4)


def test_grid_contact_full_condition(test, device):
    """Enforce the full contact condition at active grid collider nodes, with no speculative gap."""
    solver, state = _make_solver(
        device,
        positions=((0.4, 0.4, 0.6),),
        contact=True,
        collider_basis="S2",
        solver_class=SolverImplicitMPM,
        integration_scheme="pic",
        young_modulus=12000.0,
        air_drag=1.0e-4,
        max_iterations=50,
    )
    # Serendipity edge nodes at z = 0.5 are active 0.05 above the ground.
    state.particle_qd.assign(((0.0, 0.0, -0.4),))
    solver.step(state, state, None, None, 0.1)
    test.assertGreater(state.particle_qd.numpy()[0, 2], -0.01)


def test_colored_contact_dynamic_collider(test, device, capture=False):
    """Share impact momentum with a finite-mass collider across separated contact cells."""

    class FeedbackMPM(SolverImplicitMPM):
        def _solve_rheology(self, scratch, rigidity_operator):
            result = super()._solve_rheology(scratch, rigidity_operator)
            wp.copy(self.feedback_velocity, scratch.collider_velocity)
            return result

    for mode, velocity in product(("gs", "gs-soa"), ("Q1", "B2")):
        with test.subTest(solver=mode, velocity_basis=velocity), wp.ScopedDevice(device):
            solver, state = _make_solver(
                device,
                positions=((-0.8, 0.2, 0.4), (1.6, 0.2, 0.4)),
                contact=True,
                collider_basis="pic27",
                collider_mass=2.0,
                gs_colored_contacts=None,
                rheology_solver=mode,
                velocity_basis=velocity,
                solver_class=FeedbackMPM,
                max_iterations=50,
                young_modulus=12000.0,
                air_drag=1.0e-4,
            )
            solver.feedback_velocity = wp.empty(2, dtype=wp.vec3)
            state.particle_qd.fill_(wp.vec3(0.0, 0.0, -1.0))
            for step in range(2):
                initial_momentum = state.particle_qd.numpy().sum(axis=0)
                if capture and step > 0:
                    with wp.ScopedCapture(device=device) as graph:
                        solver.step(state, state, None, None, 0.001)
                    wp.capture_launch(graph.graph)
                else:
                    solver.step(state, state, None, None, 0.001)
                impulses, positions, _ = solver.collect_collider_impulses(state)
                impulse, position = impulses.numpy(), positions.numpy()
                linear = impulse.sum(axis=0) / 2.0
                angular = np.cross(position, impulse).sum(axis=0)
                test.assertLess(linear[2], -0.1)
                test.assertGreater(np.linalg.norm(angular), 0.01)
                np.testing.assert_allclose(
                    solver.feedback_velocity.numpy(), linear + np.cross(angular, position), atol=2.0e-6
                )
                np.testing.assert_allclose(
                    state.particle_qd.numpy().sum(axis=0) + impulse.sum(axis=0), initial_momentum, atol=1.0e-4
                )


class TestImplicitMPMGimp(unittest.TestCase):
    pass


for name, func in (
    ("test_gimp_overlap_quadrature", test_gimp_overlap_quadrature),
    ("test_gimp_particle_contact_weights", test_gimp_particle_contact_weights),
    ("test_gimp_stress_roundtrip", test_gimp_stress_roundtrip),
    ("test_gimp_uniform_stress", test_gimp_uniform_stress),
    ("test_gimp_masked_reset", test_gimp_masked_reset),
    ("test_colored_contact_velocity_bases", test_colored_contact_velocity_bases),
    ("test_colored_contact_default", test_colored_contact_default),
    ("test_pic_speculative_contact", test_pic_speculative_contact),
    ("test_pic_speculative_contact_adhesion", test_pic_speculative_contact_adhesion),
    ("test_grid_contact_full_condition", test_grid_contact_full_condition),
    ("test_colored_contact_dynamic_collider", test_colored_contact_dynamic_collider),
):
    add_function_test(TestImplicitMPMGimp, name, func, devices=get_test_devices(mode="basic"))
for name, func in (
    ("test_gimp_sparse_graph_rebuild", test_gimp_sparse_graph_rebuild),
    ("test_gimp_capture_history", test_gimp_capture_history),
    ("test_gimp_uncapped_capture", test_gimp_uncapped_capture),
):
    add_function_test(TestImplicitMPMGimp, name, func, devices=get_cuda_test_devices(mode="basic"))

add_function_test(
    TestImplicitMPMGimp,
    "test_pic_speculative_contact_capture",
    test_pic_speculative_contact,
    devices=get_cuda_test_devices(mode="basic"),
    capture=True,
)

add_function_test(
    TestImplicitMPMGimp,
    "test_colored_contact_dynamic_collider_capture",
    test_colored_contact_dynamic_collider,
    devices=get_cuda_test_devices(mode="basic"),
    capture=True,
)

add_function_test(
    TestImplicitMPMGimp,
    "test_gimp_colored_sparse_graph_rebuild",
    test_gimp_sparse_graph_rebuild,
    devices=get_cuda_test_devices(mode="basic"),
    colored=True,
)

add_function_test(
    TestImplicitMPMGimp,
    "test_gimp_colored_masked_reset",
    test_gimp_masked_reset,
    devices=get_test_devices(mode="basic"),
    colored=True,
)

for basis in ("B2", "B3"):
    add_function_test(
        TestImplicitMPMGimp,
        f"test_gimp_colored_sparse_graph_rebuild_{basis}",
        test_gimp_sparse_graph_rebuild,
        devices=get_cuda_test_devices(mode="basic"),
        colored=True,
        velocity_basis=basis,
    )

if __name__ == "__main__":
    unittest.main()
