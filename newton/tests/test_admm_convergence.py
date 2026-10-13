# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate Warp convergence checks and adaptive coupled iteration control."""

import dataclasses
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.admm_convergence import (
    AdmmConvergenceGroup,
    check_body_proximal_convergence_kernel,
    check_convergence_kernel,
    check_joint_proximal_convergence_kernel,
    check_particle_proximal_convergence_kernel,
    snapshot_forces_kernel,
)
from newton.solvers import SolverSemiImplicit, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_solver(
    device, convergence=None, *, iterations=8, substeps=1, gap=-0.099, velocity=0.0, gamma=0.0, baumgarte=0.2
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=(gap, 0.0, 0.0), vel=(velocity, 0.0, 0.0), mass=1.0, radius=0.05)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(-velocity, 0.0, 0.0), mass=1.0, radius=0.05)
    model = builder.finalize(device=device)

    def make_participant(view):
        # Each participant owns one particle, so only the coupler needs a contact grid.
        view.particle_grid = None
        return SolverSemiImplicit(view)

    solver = SolverCoupledADMM(
        model,
        entries=[
            SolverCoupled.Entry(name=name, solver=make_participant, particles=[i], substeps=substeps)
            for i, name in enumerate(("a", "b"))
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=iterations,
            rho=20.0,
            gamma=gamma,
            baumgarte=baumgarte,
            convergence=convergence,
            contact_pairs=[SolverCoupledADMM.ContactPair("a", "b")],
        ),
    )
    return model, solver


def _row_buffers(device, *, count=1, angular=False, revolute=False, dynamic=False):
    group = AdmmConvergenceGroup()
    group.W = wp.ones(count, dtype=float, device=device)
    group.u = wp.zeros(count, dtype=wp.vec3, device=device)
    group.lambda_ = wp.zeros(count, dtype=wp.vec3, device=device)
    group.Jv = wp.zeros(count, dtype=wp.vec3, device=device)
    group.offset = 0
    group.capacity = count
    group.angular = angular
    group.revolute = revolute
    if dynamic:
        group.active_count = wp.array([count], dtype=int, device=device)
    groups = wp.array([group], dtype=AdmmConvergenceGroup, device=device)
    indices = wp.zeros(count, dtype=int, device=device)
    previous = wp.empty(count, dtype=wp.vec3, device=device)
    failed = wp.zeros(1, dtype=int, device=device)
    return group, groups, indices, previous, failed


def test_criteria(test, device):
    """Check force amplification, relative limits, masking, and invalid values."""
    for angular, revolute, dynamic in [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
    ]:
        group, groups, indices, previous, failed = _row_buffers(
            device, angular=angular, revolute=revolute, dynamic=dynamic
        )

        def snapshot(groups=groups, indices=indices, previous=previous, failed=failed):
            wp.launch(snapshot_forces_kernel, dim=1, inputs=[groups, indices, 1000.0, previous, failed], device=device)

        def check(
            *,
            v=1e-5,
            av=1e-5,
            f=1e-3,
            t=1e-3,
            relative=0.0,
            groups=groups,
            indices=indices,
            previous=previous,
            failed=failed,
        ):
            wp.launch(
                check_convergence_kernel,
                dim=1,
                inputs=[groups, indices, 1000.0, v, av, f, t, relative, previous, failed],
                device=device,
            )
            return int(failed.numpy()[0])

        snapshot()
        test.assertEqual(check(), 0)
        group.W.fill_(1000.0)
        snapshot()
        group.u.assign([[0.0, 1e-6, 0.0]])
        test.assertEqual(check(), 1)  # Velocity passes; force changes by 1000 N.
        group.u.zero_()
        group.lambda_.assign([[0.0, 1.0, 0.0]])
        snapshot()
        group.lambda_.assign([[0.0, 1.001, 0.0]])
        test.assertEqual(check(relative=0.01), 0)
        snapshot()
        group.u.assign([[0.0, 1e-4, 0.0]])
        test.assertEqual(check(v=1e-3, av=1e-5, f=1e9, t=1e9), int(angular))
        group.u.zero_()
        for value in (float("nan"), float("inf"), -float("inf"), 1e38):
            group.lambda_.zero_()
            snapshot()
            group.lambda_.assign([[0.0, value, 0.0]])
            test.assertEqual(check(), 1)
        group.lambda_.zero_()
        snapshot()
        group.u.assign([[1.0, 0.0, 0.0]])
        test.assertEqual(check(), int(not revolute))
        group.u.zero_()
        snapshot()
        group.u.assign([[float("nan"), 0.0, 0.0]])
        test.assertEqual(check(), 1)  # Nonfinite inputs fail even on the free hinge axis.
        snapshot()
        group.u.zero_()
        test.assertEqual(check(), 1)  # The force used must also have been finite.


def test_capacity(test, device):
    """Ignore unused capacity while checking every populated speculative row."""
    group, groups, indices, previous, failed = _row_buffers(device, count=3, dynamic=True)
    group.active_count.fill_(1)
    group.u.assign([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]])
    for count, expected in [(1, 0), (2, 1), (4, 1), (-1, 1), (0, 0)]:
        group.active_count.fill_(count)
        wp.launch(snapshot_forces_kernel, dim=3, inputs=[groups, indices, 1.0, previous, failed], device=device)
        wp.launch(
            check_convergence_kernel,
            dim=3,
            inputs=[groups, indices, 1.0, 1e-5, 1e-5, 1e-3, 1e-3, 0.0, previous, failed],
            device=device,
        )
        test.assertEqual(int(failed.numpy()[0]), expected)


def test_stopping(test, device):
    """Stop settled contacts after complete iterations and clear status on reset."""
    config = SolverCoupledADMM.ConvergenceConfig(min_iterations=2, check_interval=3)
    model, solver = _make_solver(device, config, gap=-0.1)
    state_in, state_out = model.state(), model.state()
    solver.step(state_in, state_out, None, None, 0.001)
    test.assertEqual(int(solver.iteration_count.numpy()[0]), 2)
    test.assertEqual(int(solver.converged.numpy()[0]), 1)
    test.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
    solver.reset(state_in)
    test.assertEqual(int(solver.iteration_count.numpy()[0]), 0)
    test.assertEqual(int(solver.converged.numpy()[0]), 0)


def test_loaded_stopping(test, device):
    """Stop under sustained contact forces and match a fixed-iteration reference."""
    dt = 0.001
    for gamma, substeps, interval in ((0.0, 1, 1), (0.1, 3, 3)):
        with test.subTest(gamma=gamma, substeps=substeps, interval=interval):
            config = SolverCoupledADMM.ConvergenceConfig(check_interval=interval)
            results = []
            for convergence, iterations in ((None, 1024), (config, 512)):
                model, solver = _make_solver(
                    device, convergence, iterations=iterations, gamma=gamma, substeps=substeps, baumgarte=0.0
                )
                state_in, state_out = model.state(), model.state()
                # With penetration correction disabled, opposing 10 N loads have
                # an equilibrium of zero velocity and a 10 N coupling force.
                state_in.particle_f.assign([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
                solver.step(state_in, state_out, None, None, dt)
                group = solver._admm_dynamic_pp_contact_groups[0]
                test.assertEqual(int(group.active_count.numpy()[0]), 1)
                # The mechanics buffer contains the external load plus the coupling force.
                force = solver._admm_buffers["a"].particle_f.numpy()[0] - state_in.particle_f.numpy()[0]
                test.assertGreater(np.linalg.norm(force), 9.0)
                results.append((state_out.particle_q.numpy(), state_out.particle_qd.numpy(), force))
                if convergence is not None:
                    test.assertEqual(int(solver.converged.numpy()[0]), 1)
                    test.assertGreater(int(solver.iteration_count.numpy()[0]), config.min_iterations)
                    test.assertLess(int(solver.iteration_count.numpy()[0]), iterations)
            reference, adaptive = results
            np.testing.assert_allclose(adaptive[0], reference[0], atol=dt * config.linear_velocity_tolerance, rtol=0.0)
            np.testing.assert_allclose(adaptive[1], reference[1], atol=config.linear_velocity_tolerance, rtol=0.0)
            force_tolerance = config.force_tolerance + config.force_relative_tolerance * np.linalg.norm(reference[2])
            test.assertLessEqual(np.linalg.norm(adaptive[2] - reference[2]), force_tolerance)
            np.testing.assert_allclose(adaptive[1], 0.0, atol=config.linear_velocity_tolerance, rtol=0.0)
            np.testing.assert_allclose(adaptive[2], [-10.0, 0.0, 0.0], atol=force_tolerance, rtol=0.0)


def test_proximal_common_motion(test, device):
    """Resolve shared acceleration even when every interface residual is zero."""
    dt = 0.001
    for gamma, substeps, interval, gap in ((0.1, 1, 1, -0.099), (1.0, 3, 3, -0.11)):
        config = SolverCoupledADMM.ConvergenceConfig(check_interval=interval)
        outputs = []
        for convergence in (None, config):
            model, solver = _make_solver(
                device, convergence, iterations=256, substeps=substeps, gamma=gamma, baumgarte=0.0, gap=gap
            )
            state_in, state_out = model.state(), model.state()
            state_in.particle_f.fill_(wp.vec3(0.0, 1.0, 0.0))
            solver.step(state_in, state_out, None, None, dt)
            outputs.append(state_out.particle_qd.numpy())
            group = solver._admm_dynamic_pp_contact_groups[0]
            test.assertEqual(int(group.active_count.numpy()[0]) > 0, gap > -0.1)
            np.testing.assert_array_equal(group.u.numpy(), group.Jv.numpy())
            if convergence is not None:
                test.assertEqual(int(solver.converged.numpy()[0]), 1)
                test.assertGreater(int(solver.iteration_count.numpy()[0]), 2)
                test.assertLess(int(solver.iteration_count.numpy()[0]), 256)
        np.testing.assert_allclose(outputs[1], outputs[0], atol=config.linear_velocity_tolerance, rtol=0.0)
        np.testing.assert_allclose(outputs[1][:, 1], dt, atol=config.linear_velocity_tolerance, rtol=0.0)


def test_proximal_criteria(test, device):
    """Reject amplified increments and invalid active inputs while ignoring unshifted velocities."""
    previous = wp.zeros(1, dtype=wp.vec3, device=device)
    current = wp.zeros_like(previous)
    mass = wp.array([1.0], dtype=float, device=device)
    lump = wp.zeros(1, dtype=float, device=device)
    failed = wp.zeros(1, dtype=int, device=device)
    for added_mass, delta, expected in (
        (100.0, 1e-6, 1),  # A small increment can hide a large proximal contribution.
        (100.0, 1e-8, 0),
        (0.1, 1e-4, 1),  # Small proximal factors still require the iterate itself to settle.
        (0.0, float("nan"), 0),
        (1.0, float("nan"), 1),
        (1.0, float("inf"), 1),
        (float("nan"), 0.0, 1),
        (float("inf"), 0.0, 1),
    ):
        mass.fill_(1.0 + added_mass)
        lump.fill_(added_mass)
        current.fill_(wp.vec3(delta, 0.0, 0.0))
        failed.zero_()
        wp.launch(
            check_particle_proximal_convergence_kernel,
            dim=1,
            inputs=[previous, current, lump, mass, 1e-5, failed],
            device=device,
        )
        test.assertEqual(int(failed.numpy()[0]), expected)


def test_proximal_angular_and_joint_units(test, device):
    """Check body rotation and mixed joint layouts with separate angular tolerances."""
    previous = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    current = wp.zeros_like(previous)
    lump = wp.array([1.0], dtype=float, device=device)
    mass = wp.array([2.0], dtype=float, device=device)
    inertia = wp.array([np.eye(3) * 2.0], dtype=wp.mat33, device=device)
    failed = wp.zeros(1, dtype=int, device=device)
    for axis, expected in ((0, 0), (3, 1)):
        value = np.zeros((1, 6), dtype=np.float32)
        value[0, axis] = 1e-4
        current.assign(value)
        failed.zero_()
        wp.launch(
            check_body_proximal_convergence_kernel,
            dim=1,
            inputs=[previous, current, lump, lump, mass, inertia, 1e-3, 1e-5, failed],
            device=device,
        )
        test.assertEqual(int(failed.numpy()[0]), expected)

    # Free, revolute, and prismatic joint layouts in one participant.
    starts = wp.array([0, 6, 7, 8], dtype=int, device=device)
    dimensions = wp.array([[3, 3], [0, 1], [1, 0]], dtype=int, device=device)
    factors = wp.ones(8, dtype=float, device=device)
    previous_joint = wp.zeros(8, dtype=float, device=device)
    current_joint = wp.zeros_like(previous_joint)
    for dof, expected in ((0, 0), (3, 1), (6, 1), (7, 0)):
        value = np.zeros(8, dtype=np.float32)
        value[dof] = 1e-4
        current_joint.assign(value)
        failed.zero_()
        wp.launch(
            check_joint_proximal_convergence_kernel,
            dim=3,
            inputs=[previous_joint, current_joint, factors, starts, dimensions, 1e-3, 1e-5, failed],
            device=device,
        )
        test.assertEqual(int(failed.numpy()[0]), expected)


def test_proximal_graph_replay(test, device):
    """Replay proximal stopping with changed forcing and reset convergence state."""
    if not device.is_cuda:
        test.skipTest("CUDA graphs require CUDA")
    config = SolverCoupledADMM.ConvergenceConfig(check_interval=3)
    model, solver = _make_solver(device, config, gamma=0.1, iterations=32, baumgarte=0.0, substeps=3)
    state_in, state_out = model.state(), model.state()
    state_in.particle_f.fill_(wp.vec3(0.0, 1.0, 0.0))
    solver.step(state_in, state_out, None, None, 0.001)
    solver.reset(state_in)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, None, None, 0.001)
    for force in (1.0, 2.0, 0.0):
        solver.reset(state_in)
        state_in.particle_f.fill_(wp.vec3(0.0, force, 0.0))
        wp.capture_launch(capture.graph)
        test.assertEqual(int(solver.converged.numpy()[0]), 1)
        if force:
            test.assertGreater(int(solver.iteration_count.numpy()[0]), 2)
        else:
            test.assertEqual(int(solver.iteration_count.numpy()[0]), 2)
        np.testing.assert_allclose(
            state_out.particle_qd.numpy()[:, 1], 0.001 * force, atol=config.linear_velocity_tolerance, rtol=0.0
        )


def test_cap_parity(test, device):
    """Match fixed iterations when strict criteria exhaust a nondivisible cap."""
    strict = SolverCoupledADMM.ConvergenceConfig(
        linear_velocity_tolerance=0.0,
        angular_velocity_tolerance=0.0,
        force_tolerance=0.0,
        torque_tolerance=0.0,
        force_relative_tolerance=0.0,
        min_iterations=2,
        check_interval=4,
    )
    for substeps in (1, 2, 3):
        outputs = []
        for config in (None, strict):
            model, solver = _make_solver(device, config, substeps=substeps, velocity=0.1)
            state_in, state_out = model.state(), model.state()
            for _ in range(3):
                solver.step(state_in, state_out, None, None, 0.001)
                state_in, state_out = state_out, state_in
                if config is not None:
                    test.assertEqual(int(solver.iteration_count.numpy()[0]), 8)
                    test.assertEqual(int(solver.converged.numpy()[0]), 0)
            outputs.append((state_in.particle_q.numpy(), state_in.particle_qd.numpy()))
        np.testing.assert_array_equal(outputs[0], outputs[1])


def test_graph_replay(test, device):
    """Replay conditional stopping as contact state changes on the device."""
    if not device.is_cuda:
        test.skipTest("CUDA graphs require CUDA")
    config = SolverCoupledADMM.ConvergenceConfig(min_iterations=2, check_interval=3)
    model, solver = _make_solver(device, config, gap=-0.11, substeps=3)
    state_in, state_out = model.state(), model.state()
    solver.step(state_in, state_out, None, None, 0.001)
    with wp.ScopedDevice(device), wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, None, None, 0.001)
    wp.capture_launch(capture.graph)
    test.assertEqual(int(solver.iteration_count.numpy()[0]), 2)
    state_in.particle_q.assign([[-0.099, 0.0, 0.0], [0.0, 0.0, 0.0]])
    state_in.particle_qd.assign([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    wp.capture_launch(capture.graph)
    test.assertGreater(int(solver.iteration_count.numpy()[0]), 2)
    captured = state_out.particle_q.numpy().copy()
    iterations = int(solver.iteration_count.numpy()[0])
    solver.reset(state_in)
    solver.step(state_in, state_out, None, None, 0.001)
    test.assertEqual(int(solver.iteration_count.numpy()[0]), iterations)
    np.testing.assert_array_equal(captured, state_out.particle_q.numpy())
    state_in.particle_q.assign([[-0.11, 0.0, 0.0], [0.0, 0.0, 0.0]])
    state_in.particle_qd.zero_()
    solver.reset(state_in)
    wp.capture_launch(capture.graph)
    test.assertEqual(int(solver.iteration_count.numpy()[0]), 2)


def _make_rigid_solver(device, convergence, *, kind="contact", gamma=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    a = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    b = builder.add_body(
        xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
    )
    if kind == "contact":
        builder.add_shape_sphere(a, radius=0.05)
        builder.add_shape_sphere(b, radius=0.05)
    elif kind == "fixed":
        builder.add_joint_fixed(parent=a, child=b)
    elif kind == "revolute":
        builder.add_joint_revolute(parent=a, child=b, friction=0.2)
    builder.color()
    model = builder.finalize(device=device)
    solver = SolverCoupledADMM(
        model,
        [
            SolverCoupled.Entry(
                "a", lambda v: SolverVBD(v, iterations=1, rigid_compliant_alm=True), bodies=[a], substeps=2
            ),
            SolverCoupled.Entry(
                "b", lambda v: SolverVBD(v, iterations=1, rigid_compliant_alm=True), bodies=[b], substeps=3
            ),
        ],
        SolverCoupledADMM.Config(
            iterations=7,
            rho=20.0,
            gamma=gamma,
            baumgarte=0.1,
            convergence=convergence,
            contact_pairs=[SolverCoupledADMM.ContactPair("a", "b")] if kind == "contact" else (),
        ),
    )
    return model, solver


def test_rigid_groups(test, device):
    """Check rigid contact, point, angular, hinge and friction rows with real solvers."""
    strict = SolverCoupledADMM.ConvergenceConfig(
        linear_velocity_tolerance=0.0,
        angular_velocity_tolerance=0.0,
        force_tolerance=0.0,
        torque_tolerance=0.0,
        force_relative_tolerance=0.0,
        min_iterations=2,
        check_interval=3,
    )
    for kind in ("contact", "fixed", "revolute"):
        results = []
        for config in (None, strict):
            model, solver = _make_rigid_solver(device, config, kind=kind, gamma=0.2)
            a, b = model.state(), model.state()
            solver.step(a, b, None, None, 0.001)
            results.append((b.body_q.numpy(), b.body_qd.numpy()))
            if config is not None:
                test.assertGreater(solver._convergence_force_used.shape[0], 0)
                test.assertEqual(int(solver.iteration_count.numpy()[0]), 7)
                test.assertEqual(int(solver.converged.numpy()[0]), 0)
                if device.is_cuda:
                    solver.reset(a)
                    with wp.ScopedCapture(device=device) as capture:
                        solver.step(a, b, None, None, 0.001)
                    wp.capture_launch(capture.graph)
                    test.assertEqual(int(solver.iteration_count.numpy()[0]), 7)
                    np.testing.assert_allclose(b.body_q.numpy(), results[-1][0], atol=1e-6, rtol=1e-6)
        for fixed, adaptive in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(fixed, adaptive)


def _make_rigid_particle_solver(device, convergence, *, attachment):
    """Build a rigid-particle attachment or contact between two participants.

    Args:
        device: Warp device for the model and solvers.
        convergence: Optional ADMM convergence settings.
        attachment: Whether to couple through an attachment or a contact.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_shape_sphere(body, radius=0.05)
    particle = builder.add_particle(pos=(0.08, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.05)
    if attachment:
        SolverCoupledADMM.add_body_particle_attachment(builder, body, particle, stiffness=500.0)
    model = builder.finalize(device=device)
    solver = SolverCoupledADMM(
        model,
        [
            SolverCoupled.Entry("body", SolverSemiImplicit, bodies=[body]),
            SolverCoupled.Entry("particle", SolverSemiImplicit, particles=[particle]),
        ],
        SolverCoupledADMM.Config(
            iterations=5,
            rho=20.0,
            baumgarte=0.2,
            convergence=convergence,
            contact_pairs=() if attachment else [SolverCoupledADMM.ContactPair("body", "particle")],
        ),
    )
    return model, solver


def test_rigid_particle_groups(test, device):
    """Include rigid-particle attachments and collision rows in both criteria."""
    for attachment in (False, True):
        config = SolverCoupledADMM.ConvergenceConfig(min_iterations=1, check_interval=2)
        model, solver = _make_rigid_particle_solver(device, config, attachment=attachment)
        a, b = model.state(), model.state()
        solver.step(a, b, None, None, 0.001)
        test.assertGreater(int(solver.iteration_count.numpy()[0]), 1)
        test.assertTrue(np.isfinite(b.body_q.numpy()).all())
        test.assertTrue(np.isfinite(b.particle_q.numpy()).all())
        groups = solver._admm_rp_groups if attachment else solver._admm_dynamic_rp_contact_groups
        test.assertEqual(len(groups), 1)
        solver._snapshot_convergence_forces()
        groups[0].lambda_.fill_(wp.vec3(1e4, 0.0, 0.0))
        solver._check_convergence(5)
        test.assertEqual(int(solver.converged.numpy()[0]), 0)


def test_group_force_consistency(test, device):
    """Check every group family's convergence forces against mechanics accumulation."""
    config = SolverCoupledADMM.ConvergenceConfig(linear_velocity_tolerance=1.0, angular_velocity_tolerance=1.0)
    scenes = [
        _make_solver(device, config),
        *(_make_rigid_solver(device, config, kind=kind) for kind in ("contact", "fixed", "revolute")),
        *(_make_rigid_particle_solver(device, config, attachment=value) for value in (False, True)),
    ]
    declared = set()
    exercised = set()
    for model, solver in scenes:
        solver.step(model.state(), model.state(), None, None, 0.001)
        # Discover collections independently of the list in _setup_convergence.
        collections = {
            name: groups
            for name, groups in vars(solver).items()
            if name.startswith("_admm_") and name.endswith("_groups")
        }
        declared.update(collections)
        for name, groups in collections.items():
            for group in groups:
                with test.subTest(group=name):
                    for collection in collections:
                        setattr(solver, collection, [])
                    setattr(solver, name, [group])
                    solver._setup_convergence()
                    test.assertEqual(solver._convergence_force_used.shape[0], group.count)
                    if hasattr(group, "active_count"):
                        test.assertEqual(int(group.active_count.numpy()[0]), 1)
                    group.W.fill_(2.5)
                    group.lambda_.fill_(wp.vec3(1.0, -2.0, 3.0))
                    group.u.fill_(wp.vec3(0.04, 0.05, -0.06))
                    group.Jv.fill_(wp.vec3(-0.02, 0.01, 0.03))
                    solver._snapshot_convergence_forces()
                    for buf in solver._admm_buffers.values():
                        solver._zero_array(buf.body_f)
                        solver._zero_array(buf.particle_f)
                    solver._accumulate_admm_forces(0, 0.001, refresh_jv=False, initialize_contact_u=False)
                    recorded = solver._convergence_force_used.numpy()[0]
                    if hasattr(group, "body_entry_name_a"):
                        entry_name = group.body_entry_name_a
                        body = int(group.body_ids_a.numpy()[0])
                        spatial_force = solver._admm_buffers[entry_name].body_f.numpy()[body]
                        applied = spatial_force[3:] if "angular" in name else spatial_force[:3]
                        if hasattr(group, "frame_a"):
                            pose = solver._entries[entry_name].state_0.body_q.list()[body]
                            rotation = wp.transform_get_rotation(pose * group.frame_a.list()[0])
                            recorded = np.array(wp.quat_rotate(rotation, wp.vec3(*recorded)))
                    elif hasattr(group, "body_entry_name"):
                        body = int(group.body_ids.numpy()[0])
                        applied = solver._admm_buffers[group.body_entry_name].body_f.numpy()[body, :3]
                        if hasattr(group, "body_sign"):
                            recorded = recorded * int(group.body_sign.numpy()[0])
                    else:
                        particle = int(group.particle_ids_a.numpy()[0])
                        applied = solver._admm_buffers[group.particle_entry_name_a].particle_f.numpy()[particle]
                    np.testing.assert_allclose(recorded, applied, atol=1e-6, rtol=1e-6)
                    solver._check_convergence(2)
                    test.assertEqual(int(solver.converged.numpy()[0]), 1)
                    group.lambda_.fill_(wp.vec3(1.0, -1.5, 3.0))
                    solver._check_convergence(2)
                    test.assertEqual(int(solver.converged.numpy()[0]), 0)
                    exercised.add(name)
    test.assertSetEqual(exercised, declared, "Add a populated test scene for each ADMM group collection")


def test_empty_and_schedule(test, device):
    """Handle an empty interface and check only at the configured iterations."""
    model = newton.ModelBuilder().finalize(device=device)
    solver = SolverCoupledADMM(
        model,
        [],
        SolverCoupledADMM.Config(iterations=1, convergence=SolverCoupledADMM.ConvergenceConfig(min_iterations=1)),
    )
    solver.step(model.state(), model.state(), None, None, 0.001)
    test.assertEqual(int(solver.iteration_count.numpy()[0]), 1)
    test.assertEqual(int(solver.converged.numpy()[0]), 1)
    strict = SolverCoupledADMM.ConvergenceConfig(
        linear_velocity_tolerance=0.0,
        force_tolerance=0.0,
        force_relative_tolerance=0.0,
        min_iterations=2,
        check_interval=4,
    )
    model, solver = _make_solver(device, strict, iterations=9, velocity=0.1)
    checked = []
    check = solver._check_convergence

    def record(iterations):
        checked.append(iterations)
        check(iterations)

    solver._check_convergence = record
    solver.step(model.state(), model.state(), None, None, 0.001)
    test.assertEqual(checked, [2, 6, 9])


def test_mixed_group_offsets(test, device):
    """Reduce failures across unequal groups and preserve angular tolerances."""
    descriptors = []
    indices = []
    for count, angular, revolute in ((2, False, False), (3, True, True), (1, True, False)):
        group, _, _, _, _ = _row_buffers(device, count=count, angular=angular, revolute=revolute)
        group.offset = len(indices)
        indices.extend([len(descriptors)] * count)
        descriptors.append(group)
    groups = wp.array(descriptors, dtype=AdmmConvergenceGroup, device=device)
    indices = wp.array(indices, dtype=int, device=device)
    previous = wp.empty(6, dtype=wp.vec3, device=device)
    failed = wp.zeros(1, dtype=int, device=device)
    for group in descriptors:
        wp.launch(snapshot_forces_kernel, dim=6, inputs=[groups, indices, 1.0, previous, failed], device=device)
        values = group.lambda_.numpy()
        values[-1, 1] = 0.1
        group.lambda_.assign(values)
        wp.launch(
            check_convergence_kernel,
            dim=6,
            inputs=[groups, indices, 1.0, 1e-5, 1e-5, 1e-3, 1e-3, 0.0, previous, failed],
            device=device,
        )
        test.assertEqual(int(failed.numpy()[0]), 1)
        group.lambda_.zero_()


class TestAdmmConvergence(unittest.TestCase):
    """Exercise convergence configuration and device execution."""

    def test_invalid_config(self):
        """Reject invalid convergence settings using the existing config validation."""
        base = SolverCoupledADMM.ConvergenceConfig()
        for name in (
            "linear_velocity_tolerance",
            "angular_velocity_tolerance",
            "force_tolerance",
            "torque_tolerance",
            "force_relative_tolerance",
        ):
            for value in (-1.0, float("nan"), float("inf"), 1e39):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    SolverCoupledADMM._validate_config(
                        SolverCoupledADMM.Config(convergence=dataclasses.replace(base, **{name: value}))
                    )
        for name in ("min_iterations", "check_interval"):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError)):
                    SolverCoupledADMM._validate_config(
                        SolverCoupledADMM.Config(convergence=dataclasses.replace(base, **{name: value}))
                    )
        with self.assertRaises(ValueError):
            SolverCoupledADMM._validate_config(SolverCoupledADMM.Config(iterations=1, convergence=base))


for func in (
    test_criteria,
    test_capacity,
    test_stopping,
    test_loaded_stopping,
    test_proximal_common_motion,
    test_proximal_criteria,
    test_proximal_angular_and_joint_units,
    test_proximal_graph_replay,
    test_cap_parity,
    test_graph_replay,
    test_rigid_groups,
    test_rigid_particle_groups,
    test_group_force_consistency,
    test_empty_and_schedule,
    test_mixed_group_offsets,
):
    add_function_test(TestAdmmConvergence, func.__name__, func, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
