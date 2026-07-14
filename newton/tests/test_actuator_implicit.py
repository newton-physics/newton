# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scalar implicit (Stable-PD) actuation mode."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.actuators import (
    Actuator,
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    Controller,
    ControllerPD,
    ResponseOracle,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_WARP_NN = importlib.util.find_spec("warp_nn") is not None


def _make_actuator(model, device, kp, kd, max_effort=None, **kwargs):
    """Build an implicit Actuator over all DOFs with a PD controller and optional clamp."""
    n = model.joint_dof_count
    indices = wp.array(np.arange(n, dtype=np.uint32), device=device)
    clamping = None
    if max_effort is not None:
        clamping = [ClampingMaxEffort(max_effort=wp.array(max_effort, dtype=float, device=device))]
    eim = kwargs.setdefault("effective_inv_mass", ResponseOracle(model))
    actuator = Actuator(
        indices=indices,
        controller=ControllerPD(kp=kp, kd=kd),
        clamping=clamping,
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    actuator.set_strategy_implicit(**kwargs)
    actuator._test_oracle = eim if isinstance(eim, ResponseOracle) else None
    return actuator


def _step(actuator, state, control, dt):
    """Refresh the actuator's response oracle (if any) at *state*, then step."""
    oracle = getattr(actuator, "_test_oracle", None)
    if oracle is not None:
        oracle.refresh(state)
    actuator.step(state, control, dt=dt)


def _build_single_revolute(device, mass=1.0, com_offset=0.5):
    """Single revolute joint (world -> body) with an offset COM for real inertia."""
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(mass=mass)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.body_com[body] = wp.vec3(com_offset, 0.0, 0.0)
    j = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
    builder.add_articulation([j])
    return builder.finalize(device=device)


def _build_two_link(device):
    """Two-link revolute chain (one articulation, two scalar DOFs)."""
    builder = newton.ModelBuilder(gravity=0.0)
    base = builder.add_link(mass=1.5)
    tip = builder.add_link(mass=0.8)
    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
    builder.add_shape_box(tip, hx=0.15, hy=0.1, hz=0.1)
    builder.body_com[base] = wp.vec3(0.3, 0.0, 0.0)
    builder.body_com[tip] = wp.vec3(0.25, 0.0, 0.0)
    j0 = builder.add_joint_revolute(parent=-1, child=base, axis=newton.Axis.Z)
    j1 = builder.add_joint_revolute(
        parent=base,
        child=tip,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])
    return builder.finalize(device=device)


def _alpha_reference(model, state):
    """alpha_i = (H^{-1})_{ii} from a fresh dense mass-matrix evaluation."""
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    H = newton.eval_mass_matrix(model, state)
    H_np = H.numpy()
    n = model.joint_dof_count
    inv = np.linalg.inv(H_np[0, :n, :n])
    return np.diag(inv).copy()


def test_provider_matches_inverse_mass(test, device):
    """The model-backed oracle returns (H^{-1})_{ii} per model DOF."""
    model = _build_single_revolute(device)
    state = model.state()

    provider = ResponseOracle(model)
    provider.refresh(state)
    alpha = provider.alpha.numpy()
    test.assertEqual(alpha.shape, (model.joint_dof_count,))

    alpha_ref = _alpha_reference(model, state)
    test.assertAlmostEqual(alpha[0], alpha_ref[0], places=5)


def test_inverse_blocks_match_dense_inverse(test, device):
    """refresh(blocks=True) fills the full per-articulation inverse mass block.

    inverse_blocks[a] must equal inv(H_a), and its diagonal must equal alpha.
    """
    model = _build_two_link(device)
    n = model.joint_dof_count
    state = model.state()
    state.joint_q.assign(np.array([0.3, -0.8], dtype=np.float32))

    oracle = ResponseOracle(model)
    oracle.refresh(state, blocks=True)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    H = newton.eval_mass_matrix(model, state).numpy()[0, :n, :n]
    Hinv = np.linalg.inv(H)

    block = oracle.inverse_blocks.numpy()[0, :n, :n]
    np.testing.assert_allclose(block, Hinv, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.diag(block), oracle.alpha.numpy(), rtol=1e-5)


def test_alpha_direct_write_from_solver(test, device):
    """Alternative to refresh(): write solver-computed inverse masses into oracle.alpha.

    Demonstrates the second way to keep the oracle current: instead of
    calling ``oracle.refresh(state)`` (dense recompute from the Newton
    model), take the effective inverse mass the solver already computed —
    here MuJoCo's ``dof_invweight0`` — and write it into ``oracle.alpha``
    in place. The implicit solve reads whatever the buffer holds.
    """
    h = 0.01
    kp_val, kd_val = 500.0, 5.0
    q0, target = 0.2, 1.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    oracle = ResponseOracle(model)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([kp_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
        effective_inv_mass=oracle,
    )

    # MuJoCo's compile-time effective inverse mass per DOF. This one-joint
    # model maps MuJoCo DOF 0 to Newton DOF 0; multi-joint models must remap
    # through the solver's MuJoCo->Newton DOF tables.
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
    alpha_mjc = np.array(solver.mj_model.dof_invweight0, dtype=np.float32)
    oracle.alpha.assign(alpha_mjc)

    control.joint_f.zero_()
    actuator.step(state, control, dt=h)  # no refresh(): alpha holds the MuJoCo values

    a = float(alpha_mjc[0])
    e_q = target - q0
    expected_tau = kp_val * e_q / (1.0 + a * h * kd_val + a * h * h * kp_val)
    test.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)


def test_alpha_from_mujoco_mass_matrix(test, device):
    """Fill oracle.alpha from MuJoCo's per-step mass matrix instead of refresh().

    MuJoCo rebuilds ``qM`` at the step-start pose every step, so — unlike the
    compile-time ``dof_invweight0`` — its inverse diagonal tracks the current
    configuration. This test poses a two-link chain away from the reference
    configuration, extracts ``alpha = diag(qM^-1)`` from the solver (remapped
    to Newton DOF order), checks it against the oracle's own dense recompute,
    and drives the implicit solve with it.
    """
    h = 0.01
    kp = np.array([300.0, 200.0], dtype=np.float32)
    kd = np.array([3.0, 2.0], dtype=np.float32)
    q0 = np.array([0.3, -0.8], dtype=np.float32)  # away from qpos0: alpha differs from invweight0
    target = np.array([0.6, 0.4], dtype=np.float32)

    model = _build_two_link(device)
    state = model.state()
    state.joint_q.assign(q0)
    control = model.control()
    control.joint_target_q.assign(target)

    # One solver step populates qM at the state's pose (computed before integration).
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
    state_out = model.state()
    solver.step(state, state_out, control, None, h)

    n = model.joint_dof_count
    nv = solver.mj_model.nv
    test.assertEqual(nv, n)
    # dense layout for this small model; the allocation is padded, so slice to nv
    qM = solver.mjw_data.qM.numpy()[0][:nv, :nv]
    alpha_mjc = np.diag(np.linalg.inv(qM))

    # Remap MuJoCo DOF order to Newton DOF order.
    alpha_newton = np.zeros(n, dtype=np.float32)
    mapping = solver.mjc_dof_to_newton_dof
    if mapping is not None:
        for mjc_dof, newton_dof in enumerate(mapping.numpy()[0]):
            if 0 <= newton_dof < n:
                alpha_newton[newton_dof] = alpha_mjc[mjc_dof]
    else:
        alpha_newton[:] = alpha_mjc

    # The solver's mass matrix must agree with the oracle's own dense recompute.
    np.testing.assert_allclose(alpha_newton, _alpha_reference(model, state), rtol=1e-4)

    # Drive the implicit solve with the solver-provided values (no refresh()).
    oracle = ResponseOracle(model)
    oracle.alpha.assign(alpha_newton)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
    )
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)

    joint_f = control.joint_f.numpy()
    for i in range(n):
        e_q = target[i] - q0[i]
        a = float(alpha_newton[i])
        expected = kp[i] * e_q / (1.0 + a * h * kd[i] + a * h * h * kp[i])
        test.assertAlmostEqual(joint_f[i], expected, places=3)


def test_alpha_from_featherstone_impulse_probe(test, device):
    """ResponseOracle.refresh_from_forward_dynamics: alpha via unit-impulse probe.

    A unit generalized force at DOF i gives ``qdd = M^-1 e_i``, so
    ``qdd_i = (M^-1)_ii = alpha_i`` — no dense mass matrix formed. Checks it
    matches the dense recompute and drives the implicit solve.
    """
    h = 0.01
    kp = np.array([300.0, 200.0], dtype=np.float32)
    kd = np.array([3.0, 2.0], dtype=np.float32)
    q0 = np.array([0.3, -0.8], dtype=np.float32)  # non-reference pose: alpha is pose-dependent
    target = np.array([0.6, 0.4], dtype=np.float32)

    model = _build_two_link(device)
    n = model.joint_dof_count
    control = model.control()
    control.joint_target_q.assign(target)

    state = model.state()
    state.joint_q.assign(q0)

    solver = newton.solvers.SolverFeatherstone(model)
    oracle = ResponseOracle(model)
    oracle.refresh_from_forward_dynamics(solver, state)

    # The impulse-probe alpha must match the dense diag(M^-1).
    np.testing.assert_allclose(oracle.alpha.numpy(), _alpha_reference(model, state), rtol=1e-4)

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
    )
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)

    alpha = oracle.alpha.numpy()
    joint_f = control.joint_f.numpy()
    for i in range(n):
        e_q = target[i] - q0[i]
        a = float(alpha[i])
        expected = kp[i] * e_q / (1.0 + a * h * kd[i] + a * h * h * kp[i])
        test.assertAlmostEqual(joint_f[i], expected, places=3)


def test_full_loop_alpha_from_mujoco_matches_refresh(test, device):
    """Closed-loop actuator+solver run with per-step alpha from MuJoCo's qM.

    Runs the same simulation twice, updating ``oracle.alpha`` every step
    either from the solver's own mass matrix (``mjw_data.qM``, device-side
    Cholesky kernel — the "solver-owned oracle" path) or with the built-in
    ``oracle.refresh()``. The refresh is scheduled at the same one-step-stale
    phase as the qM data, so the trajectories must coincide. On CUDA the
    whole step (actuator + solver + alpha update) is graph-captured.
    """
    from newton._src.actuators.response_oracle import _alpha_from_mass_matrix_kernel  # noqa: PLC0415

    h = 0.005
    outer_iters = 30
    kp = np.array([300.0, 200.0], dtype=np.float32)
    kd = np.array([3.0, 2.0], dtype=np.float32)
    q_init = np.array([0.3, -0.8], dtype=np.float32)
    target = np.array([0.6, 0.4], dtype=np.float32)

    def run(use_qm):
        model = _build_two_link(device)
        n = model.joint_dof_count
        states = [model.state(), model.state()]
        control = model.control()
        control.joint_target_q.assign(target)

        solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
        oracle = ResponseOracle(model)
        actuator = _make_actuator(
            model,
            device,
            kp=wp.array(kp, dtype=float, device=device),
            kd=wp.array(kd, dtype=float, device=device),
            effective_inv_mass=oracle,
        )

        nv = solver.mj_model.nv
        test.assertEqual(nv, n)
        if solver.mjc_dof_to_newton_dof is not None:
            np.testing.assert_array_equal(solver.mjc_dof_to_newton_dof.numpy()[0], np.arange(n))
        # scratch for the Cholesky kernel run on MuJoCo's (padded) qM
        qm_pad = solver.mjw_data.qM.shape[1]
        art_start = wp.array([0], dtype=wp.int32, device=device)
        art_count = wp.array([nv], dtype=wp.int32, device=device)
        chol_l = wp.zeros((1, qm_pad, qm_pad), dtype=wp.float32, device=device)
        chol_y = wp.zeros((1, qm_pad), dtype=wp.float32, device=device)

        def update_alpha(state_prev):
            if use_qm:
                # alpha = diag(qM^-1) from the solver's matrix (pose of the
                # step that just ran) — same staleness as refresh(state_prev)
                wp.launch(
                    _alpha_from_mass_matrix_kernel,
                    dim=1,
                    inputs=[solver.mjw_data.qM, art_start, art_count, chol_l, chol_y],
                    outputs=[oracle.alpha],
                    device=device,
                )
            else:
                oracle.refresh(state_prev)

        def two_steps():
            for _ in range(2):  # even count: state buffers line up for graph replay
                control.joint_f.zero_()
                actuator.step(states[0], control, dt=h)
                solver.step(states[0], states[1], control, None, h)
                update_alpha(states[0])  # states[0] still holds the pre-step pose
                states[0], states[1] = states[1], states[0]

        def reset():
            states[0].joint_q.assign(q_init)
            states[0].joint_qd.zero_()
            newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
            oracle.refresh(states[0])  # prime alpha at the initial pose

        reset()
        two_steps()  # warm-up: module loads and lazy allocations before capture

        reset()
        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                two_steps()
            step_fn = lambda: wp.capture_launch(capture.graph)  # noqa: E731
        else:
            step_fn = two_steps

        traj = []
        for _ in range(outer_iters):
            step_fn()
            traj.append(states[0].joint_q.numpy().copy())
        return np.array(traj)

    traj_qm = run(use_qm=True)
    traj_ref = run(use_qm=False)
    test.assertTrue(np.all(np.isfinite(traj_qm)))
    np.testing.assert_allclose(traj_qm, traj_ref, atol=1e-4)


def test_pd_block_solve_matches_reference(test, device):
    """Block solve on a coupled chain matches the dense block reference, and differs from scalar.

    For a PD actuator driving both DOFs of a two-link chain, the block solve
    couples them through A = inv(H). The result must equal the numpy block
    solution J p = h f0 (J = I + h(h Kp + Kd) A), and must differ from the
    scalar solve, which drops the off-diagonal (M^-1)_12.
    """
    from newton.actuators import ActuatorImplicitOptions  # noqa: PLC0415

    h = 0.01
    kp = np.array([4000.0, 3000.0], dtype=np.float32)  # stiff, so coupling matters
    kd = np.array([40.0, 30.0], dtype=np.float32)
    q0 = np.array([0.3, -0.8], dtype=np.float32)
    target = np.array([0.6, 0.4], dtype=np.float32)

    model = _build_two_link(device)
    n = model.joint_dof_count
    state = model.state()
    state.joint_q.assign(q0)
    control = model.control()
    control.joint_target_q.assign(target)

    # Dense reference: full block solve with A = inv(H).
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    H = newton.eval_mass_matrix(model, state).numpy()[0, :n, :n]
    A = np.linalg.inv(H)
    f0 = kp * (target - q0)  # qd=0, const=0
    J = np.eye(n) + h * np.diag(h * kp + kd) @ A
    p_ref = np.linalg.solve(J, h * f0)
    tau_ref = p_ref / h

    oracle = ResponseOracle(model)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
        options=ActuatorImplicitOptions(block_solve=True),
    )
    oracle.refresh(state, blocks=True)
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)
    tau_block = control.joint_f.numpy()

    np.testing.assert_allclose(tau_block, tau_ref, rtol=1e-3, atol=1e-3)

    # Scalar solve drops the cross term → different result.
    tau_scalar = np.array(
        [kp[i] * (target[i] - q0[i]) / (1.0 + A[i, i] * h * kd[i] + A[i, i] * h * h * kp[i]) for i in range(n)]
    )
    test.assertGreater(np.max(np.abs(tau_block - tau_scalar)), 1e-2 * np.max(np.abs(tau_block)))


def test_block_solve_clamp_in_residual(test, device):
    """Block solve composes the clamp into the residual per DOF (generic, not PD-specific).

    A tight max-effort clamp on one DOF of a coupled block must bind exactly at
    the limit — the clamp is evaluated inside the block Newton, not applied
    afterwards — while the coupled DOF still solves against the block.
    """
    from newton.actuators import ActuatorImplicitOptions  # noqa: PLC0415

    h = 0.01
    kp = np.array([4000.0, 3000.0], dtype=np.float32)
    kd = np.array([40.0, 30.0], dtype=np.float32)
    q0 = np.array([0.3, -0.8], dtype=np.float32)
    target = np.array([0.6, 0.4], dtype=np.float32)

    model = _build_two_link(device)
    state = model.state()
    state.joint_q.assign(q0)
    control = model.control()
    control.joint_target_q.assign(target)

    # Unclamped block force to size a binding limit on DOF 0.
    oracle = ResponseOracle(model)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
        options=ActuatorImplicitOptions(block_solve=True),
    )
    oracle.refresh(state, blocks=True)
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)
    unclamped0 = float(control.joint_f.numpy()[0])
    limit = 0.5 * abs(unclamped0)

    clamped = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        max_effort=np.array([limit, 1.0e6], dtype=np.float32),
        effective_inv_mass=oracle,
        options=ActuatorImplicitOptions(block_solve=True),
    )
    oracle.refresh(state, blocks=True)
    control.joint_f.zero_()
    clamped.step(state, control, dt=h)
    joint_f = control.joint_f.numpy()
    test.assertAlmostEqual(abs(joint_f[0]), limit, delta=limit * 1e-3)


def test_pd_denominator_equivalence(test, device):
    """joint_f matches the analytic Stable-PD denominator solution."""
    h = 0.01
    kp_val, kd_val = 500.0, 5.0
    q0, target = 0.2, 1.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    state.joint_qd.assign(np.array([0.0], dtype=np.float32))

    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([kp_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)

    alpha = _alpha_reference(model, state)[0]
    e_q = target - q0
    expected_tau = kp_val * e_q / (1.0 + alpha * h * kd_val + alpha * h * h * kp_val)

    test.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)


def test_high_gain_stability(test, device):
    """At extreme stiffness the implicit force stays finite and far below explicit PD."""
    h = 0.01
    kp_val = 1.0e8
    q0, target = 0.0, 0.5

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    state.joint_qd.assign(np.array([0.0], dtype=np.float32))

    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([kp_val], dtype=float, device=device),
        kd=wp.zeros(model.joint_dof_count, dtype=float, device=device),
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)

    tau = control.joint_f.numpy()[0]
    alpha = _alpha_reference(model, state)[0]
    explicit_tau = kp_val * (target - q0)

    test.assertTrue(np.isfinite(tau))
    test.assertLess(abs(tau), 1.0e-3 * explicit_tau)
    # As kp -> inf the impulse drives qd_next so that q_next -> target:
    # tau -> (target - q0) / (alpha * h^2).
    test.assertAlmostEqual(tau, (target - q0) / (alpha * h * h), delta=abs(tau) * 1e-2)


def test_two_link_indexing(test, device):
    """Per-DOF solve uses the correct articulation-local column for both joints."""
    h = 0.01
    kp = np.array([300.0, 200.0], dtype=np.float32)
    kd = np.array([3.0, 2.0], dtype=np.float32)
    q0 = np.array([0.1, -0.2], dtype=np.float32)
    target = np.array([0.6, 0.4], dtype=np.float32)

    model = _build_two_link(device)
    state = model.state()
    state.joint_q.assign(q0)
    state.joint_qd.assign(np.zeros(2, dtype=np.float32))

    control = model.control()
    control.joint_target_q.assign(target)

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)

    alpha = _alpha_reference(model, state)
    joint_f = control.joint_f.numpy()
    for i in range(2):
        e_q = target[i] - q0[i]
        expected = kp[i] * e_q / (1.0 + alpha[i] * h * kd[i] + alpha[i] * h * h * kp[i])
        test.assertAlmostEqual(joint_f[i], expected, places=3)


def test_end_to_end_stable_convergence(test, device):
    """Driving a stiff pendulum through a solver converges and stays bounded."""
    h = 0.01
    kp_val, kd_val = 5.0e4, 300.0
    target = 1.0
    steps = 300

    model = _build_single_revolute(device)
    state_in = model.state()
    state_out = model.state()
    state_in.joint_q.assign(np.array([0.0], dtype=np.float32))
    state_in.joint_qd.assign(np.array([0.0], dtype=np.float32))

    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([kp_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
    )
    solver = newton.solvers.SolverFeatherstone(model)

    max_abs_q = 0.0
    for _ in range(steps):
        control.joint_f.zero_()
        _step(actuator, state_in, control, h)
        solver.step(state_in, state_out, control, None, dt=h)
        state_in, state_out = state_out, state_in
        max_abs_q = max(max_abs_q, abs(float(state_in.joint_q.numpy()[0])))

    q_final = float(state_in.joint_q.numpy()[0])
    test.assertTrue(np.isfinite(q_final))
    # Bounded (no explosion) and converged near the target.
    test.assertLess(max_abs_q, 10.0 * target)
    test.assertAlmostEqual(q_final, target, delta=0.05)


def test_effort_limit_clamps_force(test, device):
    """A max-effort clamp on the actuator bounds the implicit force exactly."""
    h = 0.01
    kp_val, kd_val = 500.0, 5.0
    q0, target = 0.0, 1.0
    max_effort = 10.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    state.joint_qd.assign(np.array([0.0], dtype=np.float32))

    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    # Unclamped reference.
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([kp_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    unclamped = float(control.joint_f.numpy()[0])
    test.assertGreater(abs(unclamped), max_effort)

    control.joint_f.zero_()
    actuator_clamped = _make_actuator(
        model,
        device,
        kp=wp.array([kp_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
        max_effort=np.array([max_effort], dtype=np.float32),
    )
    _step(actuator_clamped, state, control, h)
    clamped = float(control.joint_f.numpy()[0])
    test.assertAlmostEqual(clamped, max_effort, delta=max_effort * 1e-5)


def test_live_param_update_through_views(test, device):
    """Writes through the usual parameter attributes reach the installed implicit solve.

    After set_strategy_implicit, controller and clamp parameter attributes
    are views into the packed kernel arrays: writing to them must change the
    next solve, and reading them back must return the written values.
    """
    h = 0.01
    kp1, kd1 = 500.0, 5.0
    kp2, kd2 = 2000.0, 20.0
    q0, target = 0.2, 1.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([kp1], dtype=float, device=device),
        kd=wp.array([kd1], dtype=float, device=device),
        max_effort=np.array([1.0e6], dtype=np.float32),  # inactive for now
    )

    def expected(kp, kd):
        alpha = _alpha_reference(model, state)[0]
        return kp * (target - q0) / (1.0 + alpha * h * kd + alpha * h * h * kp)

    control.joint_f.zero_()
    _step(actuator, state, control, h)
    test.assertAlmostEqual(control.joint_f.numpy()[0], expected(kp1, kd1), delta=abs(expected(kp1, kd1)) * 1e-4)

    # Update the gains through the controller attributes and step again.
    actuator.controller.kp.assign(np.array([kp2], dtype=np.float32))
    actuator.controller.kd.assign(np.array([kd2], dtype=np.float32))
    np.testing.assert_allclose(actuator.controller.kp.numpy(), [kp2])
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    test.assertAlmostEqual(control.joint_f.numpy()[0], expected(kp2, kd2), delta=abs(expected(kp2, kd2)) * 1e-4)

    # Tighten the clamp limit through the clamp attribute below the PD force.
    limit = 0.5 * expected(kp2, kd2)
    actuator.clamping[0].max_effort.assign(np.array([limit], dtype=np.float32))
    np.testing.assert_allclose(actuator.clamping[0].max_effort.numpy(), [limit], rtol=1e-6)
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    test.assertAlmostEqual(control.joint_f.numpy()[0], limit, delta=abs(limit) * 1e-5)


def test_scatter_add_accumulates(test, device):
    """The implicit force accumulates into joint_f like the explicit pipeline."""
    h = 0.01
    model = _build_single_revolute(device)
    state = model.state()
    control = model.control()
    control.joint_target_q.assign(np.array([1.0], dtype=np.float32))

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([500.0], dtype=float, device=device),
        kd=wp.array([5.0], dtype=float, device=device),
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    once = float(control.joint_f.numpy()[0])
    _step(actuator, state, control, h)
    test.assertAlmostEqual(float(control.joint_f.numpy()[0]), 2.0 * once, delta=abs(once) * 1e-5)


def test_dc_motor_clamp_is_implicit(test, device):
    """The DC-motor envelope is enforced at the predicted end-of-step velocity."""
    h = 0.01
    kp_val = 5.0e4
    qd0 = 1.0
    sat, vel_lim, max_e = 200.0, 2.0, 1.0e6

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([0.0], dtype=np.float32))
    state.joint_qd.assign(np.array([qd0], dtype=np.float32))

    control = model.control()
    control.joint_target_q.assign(np.array([1.0], dtype=np.float32))

    indices = wp.array(np.arange(model.joint_dof_count, dtype=np.uint32), device=device)
    actuator = Actuator(
        indices=indices,
        controller=ControllerPD(
            kp=wp.array([kp_val], dtype=float, device=device),
            kd=wp.zeros(1, dtype=float, device=device),
        ),
        clamping=[
            ClampingDCMotor(
                saturation_effort=wp.array([sat], dtype=float, device=device),
                velocity_limit=wp.array([vel_lim], dtype=float, device=device),
                max_motor_effort=wp.array([max_e], dtype=float, device=device),
            )
        ],
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    oracle = ResponseOracle(model)
    actuator.set_strategy_implicit(effective_inv_mass=oracle)
    actuator._test_oracle = oracle
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    tau = float(control.joint_f.numpy()[0])

    alpha = float(_alpha_reference(model, state)[0])
    # Saturated branch, self-consistent in the end-of-step velocity:
    # tau = sat * (1 - (qd0 + alpha*h*tau) / vel_lim)
    tau_selfconsistent = sat * (1.0 - qd0 / vel_lim) / (1.0 + sat * alpha * h / vel_lim)
    envelope_at_current_qd = sat * (1.0 - qd0 / vel_lim)

    test.assertAlmostEqual(tau, tau_selfconsistent, delta=abs(tau_selfconsistent) * 1e-3)
    # A post-hoc clamp would have allowed the full envelope at the current velocity.
    test.assertLess(tau, 0.8 * envelope_at_current_qd)


def test_position_clamp_is_implicit(test, device):
    """The position-based limit is interpolated at the predicted end-of-step position."""
    h = 0.01
    kp_val = 5.0e4
    q0 = 0.2
    lookup_positions = (0.0, 1.0)
    lookup_efforts = (20.0, 0.0)  # limit(q) = 20 * (1 - q)

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    state.joint_qd.assign(np.array([0.0], dtype=np.float32))

    control = model.control()
    control.joint_target_q.assign(np.array([2.0], dtype=np.float32))

    indices = wp.array(np.arange(model.joint_dof_count, dtype=np.uint32), device=device)
    actuator = Actuator(
        indices=indices,
        controller=ControllerPD(
            kp=wp.array([kp_val], dtype=float, device=device),
            kd=wp.zeros(1, dtype=float, device=device),
        ),
        clamping=[ClampingPositionBased(lookup_positions=lookup_positions, lookup_efforts=lookup_efforts)],
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    oracle = ResponseOracle(model)
    actuator.set_strategy_implicit(effective_inv_mass=oracle)
    actuator._test_oracle = oracle
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    tau = float(control.joint_f.numpy()[0])

    # Self-consistent saturated solution via fixed point in numpy:
    # tau = limit(q_p) with q_p = q0 + h * (qd0 + alpha*h*tau).
    alpha = float(_alpha_reference(model, state)[0])
    tau_ref = 20.0 * (1.0 - q0)
    for _ in range(50):
        q_p = q0 + h * (alpha * h * tau_ref)
        tau_ref = 20.0 * (1.0 - q_p)
    test.assertAlmostEqual(tau, tau_ref, delta=abs(tau_ref) * 1e-3)


def test_strategy_switch_roundtrip(test, device):
    """Strategies are interchangeable at runtime: implicit -> explicit -> implicit."""
    h = 0.01
    kp_val, kd_val = 500.0, 5.0
    q0, target = 0.2, 1.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    actuator = _make_actuator(
        model,
        device,
        kp=wp.array([kp_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    implicit_tau = float(control.joint_f.numpy()[0])

    actuator.set_strategy_explicit()
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    explicit_tau = float(control.joint_f.numpy()[0])
    test.assertAlmostEqual(explicit_tau, kp_val * (target - q0), delta=1e-3)
    test.assertLess(implicit_tau, explicit_tau)

    actuator.set_strategy_implicit(effective_inv_mass=actuator._test_oracle)
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    test.assertAlmostEqual(float(control.joint_f.numpy()[0]), implicit_tau, delta=abs(implicit_tau) * 1e-5)


def test_neural_mlp_implicit_linear_net(test, device):
    """A 1-layer (linear) neural controller solves implicitly, exact in one Newton step.

    For a linear net (tau = w0*pos_err + w1*vel_err + b) the implicit residual
    is linear, so the solve matches the analytic Stable-PD solution with
    kp = w0, kd = w1, const = b, and extra Newton iterations do not change it.
    """
    import tempfile  # noqa: PLC0415

    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    from newton.actuators import ActuatorImplicitOptions, ControllerNeuralMLP  # noqa: PLC0415

    h = 0.01
    w0, w1, b = 400.0, 8.0, 2.5
    q0, target = 0.2, 1.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/linear.onnx"
        W = numpy_helper.from_array(np.array([[w0, w1]], dtype=np.float32), name="W")
        B = numpy_helper.from_array(np.array([b], dtype=np.float32), name="b")
        x_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])
        y_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
        gemm = helper.make_node("Gemm", ["input", "W", "b"], ["output"], alpha=1.0, beta=1.0, transB=1)
        graph = helper.make_graph([gemm], "linear", [x_vi], [y_vi], initializer=[W, B])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

        controller = ControllerNeuralMLP(model_path=path)
        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array([0], dtype=wp.uint32, device=device),
            controller=controller,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        actuator.set_strategy_implicit(effective_inv_mass=oracle)
        test.assertTrue(actuator.is_graphable())

        oracle.refresh(state)
        state_a, state_b = actuator.state(), actuator.state()
        control.joint_f.zero_()
        actuator.step(state, control, state_a, state_b, dt=h)

        alpha = _alpha_reference(model, state)[0]
        e_q = target - q0
        expected_tau = (w0 * e_q + b) / (1.0 + alpha * h * w1 + alpha * h * h * w0)
        test.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)

        # Extra Newton iterations must not change the result for a linear net
        # (Newton converges in one step on a linear residual).
        actuator.set_strategy_implicit(effective_inv_mass=oracle, options=ActuatorImplicitOptions(newton_iters=3))
        control.joint_f.zero_()
        actuator.step(state, control, state_a, state_b, dt=h)
        test.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)


def test_neural_mlp_implicit_nonlinear_converges(test, device):
    """A nonlinear neural controller's implicit solve converges to the true root.

    Builds a 2-layer ELU net and checks that the solved effort satisfies the
    implicit equation ``tau == net(state_predicted_from(tau))`` — i.e. the
    residual goes to zero as the fixed Newton-iteration count grows.
    """
    import tempfile  # noqa: PLC0415

    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    from newton.actuators import ActuatorImplicitOptions, ControllerNeuralMLP  # noqa: PLC0415

    h = 0.01
    q0, qd0, target = 0.2, 0.0, 1.0

    rng = np.random.default_rng(0)
    w1 = (rng.standard_normal((4, 2)) * 3.0).astype(np.float32)
    b1 = (rng.standard_normal(4) * 0.5).astype(np.float32)
    w2 = (rng.standard_normal((1, 4)) * 4.0).astype(np.float32)
    b2 = np.array([3.0], dtype=np.float32)

    def net_np(e_q, e_qd):
        x = np.array([e_q, e_qd], dtype=np.float32)
        hl = w1 @ x + b1
        a = np.where(hl >= 0.0, hl, np.exp(hl) - 1.0)  # ELU, alpha=1
        return float((w2 @ a + b2)[0])

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    state.joint_qd.assign(np.array([qd0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))
    alpha = _alpha_reference(model, state)[0]

    def residual(tau):
        # implicit equation: tau == net evaluated at the state this tau predicts
        qd_pred = qd0 + alpha * h * tau
        q_pred = q0 + h * qd_pred
        return tau - net_np(target - q_pred, 0.0 - qd_pred)

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/nonlinear.onnx"
        inits = [numpy_helper.from_array(a, n) for a, n in ((w1, "W1"), (b1, "b1"), (w2, "W2"), (b2, "b2"))]
        n1 = helper.make_node("Gemm", ["input", "W1", "b1"], ["hl"], alpha=1.0, beta=1.0, transB=1)
        n2 = helper.make_node("Elu", ["hl"], ["ael"], alpha=1.0)
        n3 = helper.make_node("Gemm", ["ael", "W2", "b2"], ["output"], alpha=1.0, beta=1.0, transB=1)
        x_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])
        y_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
        graph = helper.make_graph([n1, n2, n3], "nonlinear", [x_vi], [y_vi], initializer=inits)
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

        def solve(iters):
            controller = ControllerNeuralMLP(model_path=path)
            oracle = ResponseOracle(model)
            actuator = Actuator(
                indices=wp.array([0], dtype=wp.uint32, device=device),
                controller=controller,
                control_target_pos_attr="joint_target_q",
                control_target_vel_attr="joint_target_qd",
            )
            actuator.set_strategy_implicit(
                effective_inv_mass=oracle, options=ActuatorImplicitOptions(newton_iters=iters)
            )
            oracle.refresh(state)
            sa, sb = actuator.state(), actuator.state()
            control.joint_f.zero_()
            actuator.step(state, control, sa, sb, dt=h)
            return float(control.joint_f.numpy()[0])

        # One step already lands near the root; a few iterations drive the
        # implicit residual to (float32) zero and further iterations are stable.
        tau3 = solve(3)
        test.assertLess(abs(residual(tau3)), 1e-2)
        tau8 = solve(8)
        test.assertLess(abs(residual(tau8)), 1e-3)
        test.assertAlmostEqual(tau3, tau8, delta=abs(tau8) * 1e-3)


def test_unsupported_controller_raises(test, device):
    """A controller without evaluate_force is rejected at construction."""

    class _NoForceController(Controller):
        def is_stateful(self):
            return False

        def is_graphable(self):
            return True

    model = _build_single_revolute(device)
    indices = wp.array(np.arange(model.joint_dof_count, dtype=np.uint32), device=device)
    actuator = Actuator(
        indices=indices,
        controller=_NoForceController(),
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    with test.assertRaises(NotImplementedError):
        actuator.set_strategy_implicit(effective_inv_mass=ResponseOracle(model))


def test_validation_errors(test, device):
    """Missing inv-mass / dt and non-scalar joints raise clearly."""
    model = _build_single_revolute(device)
    indices = wp.array(np.arange(model.joint_dof_count, dtype=np.uint32), device=device)
    kp = wp.array([100.0], dtype=float, device=device)
    kd = wp.array([1.0], dtype=float, device=device)

    actuator = Actuator(
        indices=indices,
        controller=ControllerPD(kp=kp, kd=kd),
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    with test.assertRaisesRegex(ValueError, "effective_inv_mass"):
        actuator.set_strategy_implicit(effective_inv_mass=None)

    actuator = _make_actuator(model, device, kp=kp, kd=kd)
    with test.assertRaisesRegex(ValueError, "requires dt"):
        actuator.step(model.state(), model.control())


devices = get_test_devices()


class TestActuatorImplicit(unittest.TestCase):
    pass


add_function_test(
    TestActuatorImplicit, "test_provider_matches_inverse_mass", test_provider_matches_inverse_mass, devices=devices
)
add_function_test(
    TestActuatorImplicit,
    "test_inverse_blocks_match_dense_inverse",
    test_inverse_blocks_match_dense_inverse,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_pd_block_solve_matches_reference",
    test_pd_block_solve_matches_reference,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_block_solve_clamp_in_residual",
    test_block_solve_clamp_in_residual,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit, "test_pd_denominator_equivalence", test_pd_denominator_equivalence, devices=devices
)
add_function_test(
    TestActuatorImplicit,
    "test_alpha_direct_write_from_solver",
    test_alpha_direct_write_from_solver,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_alpha_from_mujoco_mass_matrix",
    test_alpha_from_mujoco_mass_matrix,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_alpha_from_featherstone_impulse_probe",
    test_alpha_from_featherstone_impulse_probe,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_full_loop_alpha_from_mujoco_matches_refresh",
    test_full_loop_alpha_from_mujoco_matches_refresh,
    devices=devices,
)
add_function_test(TestActuatorImplicit, "test_high_gain_stability", test_high_gain_stability, devices=devices)
add_function_test(TestActuatorImplicit, "test_two_link_indexing", test_two_link_indexing, devices=devices)
add_function_test(
    TestActuatorImplicit, "test_end_to_end_stable_convergence", test_end_to_end_stable_convergence, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_effort_limit_clamps_force", test_effort_limit_clamps_force, devices=devices
)
add_function_test(TestActuatorImplicit, "test_scatter_add_accumulates", test_scatter_add_accumulates, devices=devices)
add_function_test(
    TestActuatorImplicit,
    "test_live_param_update_through_views",
    test_live_param_update_through_views,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit, "test_dc_motor_clamp_is_implicit", test_dc_motor_clamp_is_implicit, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_position_clamp_is_implicit", test_position_clamp_is_implicit, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_strategy_switch_roundtrip", test_strategy_switch_roundtrip, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_unsupported_controller_raises", test_unsupported_controller_raises, devices=devices
)
if _HAS_ONNX and _HAS_WARP_NN:
    add_function_test(
        TestActuatorImplicit,
        "test_neural_mlp_implicit_linear_net",
        test_neural_mlp_implicit_linear_net,
        devices=devices,
    )
    add_function_test(
        TestActuatorImplicit,
        "test_neural_mlp_implicit_nonlinear_converges",
        test_neural_mlp_implicit_nonlinear_converges,
        devices=devices,
    )
add_function_test(TestActuatorImplicit, "test_validation_errors", test_validation_errors, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
