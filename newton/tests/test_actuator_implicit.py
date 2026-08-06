# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the coupled implicit effort mode."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.actuators import (
    Actuator,
    ActuatorImplicitOptions,
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    Controller,
    ControllerPD,
    ControllerPID,
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
    actuator.set_effort_mode_implicit(**kwargs)
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


def _two_link_builder(armature: float = 0.0):
    """Builder for a two-link revolute chain (one articulation, two scalar DOFs)."""
    builder = newton.ModelBuilder(gravity=0.0)
    base = builder.add_link(mass=1.5)
    tip = builder.add_link(mass=0.8)
    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
    builder.add_shape_box(tip, hx=0.15, hy=0.1, hz=0.1)
    builder.body_com[base] = wp.vec3(0.3, 0.0, 0.0)
    builder.body_com[tip] = wp.vec3(0.25, 0.0, 0.0)
    j0 = builder.add_joint_revolute(parent=-1, child=base, axis=newton.Axis.Z, armature=armature)
    j1 = builder.add_joint_revolute(
        parent=base,
        child=tip,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
        armature=armature,
    )
    builder.add_articulation([j0, j1])
    return builder


def _build_two_link(device):
    """Two-link revolute chain (one articulation, two scalar DOFs)."""
    return _two_link_builder().finalize(device=device)


def _alpha_reference(model, state):
    """alpha_i = (H^{-1})_{ii} from a fresh dense mass-matrix evaluation."""
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    H = newton.eval_mass_matrix(model, state)
    H_np = H.numpy()
    n = model.joint_dof_count
    inv = np.linalg.inv(H_np[0, :n, :n])
    return np.diag(inv).copy()


def test_provider_matches_inverse_mass(test, device):
    """The oracle's block diagonal equals (H^{-1})_{ii} per model DOF."""
    model = _build_single_revolute(device)
    state = model.state()

    provider = ResponseOracle(model)
    provider.refresh(state)
    n = model.joint_dof_count
    alpha = np.diag(provider.inverse_blocks.numpy()[0, :n, :n])
    test.assertEqual(alpha.shape, (n,))

    alpha_ref = _alpha_reference(model, state)
    test.assertAlmostEqual(alpha[0], alpha_ref[0], places=5)


def test_inverse_blocks_match_dense_inverse(test, device):
    """refresh() fills the full per-articulation inverse mass block.

    inverse_blocks[a] must equal inv(H_a).
    """
    model = _build_two_link(device)
    n = model.joint_dof_count
    state = model.state()
    state.joint_q.assign(np.array([0.3, -0.8], dtype=np.float32))

    oracle = ResponseOracle(model)
    oracle.refresh(state)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    H = newton.eval_mass_matrix(model, state).numpy()[0, :n, :n]
    Hinv = np.linalg.inv(H)

    block = oracle.inverse_blocks.numpy()[0, :n, :n]
    np.testing.assert_allclose(block, Hinv, rtol=1e-4, atol=1e-6)


def test_direct_write_from_solver(test, device):
    """Use a solver-computed inverse mass written straight into the oracle.

    MuJoCo's ``dof_invweight0`` gives the effective inverse mass per DOF; it is
    written onto the articulation block diagonal and the solve reads it directly.
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
    blocks = np.zeros(oracle.inverse_blocks.shape, dtype=np.float32)
    for i, v in enumerate(alpha_mjc):
        blocks[0, i, i] = v
    oracle.inverse_blocks.assign(blocks)

    control.joint_f.zero_()
    actuator.step(state, control, dt=h)  # no refresh(): alpha holds the MuJoCo values

    a = float(alpha_mjc[0])
    e_q = target - q0
    expected_tau = kp_val * e_q / (1.0 + a * h * kd_val + a * h * h * kp_val)
    test.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)


def test_response_from_mujoco_mass_matrix(test, device):
    """Fill the oracle response from MuJoCo's per-step mass matrix.

    MuJoCo rebuilds ``qM`` at the step-start pose every step, so — unlike the
    compile-time ``dof_invweight0`` — its complete inverse tracks inertial
    coupling at the current configuration. This test remaps that inverse to
    Newton DOF order, checks it against the built-in oracle, and drives the
    coupled implicit solve with it.
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
    response_mjc = np.linalg.inv(qM)

    # Remap MuJoCo DOF order to Newton DOF order.
    response_newton = np.zeros((n, n), dtype=np.float32)
    mapping = solver.mjc_dof_to_newton_dof
    if mapping is not None:
        mapping_np = mapping.numpy()[0]
        for mjc_i, newton_i in enumerate(mapping_np):
            for mjc_j, newton_j in enumerate(mapping_np):
                if 0 <= newton_i < n and 0 <= newton_j < n:
                    response_newton[newton_i, newton_j] = response_mjc[mjc_i, mjc_j]
    else:
        response_newton[:] = response_mjc

    # The solver's mass matrix must agree with the oracle's own dense recompute.
    oracle_ref = ResponseOracle(model)
    oracle_ref.refresh(state)
    np.testing.assert_allclose(response_newton, oracle_ref.inverse_blocks.numpy()[0, :n, :n], rtol=1e-4)

    # Drive the implicit solve with the solver-provided values (no refresh()).
    oracle = ResponseOracle(model)
    blocks = np.zeros(oracle.inverse_blocks.shape, dtype=np.float32)
    blocks[0, :n, :n] = response_newton
    oracle.inverse_blocks.assign(blocks)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
    )
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)

    f0 = kp * (target - q0)
    jacobian = np.eye(n) + h * np.diag(h * kp + kd) @ response_newton
    expected = np.linalg.solve(jacobian, h * f0) / h
    np.testing.assert_allclose(control.joint_f.numpy(), expected, rtol=1e-3, atol=1e-3)


def test_prediction_matches_featherstone_step(test, device):
    """The state the solve predicts is the state the solver actually reaches.

    The whole scheme rests on ``qd(p) = qd + A p`` and ``q(p) = q + h qd(p)``.
    With no gravity and zero initial velocity that is exactly what semi-implicit
    Euler produces, so applying the solved effort through Featherstone must land
    on the predicted velocity and position.
    """
    h = 0.01
    kp = np.array([300.0, 200.0], dtype=np.float32)
    kd = np.array([3.0, 2.0], dtype=np.float32)
    q0 = np.array([0.3, -0.8], dtype=np.float32)
    target = np.array([0.6, 0.4], dtype=np.float32)

    model = _build_two_link(device)
    n = model.joint_dof_count
    state_in, state_out = model.state(), model.state()
    state_in.joint_q.assign(q0)
    control = model.control()
    control.joint_target_q.assign(target)

    oracle = ResponseOracle(model)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
    )
    oracle.refresh(state_in)
    control.joint_f.zero_()
    actuator.step(state_in, control, dt=h)
    tau = control.joint_f.numpy().copy()

    # What the solve assumed the step would do.
    A = oracle.inverse_blocks.numpy()[0, :n, :n]
    qd_pred = A @ (h * tau)
    q_pred = q0 + h * qd_pred

    # What the solver actually does with that effort.
    solver = newton.solvers.SolverFeatherstone(model)
    solver.step(state_in, state_out, control, None, dt=h)

    np.testing.assert_allclose(state_out.joint_qd.numpy(), qd_pred, rtol=2e-3, atol=1e-5)
    np.testing.assert_allclose(state_out.joint_q.numpy(), q_pred, rtol=2e-3, atol=1e-6)


def test_full_loop_response_from_mujoco_matches_refresh(test, device):
    """Closed-loop run with the coupled response from MuJoCo's qM.

    Runs the same simulation twice, updating the oracle response every step
    either from the solver's own mass matrix (``mjw_data.qM``, device-side
    Cholesky kernel — the "solver-owned oracle" path) or with the built-in
    ``oracle.refresh()``. The refresh is scheduled at the same one-step-stale
    phase as the qM data, so the trajectories must coincide. On CUDA the
    whole step (actuator + solver + response update) is graph-captured.
    """
    from newton._src.actuators.response_oracle import _inverse_block_from_mass_matrix_kernel  # noqa: PLC0415

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
        art_count = wp.array([nv], dtype=wp.int32, device=device)
        chol_l = wp.zeros((1, qm_pad, qm_pad), dtype=wp.float32, device=device)

        def update_response(state_prev):
            if use_qm:
                # Full inverse response from the solver's matrix at the pose
                # of the step that just ran — same staleness as refresh(state_prev).
                wp.launch(
                    _inverse_block_from_mass_matrix_kernel,
                    dim=1,
                    inputs=[solver.mjw_data.qM, art_count, chol_l],
                    outputs=[oracle.inverse_blocks],
                    device=device,
                )
            else:
                oracle.refresh(state_prev)

        def two_steps():
            for _ in range(2):  # even count: state buffers line up for graph replay
                control.joint_f.zero_()
                actuator.step(states[0], control, dt=h)
                solver.step(states[0], states[1], control, None, h)
                update_response(states[0])  # states[0] still holds the pre-step pose
                states[0], states[1] = states[1], states[0]

        def reset():
            states[0].joint_q.assign(q_init)
            states[0].joint_qd.zero_()
            newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
            oracle.refresh(states[0])  # prime the response at the initial pose

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


def test_pd_coupled_solve_matches_reference(test, device):
    """The implicit solve on a coupled chain matches the dense block reference.

    For a PD actuator driving both DOFs of a two-link chain, the solve couples
    them through A = inv(H). The result must equal the NumPy solution
    J p = h f0 (J = I + h(h Kp + Kd) A).
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

    # Dense reference: coupled solve with A = inv(H).
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
        options=ActuatorImplicitOptions(max_iters=1, warm_start="zero"),
    )
    oracle.refresh(state)
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)
    np.testing.assert_allclose(control.joint_f.numpy(), tau_ref, rtol=1e-3, atol=1e-3)


def test_coupled_solve_clamp_in_residual(test, device):
    """The clamp is composed into the coupled residual, not applied afterwards.

    A tight max-effort clamp on DOF 0 of a two-link block must bind exactly at
    the limit, and — because the clamp lives inside the block Newton — DOF 1
    must re-solve against the *clamped* DOF-0 impulse through the off-diagonal
    coupling. A post-hoc clamp would leave DOF 1 at its unclamped value, so the
    test asserts DOF 1 both moves away from that value and matches the analytic
    solution of the pinned system.
    """
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

    # Coupled response A = inv(H) at this pose.
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    H = newton.eval_mass_matrix(model, state).numpy()[0, :2, :2]
    A = np.linalg.inv(H)

    # Unclamped block solve, to size a binding limit on DOF 0.
    oracle = ResponseOracle(model)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
    )
    oracle.refresh(state)
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)
    unclamped = control.joint_f.numpy().copy()
    limit = 0.5 * abs(unclamped[0])

    clamped = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        max_effort=np.array([limit, 1.0e6], dtype=np.float32),
        effective_inv_mass=oracle,
    )
    oracle.refresh(state)
    control.joint_f.zero_()
    clamped.step(state, control, dt=h)
    joint_f = control.joint_f.numpy()

    # DOF 0 binds exactly at the limit (same sign as the unclamped force).
    test.assertAlmostEqual(abs(joint_f[0]), limit, delta=limit * 1e-3)

    # Analytic DOF-1 solve with DOF 0 pinned at its clamped impulse p0 = h*tau0.
    # qd1(p) = A10*p0 + A11*p1, q1(p) = q0[1] + h*qd1; PD with qd0 = target_vel = 0:
    #   tau1 = [kp1*(t1 - q0[1]) - (h*kp1 + kd1)*A10*p0] / (1 + h*(h*kp1 + kd1)*A11)
    tau0_clamped = np.sign(unclamped[0]) * limit
    p0 = h * tau0_clamped
    g1 = h * kp[1] + kd[1]
    tau1_ref = (kp[1] * (target[1] - q0[1]) - g1 * A[1, 0] * p0) / (1.0 + h * g1 * A[1, 1])
    test.assertAlmostEqual(joint_f[1], tau1_ref, delta=abs(tau1_ref) * 1e-3)

    # And DOF 1 genuinely responded to DOF 0's saturation (a post-hoc clamp would not).
    test.assertGreater(abs(joint_f[1] - unclamped[1]), abs(unclamped[1]) * 1e-3)


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
    """The coupled solve uses the correct articulation-local indices."""
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

    oracle = actuator._test_oracle
    response = oracle.inverse_blocks.numpy()[0, :2, :2]
    f0 = kp * (target - q0)
    jacobian = np.eye(2) + h * np.diag(h * kp + kd) @ response
    expected = np.linalg.solve(jacobian, h * f0) / h
    np.testing.assert_allclose(control.joint_f.numpy(), expected, rtol=1e-3, atol=1e-3)


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
    control.raw_joint_f = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    actuator_clamped.control_computed_output_attr = "raw_joint_f"
    _step(actuator_clamped, state, control, h)
    clamped = float(control.joint_f.numpy()[0])
    computed = float(control.raw_joint_f.numpy()[0])
    test.assertAlmostEqual(clamped, max_effort, delta=max_effort * 1e-5)
    alpha = float(_alpha_reference(model, state)[0])
    qd_pred = alpha * h * clamped
    q_pred = q0 + h * qd_pred
    expected_computed = kp_val * (target - q_pred) - kd_val * qd_pred
    test.assertAlmostEqual(computed, expected_computed, delta=abs(expected_computed) * 1e-4)
    test.assertGreater(computed, clamped)


def test_live_param_update_through_views(test, device):
    """Writes through the usual parameter attributes reach the installed implicit solve.

    After set_effort_mode_implicit, controller and clamp parameter attributes
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
    actuator.set_effort_mode_implicit(effective_inv_mass=oracle)
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
    # Steep table, so the predicted-position limit is far from the current one.
    lookup_positions = (0.0, 1.0)
    lookup_efforts = (200.0, 0.0)

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
    actuator.set_effort_mode_implicit(effective_inv_mass=oracle)
    actuator._test_oracle = oracle
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    tau = float(control.joint_f.numpy()[0])

    # The limit is the actuator's own lookup table, interpolated at position.
    def limit_at(q):
        return float(np.interp(q, lookup_positions, lookup_efforts))

    # Self-consistent saturated solution via fixed point in numpy:
    # tau = limit(q_p) with q_p = q0 + h * (qd0 + alpha*h*tau), qd0 = 0.
    alpha = float(_alpha_reference(model, state)[0])
    tau_ref = limit_at(q0)
    for _ in range(50):
        q_p = q0 + h * (alpha * h * tau_ref)
        tau_ref = limit_at(q_p)
    test.assertAlmostEqual(tau, tau_ref, delta=abs(tau_ref) * 1e-4)
    # Below the current-position limit, which is what a post-hoc clamp would give.
    test.assertLess(tau, limit_at(q0) * (1.0 - 1e-3))


def test_effort_mode_switch_roundtrip(test, device):
    """Effort modes are interchangeable at runtime: implicit -> explicit -> implicit."""
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

    actuator.set_effort_mode_explicit()
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    explicit_tau = float(control.joint_f.numpy()[0])
    test.assertAlmostEqual(explicit_tau, kp_val * (target - q0), delta=1e-3)
    test.assertLess(implicit_tau, explicit_tau)

    actuator.set_effort_mode_implicit(effective_inv_mass=actuator._test_oracle)
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    test.assertAlmostEqual(float(control.joint_f.numpy()[0]), implicit_tau, delta=abs(implicit_tau) * 1e-5)


def test_neural_mlp_implicit_linear_net(test, device):
    """A 1-layer (linear) neural controller solves implicitly, exact.

    For a linear net (tau = w0*pos_err + w1*vel_err + b) the linearization is
    the net itself, so the solve matches the analytic Stable-PD solution with
    kp = w0, kd = w1, const = b.
    """
    import tempfile  # noqa: PLC0415

    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    from newton.actuators import ControllerNeuralMLP  # noqa: PLC0415

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
        actuator.set_effort_mode_implicit(effective_inv_mass=oracle)
        test.assertTrue(actuator.is_graphable())

        oracle.refresh(state)
        state_a, state_b = actuator.state(), actuator.state()
        control.joint_f.zero_()
        actuator.step(state, control, state_a, state_b, dt=h)

        alpha = _alpha_reference(model, state)[0]
        e_q = target - q0
        expected_tau = (w0 * e_q + b) / (1.0 + alpha * h * w1 + alpha * h * h * w0)
        test.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)


def test_neural_mlp_implicit_nonlinear_linearized(test, device):
    """A nonlinear neural controller enters the solve as a linearization.

    Builds a 2-layer ELU net. Implicit actuation linearizes it once about the
    current state, ``tau ~= tau0 + a*(q-q0) + b*(qd-qd0)`` with
    ``a = d(tau)/dq``, ``b = d(tau)/dqd``, then solves the resulting linear
    Stable-PD system exactly. The solved effort must match the closed-form
    solution of that linear system.
    """
    import tempfile  # noqa: PLC0415

    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    from newton.actuators import ControllerNeuralMLP  # noqa: PLC0415

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

    def dnet_np(e_q, e_qd):
        # d(net)/d(e_q), d(net)/d(e_qd); ELU'(x) = 1 (x>=0) else exp(x)
        hl = w1 @ np.array([e_q, e_qd], dtype=np.float32) + b1
        elu_p = np.where(hl >= 0.0, 1.0, np.exp(hl))
        g = w2[0] * elu_p  # (4,)
        return float(g @ w1[:, 0]), float(g @ w1[:, 1])

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    state.joint_qd.assign(np.array([qd0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))
    alpha = _alpha_reference(model, state)[0]

    # Linearize tau(q, qd) = net(target - q, -qd) about (q0, qd0):
    #   a = d(tau)/dq = -d(net)/d(e_q),  b = d(tau)/dqd = -d(net)/d(e_qd).
    tau0 = net_np(target - q0, 0.0 - qd0)
    dneq, dneqd = dnet_np(target - q0, 0.0 - qd0)
    a, b = -dneq, -dneqd
    # Solve p/h from p = h*(tau0 + a*(q(p)-q0) + b*(qd(p)-qd0)),
    #   qd(p) = qd0 + alpha*p, q(p) = q0 + h*qd(p).
    expected_tau = (tau0 + a * h * qd0) / (1.0 - alpha * h * (a * h + b))

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/nonlinear.onnx"
        inits = [numpy_helper.from_array(a_, n) for a_, n in ((w1, "W1"), (b1, "b1"), (w2, "W2"), (b2, "b2"))]
        n1 = helper.make_node("Gemm", ["input", "W1", "b1"], ["hl"], alpha=1.0, beta=1.0, transB=1)
        n2 = helper.make_node("Elu", ["hl"], ["ael"], alpha=1.0)
        n3 = helper.make_node("Gemm", ["ael", "W2", "b2"], ["output"], alpha=1.0, beta=1.0, transB=1)
        x_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])
        y_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
        graph = helper.make_graph([n1, n2, n3], "nonlinear", [x_vi], [y_vi], initializer=inits)
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

        controller = ControllerNeuralMLP(model_path=path)
        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array([0], dtype=wp.uint32, device=device),
            controller=controller,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        actuator.set_effort_mode_implicit(effective_inv_mass=oracle)
        oracle.refresh(state)
        sa, sb = actuator.state(), actuator.state()
        control.joint_f.zero_()
        actuator.step(state, control, sa, sb, dt=h)
        test.assertAlmostEqual(float(control.joint_f.numpy()[0]), expected_tau, delta=abs(expected_tau) * 3e-3)


def test_neural_lstm_implicit_rejected(test, device):
    """LSTM implicit actuation is refused rather than silently degrading.

    Warp-NN's LSTM op drops the input adjoint, so the per-step linearization
    would come back with zero slopes and the solve would reduce to the explicit
    impulse while still paying for the Newton loop. Installing implicit mode must
    raise instead.
    """
    import tempfile  # noqa: PLC0415

    from newton.actuators import ControllerNeuralLSTM  # noqa: PLC0415
    from newton.tests.test_actuators import _build_lstm_onnx  # noqa: PLC0415

    model = _build_single_revolute(device)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/lstm.onnx"
        _build_lstm_onnx(path, hidden_size=8, metadata={"effort_scale": 10.0})
        controller = ControllerNeuralLSTM(model_path=path)
        actuator = Actuator(
            indices=wp.array([0], dtype=wp.uint32, device=device),
            controller=controller,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        test.assertIsNone(controller.bind_params())
        with test.assertRaises(NotImplementedError):
            actuator.set_effort_mode_implicit(effective_inv_mass=ResponseOracle(model))


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
        actuator.set_effort_mode_implicit(effective_inv_mass=ResponseOracle(model))


def test_validation_errors(test, device):
    """A non-ResponseOracle inverse mass and a missing dt raise clearly."""
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
        actuator.set_effort_mode_implicit(effective_inv_mass=None)

    actuator = _make_actuator(model, device, kp=kp, kd=kd)
    with test.assertRaisesRegex(ValueError, "requires dt"):
        actuator.step(model.state(), model.control())


def test_pid_implicit_matches_reference(test, device):
    """Implicit PID: full effort solved implicitly, integral computed upfront.

    With the integral advanced from the current-step error and folded in as a
    constant, one step must match the closed-form Stable-PD solution
    ``tau = e_q*(kp + ki*h) / (1 + alpha*h*(kp*h + kd))`` and the integral state
    must advance to ``e_q*h``.
    """
    h = 0.01
    kp_val, ki_val, kd_val = 400.0, 50.0, 6.0
    q0, target = 0.2, 1.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    oracle = ResponseOracle(model)
    controller = ControllerPID(
        kp=wp.array([kp_val], dtype=float, device=device),
        ki=wp.array([ki_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
        integral_max=wp.array([1.0e9], dtype=float, device=device),
    )
    actuator = Actuator(
        indices=wp.array([0], dtype=wp.uint32, device=device),
        controller=controller,
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    actuator.set_effort_mode_implicit(effective_inv_mass=oracle)
    test.assertTrue(actuator.is_graphable())

    oracle.refresh(state)
    sa, sb = actuator.state(), actuator.state()
    control.joint_f.zero_()
    actuator.step(state, control, sa, sb, dt=h)

    alpha = _alpha_reference(model, state)[0]
    e_q = target - q0
    expected = e_q * (kp_val + ki_val * h) / (1.0 + alpha * h * (kp_val * h + kd_val))
    test.assertAlmostEqual(control.joint_f.numpy()[0], expected, delta=abs(expected) * 1e-4)
    test.assertAlmostEqual(sb.controller_state.integral.numpy()[0], e_q * h, delta=abs(e_q * h) * 1e-5)


def test_pid_implicit_multistep_antiwindup(test, device):
    """The integral accumulates across steps and saturates at integral_max.

    Held at a fixed pose, the integral grows by ``e_q*h`` each step until it
    hits ``integral_max`` (anti-windup binds). Each step's effort matches the
    Stable-PD solution with that step's (possibly saturated) integral folded in:
    ``tau = (kp*e_q + ki*I_k) / (1 + alpha*h*(kp*h + kd))``.
    """
    h = 0.01
    kp_val, ki_val, kd_val = 400.0, 50.0, 6.0
    q0, target = 0.2, 1.0
    e_q = target - q0
    integral_max = 0.02  # e_q*h = 0.008 per step, so anti-windup binds on step 3

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    oracle = ResponseOracle(model)
    controller = ControllerPID(
        kp=wp.array([kp_val], dtype=float, device=device),
        ki=wp.array([ki_val], dtype=float, device=device),
        kd=wp.array([kd_val], dtype=float, device=device),
        integral_max=wp.array([integral_max], dtype=float, device=device),
    )
    actuator = Actuator(
        indices=wp.array([0], dtype=wp.uint32, device=device),
        controller=controller,
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    actuator.set_effort_mode_implicit(effective_inv_mass=oracle)
    oracle.refresh(state)  # pose is fixed, so alpha is constant across steps
    alpha = _alpha_reference(model, state)[0]
    denom = 1.0 + alpha * h * (kp_val * h + kd_val)

    # sa is the current state (integral_prev), sb receives the advanced integral.
    sa, sb = actuator.state(), actuator.state()
    for k in range(5):
        control.joint_f.zero_()
        actuator.step(state, control, sa, sb, dt=h)
        integral = min((k + 1) * e_q * h, integral_max)  # accumulate, then saturate
        expected = (kp_val * e_q + ki_val * integral) / denom
        test.assertAlmostEqual(sb.controller_state.integral.numpy()[0], integral, delta=abs(integral) * 1e-4)
        test.assertAlmostEqual(control.joint_f.numpy()[0], expected, delta=abs(expected) * 1e-4)
        sa, sb = sb, sa  # next step reads the integral just written

    # Anti-windup genuinely bound: the final integral is capped below raw accumulation.
    final_integral = float(sa.controller_state.integral.numpy()[0])
    test.assertLess(final_integral, 5 * e_q * h)
    test.assertAlmostEqual(final_integral, integral_max, delta=integral_max * 1e-4)


def test_selection_api_updates_implicit_solve(test, device):
    """Writing a gain through the selection API reaches the installed implicit solve.

    ``set_actuator_parameter`` scatters into the packed parameter views that
    ``set_effort_mode_implicit`` binds, so the next solve must use the new gain.
    This exercises the masked-scatter write path, not the direct ``.assign`` one.
    """
    from newton._src.utils.selection import ArticulationView  # noqa: PLC0415

    h = 0.01
    kp1, kp2, kd_val = 500.0, 2000.0, 5.0
    q0, target = 0.2, 1.0

    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([q0], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([target], dtype=np.float32))

    oracle = ResponseOracle(model)
    actuator = Actuator(
        indices=wp.array([0], dtype=wp.uint32, device=device),
        controller=ControllerPD(
            kp=wp.array([kp1], dtype=float, device=device),
            kd=wp.array([kd_val], dtype=float, device=device),
        ),
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    actuator.set_effort_mode_implicit(effective_inv_mass=oracle)

    view = ArticulationView(model, "*", verbose=False)
    np.testing.assert_allclose(view.get_actuator_parameter(actuator, actuator.controller, "kp").numpy(), [[kp1]])
    view.set_actuator_parameter(actuator, actuator.controller, "kp", wp.array([[kp2]], dtype=float, device=device))
    np.testing.assert_allclose(actuator.controller.kp.numpy(), [kp2])

    def expected(kp):
        alpha = _alpha_reference(model, state)[0]
        return kp * (target - q0) / (1.0 + alpha * h * kd_val + alpha * h * h * kp)

    oracle.refresh(state)
    control.joint_f.zero_()
    actuator.step(state, control, dt=h)
    test.assertAlmostEqual(control.joint_f.numpy()[0], expected(kp2), delta=abs(expected(kp2)) * 1e-4)


def test_singular_jacobian_stays_finite(test, device):
    """A degenerate Jacobian must not leak Inf/NaN into the effort.

    ``derivative_floor`` bounds the pivot used by the elimination; the same
    floored value has to reach the back-substitution divide, otherwise a
    vanishing diagonal produces a non-finite impulse. Driving kp/kd to zero makes
    the residual flat in the clamped region, so the solve leans on that floor.
    """
    h = 0.01
    model = _build_single_revolute(device)
    state = model.state()
    state.joint_q.assign(np.array([0.2], dtype=np.float32))
    control = model.control()
    control.joint_target_q.assign(np.array([1.0], dtype=np.float32))

    oracle = ResponseOracle(model)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.zeros(model.joint_dof_count, dtype=float, device=device),
        kd=wp.zeros(model.joint_dof_count, dtype=float, device=device),
        effective_inv_mass=oracle,
        options=ActuatorImplicitOptions(derivative_floor=1.0e-8),
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    test.assertTrue(np.all(np.isfinite(control.joint_f.numpy())))


def test_implicit_beats_explicit_at_stiff_gains(test, device):
    """The reason implicit mode exists: explicit diverges, implicit does not.

    Same model, same stiff gains, same timestep, stepped through a solver.
    """
    h = 0.01
    kp_val, kd_val, target, steps = 5.0e4, 300.0, 1.0, 300

    def run(implicit):
        model = _build_single_revolute(device)
        s_in, s_out = model.state(), model.state()
        control = model.control()
        control.joint_target_q.assign(np.array([target], dtype=np.float32))
        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array(np.arange(model.joint_dof_count, dtype=np.uint32), device=device),
            controller=ControllerPD(
                kp=wp.array([kp_val], dtype=float, device=device),
                kd=wp.array([kd_val], dtype=float, device=device),
            ),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        if implicit:
            actuator.set_effort_mode_implicit(effective_inv_mass=oracle)
        solver = newton.solvers.SolverFeatherstone(model)
        for _ in range(steps):
            control.joint_f.zero_()
            oracle.refresh(s_in)
            actuator.step(s_in, control, dt=h)
            solver.step(s_in, s_out, control, None, dt=h)
            s_in, s_out = s_out, s_in
        return float(s_in.joint_q.numpy()[0])

    explicit_q = run(implicit=False)
    implicit_q = run(implicit=True)
    test.assertFalse(np.isfinite(explicit_q) and abs(explicit_q) < 10.0 * target)
    test.assertTrue(np.isfinite(implicit_q))
    test.assertAlmostEqual(implicit_q, target, delta=0.05)


def test_multi_articulation_indexing(test, device):
    """Two articulations in one model solve with their own response blocks.

    Every other test uses a single articulation, so ``art_base`` is always 0 and
    an articulation-local index bug would be invisible. Here the second
    articulation's DOFs start at a nonzero base and carry different gains.
    """
    h = 0.01
    kp = np.array([300.0, 200.0, 500.0, 400.0], dtype=np.float32)
    kd = np.array([3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    q0 = np.array([0.3, -0.8, 0.1, -0.2], dtype=np.float32)
    target = np.array([0.6, 0.4, -0.3, 0.5], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    for _ in range(2):
        builder.add_builder(_two_link_builder())
    model = builder.finalize(device=device)
    test.assertEqual(model.articulation_count, 2)
    n = model.joint_dof_count
    test.assertEqual(n, 4)

    state = model.state()
    state.joint_q.assign(q0)
    control = model.control()
    control.joint_target_q.assign(target)

    oracle = ResponseOracle(model)
    actuator = _make_actuator(
        model,
        device,
        kp=wp.array(kp, dtype=float, device=device),
        kd=wp.array(kd, dtype=float, device=device),
        effective_inv_mass=oracle,
    )
    control.joint_f.zero_()
    _step(actuator, state, control, h)

    # Each articulation is its own 2x2 coupled solve.
    blocks = oracle.inverse_blocks.numpy()
    expected = np.zeros(n, dtype=np.float64)
    for a in range(2):
        sl = slice(2 * a, 2 * a + 2)
        A = blocks[a, :2, :2]
        f0 = kp[sl] * (target[sl] - q0[sl])
        J = np.eye(2) + h * np.diag(h * kp[sl] + kd[sl]) @ A
        expected[sl] = np.linalg.solve(J, h * f0) / h
    np.testing.assert_allclose(control.joint_f.numpy(), expected, rtol=1e-3, atol=1e-3)


def test_refresh_does_not_mutate_state(test, device):
    """``refresh()`` must not write back into the caller's state.

    It runs forward kinematics internally; doing that on the caller's state would
    overwrite ``body_q``/``body_qd``, which are authoritative for
    maximal-coordinate solvers.
    """
    model = _build_two_link(device)
    state = model.state()
    state.joint_q.assign(np.array([0.3, -0.8], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Move the body pose away from FK of joint_q.
    body_q = state.body_q.numpy().copy()
    body_q[:, 0] += 5.0
    state.body_q.assign(body_q)

    ResponseOracle(model).refresh(state)
    np.testing.assert_allclose(state.body_q.numpy(), body_q, rtol=0, atol=0)


def test_armature_enters_the_response(test, device):
    """Joint armature is rotor inertia the solver feels, so it must reduce alpha."""
    q0 = np.array([0.3, -0.8], dtype=np.float32)

    def alpha_for(armature):
        builder = _two_link_builder(armature=armature)
        m = builder.finalize(device=device)
        st = m.state()
        st.joint_q.assign(q0)
        o = ResponseOracle(m)
        o.refresh(st)
        return np.diag(o.inverse_blocks.numpy()[0, :2, :2]).copy()

    bare = alpha_for(0.0)
    with_armature = alpha_for(0.5)
    test.assertTrue(np.all(with_armature < bare))

    # alpha must match a dense inverse of (H + diag(armature)).
    builder = _two_link_builder(armature=0.5)
    m = builder.finalize(device=device)
    st = m.state()
    st.joint_q.assign(q0)
    newton.eval_fk(m, st.joint_q, st.joint_qd, st)
    H = newton.eval_mass_matrix(m, st).numpy()[0, :2, :2] + np.diag([0.5, 0.5])
    np.testing.assert_allclose(with_armature, np.diag(np.linalg.inv(H)), rtol=1e-4)


def test_bind_params_is_idempotent(test, device):
    """Re-binding must not detach the installed solve from later writes."""
    h = 0.01
    kp1, kp2, kd_val = 500.0, 2000.0, 5.0
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
        kd=wp.array([kd_val], dtype=float, device=device),
    )

    def expected(kp):
        alpha = _alpha_reference(model, state)[0]
        return kp * (target - q0) / (1.0 + alpha * h * kd_val + alpha * h * h * kp)

    pack = actuator.controller.bind_params()
    test.assertIs(actuator.controller.bind_params(), pack)

    actuator.controller.kp.assign(np.array([kp2], dtype=np.float32))
    control.joint_f.zero_()
    _step(actuator, state, control, h)
    test.assertAlmostEqual(control.joint_f.numpy()[0], expected(kp2), delta=abs(expected(kp2)) * 1e-4)


def test_validation_rejects_bad_options(test, device):
    """dt and warm_start are validated rather than silently misbehaving."""
    model = _build_single_revolute(device)
    kp = wp.array([100.0], dtype=float, device=device)
    kd = wp.array([1.0], dtype=float, device=device)

    with test.assertRaisesRegex(ValueError, "warm_start"):
        _make_actuator(model, device, kp=kp, kd=kd, options=ActuatorImplicitOptions(warm_start="Zero"))

    actuator = _make_actuator(model, device, kp=kp, kd=kd)
    with test.assertRaisesRegex(ValueError, "dt > 0"):
        actuator.step(model.state(), model.control(), dt=0.0)


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
    "test_pd_coupled_solve_matches_reference",
    test_pd_coupled_solve_matches_reference,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_coupled_solve_clamp_in_residual",
    test_coupled_solve_clamp_in_residual,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit, "test_pd_denominator_equivalence", test_pd_denominator_equivalence, devices=devices
)
add_function_test(
    TestActuatorImplicit,
    "test_direct_write_from_solver",
    test_direct_write_from_solver,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_response_from_mujoco_mass_matrix",
    test_response_from_mujoco_mass_matrix,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_prediction_matches_featherstone_step",
    test_prediction_matches_featherstone_step,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_full_loop_response_from_mujoco_matches_refresh",
    test_full_loop_response_from_mujoco_matches_refresh,
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
    TestActuatorImplicit, "test_effort_mode_switch_roundtrip", test_effort_mode_switch_roundtrip, devices=devices
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
        "test_neural_mlp_implicit_nonlinear_linearized",
        test_neural_mlp_implicit_nonlinear_linearized,
        devices=devices,
    )
    add_function_test(
        TestActuatorImplicit,
        "test_neural_lstm_implicit_rejected",
        test_neural_lstm_implicit_rejected,
        devices=devices,
    )
add_function_test(
    TestActuatorImplicit, "test_pid_implicit_matches_reference", test_pid_implicit_matches_reference, devices=devices
)
add_function_test(
    TestActuatorImplicit,
    "test_pid_implicit_multistep_antiwindup",
    test_pid_implicit_multistep_antiwindup,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit,
    "test_selection_api_updates_implicit_solve",
    test_selection_api_updates_implicit_solve,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit, "test_singular_jacobian_stays_finite", test_singular_jacobian_stays_finite, devices=devices
)
add_function_test(
    TestActuatorImplicit,
    "test_implicit_beats_explicit_at_stiff_gains",
    test_implicit_beats_explicit_at_stiff_gains,
    devices=devices,
)
add_function_test(
    TestActuatorImplicit, "test_multi_articulation_indexing", test_multi_articulation_indexing, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_refresh_does_not_mutate_state", test_refresh_does_not_mutate_state, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_armature_enters_the_response", test_armature_enters_the_response, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_bind_params_is_idempotent", test_bind_params_is_idempotent, devices=devices
)
add_function_test(
    TestActuatorImplicit, "test_validation_rejects_bad_options", test_validation_rejects_bad_options, devices=devices
)
add_function_test(TestActuatorImplicit, "test_validation_errors", test_validation_errors, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
