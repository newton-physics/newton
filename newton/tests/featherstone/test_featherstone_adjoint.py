# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Featherstone differentiable-path adjoints.

Every test is contact-free and runs only a handful of solver steps, so the
module adds no settle phase and no kernels beyond the Featherstone ones.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.featherstone.kernels import (
    compute_body_parent_f,
    eval_dense_cholesky_batched,
    eval_dense_solve_batched,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

STAND = np.array([0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6], dtype=np.float32)
SIM_DT = 1.0e-3


def _build_model(device, requires_grad=True, kinematic_pendulum=False):
    """Build the floating quadruped (inflated inertias, armature 0.1) on the test device.

    With ``kinematic_pendulum``, a separate one-link articulation whose body
    is kinematic is appended after the quadruped (its DOF comes last)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg.armature = 0.1
    builder.default_joint_cfg.target_ke = 0.0
    builder.default_joint_cfg.target_kd = 0.0
    builder.default_joint_cfg.damping = 5.0
    builder.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
        joint_damping_mode="drive",
    )
    for body in range(builder.body_count):
        inertia_np = np.asarray(builder.body_inertia[body], dtype=np.float32).reshape(3, 3)
        inertia_np += np.eye(3, dtype=np.float32) * 0.01
        builder.body_inertia[body] = wp.mat33(inertia_np)
    builder.joint_q[-12:] = STAND
    if kinematic_pendulum:
        link = builder.add_link(
            mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), is_kinematic=True, label="kinematic_pendulum"
        )
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Y)
        builder.add_articulation([joint])
    return builder.finalize(device=device, requires_grad=requires_grad)


def test_multistep_tape_matches_chained_steps(test, device):
    """Check that one K-step tape gives the same gradient as K chained single-step tapes.

    A single-step tape has nothing to alias, so backpropagating through the
    steps one tape at a time (each seeded with the next step's input
    adjoint) is an exact reference for the shared K-step tape: both run the
    same kernels on bitwise-identical inputs. Before the per-state
    system-matrix scratch, every step of the shared tape rewrote the
    solver-owned J/M/P/H/L buffers, so the earlier steps' adjoints read the
    later steps' matrices: on this scene the K=10 gradient deviated by 3.5%
    of its largest entry (0.5% at K=4). With per-state scratch the two
    gradients agree to the last bit.
    """
    K = 10
    model = _build_model(device)
    solver = newton.solvers.SolverFeatherstone(model)
    rng = np.random.default_rng(0)
    q0 = model.joint_q.numpy().copy()
    q0[7:] += rng.uniform(-0.2, 0.2, 12).astype(np.float32)
    qd0 = rng.uniform(-1.0, 1.0, model.joint_dof_count).astype(np.float32)
    tau = rng.uniform(-20.0, 20.0, model.joint_dof_count).astype(np.float32)
    tau[:6] = 0.0  # actuate the hinges only
    seed_q = rng.standard_normal(model.joint_coord_count).astype(np.float32)
    seed_qd = rng.standard_normal(model.joint_dof_count).astype(np.float32)
    control = model.control(requires_grad=True)
    control.joint_f.assign(tau)

    def make_state(q, qd):
        state = model.state(requires_grad=True)
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return state

    def backward(tape, state, adj_q, adj_qd):
        tape.backward(
            grads={
                state.joint_q: wp.array(adj_q, dtype=float, device=device),
                state.joint_qd: wp.array(adj_qd, dtype=float, device=device),
            }
        )

    # one shared tape over all K steps
    states = [make_state(q0, qd0)] + [model.state(requires_grad=True) for _ in range(K)]
    tape = wp.Tape()
    with tape:
        for t in range(K):
            states[t].clear_forces()
            solver.step(states[t], states[t + 1], control, None, SIM_DT)
    backward(tape, states[K], seed_q, seed_qd)
    shared = np.concatenate([states[0].joint_q.grad.numpy(), states[0].joint_qd.grad.numpy()])
    inputs = [(s.joint_q.numpy(), s.joint_qd.numpy()) for s in states[:K]]

    # the same K steps, one tape per step, chained from the last step back
    adj_q, adj_qd = seed_q, seed_qd
    for t in reversed(range(K)):
        state_in = make_state(*inputs[t])
        state_out = model.state(requires_grad=True)
        step_tape = wp.Tape()
        with step_tape:
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, SIM_DT)
        backward(step_tape, state_out, adj_q, adj_qd)
        adj_q, adj_qd = state_in.joint_q.grad.numpy().copy(), state_in.joint_qd.grad.numpy().copy()
    chained = np.concatenate([adj_q, adj_qd])

    scale = float(np.abs(chained).max())
    test.assertGreater(scale, 1e-3, "chained gradient is zero - harness broken")
    err = float(np.abs(shared - chained).max()) / scale
    test.assertLessEqual(
        err,
        1e-4,
        f"K={K} shared-tape gradient deviates from the chained single-step tapes by {err:.2e} "
        "of its largest entry - tape steps alias the solver's system-matrix scratch",
    )


class TestFeatherstoneAdjoint(unittest.TestCase):
    """Regression tests for the Featherstone differentiable-path adjoints."""


add_function_test(
    TestFeatherstoneAdjoint,
    "test_multistep_tape_matches_chained_steps",
    test_multistep_tape_matches_chained_steps,
    devices=get_test_devices(),
)


@wp.kernel
def _weighted_qd_loss_kernel(qd: wp.array[float], w: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, w[i] * qd[i])


def _check_armature_gradient(test, device, kinematic_pendulum):
    """Compare one step's tape gradient w.r.t. Model.joint_armature with central differences.

    The hinges start off the stand pose with random velocities and torques,
    so every quadruped DOF's armature matters, and each DOF is checked on
    its own. With ``kinematic_pendulum`` the model also holds a kinematic
    articulation, which switches the solver to its kinematic-armature
    substitution; the quadruped's gradients must survive it and the
    kinematic DOF must get none.
    """
    model = _build_model(device, kinematic_pendulum=kinematic_pendulum)
    solver = newton.solvers.SolverFeatherstone(model)
    n_dyn = 18  # the quadruped's DOFs; the kinematic pendulum's DOF comes after them
    rng = np.random.default_rng(1)
    q0 = model.joint_q.numpy().copy()
    q0[7:19] += rng.uniform(-0.2, 0.2, 12).astype(np.float32)
    qd0 = np.zeros(model.joint_dof_count, dtype=np.float32)
    qd0[:n_dyn] = rng.uniform(-1.0, 1.0, n_dyn)
    tau = np.zeros(model.joint_dof_count, dtype=np.float32)
    tau[6:n_dyn] = rng.uniform(-20.0, 20.0, n_dyn - 6)
    w = np.zeros(model.joint_dof_count, dtype=np.float32)
    w[:n_dyn] = rng.standard_normal(n_dyn)
    w_arr = wp.array(w, dtype=float, device=device)
    armature0 = model.joint_armature.numpy().astype(np.float64).copy()

    def make_state(requires_grad):
        state = model.state(requires_grad=requires_grad)
        state.joint_q.assign(q0)
        state.joint_qd.assign(qd0)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return state

    def control():
        ctrl = model.control(requires_grad=True)
        ctrl.joint_f.assign(tau)
        return ctrl

    def step_loss(armature64):
        """One eager step with the given armature; returns the weighted joint-velocity loss."""
        model.joint_armature.assign(armature64.astype(np.float32))
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        st1 = model.state(requires_grad=False)
        solver.step(make_state(False), st1, control(), None, SIM_DT)
        return float(np.dot(w.astype(np.float64), st1.joint_qd.numpy().astype(np.float64)))

    model.joint_armature.grad.zero_()
    st0 = make_state(True)
    st1 = model.state(requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        solver.step(st0, st1, control(), None, SIM_DT)
        wp.launch(
            _weighted_qd_loss_kernel, dim=model.joint_dof_count, inputs=[st1.joint_qd, w_arr, loss], device=device
        )
    tape.backward(loss)
    ana = model.joint_armature.grad.numpy().astype(np.float64).copy()
    tape.zero()

    eps = 1e-2
    fd = np.zeros(n_dyn)
    for c in range(n_dyn):
        ap = armature0.copy()
        ap[c] += eps
        am = armature0.copy()
        am[c] -= eps
        fd[c] = (step_loss(ap) - step_loss(am)) / (2.0 * eps)
    model.joint_armature.assign(armature0.astype(np.float32))
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    scale = float(np.abs(fd).max())
    hinge_resolved = int((np.abs(fd[6:]) > 1e-2 * scale).sum())
    test.assertGreaterEqual(
        hinge_resolved, 8, f"only {hinge_resolved} hinge armature channels resolvable - harness broken"
    )
    for c in range(n_dyn):
        err = abs(ana[c] - fd[c])
        test.assertLessEqual(
            err,
            0.02 * abs(fd[c]) + 1e-4 * scale,
            f"armature DOF {c}: tape {ana[c]:.5e} vs FD {fd[c]:.5e} - the Cholesky-solve adjoint "
            "drops the armature dependence",
        )
    if kinematic_pendulum:
        test.assertEqual(float(ana[n_dyn]), 0.0, "the kinematic DOF's substituted armature must not get a gradient")


def test_armature_gradient(test, device):
    """Verify the joint-armature gradient on the default dense-Cholesky path.

    The armature regularizes the mass-matrix diagonal before the Cholesky
    factorization. Before the fix, the solve adjoint dropped this dependence
    and the tape gradient w.r.t. ``Model.joint_armature`` was exactly zero.
    """
    _check_armature_gradient(test, device, kinematic_pendulum=False)


def test_armature_gradient_with_kinematic_joint(test, device):
    """Verify the dynamic DOFs keep their armature gradient when the model has a kinematic joint.

    Kinematic joints make the solver substitute a huge armature for their
    DOFs; it used to do so with a detached copy of ``Model.joint_armature``,
    which zeroed the gradient for every DOF of the model.
    """
    _check_armature_gradient(test, device, kinematic_pendulum=True)


def test_dense_cholesky_solve_contract(test, device):
    """Check the dense Cholesky/solve adjoint against float64 central differences.

    ``dense_cholesky`` factors the lower triangle of ``A`` plus ``diag(R)``
    and never reads the upper triangle, so the solve adjoint must deposit
    the symmetrized gradient on the lower triangle, nothing on the upper
    triangle, and the diagonal into ``R``. The upper triangle here holds
    values unrelated to the lower one, so a full-matrix deposit fails. The
    factor ``L`` is internal and must get no adjoint.
    """
    n = 6
    rng = np.random.default_rng(7)
    m = rng.standard_normal((n, n))
    spd = m @ m.T + 0.5 * np.eye(n)
    a64 = np.tril(spd) + np.triu(rng.standard_normal((n, n)), 1)  # upper triangle is never read
    r64 = 0.1 + 0.05 * np.abs(rng.standard_normal(n))
    b64 = rng.standard_normal(n)
    g64 = rng.standard_normal(n)

    def phi(a, r, b):
        """g . (lower-symmetrized A + diag(R))^-1 b in float64."""
        low = np.tril(a)
        sym = low + np.tril(low, -1).T + np.diag(r)
        return float(g64 @ np.linalg.solve(sym, b))

    def fd(which, eps=1e-6):
        """Central differences of phi w.r.t. every entry of A, R or b."""
        base = {"a": a64, "r": r64, "b": b64}
        out = np.zeros(base[which].size)
        for c in range(out.size):
            args = {k: v.copy() for k, v in base.items()}
            args[which].flat[c] += eps
            hi = phi(args["a"], args["r"], args["b"])
            args[which].flat[c] -= 2.0 * eps
            out[c] = (hi - phi(args["a"], args["r"], args["b"])) / (2.0 * eps)
        return out

    a = wp.array(a64.astype(np.float32).reshape(-1), dtype=float, requires_grad=True, device=device)
    r = wp.array(r64.astype(np.float32), dtype=float, requires_grad=True, device=device)
    b = wp.array(b64.astype(np.float32), dtype=float, requires_grad=True, device=device)
    w = wp.array(g64.astype(np.float32), dtype=float, device=device)
    lfac = wp.zeros(n * n, dtype=float, requires_grad=True, device=device)
    x = wp.zeros(n, dtype=float, requires_grad=True, device=device)
    tmp = wp.zeros(n, dtype=float, requires_grad=True, device=device)
    starts = wp.array([0], dtype=wp.int32, device=device)
    dims = wp.array([n], dtype=wp.int32, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            eval_dense_cholesky_batched, dim=1, inputs=[starts, dims, starts, a, r], outputs=[lfac], device=device
        )
        wp.launch(
            eval_dense_solve_batched,
            dim=1,
            inputs=[starts, dims, starts, a, r, starts, lfac, b],
            outputs=[x, tmp],
            device=device,
        )
        wp.launch(_weighted_qd_loss_kernel, dim=n, inputs=[x, w, loss], device=device)
    tape.backward(loss)

    upper = np.triu(np.ones((n, n), dtype=bool), 1).reshape(-1)
    grad_a = a.grad.numpy().astype(np.float64)
    test.assertTrue(np.all(grad_a[upper] == 0.0), f"upper-triangle gradient must be zero, got {grad_a[upper]}")
    for name, tape_g, ref in (
        ("A (lower triangle)", grad_a[~upper], fd("a")[~upper]),
        ("R (regularization)", r.grad.numpy().astype(np.float64), fd("r")),
        ("b", b.grad.numpy().astype(np.float64), fd("b")),
    ):
        np.testing.assert_allclose(tape_g, ref, rtol=2e-3, atol=1e-4 * np.abs(ref).max(), err_msg=name)
    test.assertTrue(np.all(lfac.grad.numpy() == 0.0), "the internal Cholesky factor must get no adjoint")


def test_rigid_tau_external_wrench(test, device):
    """Verify gradients through eval_rigid_tau's external-wrench frame shift.

    Before the fix, eval_rigid_tau overwrote its body_f_ext input with the
    shifted wrench, so the recompute-based adjoint replayed the wrong force.
    The wrench-channel gradients survive that defect; the root-pose channels
    of the initial state do not (sign flips, >200% off FD), so they are the
    discriminating check. One step with a nonzero wrench on a non-root body,
    linear loss in hinge velocities, compared against central differences.
    """
    model = _build_model(device)
    solver = newton.solvers.SolverFeatherstone(model)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)

    BODY = 1
    WRENCH = np.array([5.0, -3.0, 2.0, 0.5, -0.25, 0.75], dtype=np.float64)
    C = np.array([0.3, -0.7, 1.1, -0.4, 0.9, -0.2], dtype=np.float64)

    def rollout_loss(wrench64=None, q64=None):
        """One eager solver step under the wrench; returns the hinge-velocity probe loss."""
        # a fresh state's force buffers are already zero - clear_forces()
        # must NOT be called after the assign (it would zero the wrench)
        st0 = model.state()
        st1 = model.state()
        if q64 is not None:
            st0.joint_q.assign(q64.astype(np.float32))
        f = np.zeros((model.body_count, 6), dtype=np.float32)
        f[BODY] = (WRENCH if wrench64 is None else wrench64).astype(np.float32)
        st0.body_f.assign(f)
        ctrl = model.control()
        solver.step(st0, st1, ctrl, None, SIM_DT)
        # loss on a HINGE dof velocity: the gradient must flow through
        # eval_rigid_tau's tau path (a root-body twist loss can bypass it -
        # FREE-joint wrenches are routed separately)
        qd = st1.joint_qd.numpy().astype(np.float64)
        return float(C[0] * qd[9] + C[1] * qd[13] + C[2] * qd[3] + C[3] * qd[16])

    # analytic: one taped step with the wrench requiring grad
    st0 = model.state(requires_grad=True)
    st1 = model.state(requires_grad=True)
    f0 = np.zeros((model.body_count, 6), dtype=np.float32)
    f0[BODY] = WRENCH.astype(np.float32)
    st0.body_f.assign(f0)
    ctrl = model.control(requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        solver.step(st0, st1, ctrl, None, SIM_DT)
        wp.launch(_joint_qd_loss_kernel, dim=1, inputs=[st1.joint_qd, loss], device=device)
    tape.backward(loss)
    ana = st0.body_f.grad.numpy().astype(np.float64)[BODY]
    ana_q = st0.joint_q.grad.numpy().astype(np.float64).copy()
    tape.zero()

    eps = 1e-3
    for c in range(6):
        wp_ = WRENCH.copy()
        wp_[c] += eps
        l1 = rollout_loss(wp_)
        wm = WRENCH.copy()
        wm[c] -= eps
        l0 = rollout_loss(wm)
        fd = (l1 - l0) / (2 * eps)
        test.assertGreater(np.abs(fd), 1e-8, f"wrench channel {c}: FD is zero - harness broken")
        rel = abs(ana[c] - fd) / max(abs(fd), 1e-12)
        test.assertLessEqual(
            rel,
            0.02,
            f"wrench channel {c}: tape {ana[c]:.5e} vs FD {fd:.5e} (rel {rel * 100:.2f}%) - "
            "external-wrench gradient through eval_rigid_tau is corrupted",
        )

    # root-pose channels discriminate the defect; channels whose two FD
    # estimates disagree by more than 15% cannot be judged and are skipped
    q_base = st0.joint_q.numpy().astype(np.float64).copy()
    resolved = 0
    for c in (3, 4, 5, 6):
        fds = []
        for eps_q in (1e-3, 1e-2):
            qp = q_base.copy()
            qp[c] += eps_q
            qm = q_base.copy()
            qm[c] -= eps_q
            fds.append((rollout_loss(q64=qp) - rollout_loss(q64=qm)) / (2.0 * eps_q))
        scale = max(abs(fds[0]), abs(fds[1]), 1e-12)
        if scale < 1e-8 and abs(ana_q[c]) < 1e-8:
            continue  # dead channel: nothing to adjudicate
        spread = abs(fds[0] - fds[1]) / scale
        if spread > 0.15:
            continue  # FD-unresolvable: finite differences cannot judge
        resolved += 1
        rel = abs(ana_q[c] - fds[0]) / scale
        test.assertLessEqual(
            rel,
            max(0.08, spread),
            f"root-pose joint_q[{c}]: tape {ana_q[c]:.5e} vs FD {fds[0]:.5e} (rel {rel * 100:.2f}%, "
            f"spread {spread * 100:.2f}%) - pose gradient through eval_rigid_tau's wrench "
            "frame shift is corrupted (replayed f_ext from the overwritten array)",
        )
    if resolved == 0:
        test.skipTest(
            "all four root-pose channels are dead or FD-unresolvable on this device; "
            "the discriminating root-pose check did not run (wrench channels only)"
        )


def test_body_parent_f_wrench_contract(test, device):
    """Verify compute_body_parent_f shifts the public wrench into the solve frame.

    This guards a contract this change introduces rather than a defect on
    main: eval_rigid_tau no longer leaves the shifted wrench in body_f_ext,
    so compute_body_parent_f now receives the public wrench and must shift
    it itself (main's kernel expects the shifted one, and the solver's
    body_parent_f observable is the same before and after). With zero bias
    and no descendants the transmitted wrench at the COM must read
    (-F, -tau): the solve-frame r x F term cancels against the COM
    re-reference. The input buffer must stay unmodified.
    """
    n = 3
    F = np.array([[1.5, -2.0, 0.75], [-0.4, 0.9, 2.2], [0.6, 0.3, -1.1]], dtype=np.float64)
    TAU = np.array([[0.1, 0.2, -0.3], [0.5, -0.15, 0.05], [-0.25, 0.4, 0.2]], dtype=np.float64)
    COM = np.array([[0.7, -0.35, 1.1], [-1.2, 0.5, 0.3], [0.2, 0.8, -0.9]], dtype=np.float64)
    ORIGIN = np.array([[0.15, 0.4, -0.6], [0.3, -0.2, 0.55], [-0.4, 0.1, 0.7]], dtype=np.float64)
    r = COM - ORIGIN
    test.assertTrue(
        np.all(np.linalg.norm(r, axis=1) > 0.1),
        "harness broken - the COM offsets are not nonzero as the setup requires",
    )

    body_q_com = wp.array(
        [wp.transform(tuple(COM[i]), (0.0, 0.0, 0.0, 1.0)) for i in range(n)],
        dtype=wp.transform,
        device=device,
    )
    body_solve_origin = wp.array(ORIGIN.astype(np.float32), dtype=wp.vec3, device=device)
    body_f_s = wp.zeros(n, dtype=wp.spatial_vector, device=device)
    body_ft_s = wp.zeros(n, dtype=wp.spatial_vector, device=device)
    f_ext_np = np.concatenate([F, TAU], axis=1).astype(np.float32)
    body_f_ext = wp.array(f_ext_np, dtype=wp.spatial_vector, device=device)
    body_parent_f = wp.empty(n, dtype=wp.spatial_vector, device=device)

    wp.launch(
        compute_body_parent_f,
        dim=n,
        inputs=[body_q_com, body_solve_origin, body_f_s, body_ft_s, body_f_ext],
        outputs=[body_parent_f],
        device=device,
    )

    out = body_parent_f.numpy().astype(np.float64)
    for i in range(n):
        test.assertTrue(
            np.allclose(out[i, :3], -F[i], atol=1e-5),
            f"body {i}: linear part {out[i, :3]} != {-F[i]} - the public wrench's "
            "force was not negated before the sum (raw-buffer contract)",
        )
        test.assertTrue(
            np.allclose(out[i, 3:], -TAU[i], atol=1e-5),
            f"body {i}: torque at COM {out[i, 3:]} != {-TAU[i]} - the wrench was not "
            "shifted into the solve frame (the r x F shift term must cancel against "
            "the COM re-reference)",
        )
    test.assertTrue(
        np.array_equal(body_f_ext.numpy().astype(np.float64), f_ext_np.astype(np.float64)),
        "body_f_ext was mutated - the public -> solve-frame shift must stay a "
        "pure local computation (the write-back the adjoint fix retired)",
    )


@wp.kernel
def _joint_qd_loss_kernel(joint_qd: wp.array[float], loss: wp.array[float]):
    loss[0] = 0.3 * joint_qd[9] - 0.7 * joint_qd[13] + 1.1 * joint_qd[3] - 0.4 * joint_qd[16]


def test_non_grad_step_reuses_aux_arrays(test, device):
    """Verify repeated non-grad steps reuse the solver's auxiliary arrays.

    This guards this change's per-state allocation against reallocating on
    every non-grad step; main passes it too."""
    model = _build_model(device, requires_grad=False)
    solver = newton.solvers.SolverFeatherstone(model)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    st0 = model.state()
    st1 = model.state()
    ctrl = model.control()
    solver.step(st0, st1, ctrl, None, SIM_DT)
    test.assertTrue(
        getattr(solver, "_featherstone_augmented", False),
        "non-grad step must mark the solver augmented so the next step reuses the aux arrays",
    )
    body_f_ext = solver.body_f_ext
    solver.step(st1, st0, ctrl, None, SIM_DT)
    test.assertIs(
        body_f_ext,
        solver.body_f_ext,
        "second non-grad step reallocated the state-side aux arrays",
    )


add_function_test(
    TestFeatherstoneAdjoint,
    "test_armature_gradient",
    test_armature_gradient,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherstoneAdjoint,
    "test_armature_gradient_with_kinematic_joint",
    test_armature_gradient_with_kinematic_joint,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherstoneAdjoint,
    "test_dense_cholesky_solve_contract",
    test_dense_cholesky_solve_contract,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherstoneAdjoint,
    "test_rigid_tau_external_wrench",
    test_rigid_tau_external_wrench,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherstoneAdjoint,
    "test_body_parent_f_wrench_contract",
    test_body_parent_f_wrench_contract,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherstoneAdjoint,
    "test_non_grad_step_reuses_aux_arrays",
    test_non_grad_step_reuses_aux_arrays,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
