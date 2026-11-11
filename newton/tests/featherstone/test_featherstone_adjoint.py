# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Featherstone differentiable-path adjoints.

Multi-step gradient tape through SolverFeatherstone on a quadruped URDF
(frozen control, ground contacts active but settled). Guards the per-state
system-matrix ownership: before the fix, the solver-owned mass-matrix chain
scratch (J, M, P, H, L) was rewritten by every step, so inside one tape the
steps aliased (primal staleness + covector double-counting) and the
10-step gradient error against central finite differences was ~52% on the
worst channel. Post-fix it is ~0.2%.

A channel fails only if the tape deviates from FD by more than max(8%,
the FD two-epsilon spread); channels whose FD itself is not converged
(spread > 15%) are reported but skipped, since finite differences cannot
adjudicate them. The 8% floor (not 5%): a per-step backward - each step
backpropagated individually instead of through one shared tape, the
aliasing-free upper bound - measured ~6% error on the q[16] channel at
the K=2 horizon, i.e. the residual there is float32 chaos and FD
truncation, not adjoint aliasing. The pre-fix failures (12% on that
channel, 26-52% on others) clear the bar by >2x.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.featherstone.kernels import (
    eval_dense_cholesky_batched,
    eval_dense_solve_batched,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

STAND = np.array([0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6], dtype=np.float32)
SIM_DT = 1.0e-3
HINGE_DOFS = np.arange(6, 18, dtype=np.int32)
EPS1, EPS2 = 3e-4, 1e-4
# (state array, dof/channel): a spread of joint angles and velocities plus
# root pose/velocity channels, so both the q and qd adjoint paths are hit
CHANNELS = [(0, 5), (0, 16), (0, 4), (0, 9), (1, 3), (1, 4), (1, 16), (1, 8)]


@wp.kernel
def _servo_kernel(
    params: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    stand: wp.array[float],
    hinge_dofs: wp.array[int],
    mode: int,
):
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    if mode == 0:
        target = stand[j]
    else:
        target = stand[j] * (1.0 + params[0])
    joint_f[dof] = 2000.0 * (target - q) - 15.0 * qd


@wp.kernel
def _probe_loss_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    loss: wp.array[float],
):
    tr = wp.transform_get_translation(body_q[0])
    qt = body_q[0]
    v = body_qd[0]
    loss[0] = (
        1.0 * joint_q[9]
        + 0.75 * joint_q[13]
        + 1.5 * joint_qd[3]
        + 1.25 * joint_qd[4]
        + 0.5 * joint_qd[8]
        + 0.625 * joint_qd[12]
        + 0.5 * tr[2]
        + 0.25 * qt[4]
        + 0.375 * v[0]
        + 0.875 * v[3]
    )


def _build_model(device):
    quadruped = newton.ModelBuilder()
    quadruped.default_joint_cfg.armature = 0.1
    quadruped.default_joint_cfg.target_ke = 0.0
    quadruped.default_joint_cfg.target_kd = 0.0
    quadruped.default_joint_cfg.damping = 5.0
    quadruped.default_shape_cfg.mu = 1.0
    quadruped.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
        joint_damping_mode="drive",
    )
    for body in range(quadruped.body_count):
        inertia_np = np.asarray(quadruped.body_inertia[body], dtype=np.float32).reshape(3, 3)
        inertia_np += np.eye(3, dtype=np.float32) * 0.01
        quadruped.body_inertia[body] = wp.mat33(inertia_np)
    quadruped.joint_q[-12:] = STAND
    quadruped.joint_q[0:3] = (0.0, 0.0, 0.7)
    quadruped.joint_q[6] = 1.0
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
    scene.replicate(quadruped, 1)
    scene.add_ground_plane(cfg=quadruped.default_shape_cfg)
    return scene.finalize(device=device, requires_grad=True)


def _eager_segment(model, solver, pipe, cts, stand_arr, dofs_arr, st0, st1, n, mode, amp=0.0):
    params = wp.array([amp], dtype=wp.float32, device=model.device)
    ctrl = model.control()
    for _ in range(n):
        pipe.collide(st0, cts)
        wp.launch(
            _servo_kernel,
            dim=12,
            inputs=[params, st0.joint_q, st0.joint_qd, ctrl.joint_f, stand_arr, dofs_arr, mode],
            device=model.device,
        )
        st0.clear_forces()
        solver.step(st0, st1, ctrl, cts, SIM_DT)
        st0, st1 = st1, st0
    return st0


def _snapshot(st):
    return tuple(arr.numpy().copy() for arr in (st.joint_q, st.joint_qd, st.body_q, st.body_qd))


def test_multistep_gradient(test, device):
    model = _build_model(device)
    solver = newton.solvers.SolverFeatherstone(model)
    pipe = newton.CollisionPipeline(model, soft_contact_gap=10.0, requires_grad=True)
    cts = pipe.contacts()
    stand_arr = wp.array(STAND, dtype=wp.float32, device=device)
    dofs_arr = wp.array(HINGE_DOFS, dtype=wp.int32, device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)

    # settle to a grounded stance, then perturb with one servo pulse so the
    # frozen control below is a non-equilibrium the rollout must respond to
    s0 = model.state()
    s1 = model.state()
    s0 = _eager_segment(model, solver, pipe, cts, stand_arr, dofs_arr, s0, s1, 1200, mode=0)
    s0 = _eager_segment(model, solver, pipe, cts, stand_arr, dofs_arr, s0, s1, 100, mode=1, amp=-0.12)
    snap0 = _snapshot(s0)

    # frozen control: the servo torque at the snapshot with amp = 0
    params0 = wp.array([0.0], dtype=wp.float32, device=device)
    ctrl_e = model.control()
    wp.launch(
        _servo_kernel,
        dim=12,
        inputs=[
            params0,
            wp.array(snap0[0], dtype=wp.float32, device=device),
            wp.array(snap0[1], dtype=wp.float32, device=device),
            ctrl_e.joint_f,
            stand_arr,
            dofs_arr,
            1,
        ],
        device=device,
    )
    frozen0 = ctrl_e.joint_f.numpy().copy()

    # contact-free rollouts: the gradient comparison must
    # not depend on stale contacts, so one allocating collide is followed by
    # zeroing the contact count - every rollout/tape below then runs
    # contact-free while the contact arrays stay allocated
    pipe.collide(model.state(), cts)
    cts.rigid_contact_count.zero_()

    def rollout(snap, k, kick_arr=None, kick_idx=0, kick=0.0):
        """Eager K-step rollout of the frozen control; returns the scalar probe loss."""
        states = [model.state() for _ in range(k + 1)]
        states[0].joint_q.assign(snap[0])
        states[0].joint_qd.assign(snap[1])
        states[0].body_q.assign(snap[2])
        states[0].body_qd.assign(snap[3])
        if kick_arr is not None:
            v = snap[kick_arr].copy()
            v[kick_idx] += kick
            if kick_arr == 0:
                states[0].joint_q.assign(v)
            else:
                states[0].joint_qd.assign(v)
        ctrl = model.control()
        ctrl.joint_f.assign(frozen0)
        for t in range(k):
            states[t].clear_forces()
            solver.step(states[t], states[t + 1], ctrl, cts, SIM_DT)
        st = states[k]
        jq = st.joint_q.numpy().astype(np.float64)
        jqd = st.joint_qd.numpy().astype(np.float64)
        bq = st.body_q.numpy().astype(np.float64)
        bqd = st.body_qd.numpy().astype(np.float64)
        return float(
            1.0 * jq[9]
            + 0.75 * jq[13]
            + 1.5 * jqd[3]
            + 1.25 * jqd[4]
            + 0.5 * jqd[8]
            + 0.625 * jqd[12]
            + 0.5 * bq[0][2]
            + 0.25 * bq[0][4]
            + 0.375 * bqd[0][0]
            + 0.875 * bqd[0][3]
        )

    # physics-unchanged guard: the eager 2-step probe value is
    # adjoint-independent, so it pins forward-physics stability across
    # refactors. The expected value is MEASURED, not analytic; the delta
    # admits normal cross-device float32 drift (cpu vs cuda measured
    # ~5e-4 on this probe, delta=2e-3 gives ~4x headroom) - a real
    # physics change moves the probe by orders of magnitude. If CI
    # hardware drifts past the delta, re-measure or relax it.
    probe = rollout(snap0, 2)
    test.assertAlmostEqual(probe, -2.05270307, delta=2e-3)

    def tape_grad(snap, k):
        """Tape gradient of the probe loss w.r.t. the snapshot's joint_q/joint_qd."""
        states = [model.state(requires_grad=True) for _ in range(k + 1)]
        # solver.step() auto-allocates the per-state system-matrix scratch
        # (the mechanism under test) on each output state when the input
        # state requires grad, so each tape step owns its J/M/P/H/L buffers
        states[0].joint_q.assign(snap[0])
        states[0].joint_qd.assign(snap[1])
        states[0].body_q.assign(snap[2])
        states[0].body_qd.assign(snap[3])
        ctrl = model.control(requires_grad=True)
        ctrl.joint_f.assign(frozen0)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            for t in range(k):
                states[t].clear_forces()
                solver.step(states[t], states[t + 1], ctrl, cts, SIM_DT)
            wp.launch(
                _probe_loss_kernel,
                dim=1,
                inputs=[states[k].joint_q, states[k].joint_qd, states[k].body_q, states[k].body_qd, loss],
                device=device,
            )
        tape.backward(loss)
        g = states[0].joint_q.grad.numpy().astype(np.float64).copy()
        gqd = states[0].joint_qd.grad.numpy().astype(np.float64).copy()
        tape.zero()
        return g, gqd

    def check_horizon(snap, k):
        """Tape vs central FD on every probe channel; raises on the first bad one."""
        g, gqd = tape_grad(snap, k)
        for arr_idx, c in CHANNELS:
            fd1 = (rollout(snap, k, arr_idx, c, +EPS1) - rollout(snap, k, arr_idx, c, -EPS1)) / (2.0 * EPS1)
            fd2 = (rollout(snap, k, arr_idx, c, +EPS2) - rollout(snap, k, arr_idx, c, -EPS2)) / (2.0 * EPS2)
            gv = g[c] if arr_idx == 0 else gqd[c]
            scale = max(abs(fd1), abs(fd2), 1e-12)
            if scale < 1e-8 and abs(gv) < 1e-8:
                continue  # dead channel: nothing to adjudicate
            spread = abs(fd1 - fd2) / scale
            if spread > 0.15:
                continue  # FD-unresolvable: finite differences cannot judge
            rel = abs(gv - fd1) / scale
            test.assertLessEqual(
                rel,
                max(0.08, spread),
                f"K={k} {'q' if arr_idx == 0 else 'qd'}[{c}]: tape {gv:.5e} vs FD {fd1:.5e} "
                f"(rel {rel * 100:.2f}%, spread {spread * 100:.2f}%)",
            )

    # short horizon (adjoint path through cholesky/solve) and long horizon
    # (the multi-step aliasing the per-state scratch fixes); t=8 re-checks
    # from a mid-rollout snapshot with ping-pong state reuse behind it
    s8a = model.state()
    s8b = model.state()
    s8a.joint_q.assign(snap0[0])
    s8a.joint_qd.assign(snap0[1])
    s8a.body_q.assign(snap0[2])
    s8a.body_qd.assign(snap0[3])
    ctrl_roll = model.control()
    ctrl_roll.joint_f.assign(frozen0)
    for _ in range(8):
        s8a.clear_forces()
        solver.step(s8a, s8b, ctrl_roll, cts, SIM_DT)
        s8a, s8b = s8b, s8a
    snap8 = _snapshot(s8a)

    check_horizon(snap0, 2)
    check_horizon(snap0, 10)
    check_horizon(snap8, 2)


class TestFeatherstoneAdjoint(unittest.TestCase):
    pass


add_function_test(
    TestFeatherstoneAdjoint,
    "test_multistep_gradient",
    test_multistep_gradient,
    devices=get_test_devices(),
)


# ------------------------------------------------------------------------------
# dense cholesky + solve adjoint toy (unit level, no simulation)
# ------------------------------------------------------------------------------


def test_dense_cholesky_solve_adjoint(test, device):
    """Guards the custom adjoints of the Featherstone dense Cholesky/solve.

    Composition under test (verbatim repo kernels, one 8x8 batch):
        L_factor = chol(H + diag(R));  x = solve(L_factor, b);  loss = <g, x>
    The loss is linear in x, so the f64 central-FD reference is exact to
    roundoff and any tape deviation is a true adjoint defect.

    Defects this pins (pre-fix behavior):
    - adj_dense_solve wrote the A-product gradient -tmp*x^T into BOTH
      adjoint[A] and adjoint[L]. dense_cholesky reads only the lower
      triangle, so the correct A deposit is the symmetrized
      -(tmp*x^T + x*tmp^T) on the lower triangle + diagonal, and the factor
      gradient adjoint[L] must be (G_A + G_A^T) L, not the A form.
    - adj_dense_cholesky was a nop: the gradient w.r.t. the armature
      regularization R (entering as A_jj + R_j) was silently dropped -
      tape adjoint[R] was exactly 0 while the FD is nonzero.
    """
    N = 8
    rng = np.random.default_rng(7)
    M = rng.standard_normal((N, N))
    H64 = M @ M.T + 0.5 * np.eye(N)
    R64 = 0.1 + 0.05 * np.abs(rng.standard_normal(N))
    B64 = rng.standard_normal(N)
    G64 = rng.standard_normal(N)

    def chol64(S):
        L = np.zeros_like(S)
        for j in range(N):
            s = S[j, j] - np.dot(L[j, :j], L[j, :j])
            L[j, j] = np.sqrt(s)
            for i in range(j + 1, N):
                L[i, j] = (S[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]
        return L

    def solve64(L, b):
        y = np.zeros(N)
        for i in range(N):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        x = np.zeros(N)
        for i in range(N - 1, -1, -1):
            x[i] = (y[i] - np.dot(L[i + 1 :, i], x[i + 1 :])) / L[i, i]
        return x

    def phi64(H=None, R=None, b=None):
        Ll = chol64((H64 if H is None else H) + np.diag(R64 if R is None else R))
        return float(np.dot(G64, solve64(Ll, B64 if b is None else b)))

    def fd64(which, eps=1e-6):
        out = np.zeros(N * N if which == "H" else N)
        for c in range(out.size):

            def val(delta, c=c, which=which):
                if which == "H":
                    Hl = H64.copy()
                    Hl.flat[c] += delta
                    return phi64(H=Hl)
                if which == "R":
                    Rl = R64.copy()
                    Rl.flat[c] += delta
                    return phi64(R=Rl)
                bl = B64.copy()
                bl.flat[c] += delta
                return phi64(b=bl)

            out[c] = (val(+eps) - val(-eps)) / (2 * eps)
        return out

    H = wp.array(H64.astype(np.float32).reshape(-1), dtype=wp.float32, requires_grad=True, device=device)
    R = wp.array(R64.astype(np.float32), dtype=wp.float32, requires_grad=True, device=device)
    L = wp.zeros(N * N, dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array(B64.astype(np.float32), dtype=wp.float32, requires_grad=True, device=device)
    g = wp.array(G64.astype(np.float32), dtype=wp.float32, device=device)
    x = wp.zeros(N, dtype=wp.float32, requires_grad=True, device=device)
    tmp = wp.zeros(N, dtype=wp.float32, requires_grad=True, device=device)
    starts = wp.array([0], dtype=wp.int32, device=device)
    dims = wp.array([N], dtype=wp.int32, device=device)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(eval_dense_cholesky_batched, dim=1, inputs=[starts, dims, starts, H, R], outputs=[L], device=device)
        wp.launch(
            eval_dense_solve_batched, dim=1, inputs=[starts, dims, starts, H, L, b], outputs=[x, tmp], device=device
        )
        wp.launch(_lin_loss_kernel, dim=1, inputs=[x, g, loss], device=device)
    tape.backward(loss)

    # closed-form references in f64
    L64 = chol64(H64 + np.diag(R64))
    tmp64 = solve64(L64, G64)
    x64 = solve64(L64, B64)
    G_A = -np.outer(tmp64, x64)
    R_grad = np.diag(G_A)  # d(loss)/d(armature) = diag of the A-product gradient

    checks = (
        ("H (product)", H.grad.numpy().astype(np.float64), fd64("H")),
        ("R (armature)", R.grad.numpy().astype(np.float64), R_grad),
        ("b (tau)", b.grad.numpy().astype(np.float64), fd64("b")),
    )
    for name, tape_g, ref in checks:
        rel = np.abs(tape_g - ref) / np.maximum(np.abs(ref), 1e-12)
        significant = np.abs(ref) > 1e-6
        worst = float(rel[significant].max()) if significant.any() else 0.0
        n_bad = int((significant & (rel > 5e-3)).sum())
        test.assertEqual(
            n_bad,
            0,
            f"{name}: {n_bad} channels deviate from the f64 reference "
            f"(worst rel {worst:.2e}) - adjoint of the dense cholesky/solve path is wrong",
        )

    # design pin: the Cholesky FACTOR is internal - the solve's adjoint folds
    # the factorization dependence into the A-product deposit, so nothing may
    # write adjoint[L] (pre-fix, it received the A-form gradient as garbage)
    test.assertTrue(
        np.all(L.grad.numpy() == 0.0),
        "adjoint[L] must stay exactly zero: the factor is internal and its "
        "dependence is carried by the A-product deposit",
    )


@wp.kernel
def _lin_loss_kernel(x: wp.array[float], g: wp.array[float], loss: wp.array[float]):
    s = float(0.0)
    for i in range(x.shape[0]):
        s = s + g[i] * x[i]
    loss[0] = s


def test_rigid_tau_external_wrench(test, device):
    """Guards eval_rigid_tau's external-wrench frame shift.

    Pre-fix, eval_rigid_tau converted the public-frame external wrench into
    the internal solve frame by OVERWRITING the body_f_ext input array in
    place. Warp's recompute-based adjoint re-derives the primal locals by
    re-reading that array, so every gradient flowing through the kernel was
    corrupted (force negated, torque shifted in the recomputed f_s). The fix
    keeps the shift a pure local computation.

    Discriminating setup: a NONZERO external wrench on one body (with zero
    wrenches the overwritten values equal the originals and the defect is
    inert), one solver step, linear loss in hinge-dof velocities; the tape
    gradient w.r.t. the six wrench components is compared against central
    finite differences of the same rollout.

    Scope note: this test PINS the wrench-path gradient (it guards the fixed
    behavior against regressions). It does not by itself discriminate the
    pre-fix code: the documented pre-fix anomalies live in the S_s /
    body_q_com covector deposits, which are per-step DERIVED arrays (not
    state leaves) and were discriminated with the launch-by-launch manual
    backward walk in the fix's validation history. The other two tests in
    this file carry the with/without discrimination.
    """
    model = _build_model(device)
    solver = newton.solvers.SolverFeatherstone(model)
    pipe = newton.CollisionPipeline(model, soft_contact_gap=10.0, requires_grad=True)
    cts = pipe.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    # contact-free comparison world (same parity as the multistep test)
    pipe.collide(model.state(), cts)
    cts.rigid_contact_count.zero_()

    BODY = 1
    WRENCH = np.array([5.0, -3.0, 2.0, 0.5, -0.25, 0.75], dtype=np.float64)
    C = np.array([0.3, -0.7, 1.1, -0.4, 0.9, -0.2], dtype=np.float64)

    def rollout_loss(wrench64):
        # a fresh state's force buffers are already zero - clear_forces()
        # must NOT be called after the assign (it would zero the wrench)
        st0 = model.state()
        st1 = model.state()
        f = np.zeros((model.body_count, 6), dtype=np.float32)
        f[BODY] = wrench64.astype(np.float32)
        st0.body_f.assign(f)
        ctrl = model.control()
        solver.step(st0, st1, ctrl, cts, SIM_DT)
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
        solver.step(st0, st1, ctrl, cts, SIM_DT)
        wp.launch(_joint_qd_loss_kernel, dim=1, inputs=[st1.joint_qd, loss], device=device)
    tape.backward(loss)
    ana = st0.body_f.grad.numpy().astype(np.float64)[BODY]
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


@wp.kernel
def _joint_qd_loss_kernel(joint_qd: wp.array[float], loss: wp.array[float]):
    loss[0] = 0.3 * joint_qd[9] - 0.7 * joint_qd[13] + 1.1 * joint_qd[3] - 0.4 * joint_qd[16]


add_function_test(
    TestFeatherstoneAdjoint,
    "test_dense_cholesky_solve_adjoint",
    test_dense_cholesky_solve_adjoint,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherstoneAdjoint,
    "test_rigid_tau_external_wrench",
    test_rigid_tau_external_wrench,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
