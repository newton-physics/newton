# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for the single-kernel sparse Conjugate Residual solver.

Covers:
  * the fused CR kernel (raw-Jacobian gather/sort transpose, on-the-fly P and M^-1) against
    numpy's dense solve on synthetic block-sparse systems structured like Kamino's sparse
    Delassus operator, with and without a diagonal preconditioner, and
  * an end-to-end PADMM solve on a contact problem, checking that the fused solver (``CRF``)
    produces the same dual solution as the multi-launch ``CR`` solver.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg import ConjugateResidualSolver, ConjugateResidualSolverFused
from newton._src.solvers.kamino._src.linalg.conjugate_fused import (
    MAX_BLOCKS_PER_ROW,
    build_row_index,
    build_transpose_index,
    make_fused_cr_kernel,
)
from newton._src.solvers.kamino._src.models.builders import basics
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver, PADMMWarmStartMode
from newton._src.solvers.kamino.tests import setup_tests
from newton._src.solvers.kamino.tests.utils.make import make_containers, update_containers

MB = MAX_BLOCKS_PER_ROW


def _spd_inertia(rng):
    Q = rng.standard_normal((3, 3))
    return (Q @ Q.T + 3.0 * np.eye(3)).astype(np.float32)


def _make_world(rng, ncts, nbodies):
    """Synthesize one world's Jacobian (1-2 vec6 blocks/row) and SPD block-diagonal M^-1."""
    nbd = 6 * nbodies
    J = np.zeros((ncts, nbd), np.float32)
    blocks = []  # (row, col, vec6)
    for r in range(ncts):
        for bdy in rng.choice(nbodies, size=min(1 + (r % 2), nbodies), replace=False):
            blk = rng.standard_normal(6).astype(np.float32)
            J[r, 6 * bdy : 6 * bdy + 6] = blk
            blocks.append((r, 6 * bdy, blk))
    inv_m = (0.5 + rng.random(nbodies)).astype(np.float32)
    inv_I = np.stack([_spd_inertia(rng) for _ in range(nbodies)]).astype(np.float32)
    Minv = np.zeros((nbd, nbd), np.float32)
    for bdy in range(nbodies):
        Minv[6 * bdy : 6 * bdy + 3, 6 * bdy : 6 * bdy + 3] = inv_m[bdy] * np.eye(3)
        Minv[6 * bdy + 3 : 6 * bdy + 6, 6 * bdy + 3 : 6 * bdy + 6] = inv_I[bdy]
    eta = (1e-3 + 1e-2 * rng.random(ncts)).astype(np.float32)
    return {
        "J": J,
        "blocks": blocks,
        "inv_m": inv_m,
        "inv_I": inv_I,
        "Minv": Minv,
        "eta": eta,
        "ncts": ncts,
        "nbd": nbd,
        "nbodies": nbodies,
    }


def _solve_fused(worlds, use_precond, device, rng):
    """Run the fused CR kernel on packed synthetic worlds; return (x, P, b_ref, A_ref)."""
    W = len(worlds)
    R = 256
    maxbodies = max(w["nbodies"] for w in worlds)
    C = 6 * maxbodies
    nnz = [len(w["blocks"]) for w in worlds]
    max_nnz = max(nnz)
    total_nnz = W * max_nnz

    ncts = np.array([w["ncts"] for w in worlds], np.int32)
    nbd = np.array([w["nbd"] for w in worlds], np.int32)
    off = np.array([w * R for w in range(W)], np.int32)
    bodies_off = np.array([w * maxbodies for w in range(W)], np.int32)
    num_nzb = np.array(nnz, np.int32)
    nzb_start = np.array([w * max_nnz for w in range(W)], np.int32)

    coords = np.zeros((total_nnz, 2), np.int32)
    values = np.zeros((total_nnz, 6), np.float32)
    eta = np.zeros(W * R, np.float32)
    b = np.zeros(W * R, np.float32)
    P = np.ones(W * R, np.float32)
    inv_m = np.zeros(W * maxbodies, np.float32)
    inv_I = np.tile(np.eye(3, dtype=np.float32), (W * maxbodies, 1, 1))
    b_ref, A_ref, P_ref = [], [], []
    for wi, wd in enumerate(worlds):
        for k, (r, c, blk) in enumerate(wd["blocks"]):
            coords[wi * max_nnz + k] = (r, c)
            values[wi * max_nnz + k] = blk
        eta[wi * R : wi * R + wd["ncts"]] = wd["eta"]
        bb = rng.standard_normal(wd["ncts"]).astype(np.float32)
        b[wi * R : wi * R + wd["ncts"]] = bb
        b_ref.append(bb)
        Pw = (0.5 + rng.random(wd["ncts"])).astype(np.float32) if use_precond else np.ones(wd["ncts"], np.float32)
        P[wi * R : wi * R + wd["ncts"]] = Pw
        P_ref.append(Pw)
        inv_m[wi * maxbodies : wi * maxbodies + wd["nbodies"]] = wd["inv_m"]
        inv_I[wi * maxbodies : wi * maxbodies + wd["nbodies"]] = wd["inv_I"]
        PJ = Pw[:, None] * wd["J"]
        A_ref.append(PJ @ wd["Minv"] @ PJ.T + np.diag(wd["eta"]))

    d = lambda a, dt: wp.array(a, dtype=dt, device=device)  # noqa: E731
    cj_num_nzb = d(num_nzb, wp.int32)
    cj_nzb_start = d(nzb_start, wp.int32)
    cj_coords = d(coords, wp.int32)
    cj_values = wp.array(values.reshape(total_nnz, 6), dtype=vec6f, device=device)
    row_off = d(off, wp.int32)

    row_blk = build_row_index(
        num_nzb=cj_num_nzb,
        nzb_start=cj_nzb_start,
        nzb_coords=cj_coords,
        row_offset=row_off,
        total_rows=W * R,
        max_of_num_nzb=max_nnz,
        device=device,
    )
    row_idx_sorted = wp.zeros((max(2 * total_nnz, 2),), dtype=wp.int32, device=device)
    sort_key, sort_val, cursor = build_transpose_index(
        num_nzb=cj_num_nzb,
        nzb_start=cj_nzb_start,
        nzb_coords=cj_coords,
        total_nnz=total_nnz,
        max_of_num_nzb=max_nnz,
        max_major_cols=maxbodies,
        out_row_idx_sorted=row_idx_sorted,
        device=device,
    )

    x = wp.zeros(W * R, dtype=wp.float32, device=device)
    iters = wp.zeros(W, dtype=wp.int32, device=device)
    resid = wp.zeros(W, dtype=wp.float32, device=device)
    maxiter = wp.full(W, 1000, dtype=wp.int32, device=device)
    atol = wp.full(W, 1e-10, dtype=wp.float32, device=device)
    rtol = wp.full(W, 1e-10, dtype=wp.float32, device=device)
    # block_dim < R so each thread owns NR = R / block_dim > 1 rows (multi-row-per-thread path).
    block_dim = 64
    kernel = make_fused_cr_kernel(R, C, MB, block_dim)
    wp.launch_tiled(
        kernel,
        dim=W,
        inputs=[
            d(ncts, wp.int32),
            d(nbd, wp.int32),
            row_off,
            row_off,
            wp.array(np.ones(W, np.uint8), dtype=wp.bool, device=device),
            cj_num_nzb,
            cj_nzb_start,
            cj_coords,
            cj_values,
            row_blk,
            sort_key,
            sort_val,
            row_idx_sorted,
            cursor,
            d(inv_m, wp.float32),
            wp.array(inv_I, dtype=wp.mat33f, device=device),
            d(bodies_off, wp.int32),
            d(P, wp.float32),
            int(use_precond),
            d(eta, wp.float32),
            d(b, wp.float32),
            x,
            maxiter,
            atol,
            rtol,
        ],
        outputs=[iters, resid],
        block_dim=block_dim,
        device=device,
    )
    return x.numpy(), b_ref, A_ref


class TestFusedCRKernel(unittest.TestCase):
    """Numeric correctness of the fused CR kernel against numpy."""

    @classmethod
    def setUpClass(cls):
        setup_tests(clear_cache=False)
        cls.device = wp.get_preferred_device()

    def _check(self, use_precond):
        rng = np.random.default_rng(7 if use_precond else 11)
        R = 256
        # Include a world with more rows than block_dim (64) so threads span multiple row-strips.
        worlds = [_make_world(rng, n, nb) for n, nb in [(40, 9), (17, 5), (96, 18), (200, 30)]]
        x, b_ref, A_ref = _solve_fused(worlds, use_precond, self.device, rng)
        for wi, wd in enumerate(worlds):
            xw = x[wi * R : wi * R + wd["ncts"]]
            xref = np.linalg.solve(A_ref[wi].astype(np.float64), b_ref[wi].astype(np.float64))
            rel = np.linalg.norm(xw - xref) / (np.linalg.norm(xref) + 1e-30)
            self.assertLess(rel, 2e-3, f"world {wi} (precond={use_precond}) rel_err {rel:.2e}")

    def test_fused_cr_vs_numpy(self):
        """Fused CR (no preconditioner) must match numpy's dense solve."""
        if not self.device.is_cuda:
            self.skipTest("requires CUDA")
        self._check(use_precond=False)

    def test_fused_cr_vs_numpy_preconditioned(self):
        """Fused CR with a diagonal preconditioner must match numpy's dense solve."""
        if not self.device.is_cuda:
            self.skipTest("requires CUDA")
        self._check(use_precond=True)


def _run_padmm_sparse(solver_cls, device):
    """Run a box-on-plane contact PADMM solve with the given sparse linear solver."""
    builder = basics.build_box_on_plane()
    builder.gravity[0].enabled = True
    model, data, state, limits, detector, jacobians = make_containers(
        builder=builder, max_world_contacts=8, device=device, sparse=True
    )
    contacts = detector.contacts
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        solver=solver_cls,
        sparse=True,
    )
    update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

    config = PADMMSolver.Config()
    config.primal_tolerance = 1e-6
    config.dual_tolerance = 1e-6
    config.compl_tolerance = 1e-6
    config.eta = 1e-5
    config.rho_0 = 1.0
    config.max_iterations = 200
    solver = PADMMSolver(
        model=model, config=config, warmstart=PADMMWarmStartMode.NONE, use_acceleration=False, collect_info=False
    )

    problem.build(model=model, data=data, limits=limits, contacts=contacts, jacobians=jacobians)
    solver.reset()
    solver.coldstart()
    solver.solve(problem=problem)
    return {
        "lambdas": solver.data.solution.lambdas.numpy().copy(),
        "v_plus": solver.data.solution.v_plus.numpy().copy(),
        "status": solver.data.status.numpy().copy(),
    }


class TestFusedCRIntegration(unittest.TestCase):
    """End-to-end: fused CRF must match the multi-launch CR inside the PADMM solver."""

    @classmethod
    def setUpClass(cls):
        setup_tests(clear_cache=False)
        cls.device = wp.get_preferred_device()

    def test_padmm_box_on_plane_cr_vs_crf(self):
        if not self.device.is_cuda:
            self.skipTest("requires CUDA")
        ref = _run_padmm_sparse(ConjugateResidualSolver, self.device)
        got = _run_padmm_sparse(ConjugateResidualSolverFused, self.device)

        self.assertTrue(bool(ref["status"][0][0]), "multi-launch CR did not converge")
        self.assertTrue(bool(got["status"][0][0]), "fused CRF did not converge")

        lam_err = np.linalg.norm(ref["lambdas"] - got["lambdas"]) / (np.linalg.norm(ref["lambdas"]) + 1e-12)
        vp_err = np.linalg.norm(ref["v_plus"] - got["v_plus"]) / (np.linalg.norm(ref["v_plus"]) + 1e-12)
        self.assertLess(lam_err, 1e-3, f"lambda mismatch CR vs CRF: rel {lam_err:.2e}")
        self.assertLess(vp_err, 1e-3, f"v_plus mismatch CR vs CRF: rel {vp_err:.2e}")


if __name__ == "__main__":
    unittest.main()
