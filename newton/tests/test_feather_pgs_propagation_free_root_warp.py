# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical equivalence of the free-root warp tree kernels vs serial fallbacks.

Articulations shaped "one multi-DOF world-rooted joint (floating base) + only
0/1-DOF joints below it" qualify for the one-warp propagation tree kernels
with a compiled free-root special case. These tests compare the warp variants
against the serial reference kernels (forced by nulling the per-size kernel
dicts) two ways:

1. Kernel-level: identical solver states (contact-free trajectories are
   bitwise deterministic), synthetic deferred impulses / generalized
   velocities, one kernel invocation each, tight tolerances. This is the
   primary gate — it compares exactly the code paths this feature adds.
2. End-to-end: full stepping over a ground plane. The contact pipeline
   allocates row slots atomically, so trajectories are only reproducible when
   each world produces a single contact point; the scene uses one sphere per
   world so warp-vs-serial trajectories stay comparable at 1e-4.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

N_WORLDS = 4
N_LINKS = 6
DT = 1.0 / 240.0

_WARP_KERNEL_DICTS = (
    "_propagate_tree_warp_kernels_by_size",
    "_factor_tree_warp_kernels_by_size",
    "_response_tree_warp_kernels_by_size",
    "_refresh_tree_warp_kernels_by_size",
)


def _make_floating_chain_world(*, base_z: float, contact_sphere: str | None) -> newton.ModelBuilder:
    """A floating box base (free joint) trailing N_LINKS revolute links.

    ``contact_sphere`` places a single collision sphere on the base or on the
    deepest link; all other shapes carry no collision geometry so each world
    produces at most one contact point (keeps the contact pipeline
    deterministic for the end-to-end comparison). ``None`` builds density-only
    box shapes with collision enabled (used for the contact-free tests).
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.7

    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()))
    cfg = builder.default_shape_cfg if contact_sphere is None else no_collision
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    if contact_sphere == "base":
        builder.add_shape_sphere(base, radius=0.1)
    joints = [builder.add_joint_free(base)]
    prev = base
    for i in range(N_LINKS):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=cfg)
        if contact_sphere == "deep" and i == N_LINKS - 1:
            builder.add_shape_sphere(link, radius=0.05)
        offset = 0.22 if i == 0 else 0.12
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder


def _build_model(device: str, *, base_z: float, contact_sphere: str | None, ground: bool) -> newton.Model:
    scene = newton.ModelBuilder()
    for _ in range(N_WORLDS):
        scene.add_world(_make_floating_chain_world(base_z=base_z, contact_sphere=contact_sphere))
    if ground:
        scene.add_ground_plane()
    return scene.finalize(device=device)


def _make_solver(model: newton.Model, response: str) -> SolverFeatherPGS:
    # propagation_cached_response=False pins the per-iteration tree-walk
    # propagation path this file compares (warp vs serial variants). The
    # cached-response GEMV replacement has its own equivalence gate in
    # test_feather_pgs_propagation_cached_response.py.
    return SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response=response,
        pgs_iterations=8,
        pgs_warmstart=False,
        mf_warmstart=False,
        propagation_cached_response=False,
    )


def _force_serial_fallback(solver: SolverFeatherPGS) -> None:
    for name in _WARP_KERNEL_DICTS:
        kernels = getattr(solver, name, None)
        if kernels:
            setattr(solver, name, dict.fromkeys(kernels, None))


def _seed_joint_qd(model: newton.Model) -> np.ndarray:
    n = int(model.joint_dof_count)
    return (0.4 * np.sin(0.7 * np.arange(n, dtype=np.float64) + 0.3)).astype(np.float32)


def _step_n(model, solver, n_steps, *, state_in=None):
    state_in = model.state() if state_in is None else state_in
    state_out = model.state()
    control = model.control()
    state_in.joint_qd.assign(_seed_joint_qd(model))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    for _ in range(n_steps):
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        state_in, state_out = state_out, state_in
    wp.synchronize()
    return state_in


# ── numpy reference of the serial propagation tree algorithms ──────────────


def _tree_inputs(model, solver, art):
    return {
        "S": solver.propagation_joint_S_flat.numpy().astype(np.float64),
        "U": solver.propagation_tree_U.numpy().astype(np.float64),
        "Dinv": solver.propagation_tree_D_inv.numpy().astype(np.float64),
        "com": solver.propagation_body_com_rel.numpy().astype(np.float64),
        "jp": model.joint_parent.numpy(),
        "jc": model.joint_child.numpy(),
        "qd_start": model.joint_qd_start.numpy(),
        "j0": int(model.articulation_start.numpy()[art]),
        "j1": int(model.articulation_start.numpy()[art + 1]),
    }


def _xlt_wrench(w, e):
    return np.concatenate([w[:3], w[3:] + np.cross(e, w[:3])])


def _xlt_twist(v, e):
    return np.concatenate([v[:3] + np.cross(v[3:], e), v[3:]])


def _ref_propagate(ti, imp_by_body, v_out):
    """Mirror of kernels.propagate_tree_impulses_for_size (one articulation)."""
    S, U, Dinv, com = ti["S"], ti["U"], ti["Dinv"], ti["com"]
    jp, jc, qd_start = ti["jp"], ti["jc"], ti["qd_start"]
    j0, j1 = ti["j0"], ti["j1"]
    pA = {}
    u = np.zeros(S.shape[0])
    body_delta = {}
    for j in range(j0, j1):
        pA[int(jc[j])] = -np.asarray(imp_by_body.get(int(jc[j]), np.zeros(6)), dtype=np.float64)
        body_delta[int(jc[j])] = np.zeros(6)
    for j in range(j1 - 1, j0 - 1, -1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        for g in range(d0, d1):
            u[g] = -np.dot(S[g], pA[child])
        if parent >= 0:
            p = pA[child].copy()
            for a in range(d0, d1):
                coeff = sum(Dinv[j, a - d0, b - d0] * u[b] for b in range(d0, d1))
                p += U[a] * coeff
            pA[parent] += _xlt_wrench(p, com[child] - com[parent])
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        pd = np.zeros(6)
        if parent >= 0:
            pd = _xlt_twist(body_delta[parent], com[child] - com[parent])
        qdd = np.zeros(d1 - d0)
        for a in range(d0, d1):
            acc = 0.0
            for b in range(d0, d1):
                pt = np.dot(U[b], pd) if parent >= 0 else 0.0
                acc += Dinv[j, a - d0, b - d0] * (u[b] - pt)
            qdd[a - d0] = acc
            v_out[a] += acc
        body_delta[child] = pd + sum(S[a] * qdd[a - d0] for a in range(d0, d1))
    # third pass: recompute body twists from the updated v_out
    body_qd = {}
    twist = {}
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        val = np.zeros(6)
        if parent >= 0:
            val = _xlt_twist(twist[parent], com[child] - com[parent])
        for a in range(d0, d1):
            val = val + S[a] * v_out[a]
        twist[child] = val
        body_qd[child] = val
    return v_out, body_qd


def _ref_body_response(ti, body):
    """6x6 response of ``body``: column = body twist delta per unit basis wrench."""
    R = np.zeros((6, 6))
    for basis in range(6):
        imp = np.zeros(6)
        imp[basis] = 1.0
        # At zero initial generalized velocity the recomputed body twist for a
        # unit test wrench equals the body's velocity response column.
        _, body_qd = _ref_propagate(ti, {body: imp}, np.zeros(ti["S"].shape[0]))
        R[:, basis] = body_qd[body]
    return R


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationFreeRootWarp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"
        # contact-free model: floating chains tumbling in space
        cls.model_free = _build_model(cls.device, base_z=2.0, contact_sphere=None, ground=False)

    def test_free_root_shape_detected(self):
        solver = _make_solver(self.model_free, "propagation-colored")
        sizes = [int(s) for s in solver.size_groups if solver.n_arts_by_size[s] > 0]
        self.assertEqual(len(sizes), 1)
        size = sizes[0]
        self.assertFalse(solver._propagation_tree_single_dof_by_size[size])
        self.assertTrue(solver._propagation_tree_free_root_by_size[size])
        for name in _WARP_KERNEL_DICTS:
            kernels = getattr(solver, name, None)
            self.assertIsNotNone(kernels, name)
            self.assertIsNotNone(kernels.get(size), f"{name}[{size}] should hold a warp variant")

    def _paired_solvers(self, n_steps):
        """Two solvers advanced identically on a contact-free trajectory.

        Without contact rows the tree kernels do not influence the
        trajectory, and no atomically-ordered contact bookkeeping runs, so
        both solvers reach bitwise-identical states and tree factorizations
        are computed from identical inputs.
        """
        solver_w = _make_solver(self.model_free, "propagation-colored")
        solver_s = _make_solver(self.model_free, "propagation-colored")
        _force_serial_fallback(solver_s)
        state_w = _step_n(self.model_free, solver_w, n_steps)
        state_s = _step_n(self.model_free, solver_s, n_steps)
        np.testing.assert_array_equal(state_w.joint_q.numpy(), state_s.joint_q.numpy())
        return solver_w, solver_s

    def test_factor_matches_serial(self):
        model = self.model_free
        qd_start = model.joint_qd_start.numpy()
        n_joints = model.joint_count
        for n_steps in (1, 12, 25):
            with self.subTest(n_steps=n_steps):
                solver_w, solver_s = self._paired_solvers(n_steps)
                U_w = solver_w.propagation_tree_U.numpy()
                U_s = solver_s.propagation_tree_U.numpy()
                Dc_w = solver_w.propagation_tree_D_chol.numpy()
                Dc_s = solver_s.propagation_tree_D_chol.numpy()
                Di_w = solver_w.propagation_tree_D_inv.numpy()
                Di_s = solver_s.propagation_tree_D_inv.numpy()
                for j in range(n_joints):
                    dc = int(qd_start[j + 1] - qd_start[j])
                    if dc == 0:
                        continue
                    d0 = int(qd_start[j])
                    scale = max(1.0, float(np.max(np.abs(U_s[d0 : d0 + dc]))))
                    self.assertLess(float(np.max(np.abs(U_w[d0 : d0 + dc] - U_s[d0 : d0 + dc]))), 1e-5 * scale)
                    scale = max(1.0, float(np.max(np.abs(Di_s[j, :dc, :dc]))))
                    self.assertLess(float(np.max(np.abs(Di_w[j, :dc, :dc] - Di_s[j, :dc, :dc]))), 1e-5 * scale)
                    scale = max(1.0, float(np.max(np.abs(Dc_s[j, :dc, :dc]))))
                    self.assertLess(float(np.max(np.abs(Dc_w[j, :dc, :dc] - Dc_s[j, :dc, :dc]))), 1e-5 * scale)

    def test_propagate_matches_serial(self):
        model = self.model_free
        rng = np.random.default_rng(1234)
        solver_w, solver_s = self._paired_solvers(20)
        bodies_per_world = model.body_count // N_WORLDS
        patterns = [rng.normal(scale=0.5, size=(model.body_count, 6)).astype(np.float32)]
        for body in (0, 1, bodies_per_world - 1):  # root, first link, deepest link
            p = np.zeros((model.body_count, 6), dtype=np.float32)
            p[body] = rng.normal(scale=1.0, size=6)
            patterns.append(p)
        for pi, imps in enumerate(patterns):
            with self.subTest(pattern=pi):
                v0 = rng.normal(scale=0.1, size=solver_w.v_out.shape).astype(np.float32)
                for s in (solver_w, solver_s):
                    s.propagation_body_impulses.assign(imps)
                    s.v_out.assign(v0)
                    s.propagation_body_qd.zero_()
                solver_w._propagate_response()
                solver_s._propagate_response()
                wp.synchronize()
                dv = solver_w.v_out.numpy() - solver_s.v_out.numpy()
                dqd = solver_w.propagation_body_qd.numpy() - solver_s.propagation_body_qd.numpy()
                scale = max(1.0, float(np.max(np.abs(solver_s.v_out.numpy()))))
                self.assertLess(float(np.max(np.abs(dv))), 1e-5 * scale)
                self.assertLess(float(np.max(np.abs(dqd))), 1e-5 * scale)
                # deferred impulses must be consumed
                self.assertEqual(float(np.max(np.abs(solver_w.propagation_body_impulses.numpy()))), 0.0)

    def test_propagate_matches_numpy_reference(self):
        model = self.model_free
        rng = np.random.default_rng(99)
        solver_w, _ = self._paired_solvers(20)
        ti = _tree_inputs(model, solver_w, art=0)
        bodies_per_world = model.body_count // N_WORLDS
        imps = np.zeros((model.body_count, 6), dtype=np.float32)
        imps[:bodies_per_world] = rng.normal(scale=0.5, size=(bodies_per_world, 6))
        solver_w.propagation_body_impulses.assign(imps)
        solver_w.v_out.zero_()
        solver_w.propagation_body_qd.zero_()
        solver_w._propagate_response()
        wp.synchronize()
        n_dofs_w0 = int(model.joint_dof_count) // N_WORLDS
        ref_v, ref_body_qd = _ref_propagate(
            ti,
            {b: imps[b].astype(np.float64) for b in range(bodies_per_world)},
            np.zeros(ti["S"].shape[0]),
        )
        got_v = solver_w.v_out.numpy()[:n_dofs_w0]
        self.assertLess(float(np.max(np.abs(got_v - ref_v[:n_dofs_w0]))), 1e-4)
        got_qd = solver_w.propagation_body_qd.numpy()
        for b in range(bodies_per_world):
            self.assertLess(float(np.max(np.abs(got_qd[b] - ref_body_qd[b]))), 1e-4)

    def test_refresh_matches_serial(self):
        rng = np.random.default_rng(4321)
        solver_w, solver_s = self._paired_solvers(20)
        v0 = rng.normal(scale=0.5, size=solver_w.v_out.shape).astype(np.float32)
        for s in (solver_w, solver_s):
            s.v_out.assign(v0)
            s.propagation_body_qd.zero_()
        solver_w._refresh_propagation_body_qd_from_vout(force=True)
        solver_s._refresh_propagation_body_qd_from_vout(force=True)
        wp.synchronize()
        qd_w = solver_w.propagation_body_qd.numpy()
        qd_s = solver_s.propagation_body_qd.numpy()
        scale = max(1.0, float(np.max(np.abs(qd_s))))
        self.assertLess(float(np.max(np.abs(qd_w - qd_s))), 1e-5 * scale)

    def test_response_matches_numpy_reference(self):
        """Warp free-root response vs the serial tree-solve semantics.

        The serial fallback for free-root groups is the path-restricted
        generic kernel, which only fills rows for contact-active bodies; the
        warp variant fills every link (the single-DOF revolute contract). Both
        must equal the exact articulated-body response, so the warp output is
        gated against a float64 reference of the serial algorithm.
        """
        model = self.model_free
        solver_w, _ = self._paired_solvers(20)
        size = next(int(s) for s in solver_w.size_groups if solver_w.n_arts_by_size[s] > 0)
        self.assertIsNotNone(solver_w._response_tree_warp_kernels_by_size.get(size))
        ti = _tree_inputs(model, solver_w, art=0)
        resp = solver_w.propagation_body_response.numpy()
        bodies_per_world = model.body_count // N_WORLDS
        for body in range(bodies_per_world):
            ref = _ref_body_response(ti, body)
            scale = max(1.0, float(np.max(np.abs(ref))))
            self.assertLess(
                float(np.max(np.abs(resp[body] - ref))),
                1e-4 * scale,
                f"body {body} response mismatch",
            )

    def test_end_to_end_single_contact(self):
        """Full stepping over ground with one contact sphere per world.

        A single contact point keeps the contact pipeline's atomic slot
        allocation trivially deterministic, so the warp-vs-serial trajectory
        difference reflects only the tree-kernel float reassociation.
        """
        for contact_sphere, base_z in (("base", 0.115), ("deep", 0.12)):
            model = _build_model(self.device, base_z=base_z, contact_sphere=contact_sphere, ground=True)
            for response in ("propagation-colored", "propagation"):
                with self.subTest(contact=contact_sphere, articulated_contact_response=response):
                    results = {}
                    for mode in ("warp", "serial", "serial2"):
                        solver = _make_solver(model, response)
                        if mode != "warp":
                            _force_serial_fallback(solver)
                        state = _step_n(model, solver, 40)
                        results[mode] = {
                            "joint_q": state.joint_q.numpy().copy(),
                            "joint_qd": state.joint_qd.numpy().copy(),
                            "propagation_body_qd": solver.propagation_body_qd.numpy().copy(),
                            "rows": int(solver.propagation_constraint_count.numpy().sum()),
                        }
                    self.assertGreater(results["warp"]["rows"], 0, "no propagation contact rows produced")
                    # precondition: the scene is pipeline-deterministic
                    for key in ("joint_q", "joint_qd"):
                        np.testing.assert_array_equal(
                            results["serial"][key],
                            results["serial2"][key],
                            err_msg=f"serial trajectory not reproducible for {key}; scene invalidates the gate",
                        )
                    # The warp and serial paths differ only by float reassociation, but 40 steps of
                    # contact dynamics amplify that seed chaotically: measured joint_qd diffs are
                    # ~5e-7 at 10 steps, 1e-5..4e-5 at 20 and 4e-5..1.2e-4 at 40, with the 40-step
                    # value swinging by 3x across GPU architectures (8.1e-5 on sm_120 vs 1.2e-4 on
                    # sm_86 at the same commit) and across physically equivalent trajectory changes.
                    # The gate is set ~4x above the observed band to stay deterministic per
                    # hardware; structural kernel bugs produce diffs of 1e-2 or more and still trip
                    # it comfortably.
                    for key in ("joint_q", "joint_qd", "propagation_body_qd"):
                        diff = float(np.max(np.abs(results["warp"][key] - results["serial"][key])))
                        self.assertLessEqual(
                            diff,
                            5e-4,
                            f"{key} max abs diff {diff:.3e} exceeds 5e-4",
                        )


if __name__ == "__main__":
    unittest.main()
