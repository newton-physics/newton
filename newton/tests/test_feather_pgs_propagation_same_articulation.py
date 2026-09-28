# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-articulation link-link contacts on the propagation path.

A row whose two bodies are links of one articulation needs the cross
operational-space response J_a (X_a H^-1 X_b^T) J_b^T in its effective mass;
per-link diagonal responses alone overestimate it. These tests build a
"scissor" (two sibling links overlapping through their common parent) and
compare the propagation row diagonal against the dense J_world/Y_world
reference operator.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

CONTACT_ROW_TYPE = 0
DENSE_PATH = 0
PROPAGATION_PATH = 2


def _build_scissor_model(device: str) -> newton.Model:
    """One articulation: base link + two sibling links whose boxes overlap.

    The siblings are not directly jointed, so collision produces a
    link-link contact within a single articulation (common ancestor = base).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = 0.75
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    base = builder.add_link()
    builder.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    j_base = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )

    joints = [j_base]
    for side in (1.0, -1.0):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04)
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.06, side * 0.028, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _zero_iteration_step(model: newton.Model, solver: SolverFeatherPGS):
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    state_in.clear_forces()
    collision_pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)
    solver._test_live_contact_count = int(contacts.rigid_contact_count.numpy()[0])
    return state_in


def _make_solver(model: newton.Model, mode: str, **kwargs) -> SolverFeatherPGS:
    return SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response=mode,
        pgs_iterations=0,
        pgs_velocity_iterations=0,
        enable_contact_friction=False,
        dense_max_constraints=64,
        mf_max_constraints=16,
        pgs_warmstart=False,
        mf_warmstart=False,
        **kwargs,
    )


def _contact_rows(solver: SolverFeatherPGS, path_id: int, contact_count: int | None = None) -> dict[int, int]:
    """contact index -> row slot for contacts routed to the given path."""
    contact_path = solver.contact_path.numpy().astype(np.int32)
    contact_slot = solver.contact_slot.numpy().astype(np.int32)
    contact_world = solver.contact_world.numpy().astype(np.int32)
    out = {}
    if contact_count is None:
        contact_count = solver._test_live_contact_count
    for c in range(contact_count):
        if int(contact_world[c]) == 0 and int(contact_path[c]) == path_id and int(contact_slot[c]) >= 0:
            out[c] = int(contact_slot[c])
    return out


def _analytic_H_and_X(model: newton.Model, state) -> tuple[np.ndarray, np.ndarray]:
    """Joint-space inertia H [D,D] and per-body kinematic maps X [bodies,6,D]
    for single-world revolute trees, world frame, velocities as
    [v_com_lin, omega] (the propagation body convention)."""
    body_q = state.body_q.numpy().astype(np.float64)
    body_com = model.body_com.numpy().astype(np.float64)
    body_mass = model.body_mass.numpy().astype(np.float64)
    body_inertia = model.body_inertia.numpy().astype(np.float64)
    joint_parent = model.joint_parent.numpy().astype(np.int32)
    joint_child = model.joint_child.numpy().astype(np.int32)
    joint_axis = model.joint_axis.numpy().astype(np.float64)
    joint_X_p = model.joint_X_p.numpy().astype(np.float64)
    joint_qd_start = model.joint_qd_start.numpy().astype(np.int32)
    armature = model.joint_armature.numpy().astype(np.float64)

    def rot(q):
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    D = int(model.joint_dof_count)
    n_bodies = model.body_count
    com_w = np.zeros((n_bodies, 3))
    R_w = np.zeros((n_bodies, 3, 3))
    for b in range(n_bodies):
        p, q = body_q[b, :3], body_q[b, 3:]
        R_w[b] = rot(q)
        com_w[b] = p + R_w[b] @ body_com[b]

    # world-frame joint axis and anchor per joint
    n_joints = joint_parent.shape[0]
    axis_w = np.zeros((n_joints, 3))
    anchor_w = np.zeros((n_joints, 3))
    parent_joint_of_body = np.full(n_bodies, -1, dtype=np.int32)
    for j in range(n_joints):
        parent = int(joint_parent[j])
        if parent >= 0:
            Rp = R_w[parent]
            pp = body_q[parent, :3]
        else:
            Rp = np.eye(3)
            pp = np.zeros(3)
        Xp_p, Xp_q = joint_X_p[j, :3], joint_X_p[j, 3:]
        anchor_w[j] = pp + Rp @ Xp_p
        axis_w[j] = (
            Rp @ rot(Xp_q) @ joint_axis[joint_qd_start[j]] if joint_axis.ndim == 2 else Rp @ rot(Xp_q) @ joint_axis[j]
        )
        parent_joint_of_body[int(joint_child[j])] = j

    X = np.zeros((n_bodies, 6, D))
    for b in range(n_bodies):
        j = int(parent_joint_of_body[b])
        while j >= 0:
            dof = int(joint_qd_start[j])
            a = axis_w[j]
            X[b, 0:3, dof] = np.cross(a, com_w[b] - anchor_w[j])
            X[b, 3:6, dof] = a
            parent = int(joint_parent[j])
            j = int(parent_joint_of_body[parent]) if parent >= 0 else -1

    H = np.zeros((D, D))
    for b in range(n_bodies):
        M = np.zeros((6, 6))
        M[0:3, 0:3] = body_mass[b] * np.eye(3)
        M[3:6, 3:6] = R_w[b] @ body_inertia[b] @ R_w[b].T
        H += X[b].T @ M @ X[b]
    H += np.diag(armature[:D])
    return H, X


def _propagation_operator_diag(solver: SolverFeatherPGS) -> np.ndarray:
    count = int(solver.propagation_constraint_count.numpy()[0])
    J_a = solver.propagation_J_a.numpy()[0, :count].astype(np.float64)
    J_b = solver.propagation_J_b.numpy()[0, :count].astype(np.float64)
    MiJt_a = solver.propagation_MiJt_a.numpy()[0, :count].astype(np.float64)
    MiJt_b = solver.propagation_MiJt_b.numpy()[0, :count].astype(np.float64)
    body_a = solver.propagation_body_a.numpy()[0, :count].astype(np.int32)
    body_b = solver.propagation_body_b.numpy()[0, :count].astype(np.int32)
    diag = np.zeros(count, dtype=np.float64)
    for row in range(count):
        if body_a[row] >= 0:
            diag[row] += float(np.dot(J_a[row], MiJt_a[row]))
        if body_b[row] >= 0:
            diag[row] += float(np.dot(J_b[row], MiJt_b[row]))
    return diag


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationSameArticulation(unittest.TestCase):
    def test_scissor_scene_produces_same_articulation_contact(self):
        model = _build_scissor_model("cuda:0")
        solver = _make_solver(model, "immediate")
        _zero_iteration_step(model, solver)
        dense_rows = _contact_rows(solver, DENSE_PATH)
        self.assertGreater(len(dense_rows), 0, "scissor scene produced no dense link-link contact")

    def test_default_routing_keeps_same_articulation_rows_dense(self):
        model = _build_scissor_model("cuda:0")
        solver = _make_solver(model, "propagation")
        _zero_iteration_step(model, solver)
        self.assertEqual(int(solver.propagation_constraint_count.numpy()[0]), 0)
        self.assertGreater(len(_contact_rows(solver, DENSE_PATH)), 0)

    def test_flag_requires_propagation_mode(self):
        model = _build_scissor_model("cuda:0")
        with self.assertRaises(ValueError):
            _make_solver(model, "immediate", propagation_same_articulation_rows=True)

    def test_cross_response_effective_mass_matches_reference_operator(self):
        """The defining check: with same-articulation rows routed to propagation,
        each row's J M^-1 J^T must match the analytic reference including the
        cross term between the two links."""
        model = _build_scissor_model("cuda:0")

        reference = _make_solver(model, "immediate")
        _zero_iteration_step(model, reference)
        dense_rows = _contact_rows(reference, DENSE_PATH)
        self.assertGreater(len(dense_rows), 0)

        solver = _make_solver(model, "propagation", propagation_same_articulation_rows=True)
        state = _zero_iteration_step(model, solver)
        prop_rows = _contact_rows(solver, PROPAGATION_PATH)
        self.assertEqual(
            set(prop_rows.keys()),
            set(dense_rows.keys()),
            "same-articulation contacts were not routed to propagation rows",
        )
        prop_diag = _propagation_operator_diag(solver)

        # Analytic reference: generalized row Jacobian g = X_a^T j_a + X_b^T j_b
        # from the solver's own row geometry, W_kk = g^T H^-1 g.
        H, X = _analytic_H_and_X(model, state)
        count = int(solver.propagation_constraint_count.numpy()[0])
        J_a = solver.propagation_J_a.numpy()[0, :count].astype(np.float64)
        J_b = solver.propagation_J_b.numpy()[0, :count].astype(np.float64)
        body_a = solver.propagation_body_a.numpy()[0, :count].astype(np.int32)
        body_b = solver.propagation_body_b.numpy()[0, :count].astype(np.int32)

        # Convention check: the reference solver's Y rows must equal H^-1 g
        # (up to overall row sign) — this pins frames/ordering before we
        # trust the reference for the real assertion.
        width = int(reference.world_dof_count.numpy()[0])
        Y = reference.Y_world.numpy()[0].astype(np.float64)
        for contact, dense_slot in sorted(dense_rows.items()):
            row = prop_rows[contact]
            g = X[body_a[row]].T @ J_a[row] + X[body_b[row]].T @ J_b[row]
            y_ref = np.linalg.solve(H, g)
            y_got = Y[dense_slot, :width]
            err = min(np.linalg.norm(y_got - y_ref), np.linalg.norm(y_got + y_ref))
            self.assertLess(
                err / max(np.linalg.norm(y_ref), 1e-12),
                1.0e-3,
                f"analytic reference disagrees with solver Y row {dense_slot}: {y_got} vs {y_ref}",
            )

        checked = 0
        for contact in sorted(dense_rows):
            row = prop_rows[contact]
            g = X[body_a[row]].T @ J_a[row] + X[body_b[row]].T @ J_b[row]
            ref = float(g @ np.linalg.solve(H, g))
            got = prop_diag[row]
            if ref < 1.0e-6:
                # Kinematically locked direction: true W_kk is zero, both
                # sides must be at float noise level.
                self.assertLess(abs(got), 1.0e-6)
                continue
            checked += 1
            rel = abs(got - ref) / ref
            self.assertLess(
                rel,
                1.0e-3,
                f"contact {contact}: propagation W_kk {got:.6e} vs reference {ref:.6e} "
                f"(rel {rel:.3e}) — cross response term missing or wrong",
            )
        self.assertGreater(checked, 0, "no non-degenerate same-articulation rows were checked")


if __name__ == "__main__":
    unittest.main()
