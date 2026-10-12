# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Link-link contacts between links of one articulation (dense and propagation rows).

A row whose two bodies are links of one articulation needs the cross response
J_a (X_a H^-1 X_b^T) J_b^T in its effective mass. These tests build a "scissor"
(two sibling links overlapping through their common parent), check that the
contact is routed to the dense articulated rows, and compare the dense row
diagonal against an analytic joint-space reference. With
``propagation_same_articulation_rows`` the contact becomes a propagation row whose
effective mass must include the same cross term.
"""

import unittest

import numpy as np

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import _build_scissor_model
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DENSE_PATH = 0
PROPAGATION_PATH = 2
PGS_CFM = 1.0e-6


def _step_once(model, solver):
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)
    return state_in, int(contacts.rigid_contact_count.numpy()[0])


def _analytic_H_and_X(model, state):
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


def test_scissor_scene_produces_same_articulation_contact(test, device, solve_mode="matrix_free"):
    """Route a contact between two links of one articulation to the dense articulated rows."""
    model = _build_scissor_model(device)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=0, cfm=PGS_CFM)
    _, contact_count = _step_once(model, solver)
    test.assertGreater(contact_count, 0, "scissor scene produced no link-link contact")
    paths = solver._contact_path.numpy()[:contact_count]
    slots = solver._contact_slot.numpy()[:contact_count]
    routed = (paths == DENSE_PATH) & (slots >= 0)
    test.assertTrue(routed.any(), "no same-articulation contact reached the dense rows")
    test.assertEqual(int(solver._mf_constraint_count.numpy()[0]), 0)


def test_scissor_dense_diagonal_matches_reference(test, device, solve_mode="matrix_free"):
    """Match each dense row's effective mass to ``J H^-1 J^T`` from an analytic joint-space inertia."""
    model = _build_scissor_model(device)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=0, cfm=PGS_CFM)
    state, _ = _step_once(model, solver)
    H, _ = _analytic_H_and_X(model, state)
    count = int(solver._constraint_count.numpy()[0])
    test.assertGreater(count, 0)
    D = H.shape[0]
    # One articulation: the world rows are that articulation's grouped rows.
    J = solver._J_by_size[D].numpy()[0, :count, :D].astype(np.float64)
    reference = np.einsum("rd,rd->r", J, np.linalg.solve(H, J.T).T) + PGS_CFM
    got = solver._diag.numpy()[0, :count].astype(np.float64)
    np.testing.assert_allclose(got, reference, rtol=1.0e-3, atol=1.0e-6)


def _routed_slots(solver, contact_count, path):
    """Map each contact routed to ``path`` in world 0 to its normal-row slot."""
    paths = solver._contact_path.numpy()[:contact_count]
    slots = solver._contact_slot.numpy()[:contact_count]
    worlds = solver._contact_world.numpy()[:contact_count]
    return {c: int(slots[c]) for c in range(contact_count) if paths[c] == path and slots[c] >= 0 and worlds[c] == 0}


def test_default_routing_keeps_same_articulation_rows_dense(test, device):
    """Keep same-articulation contacts on the dense rows under the default propagation response."""
    model = _build_scissor_model(device)
    solver = SolverFeatherPGS(model, iterations=0, cfm=PGS_CFM, articulated_contact_response="propagation")
    _, contact_count = _step_once(model, solver)
    test.assertGreater(len(_routed_slots(solver, contact_count, DENSE_PATH)), 0)
    test.assertEqual(int(solver._propagation_constraint_count.numpy()[0]), 0)


def test_flag_requires_propagation_mode(test, device):
    """Reject same-articulation propagation rows without the response that solves them."""
    model = _build_scissor_model(device)
    for response in ("immediate", "propagation_fused", "propagation_colored"):
        with test.subTest(response=response):
            with test.assertRaisesRegex(ValueError, "propagation_same_articulation_rows"):
                SolverFeatherPGS(model, articulated_contact_response=response, propagation_same_articulation_rows=True)


def test_cross_response_effective_mass_matches_reference_operator(test, device):
    """Match each same-articulation propagation row's ``J M^-1 J^T`` to the analytic operator.

    The generalized row Jacobian is ``g = X_a^T j_a + X_b^T j_b`` from the solver's own
    body-space rows; the reference effective mass is ``g^T H^-1 g``. Per-link responses
    alone would miss the cross term between the two links.
    """
    model = _build_scissor_model(device)
    reference = SolverFeatherPGS(model, iterations=0, cfm=PGS_CFM)
    _, contact_count = _step_once(model, reference)
    dense_slots = _routed_slots(reference, contact_count, DENSE_PATH)
    test.assertGreater(len(dense_slots), 0)

    solver = SolverFeatherPGS(
        model,
        iterations=0,
        cfm=PGS_CFM,
        articulated_contact_response="propagation",
        propagation_same_articulation_rows=True,
    )
    state, routed_count = _step_once(model, solver)
    test.assertEqual(routed_count, contact_count)
    rows = _routed_slots(solver, contact_count, PROPAGATION_PATH)
    test.assertEqual(set(rows), set(dense_slots), "same-articulation contacts were not routed to propagation rows")

    count = int(solver._propagation_constraint_count.numpy()[0])
    J_a = solver._propagation_J_a.numpy()[0, :count].astype(np.float64)
    J_b = solver._propagation_J_b.numpy()[0, :count].astype(np.float64)
    MiJt_a = solver._propagation_MiJt_a.numpy()[0, :count].astype(np.float64)
    MiJt_b = solver._propagation_MiJt_b.numpy()[0, :count].astype(np.float64)
    body_a = solver._propagation_body_a.numpy()[0, :count]
    body_b = solver._propagation_body_b.numpy()[0, :count]
    eff_mass_inv = solver._propagation_eff_mass_inv.numpy()[0, :count].astype(np.float64)
    H, X = _analytic_H_and_X(model, state)
    width = int(reference._world_dof_count.numpy()[0])
    Y = reference._Y_world.numpy()[0].astype(np.float64)

    checked = 0
    for contact, row in sorted(rows.items()):
        test.assertGreaterEqual(int(body_a[row]), 0)
        test.assertGreaterEqual(int(body_b[row]), 0)
        g = X[body_a[row]].T @ J_a[row] + X[body_b[row]].T @ J_b[row]
        y_ref = np.linalg.solve(H, g)
        # Pin frames and ordering first: the dense row response is H^-1 g up to the row sign.
        y_got = Y[dense_slots[contact], :width]
        err = min(np.linalg.norm(y_got - y_ref), np.linalg.norm(y_got + y_ref))
        test.assertLess(err / max(np.linalg.norm(y_ref), 1.0e-12), 1.0e-3)

        expected = float(g @ y_ref)
        diagonal = float(J_a[row] @ MiJt_a[row] + J_b[row] @ MiJt_b[row])
        test.assertAlmostEqual(1.0 / eff_mass_inv[row], diagonal + PGS_CFM, delta=1.0e-6 * (diagonal + 1.0))
        if expected < 1.0e-6:
            # Kinematically locked direction: the true diagonal is zero.
            test.assertLess(abs(diagonal), 1.0e-6)
            continue
        checked += 1
        test.assertLess(abs(diagonal - expected) / expected, 1.0e-3, f"contact {contact}: cross response term missing")
    test.assertGreater(checked, 0, "no non-degenerate same-articulation rows were checked")


class TestPropagationSameArticulation(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_scissor_scene_produces_same_articulation_contact",
    "test_scissor_dense_diagonal_matches_reference",
    "test_default_routing_keeps_same_articulation_rows_dense",
    "test_flag_requires_propagation_mode",
    "test_cross_response_effective_mass_matches_reference_operator",
):
    add_function_test(TestPropagationSameArticulation, _name, globals()[_name], devices=devices)
# The dense reference rows also run in the split solve; the propagation responses are matrix-free only.
for _name in (
    "test_scissor_scene_produces_same_articulation_contact",
    "test_scissor_dense_diagonal_matches_reference",
):
    add_function_test(
        TestPropagationSameArticulation,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        solve_mode="split",
    )


if __name__ == "__main__":
    unittest.main()
