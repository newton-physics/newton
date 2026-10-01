# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mass-matrix refresh cadence of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 60.0
# Per-articulation initial pose/velocity: [base, left branch, right branch].
INITIAL_JOINT_Q = (0.7, 0.3, -0.4)
INITIAL_JOINT_QD = (0.5, -0.2, 0.3)


def _build_model(device, num_worlds=2, ground=True):
    """Two-branch pendulum: base revolute joint plus two sibling branch links.

    The sibling branches are not ancestor-related, so the mass matrix has structural zeros
    between their DOFs; with a ground plane the branch tips swing into contact.
    """
    env = newton.ModelBuilder()
    env.default_shape_cfg.density = 1000.0
    base = env.add_link()
    env.add_shape_box(base, hx=0.08, hy=0.08, hz=0.08)
    left = env.add_link()
    env.add_shape_box(left, hx=0.25, hy=0.05, hz=0.05)
    right = env.add_link()
    env.add_shape_box(right, hx=0.25, hy=0.05, hz=0.05)
    drive = {"axis": newton.Axis.Y, "target_ke": 15.0, "target_kd": 1.5}
    j_base = env.add_joint_revolute(
        -1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()), **drive
    )
    j_left = env.add_joint_revolute(
        base,
        left,
        parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        **drive,
    )
    j_right = env.add_joint_revolute(
        base,
        right,
        parent_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        **drive,
    )
    env.add_articulation([j_base, j_left, j_right])
    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _make_initial_state(model):
    state = model.state()
    num_arts = model.articulation_count
    state.joint_q.assign(np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), num_arts))
    state.joint_qd.assign(np.tile(np.asarray(INITIAL_JOINT_QD, dtype=np.float32), num_arts))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_trajectory(model, solver, num_steps):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    history = []
    for _ in range(num_steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        history.append(state_0.joint_q.numpy().copy())
    return np.stack(history)


def test_mass_refresh_cadence_follows_interval(test, device):
    """Refresh every articulation's factorization on every ``interval``-th step only."""
    model = _build_model(device)
    solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    for step_index, expected in enumerate(([1, 1], [0, 0], [1, 1], [0, 0])):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        test.assertEqual(solver.mass_update_mask.numpy().tolist(), expected, f"step {step_index}")


def test_stale_factorization_stays_close_to_interval_one(test, device):
    """Keep an interval-2 contact trajectory close to the every-step refresh."""
    history = {}
    for interval in (1, 2):
        model = _build_model(device)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
        history[interval] = _run_trajectory(model, solver, num_steps=60)
    test.assertTrue(np.isfinite(history[2]).all())
    # A stale factorization between refreshes is the intended trade-off, so the runs differ.
    test.assertLess(float(np.abs(history[2] - history[1]).max()), 0.05)
    moved = np.abs(history[1][-1] - np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), 2)).max()
    test.assertGreater(moved, 1.0e-3)


def test_model_change_request_refreshes_a_reuse_step(test, device):
    """Refresh every articulation on a reuse step after a notification and consume the request."""
    model = _build_model(device, ground=False)
    solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    state_0, state_1 = state_1, state_0
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    solver.step(state_0, state_1, control, None, DT)
    test.assertEqual(solver.mass_update_mask.numpy().tolist(), [1, 1])
    test.assertEqual(solver._mass_update_requested.numpy().tolist(), [0] * model.articulation_count)


def test_timestep_change_forces_a_refresh(test, device):
    """Refresh the augmented mass matrix when the time step changes on a reuse step."""
    model = _build_model(device, ground=False)
    solver = SolverFeatherPGS(model, update_mass_matrix_interval=4)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    solver.step(state_1, state_0, control, None, DT)
    test.assertEqual(solver.mass_update_mask.numpy().tolist(), [0, 0])
    solver.step(state_0, state_1, control, None, 0.5 * DT)
    test.assertEqual(solver.mass_update_mask.numpy().tolist(), [1, 1])


def test_captured_two_substep_replay_matches_eager(test, device):
    """Replay two captured substeps, including the first-step allocations, like eager stepping."""
    graph_model = _build_model(device, ground=False)
    eager_model = _build_model(device, ground=False)
    graph_solver = SolverFeatherPGS(graph_model, update_mass_matrix_interval=2)
    eager_solver = SolverFeatherPGS(eager_model, update_mass_matrix_interval=2)
    graph_a, graph_b = _make_initial_state(graph_model), graph_model.state()
    eager_a, eager_b = _make_initial_state(eager_model), eager_model.state()
    graph_control, eager_control = graph_model.control(), eager_model.control()

    def two_substeps(solver, state_in, state_out, control):
        for _ in range(2):
            solver.step(state_in, state_out, control, None, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_in, name), getattr(state_out, name))

    # The very first launches of graph_solver happen inside the capture.
    with wp.ScopedCapture(device) as capture:
        two_substeps(graph_solver, graph_a, graph_b, graph_control)
    for _ in range(64):
        wp.capture_launch(capture.graph)
        two_substeps(eager_solver, eager_a, eager_b, eager_control)
    for name in ("joint_q", "joint_qd"):
        captured = getattr(graph_a, name).numpy()
        test.assertTrue(np.isfinite(captured).all(), f"{name} became non-finite under graph replay")
        np.testing.assert_allclose(captured, getattr(eager_a, name).numpy(), rtol=0.0, atol=2.0e-6, err_msg=name)


def test_mass_refresh_has_no_obsolete_limit_count_state(test, device):
    """Keep no per-step joint-limit count state for the mass refresh."""
    solver = SolverFeatherPGS(_build_model(device, ground=False))
    for attribute in ("aug_limit_counts", "aug_prev_limit_counts", "limit_change_mask"):
        test.assertFalse(hasattr(solver, attribute), f"obsolete mass-refresh state {attribute!r} was restored")


class TestFeatherPGSMassUpdateInterval(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_mass_refresh_cadence_follows_interval",
    "test_stale_factorization_stays_close_to_interval_one",
    "test_model_change_request_refreshes_a_reuse_step",
    "test_timestep_change_forces_a_refresh",
    "test_captured_two_substep_replay_matches_eager",
    "test_mass_refresh_has_no_obsolete_limit_count_state",
):
    add_function_test(TestFeatherPGSMassUpdateInterval, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
