# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model-change notifications and state refresh of SolverFeatherPGS."""

import gc
import unittest
import weakref

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 60.0
INITIAL_JOINT_Q = 0.3
NEW_COM = (0.15, 0.0, 0.05)


def _build_model(device, com=None):
    """Single-link pendulum on a Y-axis revolute joint, box COM at the pivot.

    With the COM at the pivot gravity exerts no torque, so any swing after a COM change is
    attributable to the change alone.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = INITIAL_JOINT_Q
    model = builder.finalize(device=device)
    if com is not None:
        body_com = model.body_com.numpy()
        body_com[0] = com
        model.body_com.assign(body_com)
    return model


def _run_trajectory(model, solver, num_steps):
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
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


def test_stepped_solver_releases_resources_without_cyclic_gc(test, device):
    """Release a stepped solver without relying on the cyclic garbage collector."""
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        model = _build_model(device)
        solver = SolverFeatherPGS(model)
        reference = weakref.ref(solver)
        solver.step(model.state(), model.state(), model.control(), None, DT)
        del solver
        test.assertIsNone(reference(), "stepping must not create a solver ownership cycle")
    finally:
        if gc_enabled:
            gc.enable()


def test_step_refreshes_body_pose_after_generalized_coordinate_update(test, device):
    """Derive body poses from joint_q in step, so a direct joint_q write needs no caller-side FK."""
    outputs = {}
    for caller_refreshes_fk in (False, True):
        model = _build_model(device, com=NEW_COM)
        state_0 = model.state()
        state_1 = model.state()
        joint_q = state_0.joint_q.numpy()
        joint_q[0] = 1.1
        state_0.joint_q.assign(joint_q)
        stale_body_q = state_0.body_q.numpy().copy()
        if caller_refreshes_fk:
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        SolverFeatherPGS(model).step(state_0, state_1, model.control(), None, DT)
        outputs[caller_refreshes_fk] = {
            "joint_q": state_1.joint_q.numpy().copy(),
            "joint_qd": state_1.joint_qd.numpy().copy(),
            "body_q": state_0.body_q.numpy().copy(),
            "stale_body_q": stale_body_q,
        }
    test.assertFalse(np.allclose(outputs[False]["stale_body_q"], outputs[True]["body_q"]))
    for key in ("body_q", "joint_q", "joint_qd"):
        np.testing.assert_allclose(outputs[False][key], outputs[True][key], rtol=0.0, atol=1.0e-6)


def test_step_refreshes_reused_state_after_joint_q_write(test, device):
    """Pick up a joint_q write into a state the solver produced on the previous step."""
    model = _build_model(device)
    solver = SolverFeatherPGS(model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    joint_q = state_1.joint_q.numpy()
    joint_q[0] = -0.9
    state_1.joint_q.assign(joint_q)
    solver.step(state_1, state_0, control, None, DT)

    reference_model = _build_model(device)
    reference_state = reference_model.state()
    reference_state.joint_q.assign(state_1.joint_q)
    reference_state.joint_qd.assign(state_1.joint_qd)
    reference_out = reference_model.state()
    SolverFeatherPGS(reference_model).step(reference_state, reference_out, reference_model.control(), None, DT)
    np.testing.assert_allclose(state_0.joint_q.numpy(), reference_out.joint_q.numpy(), rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(state_0.body_q.numpy(), reference_out.body_q.numpy(), rtol=0.0, atol=1.0e-6)


def test_notify_refreshes_baked_com_and_inertia_buffers(test, device):
    """Re-derive the solver's COM and inertia buffers on BODY_INERTIAL_PROPERTIES."""
    model = _build_model(device)
    solver = SolverFeatherPGS(model)
    stale_X_com = solver.body_X_com.numpy().copy()
    stale_I_m = solver.body_I_m.numpy().copy()

    body_com = model.body_com.numpy()
    body_com[0] = NEW_COM
    model.body_com.assign(body_com)
    body_mass = model.body_mass.numpy()
    body_mass[0] *= 2.0
    model.body_mass.assign(body_mass)

    np.testing.assert_array_equal(solver.body_X_com.numpy(), stale_X_com)
    solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
    np.testing.assert_allclose(
        solver.body_X_com.numpy()[0][:3], np.asarray(NEW_COM, dtype=np.float32), rtol=0.0, atol=0.0
    )
    test.assertFalse(np.allclose(solver.body_I_m.numpy(), stale_I_m))
    test.assertEqual(solver._mass_update_requested.numpy()[0], 1)


def test_com_change_with_notify_matches_freshly_built_solver(test, device):
    """Reproduce a freshly built solver's dynamics after a mid-run COM change and notification."""
    pre_steps, post_steps = 30, 60
    reference_model = _build_model(device, com=NEW_COM)
    reference = _run_trajectory(reference_model, SolverFeatherPGS(reference_model), post_steps)

    histories = {}
    for notify in (True, False):
        model = _build_model(device)
        solver = SolverFeatherPGS(model)
        _run_trajectory(model, solver, pre_steps)
        body_com = model.body_com.numpy()
        body_com[0] = NEW_COM
        model.body_com.assign(body_com)
        if notify:
            solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
        histories[notify] = _run_trajectory(model, solver, post_steps)

    np.testing.assert_allclose(histories[True], reference, rtol=0.0, atol=1.0e-5)
    stale_drift = np.abs(histories[False] - reference).max()
    test.assertGreater(stale_drift, 1.0e-2, "a solver that was not notified should diverge")


def test_kinematic_flag_change_is_picked_up(test, device):
    """Hold a body still once it is flagged kinematic and the solver is notified."""
    model = _build_model(device, com=NEW_COM)
    solver = SolverFeatherPGS(model)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    flags = model.body_flags.numpy()
    flags[0] = int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
    for _ in range(30):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    test.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), INITIAL_JOINT_Q, places=5)


def _shift_child_frame(model):
    joint_X_c = model.joint_X_c.numpy()
    joint_X_c[0, 0] = 0.5
    model.joint_X_c.assign(joint_X_c)


def test_joint_frame_change_with_notify_matches_freshly_built_solver(test, device):
    """Refresh cached mass factors on JOINT_PROPERTIES, eagerly and under graph replay."""
    interval = 100
    reference_model = _build_model(device)
    _shift_child_frame(reference_model)
    reference = _run_trajectory(
        reference_model, SolverFeatherPGS(reference_model, update_mass_matrix_interval=interval), 5
    )

    for capture in (False, True):
        with test.subTest(capture=capture):
            model = _build_model(device)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()

            def substep(solver=solver, state_0=state_0, state_1=state_1, control=control):
                solver.step(state_0, state_1, control, None, DT)
                wp.copy(state_0.joint_q, state_1.joint_q)
                wp.copy(state_0.joint_qd, state_1.joint_qd)

            # Factor the original frame on the first step; later steps reuse it until notified.
            substep()
            initial_q = model.joint_q.numpy().copy()
            graph = None
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    substep()
            state_0.joint_q.assign(initial_q)
            state_0.joint_qd.zero_()
            _shift_child_frame(model)
            solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
            history = []
            for _ in range(5):
                if graph is None:
                    substep()
                else:
                    wp.capture_launch(graph.graph)
                history.append(state_0.joint_q.numpy().copy())
            np.testing.assert_allclose(np.asarray(history), reference, rtol=0.0, atol=1.0e-5)


class TestFeatherPGSNotifyInertial(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_stepped_solver_releases_resources_without_cyclic_gc",
    "test_step_refreshes_body_pose_after_generalized_coordinate_update",
    "test_step_refreshes_reused_state_after_joint_q_write",
    "test_notify_refreshes_baked_com_and_inertia_buffers",
    "test_com_change_with_notify_matches_freshly_built_solver",
    "test_kinematic_flag_change_is_picked_up",
    "test_joint_frame_change_with_notify_matches_freshly_built_solver",
):
    add_function_test(TestFeatherPGSNotifyInertial, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
