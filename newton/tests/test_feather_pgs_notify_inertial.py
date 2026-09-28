# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 60.0
INITIAL_JOINT_Q = 0.3
NEW_COM = (0.15, 0.0, 0.05)


def _build_model(device, com=None):
    """Single-link pendulum on a Y-axis revolute joint, box COM at the origin.

    With the COM at the body origin (the pivot) gravity exerts no torque, so
    any post-randomization swing is attributable to the COM offset alone.
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
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    joint_q_history = []
    for _ in range(num_steps):
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        joint_q_history.append(state_0.joint_q.numpy().copy())
    return np.stack(joint_q_history)


class TestFeatherPGSNotifyInertial(unittest.TestCase):
    def test_stepped_solver_releases_resources_without_cyclic_gc(self):
        """Release a stepped solver before cyclic GC can finalize its streams first."""
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            model = _build_model(wp.get_device())
            solver = SolverFeatherPGS(model)
            reference = weakref.ref(solver)
            solver.step(model.state(), model.state(), model.control(), None, DT)
            del solver
            self.assertIsNone(reference(), "Stepping must not create a solver ownership cycle")
        finally:
            if gc_enabled:
                gc.enable()

    def test_step_refreshes_body_pose_after_generalized_coordinate_update(self):
        """A direct ``joint_q`` update must not require a caller-side FK pass.

        ``SolverFeatherPGS.step`` historically derives body poses from the
        generalized coordinates before inverse dynamics.  Reset and direct
        generalized-coordinate callers rely on that public step behavior.
        """
        device = wp.get_device()
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

            SolverFeatherPGS(model, pgs_mode="split").step(
                state_0,
                state_1,
                model.control(),
                None,
                DT,
            )
            outputs[caller_refreshes_fk] = {
                "joint_q": state_1.joint_q.numpy().copy(),
                "joint_qd": state_1.joint_qd.numpy().copy(),
                "body_q": state_0.body_q.numpy().copy(),
                "stale_body_q": stale_body_q,
            }

        self.assertFalse(np.allclose(outputs[False]["stale_body_q"], outputs[True]["body_q"]))
        np.testing.assert_allclose(outputs[False]["body_q"], outputs[True]["body_q"], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outputs[False]["joint_q"], outputs[True]["joint_q"], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outputs[False]["joint_qd"], outputs[True]["joint_qd"], rtol=0.0, atol=1.0e-6)

    def test_notify_refreshes_baked_com_and_inertia_buffers(self):
        """Verify BODY_INERTIAL_PROPERTIES re-derives body_X_com and body_I_m.

        Both buffers are baked from the model at construction time; writing
        model.body_com/body_mass alone must not change them until the solver
        is notified.
        """
        device = wp.get_device()
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
        self.assertFalse(np.allclose(solver.body_I_m.numpy(), stale_I_m))
        self.assertEqual(solver._mass_update_requested.numpy()[0], 1)

    def test_com_change_with_notify_matches_freshly_built_solver(self):
        """Verify a mid-run COM change plus notify reproduces baked-COM dynamics.

        The trajectory after the change must match a solver constructed with
        the new COM already in the model and stepped from the same state; a
        stale solver (COM written without notify) must diverge, proving the
        comparison is not vacuous.
        """
        device = wp.get_device()
        pre_steps, post_steps = 30, 60

        # Reference: solver built with the new COM, run from the pre-change
        # state (which is static: zero torque while the COM sits at the pivot).
        reference_model = _build_model(device, com=NEW_COM)
        reference_solver = SolverFeatherPGS(reference_model)
        reference = _run_trajectory(reference_model, reference_solver, post_steps)

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
        self.assertGreater(stale_drift, 1.0e-2, "stale-solver trajectory should diverge without notify")


if __name__ == "__main__":
    unittest.main(verbosity=2)
