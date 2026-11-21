# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the QACC_SMOOTH observable and the qacc_smooth override of SolverMuJoCo."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import MUJOCO_MODULES, requires

_WORLD_COUNT = 2
_DT = 0.005


def _build_model(*, ground: bool = True, hinge_q: float = 0.29, loop: bool = False) -> newton.Model:
    """Build worlds with a limited hinge, a ball joint, and a free box with an offset center of mass.

    By default the hinge starts next to its upper limit and moves toward it,
    and the box rests on the ground, so contacts and the joint limit are active
    from the first step. With ``loop``, a revolute loop joint connects the
    ball joint's child to the world.
    """
    world = newton.ModelBuilder()
    link0 = world.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, -0.2), inertia=wp.diag(wp.vec3(0.01, 0.02, 0.01)))
    world.add_shape_box(link0, hx=0.05, hy=0.05, hz=0.2)
    hinge = world.add_joint_revolute(
        -1,
        link0,
        axis=newton.Axis.Y,
        parent_xform=wp.transform((0.0, 0.0, 1.5), wp.quat_identity()),
        limit_lower=-0.3,
        limit_upper=0.3,
    )
    link1 = world.add_link(mass=0.5, inertia=wp.diag(wp.vec3(0.01, 0.02, 0.03)))
    world.add_shape_sphere(link1, radius=0.1)
    ball = world.add_joint_ball(
        link0,
        link1,
        parent_xform=wp.transform((0.0, 0.0, -0.4), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4)),
        child_xform=wp.transform((0.0, 0.0, 0.2), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)),
    )
    world.add_articulation([hinge, ball])
    if loop:
        # Close the chain to the world near the ball body's initial position.
        world.add_joint_revolute(-1, link1, parent_xform=wp.transform((0.0, 0.0, 0.9), wp.quat_identity()))
    box = world.add_body(xform=wp.transform((0.6, 0.0, 0.1), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3)))
    world.add_shape_box(box, hx=0.1, hy=0.15, hz=0.1)
    world.body_com[box] = wp.vec3(0.03, -0.02, 0.01)
    world.joint_q[0] = hinge_q

    builder = newton.ModelBuilder()
    if ground:
        builder.add_ground_plane()
    for _ in range(_WORLD_COUNT):
        builder.add_world(world)
    model = builder.finalize(device="cuda:0")
    qd = np.random.default_rng(0).normal(size=model.joint_dof_count).astype(np.float32)
    qd[model.joint_qd_start.numpy()[:-1][model.joint_type.numpy() == newton.JointType.REVOLUTE]] = 2.0
    model.joint_qd.assign(qd)
    return model


def _dof_ranges(model: newton.Model, joint_type: newton.JointType) -> list[np.ndarray]:
    """Return the Newton DOF indices of each joint of ``joint_type``."""
    start = model.joint_qd_start.numpy()
    return [np.arange(start[j], start[j + 1]) for j in np.flatnonzero(model.joint_type.numpy() == joint_type)]


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
@requires(*MUJOCO_MODULES)
class TestMuJoCoQaccSmooth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _build_model()

    def _native_and_override_solvers(self, **kwargs):
        """Return a native solver and an override solver with ``QACC_SMOOTH`` observables."""
        native = SolverMuJoCo(self.model, **kwargs)
        override = SolverMuJoCo(self.model, **kwargs)
        kinds = {SolverMuJoCo.ObservableKind.QACC_SMOOTH}
        return native, native.observables(kinds), override, override.observables(kinds)

    def test_qacc_smooth_observable(self):
        """Report MuJoCo's qacc_smooth as the time derivative of Newton's joint_qd."""
        model = self.model
        solver = SolverMuJoCo(model, integrator="euler")
        observables = solver.observables({SolverMuJoCo.ObservableKind.QACC_SMOOTH})
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, model.control(), None, _DT, observables=observables)
        qacc_smooth = observables.qacc_smooth.numpy()

        # Hinge DOFs carry MuJoCo's values unchanged.
        mjc_to_newton = solver.mjc_dof_to_newton_dof.numpy()
        mjw_qacc_smooth = solver.mjw_data.qacc_smooth.numpy()
        for dofs in _dof_ranges(model, newton.JointType.REVOLUTE):
            world, column = np.argwhere(mjc_to_newton == dofs[0])[0]
            self.assertEqual(qacc_smooth[dofs[0]], mjw_qacc_smooth[world, column])

        # Without constraints, Euler integrates qacc_smooth, so a finite difference
        # of joint_qd checks the free and ball conventions too. The box's linear
        # acceleration is that of its center of mass, which is gravity.
        free_model = _build_model(ground=False, hinge_q=0.0)
        solver = SolverMuJoCo(free_model, integrator="euler", disable_contacts=True)
        observables = solver.observables({SolverMuJoCo.ObservableKind.QACC_SMOOTH})
        state_in, state_out = free_model.state(), free_model.state()
        dt = 1.0e-3
        solver.step(state_in, state_out, free_model.control(), None, dt, observables=observables)
        qacc_smooth = observables.qacc_smooth.numpy()
        finite_difference = (state_out.joint_qd.numpy() - state_in.joint_qd.numpy()) / dt
        np.testing.assert_allclose(qacc_smooth, finite_difference, atol=2.0e-2)
        for dofs in _dof_ranges(free_model, newton.JointType.FREE):
            np.testing.assert_allclose(qacc_smooth[dofs[:3]], [0.0, 0.0, -9.81], atol=1.0e-4)
        self.assertGreater(
            np.abs(qacc_smooth[np.concatenate(_dof_ranges(free_model, newton.JointType.BALL))]).max(), 1.0
        )

        # In-place stepping reads the same input frames, and loop-joint DOFs report zero.
        loop_model = _build_model(loop=True)
        results = []
        for in_place in (False, True):
            solver = SolverMuJoCo(loop_model, integrator="euler")
            observables = solver.observables({SolverMuJoCo.ObservableKind.QACC_SMOOTH})
            observables.qacc_smooth.fill_(float("nan"))
            state_in = loop_model.state()
            state_out = state_in if in_place else loop_model.state()
            solver.step(state_in, state_out, loop_model.control(), None, _DT, observables=observables)
            results.append(observables.qacc_smooth.numpy())
        np.testing.assert_array_equal(results[1], results[0])
        articulation = loop_model.joint_articulation.numpy()
        start = loop_model.joint_qd_start.numpy()
        loop_dofs = [d for j in np.flatnonzero(articulation < 0) for d in range(start[j], start[j + 1])]
        self.assertEqual(len(loop_dofs), _WORLD_COUNT)
        np.testing.assert_array_equal(results[0][loop_dofs], 0.0)
        self.assertTrue(np.isfinite(results[0]).all())

    def test_override_reproduces_native_step(self):
        """Reproduce the native step when fed native qacc_smooth, with an empty mask, and after clearing."""
        model = self.model
        cases = [
            {"integrator": "euler"},
            {"integrator": "implicit"},
            {"integrator": "implicitfast"},
            {"integrator": "implicitfast", "use_mujoco_contacts": False, "use_speculative_contacts": True},
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                speculative = kwargs.get("use_speculative_contacts", False)
                native, native_obs, fed, fed_obs = self._native_and_override_solvers(**kwargs)
                unmasked = SolverMuJoCo(model, **kwargs)
                cleared = SolverMuJoCo(model, **kwargs)
                self.assertEqual(fed.use_speculative_contacts, speculative)
                values = wp.zeros(model.joint_dof_count, dtype=wp.float32)
                fed.set_qacc_smooth_override(values)
                no_dofs = wp.zeros(model.joint_dof_count, dtype=wp.bool)
                unmasked.set_qacc_smooth_override(values, dof_mask=no_dofs)
                cleared.set_qacc_smooth_override(values)
                cleared.set_qacc_smooth_override(None)

                contacts = None
                if speculative:
                    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.1)
                    contacts = pipeline.contacts()
                state_in = model.state()
                outputs = {name: model.state() for name in ("native", "fed", "unmasked", "cleared")}
                control = model.control()
                constraint_rows = 0
                for _ in range(20):
                    if speculative:
                        pipeline.collide(state_in, contacts, dt=_DT)
                    native.step(state_in, outputs["native"], control, contacts, _DT, observables=native_obs)
                    values.assign(native_obs.qacc_smooth)
                    fed.step(state_in, outputs["fed"], control, contacts, _DT, observables=fed_obs)
                    unmasked.step(state_in, outputs["unmasked"], control, contacts, _DT)
                    cleared.step(state_in, outputs["cleared"], control, contacts, _DT)
                    constraint_rows = max(constraint_rows, int(native.mjw_data.nefc.numpy().min()))

                    expected = outputs["native"]
                    np.testing.assert_array_equal(outputs["cleared"].joint_q.numpy(), expected.joint_q.numpy())
                    np.testing.assert_array_equal(outputs["cleared"].joint_qd.numpy(), expected.joint_qd.numpy())
                    for name in ("fed", "unmasked"):
                        np.testing.assert_allclose(outputs[name].joint_q.numpy(), expected.joint_q.numpy(), atol=1e-6)
                        np.testing.assert_allclose(outputs[name].joint_qd.numpy(), expected.joint_qd.numpy(), atol=1e-4)
                    np.testing.assert_allclose(fed_obs.qacc_smooth.numpy(), native_obs.qacc_smooth.numpy(), atol=1e-4)
                    state_in.assign(expected)
                # Every world solved contact or limit constraints.
                self.assertGreater(constraint_rows, 0)

    def test_override_masked_dofs(self):
        """Override only masked DOFs, including some components of a free joint."""
        model = self.model
        native, native_obs, solver, obs = self._native_and_override_solvers()
        hinge = _dof_ranges(model, newton.JointType.REVOLUTE)[0]
        free = _dof_ranges(model, newton.JointType.FREE)[0]
        masked = np.concatenate([hinge, free[[2, 4]]])
        mask = np.zeros(model.joint_dof_count, dtype=bool)
        mask[masked] = True
        values = wp.full(model.joint_dof_count, 3.0, dtype=wp.float32)
        solver.set_qacc_smooth_override(values, dof_mask=wp.array(mask, dtype=wp.bool))

        state_in, native_out, out = model.state(), model.state(), model.state()
        control = model.control()
        native.step(state_in, native_out, control, None, _DT, observables=native_obs)
        solver.step(state_in, out, control, None, _DT, observables=obs)

        qacc_smooth, native_qacc_smooth = obs.qacc_smooth.numpy(), native_obs.qacc_smooth.numpy()
        np.testing.assert_allclose(qacc_smooth[masked], 3.0, atol=1e-5)
        self.assertEqual(qacc_smooth[hinge[0]], 3.0)
        np.testing.assert_allclose(qacc_smooth[~mask], native_qacc_smooth[~mask], atol=1e-5)
        self.assertTrue(np.all(np.abs(native_qacc_smooth[masked] - 3.0) > 0.1))
        # The override changes the motion of world 0 only.
        world_dofs = model.joint_dof_count // _WORLD_COUNT
        joint_qd, native_joint_qd = out.joint_qd.numpy(), native_out.joint_qd.numpy()
        self.assertGreater(np.abs(joint_qd[:world_dofs] - native_joint_qd[:world_dofs]).max(), 1.0e-3)
        np.testing.assert_allclose(joint_qd[world_dofs:], native_joint_qd[world_dofs:], atol=1e-5)

    def test_override_graph_capture(self):
        """Read override values written between CUDA graph launches."""
        model = self.model
        eager, graphed = SolverMuJoCo(model), SolverMuJoCo(model)
        eager_values = wp.zeros(model.joint_dof_count, dtype=wp.float32)
        graphed_values = wp.zeros(model.joint_dof_count, dtype=wp.float32)
        eager.set_qacc_smooth_override(eager_values)
        graphed.set_qacc_smooth_override(graphed_values)
        control = model.control()
        eager_states = (model.state(), model.state())
        graphed_states = (model.state(), model.state())

        # Load modules before capture.
        eager.step(model.state(), model.state(), control, None, _DT)
        eager = SolverMuJoCo(model)
        eager.set_qacc_smooth_override(eager_values)
        with wp.ScopedCapture(device=model.device) as capture:
            graphed.step(graphed_states[0], graphed_states[1], control, None, _DT)
            graphed.step(graphed_states[1], graphed_states[0], control, None, _DT)

        for k in range(5):
            for values in (eager_values, graphed_values):
                values.fill_(float(k) - 2.0)
            eager.step(eager_states[0], eager_states[1], control, None, _DT)
            eager.step(eager_states[1], eager_states[0], control, None, _DT)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(graphed_states[0].joint_q.numpy(), eager_states[0].joint_q.numpy(), atol=1e-6)
            np.testing.assert_allclose(graphed_states[0].joint_qd.numpy(), eager_states[0].joint_qd.numpy(), atol=1e-5)
        # The last launch read the values written before it.
        hinge = _dof_ranges(model, newton.JointType.REVOLUTE)[0]
        self.assertEqual(graphed.mjc_dof_to_newton_dof.numpy()[0, 0], hinge[0])
        self.assertEqual(float(graphed.mjw_data.qacc_smooth.numpy()[0, 0]), 2.0)

    def test_rejects_unsupported_configurations(self):
        """Reject unsupported backends, integrators, sleeping, data intervals, and invalid arrays."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cuda:0")
        dof_count = model.joint_dof_count
        values = wp.zeros(dof_count, dtype=wp.float32)
        for kwargs, message in (
            ({"use_mujoco_cpu": True}, "use_mujoco_cpu=False"),
            ({"integrator": "rk4"}, "got rk4"),
            ({"integrator": "discrete"}, "got discrete"),
            ({"enable_sleeping": True}, "sleeping"),
        ):
            with self.subTest(**kwargs):
                solver = SolverMuJoCo(model, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    solver.set_qacc_smooth_override(values)
                solver.set_qacc_smooth_override(None)
                if kwargs.get("use_mujoco_cpu") or kwargs.get("integrator") == "rk4":
                    self.assertNotIn(SolverMuJoCo.ObservableKind.QACC_SMOOTH, solver.supported_observables)
        with self.assertWarns(newton.NewtonDeprecationWarning):
            solver = SolverMuJoCo(model, update_data_interval=2)
        observables = solver.observables({SolverMuJoCo.ObservableKind.QACC_SMOOTH})
        with self.assertRaisesRegex(ValueError, "update_data_interval"):
            solver.step(model.state(), model.state(), model.control(), None, _DT, observables=observables)

        solver = SolverMuJoCo(model)
        with self.assertRaises(AttributeError):
            solver.use_speculative_contacts = True
        mask = wp.zeros(dof_count, dtype=wp.bool)
        for name, call, error, message in (
            ("numpy values", lambda: solver.set_qacc_smooth_override(np.zeros(dof_count)), TypeError, "Warp array"),
            (
                "float64 values",
                lambda: solver.set_qacc_smooth_override(wp.zeros(dof_count, dtype=wp.float64)),
                ValueError,
                "dtype float32",
            ),
            (
                "short values",
                lambda: solver.set_qacc_smooth_override(wp.zeros(dof_count - 1, dtype=wp.float32)),
                ValueError,
                "joint_dof_count",
            ),
            (
                "float mask",
                lambda: solver.set_qacc_smooth_override(values, dof_mask=values),
                ValueError,
                "dtype bool",
            ),
            (
                "CPU values",
                lambda: solver.set_qacc_smooth_override(wp.zeros(dof_count, dtype=wp.float32, device="cpu")),
                ValueError,
                "is on cpu",
            ),
            ("mask without values", lambda: solver.set_qacc_smooth_override(None, dof_mask=mask), ValueError, "None"),
        ):
            with self.subTest(name), self.assertRaisesRegex(error, message):
                call()
        self.assertIsNone(solver._qacc_smooth_override)


if __name__ == "__main__":
    unittest.main(verbosity=2)
