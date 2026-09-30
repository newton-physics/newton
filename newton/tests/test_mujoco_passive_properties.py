# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for narrow MuJoCo passive-DOF property updates."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverMuJoCo


@wp.kernel
def _publish_budget(values: wp.array[float], friction: wp.array[float], damping: wp.array[float]):
    dof = wp.tid()
    if dof % 2 == 0:
        friction[dof] = values[dof]
        damping[dof] = 2.0 * values[dof]


def _make_model(*, worlds: int = 2, dofs: int = 2, device="cpu", custom_attributes: bool = True) -> newton.Model:
    """Build a replicated hinge chain with passive parameters and joint limits."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if custom_attributes:
        SolverMuJoCo.register_custom_attributes(template)
    joints = []
    parent = -1
    for _ in range(dofs):
        body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(0.0))
        joints.append(
            template.add_joint_revolute(
                parent,
                body,
                armature=0.1,
                friction=0.0,
                damping=0.2,
                limit_lower=-1.0,
                limit_upper=1.0,
            )
        )
        parent = body
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if custom_attributes:
        SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template, worlds)
    return builder.finalize(device=device)


class TestMuJoCoPassiveProperties(unittest.TestCase):
    def _cuda_device(self):
        """Skip capture tests when no CUDA device with a mempool is available."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")
        return device

    def test_eager_updates_preserve_armature_and_optional_parameters(self):
        """Publish all passive fields on both backends without modifying kinematic armature."""
        for cpu in (True, False):
            for custom_attributes in (True, False):
                with self.subTest(cpu=cpu, custom_attributes=custom_attributes):
                    model = _make_model(worlds=1, custom_attributes=custom_attributes)
                    body_flags = model.body_flags.numpy()
                    body_flags[0] = int(newton.BodyFlags.KINEMATIC)
                    model.body_flags.assign(body_flags)
                    solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, separate_worlds=False, disable_contacts=True)
                    mapping = solver.mjc_dof_to_newton_dof.numpy()
                    armature = solver.mjw_model.dof_armature.numpy().copy()
                    invweight = solver.mjw_model.dof_invweight0.numpy().copy()
                    initial_solref = solver.mjw_model.dof_solref.numpy().copy()
                    initial_solimp = solver.mjw_model.dof_solimp.numpy().copy()
                    friction = np.arange(1, model.joint_dof_count + 1, dtype=np.float32)
                    model.joint_friction.assign(friction)
                    model.joint_damping.assign(2.0 * friction)
                    model.joint_armature.fill_(7.0)
                    if custom_attributes:
                        solref = model.mujoco.solreffriction.numpy()
                        solref[:, 0] = -100.0 * friction
                        solref[:, 1] = -10.0
                        model.mujoco.solreffriction.assign(solref)
                        solimp = model.mujoco.solimpfriction.numpy()
                        solimp[:, 0] = np.linspace(0.8, 0.9, model.joint_dof_count)
                        model.mujoco.solimpfriction.assign(solimp)
                    solver.update_joint_dof_passive_properties()
                    expected = {
                        "dof_frictionloss": friction[mapping],
                        "dof_damping": 2.0 * friction[mapping],
                        "dof_solref": solref[mapping] if custom_attributes else initial_solref,
                        "dof_solimp": solimp[mapping] if custom_attributes else initial_solimp,
                        "dof_armature": armature,
                        "dof_invweight0": invweight,
                    }
                    for name, values in expected.items():
                        np.testing.assert_allclose(getattr(solver.mjw_model, name).numpy(), values, err_msg=name)
                        if cpu:
                            np.testing.assert_allclose(
                                getattr(solver.mj_model, name), values[0], rtol=1.0e-5, err_msg=name
                            )

    def test_model_without_dofs(self):
        """Accept passive updates for a fixed-joint model without DOFs."""
        for cpu in (True, False):
            with self.subTest(cpu=cpu):
                builder = newton.ModelBuilder()
                body = builder.add_link()
                builder.add_shape_sphere(body, radius=0.1)
                joint = builder.add_joint_fixed(parent=-1, child=body)
                builder.add_articulation([joint])
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu)
                solver.update_joint_dof_passive_properties()
                self.assertEqual(solver.mj_model.nv, 0)

    def test_captured_publication_preserves_unrelated_properties(self):
        """Replay distinct world budgets while preserving unowned DOFs and cached inertia."""
        device = self._cuda_device()
        model = _make_model(device=device)
        solver = SolverMuJoCo(model, disable_contacts=True, iterations=2)
        untouched = ("dof_armature", "dof_invweight0", "jnt_range", "jnt_solref", "qpos0", "actuator_gainprm")
        before = {name: getattr(solver.mjw_model, name).numpy().copy() for name in untouched}
        model.joint_armature.fill_(7.0)
        model.joint_limit_upper.fill_(0.5)
        model.mujoco.dof_ref.fill_(0.2)
        values = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        mapping = solver.mjc_dof_to_newton_dof.numpy()
        with wp.ScopedCapture(device=model.device) as capture:
            wp.launch(
                _publish_budget,
                dim=model.joint_dof_count,
                inputs=[values, model.joint_friction, model.joint_damping],
                device=model.device,
            )
            solver.update_joint_dof_passive_properties()
        for scale in (1.0, 0.0, 3.0):
            values.assign(np.arange(1, model.joint_dof_count + 1, dtype=np.float32) * scale)
            solref = model.mujoco.solreffriction.numpy()
            solref[:, 0] = np.arange(1, model.joint_dof_count + 1) * -100.0
            solref[:, 1] = -10.0
            model.mujoco.solreffriction.assign(solref)
            solimp = model.mujoco.solimpfriction.numpy()
            solimp[:, 0] = np.linspace(0.8, 0.9, model.joint_dof_count)
            model.mujoco.solimpfriction.assign(solimp)
            wp.capture_launch(capture.graph)
            for target, source in (
                ("dof_frictionloss", model.joint_friction),
                ("dof_damping", model.joint_damping),
                ("dof_solref", model.mujoco.solreffriction),
                ("dof_solimp", model.mujoco.solimpfriction),
            ):
                np.testing.assert_allclose(getattr(solver.mjw_model, target).numpy(), source.numpy()[mapping])
            np.testing.assert_array_equal(model.joint_friction.numpy()[1::2], 0.0)
            np.testing.assert_allclose(model.joint_damping.numpy()[1::2], 0.2)
            for name in untouched:
                np.testing.assert_array_equal(getattr(solver.mjw_model, name).numpy(), before[name])
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.dof_armature.numpy(), 7.0)
        np.testing.assert_allclose(solver.mjw_model.dof_frictionloss.numpy(), model.joint_friction.numpy()[mapping])
        self.assertFalse(np.allclose(solver.mjw_model.dof_invweight0.numpy(), before["dof_invweight0"]))

    def test_matches_full_update_dynamics(self):
        """Match full-update trajectories with changing friction and damping on both backends."""
        for cpu in (True, False) if wp.get_cuda_device_count() else (True,):
            with self.subTest(cpu=cpu):
                model = _make_model(worlds=1 if cpu else 2, device="cpu" if cpu else "cuda:0")
                solvers = [
                    SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True, iterations=10) for _ in range(2)
                ]
                states = [[model.state(), model.state()] for _ in solvers]
                control = model.control()
                control.joint_f.fill_(0.5)
                for pair in states:
                    pair[0].joint_qd.fill_(0.3)
                    newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
                for step in range(30):
                    # Include zero-to-positive friction after initially compiling with zero.
                    budget = np.arange(1, model.joint_dof_count + 1, dtype=np.float32) * (step % 3) * 0.2
                    model.joint_friction.assign(budget)
                    model.joint_damping.assign(budget + 0.1)
                    solvers[0].notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
                    solvers[1].update_joint_dof_passive_properties()
                    if cpu:
                        np.testing.assert_allclose(solvers[1].mj_model.dof_frictionloss, budget)
                    for solver, pair in zip(solvers, states, strict=True):
                        solver.step(pair[0], pair[1], control, None, 0.005)
                        pair.reverse()
                    for field in ("joint_q", "joint_qd"):
                        np.testing.assert_allclose(
                            getattr(states[0][0], field).numpy(),
                            getattr(states[1][0], field).numpy(),
                            atol=1.0e-6,
                            rtol=1.0e-5,
                        )

    def test_captured_update_wakes_sleeping_worlds(self):
        """Wake sleeping worlds after a captured passive parameter change."""
        device = self._cuda_device()
        model = _make_model(device=device)
        sleep_policy = model.mujoco.sleep_policy.numpy()
        sleep_policy[::2] = int(SolverMuJoCo.SleepPolicy.INIT)
        model.mujoco.sleep_policy.assign(sleep_policy)
        solver = SolverMuJoCo(model, use_mujoco_contacts=True, enable_sleeping=True, solver="newton", iterations=2)
        np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), 0)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.update_joint_dof_passive_properties()
        wp.capture_launch(capture.graph)
        self.assertTrue(np.all(solver.mjw_data.tree_asleep.numpy() < 0))
        np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), 1)
        np.testing.assert_array_equal(solver.mjw_data.nv_awake.numpy(), 2)

    def test_captured_publication_and_step_match_full_notification(self):
        """Consume changing friction budgets in the same graph as the simulation step."""
        device = self._cuda_device()
        model = _make_model(worlds=2, dofs=1, device=device)
        reference = SolverMuJoCo(model, disable_contacts=True, iterations=10)
        captured = SolverMuJoCo(model, disable_contacts=True, iterations=10)
        state_in = model.state()
        reference_out = model.state()
        captured_out = model.state()
        control = model.control()
        control.joint_f.fill_(0.5)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        budget = wp.zeros(model.joint_dof_count, dtype=float, device=device)
        captured.step(state_in, captured_out, control, None, 0.005)
        with wp.ScopedCapture(device=device) as capture:
            wp.copy(model.joint_friction, budget)
            captured.update_joint_dof_passive_properties()
            captured.step(state_in, captured_out, control, None, 0.005)

        velocities = []
        for values in ([0.0, 0.0], [0.25, 2.0], [2.0, 0.25], [0.0, 0.0]):
            budget.assign(np.array(values, dtype=np.float32))
            model.joint_friction.assign(budget)
            reference.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            reference.step(state_in, reference_out, control, None, 0.005)
            wp.capture_launch(capture.graph)
            for field in ("joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(captured_out, field).numpy(),
                    getattr(reference_out, field).numpy(),
                    atol=1.0e-6,
                    rtol=1.0e-5,
                )
            velocities.append(captured_out.joint_qd.numpy().copy())
        self.assertGreater(float(velocities[0][1]), float(velocities[1][1]))


if __name__ == "__main__":
    unittest.main()
