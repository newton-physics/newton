# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverMuJoCo.reset() clearing stale internal buffers per world."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import StateFlags
from newton.exceptions import NewtonDeprecationWarning
from newton.solvers import SolverMuJoCo


def _build_two_world_model(world_count: int = 2, device=None) -> newton.Model:
    """Build a model of ``world_count`` identical free + revolute articulations."""
    template = newton.ModelBuilder()
    body0 = template.add_link(mass=0.2, xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    template.add_shape_box(body=body0, hx=0.1, hy=0.1, hz=0.1)
    joint0 = template.add_joint_free(child=body0)

    body1 = template.add_link(mass=0.1)
    template.add_shape_capsule(body=body1, radius=0.05, half_height=0.15)
    joint1 = template.add_joint_revolute(
        parent=body0,
        child=body1,
        parent_xform=wp.transform((0.0, 0.15, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, -0.15, 0.0), wp.quat_identity()),
        axis=(0.0, 0.0, 1.0),
    )
    template.add_articulation([joint0, joint1])

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i in range(world_count):
        builder.add_world(template, xform=wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity()))
    return builder.finalize(device=device)


class TestMuJoCoReset(unittest.TestCase):
    def _setup_stepped_solver(self):
        self.model = _build_two_world_model(world_count=2)
        self.solver = SolverMuJoCo(self.model, iterations=2, ls_iterations=2)
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        newton.eval_fk(self.model, self.state_in.joint_q, self.state_in.joint_qd, self.state_in)
        # One step populates the internal buffers we then expect reset to clear.
        self.collision_pipeline.collide(self.state_in, self.contacts)
        self.solver.step(self.state_in, self.state_out, self.control, self.contacts, 1.0 / 60.0)

    def _cleared_buffers(self):
        """MuJoCo buffers that reset() must zero, skipping any that are empty.

        ``qacc`` is excluded: the solver overwrites it from ``qacc_warmstart``
        at the start of every step, so reset() deliberately leaves it alone.
        """
        d = self.solver.mjw_data
        buffers = {
            "qacc_warmstart": d.qacc_warmstart,
            "qfrc_applied": d.qfrc_applied,
            "ctrl": d.ctrl,
            "act": d.act,
            "xfrc_applied": d.xfrc_applied,
        }
        return {name: buf for name, buf in buffers.items() if buf.shape[1] > 0}

    def _poison(self):
        """Fill all reset-managed buffers with a non-zero sentinel in every world."""
        for buf in self._cleared_buffers().values():
            buf.fill_(7.0)

    def test_masked_reset_supports_cuda_graph_capture(self):
        """Capture a masked GPU reset without copying its mask to the host."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA is required for graph capture")
        self._setup_stepped_solver()
        mask = wp.array([False, True, False], dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state_out, world_mask=mask)
        self._poison()
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.solver.reset(self.state_out, world_mask=mask)
        wp.capture_launch(capture.graph)
        for name, buffer in self._cleared_buffers().items():
            values = buffer.numpy()
            np.testing.assert_array_equal(values[0], 7.0, err_msg=name)
            np.testing.assert_array_equal(values[1], 0.0, err_msg=name)

    def test_reset_masked_world_only(self):
        """A per-world mask clears the selected world and leaves the others intact."""
        self._setup_stepped_solver()
        self._poison()
        self.solver.reset(
            self.state_out,
            world_mask=wp.array([False, False, True], dtype=wp.bool, device=self.model.device),
        )
        for name, buf in self._cleared_buffers().items():
            self.assertTrue(np.all(buf.numpy() == 7.0), f"{name} changed for a global-only mask")

        mask = wp.array([True, False, False], dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state_out, world_mask=mask)

        for name, buf in self._cleared_buffers().items():
            values = buf.numpy()
            self.assertTrue(np.all(values[0] == 0.0), f"{name} not cleared in masked world 0")
            self.assertTrue(np.all(values[1] == 7.0), f"{name} wrongly cleared in unmasked world 1")

    def test_reset_rejects_local_only_mask(self):
        """Reject masks without the final global slot."""
        self._setup_stepped_solver()
        self._poison()
        mask = wp.array([True, False], dtype=wp.bool, device=self.model.device)
        with self.assertRaisesRegex(ValueError, "world_count \\+ 1"):
            self.solver.reset(self.state_out, world_mask=mask)

        for name, buf in self._cleared_buffers().items():
            self.assertTrue(np.all(buf.numpy() == 7.0), f"{name} changed by a rejected mask")

    def test_native_cpu_reset_honors_each_world_mask(self):
        """Reset only the selected native MuJoCo world's persistent buffers."""
        self.model = _build_two_world_model(world_count=2)
        solver = SolverMuJoCo(self.model, separate_worlds=True, use_mujoco_cpu=True)
        self.assertEqual(len(solver.mj_data_by_world), 2)

        def buffers(data):
            return tuple(
                buffer
                for buffer in (
                    data.qacc_warmstart,
                    data.qfrc_applied,
                    data.ctrl,
                    data.act,
                    data.xfrc_applied,
                )
                if buffer.size > 0
            )

        for mask_values, expected_by_world in (
            (None, (0.0, 0.0)),
            ((False, False, False), (7.0, 7.0)),
            ((True, True, False), (0.0, 0.0)),
            ((False, False, True), (7.0, 7.0)),
            ((False, True, False), (7.0, 0.0)),
            ((True, False, False), (0.0, 7.0)),
        ):
            with self.subTest(mask=mask_values):
                for data in solver.mj_data_by_world:
                    for buffer in buffers(data):
                        buffer[:] = 7.0
                solver.reset(
                    self.model.state(),
                    world_mask=(
                        None if mask_values is None else wp.array(mask_values, dtype=wp.bool, device=self.model.device)
                    ),
                    flags=0,
                )
                for data, expected in zip(solver.mj_data_by_world, expected_by_world, strict=True):
                    for buffer in buffers(data):
                        np.testing.assert_array_equal(buffer, expected)

    def test_native_cpu_masked_joint_reset_updates_only_selected_world(self):
        """Synchronize selected native world coordinates while preserving the other."""
        self.model = _build_two_world_model(world_count=2)
        # TODO: remove with update_data_interval.
        with self.assertWarns(NewtonDeprecationWarning):
            solver = SolverMuJoCo(
                self.model,
                separate_worlds=True,
                use_mujoco_cpu=True,
                update_data_interval=0,
            )
        state = self.model.state()
        flags = StateFlags.JOINT_Q | StateFlags.JOINT_QD
        coords_per_world = self.model.joint_coord_count // self.model.world_count
        dofs_per_world = self.model.joint_dof_count // self.model.world_count
        default_q = self.model.joint_q.numpy()
        default_qd = self.model.joint_qd.numpy()
        native_defaults = tuple((data.qpos.copy(), data.qvel.copy()) for data in solver.mj_data_by_world)

        joint_q_before = np.arange(self.model.joint_coord_count, dtype=np.float32) + 30.0
        joint_qd_before = np.arange(self.model.joint_dof_count, dtype=np.float32) + 40.0
        state.joint_q.assign(joint_q_before)
        state.joint_qd.assign(joint_qd_before)
        for world, data in enumerate(solver.mj_data_by_world):
            data.qpos[:] = np.arange(data.qpos.size) + 10.0 + world * 100.0
            data.qvel[:] = np.arange(data.qvel.size) + 20.0 + world * 100.0
        world0_qpos = solver.mj_data_by_world[0].qpos.copy()
        world0_qvel = solver.mj_data_by_world[0].qvel.copy()

        solver.reset(
            state,
            world_mask=wp.array([False, True, False], dtype=wp.bool, device=self.model.device),
            flags=flags,
        )

        np.testing.assert_array_equal(solver.mj_data_by_world[0].qpos, world0_qpos)
        np.testing.assert_array_equal(solver.mj_data_by_world[0].qvel, world0_qvel)
        expected_q = joint_q_before.copy()
        expected_qd = joint_qd_before.copy()
        expected_q[coords_per_world:] = default_q[coords_per_world:]
        expected_qd[dofs_per_world:] = default_qd[dofs_per_world:]
        np.testing.assert_array_equal(state.joint_q.numpy(), expected_q)
        np.testing.assert_array_equal(state.joint_qd.numpy(), expected_qd)
        np.testing.assert_allclose(solver.mj_data_by_world[1].qpos, native_defaults[1][0], atol=1.0e-6)
        np.testing.assert_allclose(solver.mj_data_by_world[1].qvel, native_defaults[1][1], atol=1.0e-6)

    def _assert_native_cpu_steps_all_worlds_with_row_local_forces(self, model: newton.Model):
        solver = SolverMuJoCo(
            model,
            use_mujoco_cpu=True,
            disable_contacts=True,
        )
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        forces = np.zeros(model.joint_dof_count, dtype=np.float32)
        dofs_per_world = model.joint_dof_count // model.world_count
        revolute_dof = dofs_per_world - 1
        forces[revolute_dof] = 1.0
        forces[dofs_per_world + revolute_dof] = -2.0
        control.joint_f.assign(forces)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        solver.step(state_in, state_out, control, None, 0.01)

        qd = state_out.joint_qd.numpy()
        self.assertGreater(qd[revolute_dof], 0.0)
        self.assertLess(qd[dofs_per_world + revolute_dof], 0.0)
        self.assertEqual(solver._step, 1)

    def test_native_cpu_steps_all_worlds_with_row_local_forces(self):
        """Advance every native world once and apply each CUDA row's force independently."""
        self._assert_native_cpu_steps_all_worlds_with_row_local_forces(_build_two_world_model())

    def test_native_cpu_steps_all_worlds_with_cpu_state(self):
        """Advance the complete batch when Newton state also resides on CPU."""
        self._assert_native_cpu_steps_all_worlds_with_row_local_forces(_build_two_world_model(device="cpu"))

    def test_reset_all_worlds(self):
        """A ``None`` mask clears every world."""
        self._setup_stepped_solver()
        self._poison()
        self.solver.reset(self.state_out, world_mask=None)

        for name, buf in self._cleared_buffers().items():
            values = buf.numpy()
            self.assertTrue(np.all(values == 0.0), f"{name} not cleared in all worlds")

    def test_reset_rejects_wrong_length_mask(self):
        """Reject masks without local-world entries plus the global slot."""
        self._setup_stepped_solver()
        mask = wp.array([True, False, True, False], dtype=wp.bool, device=self.model.device)
        with self.assertRaises(ValueError):
            self.solver.reset(self.state_out, world_mask=mask)

    def test_reset_recovers_from_nan_warmstart(self):
        """A NaN warm-start in one world is cleared so the next step stays finite."""
        self._setup_stepped_solver()
        warmstart = self.solver.mjw_data.qacc_warmstart
        if warmstart.shape[1] == 0:
            self.skipTest("model has no DOFs to warm-start")
        poisoned = warmstart.numpy()
        poisoned[0, :] = np.nan
        warmstart.assign(poisoned)

        mask = wp.array([True, False, False], dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state_out, world_mask=mask)
        self.assertTrue(np.all(np.isfinite(self.solver.mjw_data.qacc_warmstart.numpy())))

    def test_joint_q_reset_to_model_default_masked(self):
        """JOINT_Q resets joint_q to model defaults for masked worlds only."""
        self._setup_stepped_solver()
        defaults = self.model.joint_q.numpy()
        coords_per_world = self.model.joint_coord_count // self.model.world_count
        # Corrupt the live joint coordinates in both worlds.
        self.state_out.joint_q.assign(np.full_like(defaults, 9.0))

        mask = wp.array([True, False, False], dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state_out, world_mask=mask, flags=StateFlags.JOINT_Q)

        result = self.state_out.joint_q.numpy()
        np.testing.assert_allclose(result[:coords_per_world], defaults[:coords_per_world])
        np.testing.assert_allclose(result[coords_per_world:], np.full(coords_per_world, 9.0))

    def test_joint_qd_reset_to_model_default(self):
        """JOINT_QD resets joint_qd to model defaults across all worlds."""
        self._setup_stepped_solver()
        defaults = self.model.joint_qd.numpy()
        self.state_out.joint_qd.assign(np.full_like(defaults, 5.0))

        self.solver.reset(self.state_out, world_mask=None, flags=StateFlags.JOINT_QD)

        np.testing.assert_allclose(self.state_out.joint_qd.numpy(), defaults)

    def test_flags_zero_preserves_joint_state_but_clears_buffers(self):
        """flags=0 keeps the Newton state untouched while still clearing buffers."""
        self._setup_stepped_solver()
        self._poison()
        corrupted = np.full_like(self.model.joint_q.numpy(), 9.0)
        self.state_out.joint_q.assign(corrupted)

        self.solver.reset(self.state_out, world_mask=None, flags=0)

        # Joint state preserved...
        np.testing.assert_allclose(self.state_out.joint_q.numpy(), corrupted)
        # ...but internal buffers still cleared.
        for name, buf in self._cleared_buffers().items():
            self.assertTrue(np.all(buf.numpy() == 0.0), f"{name} not cleared with flags=0")

    def test_body_flags_are_ignored(self):
        """BODY_Q/BODY_QD do not touch joint state (body poses are FK-derived)."""
        self._setup_stepped_solver()
        corrupted = np.full_like(self.model.joint_q.numpy(), 9.0)
        self.state_out.joint_q.assign(corrupted)

        self.solver.reset(self.state_out, world_mask=None, flags=StateFlags.BODY_Q | StateFlags.BODY_QD)

        np.testing.assert_allclose(self.state_out.joint_q.numpy(), corrupted)

    def test_reset_defers_qpos_sync_at_default_interval(self):
        """At the default interval (1), reset leaves qpos for the next step to sync."""
        self._setup_stepped_solver()
        qpos = self.solver.mjw_data.qpos
        qpos.assign(np.full(qpos.shape, 3.0, dtype=np.float32))

        self.solver.reset(self.state_out, world_mask=None, flags=StateFlags.JOINT_Q)

        # reset does not touch qpos at interval == 1 (step() syncs it every step).
        np.testing.assert_allclose(self.solver.mjw_data.qpos.numpy(), 3.0)

    def test_reset_syncs_qpos_when_interval_not_default(self):
        """With update_data_interval != 1, reset pushes joint state into qpos now."""
        model = _build_two_world_model(world_count=2)
        # TODO: remove with update_data_interval.
        with self.assertWarns(NewtonDeprecationWarning):
            solver = SolverMuJoCo(model, iterations=2, ls_iterations=2, update_data_interval=0)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        # Capture the same conversion path reset uses, then distinguish worlds.
        solver._update_mjc_data(solver.mjw_data, model, state)
        qpos_synced = solver.mjw_data.qpos.numpy().copy()
        poisoned = qpos_synced + np.array(((10.0,), (20.0,)), dtype=np.float32)
        solver.mjw_data.qpos.assign(poisoned)

        # With the per-step sync disabled, reset must push only the selected
        # world's state into qpos now.
        solver.reset(
            state,
            world_mask=wp.array((True, False, False), dtype=wp.bool, device=model.device),
            flags=StateFlags.JOINT_Q,
        )

        result = solver.mjw_data.qpos.numpy()
        np.testing.assert_allclose(result[0], qpos_synced[0], atol=1e-6)
        np.testing.assert_allclose(result[1], poisoned[1], atol=1e-6)


class TestMuJoCoCPUModelChanged(unittest.TestCase):
    def test_native_world_observables_match_independent_simulations(self):
        """Map body, DOF, and contact observables to each CPU world's rows."""
        model = _build_two_world_model(device="cpu")
        single_model = _build_two_world_model(world_count=1, device="cpu")
        nq = single_model.joint_coord_count
        nv = single_model.joint_dof_count
        nb = single_model.body_count
        for integrator in ("euler", "rk4"):
            with self.subTest(integrator=integrator):
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, integrator=integrator)
                state_in, state_out = model.state(), model.state()
                q = state_in.joint_q.numpy()
                q[2] = q[nq + 2] = 0.05
                state_in.joint_q.assign(q)
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                control = model.control()
                force = np.zeros(model.joint_dof_count, dtype=np.float32)
                force[nv - 1], force[2 * nv - 1] = 1.0, -2.0
                control.joint_f.assign(force)
                contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=model.device)
                obs = solver.observables(solver.supported_observables, contacts=contacts)
                solver.step(state_in, state_out, control, contacts, 0.001, observables=obs)

                total_contacts = int(contacts.rigid_contact_count.numpy()[0])
                self.assertEqual(total_contacts, sum(d.ncon for d in solver.mj_data_by_world))
                shapes = np.concatenate(
                    (
                        contacts.rigid_contact_shape0.numpy()[:total_contacts],
                        contacts.rigid_contact_shape1.numpy()[:total_contacts],
                    )
                )
                self.assertEqual(set(model.shape_world.numpy()[shapes]), {-1, 0, 1})
                self.assertTrue(np.isfinite(obs.contact_f.numpy()[:total_contacts]).all())
                self.assertGreater(np.linalg.norm(obs.contact_f.numpy()[:total_contacts]), 0.0)

                for world in range(2):
                    reference = SolverMuJoCo(single_model, use_mujoco_cpu=True, integrator=integrator)
                    ref_in, ref_out = single_model.state(), single_model.state()
                    ref_in.joint_q.assign(q[world * nq : (world + 1) * nq])
                    newton.eval_fk(single_model, ref_in.joint_q, ref_in.joint_qd, ref_in)
                    ref_control = single_model.control()
                    ref_control.joint_f.assign(force[world * nv : (world + 1) * nv])
                    ref_contacts = newton.Contacts(reference.get_max_contact_count(), 0, device="cpu")
                    ref_obs = reference.observables(reference.supported_observables, contacts=ref_contacts)
                    reference.step(ref_in, ref_out, ref_control, ref_contacts, 0.001, observables=ref_obs)
                    for name, width in (
                        ("body_qdd", nb),
                        ("body_parent_f", nb),
                        ("qfrc_actuator", nv),
                        ("qfrc_bias", nv),
                        ("qfrc_constraint", nv),
                        ("qfrc_frictionloss", nv),
                    ):
                        np.testing.assert_allclose(
                            getattr(obs, name).numpy()[world * width : (world + 1) * width],
                            getattr(ref_obs, name).numpy(),
                            rtol=1e-5,
                            atol=1e-5,
                            err_msg=f"{name} in world {world}",
                        )
                    np.testing.assert_allclose(
                        state_out.joint_q.numpy()[world * nq : (world + 1) * nq],
                        ref_out.joint_q.numpy(),
                        atol=1e-6,
                    )

    def test_notify_preserves_native_world_state(self):
        """Preserve each CPU world's live state and refresh derived dynamics."""
        import mujoco

        for world_count in (1, 2):
            with self.subTest(world_count=world_count):
                model = _build_two_world_model(world_count=world_count, device="cpu")
                with self.assertWarns(NewtonDeprecationWarning):
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True, update_data_interval=0)
                saved = []
                for world, data in enumerate(solver.mj_data_by_world):
                    data.qpos[0] += 0.3 + world
                    data.qpos[-1] = 0.2 * (world + 1)
                    data.qvel[:] = 0.1 * (world + 1)
                    data.time = 0.25
                    mujoco.mj_forward(solver.mj_model, data)
                    saved.append((data.qpos.copy(), data.qvel.copy(), data.time))
                model.body_mass.assign(model.body_mass.numpy() * 1.5)
                solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                for data, (qpos, qvel, time) in zip(solver.mj_data_by_world, saved, strict=True):
                    np.testing.assert_array_equal(data.qpos, qpos)
                    np.testing.assert_array_equal(data.qvel, qvel)
                    self.assertEqual(data.time, time)
                    reference = mujoco.MjData(solver.mj_model)
                    reference.qpos[:] = qpos
                    reference.qvel[:] = qvel
                    mujoco.mj_forward(solver.mj_model, reference)
                    for name in ("xpos", "cvel", "qacc"):
                        np.testing.assert_allclose(getattr(data, name), getattr(reference, name), atol=1e-12)


class TestMuJoCoCPUWorldInputs(unittest.TestCase):
    @staticmethod
    def _scalar_model(*, equality=False, damping_ratio=None):
        """Build two driven scalar joints with optional equality constraints."""
        equality_xml = '<equality><weld body1="body" /></equality>' if equality else ""
        damping_xml = f'dampratio="{damping_ratio}"' if damping_ratio is not None else ""
        template = newton.ModelBuilder()
        template.add_mjcf(
            f"""
            <mujoco>
              <option gravity="0 0 0"/>
              <worldbody><body name="body">
                <joint name="joint" type="slide" axis="1 0 0"/>
                <geom type="sphere" size=".1" mass="1"/>
              </body></worldbody>
              <actuator><position joint="joint" kp="10" {damping_xml}/></actuator>
              {equality_xml}
            </mujoco>
        """,
            convert_mjc_equality_constraints=False,
        )
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        return builder.finalize(device="cpu")

    def test_scalar_targets_and_damping_share_native_reference(self):
        """Preserve independent targets and physical poses with a shared native reference."""
        model = self._scalar_model(damping_ratio=1.0)
        model.mujoco.dof_ref.assign([0.25, 0.25])
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        kd = model.joint_target_kd.numpy()
        self.assertGreater(kd[0], 0.0)
        np.testing.assert_array_equal(kd, [kd[0], kd[0]])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        control = model.control()
        control.joint_target_q.assign([0.1, -0.2])
        solver.step(model.state(), model.state(), control, None, 0.001)
        np.testing.assert_allclose([d.ctrl[0] for d in solver.mj_data_by_world], [0.35, 0.05], atol=1e-6)
        np.testing.assert_allclose([d.xpos[1, 0] for d in solver.mj_data_by_world], 0.0, atol=1e-6)

    def test_gravity_without_force_storage(self):
        """Apply per-world gravity when external force arrays are absent."""
        model = _build_two_world_model(device="cpu")
        gravity = model.gravity.numpy()
        gravity[0], gravity[1] = (0.0, 0.0, -1.0), (0.0, 0.0, -3.0)
        model.gravity.assign(gravity)
        for with_control in (False, True):
            with self.subTest(with_control=with_control):
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                state_in, state_out = model.state(), model.state()
                state_in.body_f = None
                control = model.control() if with_control else None
                if control is not None:
                    control.joint_f = None
                solver.step(state_in, state_out, control, None, 0.01)
                np.testing.assert_allclose([d.qvel[2] for d in solver.mj_data_by_world], [-0.01, -0.03], atol=1e-6)

    def test_reject_mismatched_cpu_inertial_properties(self):
        """Reject unequal shared inertial properties before publishing any native update."""
        for name, native_name in (
            ("body_mass", "body_mass"),
            ("body_inertia", "body_inertia"),
            ("body_com", "body_ipos"),
        ):
            for update in (False, True):
                with self.subTest(name=name, update=update):
                    model = _build_two_world_model(device="cpu")
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True) if update else None
                    attribute = getattr(model, name)
                    values = attribute.numpy()
                    values.reshape(2, -1)[1, 0] += 0.1
                    attribute.assign(values)
                    before = getattr(solver.mj_model, native_name).copy() if update else None
                    with self.assertRaisesRegex(ValueError, name):
                        if update:
                            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                        else:
                            SolverMuJoCo(model, use_mujoco_cpu=True)
                    if update:
                        np.testing.assert_array_equal(getattr(solver.mj_model, native_name), before)

    def test_native_equality_activation_is_world_local(self):
        """A shared equality model still allows independent activation at construction and notification."""
        model = self._scalar_model(equality=True)
        model.mujoco.equality_constraint_enabled.assign([False, True])
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        np.testing.assert_array_equal([d.eq_active[0] for d in solver.mj_data_by_world], [False, True])
        model.mujoco.equality_constraint_enabled.assign([True, False])
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
        np.testing.assert_array_equal([d.eq_active[0] for d in solver.mj_data_by_world], [True, False])

    def test_reject_mismatched_cpu_equality_parameters(self):
        """Reject unequal equality solver parameters without changing the native model."""
        for name in ("eq_solref", "eq_solimp"):
            for update in (False, True):
                with self.subTest(name=name, update=update):
                    model = self._scalar_model(equality=True)
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True) if update else None
                    attribute = getattr(model.mujoco, name)
                    values = attribute.numpy()
                    values[1, 0] += 0.01
                    attribute.assign(values)
                    before = getattr(solver.mj_model, name).copy() if update else None
                    with self.assertRaisesRegex(ValueError, name):
                        if update:
                            solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
                        else:
                            SolverMuJoCo(model, use_mujoco_cpu=True)
                    if update:
                        np.testing.assert_array_equal(getattr(solver.mj_model, name), before)

    def test_reject_mismatched_cpu_shared_parameters(self):
        """Validate every parameter family before a mixed notification publishes valid earlier flags."""
        flags = newton.ModelFlags
        cases = (
            ("joint_damping", flags.JOINT_DOF_FORCE_PROPERTIES),
            ("joint_friction", flags.JOINT_DOF_PROPERTIES),
            ("joint_target_ke", flags.JOINT_DOF_FORCE_PROPERTIES),
            ("joint_target_kd", flags.JOINT_DOF_FORCE_PROPERTIES),
            ("joint_armature", flags.JOINT_DOF_INERTIAL_PROPERTIES),
            ("joint_axis", flags.JOINT_PROPERTIES),
            ("joint_X_p", flags.JOINT_PROPERTIES),
            ("joint_X_c", flags.JOINT_PROPERTIES),
            ("shape_material_mu", flags.SHAPE_PROPERTIES),
            ("shape_material_mu_torsional", flags.SHAPE_PROPERTIES),
            ("shape_material_mu_rolling", flags.SHAPE_PROPERTIES),
            ("shape_scale", flags.SHAPE_PROPERTIES),
            ("shape_transform", flags.SHAPE_PROPERTIES),
            ("mujoco.gravcomp", flags.BODY_INERTIAL_PROPERTIES),
            ("mujoco.dof_ref", flags.JOINT_REFERENCE_POSE_PROPERTIES),
            ("mujoco.dof_springref", flags.JOINT_DOF_PROPERTIES),
            ("mujoco.dof_passive_stiffness", flags.JOINT_DOF_FORCE_PROPERTIES),
            ("mujoco.actuator_gear", flags.ACTUATOR_PROPERTIES),
        )
        for name, flag in cases:
            with self.subTest(name=name):
                model = self._scalar_model()
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                source, field = (model.mujoco, name[7:]) if name.startswith("mujoco.") else (model, name)
                attribute = getattr(source, field)
                values = attribute.numpy()
                values.reshape(2, -1)[1, 0] += 0.1
                attribute.assign(values)
                # A valid inertia edit must not leak through a later invalid flag.
                before = solver.mj_model.body_mass.copy()
                model.body_mass.assign(model.body_mass.numpy() * 2)
                with self.assertRaisesRegex(ValueError, field):
                    solver.notify_model_changed(flag | flags.BODY_INERTIAL_PROPERTIES)
                np.testing.assert_array_equal(solver.mj_model.body_mass, before)
                with self.assertRaisesRegex(ValueError, field):
                    SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)

    def test_reject_mismatched_cpu_equality_data(self):
        """Joint/tendon polynomials and imported mimic coefficients share native eq_data."""
        xml = """
        <mujoco>
          <worldbody>
            <body name="a"><joint name="a" type="slide"/><geom size=".1" mass="1"/>
              <body name="b" pos="1 0 0"><joint name="b" type="slide"/><geom size=".1" mass="1"/></body>
            </body>
          </worldbody>
          <tendon>
            <fixed name="a"><joint joint="a" coef="1"/></fixed>
            <fixed name="b"><joint joint="b" coef="1"/></fixed>
          </tendon>
          <equality>{constraint}</equality>
        </mujoco>
        """
        for kind, constraint, convert in (
            ("joint", '<joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/>', False),
            ("tendon", '<tendon tendon1="a" tendon2="b" polycoef="0 1 0 0 0"/>', False),
            ("mimic", '<joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/>', True),
        ):
            with self.subTest(kind=kind):
                template = newton.ModelBuilder()
                template.add_mjcf(
                    xml.format(constraint="" if convert else constraint), convert_mjc_equality_constraints=False
                )
                if convert:
                    template.set_joint_mimic(0, 1, coeffs=(0.0, 1.0))
                builder = newton.ModelBuilder()
                builder.replicate(template, 2)
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                name = "joint_mimic_coeffs" if convert else "equality_constraint_polycoef"
                attribute = getattr(model if convert else model.mujoco, name)
                values = attribute.numpy()
                followers = np.flatnonzero(model.joint_mimic_joint.numpy() >= 0)
                row = int(followers[1]) if convert else 1
                values[row, 0] += 0.2
                attribute.assign(values)
                before = solver.mj_model.eq_data.copy()
                with self.assertRaisesRegex(ValueError, name):
                    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
                np.testing.assert_array_equal(solver.mj_model.eq_data, before)
                with self.assertRaisesRegex(ValueError, name):
                    SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)

    def test_cpu_constraint_reference_geometry(self):
        """Common world translations preserve anchors; differing relative reference geometry cannot."""
        for kind in ("connect", "weld"):
            with self.subTest(kind=kind):
                anchor = 'anchor="0 0 0"' if kind == "connect" else ""
                template = newton.ModelBuilder()
                template.add_mjcf(
                    f"""
                <mujoco>
                  <worldbody>
                    <body name="a"><freejoint/><geom size=".1" mass="1"/></body>
                    <body name="b" pos="1 0 0"><freejoint/><geom size=".1" mass="1"/></body>
                  </worldbody>
                  <equality><{kind} body1="a" body2="b" {anchor}/></equality>
                </mujoco>
                """,
                    convert_mjc_equality_constraints=False,
                )
                builder = newton.ModelBuilder()
                builder.replicate(template, 2, spacing=(2.0, 0.0, 0.0))
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                # Reference geometry is body-local, so translating both endpoints together is supported.
                np.testing.assert_allclose(
                    solver.mjw_model.eq_data.numpy()[0], solver.mjw_model.eq_data.numpy()[1], atol=1e-6
                )
                q = model.joint_q.numpy()
                q[model.joint_coord_count // 2 + 7] += 0.25
                model.joint_q.assign(q)
                body_q = model.body_q.numpy()
                body_q[3, 0] += 0.25
                model.body_q.assign(body_q)
                before = solver.mj_model.eq_data.copy()
                flags = newton.ModelFlags.JOINT_PROPERTIES | newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES
                with self.assertRaisesRegex(ValueError, "reference geometry"):
                    solver.notify_model_changed(flags)
                np.testing.assert_array_equal(solver.mj_model.eq_data, before)
                with self.assertRaisesRegex(ValueError, "reference geometry"):
                    SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)

    def test_cpu_solver_options_require_shared_values(self):
        """Native options are shared unless a constructor override replaces every world."""
        model = self._scalar_model()
        for name, override in (("impratio", 2.0), ("wind", (0.0, 0.0, 0.0))):
            with self.subTest(name=name):
                attribute = getattr(model.mujoco, name)
                original = attribute.numpy().copy()
                values = original.copy()
                values[1] += 0.1
                attribute.assign(values)
                with self.assertRaisesRegex(ValueError, name):
                    SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True, **{name: override})
                np.testing.assert_allclose(getattr(solver.mj_model.opt, name), override)
                values[: model.world_count] = override
                attribute.assign(values)
                shared = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                np.testing.assert_allclose(getattr(shared.mj_model.opt, name), override)
                attribute.assign(original)

    def test_reject_global_cpu_actuators(self):
        """A global actuator cannot share a rectangular world-local control stride."""
        template = newton.ModelBuilder()
        template.add_mjcf("""
        <mujoco>
          <worldbody><body><joint name="joint" type="slide"/><geom size=".1" mass="1"/></body></worldbody>
          <actuator><motor joint="joint"/></actuator>
        </mujoco>
        """)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        global_actuator = {
            key: attribute.values[0]
            for key, attribute in builder.custom_attributes.items()
            if attribute.frequency == "mujoco:actuator" and attribute.values
        }
        global_actuator["mujoco:actuator_world"] = -1
        builder.add_custom_values(**global_actuator)
        model = builder.finalize(device="cpu")
        self.assertEqual(model.control().mujoco.ctrl.shape[0], 3)
        with self.assertRaisesRegex(ValueError, "global actuators"):
            SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)

    def test_default_contact_capacity_covers_every_world(self):
        """The later world's contacts exceed the backend's 48-contact default floor."""
        template = newton.ModelBuilder()
        body = template.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()))
        joint = template.add_joint_free(child=body)
        template.add_articulation([joint])
        for i in range(13):
            template.add_shape_box(
                body, xform=wp.transform((i * 0.3, 0, 0), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1
            )
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(template, 2)
        model = builder.finalize(device="cpu")
        q = model.joint_q.numpy()
        q[model.joint_coord_count // 2 + 2] = 0.05
        model.joint_q.assign(q)
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        counts = [data.ncon for data in solver.mj_data_by_world]
        self.assertEqual(counts[0], 0)
        self.assertGreater(counts[1], 48)
        self.assertGreaterEqual(solver.get_max_contact_count() // 2, max(counts))
        self.assertGreaterEqual(solver.mjw_data.njmax, max(data.nefc for data in solver.mj_data_by_world))
        contacts = newton.Contacts(solver.get_max_contact_count(), 0, device="cpu")
        solver.update_contacts(contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), sum(counts))

    def test_initial_contact_capacity_covers_every_world(self):
        """Raise the native contact quota for an initially contacting later world."""
        model = _build_two_world_model(device="cpu")
        q = model.joint_q.numpy()
        q[model.joint_coord_count // 2 + 2] = 0.05
        model.joint_q.assign(q)
        with self.assertWarnsRegex(newton.exceptions.NewtonWarning, "nconmax"):
            solver = SolverMuJoCo(model, use_mujoco_cpu=True, nconmax=1)
        counts = [data.ncon for data in solver.mj_data_by_world]
        self.assertEqual(counts[0], 0)
        self.assertGreater(counts[1], 1)
        self.assertGreaterEqual(solver.get_max_contact_count() // 2, max(counts))
        contacts = newton.Contacts(solver.get_max_contact_count(), 0, device="cpu")
        solver.update_contacts(contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), sum(counts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
