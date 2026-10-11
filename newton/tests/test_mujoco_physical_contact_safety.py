# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check rejection, recovery, and checked replay of physical contact errors.

The first three regressions also run as a standalone file against the unchanged
Phase 2 packages; their failures must be behavioral, not missing-API failures.
"""

import unittest

import numpy as np
import warp as wp

from newton.tests.test_mujoco_contact_partition import _fill_contacts, _make_fixture
from newton.tests.test_mujoco_physical_contact import _fixture, _publish


class TestMuJoCoPhysicalContactSafety(unittest.TestCase):
    def test_eager_invalid_step_rejects_before_publishing_output(self):
        """Reject invalid active coefficients before publishing a Newton state."""
        coefficients = ((-1.0, 0.0), (0.0, 1.0), (1.0, -1.0), (np.nan, 0.0), (1.0, np.inf))
        for stiffness, damping in coefficients:
            with self.subTest(stiffness=stiffness, damping=damping):
                model, solver, contacts, state_in, state_out, shapes = _fixture(4)
                _publish(contacts, shapes, [1.0], stiffness=stiffness, damping=damping, gap=-0.001)
                state_out.joint_qd.fill_(17.0)
                state_out.body_qd.fill_(19.0)
                before_joint = state_out.joint_qd.numpy().copy()
                before_body = state_out.body_qd.numpy().copy()
                caught = None
                try:
                    solver.step(state_in, state_out, model.control(), contacts, 0.001)
                except ValueError as error:
                    caught = error
                self.assertIsNotNone(
                    caught,
                    "Invalid physical step returned normally; "
                    f"published body velocity is finite={np.isfinite(state_out.body_qd.numpy()).all()}",
                )
                self.assertRegex(str(caught), "physical contact")
                np.testing.assert_array_equal(state_out.joint_qd.numpy(), before_joint)
                np.testing.assert_array_equal(state_out.body_qd.numpy(), before_body)
                self.assertNotEqual(int(solver.mjw_data.contact.force_error.numpy()[0]), 0)

    def test_newton_reset_clears_error_and_allows_corrected_contact(self):
        """Recover from an actual invalid step through Newton's public reset."""
        model, solver, contacts, state_in, state_out, shapes = _fixture(4)
        _publish(contacts, shapes, [1.0], stiffness=-1.0, gap=-0.001)
        # Phase 2 returns normally; the hardened implementation raises. Both
        # paths must leave a real device error before the reset is exercised.
        try:
            solver.step(state_in, state_out, model.control(), contacts, 0.001)
        except ValueError:
            pass
        self.assertNotEqual(int(solver.mjw_data.contact.force_error.numpy()[0]), 0)
        solver.reset(state_in)
        np.testing.assert_array_equal(solver.mjw_data.contact.force_error.numpy(), 0)
        np.testing.assert_array_equal(solver.mjw_data.contact.force_params.numpy(), 0.0)
        _publish(contacts, shapes, [0.25] * 4, stiffness=10000.0, damping=20.0, gap=-0.001)
        solver.step(state_in, state_out, model.control(), contacts, 0.001)
        solver.check_hydroelastic_force_response()
        # Independent scalar material law and momentum balance, m=1 and v=0.
        expected = 0.001 * 10000.0 * 0.001 / (1.0 + 0.001 * 20.0 + 0.001**2 * 10000.0)
        np.testing.assert_allclose(state_out.body_qd.numpy()[0, 2], expected, rtol=2.0e-5, atol=2.0e-7)

    def test_empty_physical_step_rejects_invalid_timestep(self):
        """Validate the timestep even when no physical constraint row exists."""
        for step in (0.0, -0.001, np.nan, np.inf, -np.inf):
            with self.subTest(step=step):
                model, solver, contacts, state_in, state_out, shapes = _fixture(4)
                _publish(contacts, shapes, [], stiffness=1.0)
                state_out.body_qd.fill_(19.0)
                before = state_out.body_qd.numpy().copy()
                with self.assertRaisesRegex(ValueError, "timestep|time step|dt"):
                    solver.step(state_in, state_out, model.control(), contacts, step)
                np.testing.assert_array_equal(state_out.body_qd.numpy(), before)

    def test_corrected_values_do_not_erase_sticky_error(self):
        """Reject later steps until reset even after invalid contacts disappear."""
        model, solver, contacts, state_in, state_out, shapes = _fixture(4)
        _publish(contacts, shapes, [1.0], stiffness=-1.0, gap=-0.001)
        with self.assertRaisesRegex(ValueError, "physical contact"):
            solver.step(state_in, state_out, model.control(), contacts, 0.001)
        for weights in ([], [1.0]):
            with self.subTest(weights=weights):
                _publish(contacts, shapes, weights, stiffness=10000.0, gap=-0.001)
                with self.assertRaisesRegex(ValueError, "physical contact"):
                    solver.step(state_in, state_out, model.control(), contacts, 0.001)
        solver.reset(state_in)
        solver.step(state_in, state_out, model.control(), contacts, 0.001)
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_partial_reset_preserves_other_world_errors_and_parameters(self):
        """Clear selected physical worlds without clearing another world's failure."""
        model, solver, contacts, state_in, state_out, shapes = _make_fixture(worlds=2)
        _fill_contacts(contacts, shapes, [1.0, 1.0], stiffness=-10.0)
        shape0 = contacts.rigid_contact_shape0.numpy()
        shape1 = contacts.rigid_contact_shape1.numpy()
        shape0[1], shape1[1] = shapes[0] + 2, shapes[1] + 2
        contacts.rigid_contact_shape0.assign(shape0)
        contacts.rigid_contact_shape1.assign(shape1)
        with self.assertRaisesRegex(ValueError, "physical contact"):
            solver.step(state_in, state_out, model.control(), contacts, 0.001)
        data = solver.mjw_data
        np.testing.assert_array_equal(data.contact.force_error.numpy(), [1, 1])
        count = int(data.nacon.numpy()[0])
        worlds = data.contact.worldid.numpy()[:count]
        before = data.contact.force_params.numpy()[:count].copy()
        self.assertEqual(set(worlds.tolist()), {0, 1})
        solver.reset(state_in, world_mask=wp.array([True, False, False], dtype=bool, device=model.device))
        np.testing.assert_array_equal(data.contact.force_error.numpy(), [0, 1])
        np.testing.assert_array_equal(data.contact.force_params.numpy()[:count][worlds == 0], 0.0)
        np.testing.assert_array_equal(data.contact.force_params.numpy()[:count][worlds == 1], before[worlds == 1])
        with self.assertRaisesRegex(ValueError, "worlds.*1"):
            solver.check_hydroelastic_force_response()
        solver.reset(state_in, world_mask=wp.array([False, True, False], dtype=bool, device=model.device))
        solver.check_hydroelastic_force_response()
        np.testing.assert_array_equal(data.contact.force_params.numpy()[:count], 0.0)

        # Full reset must also clear allocated tail slots outside nacon.
        data.contact.force_params.fill_(wp.vec2(7.0, 3.0))
        data.contact.force_error.assign(np.array([1, 2], dtype=np.int32))
        solver.reset(state_in)
        np.testing.assert_array_equal(data.contact.force_params.numpy(), 0.0)
        np.testing.assert_array_equal(data.contact.force_error.numpy(), 0)

    def test_backend_forward_and_split_step_reject_errors(self):
        """Enforce the physical error contract on backend forward and split steps."""
        import mujoco_warp as mjw

        model, solver, contacts, state_in, state_out, shapes = _fixture(4)
        _publish(contacts, shapes, [1.0], stiffness=10000.0, gap=-0.001)
        solver.step(state_in, state_out, model.control(), contacts, 0.001)
        backend, data = solver.mjw_model, solver.mjw_data
        for entry in (mjw.forward, mjw.step1):
            with self.subTest(entry=entry.__name__):
                solver.reset(state_in)
                data.contact.force_params.fill_(wp.vec2(-1.0, 0.0))
                before = data.qpos.numpy().copy()
                with self.assertRaisesRegex(ValueError, "physical contact"):
                    entry(backend, data)
                np.testing.assert_array_equal(data.qpos.numpy(), before)

        solver.reset(state_in)
        data.contact.force_params.fill_(wp.vec2(-1.0, 0.0))
        mjw.make_constraint(backend, data)
        self.assertNotEqual(int(data.contact.force_error.numpy()[0]), 0)
        before = data.qpos.numpy().copy()
        with self.assertRaisesRegex(ValueError, "physical contact"):
            mjw.step2(backend, data)
        np.testing.assert_array_equal(data.qpos.numpy(), before)

    def test_public_backend_reset_validates_mask(self):
        """Reject malformed reset masks without clearing a sticky failure."""
        import mujoco_warp as mjw

        _, solver, _, _, _, _ = _fixture(4)
        data = solver.mjw_data
        data.contact.force_error.fill_(1)
        for mask in (
            wp.array([True, False], dtype=bool, device="cpu"),
            wp.array([1.0], dtype=float, device="cpu"),
            wp.array([1], dtype=int, device="cpu"),
        ):
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaises((ValueError, TypeError)):
                mjw.reset_contact_force_params(data, mask)
        np.testing.assert_array_equal(data.contact.force_error.numpy(), [1])
        mjw.reset_contact_force_params(data, wp.array([True], dtype=bool, device="cpu"))
        np.testing.assert_array_equal(data.contact.force_error.numpy(), 0)

    def test_valid_zero_tuple_and_default_off_remain_usable(self):
        """Preserve valid zero-sentinel and ordinary default-off stepping."""
        outputs = []
        for enabled in (False, True):
            model, solver, contacts, state_in, state_out, shapes = _fixture(4, enabled=enabled)
            _publish(contacts, shapes, [1.0], stiffness=0.0, damping=0.0, gap=-0.001)
            solver.step(state_in, state_out, model.control(), contacts, 0.001)
            solver.check_hydroelastic_force_response()
            outputs.append(state_out.body_qd.numpy())
        np.testing.assert_array_equal(*outputs)

    def test_cuda_checked_replay_rejects_invalid_then_recovers(self):
        """Reject changed device coefficients during checked replay and recover after reset."""
        import mujoco_warp as mjw

        if not wp.get_cuda_device_count():
            self.skipTest("CUDA is unavailable")
        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the mempool allocator")
        model, solver, contacts, state_in, state_out, shapes = _fixture(4, device=device)
        control = model.control()
        _publish(contacts, shapes, [0.25] * 4, stiffness=10000.0, damping=20.0, gap=-0.001)
        solver.step(state_in, state_out, control, contacts, 0.001)
        expected = state_out.body_qd.numpy().copy()
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_in, state_out, control, contacts, 0.001)
        solver.launch_hydroelastic_force_response_graph(capture.graph)
        np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=2.0e-5, atol=2.0e-7)

        _publish(contacts, shapes, [1.0], stiffness=-1.0, gap=-0.001)
        with self.assertRaisesRegex(ValueError, "physical contact"):
            solver.launch_hydroelastic_force_response_graph(capture.graph)
        self.assertNotEqual(int(solver.mjw_data.contact.force_error.numpy()[0]), 0)
        _publish(contacts, shapes, [0.25] * 4, stiffness=10000.0, damping=20.0, gap=-0.001)
        with self.assertRaisesRegex(ValueError, "physical contact"):
            solver.launch_hydroelastic_force_response_graph(capture.graph)
        solver.reset(state_in)
        solver.launch_hydroelastic_force_response_graph(capture.graph)
        np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=2.0e-5, atol=2.0e-7)

        # A custom stream must finish its error writes before host validation.
        stream = wp.Stream(device)
        mjw.launch_contact_force_graph(solver.mjw_data, capture.graph, stream=stream)
        np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=2.0e-5, atol=2.0e-7)
        _publish(contacts, shapes, [1.0], stiffness=np.nan, gap=-0.001)
        with self.assertRaisesRegex(ValueError, "physical contact"):
            mjw.launch_contact_force_graph(solver.mjw_data, capture.graph, stream=stream)

    def test_cuda_checked_replay_rejects_calls_during_capture(self):
        """Require checked replay and host error readback to run outside capture."""
        if not wp.get_cuda_device_count():
            self.skipTest("CUDA is unavailable")
        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the mempool allocator")
        model, solver, contacts, state_in, state_out, shapes = _fixture(4, device=device)
        control = model.control()
        _publish(contacts, shapes, [1.0], stiffness=10000.0, gap=-0.001)
        solver.step(state_in, state_out, control, contacts, 0.001)
        with wp.ScopedCapture(device=device) as first:
            solver.step(state_in, state_out, control, contacts, 0.001)
        with wp.ScopedCapture(device=device):
            solver.step(state_in, state_out, control, contacts, 0.001)
            with self.assertRaisesRegex((ValueError, RuntimeError), "capture"):
                solver.launch_hydroelastic_force_response_graph(first.graph)


if __name__ == "__main__":
    unittest.main()
