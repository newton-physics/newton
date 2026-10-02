# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise step/reset/release guards for experimental contact compliance."""

import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.contact_compliance import _prepare_compliant_rows, start_step
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_compliance import run_fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _contacts(device, *, count=0):
    """Create bounded material buffers for guard-only tests."""
    return SimpleNamespace(
        rigid_contact_count=wp.array([count], dtype=int, device=device),
        rigid_contact_max=4,
        rigid_contact_stiffness=wp.zeros(4, device=device),
        rigid_contact_damping=wp.zeros(4, device=device),
        rigid_contact_friction=wp.zeros(4, device=device),
    )


def test_input_overflow_rejected(test, device):
    """Fail explicitly for lost contact candidates."""
    solver = SimpleNamespace(model=SimpleNamespace(device=device))
    with test.assertRaisesRegex(RuntimeError, "overflowing contact"):
        start_step(solver, _contacts(device, count=5), 0.005)


def test_capture_rejected(test, device):
    """Prevent a silent fallback under CUDA graph capture."""
    solver = SimpleNamespace(model=SimpleNamespace(device=device))
    contacts = _contacts(device)
    with wp.ScopedCapture(device=device):
        with test.assertRaisesRegex(RuntimeError, "CUDA graph"):
            start_step(solver, contacts, 0.005)


def test_solver_step_rejects_capture(test, device):
    """Reject capturing a real compliant solver step before it launches any work."""
    _, _, solver, _ = run_fixture(device=device, articulated=True, enabled=True, steps=1)
    model = solver.model
    state, output = model.state(), model.state()
    with wp.ScopedCapture(device=device):
        with test.assertRaisesRegex(RuntimeError, "CUDA graph"):
            solver.step(state, output, model.control(), None, 0.005)


def test_nonfinite_inputs_rejected(test, device):
    """Reject invalid time steps instead of emitting infinite row coefficients."""
    solver = SimpleNamespace(model=SimpleNamespace(device=device))
    contacts = _contacts(device)
    for dt in (0, -1, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "dt"):
            start_step(solver, contacts, dt)


def test_persistent_patch_guard(test, device):
    """Reject positive persistent-anchor bias but permit a disabled dummy patch buffer."""
    solver = SimpleNamespace(
        model=SimpleNamespace(device=device),
        friction_anchor_beta=0.0,
        _friction_patches=object(),
    )
    start_step(solver, _contacts(device), 0.005)
    solver.friction_anchor_beta = 0.2
    with test.assertRaisesRegex(ValueError, "friction_anchor_beta"):
        start_step(solver, _contacts(device), 0.005)
    solver.contact_compliance = True
    with test.assertRaisesRegex(ValueError, "friction_anchor_beta"):
        SolverFeatherPGS.step(solver, None, None, None, None, 0.005)


def test_filtered_speculative_contacts_are_skipped(test, device):
    """Honor the normal allocator's positive-gap exclusion for compliant contacts."""
    for articulated in (False, True):
        _, paths, solver, _ = run_fixture(
            device=device,
            articulated=articulated,
            enabled=True,
            steps=1,
            height=0.052,
            solver_options={"contact_gap_gate": 0.001},
        )
        test.assertEqual(paths, {-1})
        test.assertEqual(solver.compliance_contact_count, 0)
        test.assertEqual(solver.compliance_skipped_contact_count, 1)


def test_nonresponding_contact_is_skipped(test, device):
    """Ignore a kinematic-ground pair without treating it as lost capacity."""
    _, paths, solver, _ = run_fixture(device=device, articulated=True, enabled=True, steps=1, kinematic=True)
    test.assertEqual(paths, {-1})
    test.assertEqual(solver.compliance_contact_count, 0)
    test.assertEqual(solver.compliance_skipped_contact_count, 1)


def test_rolled_back_capacity_fails_without_warnings(test, device):
    """Reject an actual failed reservation even when its counter was rolled back."""
    for articulated in (False, True):
        with test.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
            run_fixture(
                device=device,
                articulated=articulated,
                enabled=True,
                steps=1,
                solver_options={
                    "dense_max_constraints": 1,
                    "mf_max_constraints": 1,
                    "warn_constraint_overflow": False,
                },
            )


def test_solver_overflow_and_dropped_contact_rejected(test, device):
    """Reject overflowing rows and missing mappings for positive stiffness."""
    _, _, solver, _ = run_fixture(device=device, articulated=True, enabled=True, steps=1)
    solver.constraint_count.fill_(solver.dense_max_constraints + 1)
    with test.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
        _prepare_compliant_rows(solver)
    solver.constraint_count.fill_(1)
    solver.contact_slot.fill_(-1)
    with test.assertRaisesRegex(RuntimeError, "dropped"):
        _prepare_compliant_rows(solver)


def test_reset_and_open_gap_release(test, device):
    """Clear only step-local compliance data and exert no force across an open gap."""
    _, _, solver, _ = run_fixture(device=device, articulated=True, enabled=True, steps=5)
    model = solver.model
    state = model.state()
    state.joint_q.assign(np.array([0.01], dtype=np.float32))
    state.joint_qd.assign(np.array([0.2], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    before = {name: getattr(state, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    solver.reset(state)
    test.assertIsNone(solver._compliant_contacts)
    test.assertEqual(solver.compliance_contact_count, 0)
    test.assertEqual(solver.compliance_skipped_contact_count, 0)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name).numpy(), value)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    contacts = pipeline.contacts()
    for field in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
        setattr(contacts, field, wp.zeros(32, device=device))
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    output = model.state()
    solver.step(state, output, model.control(), contacts, 0.005)
    test.assertEqual(solver.compliance_contact_count, 0)
    test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 0.2 - 9.81 * 0.005, delta=1e-6)
    test.assertTrue(np.isfinite(output.body_q.numpy()).all())


def test_actual_timestep_and_iterations(test, device):
    """Keep native row stiffness invariant under smaller timesteps and more iterations."""
    for dt, iterations in ((0.005, 1), (0.0025, 16)):
        trace, _, _, mass = run_fixture(
            device=device, articulated=True, enabled=True, dt=dt, iterations=iterations, steps=round(1 / dt)
        )
        test.assertAlmostEqual(float(trace[-1, 2]), 0.05 - mass * 9.81 / 3000, delta=2e-6)


def test_public_defaults_remain_hard(test, device):
    """Expose an explicit default-off public constructor option."""
    parameter = inspect.signature(SolverFeatherPGS.__init__).parameters["contact_compliance"]
    test.assertIs(parameter.default, False)
    test.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)


class TestContactComplianceSafety(unittest.TestCase):
    """Reject invalid routing and preserve the physical reset and release behavior."""


devices = get_cuda_test_devices()
for _fn in (
    test_input_overflow_rejected,
    test_capture_rejected,
    test_solver_step_rejects_capture,
    test_nonfinite_inputs_rejected,
    test_persistent_patch_guard,
    test_filtered_speculative_contacts_are_skipped,
    test_nonresponding_contact_is_skipped,
    test_rolled_back_capacity_fails_without_warnings,
    test_solver_overflow_and_dropped_contact_rejected,
    test_reset_and_open_gap_release,
    test_actual_timestep_and_iterations,
    test_public_defaults_remain_hard,
):
    add_function_test(TestContactComplianceSafety, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
