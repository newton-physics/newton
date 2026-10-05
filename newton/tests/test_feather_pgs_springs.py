# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Passive joint damping, and the unsupported passive joint springs, of SolverFeatherPGS.

FeatherPGS does not apply passive joint springs yet (newton-physics/newton#4516); it warns
when a model carries nonzero MuJoCo spring stiffness.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0
SPRING_KEYS = ("mujoco:dof_passive_stiffness", "mujoco:dof_springref", "mujoco:dof_ref")
_SPRING_WARNING = r"does not support passive joint springs yet .*#4516"


def _hinge_model(device, *, damping: float, spring_k: float | None = None, spring_ref: float = 0.5):
    """A fixed-base link on a revolute Z joint, so gravity exerts no torque about the axis.

    ``spring_k`` registers the MuJoCo spring attributes and sets the joint's stiffness.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
    custom = {}
    if spring_k is not None:
        SolverMuJoCo.register_custom_attributes(builder)
        custom = {"mujoco:dof_passive_stiffness": spring_k, "mujoco:dof_springref": spring_ref}
    link = builder.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        damping=damping,
        custom_attributes=custom,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _run(model, solver, steps: int, qd0: float = 0.0):
    state_0, state_1 = model.state(), model.state()
    if qd0 != 0.0:
        qd = state_0.joint_qd.numpy()
        qd[0] = qd0
        state_0.joint_qd.assign(qd)
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    return state_0


def _construct(test, model, *, pgs_mode, expect_warning):
    """Construct the solver, requiring exactly the spring warning or no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, pgs_iterations=8)
    messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    test.assertEqual(len(caught), int(expect_warning), messages)
    if expect_warning:
        test.assertIs(caught[0].category, UserWarning)
        test.assertRegex(str(caught[0].message), _SPRING_WARNING)
    return solver


def test_passive_damping_decays_velocity(test, device, pgs_mode="matrix_free"):
    """Decay joint velocity through passive damping; without damping the joint keeps coasting."""
    damped_model = _hinge_model(device, damping=0.02)
    undamped_model = _hinge_model(device, damping=0.0)
    damped = _run(damped_model, _construct(test, damped_model, pgs_mode=pgs_mode, expect_warning=False), 480, qd0=5.0)
    undamped = _run(
        undamped_model, _construct(test, undamped_model, pgs_mode=pgs_mode, expect_warning=False), 480, qd0=5.0
    )
    qd_damped = abs(float(damped.joint_qd.numpy()[0]))
    qd_undamped = abs(float(undamped.joint_qd.numpy()[0]))
    test.assertGreater(qd_undamped, 4.0, "undamped joint should keep coasting")
    test.assertLess(qd_damped, 0.2 * qd_undamped)


def test_nonzero_spring_stiffness_warns_and_is_ignored(test, device, pgs_mode="matrix_free"):
    """Warn once at construction for nonzero MuJoCo spring stiffness and apply no spring torque."""
    model = _hinge_model(device, damping=0.0, spring_k=4.0, spring_ref=0.5)
    solver = _construct(test, model, pgs_mode=pgs_mode, expect_warning=True)
    np.testing.assert_array_equal(solver._passive_spring_stiffness.numpy(), [0.0])
    state = _run(model, solver, 240)
    np.testing.assert_allclose(state.joint_q.numpy(), [0.0], atol=1.0e-6)
    test.assertEqual(float(solver.joint_tau.numpy()[0]), 0.0)


def test_zero_or_absent_spring_stiffness_does_not_warn(test, device, pgs_mode="matrix_free"):
    """Construct silently without the spring attributes, and with zero registered stiffness."""
    _construct(test, _hinge_model(device, damping=0.1), pgs_mode=pgs_mode, expect_warning=False)
    _construct(test, _hinge_model(device, damping=0.1, spring_k=0.0), pgs_mode=pgs_mode, expect_warning=False)


def test_spring_attribute_edits_leave_damping_refresh_intact(test, device, pgs_mode="matrix_free"):
    """A JOINT_DOF_PROPERTIES notify refreshes damping and still applies no spring torque."""
    model = _hinge_model(device, damping=0.0, spring_k=0.0, spring_ref=0.5)
    solver = _construct(test, model, pgs_mode=pgs_mode, expect_warning=False)
    model.mujoco.dof_passive_stiffness.assign([4.0])
    model.joint_damping.assign([3.0])
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    state_0, state_1 = model.state(), model.state()
    state_0.joint_qd.fill_(1.0)
    solver.step(state_0, state_1, model.control(), None, DT)
    np.testing.assert_allclose(solver.joint_tau.numpy(), [-3.0], atol=1.0e-6)


def test_registration_adds_no_spring_attributes(test, device):
    """Register only FeatherPGS's own body attributes, not the MuJoCo spring attributes."""
    builder = newton.ModelBuilder()
    SolverFeatherPGS.register_custom_attributes(builder)
    test.assertFalse(any(key in builder.custom_attributes for key in SPRING_KEYS))


class TestFeatherPGSSprings(unittest.TestCase):
    pass


for _name, _func in (
    ("test_passive_damping_decays_velocity", test_passive_damping_decays_velocity),
    ("test_nonzero_spring_stiffness_warns_and_is_ignored", test_nonzero_spring_stiffness_warns_and_is_ignored),
    ("test_zero_or_absent_spring_stiffness_does_not_warn", test_zero_or_absent_spring_stiffness_does_not_warn),
    (
        "test_spring_attribute_edits_leave_damping_refresh_intact",
        test_spring_attribute_edits_leave_damping_refresh_intact,
    ),
):
    add_function_test(TestFeatherPGSSprings, _name, _func, devices=get_cuda_test_devices())
    add_function_test(TestFeatherPGSSprings, f"{_name}_split", _func, devices=get_test_devices(), pgs_mode="split")
add_function_test(
    TestFeatherPGSSprings,
    "test_registration_adds_no_spring_attributes",
    test_registration_adds_no_spring_attributes,
    devices=None,
)


if __name__ == "__main__":
    unittest.main()
