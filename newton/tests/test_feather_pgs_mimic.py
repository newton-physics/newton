# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mimic rows of SolverFeatherPGS."""

import inspect
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_MIMIC
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0


def _solver_accepts(option: str) -> bool:
    return option in inspect.signature(SolverFeatherPGS).parameters


def _build_mimic_layout(joint_type):
    """Build a joint-owned mimic pair after a free base in one articulation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_free(child=base)]
    parent = base
    for _ in range(2):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        linear = []
        angular = []
        if joint_type == newton.JointType.PRISMATIC:
            linear = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
        elif joint_type == newton.JointType.REVOLUTE:
            angular = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)]
        elif joint_type == newton.JointType.D6:
            linear = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
            angular = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
        if joint_type == newton.JointType.BALL:
            joint = builder.add_joint_ball(parent=parent, child=child)
        elif joint_type == newton.JointType.FREE:
            joint = builder.add_joint_free(parent=parent, child=child)
        elif joint_type == newton.JointType.DISTANCE:
            joint = builder.add_joint_distance(parent=parent, child=child)
        else:
            joint = builder.add_joint(joint_type, parent=parent, child=child, linear_axes=linear, angular_axes=angular)
        joints.append(joint)
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.1, -0.5))
    return builder, joints[1], joints[2]


def _build_two_revolute_chain(coef0: float, coef1: float, legacy: bool = False):
    """Build a fixed-base chain of two revolute Z-joints with a mimic between them.

    The leader joint is position-driven; the follower joint has no drive and no spring,
    so any tracking it does comes from the mimic row alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    # add_link (not add_body): add_body wraps each body in its own free-joint articulation.
    link_a = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_a, hx=0.1, hy=0.02, hz=0.02)
    j_leader = b.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    link_b = b.add_link(xform=wp.transform(wp.vec3(0.6, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_b, hx=0.1, hy=0.02, hz=0.02)
    j_follower = b.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    b.add_articulation([j_leader, j_follower], label="mimic_chain")
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    # follower: q_follower = coef0 + coef1 * q_leader
    if legacy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)
    else:
        b.set_joint_mimic(j_follower, j_leader, coeffs=(coef0, coef1))
    return b, j_leader, j_follower


def _run_model(model, leader_target: float, steps: int, **solver_kwargs):
    solver = SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = leader_target
    control.joint_target_q.assign(targets)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    return solver, state_0.joint_q.numpy()


def _run_chain(device, coef0: float, coef1: float, leader_target: float, steps: int = 600, legacy: bool = False):
    builder, _, _ = _build_two_revolute_chain(coef0, coef1, legacy=legacy)
    return _run_model(builder.finalize(device=device), leader_target, steps)


# -- Construction layout ---------------------------------------------------------------


def check_reject_quaternion_mimics_at_construction(test, device, **solver_kwargs):
    """Reject joint-owned mimics whose position and velocity layouts differ."""
    for joint_type in (newton.JointType.BALL, newton.JointType.FREE, newton.JointType.DISTANCE):
        with test.subTest(joint_type=joint_type):
            builder, _, follower = _build_mimic_layout(joint_type)
            model = builder.finalize(device=device)
            q_start = model.joint_q_start.numpy()
            qd_start = model.joint_qd_start.numpy()
            test.assertNotEqual(q_start[follower + 1] - q_start[follower], qd_start[follower + 1] - qd_start[follower])
            with test.assertRaisesRegex(NotImplementedError, "mimic joint .* position and velocity"):
                SolverFeatherPGS(model, **solver_kwargs)


def check_scalar_and_d6_mimic_coordinate_maps(test, device, **solver_kwargs):
    """Preserve componentwise mimic maps after a quaternion-layout free base."""
    for joint_type, dimensions in (
        (newton.JointType.REVOLUTE, 1),
        (newton.JointType.PRISMATIC, 1),
        (newton.JointType.D6, 6),
    ):
        with test.subTest(joint_type=joint_type):
            builder, leader, follower = _build_mimic_layout(joint_type)
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, **solver_kwargs)
            test.assertEqual(solver._mimic_count, dimensions)
            for suffix, start, joint in (
                ("q0", model.joint_q_start, follower),
                ("q1", model.joint_q_start, leader),
                ("dof0", model.joint_qd_start, follower),
                ("dof1", model.joint_qd_start, leader),
            ):
                expected = np.arange(int(start.numpy()[joint]), int(start.numpy()[joint]) + dimensions)
                np.testing.assert_array_equal(getattr(solver, f"_mimic_{suffix}").numpy(), expected)
            np.testing.assert_array_equal(solver._mimic_legacy.numpy(), np.full(dimensions, -1))


def check_invalid_legacy_mimic_keeps_precedence(test, device, **solver_kwargs):
    """A legacy mimic replaces the joint-owned mimic of its follower, also when it is rejected."""
    builder, leader, follower = _build_mimic_layout(newton.JointType.BALL)
    with test.assertWarns(DeprecationWarning):
        builder.add_constraint_mimic(joint0=follower, joint1=leader)
    model = builder.finalize(device=device)
    # The legacy entry is checked instead of the joint-owned BALL mimic it replaces.
    with test.assertRaisesRegex(NotImplementedError, "must couple REVOLUTE or PRISMATIC"):
        SolverFeatherPGS(model, **solver_kwargs)


def test_reject_quaternion_mimics_at_construction(test, device):
    check_reject_quaternion_mimics_at_construction(test, device)


def test_scalar_and_d6_mimic_coordinate_maps(test, device):
    check_scalar_and_d6_mimic_coordinate_maps(test, device)


def test_invalid_legacy_mimic_keeps_precedence(test, device):
    check_invalid_legacy_mimic_keeps_precedence(test, device)


def test_cross_articulation_mimic_is_rejected(test, device):
    """A mimic between two articulations raises instead of being ignored."""
    template, j_leader, j_follower = _build_two_revolute_chain(0.0, 1.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_builder(template)
    builder.add_builder(template)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # The follower of the second copy mimics the leader of the first copy.
        builder.add_constraint_mimic(joint0=j_follower + 2, joint1=j_leader, coef0=0.0, coef1=1.0)
    with test.assertRaisesRegex(NotImplementedError, "different articulations"):
        SolverFeatherPGS(builder.finalize(device=device))


# -- Dynamics --------------------------------------------------------------------------


def test_identity_mimic_tracks_leader(test, device):
    """A 1:1 mimic makes the undriven follower joint track the driven leader."""
    _, q = _run_chain(device, coef0=0.0, coef1=1.0, leader_target=0.5)
    test.assertAlmostEqual(q[0], 0.5, delta=0.05)
    test.assertAlmostEqual(q[1], q[0], delta=0.02)


def test_legacy_constraint_mimic_tracks_leader(test, device):
    """The deprecated mimic constraints couple the follower."""
    _, q = _run_chain(device, coef0=0.1, coef1=-0.5, leader_target=0.6, legacy=True)
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_legacy_constraint_overrides_joint_mimic(test, device):
    """A legacy constraint on the same follower replaces its joint-owned mimic, as in SolverMuJoCo."""
    builder, j_leader, j_follower = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=0.1, coef1=-0.5)
    solver, q = _run_model(builder.finalize(device=device), leader_target=0.6, steps=600)
    test.assertEqual(solver._mimic_count, 1)
    test.assertEqual(solver._mimic_legacy.numpy().tolist(), [0])
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_scaled_offset_mimic(test, device):
    """q_follower converges to coef0 + coef1 * q_leader for a scaled, offset mimic."""
    _, q = _run_chain(device, coef0=0.1, coef1=-0.5, leader_target=0.6)
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_runtime_coefficient_change(test, device):
    """Joint-owned coefficients are read every step, also from a captured graph."""
    builder, _, follower = _build_two_revolute_chain(0.0, 1.0)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_iterations=16, pgs_beta=0.1)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)

    def simulate():
        for _ in range(2):
            solver.step(state_0, state_1, control, None, DT)
            solver.step(state_1, state_0, control, None, DT)

    simulate()
    with wp.ScopedCapture(device=device) as capture:
        simulate()
    for _ in range(150):
        wp.capture_launch(capture.graph)
    q = state_0.joint_q.numpy()
    test.assertAlmostEqual(q[1], q[0], delta=0.02)

    coeffs = model.joint_mimic_coeffs.numpy()
    coeffs[follower] = (0.1, -0.5)
    model.joint_mimic_coeffs.assign(coeffs)
    for _ in range(150):
        wp.capture_launch(capture.graph)
    q = state_0.joint_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_mimic_row_is_assembled(test, device):
    """The solver assembles a mimic row that carries the coupling."""
    solver, _ = _run_chain(device, coef0=0.0, coef1=1.0, leader_target=0.5, steps=10)
    row_types = solver.row_type.numpy()
    counts = solver.constraint_count.numpy()
    rows = [int(t) for w in range(row_types.shape[0]) for t in row_types[w, : counts[w]]]
    test.assertIn(PGS_CONSTRAINT_TYPE_MIMIC, rows)


def test_replicated_mimics_have_articulation_local_ranges(test, device):
    """Replicated mimic rows use one compact lookup range per articulation."""
    template, _, _ = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count=4)
    solver = SolverFeatherPGS(builder.finalize(device=device))
    test.assertEqual(solver._mimic_art_start.numpy().tolist(), [0, 1, 2, 3, 4])
    test.assertEqual(solver._mimic_art_list.numpy().tolist(), [0, 1, 2, 3])


def test_disabled_mimic_is_ignored(test, device):
    """A disabled legacy mimic constraint leaves the follower joint uncoupled."""
    builder, _, _ = _build_two_revolute_chain(0.0, 1.0, legacy=True)
    builder.constraint_mimic_enabled[0] = False
    _, q = _run_model(builder.finalize(device=device), leader_target=0.5, steps=600)
    test.assertAlmostEqual(q[0], 0.5, delta=0.05)
    # The undriven follower swings freely; it must not sit at the leader angle.
    test.assertNotAlmostEqual(q[1], q[0], delta=0.02)


def test_mimic_row_overflow_is_reported(test, device):
    """A mimic row beyond dense_max_constraints is dropped and flagged, not partially kept."""
    builder, _, _ = _build_two_revolute_chain(0.0, 1.0)
    model = builder.finalize(device=device)
    # A zero activation gap keeps the (far) joint-limit rows out of the single row slot.
    options = {"dense_max_constraints": 1, "joint_limit_activation_gap": 0.0, "warn_constraint_overflow": False}
    solver = SolverFeatherPGS(model, **options)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    test.assertEqual(int(solver.mimic_slot.numpy()[0]), 0)
    test.assertFalse(bool(solver.constraint_overflow.numpy()[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # A second, consistent row on the other joint (identity coupling in both directions).
        builder.add_constraint_mimic(joint0=0, joint1=1, coef0=0.0, coef1=1.0)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, **options)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    test.assertEqual(int(solver.constraint_count.numpy()[0]), 1)
    test.assertTrue(bool(solver.constraint_overflow.numpy()[0]))
    with test.assertRaises(RuntimeError):
        solver.check_constraint_capacity()


# -- Split solve (cumulative: needs the split/CPU solve) --------------------------------


def test_split_reject_quaternion_mimics_at_construction(test, device):
    check_reject_quaternion_mimics_at_construction(test, device, pgs_mode="split")


def test_split_scalar_and_d6_mimic_coordinate_maps(test, device):
    check_scalar_and_d6_mimic_coordinate_maps(test, device, pgs_mode="split")


def test_split_invalid_legacy_mimic_keeps_precedence(test, device):
    check_invalid_legacy_mimic_keeps_precedence(test, device, pgs_mode="split")


class TestFeatherPGSMimicLayout(unittest.TestCase):
    pass


class TestFeatherPGSMimic(unittest.TestCase):
    pass


@unittest.skipUnless(_solver_accepts("pgs_mode"), "requires the FeatherPGS split solve")
class TestFeatherPGSMimicLayoutSplit(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _name in (
    "test_reject_quaternion_mimics_at_construction",
    "test_scalar_and_d6_mimic_coordinate_maps",
    "test_invalid_legacy_mimic_keeps_precedence",
    "test_cross_articulation_mimic_is_rejected",
):
    add_function_test(TestFeatherPGSMimicLayout, _name, globals()[_name], devices=cuda_devices)
for _name in (
    "test_identity_mimic_tracks_leader",
    "test_legacy_constraint_mimic_tracks_leader",
    "test_legacy_constraint_overrides_joint_mimic",
    "test_scaled_offset_mimic",
    "test_runtime_coefficient_change",
    "test_mimic_row_is_assembled",
    "test_replicated_mimics_have_articulation_local_ranges",
    "test_disabled_mimic_is_ignored",
    "test_mimic_row_overflow_is_reported",
):
    add_function_test(TestFeatherPGSMimic, _name, globals()[_name], devices=cuda_devices)
for _name in (
    "test_split_reject_quaternion_mimics_at_construction",
    "test_split_scalar_and_d6_mimic_coordinate_maps",
    "test_split_invalid_legacy_mimic_keeps_precedence",
):
    add_function_test(TestFeatherPGSMimicLayoutSplit, _name, globals()[_name], devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
