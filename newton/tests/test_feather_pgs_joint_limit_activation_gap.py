# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-limit row activation and the per-world row-family layout of SolverFeatherPGS."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_joint_limit_warp_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _built_rows(device, q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0):
    """Build the joint-limit rows of one single-DOF articulation and return ``(J, phi)``."""
    max_constraints = 8
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    J_group = wp.zeros((1, max_constraints, 1), dtype=wp.float32, device=device)
    world_phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    kernel = _get_joint_limit_warp_kernel(1, wp.get_device(device).arch, 1)
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            1,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([lower], dtype=wp.float32, device=device),
            wp.array([upper], dtype=wp.float32, device=device),
            wp.array([q], dtype=wp.float32, device=device),
            gap,
            max_constraints,
        ],
        outputs=[
            world_slot_counter,
            J_group,
            wp.zeros((1, max_constraints), dtype=wp.int32, device=device),
            wp.zeros((1, max_constraints), dtype=wp.int32, device=device),
            wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
            world_phi,
            wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
        ],
        block_dim=32,
        device=device,
    )
    count = int(world_slot_counter.numpy()[0])
    return J_group.numpy()[0, :count, 0].tolist(), world_phi.numpy()[0, :count].tolist()


def _make_layout_run(device, *, enable_joint_velocity_limits=True):
    """Build a scene that produces every row family: limits, velocity limits and contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.05), wp.quat_identity()),
        limit_lower=-0.1,
        limit_upper=0.1,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2

    free_body = builder.add_body(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()),
        custom_attributes={
            "rigid_body_max_linear_velocity": 1.0,
            "rigid_body_max_angular_velocity": 1.0,
        },
    )
    builder.add_shape_box(free_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    velocity_limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
    velocity_limits[0] = 0.25
    model.joint_velocity_limit.assign(velocity_limits)
    solver = SolverFeatherPGS(
        model,
        enable_joint_limits=True,
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        velocity_limit_activation_fraction=0.5,
        dense_max_constraints=64,
        mf_max_constraints=64,
        pgs_iterations=0,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0] = 0.5
    free_dof = int(model.joint_qd_start.numpy()[1])
    joint_qd[free_dof] = 1.5
    joint_qd[free_dof + 3] = 1.5
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 60.0)
    return solver


def test_combined_row_families_follow_the_documented_layout(test, device):
    """Lay out dense rows as limits, velocity limits, contacts; free-body rows as contacts, velocity limits."""
    solver = _make_layout_run(device)
    dense_count = int(solver.constraint_count.numpy()[0])
    dense_types = solver.row_type.numpy()[0, :dense_count]
    limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    contact_rows = np.flatnonzero(
        (dense_types == PGS_CONSTRAINT_TYPE_CONTACT) | (dense_types == PGS_CONSTRAINT_TYPE_FRICTION)
    )
    test.assertGreater(limit_rows.size, 0)
    test.assertGreater(velocity_limit_rows.size, 0)
    test.assertGreater(contact_rows.size, 0)
    test.assertLess(limit_rows.max(), velocity_limit_rows.min())
    test.assertLess(velocity_limit_rows.max(), contact_rows.min())

    mf_count = int(solver.mf_constraint_count.numpy()[0])
    mf_types = solver.mf_row_type.numpy()[0, :mf_count]
    mf_contact_end = int(solver.mf_contact_rows_end.numpy()[0])
    mf_contacts = np.flatnonzero((mf_types == PGS_CONSTRAINT_TYPE_CONTACT) | (mf_types == PGS_CONSTRAINT_TYPE_FRICTION))
    mf_velocity_limits = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertGreater(mf_contacts.size, 0)
    test.assertGreater(mf_velocity_limits.size, 0)
    test.assertTrue(np.all(mf_contacts < mf_contact_end))
    test.assertTrue(np.all(mf_velocity_limits >= mf_contact_end))


def test_velocity_limit_rows_require_the_option(test, device):
    """Build no joint velocity-limit rows unless velocity limits are enabled."""
    solver = _make_layout_run(device, enable_joint_velocity_limits=False)
    dense_count = int(solver.constraint_count.numpy()[0])
    dense_types = solver.row_type.numpy()[0, :dense_count]
    test.assertEqual(int(np.sum(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 0)
    test.assertGreater(int(np.sum(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)), 0)


def test_finite_gap_builds_only_near_limit_rows(test, device):
    """Create a limit row only within the activation gap of that limit."""
    test.assertEqual(_built_rows(device, 0.0, gap=0.2), ([], []))
    jacobian, phi = _built_rows(device, -0.85, gap=0.2)
    test.assertEqual(jacobian, [1.0])
    test.assertAlmostEqual(phi[0], 0.15, places=6)
    jacobian, phi = _built_rows(device, 0.85, gap=0.2)
    test.assertEqual(jacobian, [-1.0])
    test.assertAlmostEqual(phi[0], 0.15, places=6)


def test_finite_gap_does_not_activate_unlimited_sentinel_limits(test, device):
    """Treat the builder's unlimited sentinel as no limit."""
    test.assertEqual(_built_rows(device, 0.0, gap=0.2, lower=-MAXVAL, upper=MAXVAL), ([], []))


def test_infinite_gap_allocates_every_finite_limit(test, device):
    """Create both rows of every finite limit with the default infinite gap."""
    jacobian, phi = _built_rows(device, 0.0, gap=float("inf"))
    test.assertEqual(jacobian, [1.0, -1.0])
    test.assertEqual(phi, [1.0, 1.0])


def _driven_limited_joint_position(device, **solver_kwargs) -> float:
    """Drive a revolute joint limited to ``[-0.3, 0.3]`` toward 1 rad and return its final position."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    joint = builder.add_joint_revolute(
        -1, link, axis=newton.Axis.Z, limit_lower=-0.3, limit_upper=0.3, target_ke=500.0, target_kd=10.0
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_iterations=8, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    control.joint_target_q.fill_(1.0)
    for _ in range(240):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return float(state_0.joint_q.numpy()[0])


def test_limit_holds_a_driven_joint(test, device):
    """Hold a joint driven past its upper limit at the limit when joint limits are enabled."""
    test.assertAlmostEqual(_driven_limited_joint_position(device, enable_joint_limits=True), 0.3, delta=2.0e-3)


def test_disabled_joint_limits_are_not_enforced(test, device):
    """Leave finite joint limits unenforced by default, so the drive reaches its target."""
    test.assertAlmostEqual(_driven_limited_joint_position(device), 1.0, delta=2.0e-3)
    test.assertAlmostEqual(_driven_limited_joint_position(device, enable_joint_limits=False), 1.0, delta=2.0e-3)


def _build_default_limit_chain(device, count: int):
    """Build ``count`` independent revolute links with the builder's default (+/-1e10) limits."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    for _ in range(count):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        joints.append(builder.add_joint_revolute(-1, link, axis=newton.Axis.Z))
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def test_default_finite_limits_keep_rows_and_warn_at_capacity(test, device):
    """Keep the rows of the builder's large finite default limits and warn when they exceed the capacity."""
    for count, expect_warning in ((16, False), (17, True)):
        with test.subTest(dofs=count):
            model = _build_default_limit_chain(device, count)
            test.assertEqual(float(model.joint_limit_upper.numpy()[0]), MAXVAL)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver = SolverFeatherPGS(model, enable_joint_limits=True, warn_constraint_overflow=False)
            messages = [str(w.message) for w in caught if "joint position limits" in str(w.message)]
            test.assertEqual(len(messages), int(expect_warning), messages)
            if expect_warning:
                test.assertIn("at least 34 dense rows", messages[0])
                test.assertIn("dense_max_constraints=32", messages[0])
            solver.step(model.state(), model.state(), model.control(), None, 0.01)
            # Large finite bounds are not reclassified as unlimited: two rows per DOF, up to
            # the capacity.
            test.assertEqual(int(solver.constraint_count.numpy()[0]), min(2 * count, 32))
            test.assertEqual(bool(solver.constraint_overflow.numpy()[0]), expect_warning)

    model = _build_default_limit_chain(device, 17)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SolverFeatherPGS(model, enable_joint_limits=True, dense_max_constraints=34)
        gap_solver = SolverFeatherPGS(model, enable_joint_limits=True, joint_limit_activation_gap=0.5)
    test.assertFalse([w for w in caught if "joint position limits" in str(w.message)])
    gap_solver.step(model.state(), model.state(), model.control(), None, 0.01)
    test.assertEqual(int(gap_solver.constraint_count.numpy()[0]), 0)
    test.assertFalse(bool(gap_solver.constraint_overflow.numpy()[0]))


def test_disabled_joint_limits_build_no_rows(test, device):
    """Build no limit rows and use no capacity for finite limits when joint limits are disabled."""
    model = _build_default_limit_chain(device, 17)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solvers = (SolverFeatherPGS(model), SolverFeatherPGS(model, enable_joint_limits=False))
    # The 34 rows of the +/-1e10 defaults would exceed the default capacity of 32 if limits were enabled.
    test.assertFalse([w for w in caught if "joint position limits" in str(w.message)])
    for solver in solvers:
        test.assertFalse(solver.enable_joint_limits)
        solver.step(model.state(), model.state(), model.control(), None, 0.01)
        test.assertEqual(int(solver.constraint_count.numpy()[0]), 0)
        test.assertFalse(bool(solver.constraint_overflow.numpy()[0]))

    # A joint outside its limit gets no row either, whatever the activation gap.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    joint = builder.add_joint_revolute(-1, link, axis=newton.Axis.Z, limit_lower=-0.1, limit_upper=0.1)
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, joint_limit_activation_gap=0.5)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, 0.01)
    test.assertEqual(int(solver.constraint_count.numpy()[0]), 0)
    test.assertAlmostEqual(float(state_1.joint_q.numpy()[0]), 0.2, places=6)


class TestFeatherPGSJointLimitActivationGap(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    (
        "test_combined_row_families_follow_the_documented_layout",
        test_combined_row_families_follow_the_documented_layout,
    ),
    ("test_velocity_limit_rows_require_the_option", test_velocity_limit_rows_require_the_option),
    ("test_finite_gap_builds_only_near_limit_rows", test_finite_gap_builds_only_near_limit_rows),
    (
        "test_finite_gap_does_not_activate_unlimited_sentinel_limits",
        test_finite_gap_does_not_activate_unlimited_sentinel_limits,
    ),
    ("test_infinite_gap_allocates_every_finite_limit", test_infinite_gap_allocates_every_finite_limit),
    ("test_limit_holds_a_driven_joint", test_limit_holds_a_driven_joint),
    ("test_disabled_joint_limits_are_not_enforced", test_disabled_joint_limits_are_not_enforced),
    ("test_disabled_joint_limits_build_no_rows", test_disabled_joint_limits_build_no_rows),
    (
        "test_default_finite_limits_keep_rows_and_warn_at_capacity",
        test_default_finite_limits_keep_rows_and_warn_at_capacity,
    ),
):
    add_function_test(TestFeatherPGSJointLimitActivationGap, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main()
